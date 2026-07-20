#!/usr/bin/env python3
"""
deploy_goto_go2.py — run the quiet goto policy on a REAL Unitree Go2 (low-level).

This is the real-robot host loop the README's "Real Go2" section describes but does
not ship. It:
  * reads LowState (IMU + joint encoders) over CycloneDDS at 50 Hz,
  * assembles the exact 51-dim observation from policy_descriptor.goto.json,
  * runs policy.onnx (pure-numpy MLP by default — no onnxruntime needed; the robot
    only needs numpy + unitree_sdk2py, both already present),
  * sends 0.25*action + default_pos as joint POSITION targets via LowCmd with the
    descriptor PD gains (kp=20/kd=1, calf kp=40/kd=2).

RUN IT *ON THE ROBOT* over SSH (a 50 Hz motor loop over WiFi is unsafe).

============================  SAFETY — READ FIRST  ============================
This takes DIRECT low-level motor control and BYPASSES Unitree's sport-mode
safety controller. A sim-trained policy has never touched this hardware. Mistakes
(joint-map, sign, gains) produce violent motion that can injure people or wreck
the robot.

MANDATORY first-run sequence:
  1.  Robot HOISTED / suspended, all four feet OFF the ground, area clear, hand on
      the remote power kill.
  2.  `--check`  : reads state and runs inference but sends NO motor command. Verify
      the printed per-leg joint angles and projected-gravity match reality.
  3.  `--mode stand --hold-default` (still suspended): engages PD to the default
      stance only, policy NOT driving. Confirm no snap, legs go to stance pose.
  4.  `--mode stand --armed` (still suspended): policy holds a stand, zero command.
      Watch for tremor / drift for 10-20 s.
  5.  Only then, feet down on a stand test, then `--mode walk --goal-bx 1.5`.

The policy was trained with an AgileX Piper arm mounted (mass distribution). WITHOUT
the arm it is OUT OF DISTRIBUTION — expect degraded stability. Do not skip step 1.

Nothing sends torque unless you pass --armed. Ctrl-C / any error -> emergency damp.
=============================================================================
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Descriptor constants (policies/policy_descriptor.goto.json) — POLICY joint order
#   idx: 0..2 FL(hip,thigh,calf) 3..5 FR 6..8 RL 9..11 RR
# ----------------------------------------------------------------------------
POLICY_JOINTS = [
    "FL_hip", "FL_thigh", "FL_calf", "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf", "RR_hip", "RR_thigh", "RR_calf",
]
DEFAULT_POS = np.array([0.1, 0.9, -1.8,  -0.1, 0.9, -1.8,
                        0.1, 0.9, -1.8,  -0.1, 0.9, -1.8], dtype=np.float32)
ACTION_SCALE = 0.25
KP = np.array([20, 20, 40,  20, 20, 40,  20, 20, 40,  20, 20, 40], dtype=np.float32)
KD = np.array([1,  1,  2,   1,  1,  2,   1,  1,  2,   1,  1,  2],   dtype=np.float32)
BODY_HEIGHT = 0.32
CONTROL_HZ = 50.0
DT = 1.0 / CONTROL_HZ
GAIT_PERIOD = 0.6  # s

# Unitree Go2 LowCmd/LowState motor index order is FR,FL,RR,RL (hip,thigh,calf):
#   0 FR_hip 1 FR_thigh 2 FR_calf | 3 FL_hip 4 FL_thigh 5 FL_calf
#   6 RR_hip 7 RR_thigh 8 RR_calf | 9 RL_hip 10 RL_thigh 11 RL_calf
# POLICY_TO_UNITREE[p] = unitree motor index that holds policy joint p.
POLICY_TO_UNITREE = np.array([3, 4, 5,  0, 1, 2,  9, 10, 11,  6, 7, 8], dtype=np.int32)
UNITREE_TO_POLICY = np.argsort(POLICY_TO_UNITREE)  # inverse map (self-inverse here)

# ----------------------------------------------------------------------------
# Pure-numpy policy (parses the ONNX protobuf directly; falls back to onnxruntime
# if it happens to be installed). Verified to match the exported actor.
# ----------------------------------------------------------------------------
def _read_varint(buf, i):
    shift = val = 0
    while True:
        b = buf[i]; i += 1
        val |= (b & 0x7F) << shift
        if not (b & 0x80):
            return val, i
        shift += 7


def _fields(buf):
    i, n, out = 0, len(buf), {}
    while i < n:
        tag, i = _read_varint(buf, i); fn, wt = tag >> 3, tag & 7
        if wt == 0:
            v, i = _read_varint(buf, i)
        elif wt == 2:
            ln, i = _read_varint(buf, i); v = buf[i:i + ln]; i += ln
        elif wt == 1:
            v = buf[i:i + 8]; i += 8
        elif wt == 5:
            v = buf[i:i + 4]; i += 4
        else:
            raise ValueError(f"bad wire type {wt}")
        out.setdefault(fn, []).append(v)
    return out


class NumpyMLP:
    """ELU MLP reconstructed from the ONNX initializers (Gemm/Elu x N)."""

    def __init__(self, onnx_path):
        data = Path(onnx_path).read_bytes()
        graph = _fields(_fields(data)[7][0])
        inits = {}
        for raw in graph.get(5, []):
            f = _fields(raw)
            name = f[8][0].decode()
            dims = list(f.get(1, []))          # repeated int64 dims (already decoded ints)
            floats = np.frombuffer(f[9][0], dtype="<f4").copy() if 9 in f else None
            inits[name] = (dims, floats)
        idx = lambda nm: int(nm.split(".")[1])
        w = sorted([(idx(n), d, f) for n, (d, f) in inits.items()
                    if f is not None and len(d) == 2])
        b = {d[0]: f for n, (d, f) in inits.items() if f is not None and len(d) == 1}
        self.layers = []
        for _, d, f in w:
            out_dim, in_dim = d
            self.layers.append((f.reshape(out_dim, in_dim), b[out_dim]))

    @staticmethod
    def _elu(x):
        return np.where(x > 0, x, np.exp(np.clip(x, -30, 0)) - 1.0)

    def __call__(self, obs):
        x = np.asarray(obs, dtype=np.float32)
        for i, (W, bb) in enumerate(self.layers):
            x = x @ W.T + bb
            if i < len(self.layers) - 1:
                x = self._elu(x)
        return x


def load_policy(onnx_path):
    try:
        import onnxruntime as ort  # noqa
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        print("[policy] using onnxruntime")
        return lambda o: sess.run(None, {name: o.reshape(1, -1).astype(np.float32)})[0].reshape(-1)
    except Exception:
        print("[policy] using pure-numpy MLP (onnxruntime not present)")
        mlp = NumpyMLP(onnx_path)
        return lambda o: mlp(o).reshape(-1)


# ----------------------------------------------------------------------------
# Math + host-side command generation (mirror of mdp_ext.GotoCommand)
# ----------------------------------------------------------------------------
def quat_rotate_inverse(q_wxyz, v):
    w, x, y, z = q_wxyz
    qv = np.array([x, y, z], dtype=np.float32)
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(qv, v) * (2.0 * w)
    c = qv * (2.0 * float(qv @ v))
    return a - b + c


class GotoCommand:
    """Body-frame goal (bx,by) held constant -> velocity command + gait clock.

    Holding the goal fixed in the BODY frame gives open-loop locomotion without an
    odometer: (bx>0,by=0) walks forward forever; goal behind -> turn; lateral ->
    sidestep. (0,0) = stand. Real world-goal chasing needs pose integration (TODO).
    """

    def __init__(self, goal_bx=0.0, goal_by=0.0):
        self.goal = np.array([goal_bx, goal_by], dtype=np.float32)
        self.v = np.zeros(2, dtype=np.float32)
        self.wz = 0.0
        self.phase = 0.0

    def step(self):
        tb = float(np.linalg.norm(self.goal))
        gain = min(tb / 0.8, 1.0) if tb > 0 else 0.0
        v_des = np.clip(1.2 * self.goal, -0.6, 0.6) * gain
        wz_des = np.clip(1.2 * math.atan2(self.goal[1], self.goal[0]), -0.8, 0.8) if tb > 0.3 else 0.0
        # slew-limit: 0.75 m/s^2 linear, 1.5 rad/s^2 yaw
        dv = np.clip(v_des - self.v, -0.75 * DT, 0.75 * DT)
        self.v = self.v + dv
        self.wz += float(np.clip(wz_des - self.wz, -1.5 * DT, 1.5 * DT))
        moving = (abs(self.v[0]) + abs(self.v[1]) + abs(self.wz)) > 0.1
        if moving:
            self.phase = (self.phase + DT / GAIT_PERIOD) % 1.0
            gs, gc = math.sin(2 * math.pi * self.phase), math.cos(2 * math.pi * self.phase)
        else:
            gs = gc = 0.0
        cmd5 = np.array([self.v[0], self.v[1], self.wz, gs, gc], dtype=np.float32)
        target_b = np.array([self.goal[0], self.goal[1], 0.0], dtype=np.float32)
        return cmd5, target_b


def build_obs(low_state, cmd5, target_b, last_action):
    """Assemble the 51-dim observation in the exact descriptor order."""
    imu = low_state.imu_state
    ang_vel = np.array(imu.gyroscope, dtype=np.float32)                       # [0:3]
    grav = quat_rotate_inverse(np.array(imu.quaternion, dtype=np.float32),
                               np.array([0, 0, -1], dtype=np.float32))         # [3:6]
    q_u = np.array([low_state.motor_state[i].q for i in range(12)], dtype=np.float32)
    dq_u = np.array([low_state.motor_state[i].dq for i in range(12)], dtype=np.float32)
    q_p = q_u[POLICY_TO_UNITREE]      # unitree order -> policy order
    dq_p = dq_u[POLICY_TO_UNITREE]
    joint_pos_rel = q_p - DEFAULT_POS                                         # [11:23]
    obs = np.concatenate([
        ang_vel, grav, cmd5,
        joint_pos_rel, dq_p, last_action,
        np.array([BODY_HEIGHT], dtype=np.float32), target_b,
    ]).astype(np.float32)
    assert obs.shape[0] == 51, obs.shape
    return obs


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Deploy quiet goto policy on a real Go2 (low-level).")
    ap.add_argument("--policy", default=str(Path(__file__).resolve().parents[1] / "policies" / "policy.onnx"))
    ap.add_argument("--iface", default=None, help="network interface for DDS (default: rely on CYCLONEDDS_URI)")
    ap.add_argument("--mode", choices=["stand", "walk"], default="stand")
    ap.add_argument("--goal-bx", type=float, default=0.0, help="body-frame goal x (m), held constant")
    ap.add_argument("--goal-by", type=float, default=0.0, help="body-frame goal y (m), held constant")
    ap.add_argument("--check", action="store_true", help="read state + infer, send NO motor command, print and exit-loop")
    ap.add_argument("--armed", action="store_true", help="REQUIRED to actually send torque; without it kp=kd=0")
    ap.add_argument("--hold-default", action="store_true", help="ignore policy, hold default stance (PD bring-up test)")
    ap.add_argument("--release-sport", action="store_true", help="release Unitree sport-mode so low-level is honored")
    ap.add_argument("--duration", type=float, default=20.0, help="run seconds (excl. ramp)")
    ap.add_argument("--ramp", type=float, default=2.0, help="seconds to ramp PD gains 0->full and glide to default")
    ap.add_argument("--max-action", type=float, default=6.0, help="clamp on raw policy action")
    args = ap.parse_args()

    if "CYCLONEDDS_URI" not in os.environ:
        print("WARNING: CYCLONEDDS_URI not set; expected /unitree/etc/cyclonedds.xml", file=sys.stderr)

    from unitree_sdk2py.core.channel import (ChannelPublisher, ChannelSubscriber,
                                             ChannelFactoryInitialize)
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
    from unitree_sdk2py.utils.crc import CRC

    if args.iface:
        ChannelFactoryInitialize(0, args.iface)
    else:
        ChannelFactoryInitialize(0)

    if args.release_sport:
        try:
            from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
            msc = MotionSwitcherClient(); msc.SetTimeout(5.0); msc.Init()
            for _ in range(10):
                status, result = msc.CheckMode()
                if not result.get("name"):
                    break
                print(f"[sport] releasing mode '{result['name']}' ...")
                msc.ReleaseMode(); time.sleep(1.0)
            print("[sport] sport-mode released (low-level should now be honored)")
        except Exception as e:
            print(f"[sport] could NOT release sport mode: {e}\n"
                  f"        On some Go2 firmware low-level control is locked. Aborting.", file=sys.stderr)
            return 2

    crc = CRC()
    state = {"low": None, "t": 0.0}

    def on_state(msg: LowState_):
        state["low"] = msg
        state["t"] = time.time()

    ChannelSubscriber("rt/lowstate", LowState_).Init(on_state, 10)
    pub = ChannelPublisher("rt/lowcmd", LowCmd_); pub.Init()

    # wait for first state
    print("[dds] waiting for LowState ...")
    t0 = time.time()
    while state["low"] is None:
        if time.time() - t0 > 5.0:
            print("ERROR: no LowState in 5 s — DDS not connected (URI/iface? robot up?).", file=sys.stderr)
            return 3
        time.sleep(0.05)
    print("[dds] LowState connected.")

    policy = load_policy(args.policy)
    cmdgen = GotoCommand(args.goal_bx, args.goal_by) if args.mode == "walk" else GotoCommand(0.0, 0.0)
    last_action = np.zeros(12, dtype=np.float32)

    # --check: one read + inference, print, no command
    def snapshot():
        cmd5, target_b = cmdgen.step()
        obs = build_obs(state["low"], cmd5, target_b, last_action)
        act = np.clip(policy(obs), -args.max_action, args.max_action)
        tgt_p = ACTION_SCALE * act + DEFAULT_POS
        return obs, cmd5, target_b, act, tgt_p

    if args.check:
        obs, cmd5, target_b, act, tgt_p = snapshot()
        np.set_printoptions(precision=3, suppress=True)
        print("\n--- CHECK (no command sent) ---")
        print("base_ang_vel :", obs[0:3])
        print("proj_gravity :", obs[3:6], " (upright ~ [0 0 -1])")
        print("command 5    :", cmd5)
        print("q_rel (policy order FL,FR,RL,RR x hip,thigh,calf):\n ", obs[11:23])
        print("target_b     :", target_b)
        print("action       :", act)
        print("target_q pol :", tgt_p)
        print("Verify: legs at rest read near default (thigh~0.9, calf~-1.8), "
              "proj_gravity ~ [0,0,-1] when level. Then re-run without --check.")
        return 0

    if not args.armed:
        print("[safe] --armed NOT set: running the loop with kp=kd=0 (zero torque dry run).")

    cmd = unitree_go_msg_dds__LowCmd_()
    cmd.head = [0xFE, 0xEF]
    cmd.level_flag = 0xFF  # LOWLEVEL
    for i in range(20):
        cmd.motor_cmd[i].mode = 0x01
        cmd.motor_cmd[i].q = 0.0
        cmd.motor_cmd[i].dq = 0.0
        cmd.motor_cmd[i].kp = 0.0
        cmd.motor_cmd[i].kd = 0.0
        cmd.motor_cmd[i].tau = 0.0

    # starting joint pose (unitree order) to glide from -> default, avoids snap
    q_start_u = np.array([state["low"].motor_state[i].q for i in range(12)], dtype=np.float32)

    def emergency_damp(reason=""):
        for _ in range(10):
            for i in range(20):
                cmd.motor_cmd[i].q = 0.0
                cmd.motor_cmd[i].dq = 0.0
                cmd.motor_cmd[i].kp = 0.0
                cmd.motor_cmd[i].kd = 3.0 if i < 12 else 0.0  # gentle joint damping
                cmd.motor_cmd[i].tau = 0.0
            cmd.crc = crc.Crc(cmd)
            pub.Write(cmd)
            time.sleep(0.02)
        print(f"[safe] emergency damp sent. {reason}")

    steps = int((args.duration + args.ramp) / DT)
    ramp_steps = int(args.ramp / DT)
    next_t = time.time()
    print(f"[run] mode={args.mode} armed={args.armed} hold_default={args.hold_default} "
          f"goal=({args.goal_bx},{args.goal_by}) — Ctrl-C to stop.")
    try:
        for k in range(steps):
            low = state["low"]
            if time.time() - state["t"] > 0.1:
                raise RuntimeError("LowState stale > 100 ms")

            cmd5, target_b = cmdgen.step()
            if args.hold_default:
                target_p = DEFAULT_POS.copy()
                act = np.zeros(12, dtype=np.float32)
            else:
                obs = build_obs(low, cmd5, target_b, last_action)
                act = np.clip(policy(obs), -args.max_action, args.max_action)
                target_p = ACTION_SCALE * act + DEFAULT_POS
            last_action = act

            ramp = min(1.0, (k + 1) / max(1, ramp_steps))
            # glide the target from the captured start pose to the policy/default target
            target_u_full = target_p[UNITREE_TO_POLICY]  # policy order -> unitree order
            target_u = (1.0 - ramp) * q_start_u + ramp * target_u_full
            kp = KP[UNITREE_TO_POLICY] * ramp * (1.0 if args.armed else 0.0)
            kd = KD[UNITREE_TO_POLICY] * (1.0 if args.armed else 0.0)

            for j in range(12):
                cmd.motor_cmd[j].q = float(target_u[j])
                cmd.motor_cmd[j].dq = 0.0
                cmd.motor_cmd[j].kp = float(kp[j])
                cmd.motor_cmd[j].kd = float(kd[j])
                cmd.motor_cmd[j].tau = 0.0
            cmd.crc = crc.Crc(cmd)
            pub.Write(cmd)

            if k % 25 == 0:
                print(f"  t={k*DT:5.2f}s ramp={ramp:.2f} v={cmd5[:3]} max|a|={np.max(np.abs(act)):.2f}")

            next_t += DT
            sleep = next_t - time.time()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.time()  # fell behind; don't spiral
        print("[run] duration complete — damping.")
        emergency_damp("normal completion")
    except KeyboardInterrupt:
        emergency_damp("KeyboardInterrupt")
    except Exception as e:
        emergency_damp(f"exception: {e}")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
