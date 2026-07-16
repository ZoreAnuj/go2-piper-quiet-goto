"""Record single-robot scenario videos of the trained quiet policy.

Each clip drives the goto command deterministically (goal placement is forced
every step, resampling disabled) so one behavior is shown per video.
"""

import os
import sys
from pathlib import Path

os.environ["MUJOCO_GL"] = "wgl"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import onnxruntime as ort
import torch

import mjlab_tasks.go2_piper_quiet  # noqa: F401
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

ONNX = sys.argv[1] if len(sys.argv) > 1 else (
  r"D:\go2_bungalow_bc\logs\rsl_rl\go2_piper_quiet\2026-07-15_19-21-53"
  r"\2026-07-15_19-21-53.onnx"
)
OUT = Path(r"D:\go2_bungalow_bc\cache\quiet_videos\actions")
OUT.mkdir(parents=True, exist_ok=True)

sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

cfg = load_env_cfg("Mjlab-Velocity-Flat-Go2Piper-Quiet", play=True)
cfg.scene.num_envs = 1
cfg.viewer.height, cfg.viewer.width = 480, 640
cfg.viewer.distance = 2.2
env = ManagerBasedRlEnv(cfg, device=DEVICE, render_mode="rgb_array")

robot = env.scene["robot"]
goto = env.command_manager.get_term("goto")
ARM_CTRL = torch.tensor(
  [i for i, n in enumerate(robot.actuator_names) if "piper" in n or "gripper" in n],
  dtype=torch.int, device=DEVICE,
)
FOLDED = torch.tensor([[0.0, 1.2, -1.4, 0.0, 0.0, 0.0, 0.035, -0.035]], device=DEVICE)


def policy(o: torch.Tensor) -> torch.Tensor:
  a = sess.run(None, {in_name: o.cpu().numpy().astype(np.float32)})[0]
  return torch.from_numpy(a).to(DEVICE)


def body_goal(dx: float, dy: float) -> torch.Tensor:
  """World goal at body-frame offset (dx, dy) from the robot's CURRENT pose."""
  pos = robot.data.root_link_pos_w[0, :2]
  h = robot.data.heading_w[0]
  c, s = torch.cos(h), torch.sin(h)
  return torch.stack([pos[0] + c * dx - s * dy, pos[1] + s * dx + c * dy]).unsqueeze(0)


def record(name: str, seconds: float, script):
  """script(i, t) is called every step to set goals / arm / pushes."""
  obs, _ = env.reset()
  robot.write_ctrl_to_sim(FOLDED, ARM_CTRL)
  vw = cv2.VideoWriter(str(OUT / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 480))
  steps = int(seconds * 50)
  for i in range(steps):
    goto.time_left[:] = 100.0  # freeze random resampling
    script(i, i / 50.0)
    obs, *_ = env.step(policy(obs["actor"]))
    if i % 2 == 0:
      vw.write(cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR))
  vw.release()
  print("wrote", OUT / f"{name}.mp4")


# --- scenarios (goals set in the robot's CURRENT body frame at key moments) ---

state = {}

def walk_forward(i, t):
  if i == 0 or i == 250:  # 2.5 m ahead, twice
    state["g"] = body_goal(2.5, 0.0)
  goto.goal_w[:] = state["g"]

def turn_around(i, t):
  if i == 0:
    state["g"] = body_goal(-2.0, 0.1)  # directly behind -> 180 deg turn, then walk
  goto.goal_w[:] = state["g"]

def sidestep(i, t):
  if i == 0:
    state["g"] = body_goal(0.3, 1.5)   # left
  elif i == 200:
    state["g"] = body_goal(0.3, -1.5)  # then right
  goto.goal_w[:] = state["g"]

def stand_quiet(i, t):
  if i == 0:
    state["g"] = robot.data.root_link_pos_w[:, :2].clone()
  goto.goal_w[:] = state["g"]  # goal under the robot -> stand
  if i % 75 == 0 and i > 0:    # arm waves while the base must stay put
    tgt = FOLDED + (torch.rand_like(FOLDED) * 2 - 1) * torch.tensor(
      [[0.8, 0.5, 0.5, 0.5, 0.5, 0.8, 0.0, 0.0]], device=DEVICE)
    robot.write_ctrl_to_sim(tgt, ARM_CTRL)

def push_recovery(i, t):
  if i == 0:
    state["g"] = body_goal(3.0, 0.0)
  goto.goal_w[:] = state["g"]
  if i in (150, 300):  # lateral shove mid-walk
    envs_mdp.push_by_setting_velocity(
      env, torch.tensor([0], device=DEVICE),
      {"y": (0.7, 0.7), "yaw": (0.5, 0.5)})

def arm_swing_walk(i, t):
  if i == 0 or i == 250:
    state["g"] = body_goal(2.5, 0.0)
  goto.goal_w[:] = state["g"]
  if i % 60 == 0:  # aggressive arm target changes while walking
    tgt = FOLDED + (torch.rand_like(FOLDED) * 2 - 1) * torch.tensor(
      [[1.0, 0.7, 0.6, 0.6, 0.6, 1.0, 0.0, 0.0]], device=DEVICE)
    robot.write_ctrl_to_sim(tgt, ARM_CTRL)


for name, seconds, fn in [
  ("walk_forward", 10, walk_forward),
  ("turn_around", 10, turn_around),
  ("sidestep", 9, sidestep),
  ("stand_quiet_arm_wave", 10, stand_quiet),
  ("push_recovery", 10, push_recovery),
  ("arm_swing_walk", 10, arm_swing_walk),
]:
  state.clear()
  record(name, seconds, fn)
env.close()
