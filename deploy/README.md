# Deploy the quiet goto policy on a real Go2 (low-level)

`deploy_goto_go2.py` is the real-robot host loop the top-level README describes but
does not ship. It reads `LowState`, assembles the 51-dim observation, runs
`policies/policy.onnx` (pure-numpy — **no onnxruntime needed**, only numpy +
`unitree_sdk2py`, both already on the robot), and streams joint-position `LowCmd`
at 50 Hz with the descriptor PD gains.

## Prerequisites (all must be true before you arm it)

1. **Run it ON the robot over SSH.** A 50 Hz motor loop over WiFi is unsafe.
   ```bash
   scp -r deploy policies root@<robot-ip>:/unitree/module/pet_go/go2quiet/
   ssh root@<robot-ip>
   cd /unitree/module/pet_go/go2quiet
   export CYCLONEDDS_URI=/unitree/etc/cyclonedds.xml
   export PYTHONPATH=/unitree/module/pet_go
   ```
2. **Low-level control must be unlocked.** This policy needs `rt/lowcmd` (direct joint
   control), which is a *different* stack from the `SportClient` in `OPERATIONS.md`.
   On some Go2 firmware low-level is locked; `--release-sport` tries to release the
   sport controller via `MotionSwitcherClient` and aborts if it can't. **Unverified on
   this unit — this is the #1 feasibility risk.**
3. **The Piper arm should be mounted.** The policy was trained with it (mass
   distribution). Without it the policy is **out-of-distribution** — degraded stability,
   expect it may not hold a clean stand. Test extra carefully or not at all without the arm.
4. **Robot HOISTED, feet off the ground, area clear, hand on the power kill.**

## Bring-up ladder — do not skip a rung

| # | Command | What it proves | Rig |
|---|---------|----------------|-----|
| 1 | `python3 deploy_goto_go2.py --check` | obs mapping/signs are right (prints per-leg angles + projected gravity — **verify against reality**) | suspended |
| 2 | `python3 deploy_goto_go2.py --release-sport --mode stand --hold-default --armed` | PD engages, legs glide to default stance, no snap | suspended |
| 3 | `python3 deploy_goto_go2.py --release-sport --mode stand --armed` | policy holds a stand, zero command, no tremor/drift (10–20 s) | suspended |
| 4 | repeat rung 3 | stand holds with feet loaded | on a low stand, then floor |
| 5 | `python3 deploy_goto_go2.py --release-sport --mode walk --goal-bx 1.5 --armed` | open-loop forward walk (body-frame goal held constant) | floor, clear space, spotter |

- **Nothing sends torque without `--armed`.** Without it the loop runs with `kp=kd=0`
  (true dry run) so you can watch obs/actions safely.
- **Ctrl-C or any error → emergency joint damp** (kp=0, gentle kd), then exit. The loop
  also damps if `LowState` goes stale (>100 ms).
- `--goal-bx/--goal-by` set a **body-frame** goal held constant: `+bx` walks forward
  indefinitely, lateral → sidestep, behind → turn. True world-goal chasing needs pose
  integration (not yet implemented — see `GotoCommand` TODO).

## What is validated vs. not

**Validated offline** (see the numpy checks): policy I/O contract (51→12 ELU MLP),
weight values, joint remap round-trip, command generator, projected-gravity math,
full observation assembly → bounded actions.

**NOT yet validated on hardware** (verify with rung 1 before arming): exact
`unitree_sdk2py` field names/CRC on this SDK version, real encoder sign/zero
conventions, and whether low-level commands are honored at all on this firmware.
