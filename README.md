# go2-piper-quiet-goto

Quiet, smooth **goto locomotion policy for the Unitree Go2 (Pro) with a back-mounted AgileX Piper arm**, trained with [mjlab](https://github.com/mujocolab/mjlab) (MuJoCo-Warp, rsl_rl PPO). It is a drop-in replacement for the LuckyEngine `Go2PiperGoto` policy slot (same 51-dim observation contract, same PD constants), retrained for:

- **Silence**: soft footfalls (contact-force + landing-impact penalties), no foot scuffing, low joint accel/torque
- **Smoothness**: action-rate/jerk penalties, slew-limited velocity commands, softened goal P-law
- **External-force resistance**: random base pushes, ±15 N / 5 N·m base wrench bursts, ±8 N end-effector tugs during training
- **Arm robustness**: trained with the Piper attached — randomized arm poses, moving servo targets, 0–0.5 kg wrist payload

Final training stats (6200 iters, 4096 envs): **0 falls, 0 NaNs**, vel-tracking error 0.23 m/s, mean landing force **11.9 N**, mean foot slip **0.025 m/s**.

<p align="center"><img src="videos/quiet_rollout.gif" alt="quiet rollout" width="640"></p>

**In-sim videos** (offscreen render of the training environment): [`videos/quiet_rollout.mp4`](videos/quiet_rollout.mp4) — 20 s goal-chasing rollout, randomized arm poses · [`videos/disturbance_test.mp4`](videos/disturbance_test.mp4) — same policy under training-strength pushes and ±15 N/5 N·m wrench bursts. Re-record for any checkpoint with `python scripts/record_video_quiet.py <policy.onnx>`.

**Action samples — before/after** (baseline goto policy vs the quiet retrain, same scripted goals per clip; full-quality mp4s in [`videos/actions/`](videos/actions/) and [`videos/actions/baseline/`](videos/actions/baseline/), recorded via `scripts/record_actions_quiet.py`):

*walk forward — two consecutive 2.5 m goals*
<p align="center"><img src="videos/actions/compare_walk_forward.gif" alt="walk forward old vs new" width="760"></p>

*turn around — goal placed directly behind → 180° turn*
<p align="center"><img src="videos/actions/compare_turn_around.gif" alt="turn around old vs new" width="760"></p>

*sidestep — lateral goals left, then right*
<p align="center"><img src="videos/actions/compare_sidestep.gif" alt="sidestep old vs new" width="760"></p>

*stand quiet — zero command, arm waves while the base must hold*
<p align="center"><img src="videos/actions/compare_stand_quiet_arm_wave.gif" alt="stand quiet old vs new" width="760"></p>

*push recovery — 0.7 m/s lateral shoves mid-walk*
<p align="center"><img src="videos/actions/compare_push_recovery.gif" alt="push recovery old vs new" width="760"></p>

*arm-swing walk — aggressive arm target changes every 1.2 s*
<p align="center"><img src="videos/actions/compare_arm_swing_walk.gif" alt="arm swing walk old vs new" width="760"></p>

W&B runs: [`dz8wkx58`](https://wandb.ai/zeroanuj/go2-quiet/runs/dz8wkx58) (iters 0–1200) · [`pm1t1u9i`](https://wandb.ai/zeroanuj/go2-quiet/runs/pm1t1u9i) (1200–6200)

## Sim2real



https://github.com/user-attachments/assets/82b8c323-6b40-440a-8d7e-92979e743e3b





## Repo layout

```
policies/policy.onnx                  # QUIET actor (obs[1,51] -> actions[1,12]) - this work
policies/baseline/policy.onnx         # ORIGINAL goto policy it replaces (same contract) - for A/B
policies/policy_descriptor.goto.json  # the deployment contract both policies satisfy
checkpoints/model_6199.pt             # rsl_rl checkpoint (actor+critic+optimizer)
checkpoints/env.yaml, agent.yaml      # full training config dumps
mjlab_tasks/go2_piper_quiet/          # mjlab task package (env cfg, command, rewards, events)
scripts/                              # train / smoke / preview / live-viewer entry points
media/                                # rollout preview
```

## The interface (what the policy expects)

**Observation — 51 floats, exact order** (`policy_descriptor.goto.json`):

| # | slice | content | source on robot |
|---|---|---|---|
| 1 | 0:3 | base angular velocity (body frame) | IMU gyro |
| 2 | 3:6 | projected gravity (body frame) | IMU orientation |
| 3 | 6:11 | commands `[Vx, Vy, YawRate, GaitSin, GaitCos]` | host controller |
| 4 | 11:23 | joint_pos − default_pos (12 legs, FL/FR/RL/RR × hip/thigh/calf) | encoders |
| 5 | 23:35 | joint velocities (12 legs) | encoders |
| 6 | 35:47 | previous raw action (12) | host controller |
| 7 | 47:48 | command `BodyHeight` (0.32 stand) | host controller |
| 8 | 48:51 | command `[TargetBX, TargetBY, TargetBZ]` — body-frame goal, TBZ=0 | host controller |

Note: **no base linear velocity** (no state estimator needed) and **no arm joints** in obs.

**Action — 12 floats** → leg joint position targets: `target = 0.25 * action + default_pos`, PD-tracked at **kp=20, kd=1 (calf: kp=40, kd=2)**, effort limit 23.5 N·m (calf 45), 50 Hz control (sim: 0.005 s × decimation 4). `default_pos`: hips ±0.1 (L+/R−), thigh 0.9, calf −1.8.

**Host-side command generation** (mirror of training, see `mdp_ext.GotoCommand`):
- Goal → body frame: `tbx = cosψ·dx + sinψ·dy`, `tby = −sinψ·dx + cosψ·dy`
- P-law: `v = clip(1.2·tb, ±0.6) · min(|tb|/0.8, 1)`; `wz = clip(1.2·atan2(tby,tbx), ±0.8)` if `|tb| > 0.3` else 0
- Slew-limit commands: ≤0.75 m/s² linear, ≤1.5 rad/s² yaw
- Gait clock: period **0.6 s**, advances only while `|v|+|wz| > 0.1`; send `sin/cos(2πφ)`, zeros when standing
- `BodyHeight = 0.32`

## Deploy

### LuckyEngine (Go2PiperGoto slot)

The obs layout is byte-identical to the shipped descriptor — only the weights change:

```powershell
# back up, then swap
cd "...\Assets\ContentVault\Robots\Unitree Go2 Piper\policies\goto"
copy policy.onnx policy.onnx.bak
copy <this repo>\policies\policy.onnx policy.onnx
# restart the scene; slot 4 (GOTO) now runs the quiet policy
```

Drive it exactly as before over gRPC (`SetPolicyCommandFloat`, cmd ids 1–9). Recommended host-gain softening to match training: KPOS 2.0→1.2, VMAX 1.0→0.6, OMEGA_MAX 1.5→0.8, SLOW_R 0.5→0.8.

### Real Go2

Run a 50 Hz loop: assemble the 51-dim obs from IMU + encoders + your command generator (table above), run `policy.onnx` (onnxruntime, CPU is plenty for a 512-256-128 MLP), send `0.25·a + default_pos` as joint position targets with the PD gains above (Unitree low-level SDK `q, kp, kd`). Start suspended, verify stand, then feet on ground. The policy expects the Piper mounted; without the arm the mass distribution is out-of-distribution — retrain or test carefully.

## Retrain / evaluate

Requirements: Python 3.10+, `mjlab==1.3.0` (+ mujoco ≥3.8, mujoco-warp, torch CUDA, rsl-rl), `onnxruntime`, `wandb login`. Works on native Windows (`MUJOCO_GL=wgl` for offscreen rendering).

One path to edit: `mjlab_tasks/go2_piper_quiet/assets.py::GO2_PIPER_XML` must point at your `go2_piper.xml` (the Go2+Piper MJCF with meshes — not redistributed here; it ships with LuckyEngine, and the Go2 leg meshes are from [unitree_ros](https://github.com/unitreerobotics/unitree_ros), BSD-3). The XML must contain the `imu` site sensors and `FL/FR/RL/RR` foot sites/geoms (stock naming).

```bash
python scripts/smoke_quiet.py                       # asserts obs==51, steps 4 envs
python scripts/train_quiet.py Mjlab-Velocity-Flat-Go2Piper-Quiet --env.scene.num-envs 4096
# resume:  --agent.resume True --agent.load-run <timestamp-dir>
python scripts/preview_policy_quiet.py policies/policy.onnx   # PNG rollout + quiet metrics
python scripts/play_quiet.py Mjlab-Velocity-Flat-Go2Piper-Quiet \
    --agent trained --checkpoint-file checkpoints/model_6199.pt --viewer viser  # localhost:8080
```

### Training recipe (see `env_cfg.py` for exact numbers)

- **Rewards**: velocity tracking (2.0/2.0) dominant; quiet penalties — contact force >40 N, landing impact, action rate/jerk, joint accel/torque, slip, 6 cm foot clearance — **ramped in 3 stages** (iters 0/500/1500) so the policy never collapses to standing
- **Commands**: wandering goal → P-law (the deploy distribution), 15% standing envs
- **Arm**: env-controlled servos; per-episode 80% folded±0.15 rad / 20% random-envelope pose, targets resampled every 3–5 s, 0–0.5 kg wrist payload
- **Forces**: velocity pushes every 8–15 s; `apply_body_impulse` wrench bursts on base and wrist
- **DR**: foot friction (slide/spin/roll), base mass ×0.9–1.15, COM ±2.5 cm, PD gains ±10%, encoder bias ±0.02 rad, obs noise on

## Attribution

Unitree Go2 model/meshes © Unitree Robotics (BSD-3 via unitree_ros); Piper arm © AgileX Robotics; trained with mjlab (Apache-2.0) and rsl_rl. Policy weights in this repo are released under MIT (see LICENSE).
