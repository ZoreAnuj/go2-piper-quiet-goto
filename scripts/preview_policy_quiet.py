"""Preview the TRAINED quiet policy (exported ONNX) — frames + quiet metrics."""

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
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

ONNX = sys.argv[1] if len(sys.argv) > 1 else (
  r"D:\go2_bungalow_bc\logs\rsl_rl\go2_piper_quiet\2026-07-15_18-31-44"
  r"\2026-07-15_18-31-44.onnx"
)
OUT = Path(r"D:\go2_bungalow_bc\cache\quiet_preview")
OUT.mkdir(parents=True, exist_ok=True)

sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name
batch_dim = sess.get_inputs()[0].shape[0]
print("onnx input:", in_name, sess.get_inputs()[0].shape)

cfg = load_env_cfg("Mjlab-Velocity-Flat-Go2Piper-Quiet", play=True)
cfg.scene.num_envs = 3
cfg.viewer.height, cfg.viewer.width = 480, 640
device = "cuda" if torch.cuda.is_available() else "cpu"
env = ManagerBasedRlEnv(cfg, device=device, render_mode="rgb_array")
obs, _ = env.reset()


def policy(o: torch.Tensor) -> torch.Tensor:
  x = o.cpu().numpy().astype(np.float32)
  if isinstance(batch_dim, int) and batch_dim == 1:
    a = np.vstack([sess.run(None, {in_name: x[i : i + 1]})[0] for i in range(len(x))])
  else:
    a = sess.run(None, {in_name: x})[0]
  return torch.from_numpy(a).to(device)


frames, landing, slip = [], [], []
for i in range(400):  # ~8 s
  obs, rew, term, trunc, extras = env.step(policy(obs["actor"]))
  log = env.extras.get("log", {})
  if "Metrics/landing_force_mean" in log:
    landing.append(float(log["Metrics/landing_force_mean"]))
  if "Metrics/slip_velocity_mean" in log:
    slip.append(float(log["Metrics/slip_velocity_mean"]))
  if i in (10, 130, 260, 390):
    frames.append(env.render())

grid = np.vstack([np.hstack(frames[:2]), np.hstack(frames[2:])])
cv2.imwrite(str(OUT / "policy_preview_grid.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
cmd = env.command_manager.get_command("goto")[0]
print("wrote", OUT / "policy_preview_grid.png")
print(f"mean landing force: {np.mean(landing):.1f} N   mean slip vel: {np.mean(slip):.3f} m/s")
print("cmd sample:", [round(float(v), 2) for v in cmd])
env.close()
