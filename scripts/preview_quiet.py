"""Render preview frames of the quiet Go2+Piper training env -> PNG grid."""

import os
import sys
from pathlib import Path

os.environ["MUJOCO_GL"] = "wgl"  # Windows offscreen GL (before mujoco import)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

import mjlab_tasks.go2_piper_quiet  # noqa: F401
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

OUT = Path(r"D:\go2_bungalow_bc\cache\quiet_preview")
OUT.mkdir(parents=True, exist_ok=True)

cfg = load_env_cfg("Mjlab-Velocity-Flat-Go2Piper-Quiet")
cfg.scene.num_envs = 3
cfg.viewer.height, cfg.viewer.width = 480, 640
device = "cuda" if torch.cuda.is_available() else "cpu"
env = ManagerBasedRlEnv(cfg, device=device, render_mode="rgb_array")
env.reset()

frames = []
for i in range(120):  # ~2.4 s sim time
  a = torch.zeros(env.num_envs, 12, device=device)
  env.step(a)
  if i in (0, 30, 60, 110):
    frames.append(env.render())

grid = np.vstack(
  [np.hstack(frames[:2]), np.hstack(frames[2:])]
)
import cv2

cv2.imwrite(str(OUT / "preview_grid.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
print("wrote", OUT / "preview_grid.png", grid.shape)
env.close()
