"""Record rollout videos of the trained quiet policy (offscreen sim render).

Writes: quiet_rollout.mp4 (play cfg), disturbance_test.mp4 (train cfg with
pushes + wrench bursts), and quiet_rollout.gif (README inline preview).
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
from PIL import Image

import mjlab_tasks.go2_piper_quiet  # noqa: F401
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

ONNX = sys.argv[1] if len(sys.argv) > 1 else (
  r"D:\go2_bungalow_bc\logs\rsl_rl\go2_piper_quiet\2026-07-15_19-21-53"
  r"\2026-07-15_19-21-53.onnx"
)
OUT = Path(r"D:\go2_bungalow_bc\cache\quiet_videos")
OUT.mkdir(parents=True, exist_ok=True)

sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name
batch_dim = sess.get_inputs()[0].shape[0]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def policy(o: torch.Tensor) -> torch.Tensor:
  x = o.cpu().numpy().astype(np.float32)
  if isinstance(batch_dim, int) and batch_dim == 1:
    a = np.vstack([sess.run(None, {in_name: x[i : i + 1]})[0] for i in range(len(x))])
  else:
    a = sess.run(None, {in_name: x})[0]
  return torch.from_numpy(a).to(DEVICE)


def record(play: bool, steps: int, path: Path) -> list[np.ndarray]:
  cfg = load_env_cfg("Mjlab-Velocity-Flat-Go2Piper-Quiet", play=play)
  if not play:  # keep training disturbances but don't reset mid-video
    cfg.episode_length_s = 60.0
    cfg.observations["actor"].enable_corruption = False
  cfg.scene.num_envs = 3
  cfg.viewer.height, cfg.viewer.width = 480, 640
  env = ManagerBasedRlEnv(cfg, device=DEVICE, render_mode="rgb_array")
  obs, _ = env.reset()
  vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 480))
  frames = []
  for i in range(steps):
    obs, *_ = env.step(policy(obs["actor"]))
    if i % 2 == 0:  # 50 Hz sim -> 25 fps video
      f = env.render()
      vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
      frames.append(f)
  vw.release()
  env.close()
  print("wrote", path, f"({len(frames)} frames)")
  return frames


frames = record(play=True, steps=1000, path=OUT / "quiet_rollout.mp4")  # 20 s
record(play=False, steps=1000, path=OUT / "disturbance_test.mp4")  # pushes+wrenches

# Inline README gif: 10 s, 12 fps, 480px wide.
gif = [
  Image.fromarray(f).resize((480, 360))
  for f in frames[: 12 * 10 * 2 : 2]  # every 2nd of the 25fps frames -> ~12fps
]
gif[0].save(
  OUT / "quiet_rollout.gif",
  save_all=True,
  append_images=gif[1:],
  duration=83,
  loop=0,
)
print("wrote", OUT / "quiet_rollout.gif")
