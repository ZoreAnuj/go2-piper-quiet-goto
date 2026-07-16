"""Smoke check: build 4 envs, verify actor obs == 51 (descriptor), step random.

The single runnable check for the task package: fails loudly if obs layout,
action dim, arm events, or the goto command break.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import mjlab_tasks.go2_piper_quiet  # noqa: F401
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

cfg = load_env_cfg("Mjlab-Velocity-Flat-Go2Piper-Quiet")
cfg.scene.num_envs = 4
device = "cuda" if torch.cuda.is_available() else "cpu"
env = ManagerBasedRlEnv(cfg, device=device)
obs, _ = env.reset()
actor = obs["actor"]
print("actor obs:", tuple(actor.shape), "critic obs:", tuple(obs["critic"].shape))
assert actor.shape[-1] == 51, f"descriptor wants 51 dims, got {actor.shape[-1]}"
assert env.action_manager.total_action_dim == 12, env.action_manager.total_action_dim

for i in range(50):
  a = torch.randn(env.num_envs, 12, device=device) * 0.1
  obs, rew, term, trunc, extras = env.step(a)
  assert torch.isfinite(obs["actor"]).all(), f"non-finite obs at step {i}"
cmd = env.command_manager.get_command("goto")
print("goto cmd sample:", cmd[0].tolist())
print("reward terms:", list(cfg.rewards.keys()))
print("SMOKE OK")
env.close()
