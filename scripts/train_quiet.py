"""Train the quiet Go2+Piper goto policy in mjlab (wandb-logged).

Usage:
  python scripts/train_quiet.py Mjlab-Velocity-Flat-Go2Piper-Quiet \
      --env.scene.num-envs 4096
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mjlab_tasks.go2_piper_quiet  # noqa: F401  (registers the task)
from mjlab.scripts.train import main

if __name__ == "__main__":
  main()
