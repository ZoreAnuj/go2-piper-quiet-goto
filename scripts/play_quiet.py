"""Live-view the quiet Go2+Piper policy in mjlab's viser web viewer.

Usage:
  python scripts/play_quiet.py Mjlab-Velocity-Flat-Go2Piper-Quiet \
      --agent trained --checkpoint-file <model.pt> --viewer viser
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mjlab_tasks.go2_piper_quiet  # noqa: F401  (registers the task)
from mjlab.scripts.play import main

if __name__ == "__main__":
  main()
