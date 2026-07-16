"""Registers Mjlab-Velocity-Flat-Go2Piper-Quiet (import side effect)."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from mjlab_tasks.go2_piper_quiet.env_cfg import go2_piper_quiet_env_cfg
from mjlab_tasks.go2_piper_quiet.rl_cfg import go2_piper_quiet_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Go2Piper-Quiet",
  env_cfg=go2_piper_quiet_env_cfg(),
  play_env_cfg=go2_piper_quiet_env_cfg(play=True),
  rl_cfg=go2_piper_quiet_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
