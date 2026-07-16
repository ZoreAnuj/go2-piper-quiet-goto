"""Custom MDP terms for the quiet Go2+Piper goto task.

GotoCommand reproduces the deploy-side command distribution exactly: a wandering
world goal is converted to a body-frame target (TargetBX/BY/BZ) and velocities
via the same P-law tier2.position_goal_cmd runs at deploy (softened gains), plus
the deploy gait clock (0.6 s period, advances only when |cmd| > stand mask).
Command layout (9): [vx, vy, wz, gait_sin, gait_cos, body_height, tbx, tby, tbz]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


##
# Goto command (velocity + gait clock + height + body-frame target).
##


class GotoCommand(CommandTerm):
  cfg: GotoCommandCfg

  def __init__(self, cfg: GotoCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    n, dev = self.num_envs, self.device
    self.goal_w = torch.zeros(n, 2, device=dev)
    self.phase = torch.zeros(n, device=dev)
    self.cmd = torch.zeros(n, 9, device=dev)
    self.cmd[:, 5] = cfg.body_height
    self.is_standing_env = torch.zeros(n, dtype=torch.bool, device=dev)
    self.metrics["error_vel_xy"] = torch.zeros(n, device=dev)
    self.metrics["error_vel_yaw"] = torch.zeros(n, device=dev)

  @property
  def command(self) -> torch.Tensor:
    return self.cmd

  def _update_metrics(self) -> None:
    max_t = self.cfg.resampling_time_range[1] / self._env.step_dt
    self.metrics["error_vel_xy"] += (
      torch.norm(self.cmd[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1)
      / max_t
    )
    self.metrics["error_vel_yaw"] += (
      torch.abs(self.cmd[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2]) / max_t
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    r = torch.empty(n, device=self.device)
    radius = r.uniform_(*self.cfg.goal_radius).clone()
    theta = torch.empty(n, device=self.device).uniform_(-math.pi, math.pi)
    pos = self.robot.data.root_link_pos_w[env_ids, :2]
    self.is_standing_env[env_ids] = (
      torch.empty(n, device=self.device).uniform_(0, 1) <= self.cfg.rel_standing_envs
    )
    offset = torch.stack([radius * torch.cos(theta), radius * torch.sin(theta)], dim=-1)
    offset[self.is_standing_env[env_ids]] = 0.0  # standing envs: goal under the robot
    self.goal_w[env_ids] = pos + offset
    self.cmd[env_ids, 5] = self.cfg.body_height

  def _update_command(self) -> None:
    dt = self._env.step_dt
    c = self.cfg
    pos = self.robot.data.root_link_pos_w[:, :2]
    heading = self.robot.data.heading_w
    d = self.goal_w - pos
    ch, sh = torch.cos(heading), torch.sin(heading)
    tbx = ch * d[:, 0] + sh * d[:, 1]
    tby = -sh * d[:, 0] + ch * d[:, 1]

    # StepGoto P-law (softened): ramp inside slow_r, no heading spin inside head_dz.
    dxy = torch.sqrt(tbx**2 + tby**2)
    ramp = torch.clamp(dxy / c.slow_radius, max=1.0)
    vx = torch.clamp(c.kpos * tbx, -c.vmax, c.vmax) * ramp
    vy = torch.clamp(c.kpos * tby, -c.vmax, c.vmax) * ramp
    yaw_err = torch.atan2(tby, tbx)
    wz = torch.clamp(c.kyaw * yaw_err, -c.omega_max, c.omega_max) * (
      dxy > c.head_deadzone
    )

    # Slew-limit velocity commands (quiet: no step commands).
    dv, dw = c.accel_max * dt, c.alpha_max * dt
    vx = self.cmd[:, 0] + torch.clamp(vx - self.cmd[:, 0], -dv, dv)
    vy = self.cmd[:, 1] + torch.clamp(vy - self.cmd[:, 1], -dv, dv)
    wz = self.cmd[:, 2] + torch.clamp(wz - self.cmd[:, 2], -dw, dw)

    # Deploy gait clock: advance only when commanded to move.
    moving = (torch.sqrt(vx**2 + vy**2) + torch.abs(wz)) > c.stand_mask
    self.phase = torch.where(
      moving, (self.phase + dt / c.gait_period) % 1.0, self.phase
    )
    two_pi = 2.0 * math.pi
    gsin = torch.where(moving, torch.sin(two_pi * self.phase), torch.zeros_like(vx))
    gcos = torch.where(moving, torch.cos(two_pi * self.phase), torch.zeros_like(vx))

    self.cmd[:, 0], self.cmd[:, 1], self.cmd[:, 2] = vx, vy, wz
    self.cmd[:, 3], self.cmd[:, 4] = gsin, gcos
    self.cmd[:, 6], self.cmd[:, 7] = tbx, tby
    self.cmd[:, 8] = 0.0


@dataclass(kw_only=True)
class GotoCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  goal_radius: tuple[float, float] = (0.5, 3.0)
  rel_standing_envs: float = 0.15
  # Softened deploy P-law gains (tier2: KPOS 2.0, VMAX 1.0, KYAW 1.5, OMEGA 1.5).
  kpos: float = 1.2
  vmax: float = 0.6
  kyaw: float = 1.2
  omega_max: float = 0.8
  slow_radius: float = 0.8
  head_deadzone: float = 0.3
  accel_max: float = 0.75  # m/s^2 command slew
  alpha_max: float = 1.5  # rad/s^2 command slew
  gait_period: float = 0.6  # deploy tier2.GAIT_PERIOD
  stand_mask: float = 0.1  # deploy tier2.STAND_MASK
  body_height: float = 0.32  # deploy tier2.H_STAND

  def build(self, env: ManagerBasedRlEnv) -> GotoCommand:
    return GotoCommand(self, env)


##
# Observations.
##


def command_slice(
  env: ManagerBasedRlEnv, command_name: str, start: int, end: int
) -> torch.Tensor:
  cmd = env.command_manager.get_command(command_name)
  assert cmd is not None
  return cmd[:, start:end]


##
# Rewards.
##


def contact_force_penalty(
  env: ManagerBasedRlEnv, sensor_name: str, threshold: float = 40.0
) -> torch.Tensor:
  """Penalize foot contact force above threshold (impact noise)."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  mag = torch.norm(sensor.data.force, dim=-1)  # [B, F]
  excess = torch.clamp(mag - threshold, min=0.0)
  env.extras["log"]["Metrics/contact_force_max"] = mag.max()
  return torch.sum(excess**2, dim=1)


##
# Arm events (env-controlled Piper: legs are the only policy dofs).
##


class arm_pose_reset:
  """Per-episode arm pose: folded+jitter (default) or random-in-envelope."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._entity: Entity = env.scene[cfg.params["asset_cfg"].name]
    jn = list(cfg.params["joint_names"])
    self._joint_ids, names = self._entity.find_joints(jn, preserve_order=True)
    self._joint_ids = torch.tensor(self._joint_ids, device=env.device)
    act_names = self._entity.actuator_names
    ctrl_ids = [i for i, n in enumerate(act_names) if ("piper" in n or "gripper" in n)]
    assert len(ctrl_ids) == len(names) == 8, (act_names, names)
    self._ctrl_ids = torch.tensor(ctrl_ids, dtype=torch.int, device=env.device)
    self._device = env.device

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    folded: tuple[float, ...],
    lo: tuple[float, ...],
    hi: tuple[float, ...],
    jitter: float = 0.15,
    random_prob: float = 0.2,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_names: tuple[str, ...] = (),
  ) -> None:
    del asset_cfg, joint_names
    n = len(env_ids)
    dev = self._device
    folded_t = torch.tensor(folded, device=dev).expand(n, -1)
    lo_t = torch.tensor(lo, device=dev)
    hi_t = torch.tensor(hi, device=dev)
    rand_pose = lo_t + torch.rand(n, 8, device=dev) * (hi_t - lo_t)
    jittered = folded_t + (torch.rand(n, 8, device=dev) * 2 - 1) * jitter
    use_rand = (torch.rand(n, 1, device=dev) < random_prob).expand(-1, 8)
    pose = torch.where(use_rand, rand_pose, jittered).clamp(lo_t, hi_t)
    self._entity.write_joint_position_to_sim(pose, self._joint_ids, env_ids)
    self._entity.write_joint_velocity_to_sim(
      torch.zeros_like(pose), self._joint_ids, env_ids
    )
    self._entity.write_ctrl_to_sim(pose, self._ctrl_ids, env_ids)


class arm_motion_resample:
  """Interval event: move the arm servo targets (reaction-force robustness)."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._entity: Entity = env.scene[cfg.params["asset_cfg"].name]
    jn = list(cfg.params["joint_names"])
    joint_ids, names = self._entity.find_joints(jn, preserve_order=True)
    self._joint_ids = torch.tensor(joint_ids, device=env.device)
    act_names = self._entity.actuator_names
    ctrl_ids = [i for i, n in enumerate(act_names) if ("piper" in n or "gripper" in n)]
    assert len(ctrl_ids) == 8
    self._ctrl_ids = torch.tensor(ctrl_ids, dtype=torch.int, device=env.device)
    self._device = env.device

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    lo: tuple[float, ...],
    hi: tuple[float, ...],
    delta: float = 0.3,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_names: tuple[str, ...] = (),
  ) -> None:
    del asset_cfg, joint_names
    n = len(env_ids)
    dev = self._device
    cur = self._entity.data.joint_pos[env_ids][:, self._joint_ids]
    lo_t = torch.tensor(lo, device=dev)
    hi_t = torch.tensor(hi, device=dev)
    target = cur + (torch.rand(n, 8, device=dev) * 2 - 1) * delta
    self._entity.write_ctrl_to_sim(target.clamp(lo_t, hi_t), self._ctrl_ids, env_ids)


##
# Curriculum: ramp quiet penalties in stages (anti stand-collapse).
##


def penalty_ramp(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  stages: list[dict],
) -> dict[str, torch.Tensor]:
  """stages: [{"step": int, "weights": {reward_name: weight}}, ...]."""
  del env_ids
  scale_applied = 0
  for i, stage in enumerate(stages):
    if env.common_step_counter >= stage["step"]:
      scale_applied = i
      for name, w in stage["weights"].items():
        env.reward_manager.get_term_cfg(name).weight = w
  return {"penalty_stage": torch.tensor(float(scale_applied))}
