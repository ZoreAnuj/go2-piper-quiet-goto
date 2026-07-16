"""Quiet Go2+Piper goto-velocity task.

Actor obs = EXACTLY the 51-dim layout of policy_descriptor.goto.json:
  base_ang_vel(3) + projected_gravity(3) + [vx,vy,wz,gsin,gcos](5)
  + joint_pos_rel(12 legs) + joint_vel(12) + last_action(12)
  + body_height(1) + target_bxyz(3)
so the exported ONNX is a drop-in replacement for the LE goto slot.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, RingPatternCfg, TerrainHeightSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_tasks.go2_piper_quiet import mdp_ext
from mjlab_tasks.go2_piper_quiet.assets import (
  ACTION_SCALE,
  ARM_ENVELOPE_HI,
  ARM_ENVELOPE_LO,
  ARM_FOLDED,
  ARM_WRIST_BODY,
  BASE_BODY,
  FOOT_GEOMS,
  FOOT_SITES,
  LEG_JOINT_EXPR,
  get_go2_piper_robot_cfg,
)

LEG_ASSET = lambda: SceneEntityCfg("robot", joint_names=list(LEG_JOINT_EXPR))  # noqa: E731
LEG_ACTUATOR_EXPR = (".*_hip_joint", ".*_thigh_joint", ".*_calf_joint")
ARM_JOINT_NAMES = tuple(
  [f".*piper_joint{i}" for i in range(1, 7)] + [".*piper_finger1", ".*piper_finger2"]
)

# Quiet-penalty ramp: stage 0 weights live in the rewards dict; later stages
# tighten them (anti stand-collapse). Steps are env steps (24/iter).
PENALTY_STAGES = [
  {"step": 0, "weights": {}},
  {
    "step": 500 * 24,
    "weights": {
      "action_rate_l2": -0.1,
      "action_acc": -0.05,
      "joint_acc": -1.25e-7,
      "joint_torques": -1e-4,
      "contact_force": -1e-4,
      "soft_landing": -2e-3,
      "foot_slip": -0.15,
    },
  },
  {
    "step": 1500 * 24,
    "weights": {
      "action_rate_l2": -0.25,
      "action_acc": -0.1,
      "joint_acc": -2.5e-7,
      "joint_torques": -2e-4,
      "contact_force": -4e-4,
      "soft_landing": -5e-3,
      "foot_slip": -0.25,
    },
  },
]


def go2_piper_quiet_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_velocity_env_cfg()

  # --- scene: flat plane, deploy MJCF ---
  cfg.scene.entities = {"robot": get_go2_piper_robot_cfg()}
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.sim.njmax = 300
  cfg.sim.nconmax = None
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.mujoco.impratio = 10
  cfg.sim.mujoco.cone = "elliptic"
  cfg.sim.contact_sensor_maxmatch = 64

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=FOOT_GEOMS, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  sensors = []
  for s in cfg.scene.sensors or ():
    if s.name == "foot_height_scan":
      assert isinstance(s, TerrainHeightSensorCfg)
      s.frame = tuple(ObjRef(type="site", name=n, entity="robot") for n in FOOT_SITES)
      s.pattern = RingPatternCfg.single_ring(radius=0.04, num_samples=4)
      sensors.append(s)
  cfg.scene.sensors = tuple(sensors) + (feet_ground_cfg,)

  # --- actions: 12 legs only, descriptor semantics (0.25 * a + default_pos) ---
  cfg.actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=LEG_ACTUATOR_EXPR,
      scale=ACTION_SCALE,
      use_default_offset=True,
    )
  }

  # --- command: deploy-matched goto (P-law from wandering goal) ---
  cfg.commands = {
    "goto": mdp_ext.GotoCommandCfg(
      resampling_time_range=(5.0, 10.0),
      debug_vis=True,
    )
  }

  # --- observations: actor EXACTLY per descriptor (51 dims) ---
  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "cmd_twist_gait": ObservationTermCfg(
      func=mdp_ext.command_slice,
      params={"command_name": "goto", "start": 0, "end": 5},
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"asset_cfg": LEG_ASSET()},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      params={"asset_cfg": LEG_ASSET()},
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "cmd_height": ObservationTermCfg(
      func=mdp_ext.command_slice,
      params={"command_name": "goto", "start": 5, "end": 6},
    ),
    "cmd_target": ObservationTermCfg(
      func=mdp_ext.command_slice,
      params={"command_name": "goto", "start": 6, "end": 9},
    ),
  }
  critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}
    ),
    "foot_height": ObservationTermCfg(
      func=mdp.foot_height, params={"sensor_name": "foot_height_scan"}
    ),
    "foot_air_time": ObservationTermCfg(
      func=mdp.foot_air_time, params={"sensor_name": "feet_ground_contact"}
    ),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact, params={"sensor_name": "feet_ground_contact"}
    ),
    "foot_contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces, params={"sensor_name": "feet_ground_contact"}
    ),
  }
  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms, concatenate_terms=True, enable_corruption=not play
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms, concatenate_terms=True, enable_corruption=False
    ),
  }

  # --- rewards: tracking dominant, quiet penalties ramped by curriculum ---
  cfg.rewards["track_linear_velocity"].params["command_name"] = "goto"
  cfg.rewards["track_angular_velocity"].params["command_name"] = "goto"
  cfg.rewards["upright"].params["asset_cfg"].body_names = (BASE_BODY,)
  cfg.rewards["upright"].params.pop("terrain_sensor_names", None)
  cfg.rewards["pose"].params["command_name"] = "goto"
  cfg.rewards["pose"].params["asset_cfg"] = LEG_ASSET()
  cfg.rewards["pose"].params["std_standing"] = {
    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.05,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.1,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
  }
  del cfg.rewards["body_ang_vel"]
  del cfg.rewards["angular_momentum"]
  cfg.rewards["dof_pos_limits"].params = {"asset_cfg": LEG_ASSET()}

  cfg.rewards["air_time"].weight = 0.2
  cfg.rewards["air_time"].params["command_name"] = "goto"
  cfg.rewards["air_time"].params["command_threshold"] = 0.1
  for name in ("foot_clearance", "foot_swing_height", "foot_slip", "soft_landing"):
    cfg.rewards[name].params["command_name"] = "goto"
  cfg.rewards["foot_clearance"].params["target_height"] = 0.06
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = FOOT_SITES
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.06
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = FOOT_SITES

  # Stage-0 quiet weights (ramped up by curriculum).
  cfg.rewards["action_rate_l2"].weight = -0.02
  cfg.rewards["foot_slip"].weight = -0.1
  cfg.rewards["soft_landing"].weight = -5e-4
  cfg.rewards["action_acc"] = RewardTermCfg(func=envs_mdp.action_acc_l2, weight=-0.01)
  cfg.rewards["joint_acc"] = RewardTermCfg(
    func=envs_mdp.joint_acc_l2, weight=-5e-8, params={"asset_cfg": LEG_ASSET()}
  )
  cfg.rewards["joint_torques"] = RewardTermCfg(
    func=envs_mdp.joint_torques_l2, weight=-5e-5, params={"asset_cfg": LEG_ASSET()}
  )
  cfg.rewards["contact_force"] = RewardTermCfg(
    func=mdp_ext.contact_force_penalty,
    weight=-2e-5,
    params={"sensor_name": "feet_ground_contact", "threshold": 40.0},
  )

  # --- events: arm handling + external forces + tier-2 DR ---
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.05, 0.05)
  cfg.events["reset_robot_joints"].params["asset_cfg"] = LEG_ASSET()
  cfg.events["arm_pose"] = EventTermCfg(
    func=mdp_ext.arm_pose_reset,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "joint_names": ARM_JOINT_NAMES,
      "folded": ARM_FOLDED,
      "lo": ARM_ENVELOPE_LO,
      "hi": ARM_ENVELOPE_HI,
      "jitter": 0.15,
      "random_prob": 0.2,
    },
  )
  cfg.events["arm_motion"] = EventTermCfg(
    func=mdp_ext.arm_motion_resample,
    mode="interval",
    interval_range_s=(3.0, 5.0),
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "joint_names": ARM_JOINT_NAMES,
      "lo": ARM_ENVELOPE_LO,
      "hi": ARM_ENVELOPE_HI,
      "delta": 0.3,
    },
  )
  cfg.events["payload_mass"] = EventTermCfg(
    mode="startup",
    func=dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(ARM_WRIST_BODY,)),
      "operation": "add",
      "ranges": (0.0, 0.5),
    },
  )
  cfg.events["push_robot"].interval_range_s = (8.0, 15.0)
  cfg.events["push_robot"].params["velocity_range"] = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.2, 0.2),
    "pitch": (-0.2, 0.2),
    "yaw": (-0.3, 0.3),
  }
  cfg.events["base_wrench"] = EventTermCfg(
    func=envs_mdp.apply_body_impulse,
    mode="step",
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY,)),
      "force_range": (-15.0, 15.0),
      "torque_range": (-5.0, 5.0),
      "duration_s": (0.2, 0.5),
      "cooldown_s": (8.0, 15.0),
    },
  )
  cfg.events["ee_wrench"] = EventTermCfg(
    func=envs_mdp.apply_body_impulse,
    mode="step",
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(ARM_WRIST_BODY,)),
      "force_range": (-8.0, 8.0),
      "torque_range": (-1.0, 1.0),
      "duration_s": (0.2, 0.4),
      "cooldown_s": (10.0, 20.0),
    },
  )
  # Friction DR (go1-flat style, condim 6 feet).
  del cfg.events["foot_friction"]
  foot_asset = SceneEntityCfg("robot", geom_names=FOOT_GEOMS)
  cfg.events["foot_friction_slide"] = EventTermCfg(
    mode="startup",
    func=dr.geom_friction,
    params={
      "asset_cfg": foot_asset,
      "operation": "abs",
      "axes": [0],
      "ranges": (0.3, 1.25),
      "shared_random": True,
    },
  )
  cfg.events["foot_friction_spin"] = EventTermCfg(
    mode="startup",
    func=dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOMS),
      "operation": "abs",
      "distribution": "log_uniform",
      "axes": [1],
      "ranges": (1e-4, 2e-2),
      "shared_random": True,
    },
  )
  cfg.events["foot_friction_roll"] = EventTermCfg(
    mode="startup",
    func=dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOMS),
      "operation": "abs",
      "distribution": "log_uniform",
      "axes": [2],
      "ranges": (1e-5, 5e-3),
      "shared_random": True,
    },
  )
  cfg.events["base_com"].params["asset_cfg"].body_names = (BASE_BODY,)
  cfg.events["base_mass"] = EventTermCfg(
    mode="startup",
    func=dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY,)),
      "operation": "scale",
      "ranges": (0.9, 1.15),
    },
  )
  cfg.events["pd_gains"] = EventTermCfg(
    mode="startup",
    func=dr.pd_gains,
    params={
      # ponytail: all actuator groups (legs + arm servos) get +/-10% gain DR;
      # per-group selection indexes group objects, not per-actuator names.
      "asset_cfg": SceneEntityCfg("robot"),
      "kp_range": (0.9, 1.1),
      "kd_range": (0.9, 1.1),
      "operation": "scale",
    },
  )
  cfg.events["encoder_bias"].params["bias_range"] = (-0.02, 0.02)

  # --- terminations ---
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)}
    ),
    "nan_detection": TerminationTermCfg(func=envs_mdp.nan_detection, time_out=True),
  }

  # --- curriculum: penalty ramp only ---
  cfg.curriculum = {
    "penalty_ramp": CurriculumTermCfg(
      func=mdp_ext.penalty_ramp, params={"stages": PENALTY_STAGES}
    ),
  }

  cfg.viewer.body_name = BASE_BODY
  cfg.viewer.distance = 1.8
  cfg.viewer.elevation = -10.0

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.events.pop("push_robot", None)
    cfg.events.pop("base_wrench", None)
    cfg.events.pop("ee_wrench", None)
    cfg.curriculum = {}
    # Play at final penalty weights.
    for name, w in PENALTY_STAGES[-1]["weights"].items():
      cfg.rewards[name].weight = w

  return cfg
