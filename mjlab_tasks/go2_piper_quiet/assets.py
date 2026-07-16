"""Go2+Piper entity for mjlab, matching the LE Go2PiperGoto slot descriptor.

Model: the exact MJCF LuckyEngine deploys (go2_piper.xml). The XML ships only
the 8 Piper arm/gripper position actuators; leg actuators are added here with
the PD constants from policy_descriptor.goto.json (kp 20/kd 1, calf 40/2,
effort 23.5/45, armature 0.01/0.02) so training dynamics == deploy dynamics.
"""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

GO2_PIPER_XML = Path(
  r"D:\FInal_Setup\LuckyEngine\go2_bungalow\Assets\ContentVault\Robots"
  r"\Unitree Go2 Piper\go2_piper.xml"
)
assert GO2_PIPER_XML.exists()

LEG_JOINT_EXPR = (
  "F[LR]_(hip|thigh|calf)_joint",
  "R[LR]_(hip|thigh|calf)_joint",
)
HIP_THIGH_EXPR = (".*_hip_joint", ".*_thigh_joint")
CALF_EXPR = (".*_calf_joint",)
FOOT_GEOMS = ("FL_foot_collision", "FR_foot_collision", "RL_foot_collision", "RR_foot_collision")
FOOT_SITES = ("FL", "FR", "RL", "RR")
BASE_BODY = "base_link"
ARM_WRIST_BODY = "piper_link6"

# Deploy-side folded arm home (DEFAULT_ARM8 in the rung gates / `home` keyframe).
ARM_FOLDED = (0.0, 1.2, -1.4, 0.0, 0.0, 0.0, 0.035, -0.035)
# Safe randomization envelope per arm dof (fingers held at grasp width).
ARM_ENVELOPE_LO = (-1.5, 0.3, -2.2, -1.0, -1.0, -1.5, 0.035, -0.035)
ARM_ENVELOPE_HI = (1.5, 2.4, -0.3, 1.0, 1.0, 1.5, 0.035, -0.035)


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(GO2_PIPER_XML))


# PD constants straight from policy_descriptor.goto.json.
HIP_THIGH_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=HIP_THIGH_EXPR,
  stiffness=20.0,
  damping=1.0,
  effort_limit=23.5,
  armature=0.01,
)
CALF_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=CALF_EXPR,
  stiffness=40.0,
  damping=2.0,
  effort_limit=45.0,
  armature=0.02,
)

ARTICULATION = EntityArticulationInfoCfg(
  actuators=(HIP_THIGH_ACTUATOR, CALF_ACTUATOR),
  soft_joint_pos_limit_factor=0.9,
)

_foot_regex = "^[FR][LR]_foot_collision$"
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  solref=(0.01, 1),
  condim={_foot_regex: 6, ".*_collision": 1},
  priority={_foot_regex: 1},
  friction={_foot_regex: (1, 5e-3, 5e-4)},
)

# Legs: descriptor default_pos (hips L +0.1 / R -0.1). Arm: folded home.
INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.30),
  joint_pos={
    "F[LR]_thigh_joint": 0.9,
    "R[LR]_thigh_joint": 0.9,
    ".*_calf_joint": -1.8,
    "FL_hip_joint": 0.1,
    "FR_hip_joint": -0.1,
    "RL_hip_joint": 0.1,
    "RR_hip_joint": -0.1,
    ".*piper_joint2": 1.2,
    ".*piper_joint3": -1.4,
    ".*piper_joint[1456]": 0.0,
    ".*piper_finger1": 0.035,
    ".*piper_finger2": -0.035,
  },
  joint_vel={".*": 0.0},
)


def get_go2_piper_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=ARTICULATION,
  )


# Descriptor action_scales: 0.25 uniform.
ACTION_SCALE = 0.25
