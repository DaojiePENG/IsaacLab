# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from .lm_rough_env_cfg import UnitreeGo2RoughEnvCfg


@configclass
class UnitreeGo2FlatEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        
        # pdj: disable default vel commands
        # self.commands.base_velocity = None

        # override rewards
        self.rewards.lm_flat_orientation_l2.weight = -2.5
        self.rewards.lm_feet_air_time.weight = 0.25

        # pdj: set a small num for developing, eg: 8; set a proper num for real training, eg: 4096.
        self.scene.num_envs = 8 * 1
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        


class UnitreeGo2FlatEnvCfg_PLAY(UnitreeGo2FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None

'''
launch commands:

python source/standalone/workflows/rsl_rl/train.py --task Isaac-LM-Velocity-Flat-Unitree-Go2-v0 --headless

python source/standalone/workflows/rsl_rl/play.py --task Isaac-LM-Velocity-Flat-Unitree-Go2-v0

'''