# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoActorCriticCfgTv0,
    RslRlPpoAlgorithmCfg,
)


@configclass
class UnitreeGo2RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "unitree_go2_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name= "ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class UnitreeGo2RoughPPORunnerCfgTv0(UnitreeGo2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 2000
        self.experiment_name = "unitree_go2_rough_tv0" # Transformer version 0
        self.policy = RslRlPpoActorCriticCfgTv0(
        class_name= "ActorCriticTransformerV0",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        d_actor=48,
        nhead_actor=1,
        num_layers_actor=2,
        dim_feedforward_actor=1024,
        dropout_actor=0.0,
        d_critic=48,
        nhead_critic=1,
        num_layers_critic=2,
        dim_feedforward_critic=1024,
        dropout_critic=0.0,
    )

        # self.algorithm.learning_rate = 1.0e-4
        



@configclass
class UnitreeGo2FlatPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 300
        self.experiment_name = "unitree_go2_flat"
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]




@configclass
class UnitreeGo2FlatPPORunnerCfgTv0(UnitreeGo2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 300*2
        self.experiment_name = "unitree_go2_flat_tv0" # Transformer version 0
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]
        self.policy.class_name = "ActorCriticTransformerV0"

        self.policy.d_actor = 48
        self.policy.nhead_actor = 1
        self.policy.num_layers_actor = 1
        self.policy.dim_feedforward_actor = 1024
        self.policy.dropout_actor = 0.0

        self.policy.d_critic = 48
        self.policy.nhead_critic = 1
        self.policy.num_layers_critic = 1
        self.policy.dim_feedforward_critic = 1024
        self.policy.dropout_critic = 0.0

        self.algorithm.learning_rate = 1.0e-3