# Copyright (c) 2022-2024, @Daojie PENG: Daojie.PENG@qq.com.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`omni.isaac.lab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

"""
Commands.
"""

def vlnm_generated_commands_d(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    '''The generated digital command from language command term in the command manager with the given name.'''
    return env.command_manager.get_command(command_name)[:, :3] # pdj: only input encoded language commands [num_envs, 768]

def vlnm_generated_commands_l(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    '''The generated language command from language command term in the command manager with the given name.'''
    return env.command_manager.get_command(command_name)[:, 3:] # pdj: only input encoded language commands [num_envs, 768]