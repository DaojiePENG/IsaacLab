# Copyright (c) 2022-2024, @Daojie PENG: Daojie.PENG@qq.com.
# All rights reserved.
#

"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import CommandTerm
from omni.isaac.lab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

    from .lm_commands_cfg import LMVelocityCommandCfg # pdj: configuration class for language-motion velocity commands 

# pdj: import language related packages
from transformers import AutoTokenizer, AutoModel 


class LMVelocityCommand(CommandTerm):
    r"""Command generator that generates a velocity command in SE(2) from uniform distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    If the :attr:`cfg.heading_command` flag is set to True, the angular velocity is computed from the heading
    error similar to doing a proportional control on the heading error. The target heading is sampled uniformly
    from the provided range. Otherwise, the angular velocity is sampled uniformly from the provided range.

    Mathematically, the angular velocity is computed as follows from the heading command:

    .. math::

        \omega_z = \frac{1}{2} \text{wrap_to_pi}(\theta_{\text{target}} - \theta_{\text{current}})

    """

    cfg: LMVelocityCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: LMVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # crete buffers to store the command
        # -- command: x vel, y vel, yaw vel, heading
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)

        # pdj: using dragon-multiturn context encoder to encode the velocity commands
        # 1. load language model. device_map=self.device; , device_map='cuda:0'
        self.tokenizer = AutoTokenizer.from_pretrained('nvidia/dragon-multiturn-query-encoder', device_map=self.device)
        self.context_encoder = AutoModel.from_pretrained('nvidia/dragon-multiturn-context-encoder', device_map=self.device)
        # 2. tokenize the expression string
        self.env_id_tokens = self.tokenizer(
            "I am running at longitudinal speed: {} m/s, lateral speed: {} m/s, spinning speed: {} m/s.".format(0.0, 0.0, 0.0), 
            padding=cfg.encodings.tokens_padding, 
            truncation=cfg.encodings.tokens_truncation, 
            max_length=cfg.encodings.tokens_max_length, 
            return_tensors='pt').to(self.device)
        # 3. encode the language tokens
        self.env_id_context = self.context_encoder(self.env_id_tokens['input_ids']).pooler_output # (1, emb_dim=768)
        # 4. init language_vel_command and apply the pooler to it
        # I want to embed all 3 vel into one embedding, so the shape should be [num_envs, 768] instead of the orignal [num_envs, 3];
        self.language_vel_command = torch.zeros(self.num_envs, self.env_id_context.shape[1], device=self.device) # define a tensor, data type is str, shape is [num_envs, 768]
        self.language_vel_command[:] = self.env_id_context
        # pdj: using dragon-multiturn context encoder to encode the velocity commands


        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)
        # -- metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "LMVelocityCommandCfg:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3 + 768)."""
        # return self.vel_command_b
        return torch.cat([self.vel_command_b, self.language_vel_command], dim=1)
    

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- ang vel yaw - rotation around z
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)

        # pdj: prepare the language_vel_command. directly apply language_vel_command;
        # 1. tokenize all with self.vel_command_b[0, x], because all the envs are using the same vel_command_b
        self.env_id_tokens = self.tokenizer(
                "I am running at longitudinal speed: {} m/s, lateral speed: {} m/s, spinning speed: {} m/s.".format(self.vel_command_b[0, 0], self.vel_command_b[0, 1], self.vel_command_b[0, 2]), 
                padding=self.cfg.encodings.tokens_padding, 
                truncation=self.cfg.encodings.tokens_truncation, 
                max_length=self.cfg.encodings.tokens_max_length, 
                return_tensors='pt').to(self.device)
        # 2. pooling the expression tokens
        self.env_id_context = self.context_encoder(self.env_id_tokens['input_ids']).pooler_output # (1, emb_dim=768)
        # 3. update language commands tokens
        self.language_vel_command[env_ids] = self.env_id_context
        # pdj: prepare the language_vel_command. directly apply language_vel_command.


        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        # Compute angular velocity from heading direction
        if self.cfg.heading_command:
            # resolve indices of heading envs
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            # compute angular velocity
            heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
        # Enforce standing (i.e., zero velocity command) for standing envs
        # TODO: check if conversion is needed
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

        # pdj: prepare the language_vel_command. directly apply language_vel_command;
        # 1. tokenize all with self.vel_command_b[0, x], because all the envs are using the same vel_command_b
        self.env_id_tokens = self.tokenizer(
                "I am running at longitudinal speed: {} m/s, lateral speed: {} m/s, spinning speed: {} m/s.".format(self.vel_command_b[0, 0], self.vel_command_b[0, 1], self.vel_command_b[0, 2]), 
                padding=self.cfg.encodings.tokens_padding, 
                truncation=self.cfg.encodings.tokens_truncation, 
                max_length=self.cfg.encodings.tokens_max_length, 
                return_tensors='pt').to(self.device)
        # 2. pooling the expression tokens
        self.env_id_context = self.context_encoder(self.env_id_tokens['input_ids']).pooler_output # (1, emb_dim=768)
        # 3. update language commands tokens
        self.language_vel_command[standing_env_ids] = self.env_id_context
        # pdj: prepare the language_vel_command. directly apply language_vel_command.



    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat
  