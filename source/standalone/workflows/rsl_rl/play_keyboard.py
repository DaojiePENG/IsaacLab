# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.utils.dict import print_dict

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
import omni.appwindow
import carb.input
from carb.input import KeyboardEventType

class KeyboardControl():
    '''Class for handling the keyboard events to generate commands for robot velocity control.
    Follow this description to define and use keyboard envents of omniverse:
    https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html#handle-keyboard-events
    '''
    def __init__(self, commands: torch.tensor, device: str):
        # subscribe to keyboard events
        app_window = omni.appwindow.get_default_app_window()
        self.keyboard = app_window.get_keyboard()
        input = carb.input.acquire_input_interface()
        self.keyboard_sub_id = input.subscribe_to_keyboard_events(self.keyboard, self.on_keyboard_input)
        # initialize commands
        self.commands = commands
        self.device = device
        self._input_keyboard_mapping = {
            # forward command
            "NUMPAD_8": [1.0, 0.0, 0.0],
            "UP": [1.0, 0.0, 0.0],
            # back command
            "NUMPAD_2": [-1.0, 0.0, 0.0],
            "DOWN": [-1.0, 0.0, 0.0],
            # left command
            "NUMPAD_6": [0.0, -0.5, 0.0],
            "RIGHT": [0.0, -0.5, 0.0],
            # right command
            "NUMPAD_4": [0.0, 0.5, 0.0],
            "LEFT": [0.0, 0.5, 0.0],
            # yaw command (positive)
            "NUMPAD_7": [0.0, 0.0, 2.0],
            "N": [0.0, 0.0, 2.0],
            # yaw command (negative)
            "NUMPAD_9": [0.0, 0.0, -2.0],
            "M": [0.0, 0.0, -2.0],
        }
    def __del__(self, name):
        # unsubscribe the keyboard event handler
        self.keyboard_sub_id = None

    def on_keyboard_input(self, e):
        # '''when a key is pressedor released  the command is adjusted w.r.t the key-mapping for position control'''
        # if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
        #     # on pressing, the command is incremented
        #     if e.input.name in self._input_keyboard_mapping:
        #         self.commands[:, :3] += torch.tensor(self._input_keyboard_mapping[e.input.name], device=self.device)
        # elif e.type == carb.input.KeyboardEventType.KEY_RELEASE:
        #     # on release, the command is decremented
        #     if e.input.name in self._input_keyboard_mapping:
        #         self.commands[:, :3] -= torch.tensor(self._input_keyboard_mapping[e.input.name], device=self.device)
        
        # v_x: longitudinal forward and backward
        if e.input == carb.input.KeyboardInput.UP:
            if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                self.commands[:, 0] = 1.0
                self._handle_key_pressed(e.input, self.commands)
            elif e.type == KeyboardEventType.KEY_RELEASE:
                self.commands[:, 0] = 0.0
                self._shandle_key_released(e.input, self.commands)
        if e.input == carb.input.KeyboardInput.DOWN:
            if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                self.commands[:, 0] = -1.0
                self._handle_key_pressed(e.input, self.commands)
            elif e.type == KeyboardEventType.KEY_RELEASE:
                self.commands[:, 0] = 0.0
                self._shandle_key_released(e.input, self.commands)
        # v_y: lateral left and right
        if e.input == carb.input.KeyboardInput.COMMA:
            if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                self.commands[:, 1] = 0.5
                self._handle_key_pressed(e.input, self.commands)
            elif e.type == KeyboardEventType.KEY_RELEASE:
                self.commands[:, 1] = 0.0
                self._shandle_key_released(e.input, self.commands)
        if e.input == carb.input.KeyboardInput.PERIOD:
            if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                self.commands[:, 1] = -0.5
                self._handle_key_pressed(e.input, self.commands)
            elif e.type == KeyboardEventType.KEY_RELEASE:
                self.commands[:, 1] = 0.0
                self._shandle_key_released(e.input, self.commands)
        # w_z: spin left and right
        if e.input == carb.input.KeyboardInput.LEFT:
            if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                self.commands[:, 2] = 1.5
                self._handle_key_pressed(e.input, self.commands)
            elif e.type == KeyboardEventType.KEY_RELEASE:
                self.commands[:, 2] = 0.0
                self._shandle_key_released(e.input, self.commands)
        if e.input == carb.input.KeyboardInput.RIGHT:
            if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                self.commands[:, 2] = -1.5
                self._handle_key_pressed(e.input, self.commands)
            elif e.type == KeyboardEventType.KEY_RELEASE:
                self.commands[:, 2] = 0.0
                self._shandle_key_released(e.input, self.commands)

    def _handle_key_pressed(self, e_input, commands):
        print('-' * 16 + e_input.name + ' is press or repeat' + '-' * 16 )
        print('Present commands: v_x, v_y, w_z:\n     {}'.format(commands))

    def _shandle_key_released(self, e_input, commands):
        print('-' * 16 + e_input.name + ' is released' + '-' * 16 )
        print('Present commands: v_x, v_y, w_z:\n     {}'.format(commands))

    



def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # pdj: modify env_cfg
    env_cfg.scene.num_envs = 1 # only create 1 robot for keyboard control
    env_cfg.episode_length_s = 1200 # set a longer episode length for continuous exploring

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # instanciate keyboard control
    keyboard_control: KeyboardControl = KeyboardControl(torch.zeros_like(obs[:, 9:12]), env.unwrapped.device)
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            obs[:, 9:12] = keyboard_control.commands # call keyboard function to update commands
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)

            """ keyboard contrl for robot """
            # obs[:, 9:12] = torch.tensor([0.5, 0.0, 0.0]) # manually set all agent to go forward.
            # however, this doesn't influence the debug visuls. 
            # I need get familiar with isaacsim and isaaclab API.
            # 1. monitor the keyboard events;
            # 2. define the commands for each keybard event by modify the obs tensor;
            # 3. change the corresponding debug visul arrows according to the changed commands;
            """ keyboard contrl for robot """

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
