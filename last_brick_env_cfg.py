# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class LastBrickEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    action_space = 3
    observation_space = # TODO how much observation to use?
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)


    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=32, env_spacing=4.0, replicate_physics=True)

    # custom parameters/scales
    # - action scale
    action_scale = 100.0  # [N]
    # - reward scales
    rew_scale_x_pos = -1.0
    rew_scale_y_pos = -1.0
    rew_scale_z_pos = -3.0
    rew_scale_x_ori = -1.0
    rew_scale_y_ori = -1.0
    rew_scale_z_ori = -1.0
    rew_scale_ang_vel = -1.0

    # - reset states/conditions
"""
    last_brick_final_pos_threshold = (0.005, 0.01, 0.01)
    last_brick_final_ori_threshold = 0.1

    last_brick_pos_range = 0.1
    last_brick_ori_range = 0.1
    last_brick_vel_range = 0.1
"""
    

