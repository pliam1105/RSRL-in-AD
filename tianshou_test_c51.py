""" code based on https://github.com/thu-ml/tianshou/blob/master/test/discrete/test_c51.py and
    modified for my specific use-case
"""
import argparse
import os
import pickle

import tianshou as ts
import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import (
    Collector,
    PrioritizedVectorReplayBuffer,
    ReplayBuffer,
    VectorReplayBuffer,
)
from tianshou.env import DummyVectorEnv
from tianshou.policy import C51Policy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.space_info import SpaceInfo

from highway_env.envs.highway_env import HighwayEnv

env_config = {
    # "manual_control": True,
    # check the observation configs to align with the one the algorithms use (Atari -> RGB/Grayscale?, and check stack config if it's applied before/after input)
    "observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (128, 64),
        "stack_size": 4, #default: 4 for stacking, 1 if no stacking
        "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
        "scaling": 1.75,
    },
    "policy_frequency": 2,  # [Hz] # frequency with which actions are chosen (the agent acts)
    "action": {
        # "type": "DiscreteAction", # thankfully that works, discretizes 3x3 = 9 combinations of steering and speed controls, use that since it fits with the algorithms and q value structure
        "type": "DiscreteMetaAction"
        # "longitudinal": True,
        # "lateral": True,
        # "actions_per_axis": 5 # change that for discretization accuracy
    },
    "lanes_count": 5, # change that for randomization, less chaos
    "vehicles_count": 5, # default: 50 change that for more randomization and chaos
    "duration": 40,  # [s] # default: 40, change the episode length in seconds, maybe bigger length for more risk-aversity
    "initial_spacing": 2,
    "collision_reward": -1,  # default: -1, the reward received when colliding with a vehicle, change it to value risk
    # "reward_speed_range": [20, 30],  # [m/s] The reward for high speed is mapped linearly from this range to [0, HighwayEnv.HIGH_SPEED_REWARD].
    # "simulation_frequency": 15,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,  # [px]
    "screen_height": 150,  # [px]
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False,
    # "offroad_terminal": True # VERY IMPORTANT TO NOT GET OUT OF TRACK (LITERALLY, PUN INTENTED)
}

def create_env(render_mode):
    env: HighwayEnv = gym.make("highway-fast-v0", render_mode=render_mode)
    env.configure(env_config)
    return env

""" PARAMETERS """
num_train_envs = 10 # change that for number of environments in parallel (although sequentially) in training
#change min and max values for return range
min_return = -100
max_return = 100

batch_size = 64
training_num = 8

reward_threshold = None

if __name__ == "__main__":
    env = create_env(None)
    env.reset()
    train_envs = DummyVectorEnv([lambda: create_env(None) for _ in range(num_train_envs)])
    test_envs = DummyVectorEnv([lambda: create_env(None) for _ in range(10)])
    print(env.observation_space, env.action_space)
    print(env.observation_space.shape, env.action_space.shape or env.action_space.n)

    net = Net(
        state_shape = env.observation_space.shape,
        action_shape = env.action_space.shape or env.action_space.n,
        hidden_sizes=[128, 128, 128, 128],
        softmax=True,
        num_atoms=51
    )
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)
    policy: C51Policy = C51Policy(
        model=net,
        optim=optim,
        observation_space=env.observation_space,
        action_space=env.action_space,
        discount_factor=0.9,
        num_atoms=51,
        v_min=min_return, 
        v_max=max_return,
        # estimation_step=3,
        # target_update_freq=320
    )

    train_collector = Collector(policy, train_envs, VectorReplayBuffer(20000, num_train_envs), exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)

    # train_collector.reset()
    # train_collector.collect(n_step=batch_size*training_num)

    # may need to change that based on that environment, if not set correctly or taking forever to complete
    # reward_threshold = env.spec.reward_threshold

    #TensorBoard logging
    writer = SummaryWriter("log/C51")
    logger = TensorboardLogger(writer)

    trainer = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=50, #default: 10, preferred: 20-50? and then should work pretty well
        step_per_epoch=1000, #default: 8000, preferred: 1000-5000
        step_per_collect=10, #default: 8, preferred: 10
        # episode_per_collect=1,
        update_per_step=0.125, #default: 0.125
        episode_per_test=50, #default: 100, preferred: 10-50
        batch_size=batch_size,
        train_fn=lambda epoch, env_step: policy.set_eps(0.25), #default: 0.1
        test_fn=lambda epoch, env_step: policy.set_eps(0.05),
        # stop_fn=lambda mean_rewards: mean_rewards >= reward_threshold
        logger=logger,
    )

    for epoch_stat in trainer:
        print(epoch_stat)
        policy.eval()
        # policy.set_eps(0.05) #default: 0.05
        env = create_env("human")
        buf = VectorReplayBuffer(20000, num_train_envs)
        collector = ts.data.Collector(policy, env, buf, exploration_noise=True)
        collector.reset()
        collector.collect(n_episode=10, render=1 / 200)
        env.close()
        print(np.std(buf.act))
        policy.train()