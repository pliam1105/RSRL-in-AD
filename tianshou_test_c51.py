import argparse
import os
import pickle

import torch, numpy as np
from torch import nn

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
        "type": "DiscreteMetaAction",
        # "type": "DiscreteAction", # thankfully that works, discretizes 3x3 = 9 combinations of steering and speed controls, use that since it fits with the algorithms and q value structure
        # "longitudinal": True,
        # "lateral": True,
        # "actions_per_axis": 5 # change that for discretization accuracy
    },
    "lanes_count": 5, # change that for randomization, less chaos
    "vehicles_count": 20, # default: 50 change that for more randomization and chaos
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

""" POLICY NETWORK """
class CNNnet(nn.Module):
        def __init__(self, obs_space: gym.Space, action_shape, num_atoms):
            super().__init__()
            self.num_atoms = num_atoms
            self.cnn = nn.Sequential(
                nn.Conv2d(obs_space.shape[0], 32, kernel_size=8, stride=4, padding=0,),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten()
            )
            # Compute shape by doing one forward pass
            with torch.no_grad():
                n_flatten = self.cnn(torch.as_tensor(obs_space.sample()[None]).float()).shape[1]
            
            self.linear = nn.Sequential(
                nn.Linear(n_flatten, 256),
                nn.ReLU(),
                nn.Linear(256, int(np.prod(action_shape)) * num_atoms),
            )

        def forward(self, obs, state=None, info={}):
            if not isinstance(obs, torch.Tensor):
                obs = torch.tensor(obs, dtype=torch.float)
            logits = self.linear(self.cnn(obs))
            batch_size = logits.shape[0]
            logits = logits.view(batch_size, -1, self.num_atoms)
            logits = torch.softmax(logits, dim=-1)
            return logits, state

""" PARAMETERS """
num_train_envs = 10 # change that for number of environments in parallel (although sequentially) in training
#change min and max values for return range
min_return = -100
max_return = 100

batch_size = 64
training_num = 8

eps_train = 0.25 #default: 0.1
eps_test = 0 #default: 0.05
eps_step_init = 10000
eps_step_final = 50000

reward_threshold = None

if __name__ == "__main__":
    env = create_env(None)
    env.reset()
    train_envs = DummyVectorEnv([lambda: create_env(None) for _ in range(num_train_envs)])
    test_envs = DummyVectorEnv([lambda: create_env(None) for _ in range(10)])
    print(env.observation_space, env.action_space)
    print(env.observation_space.shape, env.action_space.shape or env.action_space.n)

    net = CNNnet(
        obs_space=env.observation_space,
        action_shape=env.action_space.shape or env.action_space.n,
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
        estimation_step=3,
        target_update_freq=320
    )

    train_collector = Collector(policy, train_envs, VectorReplayBuffer(20000, num_train_envs), exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)

    #TensorBoard logging
    writer = SummaryWriter("log/C51")
    logger = TensorboardLogger(writer)

    def train_fn(epoch: int, env_step: int) -> None:
        # eps annnealing, just a demo
        if env_step <= eps_step_init:
            policy.set_eps(eps_train)
        elif env_step <= eps_step_final:
            eps = eps_train - (env_step - 10000) / 40000 * (0.9 * eps_train)
            policy.set_eps(eps)
        else:
            policy.set_eps(0.1 * eps_train)

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
        train_fn=train_fn,
        test_fn=lambda epoch, env_step: policy.set_eps(eps_test),
        logger=logger,
    )

    # policy.load_state_dict(torch.load('C51.pth')) # to load existing policy

    for epoch_stat in trainer:
        torch.save(policy.state_dict(), 'C51.pth') # to save the policy in a file
        print(epoch_stat)
        policy.eval()
        # policy.set_eps(eps_test)
        env = create_env("human")
        buf = VectorReplayBuffer(20000, num_train_envs)
        collector = ts.data.Collector(policy, env, buf, exploration_noise=True)
        collector.reset()
        collector.collect(n_episode=10, render=1 / 200)
        env.close()
        print(np.min(buf.act))
        print(np.max(buf.act))
        policy.train()