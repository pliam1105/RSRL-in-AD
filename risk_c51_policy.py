import argparse
import os
import pickle

from typing import Callable

import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import scipy.stats as stats

import torch, numpy as np
from torch import nn, vmap

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
from tianshou.policy import C51Policy, DQNPolicy
from tianshou.policy.base import BasePolicy, TLearningRateScheduler
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
    "vehicles_count": 20, # default: 50, preferred: 10-20, change that for more randomization and chaos
    "duration": 40,  # [s] # default: 40, change the episode length in seconds, maybe bigger length for more risk-aversity
    "initial_spacing": 5, # default: 2
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

""" RISK POLICY IMPLEMENTATION """
class RiskC51Policy(C51Policy):
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        action_space: gym.spaces.Discrete,
        discount_factor: float = 0.99,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        is_double: bool = True,
        clip_loss_grad: bool = False,
        observation_space: gym.Space | None = None,
        lr_scheduler: TLearningRateScheduler | None = None,
        risk_function: Callable[[torch.Tensor, float], torch.Tensor] | None = None,
        risk_param: float = 0.0
    ) -> None:
        super().__init__(
            model=model,
            optim=optim,
            action_space=action_space,
            discount_factor=discount_factor,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            estimation_step=estimation_step,
            target_update_freq=target_update_freq,
            reward_normalization=reward_normalization,
            is_double=is_double,
            clip_loss_grad=clip_loss_grad,
            observation_space=observation_space,
            lr_scheduler=lr_scheduler
        )
        self.risk_function = risk_function
        self.risk_param = risk_param
    
    def compute_q_value(self, logits: torch.Tensor, mask: np.ndarray | None) -> torch.Tensor:
        if self.risk_function == None:
            return super().compute_q_value(logits, mask)
        else:
            # logits: [batch_size, action_size, atom_num] -> probabilities
            # self.support: [atom_num] -> values
            # asterisk does element-wise multiplication and sum is over the atoms for average, but we will do something else
            quantiles = torch.cumsum(logits, 2) # [batch_size, action_size, atom_num] -> quantiles
            # quantiles.requires_grad_(True)
            # quantile differences are actually the logits, so no need to compute them
            used_values = self.support.expand_as(quantiles) # [batch_size, action_size, atom_num], adapt values (supports) to that batched dimension
            assert used_values.shape == logits.shape
            # compute gradients of each output element wrt the respective input, not pair-wise like the jacobian
            # distorted_quantiles = torch.zeros_like(logits) # [batch_size, action_size, atom_num], quantiles passed from the distortion risk measure
            # distorted_quantile_grads = torch.zeros_like(logits) # [batch_size, action_size, atom_num], distortion risk measure derivatives at the quantiles
            # for batch in range(logits.shape[0]):
            #     for action in range(logits.shape[1]):
            #         distorted_quantiles[batch][action], distorted_quantile_grads[batch][action] = quantile_grads(quantiles[batch][action].detach().cpu().numpy(), self.risk_function, self.risk_param)
            
            def batched_quantile_grads(quant: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                return self.risk_function(quant,self.risk_param)
            # distorted_quantile_grads -> [batch_size, action_size, atom_num], distortion risk measure derivatives at the quantiles
            distorted_quantile_grads = torch.vmap(torch.vmap(batched_quantile_grads))(quantiles)
            # now we want the sum (in the atoms' dimension) of element-wise products of logits (quantile differences), distorted_quantile_grads, and used_values
            assert distorted_quantile_grads.shape == logits.shape
            distorted_expectation = torch.sum(logits*distorted_quantile_grads*used_values, 2)
            assert distorted_expectation.shape == logits.shape[:2]
            return DQNPolicy.compute_q_value(self, distorted_expectation, mask)

""" DISTORTION RISK MEASURE DERIVATIVES IMPLEMENTATION"""
def Neutral(quantiles: torch.Tensor, beta: float = 0.0) -> torch.Tensor:
    return torch.ones_like(quantiles)

def CVaR(quantiles: torch.Tensor, beta: float) -> torch.Tensor:
    return 1.0/beta*torch.le(quantiles, torch.full_like(quantiles, beta))

def Wang(quantiles: torch.Tensor, beta: float) -> torch.Tensor:
    x = torch.vmap(torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0])).icdf)(quantiles)[:,0]
    return torch.exp(-beta*x-torch.full_like(x,(beta**2)/2))

def CPW(x: torch.Tensor, b: float) -> torch.Tensor:
    return b*(x**(b-1))*((x**b+(1-x)**b)**(-1/b)) - (x**b)*(b*(x**(b-1))-b*((1-x)**(b-1)))*((x**b+(1-x)**b)**(-1/(b-1)))/b

# for a one-dimensional tensor
def RiskCompute(returns: torch.Tensor, risk_function:Callable[[torch.Tensor, float], torch.Tensor], risk_param: float):
    assert returns.dim() == 1
    logits = torch.full_like(returns, 1/returns.shape[0]) # dirac mixture
    quantiles = torch.cumsum(logits, 0)
    distorted_quantile_grads = risk_function(quantiles, risk_param)
    return torch.sum(logits*distorted_quantile_grads*returns)


""" PARAMETERS """
num_train_envs = 10 # change that for number of environments in parallel (although sequentially) in training
#change min and max values for return range
num_test_envs = 10
num_eval_envs = 50
# num_test_episodes = 10
# num_eval_episodes = 50
min_return = -100
max_return = 100

batch_size = 64
training_num = 8

eps_train = 0.25 #default: 0.1
eps_test = 0 #default: 0.05
eps_step_init = 10000
eps_step_final = 50000

reward_threshold = None

def run_experiment(risk_measure="neutral", beta=0.0, load_policy=False, train_policy=False):
    print("RUNNING EXPERIMENT:",risk_measure, beta)
    env = create_env(None)
    env.reset()
    train_envs = DummyVectorEnv([lambda: create_env(None) for _ in range(num_train_envs)])
    test_envs = DummyVectorEnv([lambda: create_env(None) for _ in range(num_test_envs)])
    # print(env.observation_space, env.action_space)
    # print(env.observation_space.shape, env.action_space.shape or env.action_space.n)

    net = CNNnet(
        obs_space=env.observation_space,
        action_shape=env.action_space.shape or env.action_space.n,
        num_atoms=51
    )
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)

    if risk_measure == "neutral":
        risk_func = Neutral
    elif risk_measure == "cvar":
        risk_func = CVaR
    elif risk_measure == "wang":
        risk_func = Wang
    elif risk_measure == "cpw":
        risk_func = CPW

    policy: RiskC51Policy = RiskC51Policy(
        model=net,
        optim=optim,
        observation_space=env.observation_space,
        action_space=env.action_space,
        discount_factor=0.9,
        num_atoms=51,
        v_min=min_return, 
        v_max=max_return,
        estimation_step=3,
        target_update_freq=320,
        risk_function=risk_func,
        risk_param=beta
    )

    if load_policy:
        policy.load_state_dict(torch.load(risk_measure+'.pth')) # to load existing policy

    if train_policy:
        train_collector = Collector(policy, train_envs, VectorReplayBuffer(20000, num_train_envs), exploration_noise=True)
        test_collector = Collector(policy, test_envs, exploration_noise=True)

        #TensorBoard logging
        writer = SummaryWriter("log/"+risk_measure)
        logger = TensorboardLogger(writer)

        def train_fn(epoch: int, env_step: int) -> None:
            # eps annnealing
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
            max_epoch=50, #default: 10, preferred: 50 (maybe even 100 if slow convergence) and then it works pretty well
            step_per_epoch=1000, #default: 8000, preferred: 1000
            step_per_collect=10, #default: 8, preferred: 10
            # episode_per_collect=1,
            update_per_step=0.125, #default: 0.125
            episode_per_test=50, #default: 100, preferred: 50
            batch_size=batch_size,
            train_fn=train_fn,
            test_fn=lambda epoch, env_step: policy.set_eps(eps_test),
            logger=logger,
        )

        epoch_risk_comps = []
        for epoch_stat in trainer:
            if(epoch_stat.epoch < trainer.max_epoch - 2): # because in the last two epochs it becomes terrible, don't know why
                torch.save(policy.state_dict(), risk_measure+'.pth') # to save the policy in a file
                print(epoch_stat)

                # collect risk measure computation from epoch testing
                risk_comp = RiskCompute(torch.Tensor(epoch_stat.test_collect_stat.returns), risk_func, beta)
                # add: (epoch, risk, mean, std, min, max)
                epoch_risk_comps.append((epoch_stat.epoch, risk_comp, epoch_stat.test_collect_stat.returns_stat.mean, epoch_stat.test_collect_stat.returns_stat.std, epoch_stat.test_collect_stat.returns_stat.min, epoch_stat.test_collect_stat.returns_stat.max))

                # view test episodes (new ones, not those used by the trainer and collected in epoch_stats) in a popup window
                policy.eval()
                # policy.set_eps(eps_test)
                env = create_env("human")
                buf = VectorReplayBuffer(20000, num_test_envs)
                collector = ts.data.Collector(policy, env, buf, exploration_noise=True)
                collector.reset()
                collector.collect(n_episode=num_test_envs, render=1 / 200) # one episode per env
                env.close()
                # print(np.min(buf.act))
                # print(np.max(buf.act))
                policy.train()

        policy.load_state_dict(torch.load(risk_measure+'.pth')) # to load previous policy before getting worse
        epochs, risks, means, stds, mins, maxs = zip(*epoch_risk_comps)
        fig = plt.figure()
        plt.plot(epochs, risks, label="risk measure")
        plt.plot(epochs, means, label="mean return")
        plt.plot(epochs, stds, label="return standard deviation")
        plt.plot(epochs, mins, label="minimum return")
        plt.plot(epochs, maxs, label="maximum return")
        plt.legend()
        plt.savefig(fname="plots/"+risk_measure+"_training", dpi=fig.dpi)
        fig.clear()
    
    # evaluate trained/stored policy
    policy.eval()
    # policy.set_eps(eps_test)
    env = create_env("human")
    buf = VectorReplayBuffer(20000, num_eval_envs)
    collector = ts.data.Collector(policy, env, buf, exploration_noise=True)
    collector.reset()
    collect_stats = collector.collect(n_episode=num_eval_envs, render=1 / 200) # one episode per env
    env.close()
    risk_comp = RiskCompute(torch.tensor(collect_stats.returns), risk_func, beta)
    mean = collect_stats.returns_stat.mean
    std = collect_stats.returns_stat.std
    min = collect_stats.returns_stat.min
    max = collect_stats.returns_stat.max
    print("Risk measure:",risk_comp,"mean:", mean, "std:", std, "min:", min, "max:", max) # print results

    # plot probability distribution
    fig = plt.figure()
    # n = collect_stats.returns.size//10
    n = 10
    p, x = np.histogram(collect_stats.returns, bins=n, density=True) # bin it into n = N//10 bins
    x_d = np.diff(x)
    x = x[:-1] + (x[1] - x[0])/2   # convert bin edges to centers
    # f = UnivariateSpline(x, p, s=n)
    # plt.plot(x, f(x))
    plt.bar(x, p*x_d)
    x = np.linspace(mean - 3*std, mean + 3*std, 100)
    plt.plot(x, stats.norm.pdf(x, mean, std)) # normal distribution using the statistics, as an approximation
    plt.figtext(0.01,0.01,"Risk measure: "+str(round(risk_comp.item(),1))+" mean: "+str(round(mean,1))+" std: "+str(round(std,1))+" min: "+str(round(min,1))+" max: "+str(round(max,1)))
    plt.savefig(fname="plots/"+risk_measure+"_eval", dpi=fig.dpi)
    fig.clear()

if __name__ == "__main__":
    # plot distortion risk measure functions to ensure they are correct
    # input = torch.linspace(0.0, 1.0, 100)
    # plt.plot(input, Neutral(input))
    # plt.plot(input, CVaR(input, 0.25))
    # plt.plot(input, Wang(input, 0.75))
    # plt.plot(input, CPW(input, 0.71))
    # plt.show()

    run_experiment("neutral", load_policy=False, train_policy=True)
    run_experiment("cvar", 0.25, load_policy=False, train_policy=True)
    run_experiment("wang", 0.75, load_policy=False, train_policy=True)
    run_experiment("cpw", 0.71, load_policy=False, train_policy=True)