import gymnasium as gym
from matplotlib import pyplot as plt

from highway_env.envs.highway_env import HighwayEnv

if __name__ == "__main__":
    env: HighwayEnv = gym.make("highway-v0", render_mode="human")
    config = {
        # "manual_control": True,
        # check the observation configs to align with the one the algorithms use (Atari -> RGB/Grayscale?, and check stack config if it's applied before/after input)
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
            "scaling": 1.75,
        },
        "policy_frequency": 2,  # [Hz] # frequency with which actions are chosen (the agent acts)
        "action": {
            "type": "DiscreteAction", # thankfully that works, discretizes 3x3 = 9 combinations of steering and speed controls, use that since it fits with the algorithms and q value structure
            "longitudinal": True,
            "lateral": True,
            "actions_per_axis": 5 # change that for discretization accuracy
        },
        "lanes_count": 4, # change that for randomization, less chaos
        "vehicles_count": 50, # change that for more randomization and chaos
        "duration": 40,  # [s] # change the episode length in seconds, maybe bigger length for more risk-aversity
        "initial_spacing": 2,
        "collision_reward": -1,  # The reward received when colliding with a vehicle. # change it to value risk
        "reward_speed_range": [20, 30],  # [m/s] The reward for high speed is mapped linearly from this range to [0, HighwayEnv.HIGH_SPEED_REWARD].
        "simulation_frequency": 15,  # [Hz]
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "screen_width": 600,  # [px]
        "screen_height": 150,  # [px]
        "centering_position": [0.3, 0.5],
        "scaling": 5.5,
        "show_trajectories": False,
        "render_agent": True,
        "offscreen_rendering": False,
        "offroad_terminal": True # VERY IMPORTANT TO NOT GET OUT OF TRACK (LITERALLY, PUN INTENTED)
    }
    env.configure(config)
    obs, info = env.reset()
    print(env.action_space)
    done = False
    while not done:
        obs, reward, done, truncated, info = env.step(env.action_space.sample())

        fig, axes = plt.subplots(ncols=4, figsize=(12, 5))
        for i, ax in enumerate(axes.flat):
            ax.imshow(obs[i, ...].T, cmap=plt.get_cmap('gray'))
        plt.show()

