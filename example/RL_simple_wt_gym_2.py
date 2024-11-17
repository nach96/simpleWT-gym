import logging

import numpy as np
import beepy

import utils
import sys
from src.simpleWT_gym.simple_wt_gym_2 import SimpleWtGym2

from stable_baselines3 import TD3
#from stable_baselines3.common.logger import configure

logging.basicConfig(level=logging.INFO)
def float_to_int(in_dict):
        for key, value in in_dict.items():
            if isinstance(value, (int, float)):  # Check if value is a numeric type
                if value % 1 == 0:
                    in_dict[key] = int(value)  # Convert to integer


# MODEL AND ENVIRONMENT PARAMETERS
model_params = {
    "learning_starts": 1e4,
    "learning_rate": 1e-3,
    "gamma": 0.98,
    "gradient_steps": 1,
    "train_freq": 100,
    "buffer_size": 1e4,
    "batch_size": 256,
    "verbose":1
    }
float_to_int(model_params)
#wg_nom = 150*(2*np.pi)/60 #rad/s
wg_nom = 40 #[rad/s]
net_size = 32
net_kwargs = dict(net_arch=[net_size, net_size])
wind = 12.3 #[m/s]

    
def main():
    env = SimpleWtGym2(Vx = wind, wg_nom = wg_nom, t_max=40)
    #obs = env.reset()

    model = TD3("MlpPolicy", env, **model_params, policy_kwargs=net_kwargs)
    model.learn(total_timesteps=4e4)

    utils.log_and_exit(env.myLog,"gym_RL")

if __name__ == "__main__":
    main()