import logging

import numpy as np
import beepy

import utils
from src.simpleWT_gym.simple_wt_gym_1 import SimpleWtGym1

logging.basicConfig(level=logging.INFO)


def pitch_stair_increment(ts,obs):
    if ts<40:
        pitch_ctrl = 0
    elif ts<80:
        if obs[1]<np.pi/4:
            pitch_ctrl = 1
        else:
            pitch_ctrl = 0
    elif ts<120:
        if obs[1]<np.pi/3:
            pitch_ctrl = 1
        else:
            pitch_ctrl = 0
    elif ts<160:
        if obs[1]<np.pi/2:
            pitch_ctrl = 1
        else:
            pitch_ctrl = 0
    else:
        pitch_ctrl = 0
    return pitch_ctrl  


def main():
    #wg_nom = 150*(2*np.pi)/60 #rad/s
    wind = 12.3
    tmax = 500

    env = SimpleWtGym1(t_max=tmax)
    obs = env.reset()

    while env.wt_sim.ti < tmax:
        #pitch_ctrl = np.deg2rad(0)
        #pitch_ctrl = pitch_stair_increment(env.wt_sim.ti,obs)
        pitch_ctrl = 1
        actions = [pitch_ctrl,wind]
        obs, reward, done, info = env.step(actions)
    
    utils.log_and_exit(env.myLog,"Gym_1")

if __name__ == "__main__":
    main()