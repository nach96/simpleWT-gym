import logging

import numpy as np
import beepy

import utils
from src.simpleWT_gym.simple_wt_gym_4 import SimpleWtGym4

logging.basicConfig(level=logging.INFO)

def pitch_stair(ts):
    if ts<40:
        pitch_ctrl = 0
    elif ts<80:
        pitch_ctrl = np.pi/4
    elif ts<120:
        pitch_ctrl = np.pi/3
    elif ts<160:
        pitch_ctrl = np.pi/2
    else:
        pitch_ctrl = 0
    return pitch_ctrl      


def main():
    #wg_nom = 150*(2*np.pi)/60 #rad/s
    wind = 12.3

    env = SimpleWtGym4()
    obs = env.reset()

    while env.wt_sim.ti < 200:
        #pitch_ctrl = np.deg2rad(0)
        pitch_ctrl = pitch_stair(env.wt_sim.ti)
        actions = [pitch_ctrl,wind]
        state = env.step(actions)
        pitch_act=state[1]
    
    utils.log_and_exit(env.myLog,33)

if __name__ == "__main__":
    main()