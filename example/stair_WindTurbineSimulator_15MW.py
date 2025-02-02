import logging

import numpy as np
import beepy

import utils

from src.simpleWT_gym.wt_dynamics_15MW import WindTurbineSimulator_15MW

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
    wind = 18
    env = WindTurbineSimulator_15MW()


    while env.ti < 200:
        pitch_ctrl = pitch_stair(env.ti)

        actions = [pitch_ctrl,wind]
        state = env.step(actions)
    
    utils.log_and_exit(env.myLog,"WT_Dynamics_15MW_0")

if __name__ == "__main__":
    main()