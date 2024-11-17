import logging

import numpy as np
import beepy

import utils
from src.simpleWT_gym.simple_wt_gym_2 import SimpleWtGym2

logging.basicConfig(level=logging.INFO)


def pitch_stair_increment(ts,pitch_ref):
    if ts<40:
        pitch_ctrl = 0
    elif ts<80:
        if pitch_ref<np.pi/4:
            pitch_ctrl = 1
        else:
            pitch_ctrl = 0
    elif ts<120:
        if pitch_ref<np.pi/3:
            pitch_ctrl = 1
        else:
            pitch_ctrl = 0
    elif ts<160:
        if pitch_ref<np.pi/2:
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

    env = SimpleWtGym2(t_max=tmax,burn_in_time=1)
    obs = env.reset()


    while env.wt_sim.ti < tmax:
        #pitch_ctrl = np.deg2rad(0)
        pitch_ref = env.wt_sim.wt.pitch_ref
        pitch_ctrl = pitch_stair_increment(env.wt_sim.ti,pitch_ref)
        actions = [pitch_ctrl,wind]
        obs, reward, done, info = env.step(actions)
    
    utils.log_and_exit(env.myLog,"Gym_2")

if __name__ == "__main__":
    main()