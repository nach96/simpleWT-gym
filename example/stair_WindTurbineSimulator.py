import logging

import numpy as np
import beepy

import utils

from src.simpleWT_gym.wt_dynamics import WindTurbineSimulator

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

def small_stair(ts):
    if ts<40:
        pitch_ctrl = 0
    elif ts<45:
        pitch_ctrl = np.pi/4/6
    elif ts<50:
        pitch_ctrl = np.pi/4/6*2
    elif ts<55:
        pitch_ctrl = np.pi/4/6*3
    elif ts<60:
        pitch_ctrl = np.pi/4/6*4
    elif ts<65:
        pitch_ctrl = np.pi/4/6*5
    elif ts<70:
        pitch_ctrl = np.pi/4/6*6
    elif ts<120:
        pitch_ctrl = np.pi/3
    elif ts<160:
        pitch_ctrl = np.pi/2
    else:
        pitch_ctrl = 0
    return pitch_ctrl  

def step_40deg(ts):
    if ts<40:
        pitch_ctrl = 0
    elif ts<80:
        pitch_ctrl = np.radians(40)
    elif ts<120:
        pitch_ctrl = np.pi/3
    elif ts<160:
        pitch_ctrl = np.pi/2
    else:
        pitch_ctrl = 0
    return pitch_ctrl

def step_2deg(ts):
    if ts<40:
        pitch_ctrl = 0
    elif ts<80:
        pitch_ctrl = np.radians(2)
    elif ts<120:
        pitch_ctrl = np.pi/3
    elif ts<160:
        pitch_ctrl = np.pi/2
    else:
        pitch_ctrl = 0
    return pitch_ctrl          
def deg2_mini_stair(ts):
    if ts<40:
        pitch_ctrl = 0
    elif ts<45:
        pitch_ctrl = np.radians(2)/6
    elif ts<50:
        pitch_ctrl = np.radians(2)/6*2
    elif ts<55:
        pitch_ctrl = np.radians(2)/6*3
    elif ts<60:
        pitch_ctrl = np.radians(2)/6*4
    elif ts<65:
        pitch_ctrl = np.radians(2)/6*5
    elif ts<70:
        pitch_ctrl = np.radians(2)/6*6
    elif ts<80:
        pitch_ctrl = np.radians(2)
    else:
        pitch_ctrl = 0
    return pitch_ctrl 

def main():
    #wg_nom = 150*(2*np.pi)/60 #rad/s
    wind = 12.3

    env = WindTurbineSimulator()


    while env.ti < 80:
        #pitch_ctrl = np.deg2rad(0)
        #pitch_ctrl = pitch_stair(env.ti)
        pitch_ctrl = step_40deg(env.ti)
        actions = [pitch_ctrl,wind]
        state = env.step(actions)
        wg = env.x[0]
    
    utils.log_and_exit(env.myLog,"WT_Dynamics_deg2_mini_stair")

if __name__ == "__main__":
    main()