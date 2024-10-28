import logging

import numpy as np
import beepy

import utils
from src.simpleWT_gym.simple_wt_gym_4 import SimpleWtGym4
#♠from src.simpleWT_gym.wt_dynamics import WindTurbineSimulator

logging.basicConfig(level=logging.INFO)

#PI controller for pitch
def pitch_pi(error,Kp,Ki,e_int,dt):
    e_int = (e_int + error)*dt #Integral error
    #PID controller
    pitch_pid = - (Kp*error + Ki*e_int)
    #pitch = np.clip(pitch_pid, np.deg2rad(0), np.deg2rad(90))
    return pitch_pid, e_int

def main():
    wind = 12.3
    wg_nom = 39.5 #rad/s
    #PI parameters
    Kp = 1
    Ki = 100
    
    logging.info("Starting simpleWT-gym example")

    env = SimpleWtGym4(Vx=wind, wg_nom=wg_nom)
    obs = env.reset()
    
    error = 0
    e_int = 0 #PI global variable

    while env.wt_sim.ti < 50:
        pitch_ctrl, e_int = pitch_pi(error,Kp,Ki,e_int,env.wt_sim.dt)
        actions = [pitch_ctrl,wind]
        state = env.step(actions)
        obs = state[0]
        error = obs[0]
    
    utils.log_and_exit(env.myLog,395)

if __name__ == "__main__":
    main()