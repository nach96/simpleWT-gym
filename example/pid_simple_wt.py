import logging

import numpy as np
import beepy

import utils
#from src.simple_wt_gym_4 import SimpleWtGym4
from src.simpleWT_gym.wt_dynamics import WindTurbineSimulator

logging.basicConfig(level=logging.INFO)

#PI controller for pitch
def pitch_pi(error,Kp,Ki,e_int):
    e_int = e_int + error #Integral error
    #PID controller
    pitch_pid = - (Kp*error + Ki*e_int)
    pitch = np.clip(pitch_pid, np.deg2rad(0), np.deg2rad(90))
    #pitch=45
    return pitch

def main():
    wg_nom = 150*(2*np.pi)/60 #rad/s
    #PI parameters
    Kp = 0.1
    Ki = 0.001
    
    logging.info("Starting simpleWT-gym example")
    #gym = SimpleWtGym4(Vx=18, wg_nom=wg_nom, t_max=40)
    #gym.reset()
    gym = WindTurbineSimulator()
    
    error = 0
    e_int = 0 #PI global variable

    while gym.ti < 40:
        pitch_ctrl = pitch_pi(error,Kp=Kp,Ki=Ki,e_int=e_int)
        actions = [pitch_ctrl,18]
        state = gym.step(actions)
        wg = gym.x[0]
        error = wg_nom - wg
    
    utils.log_and_exit(gym.myLog,5)

if __name__ == "__main__":
    main()