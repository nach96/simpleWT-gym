import logging

import numpy as np
import beepy

import utils
from src.simple_wt_gym import SimpleWtGym

logging.basicConfig(level=logging.INFO)

#PI controller for pitch increment actions
def pitch_increment_pi(error,Kp,Ki,e_int):
    e_int = e_int + error #Integral error
    #PID controller
    pitch_pid = Kp*error + Ki*e_int
    pitch_increment = np.clip(pitch_pid, -1, 1) #Normalized pitch increment
    return pitch_increment

def main():
    logging.info("Starting simpleWT-gym example")
    gym = SimpleWtGym(Vx=18, wg_nom=np.radians(7.55), t_max=40)

    gym.reset()
    terminated=False
    error = 0

    #PI parameters
    Kp = 0.2
    Ki = 0.0008
    #PI global variable
    e_int = 0

    while terminated==False:
        pitch_increment = pitch_increment_pi(error,Kp=Kp,Ki=Ki,e_int=e_int)
        observation, reward, terminated, extra = gym.step([pitch_increment])
        error = observation[0]
    
    utils.log_and_exit(gym.myLog,0)

if __name__ == "__main__":
    main()