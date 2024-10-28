import logging

import gym
from gym import spaces
import numpy as np

from .wt_dynamics import WindTurbineSimulator

"""
Action: Pitch
Observations: GenSpeed error, Pitch, Wind Speed x
"""
class SimpleWtGym4(gym.Env):
    def __init__(self, Vx=18, wg_nom=0.79, t_max=40, logging_level=logging.INFO):
        #Simulation parameters
        self.Vx = Vx
        self.wg_nom = wg_nom
        self.t_max = t_max

        #GYM API DEFINITION
        #Action: Pitch
        low_action = np.array([np.deg2rad(0)], dtype=np.float32)
        high_action = np.array([np.deg2rad(90)], dtype=np.float32)  
        #Observations: GenSpeed error, Pitch, Wind Speed x
        low_obs = np.array([-10,0,0], dtype=np.float32)
        high_obs = np.array([10,np.deg2rad(90),40], dtype=np.float32)
        self.set_spaces(low_action, high_action, low_obs, high_obs)
        

        #Logging
        self.enable_myLog = 1
        self.myLog = []
        self.pitch_ctrl = 0

    def step(self, action):
        actions = self.map_inputs(action)
        self.state = self.wt_sim.step(actions)
        obs = self.map_outputs(self.state)
        reward = self.reward(obs)
        done = self.do_terminate()
        self.log_callback()
        return obs, reward, done, {}

    def reset(self):
        #Init Wind Turbine
        self.wt_sim = WindTurbineSimulator()
        self.state = self.wt_sim.wt.x0
        obs = self.map_outputs(self.state)
        return obs

    def reward(self,obs):
        speed_error = obs[0]
        reward = -speed_error**2 
        return reward
    
    def do_terminate(self):
        terminate = False
        if (self.wt_sim.ti >= self.t_max):
            terminate = True          
        return terminate
    
    def set_spaces(self, low_action, high_action, low_obs, high_obs):
        self.action_space = spaces.Box(
            low=low_action,
            high=high_action,
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=low_obs,
            high=high_obs,
            dtype=np.float32
        )
   
    def map_inputs(self,actions):
        new_pitch = actions[0]
        #Pitch incremental inputs
        minPitch = np.radians(5)
        maxPitch = np.radians(45)
        #new_pitch = np.clip(new_pitch, minPitch, maxPitch) #Clamp between min and max pitch
        self.pitch_ctrl = new_pitch

        Vx=self.Vx=actions[1]

        return [new_pitch, Vx]
   
    def map_outputs(self, outputs):
        wg = outputs[0]
        error_wg = self.wg_nom-wg
        pitch = outputs[2]
        Vx = self.Vx
        gym_obs=[error_wg,pitch,Vx]   
        return gym_obs
    
    def log_callback(self):
        if self.enable_myLog:
            #if self.wt_sim.ti % 0.1 < 0.01:                
            self.myLog.append({
                "time": self.wt_sim.ti,
                "Pitch_ctrl": self.pitch_ctrl,
                "Cp": self.wt_sim.wt.Cp,
                "Lambda_i": self.wt_sim.wt.Lambda_i,
                "Lambda": self.wt_sim.wt.Labmda,
                "Tem": self.wt_sim.wt.Tem,
                "Tm": self.wt_sim.wt.Tm,
                "Ia": self.wt_sim.wt.Ia,
                "Ea": self.wt_sim.wt.Ea,
                "w": self.wt_sim.wt.w,
                "pitch": self.wt_sim.wt.pitch,
                "dpitch": self.wt_sim.wt.dptich,
                "pitch_ref": self.wt_sim.wt.pitch_ref
            })