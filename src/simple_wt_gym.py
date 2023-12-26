import logging

import gym
from gym import spaces
import numpy as np

from wt_dynamics import WindTurbineSimulator

class SimpleWtGym(gym.Env):
    def __init__(self, Vx=18, wg_nom=np.radians(7.55), t_max=40):
        #Simulation parameters
        self.Vx = Vx
        self.wg_nom = wg_nom
        self.t_max = t_max

        #GYM API DEFINITION
        #Action: Pitch increment (normalized)
        low_action = np.array([-1], dtype=np.float32)
        high_action = np.array([1], dtype=np.float32)  
        #Observations: GenSpeed error, Pitch, Wind Speed x
        low_obs = np.array([-10,0,0], dtype=np.float32)
        high_obs = np.array([10,90,40], dtype=np.float32)
        self.set_spaces(low_action, high_action, low_obs, high_obs)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        actions = self.map_inputs(action)
        self.state = self.wt_sim.step(actions)
        obs = self.map_outputs(self.state)
        reward = self.reward(obs)
        done = self.do_terminate()
        return np.array(self.state), reward, done, {}

    def reset(self):
        #Init Wind Turbine
        self.wt_sim = WindTurbineSimulator()
        self.state = self.wt_sim.wt.x0
        return np.array(self.state)

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
        norm_delta_pitch = actions[0]
        #Pitch incremental inputs
        minPitch = np.radians(5)
        maxPitch = np.radians(45)

        pitch_deg = norm_delta_pitch*2*self.dt # Max 2 deg/s
        new_pitch = self.pitch + np.radians(pitch_deg)    
        new_pitch = np.clip(new_pitch, minPitch, maxPitch) #Clamp between min and max pitch

        Vx = self.Vx

        return [new_pitch, Vx]
   
    def map_outputs(self, outputs):
        wg = outputs[0]
        error_wg = self.wg_nom-wg
        pitch = outputs[2]
        Vx = self.Vx
        gym_obs=[error_wg,pitch,Vx]   
        return gym_obs