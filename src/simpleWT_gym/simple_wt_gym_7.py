import logging

import gym
from gym import spaces
import numpy as np

from simpleWT_gym.wt_dynamics import WindTurbineSimulator

"""
Action: Pitch increment (normalized [-1,1]) (+Pitch_ref)
Observations: GenSpeed error, Pitch, Wind Speed x [12,13], Pitch_ref
Rewards: -speed_error^2
Wind modified in each episode
"""
class SimpleWtGym7(gym.Env):
    def __init__(self,inputFileName="", Vx=18, wg_nom=40, t_max=40, burn_in_time=0, control_time_step=0.2, Tem_ini=1.978655e7, Pitch_ini=15.55, pg_nom=1.5e7, logging_level=logging.INFO):
        #inputFileName pending. Hardcoded params in WindTurbineSimulator
        logging.debug("Initializing SimpeWTGym")

        self.control_time_step=control_time_step #s
        #Simulation parameters
        self.Vx_0 = Vx # Mean wind speed (addup  random +-0.15)
        self.Vx = Vx
        self.wg_nom = wg_nom
        self.t_max = t_max
        self.burn_in_time = burn_in_time

        #GYM API DEFINITION
        #Action: Pitch increment (normalized)
        low_action = np.array([-1], dtype=np.float32)
        high_action = np.array([1], dtype=np.float32)  
        #Observations: GenSpeed error, Pitch, Wind Speed x, Pitch_ref
        low_obs = np.array([-10,0,12,0], dtype=np.float32)
        high_obs = np.array([10,np.pi/2,13,np.pi/2], dtype=np.float32)
        self.set_spaces(low_action, high_action, low_obs, high_obs)

        #Logging
        self.enable_myLog = 1
        self.myLog = []
        self.pitch_increment = 0

    def step(self, action):
        logging.debug("Action: {}".format(action))
        self.actions = self.map_inputs(action)
        self.state = self.control_step(self.actions)
        self.obs = self.map_outputs(self.state)
        reward = self.reward(self.obs)
        done = self.do_terminate()
        self.log_callback()

        return self.obs, reward, done, {}
    
    def control_step(self, actions):
        steps = self.control_time_step/self.wt_sim.dt

        #Loop during control time step
        for i in range(int(steps)):
            state = self.wt_sim.step(actions)

        return state

    def reset(self):
        logging.debug("Resetting environment.")
        #Init Wind Turbine
        self.wt_sim = WindTurbineSimulator()
        self.state = self.wt_sim.wt.x0
        #Update wind for the step
        self.Vx = self.random_wind()
        self.obs = self.map_outputs(self.state)

        #After reset, run initial steps.
        self.obs = self.run_burn_in(self.obs)
        return self.obs
    
    def run_burn_in(self,obs):
        while (self.wt_sim.ti < self.burn_in_time):
            actions = [0.0,self.Vx]
            obs, *_ = self.step(actions)
        return obs

    def reward(self,obs):
        speed_error = obs[0]
        reward = -(speed_error**2 )
        return reward
    
    def do_terminate(self):
        terminate = False
        if (self.wt_sim.ti >= self.t_max):
            logging.info("Terminating episode. Time exceded")
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
        norm_delta_pitch = actions[0] #Norm 1 = 5 deg/s
        #Pitch incremental inputs
        minPitch = np.radians(0)
        maxPitch = np.radians(90)
        pitch_ref = self.wt_sim.wt.pitch_ref #[rad]

        self.pitch_increment = norm_delta_pitch*np.radians(5)*self.control_time_step # [rad] Norm 1 = 5 deg/s
        new_pitch = pitch_ref + self.pitch_increment   
        new_pitch = np.clip(new_pitch, minPitch, maxPitch) #Clamp between min and max pitch

        Vx = self.Vx

        return [new_pitch, Vx]
    
    def random_wind(self):
        #FRandom value between min and max
        self.Vx = self.Vx_0 + np.random.uniform(-0.15,0.15)
        
        return self.Vx
   
    def map_outputs(self, outputs):
        wg = outputs[0]
        error_wg = self.wg_nom-wg
        pitch = outputs[2]
        Vx = self.Vx
        pitch_ref = self.wt_sim.wt.pitch_ref
        
        gym_obs=[error_wg,pitch,Vx,pitch_ref]   
        return gym_obs
    
    def log_callback(self):
        if self.enable_myLog:
            self.myLog.append({
                "time": self.wt_sim.ti,
                "Pitch_increment": self.pitch_increment,
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
                "pitch_ref": self.wt_sim.wt.pitch_ref,
                "Vx": self.Vx,
                "actions.pitch": self.actions[0],
                "obs.error_wg": self.obs[0],
                "obs.pitch": self.obs[1],
                "obs.Vx": self.obs[2],
                "obs.pitch_ref": self.obs[3]
            })