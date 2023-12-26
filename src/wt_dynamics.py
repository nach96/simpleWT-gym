import logging

import numpy as np
import scipy.signal as signal
from scipy.integrate import odeint, solve_ivp

class WindTurbineSimulator():
    def __init__(self):
        logging.info("Wind turbine simulator initialized")
        self.dt = 0.01           #Simulation time step [s]
        self.wt = WindTurbineDynamics()
        self.x = self.wt.x0     #Initial state
        self.ti=0.0             #Initial time

        # Logging
        self.enable_myLog = 1
        self.myLog = []


    def step(self,u):
        tf = self.ti + self.dt
        self.x = solve_ivp(self.wt.wind_turbine_ode, [self.ti,tf], self.x, method='RK45', args=(u,))
        self.wt.wind_turbine_ode()
        self.actions = u
        self.ti = tf
        return self.x
    
    def log_callback(self):
        #log actions and observations
        if self.enable_myLog:
            if self.ti % 0.1 < 0.01:
                self.myLog.append({"time":self.ti,"pitch":self.x[2],"wg_error":self.x[0],"Pitch increment":self.actions[0],"Vx":self.actions[1]})

class WindTurbineDynamics():
    def __init__(self):
        logging.info("Wind turbine parameters initialized")
        self.c1 = 0.73              #Cp coefficients []
        self.c2 = 151
        self.c3 = 0.58
        self.c4 = 0.002
        self.c5 = 2.14
        self.c6 = 13.2
        self.c7 = 18.4
        self.c8 = -0.02
        self.c9 = -0.003
        self.rho = 1.223            #Air density [kg/m^3]
        self.R = 3.2*0.98           #Rotor radius [m]
        self.A =  np.pi*self.R**2   #Rotor area [m^2]
        self.rotor = RotorDynamics()

        logging.info("Wind turbine initial state")
        self.w0 = 1.8               #Initial rotor speed [rad/s]
        self.Ia0 = 0.001            #Initial armature current [A]
        self.pitch0 = 0             #Initial pitch angle [rad]
        self.dpitchdt0 = 0          #Initial pitch angular speed [rad/s]
        self.x0 = [self.w0,self.Ia0,self.pitch0,self.dpitchdt0]


    # Wind turbine dynamics. System with all equations
    def wind_turbine_ode(self,t,x,u):
        #State variables
        w = x[0]
        Ia = x[1]
        pitch = x[2]
        dpitchdt = x[3]
        #inputs
        pitch_ref = u[0]
        wind_speed = u[1]


        #Wind Turbine dynamics
        tip_speed_ratio = self.tip_speed_ratio(wind_speed, w)
        [dpitchdt, d2pitchd2t] = self.pitch_actuator_ode([pitch,dpitchdt],[pitch_ref])
        lambda_i = self.lambda_i(tip_speed_ratio, pitch)
        c_p = self.c_p(lambda_i,pitch)
        Tm = self.tm(c_p,wind_speed,self.rotor.w)

        #Rotor dynamics
        Tem = self.rotor.tem(Ia)
        Ea = self.rotor.Ea(w)
        dIadt = self.rotor.Ia_ode([Ia],[Ea])
        dwdt = self.rotor.w_ode([w],[Tm,Tem])

        #Return all ODEs derivatives
        dxdt = [dwdt,dIadt,dpitchdt,d2pitchd2t]
        return dxdt
    
    #Mechanical torque. Eq 1
    def tm(self,Cp,v,w):
        Tm = (Cp*self.rho*self.A*v**3)/(2*w)
        return Tm
    
    #lambda_i: Part of Cp. Eq 2
    def lambda_i(self, tip_speed_ratio, pitch):
        lambda_i = (1/(tip_speed_ratio + self.c8) - (self.c9/(pitch^3 + 1)))
        return lambda_i
    
    #Wind turbine efficency coeficient. Eq 3
    def c_p(self, lambda_i,pitch):
        Cp = self.c1*(self.c2/lambda_i - self.c3*pitch - self.c4*pitch^self.c5 - self.c6)*np.exp(-self.c7/lambda_i)
        return Cp
       
    #Tip speed ratio. Eq 4
    def tip_speed_ratio(self,v,w):
        tip_speed_ratio = w*self.R/v
        return tip_speed_ratio
    
    #pitch TF manually converted to ODE. Eq 5
    def pitch_actuator_ode(self,x,u):
        pitch = x[0]
        dpitchdt = x[1]
        pitch_ref = u[0]
        d2pitchd2t = (-dpitchdt - 0.15*(pitch-pitch_ref))/2

        dxdt =  [dpitchdt, d2pitchd2t]
        return dxdt
    
    ########## DEPRECEATED ##########
    # Pitch Transfer Function. Think how to integrate with the system.
    def pitch_actuator_TF(self):
        num = [0.15]
        den = [0.15 , 1, 0.15]
        pitch_tf = signal.TransferFunction(num, den)
        return pitch_tf

class RotorDynamics():
    def __init__(self):
        logging.info("Rotor dynamics initialized")
        self.J = 2.25*2.9       #Rotor inertia [kg*m^2]
        self.Kf = 0.025         #Friction constant [Nm*s/rad]
        self.Kg = 23.31*0.264   #generator constant []
        self.Kphi = self.Kg     #Magnetic flow coupling [V*s/rad]
        self.La = 13.5*10^(-3)  #Armature inductance [H]
        self.Vnom = 240         #Nominal voltage [V]
        self.Ra = 0.275         #Armature resistance [Ohm]
        self.Rl = 8             #Load resistance [Ohm]
        
    #Rotor speed ode. Eq 6
    def w_ode(self,x,u):
        w = x[0]
        Tm = u[0]
        Tem = u[1]
        dwdt = (Tm - Tem - self.Kf*w)/self.J
        return dwdt
    
    #Electrical torque [Nm]. Eq 7
    def tem(self,Ia):
        Tem = self.Kg*self.Kphi*Ia
        return Tem
    
    #Electrical current ode. Eq 8
    def Ia_ode(self,x,u):
        Ia = x[0]
        Ea = u[0]
        dIadt = (Ea - self.Vnom - self.Ra*Ia)/self.La
        return dIadt

    #Electrical back emf [V]. Eq 9
    def Ea(self,w):
        Ea = self.Kg*self.Kphi*w
        return Ea
