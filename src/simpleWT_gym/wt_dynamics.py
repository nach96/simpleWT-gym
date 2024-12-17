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
        solution = solve_ivp(self.wt.wind_turbine_ode, [self.ti,tf], self.x, method='RK45', args=(u,))
        self.x = [state_vec[-1] for state_vec in solution.y] #States
        logging.debug("Solver finished with success: {}".format(solution.success))
        if solution.success==False:
            logging.error("Solver failed. Stopping simulation.")
        self.log_callback()
        self.ti = tf
        return self.x
    
    def log_callback(self):
        if self.enable_myLog:
            #if self.ti % 0.1 < 0.01:                
            self.myLog.append({
                "time": self.ti,
                "Cp": self.wt.Cp,
                "Lambda_i": self.wt.Lambda_i,
                "Lambda": self.wt.Labmda,
                "Tem": self.wt.Tem,
                "Tm": self.wt.Tm,
                "Ia": self.wt.Ia,
                "Ea": self.wt.Ea,
                "w": self.wt.w,
                "pitch": self.wt.pitch,
                "dpitch": self.wt.dptich,
                "pitch_ref": self.wt.pitch_ref,
                "power": self.wt.power
            })

class WindTurbineDynamics():
    def __init__(self):
        logging.info("Wind turbine parameters initialized")
        self.c1 = 0.73              #Cp coefficients []
        self.c2 = 151
        self.c3 = 0.58
        #self.c3 = 10 #Lets hack this shit to make it more pitch dependent
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
        #self.w0 = 200*2*np.pi/60               #Initial rotor speed [rad/s]
        self.w0 = 40
        self.Ia0 = 30            #Initial armature current [A]
        #self.pitch0 = np.deg2rad(15.55)             #Initial pitch angle [rad]
        self.pitch0 = 0
        self.dpitchdt0 = 0          #Initial pitch angular speed [rad/s]
        self.x0 = [self.w0,self.Ia0,self.pitch0,self.dpitchdt0]

        #Logging signals
        self.Cp = 0
        self.Lambda_i = 0
        self.Labmda = 0
        self.Tem = 0
        self.Tm = 0
        self.Ia = self.Ia0
        self.Ea = 0
        self.w = self.w0
        self.pitch = self.pitch0
        self.dptich = 0
        self.pitch_ref = self.pitch0
        self.power = 0

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

        # Saturation tricks from Simulink to keep things in range
        w = np.clip(w, 2.5, 50)
        pitch = np.clip(pitch, np.deg2rad(0), np.deg2rad(90))
        wind_speed = np.clip(wind_speed, 1.8, 25)

        #Wind Turbine dynamics
        tip_speed_ratio = self.tip_speed_ratio(wind_speed, w)

        #[dpitchdt, d2pitchd2t] = self.pitch_actuator_ode([pitch,dpitchdt],[pitch_ref])
        [dpitchdt, d2pitchd2t] = self.pitch_actuator_ode_1st_order([pitch,dpitchdt],[pitch_ref])


        lambda_i = self.lambda_i(tip_speed_ratio, pitch)
        Cp = self.c_p(lambda_i,pitch)
        Tm = self.tm(Cp,wind_speed,w)
        #Rotor dynamics
        Tem = self.rotor.tem(Ia)
        Ea = self.rotor.Ea(w)
        dIadt = self.rotor.Ia_ode([Ia],[Ea])
        dwdt = self.rotor.w_ode([w],[Tm,Tem])
        power = self.rotor.power(Ia)

        #Log variables
        self.Cp = Cp
        self.Lambda_i = lambda_i
        self.Labmda = tip_speed_ratio
        self.Tem = Tem
        self.Tm = Tm
        self.Ia = Ia
        self.Ea = Ea
        self.w = w
        self.pitch = pitch
        self.dptich = dpitchdt
        self.pitch_ref = pitch_ref
        self.power = power

        #Return all ODEs derivatives
        dxdt = [dwdt,dIadt,dpitchdt,d2pitchd2t]
        return dxdt
    
    #Mechanical torque. Eq 1
    def tm(self,Cp,v,w):
        Tm = (Cp*self.rho*self.A*v**3)/(2*w)
        return Tm
    
    #lambda_i: Part of Cp. Eq 2
    def lambda_i(self, tip_speed_ratio, pitch):
        lambda_i = (1/(tip_speed_ratio + self.c8) - (self.c9/(pitch**3 + 1)))**(-1)
        return lambda_i
    
    #Wind turbine efficency coeficient. Eq 3
    def c_p(self, lambda_i,pitch):
        Cp = self.c1*(self.c2/lambda_i - self.c3*pitch - self.c4*(pitch**self.c5) - self.c6)*np.exp(-(self.c7/lambda_i))
        Cp = np.clip(Cp, 0.0, 1.0)
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

    def pitch_actuator_ode_1st_order(self,x,u):
        pitch_ref = u[0]
        pitch = x[0]
        d2pitchd2t = 0

        tao=0.2 #time constant [s]
        dpitchdt = 1/tao*(pitch_ref-pitch)

        max_dptich = np.radians(5) #5º/s
        dpitchdt = np.clip(dpitchdt, -max_dptich, max_dptich)

        dxdt =  [dpitchdt, d2pitchd2t]
        return dxdt

    
class RotorDynamics():
    def __init__(self):
        logging.info("Rotor dynamics initialized")
        self.J = 6.53           #Rotor inertia [kg*m^2]
        self.Kf = 0.025         #Friction constant [Nm*s/rad]
        #self.Kf = 1            #Increasing friction to make it easier to control
        self.Kg = 23.31 #*0.264 #generator constant []
        self.Kphi = 0.264       #Magnetic flow coupling [V*s/rad]
        self.La = 13.5*10**(-3) #Armature inductance [H]
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
        #Rl*Ia = V
        dIadt = (Ea - (self.Ra+self.Rl)*Ia)/self.La

        return dIadt
    
    def power(self,Ia):
        V = np.clip(self.Rl*Ia, -240, 240)
        power = V*Ia
        return power

    #Electrical back emf [V]. Eq 9
    def Ea(self,w):
        Ea = self.Kg*self.Kphi*w
        return Ea
