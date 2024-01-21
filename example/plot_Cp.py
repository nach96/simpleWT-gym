import utils
from src.simpleWT_gym.wt_dynamics import WindTurbineDynamics

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


wt = WindTurbineDynamics()

def rpm2rad(rpm):
    return rpm*(2*np.pi)/60

Vx=12
w_array = np.linspace(2,500,100)
pitch_array = np.linspace(0,90,100)

Cp_array = np.zeros((len(w_array),len(pitch_array)))
Cp_array2 = np.zeros(len(w_array))
Cp_array3 = np.zeros(len(pitch_array))
tip_speed_ratio_array = np.zeros(len(w_array))
lambda_i_array = np.zeros((len(w_array),len(pitch_array)))
lambda_i_array2 = np.zeros(len(w_array))
lambda_i_array3 = np.zeros(len(pitch_array))

for idx_w, w in enumerate(w_array):
    w = rpm2rad(w)
    tip_speed_ratio = wt.tip_speed_ratio(Vx, w)
    tip_speed_ratio_array[idx_w] = tip_speed_ratio
    lambda_i_array2[idx_w] = wt.lambda_i(tip_speed_ratio, np.deg2rad(15))
    Cp_array2[idx_w] = wt.c_p(lambda_i_array2[idx_w],np.deg2rad(15))
    for idx_pitch, pitch in enumerate(pitch_array):  
        pitch = np.deg2rad(pitch)
        lambda_i = wt.lambda_i(tip_speed_ratio, pitch)
        Cp = wt.c_p(lambda_i,pitch)
        Cp_array[idx_w][idx_pitch] = Cp
        lambda_i_array[idx_w][idx_pitch] = lambda_i

for id, pitch in enumerate(pitch_array):
    w = rpm2rad(300)
    tip_speed_ratio = wt.tip_speed_ratio(Vx, w)
    lambda_i = wt.lambda_i(tip_speed_ratio, np.deg2rad(pitch))
    lambda_i_array3[id] = lambda_i
    Cp_array3[id] = wt.c_p(lambda_i,np.deg2rad(pitch))




fig = plt.figure()
ax_cp1 = fig.add_subplot(331, projection='3d')
ax_cp2 = fig.add_subplot(332)
ax_cp3 = fig.add_subplot(333)
ax_l1 = fig.add_subplot(334, projection='3d')
ax_l2 = fig.add_subplot(335)
ax_l3 = fig.add_subplot(336)
ax_tsr = fig.add_subplot(337)

########### Cp #############
y, x = np.meshgrid(tip_speed_ratio_array, pitch_array)
z = Cp_array
ax_cp1.plot_surface(x, y, z, cmap=cm.coolwarm)
ax_cp1.set_xlabel('TSR')
ax_cp1.set_ylabel('pitch [deg]')
ax_cp1.set_zlabel('Cp')

#x = w_array
x = tip_speed_ratio_array
y = Cp_array2
ax_cp2.plot(x, y)
ax_cp2.set_xlabel('TSR')
ax_cp2.set_ylabel('Cp')
ax_cp2.set_title('pitch=15deg', loc='right')
ax_cp2.grid()

x = pitch_array
y = Cp_array3
ax_cp3.plot(x, y)
ax_cp3.set_xlabel('pitch [deg]')
ax_cp3.set_ylabel('Cp')
ax_cp3.set_title('w=300rpm',loc='right')
ax_cp3.grid()

########### lambda_i #############
y, x = np.meshgrid(w_array, pitch_array)
z = lambda_i_array
ax_l1.plot_surface(x, y, z, cmap=cm.coolwarm)
ax_l1.set_xlabel('w [rpm]')
ax_l1.set_ylabel('pitch [deg]')
ax_l1.set_zlabel('lambda_i')

x = w_array
y = lambda_i_array2
ax_l2.plot(x, y)
ax_l2.set_xlabel('w [rpm]')
ax_l2.set_ylabel('lambda_i')
ax_l2.set_title('pitch=15deg', loc='right')
ax_l2.grid()

x = pitch_array
y = lambda_i_array3
ax_l3.plot(x, y)
ax_l3.set_xlabel('pitch [deg]')
ax_l3.set_ylabel('lambda_i')
ax_l3.set_title('w=300rpm',loc='right')
ax_l3.grid()

##### tip_speed_ratio ########
x = w_array
y = tip_speed_ratio_array
ax_tsr.plot(x, y)
ax_tsr.set_xlabel('w [rpm]]')
ax_tsr.set_ylabel('tip_speed_ratio')
ax_tsr.grid()




plt.show()


        
