#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 14:35:16 2023

@author: perrielh

This is a copy of ode_system with some modifications:
   equations for NG1 and NG2 are deleted
   negative feedback effect will be accounted for only by modification of associated parameters
   
   
Training data (all malE control) from dnr_training set:
    
time (hrs): 0, 2, 4, 8, 12, 24, 48

def3 setA: 5, 6.77, 5.13, 7.07, 5.51, 3.49, 4.15

"""

###############################################################################
# Baseline Defensin-3 calibrated SIMPLIFIED model #############################
###############################################################################
#
# Training data (all malE KD control) from def3_alldata:
#
# time (hrs): 0, 2, 4, 8, 12, 24, 48
#
# def3 setA: 5, 6.77, 5.13, 7.07, 5.51, 3.49, 4.15
#
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

### List of all the ODE system parameters:

init_vals = 5
tot_I = init_vals

d_A = 21.7   
d_G = 0.0001
d_P = 0.01  
d_bP = 0.05    
d_bI = 0.02   
d_R = 0.02   
d_N = 0.16  
d_bN = 0.3

s_A = 35.9    
s_P = 0.065    
s_B = 0.21      

c_A = 67.6   #adjusts how high A peaks
c_P = 0.5 
c_I = 0.5  
c_B = 1    
c_N = 0.845
c_bN = 1.46

f_P = 14  
p_G = 1.59
m_G = 1   

K_I = 1

def f(t, y):
    A = y[0]
    G = y[1]
    P = y[2]
    bP = y[3]
    bI = y[4]
    RB = y[5]
    RP = y[6]
    RN = y[7]
    bRN = y[8]
    
    dA_dt = s_A + c_A * bRN - d_A * A                       # AMPs transcribed by Relish
    dG_dt = p_G * G - m_G * G * A - d_G * G                 # pathogen present outside the cell
    
    dP_dt = s_P - 2 * c_P * P**2 * G  - d_P * P              # unbound PGRP-LC monomer
    dbP_dt = c_P * P**2 * G  - d_bP * bP                     # PGRP-LC dimer bound to pathogen
   
    dbI_dt = c_I * (tot_I-bI/K_I) * bP - d_bI * bI                  # IMD recruited by activated PGRP-LC
   
    dRB_dt = s_B - c_B * RB * bI - d_R * RB                 # Relish B in cytoplasm
    dRP_dt = c_B * RB * bI - f_P * RP - d_R * RP            # Relish P in cytoplasm
   
    dRN_dt = f_P * RP - c_N * RN + c_bN * bRN - d_N * RN    # free nuclear Relish
    dbRN_dt = c_N * RN - c_bN * bRN - d_bN * bRN           # nuclear Relish bound to target gene
    
    return np.array([dA_dt, dG_dt, dP_dt, dbP_dt, dbI_dt, dRB_dt, dRP_dt, dRN_dt, dbRN_dt])

t_span = np.array([0, 50])
times = np.linspace(t_span[0], t_span[1], 101)

# Initial conditions of ODE system
y0 = np.array([init_vals, init_vals, init_vals, 0, 0, init_vals, 0, 0, 0])

# Solving ODE initial value problem
soln = solve_ivp(f, t_span, y0, t_eval=times)
t = soln.t
A = soln.y[0]
G = soln.y[1]
P = soln.y[2]
bP = soln.y[3]
bI = soln.y[4]
RB = soln.y[5]
RP = soln.y[6]
RN = soln.y[7]
bRN = soln.y[8]


# Plotting the solution
plt.rc("font", size=12)
plt.figure()
plt.plot(t, A, '-', label='A')
plt.plot(t, G, '-', label='G')
plt.plot(t, P, '-', label='P')
plt.plot(t, bP, '-', label='bP')
plt.plot(t, bI, '-', label='bI')
plt.plot(t, RB, '-', label='RB')
plt.plot(t, RP, '-', label='RP')
plt.plot(t, RN, '-', label='RN')
plt.plot(t, bRN, '-', label='bRN')
plt.xlabel("time")
plt.ylabel("concentration")
plt.xlim(0,50)
plt.ylim(0,10)
plt.legend(loc='center right')
plt.show()

#%%


plt.rc("font", size=12)
plt.figure()
plt.plot(t, A, '-', label='Cactus RNAi', color='red')
plt.xlabel("Hours post infection")
plt.ylabel("AMP expression")
plt.xlim(0,48)
plt.ylim(0,5)
plt.legend(loc='upper right')
plt.show()


plt.rc("font", size=12)
plt.figure()
plt.plot(t, G, '-', label='Cactus RNAi', color='red')
plt.xlabel("Hours post infection")
plt.ylabel("Bacterial load")
plt.xlim(0,48)
plt.ylim(0,5)
plt.legend(loc='upper right')
plt.show()


#%%
###############################################################################
# Defensin-3 training data calibration ########################################
###############################################################################
#
# Training data (all malE KD control) from def3_alldata:
#
# time (hrs): 0, 2, 4, 8, 12, 24, 48
#
# def3 setA: 5, 6.77, 5.13, 7.07, 5.51, 3.49, 4.15
# 
###############################################################################


from symfit import variables, Parameter, Fit, D, ODEModel
import numpy as np
import matplotlib.pyplot as plt

tdata = np.array([0, 2, 4, 8, 12, 24, 48])
concentration = np.array([5, 6.77, 5.13, 7.07, 5.51, 3.49, 4.15])
deviation = np.array([0.869, 1.567, 1.466, 1.36, 1.39, 1.17, 1.02])

# Define our ODE model
t, A, G, P, bP, bI, RB, RP, RN, bRN = variables('t, A, G, P, bP, bI, RB, RP, RN, bRN')

#d_A = Parameter('d_A', 21.7, min=0)
#d_G = Parameter('d_G', 0.0001, min =0)
#d_P = Parameter('d_P', 0.01, min = 0)
#d_bP = Parameter('d_bP', 0.05, min=0)
d_bI = Parameter('d_bI', 0.02, min=0)
d_R = Parameter('d_R', 0.02, min=0)
#d_N = Parameter('d_N', 0.16, min=0)
d_bN = Parameter('d_bN', 0.3, min=0)

#s_A = Parameter('s_A', 35.9, min=0)
#s_P = Parameter('s_P', 0.065, min=0)
#s_B = Parameter('s_B', 0.21, min=0)

#c_A = Parameter('c_A', 67.6, min=0)
#c_P = Parameter('c_P', 0.5, min=0)
c_I = Parameter('c_I', 0.5, min=0)
c_B = Parameter('c_B', 1, min=0)
#c_N = Parameter('c_N', 0.845, min=0)
#c_bN = Parameter('c_bN', 1.46, min=0)

#f_P = Parameter('f_P', 14, min=0)
#p_G = Parameter('p_G', 1.59, min=0)
#m_G = Parameter('m_G', 1, min=0)


model = ODEModel({
    D(A, t): s_A + c_A * bRN - d_A * A,
    D(G, t): p_G * G - m_G * G * A - d_G * G,
    
    D(P, t): s_P - 2 * c_P * P**2 * G  - d_P * P,
    D(bP, t): c_P * P**2 * G  - d_bP * bP,
   
    D(bI, t): c_I * (tot_I-bI/K_I) * bP  - d_bI * bI,
   
    D(RB, t): s_B - c_B * RB * bI - d_R * RB,
    D(RP, t): c_B * RB * bI - f_P * RP - d_R * RP,
   
    D(RN, t): f_P * RP - c_N * RN + c_bN * bRN - d_N * RN,
    D(bRN, t): c_N * RN - c_bN * bRN,

    },
    initial={t: tdata[0], A: concentration[0], G: init_vals, P: init_vals, bP: 0, bI: 0, RB: init_vals, RP: 0, RN: 0, bRN: 0}
)

fit = Fit(model, t=tdata, A=concentration, sigma_A=deviation, G=None, P=None, bP=None, bI=None, RB=None, RP=None, RN=None,
          bRN=None)
fit_result = fit.execute()

print(fit_result)

taxis = np.linspace(0, 50)
A_fit, G_fit, P_fit, RB_fit,  RN_fit, RP_fit, bI_fit, bP_fit, bRN_fit,  = model(t=taxis, **fit_result.params)

plt.scatter(tdata, concentration)
plt.plot(taxis, A_fit, label='[A]')
#plt.plot(taxis, G_fit, label='[G]')
#plt.plot(taxis, P_fit, label='[P]')
#plt.plot(taxis, bP_fit, label='[bP]')
#plt.plot(taxis, bI_fit, label='[bI]')
#plt.plot(taxis, RB_fit, label='[RB]')
#plt.plot(taxis, RP_fit, label='[RP]')
#plt.plot(taxis, RN_fit, label='[RN]')
#plt.plot(taxis, bRN_fit, label='[bRN]')
#plt.xlabel('Hours post infection')
#plt.ylabel('[X]')
#plt.ylim(0, 3)
plt.xlim(0, 50)
plt.ylim(0,10)
plt.legend()
plt.show()

#%%
###############################################################################
# Defensin-3 data fitting #####################################################
###############################################################################
#
# DEF3 expression test data (cactus KD vs malE KD) from def3_alldata.xlsx
#
#def3 cac KD: 8.31, 7.90, 8.28, 7.27, 7.97, 6.59, 6.44, 5.18, 5.41, 4.40
#std dev: 1.17, 1.50, 1.28, 1.01, 1.49, 2.08, 1.89, 0.99, 0.91, 1.67
#count: 8, 7, 8, 8, 8, 8, 8, 8, 8, 8
#
# def3 mal CTRL: 4.91, 8.57, 6.18, 8.81, 5.55, 6.35, 4.09, 2.94, 2.48, 0.90
# std dev: 1.38, 1.92, 1.58, 1.30, 1.90, 2.11, 2.00, 1.14, 1.98, 1.15
# count: 8, 8, 8, 8, 8, 8, 7, 8, 8, 8
#
###############################################################################


tdata = np.array([0, 2, 4, 6, 8, 10, 12, 24, 36, 48])

kd_concentration = np.array([3.98, 3.63, 3.50, 3.07, 3.02, 3.10, 1.74, 1.14, 1.54, 1.07])
kd_std_dev = np.array([1.11, 1.23, 0.97, 0.73, 0.60, 0.79, 1.39, 0.59, 0.65, 0.63])

ctrl_concentration = np.array([4.53, 5.32, 4.51, 3.74, 3.72, 3.12, 2.58, 2.30, 1.97, 1.83])
ctrl_std_dev = np.array([1.21, 0.72, 0.88, 0.64, 1.36, 0.68, 0.79, 0.49, 0.83, 0.51])

# Define our ODE model
t, A, G, P, bP, bI, RB, RP, RN, bRN = variables('t, A, G, P, bP, bI, RB, RP, RN, bRN')

d_A = Parameter('d_A', 21.7, min=0)
#d_G = Parameter('d_G', 0.0001, min =0)
#d_P = Parameter('d_P', 0.01, min = 0)
#d_bP = Parameter('d_bP', 0.05, min=0)
#d_bI = Parameter('d_bI', 0.02, min=0)
#d_R = Parameter('d_R', 0.02, min=0)
#d_N = Parameter('d_N', 0.16, min=0)
#d_bN = Parameter('d_bN', 0.3, min=0)

s_A = Parameter('s_A', 35.9, min=0)
#s_P = Parameter('s_P', 0.065, min=0)
#s_B = Parameter('s_B', 0.21, min=0)

c_A = Parameter('c_A', 67.6, min=0)
#c_P = Parameter('c_P', 0.5, min=0)
#c_I = Parameter('c_I', 0.5, min=0)
#c_B = Parameter('c_B', 1, min=0)
c_N = Parameter('c_N', 0.845, min=0)
c_bN = Parameter('c_bN', 1.46, min=0)

f_P = Parameter('f_P', 14, min=0)
#p_G = Parameter('p_G', 1.59, min=0)
#m_G = Parameter('m_G', 1, min=0)


model_dict = {
    D(A, t): s_A + c_A * bRN - d_A * A,
    D(G, t): p_G * G - m_G * G * A - d_G * G,
    
    D(P, t): s_P - 2 * c_P * P**2 * G  - d_P * P,
    D(bP, t): c_P * P**2 * G  - d_bP * bP,
   
    D(bI, t): c_I * (tot_I-bI/K_I) * bP  - d_bI * bI,
   
    D(RB, t): s_B - c_B * RB * bI - d_R * RB,
    D(RP, t): c_B * RB * bI - f_P * RP - d_R * RP,
   
    D(RN, t): f_P * RP - c_N * RN + c_bN * bRN - d_N * RN,
    D(bRN, t): c_N * RN - c_bN * bRN,
    }

kd_model = ODEModel(model_dict, initial={t: tdata[0], A: kd_concentration[0], G: init_vals, P: init_vals, bP: 0, bI: 0, RB: init_vals, RP: 0, RN: 0, bRN: 0})
ctrl_model = ODEModel(model_dict, initial={t: tdata[0], A: ctrl_concentration[0], G: init_vals, P: init_vals, bP: 0, bI: 0, RB: init_vals, RP: 0, RN: 0, bRN: 0})

taxis = np.linspace(0, 50)

kd_fit = Fit(kd_model, t=tdata, A=kd_concentration, G=None, P=None, bP=None, bI=None, RB=None, RP=None, RN=None,
          bRN=None, sigma_A=kd_std_dev)
kd_fit_result = kd_fit.execute()

print("\nResults for Def3 knockdown data:")
print(kd_fit_result)

A_kd_fit, G_kd_fit, P_kd_fit, RB_kd_fit,  RN_kd_fit, RP_kd_fit, bI_kd_fit, bP_kd_fit, bRN_kd_fit,  = kd_model(t=taxis, **kd_fit_result.params)

ctrl_fit = Fit(ctrl_model, t=tdata, A=ctrl_concentration, G=None, P=None, bP=None, bI=None, RB=None, RP=None, RN=None,
          bRN=None, sigma_A=ctrl_std_dev)
ctrl_fit_result = ctrl_fit.execute()

print("\nResults for Def3 control data:")
print(ctrl_fit_result)

A_ctrl_fit, G_ctrl_fit, P_ctrl_fit, RB_ctrl_fit,  RN_ctrl_fit, RP_ctrl_fit, bI_ctrl_fit, bP_ctrl_fit, bRN_ctrl_fit,  = ctrl_model(t=taxis, **ctrl_fit_result.params)

plt.scatter(tdata, kd_concentration, label = 'KD data', color='tomato')
plt.plot(taxis, A_kd_fit, label='KD fit', color='tomato')

plt.scatter(tdata, ctrl_concentration, label = 'Ctrl data', color='dodgerblue')
plt.plot(taxis, A_ctrl_fit, label='Ctrl fit', color='dodgerblue')

#plt.plot([0, magnitude_init], [time_max, magnitude_max], color = 'green', linestyle = '-')
#plt.annotate("Initial value = " + str(magnitude_init), (0, A_fit[0]))
#plt.annotate("Max value = " + str(magnitude_max), (time_max, magnitude_max))

plt.xlabel('Hours post infection')
plt.ylabel('Defensin-3 expression')
plt.xticks([0,6,12,18,24,30,36,42,48])
plt.xlim(-2, 50)
plt.yticks([0,2,4,6,8,10])
plt.ylim(-1,11)
plt.legend()
plt.show()

#%%

def3_kd_c_A = kd_fit_result.value(c_A)
def3_ctrl_c_A = ctrl_fit_result.value(c_A)
def3_kd_c_N = kd_fit_result.value(c_N)
def3_ctrl_c_N = ctrl_fit_result.value(c_N)
def3_kd_c_bN = kd_fit_result.value(c_bN)
def3_ctrl_c_bN = ctrl_fit_result.value(c_bN)
def3_kd_d_A = kd_fit_result.value(d_A)
def3_ctrl_d_A = ctrl_fit_result.value(d_A)
def3_kd_d_N = kd_fit_result.value(d_N)
def3_ctrl_d_N = ctrl_fit_result.value(d_N)

#%%



























