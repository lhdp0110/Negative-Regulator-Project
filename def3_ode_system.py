#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:32:22 2022

@author: perrielh


Training data (all malE control) from dnr_training set:
    
time (hrs): 0, 2, 4, 8, 12, 24, 48

def3 setA: 3.00, 4.77, 3.13, 5.07, 3.51, 1.49, 2.15
def3 setB: 2.10, 3.87, 4.22, 5.40, 3.55, 1.60, 2.17

per All_HK_Bt_GE_data.xlsx:
cactus (adjusted): 4.53, 5.32, 4.51, 3.74, 3.72, 3.12, 2.58, 2.30, 1.97, 1.83

"""
###############################################################################
# Baseline Defensin-3 calibrated COMPLEX model ################################
###############################################################################
#
# Training data (all malE KD control) from def3_alldata:
#
# time (hrs): 0, 2, 4, 8, 12, 24, 48
#
# def3 setA: 5, 6.77, 5.13, 7.07, 5.51, 3.49, 4.15
# 
# cactus: 4.53, 5.32, 4.51, 3.72, 2.58, 2.30, 1.83
#
###############################################################################

import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

### List of all the ODE system parameters:
    
init_vals = 5
tot_I = init_vals

# degradation/inactivation rates
d_A = 0.18     # 0.02 (delta_A) per Ellner Fig.3,4 /// 0.05 per Table S3 // # can only fit if 0.5
d_G = 0.01     # 0.02 (delta_P) per Ellner Fig.4
d_P = 0.01      # 0.01 (delta_P) per Ellner Table S3
d_bP = 0.05      # 0.02 (delta_P*) per Ellner Table S3
d_bI = 0.05     # 0.01 (rho_I*) per Ellner Table S3
d_R = 0.01      # 0.01 (delta_R) per Ellner Table S3
d_N = 0.05     # 0.01 (delta_N) per Ellner Table S3
d_bN = 0.05
d_NG1 = 0.15    # 0.05 (delta_H) per Ellner Table S3
d_NG2 = 0.01    # 0.05 (delta_H) per Ellner Table S3
d_NG3 = 0.01

# supply rates
s_A = 1.2      # (Q_A = delta_A) per Ellner p.15
s_P = 0.05      # (Q_P = delta_P) per Ellner p.15
s_B = 0.2      # (Q_R = delta_R) per Ellner p.15
s_NG1 = 1   # (Q_H = delta_H) per Ellner p.15
s_NG2 = 0.05    # (Q_H = delta_H) per Ellner p.15
s_NG3 = 0.05

# setting all supply and degradation rates to the same value 
# eliminates irrelevant variations and allows values to return to baseline

# reaction/binding rates
c_A = 0.15    # (default = 0.8) this parameter adjusts how high A peaks
c_P = 1.5    # (new default = 0.5)
c_I = 0.5     # (default = 0.5)
c_B = 0.5      # (default = 4) can be higher than 4 but not lower than 2
c_N = 3     # (default = 3) decreasing to 1 lowers peak of A and induces small lift around t=48, increasing beyond 3 doesn't change much
c_bN = 1     # (default = 0.5) variations alter shape and peak of A, but not by much
c_NG1 = 0.1
c_NG2 = 0.05
c_NG3 = 0.05

# carrying capacity
K_I = 1

# flow rate into nucleus
f_P =  10     # (default = 10) increasing to 20 or higher doesn't change much, decreasing to 1 flattens A and incudes small lift at t=48

# strength of feedback effect
phi_NG1 = 1
phi_NG2 = 0.3
phi_NG3 = 0.3

# proliferation rate
p_G = 0.4       # (new default = 2.8) 2.5, per Frank. alters G more than A

# mortality rate
m_G = 0.05           # (default = 2) max=5, per Frank. alters G more than A

# Likely targets for param estimation:
# c_N, c_bN, f_P, c_A  
# meaning: binding and unbinding of Relish to target sequence,
# import of Relish into nucleus, and transcription of AMPs by Relish
# maybe also c_B (conversion of Relish B into P)

# targets also depend on nature of neg reg
# ie, where is it expected to act (1 vs 2 vs 3, Lazzaro & Tate 2022 p.3)
# 1: prevent signal transduction by immune receptor (c_P, c_I)
# 2: inhibit or degrade protein or transcription factors that propagate signal (d_P, d_bP, d_bI, d_R, d_RP, d_RN, d_bRN)
# 2b: inhibit activity (c_I ....)
# 3: modify activity of resulting effector (m_G, s_A, c_A, d_A)

### System of 9 ODEs describing rate of change of concentrations
### of IMD signaling pathway components listed:

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
    NG1 = y[9]
    NG2 = y[10]
    NG3 = y[11]
    
    dA_dt = s_A + c_A * bRN - d_A * A                       # AMPs transcribed by Relish
    dG_dt = p_G * G - m_G * G * A - d_G * G                 # pathogen present outside the cell
    
    dP_dt = s_P - 2 * c_P * P**2 * G * exp(-phi_NG1 * NG1) - d_P * P              # unbound PGRP-LC monomer
    dbP_dt = c_P * P**2 * G * exp(-phi_NG1 * NG1) - d_bP * bP                     # PGRP-LC dimer bound to pathogen
   
    dbI_dt = c_I * (tot_I-bI/K_I) * bP * exp(-phi_NG2 * NG2) - d_bI * bI                  # IMD recruited by activated PGRP-LC
   
    dRB_dt = s_B - c_B * RB * bI - d_R * RB                 # Relish B in cytoplasm
    dRP_dt = c_B * RB * bI - f_P * RP * exp(-phi_NG3 * NG3)- d_R * RP            # Relish P in cytoplasm
   
    dRN_dt = f_P * RP - c_N * RN + c_bN * bRN * exp(-phi_NG3 * NG3) - d_N * RN    # free nuclear Relish
    dbRN_dt = c_N * RN - c_bN * bRN - d_bN * bRN           # nuclear Relish bound to target gene
   
    dNG1_dt = s_NG1 + c_NG1 * bRN - d_NG1 * NG1             # neg reg at level 1 (PGRP-SC2/LB, pathogen binding)
    dNG2_dt = s_NG2 + c_NG2 * bRN - d_NG2 * NG2              # neg reg at level 2 (PIRK, IMD recruitment)
    dNG3_dt = s_NG3 + c_NG3 * bRN - d_NG3 * NG3
    
    return np.array([dA_dt, dG_dt, dP_dt, dbP_dt, dbI_dt, dRB_dt, dRP_dt, dRN_dt, dbRN_dt, dNG1_dt, dNG2_dt, dNG3_dt])

t_span = np.array([0, 50])
times = np.linspace(t_span[0], t_span[1], 101)

# Initial conditions of ODE system
y0 = np.array([init_vals, init_vals, init_vals, 0, 0, init_vals, 0, 0, 0, init_vals, init_vals, init_vals])

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
NG1 = soln.y[9]
NG2 = soln.y[10]
NG3 = soln.y[11]


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
plt.plot(t, NG1, '-', label='NG1')
#plt.plot(t, NG2, '-', label='NG2')
plt.plot(t, NG3, '-', label='NG3')
plt.xlabel("time")
plt.ylabel("concentration")
plt.xlim(0,50)
plt.ylim(0,15)
plt.legend(loc='center right')
plt.show()

#%%

plt.rc("font", size=12)
plt.figure()
plt.plot(t, A, '-', label='Cactus RNAi', color='red')
plt.xlabel("Hours post infection")
plt.ylabel("AMP expression")
plt.xlim(0,48)
plt.ylim(0,3)
plt.legend(loc='upper right')
plt.show()


plt.rc("font", size=12)
plt.figure()
plt.plot(t, G, '-', label='Cactus RNAi', color='red')
plt.xlabel("Hours post infection")
plt.ylabel("Bacterial load")
plt.xlim(0,48)
plt.ylim(0,3)
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
# cactus: 4.53, 5.32, 4.51, 3.72, 2.58, 2.30, 1.83
#
###############################################################################


from symfit import variables, Parameter, Fit, D, ODEModel, exp
import numpy as np
import matplotlib.pyplot as plt

tdata = np.array([0, 2, 4, 8, 12, 24, 48])
concentration = np.array([5, 6.77, 5.13, 7.07, 5.51, 3.49, 4.15])
deviation = np.array([0.869, 1.567, 1.466, 1.36, 1.39, 1.17, 1.02])

# Define our ODE model
t, A, G, P, bP, bI, RB, RP, RN, bRN, NG1, NG2, NG3 = variables('t, A, G, P, bP, bI, RB, RP, RN, bRN, NG1, NG2, NG3')

d_A = Parameter('d_A', 0.2, min=0)
d_G = Parameter('d_G', 0.02, min =0)
d_P = Parameter('d_P', 0.02, min = 0)
d_bP = Parameter('d_bP', 0.02, min=0)
#d_bI = Parameter('d_bI', 0.02, min=0)
#d_R = Parameter('d_R', 0.02, min=0)
#d_N = Parameter('d_N', 0.02, min=0)
#d_bN = Parameter('d_bN', 0.02, min=0)
#d_NG1 = Parameter('d_NG1', 0.02, min=0)
#d_NG2 = Parameter('d_NG2', 0.02, min=0)
#d_NG3 = Parameter('d_NG3', 0.02, min=0)

s_A = Parameter('s_A', 0.5, min=0)
s_P = Parameter('s_P', 0.05, min=0)
#s_B = Parameter('s_B', 0.05, min=0)
#s_NG1 = Parameter('s_NG1', 0.05, min=0)
#s_NG2 = Parameter('s_NG2', 0.05, min=0)
#s_NG3 = Parameter('s_NG3', 0.05, min=0)

c_A = Parameter('c_A', 0.8, min=0)
c_P = Parameter('c_P', 0.5, min=0)
#c_I = Parameter('c_I', 0.5, min=0)
#c_B = Parameter('c_B', 0.5, min=0)
#c_N = Parameter('c_N', 0.5, min=0)
#c_bN = Parameter('c_bN', 0.5, min=0)
#c_NG1 = Parameter('c_NG1', 0.5, min=0)
#c_NG2 = Parameter('c_NG2', 0.5, min=0)
#c_NG3 = Parameter('c_NG3', 0.5, min=0)

#f_P = Parameter('f_P', 10, min=0)
#phi_NG1 = Parameter('phi_NG1', 0.05, min=0)
#phi_NG2 = Parameter('phi_NG2', 0.32, min=0)
#phi_NG3 = Parameter('phi_NG3', 0.32, min=0)

#p_G = Parameter('p_G', 8, min=0)
#m_G = Parameter('m_G', 2.2, min=0)

model = ODEModel({
    D(A, t): s_A + c_A * bRN - d_A * A,
    D(G, t): p_G * G - m_G * G * A - d_G * G,
    
    D(P, t): s_P - 2 * c_P * P**2 * G * exp(-phi_NG1 * NG1) - d_P * P,
    D(bP, t): c_P * P**2 * G * exp(-phi_NG1 * NG1) - d_bP * bP,
   
    D(bI, t): c_I * (tot_I-bI/K_I) * bP * exp(-phi_NG2 * NG2) - d_bI * bI,
   
    D(RB, t): s_B - c_B * RB * bI - d_R * RB,
    D(RP, t):  c_B * RB * bI - f_P * RP * exp(-phi_NG3 * NG3)- d_R * RP,
   
    D(RN, t): f_P * RP - c_N * RN + c_bN * bRN * exp(-phi_NG3 * NG3) - d_N * RN,
    D(bRN, t): c_N * RN - c_bN * bRN,
   
    D(NG1, t): s_NG1 + c_NG1 * bRN - d_NG1 * NG1,
    D(NG2, t): s_NG2 + c_NG2 * bRN - d_NG2 * NG2
    },
    initial={t: tdata[0], A: concentration[0], G: init_vals, P: init_vals, bP: 0, bI: 0, RB: init_vals, RP: 0, RN: 0, bRN: 0, NG1: init_vals, NG2: init_vals, NG3: init_vals}
)

fit = Fit(model, t=tdata, A=concentration, sigma_A=deviation, G=None, P=None, bP=None, bI=None, RB=None, RP=None, RN=None,
          bRN=None, NG1=None, NG2=None, NG3=None)
fit_result = fit.execute()

print(fit_result)

taxis = np.linspace(0, 50)
A_fit, G_fit, NG1_fit, NG2_fit, NG3_fit, P_fit, RB_fit,  RN_fit, RP_fit, bI_fit, bP_fit, bRN_fit,  = model(t=taxis, **fit_result.params)

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
#plt.plot(taxis, NG1_fit, label='[NG1]')
#plt.plot(taxis, NG2_fit, label='[NG2]')
#plt.plot(taxis, NG3_fit, label='[NG3]')
plt.xlabel('Hours post infection')
plt.ylabel('[X]')
#plt.ylim(0, 3)
plt.xlim(0, 50)
plt.legend()
plt.show()

#%%
###############################################################################
# Defensin-3 test data fitting ################################################
###############################################################################
#
# DEF3 expression test data (cactus KD vs malE KD) from def3_alldata.xlsx
#
# def3 KD: 8.31, 7.90, 8.28, 7.27, 7.97, 6.59, 6.44, 5.18, 5.41, 4.40
# std dev: 1.17, 1.50, 1.28, 1.01, 1.49, 2.08, 1.89, 0.99, 0.91, 1.67
# count: 8, 7, 8, 8, 8, 8, 8, 8, 8, 8
#
# def3 CTRL: 4.91, 8.57, 6.18, 8.81, 5.55, 6.35, 4.09, 2.94, 2.48, 0.90
# std dev: 1.38, 1.92, 1.58, 1.30, 1.90, 2.11, 2.00, 1.14, 1.98, 1.15
# count: 8, 8, 8, 8, 8, 8, 7, 8, 8, 8
#
###############################################################################
#
# CACTUS expression test data (cactus KD vs malE KD) from test_data.xlsx
#
# cactus KD: 3.98, 3.63, 3.50, 3.07, 3.02, 3.10, 1.74, 1.14, 1.54, 1.07
# std dev: 1.11, 1.23, 0.97, 0.73, 0.60, 0.79, 1.39, 0.59, 0.65, 0.63
# count: 8, 7, 8, 8, 8, 8, 8, 8, 8, 8
#
# cactus CTRL: 4.53, 5.32, 4.51, 3.74, 3.72, 3.12, 2.58, 2.30, 1.97, 1.83
# std dev: 1.21, 0.72, 0.88, 0.64, 1.36, 0.68, 0.79, 0.49, 0.83, 0.51
# count: 8, 8, 8, 8, 8, 8, 7, 8, 8, 8
#
###############################################################################

tdata = np.array([0, 2, 4, 6, 8, 10, 12, 24, 36, 48])

kd_def3_conc = np.array([8.31, 7.90, 8.28, 7.27, 7.97, 6.59, 6.44, 5.18, 5.41, 4.40])
kd_def3_dev = np.array([1.17, 1.50, 1.28, 1.01, 1.49, 2.08, 1.89, 0.99, 0.91, 1.67])

kd_cac_conc = np.array([3.98, 3.63, 3.50, 3.07, 3.02, 3.10, 1.74, 1.14, 1.54, 1.07])
kd_cac_dev = np.array([1.11, 1.23, 0.97, 0.73, 0.60, 0.79, 1.39, 0.59, 0.65, 0.63])

ctrl_def3_conc = np.array([4.91, 8.57, 6.18, 8.81, 5.55, 6.35, 4.09, 2.94, 2.48, 0.90])
ctrl_def3_dev = np.array([1.38, 1.92, 1.58, 1.30, 1.90, 2.11, 2.00, 1.14, 1.98, 1.15])

ctrl_cac_conc = np.array([4.53, 5.32, 4.51, 3.74, 3.72, 3.12, 2.58, 2.30, 1.97, 1.83])
ctrl_cac_dev = np.array([1.21, 0.72, 0.88, 0.64, 1.36, 0.68, 0.79, 0.49, 0.83, 0.51])


# Define our ODE model
t, A, G, P, bP, bI, RB, RP, RN, bRN, NG1, NG2, NG3 = variables('t, A, G, P, bP, bI, RB, RP, RN, bRN, NG1, NG2, NG3')

d_A = Parameter('d_A', 0.5, min=0)
d_N = Parameter('d_N', 0.15, min=0)
c_A = Parameter('c_A', 0.8, min=0)
c_N = Parameter('c_N', 3, min=0)
c_bN = Parameter('c_bN', 1, min=0)

model_dict = {
    D(A, t): s_A + c_A * bRN - d_A * A,
    D(G, t): p_G * G - m_G * G * A - d_G * G,
    
    D(P, t): s_P - 2 * c_P * P**2 * G * exp(-phi_NG1 * NG1) - d_P * P,
    D(bP, t): c_P * P**2 * G * exp(-phi_NG1 * NG1) - d_bP * bP,
   
    D(bI, t): c_I * (tot_I-bI/K_I) * bP * exp(-phi_NG2 * NG2) - d_bI * bI,
   
    D(RB, t): s_B - c_B * RB * bI - d_R * RB,
    D(RP, t): c_B * RB * bI - f_P * RP * exp(-phi_NG3 * NG3)- d_R * RP,
   
    D(RN, t): f_P * RP - c_N * RN + c_bN * bRN * exp(-phi_NG3 * NG3) - d_N * RN,
    D(bRN, t): c_N * RN - c_bN * bRN,
   
    D(NG1, t): s_NG1 + c_NG1 * bRN - d_NG1 * NG1,
    D(NG2, t): s_NG2 + c_NG2 * bRN - d_NG2 * NG2,
    D(NG2, t): s_NG3 + c_NG3 * bRN - d_NG3 * NG3
    }

kd_model = ODEModel(model_dict, initial={t: tdata[0], A: kd_def3_conc[0], G: init_vals, P: init_vals, bP: 0, bI: 0, RB: init_vals, RP: 0, RN: 0, bRN: 0, NG1: init_vals, NG2: init_vals})
ctrl_model = ODEModel(model_dict, initial={t: tdata[0], A: ctrl_def3_conc[0], G: init_vals, P: init_vals, bP: 0, bI: 0, RB: init_vals, RP: 0, RN: 0, bRN: 0, NG1: init_vals, NG2: init_vals})

taxis = np.linspace(0, 50)

kd_fit = Fit(kd_model, t=tdata, A=kd_def3_conc, G=None, P=None, bP=None, bI=None, RB=None, RP=None, RN=None,
          bRN=None, NG1=None, NG2=None, sigma_A=kd_def3_dev)
kd_fit_result = kd_fit.execute()

print("\nResults for Def3 knockdown data:")
print(kd_fit_result)

A_kd_fit, G_kd_fit, NG1_kd_fit, NG2_kd_fit, P_kd_fit, RB_kd_fit,  RN_kd_fit, RP_kd_fit, bI_kd_fit, bP_kd_fit, bRN_kd_fit,  = kd_model(t=taxis, **kd_fit_result.params)

ctrl_fit = Fit(ctrl_model, t=tdata, A=ctrl_def3_conc, G=None, P=None, bP=None, bI=None, RB=None, RP=None, RN=None,
          bRN=None, NG1=None, NG2=None, sigma_A=ctrl_def3_dev)
ctrl_fit_result = ctrl_fit.execute()

print("\nResults for Def3 control data:")
print(ctrl_fit_result)

A_ctrl_fit, G_ctrl_fit, NG1_ctrl_fit, NG2_ctrl_fit, P_ctrl_fit, RB_ctrl_fit,  RN_ctrl_fit, RP_ctrl_fit, bI_ctrl_fit, bP_ctrl_fit, bRN_ctrl_fit,  = ctrl_model(t=taxis, **ctrl_fit_result.params)

plt.scatter(tdata, kd_def3_conc, label = 'KD data', color='tomato')
plt.plot(taxis, A_kd_fit, label='KD fit', color='tomato')

plt.scatter(tdata, ctrl_def3_conc, label = 'Ctrl data', color='dodgerblue')
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