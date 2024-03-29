#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 15:00:51 2023

@author: perrielh
"""

#%%
###############################################################################
# Base ODE system #############################################################
###############################################################################

import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

### List of all the ODE system parameters:

# degradation/inactivation rates
d_A = 0.5      # 0.02 (delta_A) per Ellner Fig.3,4 /// 0.05 per Table S3 // # can only fit if 0.5
d_G = 0.02       # 0.02 (delta_P) per Ellner Fig.4
d_P = 0.01       # 0.01 (delta_P) per Ellner Table S3
d_bP = 0.02      # 0.02 (delta_P*) per Ellner Table S3
d_bI = 0.01      # 0.01 (rho_I*) per Ellner Table S3
d_R = 0.01       # 0.01 (delta_R) per Ellner Table S3
d_N = 0.1       # 0.01 (delta_N) per Ellner Table S3
d_bN = 0.1
d_NG1 = 0.05    # 0.05 (delta_H) per Ellner Table S3
d_NG2 = 0.05     # 0.05 (delta_H) per Ellner Table S3

# supply rates
s_A = d_A       # (Q_A = delta_A) per Ellner p.15
s_P = d_P       # (Q_P = delta_P) per Ellner p.15
s_B = d_R       # (Q_R = delta_R) per Ellner p.15
s_NG1 = d_NG1     # (Q_H = delta_H) per Ellner p.15
s_NG2 = d_NG1    # (Q_H = delta_H) per Ellner p.15

# reaction/binding rates
c_A = 0.8     # (default = 0.8) this parameter adjusts how high A peaks
c_P = 0.5       # (new default = 0.5)
c_I = 1       # (default = 0.5)
c_B = 4        # (default = 4) can be higher than 4 but not lower than 2
c_N = 3     # (default = 3) decreasing to 1 lowers peak of A and induces small lift around t=48, increasing beyond 3 doesn't change much
c_bN = 2.5      # (default = 0.5) variations alter shape and peak of A, but not by much
c_NG1 = 0.1
c_NG2 = 0.2

# carrying capacity
K_I = 1

# flow rate into nucleus
f_P =  9       # (default = 10) increasing to 20 or higher doesn't change much, decreasing to 1 flattens A and incudes small lift at t=48

# strength of feedback effect
phi_NG1 = 0.07
phi_NG2 = 0.32

# proliferation rate
p_G = 2.4       # (new default = 2.8) 2.5, per Frank. alters G more than A

# mortality rate
m_G = 2            # (default = 2) max=5, per Frank. alters G more than A

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
    
    dA_dt = s_A + c_A * bRN - d_A * A                       # AMPs transcribed by Relish
    dG_dt = p_G * G - m_G * G * A - d_G * G                 # pathogen present outside the cell
    
    dP_dt = s_P - 2 * c_P * P**2 * G * exp(-phi_NG1 * NG1) - d_P * P              # unbound PGRP-LC monomer
    dbP_dt = c_P * P**2 * G * exp(-phi_NG1 * NG1) - d_bP * bP                     # PGRP-LC dimer bound to pathogen
   
    dbI_dt = c_I * (1-bI/K_I) * bP * exp(-phi_NG2 * NG2) - d_bI * bI                  # IMD recruited by activated PGRP-LC
   
    dRB_dt = s_B - c_B * RB * bI - d_R * RB                 # Relish B in cytoplasm
    dRP_dt = c_B * RB * bI - f_P * RP - d_R * RP            # Relish P in cytoplasm
   
    dRN_dt = f_P * RP - c_N * RN + c_bN * bRN - d_N * RN    # free nuclear Relish
    dbRN_dt = c_N * RN - c_bN * bRN - d_bN * bRN           # nuclear Relish bound to target gene
   
    dNG1_dt = s_NG1 + c_NG1 * bRN - d_NG1 * NG1             # neg reg at level 1 (PGRP-SC2/LB, pathogen binding)
    dNG2_dt = s_NG2 + c_NG2 * bRN - d_NG2 *NG2              # neg reg at level 2 (PIRK, IMD recruitment)
    
    return np.array([dA_dt, dG_dt, dP_dt, dbP_dt, dbI_dt, dRB_dt, dRP_dt, dRN_dt, dbRN_dt, dNG1_dt, dNG2_dt])

t_span = np.array([0, 50])
times = np.linspace(t_span[0], t_span[1], 100)

# Initial conditions of ODE system
y0 = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1])

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


# Plotting the solution
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
plt.xlabel('Hours post infection')
plt.ylabel('Expression of component X')
plt.xticks([0,6,12,18,24,30,36,42,48])
plt.xlim(0, 50)
plt.yticks([0,1,2,3])
plt.ylim(0, 3)
plt.legend()
plt.show()


#%%
###############################################################################
# Graphical sensitivity analysis of d_A #######################################
###############################################################################
test_values = [0.25, 0.35, 0.5]
A = []
A = [0 for i in range(len(test_values))]
label = []
label = ['' for i in range(len(test_values))]
i = 0
for d_A in test_values: 
    # Solving ODE initial value problem for each parameter value
    soln = solve_ivp(f, t_span, y0, t_eval=times)
    t = soln.t
    A[i] = soln.y[0]
    label[i] = str(test_values[i])
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
    plt.plot(t, A[i], '-', label='d_A = '+label[i])
    rate_induc = (soln.y[0][12] - soln.y[0][0])/6
    rate_decay = (soln.y[0][48] - soln.y[0][12])/6
    exp_min = min(soln.y[0])
    exp_max = max(soln.y[0])
    print('For d_A = ' +label[i] +':')
    print('  Induction rate = '+str(rate_induc))
    print('  Decay rate     = '+str(rate_decay))
    print('  Minimum value  = ' + str(exp_min))
    print('  Maximum value  = ' + str(exp_max))
    print('')
    i += 1
d_A = 0.5                 # reset d_A to original value    

# Plotting the solution for all values on same graph
plt.xlabel('Hours post infection')
plt.ylabel('AMP expression')
plt.xticks([0,6,12,18,24,30,36,42,48])
plt.xlim(0, 50)
plt.yticks([0,1,2,3])
plt.ylim(0, 3)
plt.legend()
plt.show()


#%%
###############################################################################
# Graphical sensitivity analysis of d_G #######################################
###############################################################################

test_values = [0.01, 10, 100]
A = []
A = [0 for i in range(len(test_values))]
label = []
label = ['' for i in range(len(test_values))]
i = 0
for d_G in test_values: 
    # Solving ODE initial value problem for each parameter value
    soln = solve_ivp(f, t_span, y0, t_eval=times)
    t = soln.t
    A[i] = soln.y[0]
    label[i] = str(test_values[i])
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
    plt.plot(t, A[i], '-', label='d_G = '+label[i])
    rate_induc = (soln.y[0][12] - soln.y[0][0])/6
    rate_decay = (soln.y[0][48] - soln.y[0][12])/6
    exp_min = min(soln.y[0])
    exp_max = max(soln.y[0])
    print('For d_G = ' + label[i] +':')
    print('  Induction rate = ' + str(rate_induc))
    print('  Decay rate     = ' + str(rate_decay))
    print('  Minimum value  = ' + str(exp_min))
    print('  Maximum value  = ' + str(exp_max))
    print('')
    i += 1
d_G = 0.02                           # reset d_A to original value    

# Plotting the solution for all values on same graph
plt.xlabel('Hours post infection')
plt.ylabel('AMP expression')
plt.xticks([0,6,12,18,24,30,36,42,48])
plt.xlim(0, 50)
plt.yticks([0,1,2,3])
plt.ylim(0, 3)
plt.legend()
plt.show()


#%%
###############################################################################
# Graphical sensitivity analysis of d_P #######################################
###############################################################################

test_values = [0.01, 10, 100]
A = []
A = [0 for i in range(len(test_values))]
label = []
label = ['' for i in range(len(test_values))]
i = 0
for d_P in test_values: 
    # Solving ODE initial value problem for each parameter value
    soln = solve_ivp(f, t_span, y0, t_eval=times)
    t = soln.t
    A[i] = soln.y[0]
    label[i] = str(test_values[i])
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
    plt.plot(t, A[i], '-', label='d_P = '+label[i])
    rate_induc = (soln.y[0][12] - soln.y[0][0])/6
    rate_decay = (soln.y[0][48] - soln.y[0][12])/6
    exp_min = min(soln.y[0])
    exp_max = max(soln.y[0])
    print('For d_P = ' + label[i] +':')
    print('  Induction rate = ' + str(rate_induc))
    print('  Decay rate     = ' + str(rate_decay))
    print('  Minimum value  = ' + str(exp_min))
    print('  Maximum value  = ' + str(exp_max))
    print('')
    i += 1   
d_P = 0.01                           # reset d_P to original value    

# Plotting the solution for all values on same graph
plt.xlabel('Hours post infection')
plt.ylabel('AMP expression')
plt.xticks([0,6,12,18,24,30,36,42,48])
plt.xlim(0, 50)
plt.yticks([0,1,2,3])
plt.ylim(0, 3)
plt.legend()
plt.show()


#%% 
###############################################################################
# Graphical sensitivity analysis of d_bP ######################################
###############################################################################

test_values = [0.01, 10, 100]
A = []
A = [0 for i in range(len(test_values))]
label = []
label = ['' for i in range(len(test_values))]
i = 0
for d_bP in test_values: 
    # Solving ODE initial value problem for each parameter value
    soln = solve_ivp(f, t_span, y0, t_eval=times)
    t = soln.t
    A[i] = soln.y[0]
    label[i] = str(test_values[i])
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
    plt.plot(t, A[i], '-', label='d_bP = '+label[i])
    rate_induc = (soln.y[0][12] - soln.y[0][0])/6
    rate_decay = (soln.y[0][48] - soln.y[0][12])/6
    exp_min = min(soln.y[0])
    exp_max = max(soln.y[0])
    print('For d_bP = ' + label[i] +':')
    print('  Induction rate = ' + str(rate_induc))
    print('  Decay rate     = ' + str(rate_decay))
    print('  Minimum value  = ' + str(exp_min))
    print('  Maximum value  = ' + str(exp_max))
    print('')
    i += 1
d_bP = 0.02          # reset d_bP to original value    

# Plotting the solution for all values on same graph
plt.xlabel('Hours post infection')
plt.ylabel('AMP expression')
plt.xticks([0,6,12,18,24,30,36,42,48])
plt.xlim(0, 50)
plt.yticks([0,1,2,3])
plt.ylim(0, 3)
plt.legend()
plt.show()


#%%
###############################################################################
# Graphical sensitivity analysis of d_bI ######################################
###############################################################################

test_values = [0.01, 10, 100]
A = []
A = [0 for i in range(len(test_values))]
label = []
label = ['' for i in range(len(test_values))]
i = 0
for d_bI in test_values: 
    # Solving ODE initial value problem for each parameter value
    soln = solve_ivp(f, t_span, y0, t_eval=times)
    t = soln.t
    A[i] = soln.y[0]
    label[i] = str(test_values[i])
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
    plt.plot(t, A[i], '-', label='d_bI = '+label[i])
    rate_induc = (soln.y[0][12] - soln.y[0][0])/6
    rate_decay = (soln.y[0][48] - soln.y[0][12])/6
    exp_min = min(soln.y[0])
    exp_max = max(soln.y[0])
    print('For d_bI = ' + label[i] +':')
    print('  Induction rate = ' + str(rate_induc))
    print('  Decay rate     = ' + str(rate_decay))
    print('  Minimum value  = ' + str(exp_min))
    print('  Maximum value  = ' + str(exp_max))
    print('')
    i += 1
d_bI = 0.01                    # reset d_bI to original value    

# Plotting the solution for all values on same graph
plt.xlabel('Hours post infection')
plt.ylabel('AMP expression')
plt.xticks([0,6,12,18,24,30,36,42,48])
plt.xlim(0, 50)
plt.yticks([0,1,2,3])
plt.ylim(0, 3)
plt.legend()
plt.show()


#%%
###############################################################################
# Graphical sensitivity analysis of d_R #######################################
###############################################################################

test_values = [0.001, 0.2, 1]
A = []
A = [0 for i in range(len(test_values))]
label = []
label = ['' for i in range(len(test_values))]
i = 0
for d_R in test_values: 
    # Solving ODE initial value problem for each parameter value
    soln = solve_ivp(f, t_span, y0, t_eval=times)
    t = soln.t
    A[i] = soln.y[0]
    label[i] = str(test_values[i])
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
    plt.plot(t, A[i], '-', label='d_R = '+label[i])
    rate_induc = (soln.y[0][12] - soln.y[0][0])/6
    rate_decay = (soln.y[0][48] - soln.y[0][12])/6
    exp_min = min(soln.y[0])
    exp_max = max(soln.y[0])
    print('For d_R = ' + label[i] +':')
    print('  Induction rate = ' + str(rate_induc))
    print('  Decay rate     = ' + str(rate_decay))
    print('  Minimum value  = ' + str(exp_min))
    print('  Maximum value  = ' + str(exp_max))
    print('')
    i += 1
d_R = 0.01

# Plotting the solution for all values on same graph
plt.xlabel('Hours post infection')
plt.ylabel('AMP expression')
plt.xticks([0,6,12,18,24,30,36,42,48])
plt.xlim(0, 50)
plt.yticks([0,1,2,3])
plt.ylim(0, 3)
plt.legend()
plt.show()


#%%
###############################################################################
# Graphical sensitivity analysis of d_N #######################################
###############################################################################

test_values = [0.01, 0.1, 1]
A = []
A = [0 for i in range(len(test_values))]
label = []
label = ['' for i in range(len(test_values))]
i = 0
for d_N in test_values: 
    # Solving ODE initial value problem for each parameter value
    soln = solve_ivp(f, t_span, y0, t_eval=times)
    t = soln.t
    A[i] = soln.y[0]
    label[i] = str(test_values[i])
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
    plt.plot(t, A[i], '-', label='d_N = '+label[i])
    rate_induc = (soln.y[0][12] - soln.y[0][0])/6
    rate_decay = (soln.y[0][48] - soln.y[0][12])/6
    exp_min = min(soln.y[0])
    exp_max = max(soln.y[0])
    print('For d_N = ' + label[i] +':')
    print('  Induction rate = ' + str(rate_induc))
    print('  Decay rate     = ' + str(rate_decay))
    print('  Minimum value  = ' + str(exp_min))
    print('  Maximum value  = ' + str(exp_max))
    print('')
    i += 1
d_N = 0.1                           # reset d_N to original value    

# Plotting the solution for all values on same graph
plt.xlabel('Hours post infection')
plt.ylabel('AMP expression')
plt.xticks([0,6,12,18,24,30,36,42,48])
plt.xlim(0, 50)
plt.yticks([0,1,2,3])
plt.ylim(0, 3)
plt.legend()
plt.show()


#%%
###############################################################################
# Graphical sensitivity analysis of d_bN ######################################
###############################################################################

test_values = [0.001, 0.1, 1]
A = []
A = [0 for i in range(len(test_values))]
label = []
label = ['' for i in range(len(test_values))]
i = 0
for d_bN in test_values: 
    # Solving ODE initial value problem for each parameter value
    soln = solve_ivp(f, t_span, y0, t_eval=times)
    t = soln.t
    A[i] = soln.y[0]
    label[i] = str(test_values[i])
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
    plt.plot(t, A[i], '-', label='d_bN = '+label[i])
    rate_induc = (soln.y[0][12] - soln.y[0][0])/6
    rate_decay = (soln.y[0][48] - soln.y[0][12])/6
    exp_min = min(soln.y[0])
    exp_max = max(soln.y[0])
    print('For d_bN = ' + label[i] +':')
    print('  Induction rate = ' + str(rate_induc))
    print('  Decay rate     = ' + str(rate_decay))
    print('  Minimum value  = ' + str(exp_min))
    print('  Maximum value  = ' + str(exp_max))
    print('')
    i += 1
d_bN = 0.1                           # reset d_bN to original value    

# Plotting the solution for all values on same graph
plt.xlabel('Hours post infection')
plt.ylabel('AMP expression')
plt.xticks([0,6,12,18,24,30,36,42,48])
plt.xlim(0, 50)
plt.yticks([0,1,2,3])
plt.ylim(0, 3)
plt.legend()
plt.show()


#%%
###############################################################################
# Graphical sensitivity analysis of d_NG1 #####################################
###############################################################################

test_values = [0.00001, 1, 1000]
A = []
A = [0 for i in range(len(test_values))]
label = []
label = ['' for i in range(len(test_values))]
i = 0
for d_NG1 in test_values: 
    # Solving ODE initial value problem for each parameter value
    soln = solve_ivp(f, t_span, y0, t_eval=times)
    t = soln.t
    A[i] = soln.y[0]
    label[i] = str(test_values[i])
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
    plt.plot(t, A[i], '-', label='d_NG1 = '+label[i])
    rate_induc = (soln.y[0][12] - soln.y[0][0])/6
    rate_decay = (soln.y[0][48] - soln.y[0][12])/6
    exp_min = min(soln.y[0])
    exp_max = max(soln.y[0])
    print('For d_NG1 = ' + label[i] +':')
    print('  Induction rate = ' + str(rate_induc))
    print('  Decay rate     = ' + str(rate_decay))
    print('  Minimum value  = ' + str(exp_min))
    print('  Maximum value  = ' + str(exp_max))
    print('')
    i += 1
d_NG1 = 0.05                           # reset d_NG1 to original value    

# Plotting the solution for all values on same graph
plt.xlabel('Hours post infection')
plt.ylabel('AMP expression')
plt.xticks([0,6,12,18,24,30,36,42,48])
plt.xlim(0, 50)
plt.yticks([0,1,2,3])
plt.ylim(0, 3)
plt.legend()
plt.show()


#%%
###############################################################################
# Graphical sensitivity analysis of d_NG2 #####################################
###############################################################################

test_values = [0.00001, 1, 1000]
A = []
A = [0 for i in range(len(test_values))]
label = []
label = ['' for i in range(len(test_values))]
i = 0
for d_NG2 in test_values: 
    # Solving ODE initial value problem for each parameter value
    soln = solve_ivp(f, t_span, y0, t_eval=times)
    t = soln.t
    A[i] = soln.y[0]
    label[i] = str(test_values[i])
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
    plt.plot(t, A[i], '-', label='d_NG2 = '+label[i])
    rate_induc = (soln.y[0][12] - soln.y[0][0])/6
    rate_decay = (soln.y[0][48] - soln.y[0][12])/6
    exp_min = min(soln.y[0])
    exp_max = max(soln.y[0])
    print('For d_NG2 = ' + label[i] +':')
    print('  Induction rate = ' + str(rate_induc))
    print('  Decay rate     = ' + str(rate_decay))
    print('  Minimum value  = ' + str(exp_min))
    print('  Maximum value  = ' + str(exp_max))
    print('')
    i += 1
d_NG2 = 0.05                           # reset d_NG2 to original value    

# Plotting the solution for all values on same graph
plt.xlabel('Hours post infection')
plt.ylabel('AMP expression')
plt.xticks([0,6,12,18,24,30,36,42,48])
plt.xlim(0, 50)
plt.yticks([0,1,2,3])
plt.ylim(0, 3)
plt.legend()
plt.show()


#%%
###############################################################################
# Graphical sensitivity analysis of s_A #######################################
###############################################################################

test_values = [0.4, 1, 3]
A = []
A = [0 for i in range(len(test_values))]
label = []
label = ['' for i in range(len(test_values))]
i = 0
for s_A in test_values: 
    # Solving ODE initial value problem for each parameter value
    soln = solve_ivp(f, t_span, y0, t_eval=times)
    t = soln.t
    A[i] = soln.y[0]
    label[i] = str(test_values[i])
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
    plt.plot(t, A[i], '-', label='s_A = '+label[i])
    rate_induc = (soln.y[0][12] - soln.y[0][0])/6
    rate_decay = (soln.y[0][48] - soln.y[0][12])/6
    exp_min = min(soln.y[0])
    exp_max = max(soln.y[0])
    print('For s_A = ' + label[i] +':')
    print('  Induction rate = ' + str(rate_induc))
    print('  Decay rate     = ' + str(rate_decay))
    print('  Minimum value  = ' + str(exp_min))
    print('  Maximum value  = ' + str(exp_max))
    print('')
    i += 1
s_A = 0.5                          # reset c_A to original value    

# Plotting the solution for all values on same graph
plt.xlabel('Hours post infection')
plt.ylabel('AMP expression')
plt.xticks([0,6,12,18,24,30,36,42,48])
plt.xlim(0, 50)
plt.yticks([0,1,2,3])
plt.ylim(0, 3)
plt.legend()
plt.show()


#%%
###############################################################################
# Graphical sensitivity analysis of s_P #######################################
###############################################################################

test_values = [0.001, 1, 10]
A = []
A = [0 for i in range(len(test_values))]
label = []
label = ['' for i in range(len(test_values))]
i = 0
for s_P in test_values: 
    # Solving ODE initial value problem for each parameter value
    soln = solve_ivp(f, t_span, y0, t_eval=times)
    t = soln.t
    A[i] = soln.y[0]
    label[i] = str(test_values[i])
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
    plt.plot(t, A[i], '-', label='s_P = '+label[i])
    rate_induc = (soln.y[0][12] - soln.y[0][0])/6
    rate_decay = (soln.y[0][48] - soln.y[0][12])/6
    exp_min = min(soln.y[0])
    exp_max = max(soln.y[0])
    print('For s_P = ' + label[i] +':')
    print('  Induction rate = ' + str(rate_induc))
    print('  Decay rate     = ' + str(rate_decay))
    print('  Minimum value  = ' + str(exp_min))
    print('  Maximum value  = ' + str(exp_max))
    print('')
    i += 1
s_P = 0.01                          # reset c_A to original value    

# Plotting the solution for all values on same graph
plt.xlabel('Hours post infection')
plt.ylabel('AMP expression')
plt.xticks([0,6,12,18,24,30,36,42,48])
plt.xlim(0, 50)
plt.yticks([0,1,2,3])
plt.ylim(0, 3)
plt.legend()
plt.show()


#%%
###############################################################################
# Graphical sensitivity analysis of s_B #######################################
###############################################################################

test_values = [0.0001, 0.1]
A = []
A = [0 for i in range(len(test_values))]
label = []
label = ['' for i in range(len(test_values))]
i = 0
for s_B in test_values: 
    # Solving ODE initial value problem for each parameter value
    soln = solve_ivp(f, t_span, y0, t_eval=times)
    t = soln.t
    A[i] = soln.y[0]
    label[i] = str(test_values[i])
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
    plt.plot(t, A[i], '-', label='s_B = '+label[i])
    rate_induc = (soln.y[0][12] - soln.y[0][0])/6
    rate_decay = (soln.y[0][48] - soln.y[0][12])/6
    exp_min = min(soln.y[0])
    exp_max = max(soln.y[0])
    print('For s_B = ' + label[i] +':')
    print('  Induction rate = ' + str(rate_induc))
    print('  Decay rate     = ' + str(rate_decay))
    print('  Minimum value  = ' + str(exp_min))
    print('  Maximum value  = ' + str(exp_max))
    print('')
    i += 1
s_B = 0.01                          # reset c_A to original value    

# Plotting the solution for all values on same graph
plt.xlabel('Hours post infection')
plt.ylabel('AMP expression')
plt.xticks([0,6,12,18,24,30,36,42,48])
plt.xlim(0, 50)
plt.yticks([0,1,2,3])
plt.ylim(0, 3)
plt.legend()
plt.show()



#%%
###############################################################################
# Graphical sensitivity analysis of c_A #######################################
###############################################################################

test_values = [0.5, 1, 20]
A = []
A = [0 for i in range(len(test_values))]
label = []
label = ['' for i in range(len(test_values))]
i = 0
for c_A in test_values: 
    # Solving ODE initial value problem for each parameter value
    soln = solve_ivp(f, t_span, y0, t_eval=times)
    t = soln.t
    A[i] = soln.y[0]
    label[i] = str(test_values[i])
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
    plt.plot(t, A[i], '-', label='c_A = '+label[i])
    rate_induc = (soln.y[0][12] - soln.y[0][0])/6
    rate_decay = (soln.y[0][48] - soln.y[0][12])/6
    exp_min = min(soln.y[0])
    exp_max = max(soln.y[0])
    print('For c_A = ' + label[i] +':')
    print('  Induction rate = ' + str(rate_induc))
    print('  Decay rate     = ' + str(rate_decay))
    print('  Minimum value  = ' + str(exp_min))
    print('  Maximum value  = ' + str(exp_max))
    print('')
    i += 1
c_A = 0.8                           # reset c_A to original value    

# Plotting the solution for all values on same graph
plt.xlabel('Hours post infection')
plt.ylabel('AMP expression')
plt.xticks([0,6,12,18,24,30,36,42,48])
plt.xlim(0, 50)
plt.yticks([0,1,2,3])
plt.ylim(0, 3)
plt.legend()
plt.show()


#%%
###############################################################################
# Graphical sensitivity analysis of c_P #######################################
###############################################################################

test_values = [0.01, 0.1, 100]
A = []
A = [0 for i in range(len(test_values))]
label = []
label = ['' for i in range(len(test_values))]
i = 0
for c_P in test_values: 
    # Solving ODE initial value problem for each parameter value
    soln = solve_ivp(f, t_span, y0, t_eval=times)
    t = soln.t
    A[i] = soln.y[0]
    label[i] = str(test_values[i])
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
    plt.plot(t, A[i], '-', label='c_P = '+label[i])
    rate_induc = (soln.y[0][12] - soln.y[0][0])/6
    rate_decay = (soln.y[0][48] - soln.y[0][12])/6
    exp_min = min(soln.y[0])
    exp_max = max(soln.y[0])
    print('For c_P = ' + label[i] +':')
    print('  Induction rate = ' + str(rate_induc))
    print('  Decay rate     = ' + str(rate_decay))
    print('  Minimum value  = ' + str(exp_min))
    print('  Maximum value  = ' + str(exp_max))
    print('')
    i += 1
c_P = 0.5                           # reset c_P to original value    

# Plotting the solution for all values on same graph
plt.xlabel('Hours post infection')
plt.ylabel('AMP expression')
plt.xticks([0,6,12,18,24,30,36,42,48])
plt.xlim(0, 50)
plt.yticks([0,1,2,3])
plt.ylim(0, 3)
plt.legend()
plt.show()


#%%
###############################################################################
# Graphical sensitivity analysis of c_I #######################################
###############################################################################

test_values = [0.01, 0.1, 100]
A = []
A = [0 for i in range(len(test_values))]
label = []
label = ['' for i in range(len(test_values))]
i = 0
for c_I in test_values: 
    # Solving ODE initial value problem for each parameter value
    soln = solve_ivp(f, t_span, y0, t_eval=times)
    t = soln.t
    A[i] = soln.y[0]
    label[i] = str(test_values[i])
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
    plt.plot(t, A[i], '-', label='c_I = '+label[i])
    rate_induc = (soln.y[0][12] - soln.y[0][0])/6
    rate_decay = (soln.y[0][48] - soln.y[0][12])/6
    exp_min = min(soln.y[0])
    exp_max = max(soln.y[0])
    print('For c_I = ' + label[i] +':')
    print('  Induction rate = ' + str(rate_induc))
    print('  Decay rate     = ' + str(rate_decay))
    print('  Minimum value  = ' + str(exp_min))
    print('  Maximum value  = ' + str(exp_max))
    print('')
    i += 1
c_I = 1                           # reset c_I to original value    

# Plotting the solution for all values on same graph
plt.xlabel('Hours post infection')
plt.ylabel('AMP expression')
plt.xticks([0,6,12,18,24,30,36,42,48])
plt.xlim(0, 50)
plt.yticks([0,1,2,3])
plt.ylim(0, 3)
plt.legend()
plt.show()


#%%
###############################################################################
# Graphical sensitivity analysis of c_B #######################################
###############################################################################

test_values = [0.1, 0.5, 10]
A = []
A = [0 for i in range(len(test_values))]
label = []
label = ['' for i in range(len(test_values))]
i = 0
for c_B in test_values: 
    # Solving ODE initial value problem for each parameter value
    soln = solve_ivp(f, t_span, y0, t_eval=times)
    t = soln.t
    A[i] = soln.y[0]
    label[i] = str(test_values[i])
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
    plt.plot(t, A[i], '-', label='c_B = '+label[i])
    rate_induc = (soln.y[0][12] - soln.y[0][0])/6
    rate_decay = (soln.y[0][48] - soln.y[0][12])/6
    exp_min = min(soln.y[0])
    exp_max = max(soln.y[0])
    print('For c_B = ' + label[i] +':')
    print('  Induction rate = ' + str(rate_induc))
    print('  Decay rate     = ' + str(rate_decay))
    print('  Minimum value  = ' + str(exp_min))
    print('  Maximum value  = ' + str(exp_max))
    print('')
    i += 1
c_B = 4                           # reset c_B to original value    

# Plotting the solution for all values on same graph
plt.xlabel('Hours post infection')
plt.ylabel('AMP expression')
plt.xticks([0,6,12,18,24,30,36,42,48])
plt.xlim(0, 50)
plt.yticks([0,1,2,3])
plt.ylim(0, 3)
plt.legend()
plt.show()


#%%
###############################################################################
# Graphical sensitivity analysis of c_N #######################################
###############################################################################

test_values = [1, 5, 100]
A = []
A = [0 for i in range(len(test_values))]
label = []
label = ['' for i in range(len(test_values))]
i = 0
for c_N in test_values: 
    # Solving ODE initial value problem for each parameter value
    soln = solve_ivp(f, t_span, y0, t_eval=times)
    t = soln.t
    A[i] = soln.y[0]
    label[i] = str(test_values[i])
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
    plt.plot(t, A[i], '-', label='c_N = '+label[i])
    rate_induc = (soln.y[0][12] - soln.y[0][0])/6
    rate_decay = (soln.y[0][48] - soln.y[0][12])/6
    exp_min = min(soln.y[0])
    exp_max = max(soln.y[0])
    print('For c_N = ' +label[i] +':')
    print('  Induction rate = '+str(rate_induc))
    print('  Decay rate     = '+str(rate_decay))
    print('  Minimum value  = ' + str(exp_min))
    print('  Maximum value  = ' + str(exp_max))
    print('')
    i += 1
c_N = 3

# Plotting the solution for all values on same graph
plt.xlabel('Hours post infection')
plt.ylabel('AMP expression')
plt.xticks([0,6,12,18,24,30,36,42,48])
plt.xlim(0, 50)
plt.yticks([0,1,2,3])
plt.ylim(0, 3)
plt.legend()
plt.show()

#%%
###############################################################################
# Graphical sensitivity analysis of c_bN ######################################
###############################################################################

test_values = [0.1, 2, 10]
A = []
A = [0 for i in range(len(test_values))]
label = []
label = ['' for i in range(len(test_values))]
i = 0
for c_bN in test_values: 
    # Solving ODE initial value problem for each parameter value
    soln = solve_ivp(f, t_span, y0, t_eval=times)
    t = soln.t
    A[i] = soln.y[0]
    label[i] = str(test_values[i])
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
    plt.plot(t, A[i], '-', label='c_bN = '+label[i])
    rate_induc = (soln.y[0][12] - soln.y[0][0])/6
    rate_decay = (soln.y[0][48] - soln.y[0][12])/6
    exp_min = min(soln.y[0])
    exp_max = max(soln.y[0])
    print('For c_bN = ' +label[i] +':')
    print('  Induction rate = '+str(rate_induc))
    print('  Decay rate     = '+str(rate_decay))
    print('  Minimum value  = ' + str(exp_min))
    print('  Maximum value  = ' + str(exp_max))
    print('')
    i += 1
c_bN = 2.5

# Plotting the solution for all values on same graph
plt.xlabel('Hours post infection')
plt.ylabel('AMP expression')
plt.xticks([0,6,12,18,24,30,36,42,48])
plt.xlim(0, 50)
plt.yticks([0,1,2,3])
plt.ylim(0, 3)
plt.legend()
plt.show()


#%%
###############################################################################
# Graphical sensitivity analysis of c_NG1 #####################################
###############################################################################

test_values = [0.001, 1, 100]
A = []
A = [0 for i in range(len(test_values))]
label = []
label = ['' for i in range(len(test_values))]
i = 0
for c_NG1 in test_values: 
    # Solving ODE initial value problem for each parameter value
    soln = solve_ivp(f, t_span, y0, t_eval=times)
    t = soln.t
    A[i] = soln.y[0]
    label[i] = str(test_values[i])
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
    plt.plot(t, A[i], '-', label='c_NG1 = '+label[i])
    rate_induc = (soln.y[0][12] - soln.y[0][0])/6
    rate_decay = (soln.y[0][48] - soln.y[0][12])/6
    exp_min = min(soln.y[0])
    exp_max = max(soln.y[0])
    print('For c_NG1 = ' +label[i] +':')
    print('  Induction rate = '+str(rate_induc))
    print('  Decay rate     = '+str(rate_decay))
    print('  Minimum value  = ' + str(exp_min))
    print('  Maximum value  = ' + str(exp_max))
    print('')
    i += 1
c_NG1 = 0.1

# Plotting the solution for all values on same graph
plt.xlabel('Hours post infection')
plt.ylabel('AMP expression')
plt.xticks([0,6,12,18,24,30,36,42,48])
plt.xlim(0, 50)
plt.yticks([0,1,2,3])
plt.ylim(0, 3)
plt.legend()
plt.show()


#%%
###############################################################################
# Graphical sensitivity analysis of c_NG2 #####################################
###############################################################################

test_values = [0.001, 1, 100]
A = []
A = [0 for i in range(len(test_values))]
label = []
label = ['' for i in range(len(test_values))]
i = 0
for c_NG2 in test_values: 
    # Solving ODE initial value problem for each parameter value
    soln = solve_ivp(f, t_span, y0, t_eval=times)
    t = soln.t
    A[i] = soln.y[0]
    label[i] = str(test_values[i])
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
    plt.plot(t, A[i], '-', label='c_NG2 = '+label[i])
    rate_induc = (soln.y[0][12] - soln.y[0][0])/6
    rate_decay = (soln.y[0][48] - soln.y[0][12])/6
    exp_min = min(soln.y[0])
    exp_max = max(soln.y[0])
    print('For c_NG2 = ' +label[i] +':')
    print('  Induction rate = '+str(rate_induc))
    print('  Decay rate     = '+str(rate_decay))
    print('  Minimum value  = ' + str(exp_min))
    print('  Maximum value  = ' + str(exp_max))
    print('')
    i += 1
c_NG2 = 0.2

# Plotting the solution for all values on same graph
plt.xlabel('Hours post infection')
plt.ylabel('AMP expression')
plt.xticks([0,6,12,18,24,30,36,42,48])
plt.xlim(0, 50)
plt.yticks([0,1,2,3])
plt.ylim(0, 3)
plt.legend()
plt.show()


#%%
###############################################################################
# Graphical sensitivity analysis of f_P #######################################
###############################################################################

test_values = [0.05, 0.5, 10]
A = []
A = [0 for i in range(len(test_values))]
label = []
label = ['' for i in range(len(test_values))]
i = 0
for f_P in test_values: 
    # Solving ODE initial value problem for each parameter value
    soln = solve_ivp(f, t_span, y0, t_eval=times)
    t = soln.t
    A[i] = soln.y[0]
    label[i] = str(test_values[i])
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
    plt.plot(t, A[i], '-', label='f_P = '+label[i])
    rate_induc = (soln.y[0][12] - soln.y[0][0])/6
    rate_decay = (soln.y[0][48] - soln.y[0][12])/6
    exp_min = min(soln.y[0])
    exp_max = max(soln.y[0])
    print('For f_P = ' +label[i] +':')
    print('  Induction rate = '+str(rate_induc))
    print('  Decay rate     = '+str(rate_decay))
    print('  Minimum value  = ' + str(exp_min))
    print('  Maximum value  = ' + str(exp_max))
    print('')
    i += 1
f_P = 9

# Plotting the solution for all values on same graph
plt.xlabel('Hours post infection')
plt.ylabel('AMP expression')
plt.xticks([0,6,12,18,24,30,36,42,48])
plt.xlim(0, 50)
plt.yticks([0,1,2,3])
plt.ylim(0, 3)
plt.legend()
plt.show()


#%%
###############################################################################
# Graphical sensitivity analysis of phi_NG1 ###################################
###############################################################################

test_values = [0.01, 15, 20]
A = []
A = [0 for i in range(len(test_values))]
label = []
label = ['' for i in range(len(test_values))]
i = 0
for phi_NG1 in test_values: 
    # Solving ODE initial value problem for each parameter value
    soln = solve_ivp(f, t_span, y0, t_eval=times)
    t = soln.t
    A[i] = soln.y[0]
    label[i] = str(test_values[i])
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
    plt.plot(t, A[i], '-', label='phi_NG1 = '+label[i])
    rate_induc = (soln.y[0][12] - soln.y[0][0])/6
    rate_decay = (soln.y[0][48] - soln.y[0][12])/6
    exp_min = min(soln.y[0])
    exp_max = max(soln.y[0])
    print('For phi_NG1 = ' + label[i] +':')
    print('  Induction rate = '+str(rate_induc))
    print('  Decay rate     = '+str(rate_decay))
    print('  Minimum value  = ' + str(exp_min))
    print('  Maximum value  = ' + str(exp_max))
    print('')
    i += 1
phi_NG1 = 0.07

# Plotting the solution for all values on same graph
plt.xlabel('Hours post infection')
plt.ylabel('AMP expression')
plt.xticks([0,6,12,18,24,30,36,42,48])
plt.xlim(0, 50)
plt.yticks([0,1,2,3])
plt.ylim(0, 3)
plt.legend()
plt.show()


#%%
###############################################################################
# Graphical sensitivity analysis of phi_NG2 ###################################
###############################################################################

test_values = [0.01, 3, 5]
A = []
A = [0 for i in range(len(test_values))]
label = []
label = ['' for i in range(len(test_values))]
i = 0
for phi_NG2 in test_values: 
    # Solving ODE initial value problem for each parameter value
    soln = solve_ivp(f, t_span, y0, t_eval=times)
    t = soln.t
    A[i] = soln.y[0]
    label[i] = str(test_values[i])
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
    plt.plot(t, A[i], '-', label='phi_NG2 = '+label[i])
    rate_induc = (soln.y[0][12] - soln.y[0][0])/6
    rate_decay = (soln.y[0][48] - soln.y[0][12])/6
    exp_min = min(soln.y[0])
    exp_max = max(soln.y[0])
    print('For phi_NG2 = ' + label[i] +':')
    print('  Induction rate = '+str(rate_induc))
    print('  Decay rate     = '+str(rate_decay))
    print('  Minimum value  = ' + str(exp_min))
    print('  Maximum value  = ' + str(exp_max))
    print('')
    i += 1
phi_NG2 = 0.32

# Plotting the solution for all values on same graph
plt.xlabel('Hours post infection')
plt.ylabel('AMP expression')
plt.xticks([0,6,12,18,24,30,36,42,48])
plt.xlim(0, 50)
plt.yticks([0,1,2,3])
plt.ylim(0, 3)
plt.legend()
plt.show()


#%%
###############################################################################
# Graphical sensitivity analysis of p_G #######################################
###############################################################################

test_values = [0.01, 1, 2.5]
A = []
A = [0 for i in range(len(test_values))]
label = []
label = ['' for i in range(len(test_values))]
i = 0
for p_G in test_values: 
    # Solving ODE initial value problem for each parameter value
    soln = solve_ivp(f, t_span, y0, t_eval=times)
    t = soln.t
    A[i] = soln.y[0]
    label[i] = str(test_values[i])
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
    plt.plot(t, A[i], '-', label='p_G = '+label[i])
    rate_induc = (soln.y[0][12] - soln.y[0][0])/6
    rate_decay = (soln.y[0][48] - soln.y[0][12])/6
    exp_min = min(soln.y[0])
    exp_max = max(soln.y[0])
    print('For p_G = ' + label[i] +':')
    print('  Induction rate = ' + str(rate_induc))
    print('  Decay rate     = ' + str(rate_decay))
    print('  Minimum value  = ' + str(exp_min))
    print('  Maximum value  = ' + str(exp_max))
    print('')
    i += 1
p_G = 2.4                 # reset d_A to original value    

# Plotting the solution for all values on same graph
plt.xlabel('Hours post infection')
plt.ylabel('AMP expression')
plt.xticks([0,6,12,18,24,30,36,42,48])
plt.xlim(0, 50)
plt.yticks([0,1,2,3])
plt.ylim(0, 3)
plt.legend()
plt.show()


#%%
###############################################################################
# Graphical sensitivity analysis of m_G #######################################
###############################################################################

test_values = [2, 3, 5]
A = []
A = [0 for i in range(len(test_values))]
label = []
label = ['' for i in range(len(test_values))]
i = 0
for m_G in test_values: 
    # Solving ODE initial value problem for each parameter value
    soln = solve_ivp(f, t_span, y0, t_eval=times)
    t = soln.t
    A[i] = soln.y[0]
    label[i] = str(test_values[i])
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
    plt.plot(t, A[i], '-', label='m_G = '+label[i])
    rate_induc = (soln.y[0][12] - soln.y[0][0])/6
    rate_decay = (soln.y[0][48] - soln.y[0][12])/6
    exp_min = min(soln.y[0])
    exp_max = max(soln.y[0])
    print('For m_G = ' + label[i] +':')
    print('  Induction rate = ' + str(rate_induc))
    print('  Decay rate     = ' + str(rate_decay))
    print('  Minimum value  = ' + str(exp_min))
    print('  Maximum value  = ' + str(exp_max))
    print('')
    i += 1
m_G = 2                 # reset m_G to original value    

# Plotting the solution for all values on same graph
plt.xlabel('Hours post infection')
plt.ylabel('AMP expression')
plt.xticks([0,6,12,18,24,30,36,42,48])
plt.xlim(0, 50)
plt.yticks([0,1,2,3])
plt.ylim(0, 3)
plt.legend()
plt.show()

