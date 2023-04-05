#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 13:07:33 2023

@author: perrielh
"""


from scipy import integrate as sp
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
import SALib
from SALib.sample import saltelli
from SALib.analyze import sobol

# degradation/inactivation rates
d_A = 0.5      # 0.02 (delta_A) per Ellner Fig.3,4 /// 0.05 per Table S3
d_G = 0.02       # 0.02 (delta_P) per Ellner Fig.4
d_P = 0.01       # 0.01 (delta_P) per Ellner Table S3
d_bP = 0.02      # 0.02 (delta_P*) per Ellner Table S3
d_bI = 0.01      # 0.01 (rho_I*) per Ellner Table S3
d_R = 0.01       # 0.01 (delta_R) per Ellner Table S3
d_N = 0.01       # 0.01 (delta_N) per Ellner Table S3
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
p_G = 2.5       # (new default = 2.8) 2.5, per Frank. alters G more than A

# mortality rate
m_G = 2            # (default = 2) max=5, per Frank. alters G more than A


# definition of the system of ODEs
def model(y, t, d_A, d_N, c_A, c_N, c_bN):
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
    #val_base = y[11]
    #val_max = y[12]
   # rate_induc = y[13]
   # rate_decay = y[14]
    
   # val_base = A[0]
    #val_max = max(A)
    #rate_induc = (A[6]-A[0])/6
    #rate_decay = (A[12]-A[6]/6)
    
    dA_dt = s_A + c_A * bRN - d_A * A                       # AMPs transcribed by Relish
    dG_dt = p_G * G - m_G * G * A - d_G * G                 # pathogen present outside the cell
    
    dP_dt = s_P - 2 * c_P * P**2 * G * exp(-phi_NG1 * NG1) - d_P * P              # unbound PGRP-LC monomer
    dbP_dt = c_P * P**2 * G * exp(-phi_NG1 * NG1) - d_bP * bP                     # PGRP-LC dimer bound to pathogen
   
    dbI_dt = c_I * (1-bI/K_I) * bP * exp(-phi_NG2 * NG2) - d_bI * bI                  # IMD recruited by activated PGRP-LC
   
    dRB_dt = s_B - c_B * RB * bI - d_R * RB                 # Relish B in cytoplasm
    dRP_dt = c_B * RB * bI - f_P * RP - d_R * RP            # Relish P in cytoplasm
   
    dRN_dt = f_P * RP - c_N * RN + c_bN * bRN - d_N * RN    # free nuclear Relish
    dbRN_dt = c_N * RN - c_bN * bRN            # nuclear Relish bound to target gene
   
    dNG1_dt = s_NG1 + c_NG1 * bRN - d_NG1 * NG1             # neg reg at level 1 (PGRP-SC2/LB, pathogen binding)
    dNG2_dt = s_NG2 + c_NG2 * bRN - d_NG2 *NG2              # neg reg at level 2 (PIRK, IMD recruitment)

    return np.array([dA_dt, dG_dt, dP_dt, dbP_dt, dbI_dt, dRB_dt, dRP_dt, dRN_dt, dbRN_dt, dNG1_dt, dNG2_dt])

t_span = np.array([0, 50])
t = np.linspace(t_span[0], t_span[1], 101)

# Initial conditions of ODE system
y0 = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1])

#new addition here
y = sp.odeint(model, y0, t, args=(d_A,d_N,c_A,c_N,c_bN))

problem = {
  'num_vars': 5, 
  'names': ['d_A', 'd_N','c_A','c_N','c_bN'],
  'bounds': [[0.25, 0.5],
             [0.01, 1],
             [0.5, 2],
             [1, 100],
             [0.1, 10]]
}

# Generate samples
param_values = saltelli.sample(problem, 50)

#new version here
Y = np.zeros([len(param_values),11])

for i in range(len(param_values)):
    Y[i][:] = sp.odeint(model,y0,t,args=(param_values[i][0],param_values[i][1],param_values[i][2],param_values[i][3],param_values[i][4]))[len(y)-1]
    
    
print('\n\n====AMP Sobol output====\n\n')
Si_AMP = sobol.analyze(problem, Y[:,0], print_to_console=True)

Si_AMP.plot()








