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
    
cec2 setA: 3.69, 3.28, 4.25, 4.91, 4.61, 2.03, 2.78
cec2 setB: 2.82, 4.11, 4.55, 6.00, 5.11, 4.12, 4.35
cec2 avg: 3.255, 3.695, 4.4, 5.455, 4.86, 3.075, 3.565

def2 setA: 1.58, 5.93, 6.05, 6.34, 6.41, 4.91, 3.99
def2 setB: 0.22, 5.72, 8.33, 6.61, 6.06, 3.29, 2.41

def3 setA: 3.00, 4.77, 3.13, 5.07, 3.51, 1.49, 2.15
def3 setB: 2.10, 3.87, 4.22, 5.40, 3.55, 1.60, 2.17
"""


import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

### List of all the ODE system parameters:

init_A = 2.82
tot_I = init_A

# degradation/inactivation rates
d_A = 0.43      # 0.02 (delta_A) per Ellner Fig.3,4 /// 0.05 per Table S3 // # can only fit if 0.5

d_G = 0.1       # 0.02 (delta_P) per Ellner Fig.4
d_P = 0.1     # 0.01 (delta_P) per Ellner Table S3
d_bP = 0.1      # 0.02 (delta_P*) per Ellner Table S3
d_bI = 0.1     # 0.01 (rho_I*) per Ellner Table S3
d_R = 0.1       # 0.01 (delta_R) per Ellner Table S3
d_N = 0.1      # 0.01 (delta_N) per Ellner Table S3
d_bN = 0.1

# supply rates
s_A = d_A       # (Q_A = delta_A) per Ellner p.15
s_P = d_P       # (Q_P = delta_P) per Ellner p.15
s_B = d_R       # (Q_R = delta_R) per Ellner p.15

# setting all supply and degradation rates to the same value 
# eliminates irrelevant variations and allows values to return to baseline

# reaction/binding rates
c_A = 2.37     # (default = 0.8) this parameter adjusts how high A peaks
c_P = 0.5     # (new default = 0.5)
c_I = 0.5      # (default = 0.5)
c_B = 2      # (default = 4) can be higher than 4 but not lower than 2
c_N = 2    # (default = 3) decreasing to 1 lowers peak of A and induces small lift around t=48, increasing beyond 3 doesn't change much
c_bN = 5     # (default = 0.5) variations alter shape and peak of A, but not by much

# carrying capacity
K_I = 1

# flow rate into nucleus
f_P =  9.48       # (default = 10) increasing to 20 or higher doesn't change much, decreasing to 1 flattens A and incudes small lift at t=48

# proliferation rate
p_G = 3       # (new default = 2.8) 2.5, per Frank. alters G more than A

# mortality rate
m_G = 0.95         # (default = 2) max=5, per Frank. alters G more than A

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
y0 = np.array([init_A, init_A, init_A, 0, 0, init_A, 0, 0, 0])

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
plt.ylim(0,7)
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

from symfit import variables, Parameter, Fit, D, ODEModel, exp
import numpy as np
import matplotlib.pyplot as plt

tdata = np.array([0, 2, 4, 8, 12, 24, 48])
concentration = np.array([ 2.82, 4.11, 4.55, 6.00, 5.11, 4.12, 4.35])

# Define our ODE model
t, A, G, P, bP, bI, RB, RP, RN, bRN = variables('t, A, G, P, bP, bI, RB, RP, RN, bRN')


d_A = Parameter('d_A', 0.43, min=0)
c_A = Parameter('c_A', 2.4, min=0)
c_N = Parameter('c_N', 2.15, min=0)
c_bN = Parameter('c_bN', 5, min=0)
f_P = Parameter('f_P', 9.5, min=0)


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
    initial={t: tdata[0], A: concentration[0], G: init_A, P: init_A, bP: 0, bI: 0, RB: init_A, RP: 0, RN: 0, bRN: 0}
)

fit = Fit(model, t=tdata, A=concentration, G=None, P=None, bP=None, bI=None, RB=None, RP=None, RN=None,
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
plt.legend()
plt.show()