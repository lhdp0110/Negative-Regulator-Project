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
def2 setB: 1.22, 6.72, 9.33, 7.61, 7.06, 4.29, 3.41 (modified by adding +1 so init_A is >1)
def2 avg: 1.4, 6.325,7.69, 6.975, 6.735, 4.6, 3.7

def3 setA: 3.00, 4.77, 3.13, 5.07, 3.51, 1.49, 2.15
def3 setB: 2.10, 3.87, 4.22, 5.40, 3.55, 1.60, 2.17
"""


import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

### List of all the ODE system parameters:

init_A = 1.4
tot_I = init_A

d_A = 0.208     
d_G = 0.2
d_P = 0.3  
d_bP = 0.3    
d_bI = 0.3   
d_R = 0.3   
d_N = 0.3   
d_bN = 0.3

s_A = d_A      
s_P = d_P      
s_B = d_R       

c_A = 3.03   #adjusts how high A peaks
c_P = 0.5 
c_I = 0.5  
c_B = 1    
c_N = 0.62
c_bN = 1.02

f_P = 14  
p_G = 2  
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
concentration = np.array([1.4, 6.325,7.69, 6.975, 6.735, 4.6, 3.7])

# Define our ODE model
t, A, G, P, bP, bI, RB, RP, RN, bRN = variables('t, A, G, P, bP, bI, RB, RP, RN, bRN')

#d_A = Parameter('d_A', 0.2, min=0)
c_A = Parameter('c_A', 3, min=0)
c_P = Parameter('c_P', 0.5, min=0)
c_I = Parameter('c_I', 0.5, min=0)
c_B = Parameter('c_B', 1, min=0)
c_N = Parameter('c_N', 0.5, min=0)
#c_bN = Parameter('c_bN', 1, min=0)
#f_P = Parameter('f_P', 14, min=0)


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