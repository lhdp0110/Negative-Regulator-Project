#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:36:25 2022

@author: perrielh

sourced from script_16
"""

from symfit import variables, Parameter, Fit, D, ODEModel, exp
import numpy as np
import matplotlib.pyplot as plt

# degradation/inactivation rates
#d_A = 0.5   #******             # 0.02 (delta_A) per Ellner Fig.3,4 /// 0.05 per Table S3
d_G = 0.02                   # 0.02 (delta_P) per Ellner Fig.4
d_P = 0.01                  # 0.01 (delta_P) per Ellner Table S3
d_bP = 0.02                     # 0.02 (delta_P*) per Ellner Table S3
d_bI = 0.01                          # 0.01 (rho_I*) per Ellner Table S3
d_R = 0.01                       # 0.01 (delta_R) per Ellner Table S3
#d_N = 0.15 #*******             # 0.01 (delta_N) per Ellner Table S3
d_NG1 = 0.05                    # 0.05 (delta_H) per Ellner Table S3
d_NG2 = 0.05                      # 0.05 (delta_H) per Ellner Table S3


# supply rates
s_A = 0.5                       # (Q_A = delta_A) per Ellner p.15
s_P = d_P                       # (Q_P = delta_P) per Ellner p.15
s_B = d_R                         # (Q_R = delta_R) per Ellner p.15
s_NG1 = d_NG1                   # (Q_H = delta_H) per Ellner p.15
s_NG2 = d_NG1                     # (Q_H = delta_H) per Ellner p.15
    
# setting all supply and degradation rates to the same value 
# eliminates irrelevant variations and allows values to return to baseline

# reaction/binding rates
#c_A = 0.9  #*****       # (default = 0.8) this parameter adjusts how high A peaks
c_P = 0.5               # (new default = 0.5)
c_I = 0.5                # (default = 0.5)
c_B = 4                 # (default = 4) can be higher than 4 but not lower than 2
#c_N = 3  #*****          # (default = 3) decreasing to 1 lowers peak of A and induces small lift around t=48, increasing beyond 3 doesn't change much
#c_bN = 1  #******      # (default = 0.5) variations alter shape and peak of A, but not by much
c_NG1 = 0.1
c_NG2 = 0.2

# carrying capacity
K_I = 1

# flow rate into nucleus
f_P =  10      # (default = 10) increasing to 20 or higher doesn't change much, decreasing to 1 flattens A and incudes small lift at t=48

# strength of feedback effect
phi_NG1 = 0.01
phi_NG2 = 0.01

# proliferation rate
p_G = 1.9       # (new default = 2.8) 2.5, per Frank. alters G more than A

# mortality rate
m_G = 1.15 

tdata = np.array([0, 2, 4, 8, 12, 24, 48])
concentration = np.array([2.5, 4.3, 3.7, 5.2, 3.5, 1.5, 2.1])

# Define our ODE model
t, A, G, P, bP, bI, RB, RP, RN, bRN, NG1, NG2 = variables('t, A, G, P, bP, bI, RB, RP, RN, bRN, NG1, NG2')

d_A = Parameter('d_A', 0.5, min=0)
d_N = Parameter('d_N', 0.15, min=0)
c_A = Parameter('c_A', 0.8, min=0)
c_N = Parameter('c_N', 3, min=0)
c_bN = Parameter('c_bN', 1, min=0)

model = ODEModel({
    D(A, t): s_A + c_A * bRN - d_A * A,
    D(G, t): p_G * G - m_G * G * A - d_G * G,
    
    D(P, t): s_P - 2 * c_P * P**2 * G * exp(-phi_NG1 * NG1) - d_P * P,
    D(bP, t): c_P * P**2 * G * exp(-phi_NG1 * NG1) - d_bP * bP,
   
    D(bI, t): c_I * (1-bI/K_I) * bP * exp(-phi_NG2 * NG2) - d_bI * bI,
   
    D(RB, t): s_B - c_B * RB * bI - d_R * RB,
    D(RP, t): c_B * RB * bI - f_P * RP - d_R * RP,
   
    D(RN, t): f_P * RP - c_N * RN + c_bN * bRN - d_N * RN,
    D(bRN, t): c_N * RN - c_bN * bRN,
   
    D(NG1, t): s_NG1 + c_NG1 * bRN - d_NG1 * NG1,
    D(NG2, t): s_NG2 + c_NG2 * bRN - d_NG2 * NG2
    },
    initial={t: tdata[0], A: concentration[0], G: 1, P: 1, bP: 0, bI: 0, RB: 1, RP: 0, RN: 0, bRN: 0, NG1: 1, NG2: 1}
)

fit = Fit(model, t=tdata, A=concentration, G=None, P=None, bP=None, bI=None, RB=None, RP=None, RN=None,
          bRN=None, NG1=None, NG2=None)
fit_result = fit.execute()

print(fit_result)

taxis = np.linspace(0, 50)
A_fit, G_fit, NG1_fit, NG2_fit, P_fit, RB_fit,  RN_fit, RP_fit, bI_fit, bP_fit, bRN_fit,  = model(t=taxis, **fit_result.params)

plt.scatter(tdata, concentration)
plt.plot(taxis, A_fit, label='[A]')
plt.plot(taxis, G_fit, label='[G]')
plt.plot(taxis, P_fit, label='[P]')
plt.plot(taxis, bP_fit, label='[bP]')
plt.plot(taxis, bI_fit, label='[bI]')
plt.plot(taxis, RB_fit, label='[RB]')
plt.plot(taxis, RP_fit, label='[RP]')
plt.plot(taxis, RN_fit, label='[RN]')
plt.plot(taxis, bRN_fit, label='[bRN]')
plt.plot(taxis, NG1_fit, label='[NG1]')
#plt.plot(taxis, NG2_fit, label='[NG2]')
plt.xlabel('Hours post infection')
plt.ylabel('[X]')
#plt.ylim(0, 3)
plt.xlim(0, 50)
plt.legend()
plt.show()
