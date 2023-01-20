#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 13:25:21 2023

@author: perrielh
"""
from symfit import variables, Parameter, Fit, D, ODEModel, exp
from symfit.core.objectives import LogLikelihood
from symfit.core.minimizers import DifferentialEvolution
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

cec2_dataset = pd.read_excel('test_dataset.xlsx', sheet_name='cec2')
tdata = np.array([0, 2, 4, 6, 8, 10, 12, 24, 36, 48])

cec2_kd_concentration = np.array(cec2_dataset['cac_avg'].tolist())
cec2_kd_std_dev = np.array(cec2_dataset['cac_std_dev'].tolist())
cec2_ctrl_concentration = np.array(cec2_dataset['mal_avg'].tolist())
cec2_ctrl_std_dev = np.array(cec2_dataset['mal_std_dev'].tolist())

plt.scatter(tdata, cec2_kd_concentration, label = 'Cactus KD')
plt.errorbar(tdata, cec2_kd_concentration, yerr = cec2_kd_std_dev)
plt.scatter(tdata, cec2_ctrl_concentration, label = 'Control')
plt.errorbar(tdata, cec2_ctrl_concentration, yerr = cec2_ctrl_std_dev)
plt.xlabel('Hours post infection')
plt.ylabel('Cecropin-2 expression')
plt.xticks([0,6,12,18,24,30,36,42,48])
plt.xlim(-2, 50)
plt.yticks([0,2,4,6,8,10])
plt.ylim(-1,11)
plt.legend()
plt.show()

#%%

# Fitting all parameters to control data

# degradation/inactivation rates
d_A = 0.402               # 0.02 (delta_A) per Ellner Fig.3,4 /// 0.05 per Table S3
d_G = 0.053                   # 0.02 (delta_P) per Ellner Fig.4
d_P = 1.39                  # 0.01 (delta_P) per Ellner Table S3
d_bP = 0.134                     # 0.02 (delta_P*) per Ellner Table S3
d_bI = 0.132                          # 0.01 (rho_I*) per Ellner Table S3
d_R = 0.006                       # 0.01 (delta_R) per Ellner Table S3
d_N = 0.48            # 0.01 (delta_N) per Ellner Table S3
d_NG1 = 0.05                    # 0.05 (delta_H) per Ellner Table S3
d_NG2 = 0.05                      # 0.05 (delta_H) per Ellner Table S3


# supply rates
s_A = 0.561                       # (Q_A = delta_A) per Ellner p.15
s_P = d_P                       # (Q_P = delta_P) per Ellner p.15
s_B = d_R                         # (Q_R = delta_R) per Ellner p.15
s_NG1 = d_NG1                   # (Q_H = delta_H) per Ellner p.15
s_NG2 = d_NG1                     # (Q_H = delta_H) per Ellner p.15
    
# reaction/binding rates
c_A = 13.1     # (default = 0.8) this parameter adjusts how high A peaks
c_P = 0.372               # (new default = 0.5)
c_I = 0.356                # (default = 0.5)
c_B = 3.97                 # (default = 4) can be higher than 4 but not lower than 2
c_N = 4.1        # (default = 3) decreasing to 1 lowers peak of A and induces small lift around t=48, increasing beyond 3 doesn't change much
c_bN = 6.39    # (default = 0.5) variations alter shape and peak of A, but not by much
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

tdata = np.array([0, 2, 4, 6, 8, 10, 12, 24, 36, 48])

ctrl_concentration = cec2_ctrl_concentration
ctrl_std_dev = cec2_ctrl_std_dev


# Define our ODE model
t, A, G, P, bP, bI, RB, RP, RN, bRN, NG1, NG2 = variables('t, A, G, P, bP, bI, RB, RP, RN, bRN, NG1, NG2')

#d_A = Parameter('d_A', 0.5, 0.001, 100)  
#d_G = Parameter('d_G', 0.02, 0.001, 100)
#d_P = Parameter('d_P', 0.01, 0.001, 100)
#d_bP = Parameter('d_bP', 0.02, 0.001, 100)
#d_bI = Parameter('d_bI', 0.01, 0.01, 100)
#d_R = Parameter('d_R', 0.01, 0.001, 1)
#d_N = Parameter('d_N', 0.15, 0.01, 2)
#d_NG1 = Parameter('d_NG1', 0.05, 0.001, 100)
#d_NG2 = Parameter('d_NG2', 0.05, 0.001, 100)

s_A = Parameter('s_A', 0.5, 0.4, 3)

s_B = Parameter('s_B', 0.01, 0.0001, 0.1)


#c_A = Parameter('c_A', 0.8, 0.1, 20)
#c_P = Parameter('c_P', 0.5, 0.01, 100)
#c_I = Parameter('c_I', 0.5, 0.01, 100)
#c_B = Parameter('c_B', 4, 0.1, 20)
#c_N = Parameter('c_N', 3, 1, 100)
#c_bN = Parameter('c_bN', 1, 0.1, 10)
#c_NG1 = Parameter('c_NG1', 0.1, 0.01, 100)
#c_NG2 = Parameter('c_NG2', 0.2, 0.01, 100)

model_dict = {
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
    }

ctrl_model = ODEModel(model_dict, initial={t: tdata[0], A: ctrl_concentration[0], G: 1, P: 1, bP: 0, bI: 0, RB: 1, RP: 0, RN: 0, bRN: 0, NG1: 1, NG2: 1})

taxis = np.linspace(0, 50)

ctrl_fit = Fit(ctrl_model, t=tdata, A=ctrl_concentration, G=None, P=None, bP=None, bI=None, RB=None, RP=None, RN=None,
          bRN=None, NG1=None, NG2=None, sigma_A=ctrl_std_dev)
ctrl_fit_result = ctrl_fit.execute()

print("\nResults for Cec2 control data:")
print(ctrl_fit_result)

A_ctrl_fit, G_ctrl_fit, NG1_ctrl_fit, NG2_ctrl_fit, P_ctrl_fit, RB_ctrl_fit,  RN_ctrl_fit, RP_ctrl_fit, bI_ctrl_fit, bP_ctrl_fit, bRN_ctrl_fit,  = ctrl_model(t=taxis, **ctrl_fit_result.params)

plt.scatter(tdata, ctrl_concentration, label = 'Control data')
plt.plot(taxis, A_ctrl_fit, label='Control fit')

#plt.plot([0, magnitude_init], [time_max, magnitude_max], color = 'green', linestyle = '-')
#plt.annotate("Initial value = " + str(magnitude_init), (0, A_fit[0]))
#plt.annotate("Max value = " + str(magnitude_max), (time_max, magnitude_max))

plt.xlabel('Hours post infection')
plt.ylabel('Cecropin-2 expression')
plt.xticks([0,6,12,18,24,30,36,42,48])
plt.xlim(-2, 50)
plt.yticks([0,2,4,6,8,10])
plt.ylim(-1,11)
plt.legend()
plt.show()

#cec2_ctrl_c_A = ctrl_fit_result.value(c_A)
#cec2_ctrl_c_N = ctrl_fit_result.value(c_N)
# = ctrl_fit_result.value(c_bN)
#cec2_ctrl_d_A = ctrl_fit_result.value(d_A)
#cec2_ctrl_d_N = ctrl_fit_result.value(d_N)


#%%

# Fitting one parameter at a time to knockdown data

# degradation/inactivation rates
d_A = 0.402               # 0.02 (delta_A) per Ellner Fig.3,4 /// 0.05 per Table S3
d_G = 0.053                   # 0.02 (delta_P) per Ellner Fig.4
d_P = 1.39                  # 0.01 (delta_P) per Ellner Table S3
d_bP = 0.134                     # 0.02 (delta_P*) per Ellner Table S3
d_bI = 0.132                          # 0.01 (rho_I*) per Ellner Table S3
d_R = 0.006                       # 0.01 (delta_R) per Ellner Table S3
d_N = 0.48            # 0.01 (delta_N) per Ellner Table S3
d_NG1 = 0.05                    # 0.05 (delta_H) per Ellner Table S3
d_NG2 = 0.05                      # 0.05 (delta_H) per Ellner Table S3


# supply rates
s_A = 0.561                       # (Q_A = delta_A) per Ellner p.15
s_P = d_P                       # (Q_P = delta_P) per Ellner p.15
s_B = d_R                         # (Q_R = delta_R) per Ellner p.15
s_NG1 = d_NG1                   # (Q_H = delta_H) per Ellner p.15
s_NG2 = d_NG1                     # (Q_H = delta_H) per Ellner p.15
    
# reaction/binding rates
c_A = 13.1     # (default = 0.8) this parameter adjusts how high A peaks
c_P = 0.372               # (new default = 0.5)
c_I = 0.356                # (default = 0.5)
c_B = 3.97                 # (default = 4) can be higher than 4 but not lower than 2
c_N = 4.1        # (default = 3) decreasing to 1 lowers peak of A and induces small lift around t=48, increasing beyond 3 doesn't change much
c_bN = 6.39    # (default = 0.5) variations alter shape and peak of A, but not by much
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

tdata = np.array([0, 2, 4, 6, 8, 10, 12, 24, 36, 48])

kd_concentration = cec2_kd_concentration
kd_std_dev = cec2_kd_std_dev


# Define our ODE model
t, A, G, P, bP, bI, RB, RP, RN, bRN, NG1, NG2 = variables('t, A, G, P, bP, bI, RB, RP, RN, bRN, NG1, NG2')

#d_A = Parameter('d_A', 0.402, 0.001, 100)  
#d_G = Parameter('d_G', 0.02, 0.001, 100)
#d_P = Parameter('d_P', 0.01, 0.001, 100)
#d_bP = Parameter('d_bP', 0.02, 0.001, 100)
#d_bI = Parameter('d_bI', 0.01, 0.01, 100)
#d_R = Parameter('d_R', 0.01, 0.001, 1)
#d_N = Parameter('d_N', 0.15, 0.01, 2)
#d_NG1 = Parameter('d_NG1', 0.05, 0.001, 100)
#d_NG2 = Parameter('d_NG2', 0.05, 0.001, 100)

#s_A = Parameter('s_A', 0.5, 0.4, 3)

#s_B = Parameter('s_B', 0.01, 0.0001, 0.1)


c_A = Parameter('c_A', 13.1, 0.1, 20)
#c_P = Parameter('c_P', 0.5, 0.01, 100)
#c_I = Parameter('c_I', 0.5, 0.01, 100)
#c_B = Parameter('c_B', 4, 0.1, 20)
#c_N = Parameter('c_N', 3, 1, 100)
#c_bN = Parameter('c_bN', 1, 0.1, 10)
#c_NG1 = Parameter('c_NG1', 0.1, 0.01, 100)
#c_NG2 = Parameter('c_NG2', 0.2, 0.01, 100)

model_dict = {
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
    }

kd_model = ODEModel(model_dict, initial={t: tdata[0], A: kd_concentration[0], G: 1, P: 1, bP: 0, bI: 0, RB: 1, RP: 0, RN: 0, bRN: 0, NG1: 1, NG2: 1})

taxis = np.linspace(0, 50)

kd_fit = Fit(kd_model, t=tdata, A=kd_concentration, G=None, P=None, bP=None, bI=None, RB=None, RP=None, RN=None,
          bRN=None, NG1=None, NG2=None, sigma_A=kd_std_dev)
kd_fit_result = kd_fit.execute()

print("\nResults for Cec2 knockdown data:")
print(kd_fit_result)

A_kd_fit, G_kd_fit, NG1_kd_fit, NG2_kd_fit, P_kd_fit, RB_kd_fit, RN_kd_fit, RP_kd_fit, bI_kd_fit, bP_kd_fit, bRN_kd_fit,  = kd_model(t=taxis, **kd_fit_result.params)

plt.scatter(tdata, kd_concentration, label = 'Knockdown data')
plt.plot(taxis, A_kd_fit, label='Knockdown fit')

#plt.plot([0, magnitude_init], [time_max, magnitude_max], color = 'green', linestyle = '-')
#plt.annotate("Initial value = " + str(magnitude_init), (0, A_fit[0]))
#plt.annotate("Max value = " + str(magnitude_max), (time_max, magnitude_max))

plt.xlabel('Hours post infection')
plt.ylabel('Cecropin-2 expression')
plt.xticks([0,6,12,18,24,30,36,42,48])
plt.xlim(-2, 50)
plt.yticks([0,2,4,6,8,10])
plt.ylim(-1,11)
plt.legend()
plt.show()

#cec2_ctrl_c_A = ctrl_fit_result.value(c_A)
#cec2_ctrl_c_N = ctrl_fit_result.value(c_N)
# = ctrl_fit_result.value(c_bN)
#cec2_ctrl_d_A = ctrl_fit_result.value(d_A)
#cec2_ctrl_d_N = ctrl_fit_result.value(d_N)



