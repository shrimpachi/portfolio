# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 14:25:10 2022

@author: Vegt0103
"""

import os
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import signal
import pandas as pd
import math as m

path = os.path.dirname(os.path.realpath(__file__))

import sys
sys.path.append("Students")
from func_import_data import file_to_pandas
from harmfit import harmfit
# import rmse_calc

# Choose which station you want to use and set filename here
filename = 'cape_disappointment-9440581-usa-noaa'
file = path + '/' + filename # Change if your file containing the data is not in the same directory as this Python script.

data_orig = file_to_pandas(file)
# Where 'use_flag' = 0; don't use data, so 'sea_level' is set to NaN.
data_orig.loc[(data_orig['use_flag'] == 0), 'sea_level'] = np.nan

# Select start and end dates for the year you want to analyse
time_start = '2020-01-01 00:00:00'
time_end   = '2021-01-01 00:00:00'
data = data_orig.loc[time_start:time_end]

z_mean = np.nanmean(data.sea_level)

r = pd.date_range(start=data.index.min(), end=data.index.max(), freq='H')
data=data.set_index(data.index).reindex(r).fillna(z_mean)
print(len(data))

ssh = data.sea_level
time = data.index
time = time.to_pydatetime()

#%%
# Show all data for the chosen year
plt.figure()
plt.plot(time, ssh)
plt.ylim([np.floor(np.min(ssh)), np.ceil(np.max(ssh))])
plt.xlabel('Date')
plt.ylabel('Sea level (m)')
plt.title('Sea level fluctuations')# for {}'.format(time[:].year[0]))
plt.show()


M4  = 6.210300601
M6  = 4.140200401
MK3 = 8.177140247
S4  = 6
MN4 = 6.269173724
S6  = 4
M3  = 8.280400802
M8  = 3.105150301
MS4 = 6.103339275

# Semi-Diurnal
M2 = 12.4206012
S2 = 12
N2 = 12.65834751
K2 = 11.9673
V2 = 12.62600509
MU2 = 12.8717576
N22 = 12.90537297
Lam2 = 12.22177348
T2 = 12.01644934
R2 = 11.98359564
L2 = 12.19162085

# Diurnal
K1 = 23.93447213
O1 = 25.81933871
P1 = 24.06588766
Q1 = 24.06588766
S1 = 24
J1 = 23.09848146
pho = 26.72305326
Q12 = 28.00621204

# Long
SA = 8766.15265
MSF = 354.3670666
MF = 327.8599387
MM = 661.3111655
SSA = 4383.076325

# Make lists of constituent names and angular frequencies
const_keys = ["M4","M6","MK3","S4","MN4","S6","M3","M8","MS4","M2","S2","N2","K2","V2","MU2","N22","Lam2","T2","R2","L2","K1","O1","P1","Q1","S1","J1","pho","Q12","SA","MSF","MF","MM","SSA"];
const_values = [M4,M6,MK3,S4,MN4,S6,M3,M8,MS4,M2,S2,N2,K2,V2,MU2,N22,Lam2,T2,R2,L2,K1,O1,P1,Q1,S1,J1,pho,Q12,SA,MSF,MF,MM,SSA]
const_values = np.array(const_values)
const_w = 2 * np.pi/const_values

# Save in Dictionary
const = {const_keys[0]: const_w[0]}
for i in range(len(const_keys)):
    const[const_keys[i]] = const_w[i]
    

wn=const_w


a0 = np.zeros(1)
Cn = np.ones(len(wn))
Dn = np.ones(len(wn))

guess = np.concatenate((a0, Cn, Dn))

timesteps = range(len(ssh))

indep = {}  # independent variables (not to be fitted)
indep["timesteps"] = timesteps
indep["wn"] = wn

popt, pcov = optimize.curve_fit(harmfit, indep, ssh, guess)

#%%
ssh_fit = harmfit(indep, popt)
#%%
RMSE = np.sqrt(np.mean((ssh-ssh_fit)**2)) # Root Mean Squared Error
print('The RMSE has a value of ', RMSE)

plt.figure()
plt.plot(time, ssh, label = 'Observed')
plt.plot(time, ssh_fit, label = 'Fitted')
plt.ylabel('Sea surface height (m)')
plt.xlabel('Date')
plt.title('Sea surface height , both observed and fitted')
plt.show()
