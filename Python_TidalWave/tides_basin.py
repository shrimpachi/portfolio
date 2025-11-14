#%% import packages
import os
import numpy as np
import netCDF4 as nc
import datetime
import matplotlib.pyplot as plt
from scipy import optimize
import math

from harmfit_new import harmfit

# Solving the shallow water equations in shallow basins and estuaries.
# Based on Friedrichs (2011), Chapter 3 in Contemporary Issues in Estuarine Physics.


#%%               Parameter settings

deltaT = 30               # time step in seconds. Choose appropriate time step yourself based on Courant number. 
deltaX = 1000             # spatial step in meters
Lbasin = 20e3            # Length of the basin or estuary in meters
Lb     = 4e4              # e-folding length scale for width.
B0     = 1e3              # Width of the basin in meters at seaward side.
H0     = 5                # Depth of basin.
M2amp  = 1                # Amplitude of M2 tide at seaward side.
discharge = 0             # Constant river discharge at landward boundary. 
Cd     = 2e-3             # Drag coefficient
# Cd = [0.2e-3, 0.5e-3, 1e-3, 2e-3, 5e-3]
H0     = np.arange(2,11,1)  # Still water depth 
# Lbasin = np.arange(10e3, 150e3, 20e3)
# Lbasin = np.linspace(40e3, 110e3, 2)

# #%% Values Question D
# B0 = 1.6e3 #meters at x=0
# Briver = 400 #meters at x=xr
# Lbasin = 3e3 #length estuary
# Lb = -Lbasin / (np.log(Briver/B0)) #e-folding lengthscale
# Cd = 2.5e-3
# discharge = 0
# H0 = 5 #meters


# deltaX = 30
# deltaT = 4 #determine using CFL
# x_range = np.arange(0, Lbasin*1.5, deltaX) #Take in account river part



# B[pos] = B0*exp(-pos/Lbasin)


#%%
Td1  = 24*3600+50*60
Tm2  = 12*3600+25*60                # M2 tidal period in seconds
time = np.arange(0, 15*Tm2+deltaT, deltaT) # time in seconds
Nt   = len(time)                     # number of timesteps

# Define frequencies to be analysed. 
# To determine the amplitudes and phase use the code you designed in the first Matlab exercise. 
#NOTE: wn zo goed??
wn = np.zeros(3)
wn[0] = 2*np.pi/Tm2 #M2 constituent
wn[1] = 2*wn[0] #M4 constituent
wn[2] = 3*wn[0] #M6 constituent

for basinn in range(len(H0)):
    x = np.arange(0, Lbasin+deltaX, deltaX)    # x in meters
    Nx = len(x)                                # number of horizontal locations

    # B = B0 * np.exp(-x/Lb)   # width [m]
    B = np.ones(Nx)*B0;        # when basin width has to be constant.
    H = np.ones(Nx)*H0[basinn];        # depth [m]

    Z = np.zeros((Nx-1,Nt)) # Z points shifted half a grid point to the right with respect to Q points. we start with Q point, therefore 1 Z point less than Q points.      
    Q = np.zeros((Nx,Nt))
    B_H = np.array(B*H)
    A = np.matmul(B_H.reshape(len(B_H),1), np.ones((1,Nt)))      # A at Q points 
    P = B.reshape(len(B),1) * np.ones((1,Nt))                    # Wetted perimeter at Q points.

    # Initalize Inertia, Pressure Gradient and Friction for further analysis later on.
    Inertia = np.zeros((Nx,Nt)) 
    PG      = np.zeros((Nx,Nt))
    Fric    = np.zeros((Nx,Nt))

    # Boundary conditions
    Z[0,:] = M2amp * np.sin(2*np.pi*time/Tm2)    # prescribed water levels. This is M2 tide.
    Q[Nx-1,:] = -discharge                       # river discharge; most often river discharge is taken zero.

    if isinstance(H, float):
        courant = np.sqrt(9.8*H)*deltaT/deltaX
    else:
        courant = np.sqrt(9.8*max(H))*deltaT/deltaX

    # print(Z[0, 0])
    #%% 
    # For numerical part, follow thesis of Speer (1984). Staggered grid. 
    # Z points shifted half deltaX to the right of U points: 
    # Q1 Z1 Q2 Z2 ........ ZN QN+1

    # solve Bdz/dt + dQ/dx=0
    # solve dQ/dt + d/dx Q^2/A = -gA dZ/dx - Cd Q|Q|P/A*A

    # Z is water level, B=estuary width, Q is discharge, A is cross-sectional area, 
    # P is wetted perimeter (~channel width when H/B is small)

    # First start simple: rectangular basin, no advection, no river flow.
    # Numerical scheme from Speer (1984).
    

    for pt in range(Nt):
        for px in range(2, Nx):
            Z[px-1,pt] = Z[px-1,pt-1] - (deltaT/(0.5*(B[px-1]+B[px])))*(Q[px,pt-1]-Q[px-1,pt-1])/deltaX
            A[px-1,pt] = B[px-1] * (H[px] + 0.5*Z[px-1,pt] + 0.5*Z[px-2,pt])           # A at Q points
            P[px-1,pt] = B[px-1] + 2*H[px-1] + Z[px-1,pt] + Z[px-2,pt]                 # P at Q points
        
            Inertia[px-1,pt]=(Q[px-1,pt]-Q[px-1,pt-1])/deltaT
            PG[px-1,pt]=-9.81*A[px-1,pt]*(1/deltaX)*(Z[px-1,pt]-Z[px-2,pt])
            Fric[px-1,pt]=-Cd*abs(Q[px-1,pt-1])*Q[px-1,pt-1]*P[px-1,pt-1]/(A[px-1,pt-1]*A[px-1,pt-1])

            Q[px-1,pt]= (Q[px-1,pt-1]                                           # Inertia
                - 9.81*A[px-1,pt]*(deltaT/deltaX)*(Z[px-1,pt]-Z[px-2,pt])       # Pressure gradient
                - Cd*deltaT*abs(Q[px-1,pt-1])*Q[px-1,pt-1]*P[px-1,pt-1]/(A[px-1,pt-1]*A[px-1,pt-1]))    # Friction
            
        Q[0,pt]=Q[1,pt]+B[0]*deltaX*(Z[0,pt]-Z[0,pt-1])/deltaT

    U=Q/A         # Flow velocity in m/s

        #%% Analysis
    # Analyse last tidal period only. For example determine amplitude and phase of M2, M4, M6 and mean
    # of water level and flow velocity. Design you own code here. I used my code (harmfit). You
    # can determine HW level, LW level, moments of LW and HW, propagation speed of LW wave
    # and HW wave... You can determine propagation speed of wave by determining
    # the phase as a function of space. The phase (in radians) also equals k*x, so a linear
    # fit to the phase determines wave number k. 



    M2 = wn[0]
    M4 = wn[1]
    M6 = wn[2]
    # D1 = 24 + (50/60)
    # D2 = 12.41666667
    # D4 = D2/2
    # D6 = D2/3

    const_keys = ["M2", "M4", "M6"]
    const_values = [M2, M4, M6]
    const_values = np.array(const_values)
    # const_w = 2 * np.pi/const_values
    wn_cal = const_values

    const = {const_keys[0]: const_values[0]}
    for i in range(len(const_keys)):
        const[const_keys[i]] = const_values[i]

    wn_values = wn_cal
    length_wn = len(wn_cal)

    #%% Fit the harmonic function (harmfit)

    Cn = (np.ones(len(wn_values)))
    Dn = (np.ones(len(wn_values)))
    a0 = np.ones(1)
    waterlevel = np.array(Z)
    # print(Lbasin[i])
    newx = np.arange(0, Lbasin, deltaX)
    # plt.plot(newx, waterlevel[:, 9*Tm2//deltaT])
    # newtime = np.arange(9*Tm2//deltaT, 10*Tm2//deltaT, deltaT)
    Nsteps = math.floor(Tm2/deltaT)
    # plt.plot(newx, waterlevel[:, 300000//deltaT])
    # print("The mean flow velocity is:", np.mean(U[int(-Nsteps):]))
    
    # timerange = np.arange()
    # Q_last = 0
    # for q in range(len(timerange)):
    #     Q_last += Q[0][q] * deltaT

    # TP = 0.5 * Q_last
    # print(TP)

    indep = {"timesteps": time[int(-Nsteps):],
            "wn": wn_values}
    # make for loop -> take last sinus at each x position. sinus changes over time
    # thus time step also different. pls change :)


    last_sinus = [] 
    coefin = np.concatenate((a0, Cn, Dn))


    # for i in range(0,Z.shape[0], 5):
    #     last_sinus.append(np.array(Z[i, newtime]))

    phase_diff = []
    a0 = []
    amp_m2 = []
    amp_m4 = []
    u0 = []
    phase_m2 = []
    length_basin = np.arange(0, Lbasin, deltaX)

    for i in range(len(length_basin)):
        # plt.plot(newtime, last_sinus[i])
        popt_water, pcov_water = optimize.curve_fit(harmfit, indep, waterlevel[i, int(-Nsteps):], coefin)
        popt_vel, pcov_vel = optimize.curve_fit(harmfit, indep, U[i, int(-Nsteps):], coefin)
        a0.append(popt_water[0])
        Cn_m2 = popt_water[1]
        Dn_m2 = popt_water[4]
        amp_m2.append((Cn_m2**2 + Dn_m2**2)**0.5)

        Cn_m4 = popt_vel[2]
        Dn_m4 = popt_vel[5]
        amp_m4.append((Cn_m4**2 + Dn_m4**2)**0.5)

        phase_m2.append(math.atan(Cn_m2/Dn_m2))
        # phase_m4 = math.atan(Cn_m4/Dn_m4)
        # phase_diff.append(2*phase_m2-phase_m4)

        u0.append(popt_vel[0])

    
    # lengtebasin = Lbasin/1000
    plt.plot(length_basin, phase_m2, label="%i meter"%H0[basinn])

    # plt.subplot(1, 2, 1)
    # # plt.plot(length_basin, amp_m2, label='Cd =  %.e' %Cd[basinn])
    # plt.plot(length_basin, amp_m2, label="M2 amp")
    # plt.plot(length_basin, amp_m4, label='M4 amp')
    # plt.plot(length_basin, u0, label="u0")
    # plt.title("Amplitude M2/M4 flow velocity")
    # plt.xlabel("Position basin [m]")
    # plt.ylabel("Amplitude [m]")
    # plt.legend()

    # plt.subplot(1, 2, 2)
    # plt.plot(length_basin, phase_diff, label='phase diff')
    # plt.title("Relative phase difference") 
    # plt.xlabel("Position basin [m]")
    # plt.legend()

    # plt.plot(length_basin, np.mean(U[int(-Nstep):]))
    # plt.plot(length_basin, u0)
    # print(len(Z), len(newpos), len(x))
    # print(waterlevel[:][0])
    

# print('phase differences between M2 and M4',phase_diff)
# print('mean surface height a0 =', a0)






# plt.title("Amplitude over basin position")
# plt.ylabel('Amplitude [m]')
# plt.xlabel('Position in basin [m]')
# plt.legend()
# plt.savefig('depth %i m.png'%H0)
plt.xlabel("Position in basin [m]")
plt.ylabel("M2 tide phase [rad]")
plt.title("M2 tide phase for different depths")
plt.legend()
plt.show()


# print(np.shape(Z))
# for last_wave in range(9*Tm2//deltaT, 10*Tm2//deltaT, deltaT):
#     last_sinus.append(Z[:, last_wave])

# print(len(last_sinus))

# coefin = np.concatenate((a0, Cn, Dn))
# popt, pcov = optimize.curve_fit(harmfit, indep, last_sinus, coefin)
# print(popt)
# fittedwaterlevel = harmfit(indep, popt)


# plt.plot(newtime, fittedwaterlevel)
# plt.plot(newtime, last_sinus)
# plt.show()
#%% Use optimized amplitudes and phases to make the best fit function


# Nsteps = math.floor(Tm2/deltaT)


# # Create arrays to store output of analysis in.
# Z0  = np.zeros(px)
# ZM2 = np.zeros(px)
# ZM4 = np.zeros(px)
# ZM6 = np.zeros(px)
# phaseZM2 = np.zeros(px)
# phaseZM4 = np.zeros(px)
# phaseZM6 = np.zeros(px)
# U0  = np.zeros(px)
# UM2 = np.zeros(px)
# UM4 = np.zeros(px)
# UM6 = np.zeros(px)
# phaseUM2 = np.zeros(px)
# phaseUM4 = np.zeros(px)
# phaseUM6 = np.zeros(px)

# for px in range(Nx):
# Do harmonic analysis of flow velocities and water levels for each position 



