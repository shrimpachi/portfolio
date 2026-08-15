import numpy as np
import matplotlib.pyplot as plt
import math as m
import scipy
from sympy.solvers import solve
from sympy import Symbol
import os


# #%% functions
def contour_length(num_cc, num_cn):
	L_cc = 153.51 #pm
	theta_cc = np.deg2rad(109.5) #rad
	L_cn = 143.00 #pm
	theta_cn = np.deg2rad(120) #rad
	cc = num_cc * L_cc*np.sin(theta_cc/2)
	cn = num_cn * L_cn*np.sin(theta_cn/2)
	L = cc + cn #contour length of one 'monomer'
	return L


def molecular_weight(MW_backbone, sidechains_amount, n_monomer, grafting_ratio, MW_sidechain):
	MW_chains = 0
	for num in range(sidechains_amount):
		num_chains = n_monomer/grafting_ratio[num]
		MW_chains += num_chains*MW_sidechain[num]

	MW_polymer = MW_backbone + MW_chains
	return MW_polymer


def number_ratio(mix_rat, c_tot, MW_polymer, MW_polymerN3):
	num_rat = (mix_rat*c_tot) / ( ( (1-mix_rat)*c_tot*(MW_polymerN3/MW_polymer) ) + (mix_rat*c_tot))
	return num_rat


def density(number_ratio, A_polymer, A_polymerN3):
	rho = number_ratio / ( ((1-number_ratio)*A_polymer) + (number_ratio*A_polymerN3) )
	return rho


def first_order(x, a, b):
	# output = x*np.log(a) + b
	output = a*x + b
	return output

def find_value(lowerbound, upperbound, array):
	threshold_array = []
	for i in range(len(array)):
		if array[i] > lowerbound:
			if array[i] < upperbound:
				threshold_array.append(array[i])
	return threshold_array



# #%% PLL-g-PEG(-N3)
# c_tot = 0.5 #mg/mL
# mix_rat = 0.01 #PEG-N3/PEG
# num_cc = 1 #of one monomer lysine
# num_cn = 2 #of one monomer lysine
# MW_PLL = 24700 #Da // g/mol
# MW_PLLN3 = 15000 #Da
# MW_Lys = 146.19 #Da
# MW_PEG = 2164 #Da
# MW_PEG_N3 = 2000 #Da
# n_monomer_LysN3 = MW_PLLN3/MW_Lys
# grafting_rat_PEG = [3.5] #numLys/numPEG
# grafting_rat_PEG_N3 = [5] #numLys/numPEG

# #%% New PLL-g-PEG stock; spec sheet from teams
# MW_PLL = 21300
# MW_PEG = 2020
# grafting_rat_PEG = [3.47]

# MW_sides = [MW_PEG]
# MW_sidesN3 = [MW_PEG_N3]
# n_monomer_Lys = MW_PLL/MW_Lys

# l_PLL = contour_length(num_cc, num_cn)*n_monomer_Lys
# l_PLLN3 = contour_length(num_cc, num_cn)*n_monomer_LysN3
# MW_PLLPEG = molecular_weight(MW_PLL, 1, n_monomer_Lys, grafting_rat_PEG, MW_sides)
# MW_PLLPEGN3 = molecular_weight(MW_PLLN3, 1, n_monomer_LysN3, grafting_rat_PEG_N3, MW_sidesN3)
# num_rat_PP = number_ratio(mix_rat, c_tot, MW_PLLPEG, MW_PLLPEGN3)
# A_PLLPEG = l_PLL*2e3
# A_PLLPEGN3 = l_PLLN3*2e3
# density_PP = density(num_rat_PP, A_PLLPEG, A_PLLPEGN3)
# # print(MW_PLLPEG, MW_PLLPEGN3)
# print(f"Binder density for PLL-g-PEG is: {density_PP*(1e6)**2} um-2")

#%% PAcrAm-g-PMOXA(-N3) // Data obtained from Iris' spec sheet
mix_rat = 0.01
c_tot = 0.1 #mg/mL
num_cc = 2
num_cn = 0
n_monomer_AcrAm = 95 #80 to 110 repeat units
MW_PAcrAmMono = 71.08
MW_PAcrAm = MW_PAcrAmMono * n_monomer_AcrAm #Total MW of backbone
MW_PMOXA = 4048 #g/mol :: spec sheet Iris
MW_PMOXA_N3 = 6100 #g/mol :: spec sheet Iris
MW_Si = 161.32
MW_NH2 = 116.20
grafting_rat_MOXA = 5
grafting_rat_sidechain = 2.5

MW_sides = [MW_PMOXA, MW_Si, MW_NH2]
MW_sidesN3 = [MW_PMOXA_N3, MW_Si, MW_NH2]
grafting_ratios = [grafting_rat_MOXA, grafting_rat_sidechain, grafting_rat_sidechain]

l_AcrAm = contour_length(num_cc, num_cn)*n_monomer_AcrAm #pm; same length for PMOXA as PMOXA-N3
MW_PAcrAmPMOXA = molecular_weight(MW_PAcrAm, 3, n_monomer_AcrAm, grafting_ratios, MW_sides) #g/mol
MW_PAcrAmPMOXAN3 = molecular_weight(MW_PAcrAm, 3, n_monomer_AcrAm, grafting_ratios, MW_sidesN3) #g/mol
num_rat_PaS = number_ratio(mix_rat, c_tot, MW_PAcrAmPMOXA, MW_PAcrAmPMOXAN3) 
A_PAcramPMOXA = l_AcrAm*2e3 #pm^2
A_PAcramPMOXAN3 = A_PAcramPMOXA #pm^2
density_PaS = density(num_rat_PaS, A_PAcramPMOXA, A_PAcramPMOXAN3) # num/pm^2
n_MOXAN3 = n_monomer_AcrAm/grafting_rat_MOXA #no of MOXA-N3 on polymer (16 - 19 - 22 :: 19+/-3)
d_MOXA = (l_AcrAm/n_MOXAN3)*1e-3 #nm; distance between ssDNA binder strands

if d_MOXA < 2:
	bind_dens = m.floor(n_MOXAN3/2)
else:
	bind_dens = n_MOXAN3
# bind_dens = n_MOXAN3
PaS_binddens = (density_PaS*(1e6)**2)*bind_dens #num of binders per um2

print(n_MOXAN3)
print(f"Backbone length is: {l_AcrAm}")
print(f"Total molecular weight is: {MW_PAcrAmPMOXA} and {MW_PAcrAmPMOXAN3}")
print(f"Area is {l_AcrAm*1e-3*2}")
print(f"Binder density for PAcrAm-g-PMOXA is: {PaS_binddens} um-2")
print(f"Cloud density for PAcrAm-g-PMOXA is: {(density_PaS*(1e6)**2)} um-2")
print(f"Num of available binders per polymer: {PaS_binddens/(density_PaS*(1e6)**2)}")


E = 1 / (2*((density_PaS*(1e3)**2)**0.5)) #distance between two polymers in nm assuming perfectly distributed
print(f"Distance between two polymer clouds: {E} nm")









