from scipy.optimize import curve_fit
from matplotlib.pyplot import figure
from math import log10, floor

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import os

# Find Minimum of Graph
def find_minimum(neff_fit_dict, annealing_fit_dict, diode_name):
    
    min_neff = min(neff_fit_dict[diode_name])

    for i in range(0,len(neff_fit_dict[diode_name])):
        
        if min_neff == neff_fit_dict[diode_name][i]:
            
            min_anneal_time = annealing_fit_dict[diode_name][i]
            
    
    return min_neff, min_anneal_time

# Is NaN Function
def isNaN(num):
    return num != num

# Significant Figures Function
def round_sig(x, sig=3, small_value=1.0e-9):
    if isNaN(x) == True:
        return x
    else:
        return round(x, sig - int(floor(log10(max(abs(x), abs(small_value))))) - 1)

# Current Working Directory
cwd = os.getcwd()

# List all of the Files in the Current Working Directory 
search_path = '.'   # set your path here.
root, dirs, files = next(os.walk(search_path), ([],[],[]))
file_size = len(files)

# Name the Different Dictionaries That Will Contain the Relevent Information for the Analysis
neff_dict = {}
df_dict = {}
annealing_dict = {}
len_dict = {}
annealing_fit_dict = {}
neff_fit_dict = {}
params_dict = {}
covariance_dict = {}

# Fitting Items
fluence = 6.5*10**18

#guess = np.array([ 2.59046272e+00,  2.54614411e+19,  2.59153632e-01,  1.51895447e+01, -4.42517117e+00])
#guess = np.array([ 2.99080953e+00,  1.00000000e+00, -1.37453483e+00,  5.17326207e+09, 1.48335632e+01])
guess_shift = np.array([6.03709873e-01, 2.23775501e+19, 6.12224545e-01, 5.41529930e+00, 4.09569161e+01, 80])

def Hamburg(x, ga, NC, gy, ta, ty):
    return ga*np.exp(-x/ta)*fluence + gy*(1.-1./(1.+x/ty))*fluence + NC

def Hamburg_shift(x, ga, NC, gy, ta, ty, t_s):
    return ga*np.exp(-(x - t_s)/ta)*fluence + gy*(1.-1./(1.+(x - t_s)/ty))*fluence + NC

# Populate the Dictionaries and Fit the Curves
for i in range(0, file_size): 

    diode_name = files[i].strip('.csv') 
    
    if 'DIODE' in diode_name:
 
        df_dict[diode_name] = pd.read_csv(files[i])
        annealing_dict[diode_name] = df_dict[diode_name]['Annealing at 60C (min)']
        neff_dict[diode_name] = df_dict[diode_name]['Neff']
        len_dict[diode_name] = len(df_dict[diode_name]) - 1
        annealing_fit_dict[diode_name] = np.linspace(annealing_dict[diode_name][0], annealing_dict[diode_name][len_dict[diode_name]], 1000)
        
        ## Fitting for HGCAL Diodes W/O Shifting in Time
        params_dict[diode_name], covariance_dict[diode_name] = curve_fit(Hamburg, annealing_dict[diode_name], neff_dict[diode_name])#, guess)
        neff_fit_dict[diode_name] = (params_dict[diode_name][0]*np.exp(-(annealing_fit_dict[diode_name])/params_dict[diode_name][3])*fluence + params_dict[diode_name][2]*(1.-1./(1.+(annealing_fit_dict[diode_name])/params_dict[diode_name][4]))*fluence + params_dict[diode_name][1])
        
        ## Fitting for HGCAL Diodes W/ Shift in Time
        # params_dict[diode_name], covariance_dict[diode_name] = curve_fit(Hamburg_shift, annealing_dict[diode_name], neff_dict[diode_name], guess_shift)
        # neff_fit_dict[diode_name] = (params_dict[diode_name][0]*np.exp(-(annealing_fit_dict[diode_name] -  params_dict[diode_name][5])/params_dict[diode_name][3])*fluence + params_dict[diode_name][2]*(1.-1./(1.+(annealing_fit_dict[diode_name] -  params_dict[diode_name][5])/params_dict[diode_name][4]))*fluence + params_dict[diode_name][1])


#%%
# Ljubljana_Diodes Hamburg

# Get Minimum Values
min_neff_DIODE, min_anneal_time_DIODE =  find_minimum(neff_fit_dict, annealing_fit_dict, 'CO138_DIODE_Neff_vs_Annealing')

# Plot Fits 
figure(figsize=(12, 7), dpi=100)
plt.suptitle('Ljubljana DZero Hamburg Analysis', fontsize=20)

plt.scatter(annealing_dict['CO138_DIODE_Neff_vs_Annealing'], neff_dict['CO138_DIODE_Neff_vs_Annealing'], label = r'DO_138_Large_GR  $N_{eff}$ Data')
plt.plot(annealing_fit_dict['CO138_DIODE_Neff_vs_Annealing'], neff_fit_dict['CO138_DIODE_Neff_vs_Annealing'])
plt.scatter(min_anneal_time_DIODE, min_neff_DIODE, label = 'DO_138_Large_GR Minimum at ' + str(round_sig(min_anneal_time_DIODE)) + ' min')

plt.xlabel('Annealing (min)', fontsize=22)
plt.ylabel(r'$N_{eff}$  $\propto$ $V_{depletion}$', fontsize=22)
plt.legend(fontsize = 14)

plt.savefig('CO138_Neff_vs_Annealing' + '.png', dpi=900)

#%%
# HGCAL Campaign Round 2.2 Front

# Get Minimum Values
min_neff_DIODE, min_anneal_time_DIODE =  find_minimum(neff_fit_dict, annealing_fit_dict, 'N6789_24_LL_DIODE')
min_neff_DIODEQUARTER, min_anneal_time_DIODEQUARTER =  find_minimum(neff_fit_dict, annealing_fit_dict, 'N6789_24_LL_DIODEQUARTER')

# Plot Fits 
figure(figsize=(12, 7), dpi=100)
plt.suptitle('HGCAL Campaign Round 2.2 Front', fontsize=20)

plt.scatter(annealing_dict['N6789_24_LL_DIODE'], neff_dict['N6789_24_LL_DIODE'], label = r'N6789_24_LL_DIODE $N_{eff}$ Data')
plt.scatter(annealing_dict['N6789_24_LL_DIODEQUARTER'], neff_dict['N6789_24_LL_DIODEQUARTER'], label = r'N6789_24_LL_DIODEQUARTER $N_{eff}$ Data')

plt.plot(annealing_fit_dict['N6789_24_LL_DIODE'], neff_fit_dict['N6789_24_LL_DIODE'])
plt.plot(annealing_fit_dict['N6789_24_LL_DIODEQUARTER'], neff_fit_dict['N6789_24_LL_DIODEQUARTER'])

plt.scatter(min_anneal_time_DIODE, min_neff_DIODE, label = 'N6789_24_LL_DIODE Minimum at ' + str(round_sig(min_anneal_time_DIODE)) + ' min')
plt.scatter(min_anneal_time_DIODEQUARTER, min_neff_DIODEQUARTER, label = 'N6789_24_LL_DIODEQUARTER Minimum at ' + str(round_sig(min_anneal_time_DIODEQUARTER))
            + ' min')
plt.xlabel('Annealing (min)', fontsize=22)
plt.ylabel(r'$N_{eff}$  $\propto$ $V_{depletion}$', fontsize=22)
plt.legend(fontsize = 14)

plt.savefig('HGCAL_2.2_Front' + '.png', dpi=900)

#%%
# HGCAL Campaign Round 2.2 Back

# Get Minimum Values
min_neff_DIODE, min_anneal_time_DIODE =  find_minimum(neff_fit_dict, annealing_fit_dict, 'N4789_24_LL_DIODE')
min_neff_DIODEHALF, min_anneal_time_DIODEHALF =  find_minimum(neff_fit_dict, annealing_fit_dict, 'N4789_24_LL_DIODEHALF')
min_neff_DIODEQUARTER, min_anneal_time_DIODEQUARTER =  find_minimum(neff_fit_dict, annealing_fit_dict, 'N4789_24_LL_DIODEQUARTER')

# Plot Fits 
figure(figsize=(12, 7), dpi=100)
plt.suptitle('HGCAL Campaign Round 2.2 Back', fontsize=20)
plt.scatter(annealing_dict['N4789_24_LL_DIODE'], neff_dict['N4789_24_LL_DIODE'], label = r'N4789_24_LL_DIODE $N_{eff}$ Data')
plt.scatter(annealing_dict['N4789_24_LL_DIODEHALF'], neff_dict['N4789_24_LL_DIODEHALF'], label = r'N4789_24_LL_DIODEHALF $N_{eff}$ Data')
plt.scatter(annealing_dict['N4789_24_LL_DIODEQUARTER'], neff_dict['N4789_24_LL_DIODEQUARTER'], label = r'N4789_24_LL_DIODEQUARTER $N_{eff}$ Data')

plt.plot(annealing_fit_dict['N4789_24_LL_DIODE'], neff_fit_dict['N4789_24_LL_DIODE'])
plt.plot(annealing_fit_dict['N4789_24_LL_DIODEHALF'], neff_fit_dict['N4789_24_LL_DIODEHALF'])
plt.plot(annealing_fit_dict['N4789_24_LL_DIODEQUARTER'], neff_fit_dict['N4789_24_LL_DIODEQUARTER'])

plt.scatter(min_anneal_time_DIODE, min_neff_DIODE, label = 'N4789_24_LL_DIODE Minimum at ' + str(round_sig(min_anneal_time_DIODE)) + ' min')
plt.scatter(min_anneal_time_DIODEHALF, min_neff_DIODEHALF, label = 'N4789_24_LL_DIODEHALF Minimum at ' + str(round_sig(min_anneal_time_DIODEHALF)) + ' min')
plt.scatter(min_anneal_time_DIODEQUARTER, min_neff_DIODEQUARTER, label = 'N4789_24_LL_DIODEQUARTER Minimum at ' + str(round_sig(min_anneal_time_DIODEQUARTER)) + ' min')
plt.xlabel('Annealing (min)', fontsize=22)
plt.ylabel(r'$N_{eff}$  $\propto$ $V_{depletion}$', fontsize=22)
plt.legend(fontsize = 14)

plt.savefig('HGCAL_2.2_Back' + '.png', dpi=900)

#%%
# HGCAL Campaign Round 2.3 Front

# Get Minimum Values
min_neff_DIODE, min_anneal_time_DIODE =  find_minimum(neff_fit_dict, annealing_fit_dict, 'N4789_24_UR_DIODE')
min_neff_DIODEHALF, min_anneal_time_DIODEHALF =  find_minimum(neff_fit_dict, annealing_fit_dict, 'N4789_24_UR_DIODEHALF')
min_neff_DIODEQUARTER, min_anneal_time_DIODEQUARTER =  find_minimum(neff_fit_dict, annealing_fit_dict, 'N4789_24_UR_DIODEQUARTER')

# Plot Fits 
figure(figsize=(12, 7), dpi=100)
plt.suptitle('HGCAL Campaign Round 2.3 Front', fontsize=20)

plt.scatter(annealing_dict['N4789_24_UR_DIODE'], neff_dict['N4789_24_UR_DIODE'], label = r'N4789_24_UR_DIODE $N_{eff}$ Data')
plt.scatter(annealing_dict['N4789_24_UR_DIODEHALF'], neff_dict['N4789_24_UR_DIODEHALF'], label = r'N4789_24_UR_DIODEHALF $N_{eff}$ Data')
plt.scatter(annealing_dict['N4789_24_UR_DIODEQUARTER'], neff_dict['N4789_24_UR_DIODEQUARTER'], label = r'N4789_24_UR_DIODEQUARTER $N_{eff}$ Data')

plt.xlabel('Annealing (min)', fontsize=22)
plt.ylabel(r'$N_{eff}$  $\propto$ $V_{depletion}$', fontsize=22)

plt.legend(fontsize = 14)

plt.savefig('HGCAL_2.3_Front' + '.png', dpi=900)

#%%
# HGCAL Campaign Round 2.3 Back

# Get Minimum Values
min_neff_DIODE, min_anneal_time_DIODE =  find_minimum(neff_fit_dict, annealing_fit_dict, 'N4789_20_LL_DIODE')
min_neff_DIODEQUARTER, min_anneal_time_DIODEQUARTER =  find_minimum(neff_fit_dict, annealing_fit_dict, 'N4789_20_LL_DIODEQUARTER')

# Plot Fits 
figure(figsize=(12, 7), dpi=100)
plt.suptitle('HGCAL Campaign Round 2.3 Back', fontsize=20)
plt.scatter(annealing_dict['N4789_20_LL_DIODE'], neff_dict['N4789_20_LL_DIODE'], label = r'N4789_20_LL_DIODE $N_{eff}$ Data')
plt.scatter(annealing_dict['N4789_20_LL_DIODEQUARTER'], neff_dict['N4789_20_LL_DIODEQUARTER'], label = r'N4789_20_LL_DIODEQUARTER $N_{eff}$ Data')

plt.plot(annealing_fit_dict['N4789_20_LL_DIODE'], neff_fit_dict['N4789_20_LL_DIODE'])
plt.plot(annealing_fit_dict['N4789_20_LL_DIODEQUARTER'], neff_fit_dict['N4789_20_LL_DIODEQUARTER'])

plt.scatter(min_anneal_time_DIODE, min_neff_DIODE, label = 'N4789_20_LL_DIODE Minimum at ' + str(round_sig(min_anneal_time_DIODE)) + ' min')
plt.scatter(min_anneal_time_DIODEQUARTER, min_neff_DIODEQUARTER, label = 'N4789_20_LL_DIODEQUARTER Minimum at ' + str(round_sig(min_anneal_time_DIODEQUARTER)) + ' min')
plt.xlabel('Annealing (min)', fontsize=22)
plt.ylabel(r'$N_{eff}$  $\propto$ $V_{depletion}$', fontsize=22)

plt.legend(fontsize = 14)

plt.savefig('HGCAL_2.3_Back' + '.png', dpi=900)


#%%
# HGCAL Campaign Round 2.4 Back

# Plot Fits 
figure(figsize=(12, 7), dpi=100)

plt.suptitle('HGCAL Campaign Round 2.3 Back', fontsize=20)

plt.scatter(annealing_dict['N4789_20_UL_DIODE'], neff_dict['N4789_20_UL_DIODE'], label = r'N4789_20_UL_DIODE $N_{eff}$ Data')
plt.scatter(annealing_dict['N4789_20_UL_DIODEHALF'], neff_dict['N4789_20_UL_DIODEHALF'], label = r'N4789_20_UL_DIODEHALF $N_{eff}$ Data')
plt.scatter(annealing_dict['N4789_20_UL_DIODEQUARTER'], neff_dict['N4789_20_UL_DIODEQUARTER'], label = r'N4789_20_UL_DIODEQUARTER $N_{eff}$ Data')

plt.xlabel('Annealing (min)', fontsize=22)
plt.ylabel(r'$N_{eff}$  $\propto$ $V_{depletion}$', fontsize=22)

plt.legend(fontsize = 14)

plt.savefig('HGCAL_2.4_Back' + '.png', dpi=900)

#%%
# HGCAL Campaign Round 2.5 Front

# Get Minimum Values
min_neff_DIODE, min_anneal_time_DIODE =  find_minimum(neff_fit_dict, annealing_fit_dict, 'N4789_20_UR_DIODE')
min_neff_DIODEHALF, min_anneal_time_DIODEHALF =  find_minimum(neff_fit_dict, annealing_fit_dict, 'N4789_20_UR_DIODEHALF')
min_neff_DIODEQUARTER, min_anneal_time_DIODEQUARTER =  find_minimum(neff_fit_dict, annealing_fit_dict, 'N4789_20_UR_DIODEQUARTER')

# Plot Fits 
figure(figsize=(12, 7), dpi=100)
plt.suptitle('HGCAL Campaign Round 2.5 Front', fontsize=20)

plt.scatter(annealing_dict['N4789_20_UR_DIODE'], neff_dict['N4789_20_UR_DIODE'], label = r'N4789_20_UR_DIODE $N_{eff}$ Data')
plt.scatter(annealing_dict['N4789_20_UR_DIODEHALF'], neff_dict['N4789_20_UR_DIODEHALF'], label = r'N4789_20_UR_DIODEHALF $N_{eff}$ Data')
plt.scatter(annealing_dict['N4789_20_UR_DIODEQUARTER'], neff_dict['N4789_20_UR_DIODEQUARTER'], label = r'N4789_20_LL_DIODEQUARTER $N_{eff}$ Data')

plt.plot(annealing_fit_dict['N4789_20_UR_DIODE'], neff_fit_dict['N4789_20_UR_DIODE'])
plt.plot(annealing_fit_dict['N4789_20_UR_DIODEHALF'], neff_fit_dict['N4789_20_UR_DIODEHALF'])
plt.plot(annealing_fit_dict['N4789_20_UR_DIODEQUARTER'], neff_fit_dict['N4789_20_UR_DIODEQUARTER'])

plt.scatter(min_anneal_time_DIODE, min_neff_DIODE, label = 'N4789_20_UR_DIODE Minimum at ' + str(round_sig(min_anneal_time_DIODE)) + ' min')
plt.scatter(min_anneal_time_DIODEHALF, min_neff_DIODEHALF, label = 'N4789_20_UR_DIODEHALF Minimum at ' + str(round_sig(min_anneal_time_DIODEHALF)) + ' min')
plt.scatter(min_anneal_time_DIODEQUARTER, min_neff_DIODEQUARTER, label = 'N4789_20_LL_DIODEQUARTER Minimum at ' + str(round_sig(min_anneal_time_DIODEQUARTER)) + ' min')
plt.xlabel('Annealing (min)', fontsize=22)
plt.ylabel(r'$N_{eff}$  $\propto$ $V_{depletion}$', fontsize=22)
plt.legend(fontsize = 14)

plt.savefig('HGCAL_2.5_Front' + '.png', dpi=900)

#%%
# HGCAL Campaign Round 2.5 Back

# Get Minimum Values
min_neff_DIODE, min_anneal_time_DIODE =  find_minimum(neff_fit_dict, annealing_fit_dict, 'N4789_20_LL_DIODE')
min_neff_DIODEHALF, min_anneal_time_DIODEHALF =  find_minimum(neff_fit_dict, annealing_fit_dict, 'N4789_20_LL_DIODEHALF')
min_neff_DIODEQUARTER, min_anneal_time_DIODEQUARTER =  find_minimum(neff_fit_dict, annealing_fit_dict, 'N4789_20_LL_DIODEQUARTER')

# Plot Fits 
figure(figsize=(12, 7), dpi=100)
plt.suptitle('HGCAL Campaign Round 2.5 Back', fontsize=20)
plt.scatter(annealing_dict['N4789_20_LL_DIODE'], neff_dict['N4789_20_LL_DIODE'], label = r'N4789_20_LL_DIODE $N_{eff}$ Data')
plt.scatter(annealing_dict['N4789_20_LL_DIODEHALF'], neff_dict['N4789_20_LL_DIODEHALF'], label = r'N4789_20_LL_DIODEHALF $N_{eff}$ Data')
plt.scatter(annealing_dict['N4789_20_LL_DIODEQUARTER'], neff_dict['N4789_20_LL_DIODEQUARTER'], label = r'N4789_20_LL_DIODEQUARTER $N_{eff}$ Data')
plt.plot(annealing_fit_dict['N4789_20_LL_DIODE'], neff_fit_dict['N4789_20_LL_DIODE'])
plt.plot(annealing_fit_dict['N4789_20_LL_DIODEHALF'], neff_fit_dict['N4789_20_LL_DIODEHALF'])
plt.plot(annealing_fit_dict['N4789_20_LL_DIODEQUARTER'], neff_fit_dict['N4789_20_LL_DIODEQUARTER'])
plt.scatter(min_anneal_time_DIODE, min_neff_DIODE, label = 'N4789_20_LL_DIODE Minimum at ' + str(round_sig(min_anneal_time_DIODE)) + ' min')
plt.scatter(min_anneal_time_DIODEHALF, min_neff_DIODEHALF, label = 'N4789_20_LL_DIODEHALF Minimum at ' + str(round_sig(min_anneal_time_DIODEHALF)) + ' min')
plt.scatter(min_anneal_time_DIODEQUARTER, min_neff_DIODEQUARTER, label = 'N4789_20_LL_DIODEQUARTER Minimum at ' + str(round_sig(min_anneal_time_DIODEQUARTER)) + ' min')
plt.xlabel('Annealing (min)', fontsize=22)
plt.ylabel(r'$N_{eff}$  $\propto$ $V_{depletion}$', fontsize=22)
plt.legend(fontsize = 14)

plt.savefig('HGCAL_2.5_Back' + '.png', dpi=900)


#%%
# HGCAL Campaign Round 2.6 Front

# Get Minimum Values
min_neff_DIODE, min_anneal_time_DIODE =  find_minimum(neff_fit_dict, annealing_fit_dict, 'N4789_20_UR_DIODE')
min_neff_DIODEQUARTER, min_anneal_time_DIODEQUARTER =  find_minimum(neff_fit_dict, annealing_fit_dict, 'N4789_20_UR_DIODEQUARTER')

# Plot Fits 
figure(figsize=(12, 7), dpi=100)

plt.suptitle('HGCAL Campaign Round 2.6 Front', fontsize=20)

plt.scatter(annealing_dict['N4789_20_UR_DIODE'], neff_dict['N4789_20_UR_DIODE'], label = r'N4789_20_UR_DIODE $N_{eff}$ Data')
plt.scatter(annealing_dict['N4789_20_UR_DIODEQUARTER'], neff_dict['N4789_20_UR_DIODEQUARTER'], label = r'N4789_20_UR_DIODEQUARTER $N_{eff}$ Data')

plt.plot(annealing_fit_dict['N4789_20_UR_DIODE'], neff_fit_dict['N4789_20_UR_DIODE'])
# plt.plot(annealing_fit_dict['N4789_20_UR_DIODEQUARTER'], neff_fit_dict['N4789_20_UR_DIODEQUARTER'])

plt.scatter(min_anneal_time_DIODE, min_neff_DIODE, label = 'N4789_20_UR_DIODE Minimum at ' + str(round_sig(min_anneal_time_DIODE)) + ' min')
plt.scatter(min_anneal_time_DIODEQUARTER, min_neff_DIODEQUARTER, label = 'N4789_20_UR_DIODEQUARTER Minimum at ' + str(round_sig(min_anneal_time_DIODEQUARTER)) + ' min')

plt.xlabel('Annealing (min)', fontsize=22)
plt.ylabel(r'$N_{eff}$  $\propto$ $V_{depletion}$', fontsize=22)

plt.legend(fontsize = 14)

plt.savefig('HGCAL_2.6_Front' + '.png', dpi=900)


#%%
# HGCAL Campaign Round 2.6 Back

# Get Minimum Values
min_neff_DIODE, min_anneal_time_DIODE =  find_minimum(neff_fit_dict, annealing_fit_dict, 'N4789_24_UR_DIODE')
min_neff_DIODEQUARTER, min_anneal_time_DIODEQUARTER =  find_minimum(neff_fit_dict, annealing_fit_dict, 'N4789_24_UR_DIODEQUARTER')

# Plot Fits 
figure(figsize=(12, 7), dpi=100)

plt.suptitle('HGCAL Campaign Round 2.6 Back', fontsize=20)

plt.scatter(annealing_dict['N4789_24_UR_DIODE'], neff_dict['N4789_24_UR_DIODE'], label = r'N4789_24_UR_DIODE $N_{eff}$ Data')
plt.scatter(annealing_dict['N4789_24_UR_DIODEQUARTER'], neff_dict['N4789_24_UR_DIODEQUARTER'], label = r'N4789_24_UR_DIODEQUARTER $N_{eff}$ Data')

plt.plot(annealing_fit_dict['N4789_24_UR_DIODE'], neff_fit_dict['N4789_24_UR_DIODE'])
plt.plot(annealing_fit_dict['N4789_24_UR_DIODEQUARTER'], neff_fit_dict['N4789_24_UR_DIODEQUARTER'])

plt.scatter(min_anneal_time_DIODE, min_neff_DIODE, label = 'N4789_24_UR_DIODE Minimum at ' + str(round_sig(min_anneal_time_DIODE)) + ' min')
plt.scatter(min_anneal_time_DIODEQUARTER, min_neff_DIODEQUARTER, label = 'N4789_24_UR_DIODEQUARTER Minimum at ' + str(round_sig(min_anneal_time_DIODEQUARTER)) + ' min')
plt.xlabel('Annealing (min)', fontsize=22)
plt.ylabel(r'$N_{eff}$  $\propto$ $V_{depletion}$', fontsize=22)

plt.legend(fontsize = 14)

plt.savefig('HGCAL_2.6_Back' + '.png', dpi=900)


#%%
# HGCAL Campaign Round 2.7 Front

# Get Minimum Values
min_neff_DIODE, min_anneal_time_DIODE =  find_minimum(neff_fit_dict, annealing_fit_dict, 'N4789_24_UL_DIODE')
min_neff_DIODEHALF, min_anneal_time_DIODEHALF =  find_minimum(neff_fit_dict, annealing_fit_dict, 'N4789_24_UL_DIODEHALF')
min_neff_DIODEQUARTER, min_anneal_time_DIODEQUARTER =  find_minimum(neff_fit_dict, annealing_fit_dict, 'N4789_24_UL_DIODEQUARTER')

# Plot Fits 
figure(figsize=(12, 7), dpi=100)
plt.suptitle('HGCAL Campaign Round 2.7 Front', fontsize=20)
plt.scatter(annealing_dict['N4789_24_UL_DIODE'], neff_dict['N4789_24_UL_DIODE'], label = r'N4789_24_UL_DIODE $N_{eff}$ Data')
plt.scatter(annealing_dict['N4789_24_UL_DIODEHALF'], neff_dict['N4789_24_UL_DIODEHALF'], label = r'N4789_24_UL_DIODEHALF $N_{eff}$ Data')
plt.scatter(annealing_dict['N4789_24_UL_DIODEQUARTER'], neff_dict['N4789_24_UL_DIODEQUARTER'], label = r'N4789_24_UL_DIODEQUARTER $N_{eff}$ Data')

plt.plot(annealing_fit_dict['N4789_24_UL_DIODE'], neff_fit_dict['N4789_24_UL_DIODE'])
plt.plot(annealing_fit_dict['N4789_24_UL_DIODEHALF'], neff_fit_dict['N4789_24_UL_DIODEHALF'])
plt.plot(annealing_fit_dict['N4789_24_UL_DIODEQUARTER'], neff_fit_dict['N4789_24_UL_DIODEQUARTER'])

plt.scatter(min_anneal_time_DIODE, min_neff_DIODE, label = 'N4789_24_UL_DIODE Minimum at ' + str(round_sig(min_anneal_time_DIODE)) + ' min')
plt.scatter(min_anneal_time_DIODEHALF, min_neff_DIODEHALF, label = 'N4789_24_UL_DIODEHALF Minimum at ' + str(round_sig(min_anneal_time_DIODEHALF)) + ' min')
plt.scatter(min_anneal_time_DIODEQUARTER, min_neff_DIODEQUARTER, label = 'N4789_24_UL_DIODEQUARTER Minimum at ' + str(round_sig(min_anneal_time_DIODEQUARTER)) + ' min')
plt.xlabel('Annealing (min)', fontsize=22)
plt.ylabel(r'$N_{eff}$  $\propto$ $V_{depletion}$', fontsize=22)
plt.legend(fontsize = 14)

plt.savefig('HGCAL_2.7_Front' + '.png', dpi=900)


#%%
# HGCAL Campaign Round 2.7 Back

# Get Minimum Values
min_neff_DIODE, min_anneal_time_DIODE =  find_minimum(neff_fit_dict, annealing_fit_dict, 'N4789_24_LR_DIODE')
min_neff_DIODEHALF, min_anneal_time_DIODEHALF =  find_minimum(neff_fit_dict, annealing_fit_dict, 'N4789_24_LR_DIODEHALF')
min_neff_DIODEQUARTER, min_anneal_time_DIODEQUARTER =  find_minimum(neff_fit_dict, annealing_fit_dict, 'N4789_24_LR_DIODEQUARTER')

# Plot Fits 
figure(figsize=(12, 7), dpi=100)
plt.suptitle('HGCAL Campaign Round 2.7 Back', fontsize=20)
plt.scatter(annealing_dict['N4789_24_LR_DIODE'], neff_dict['N4789_24_LR_DIODE'], label = r'N4789_24_LR_DIODE $N_{eff}$ Data')
plt.scatter(annealing_dict['N4789_24_LR_DIODEHALF'], neff_dict['N4789_24_LR_DIODEHALF'], label = r'N4789_24_LR_DIODEHALF $N_{eff}$ Data')
plt.scatter(annealing_dict['N4789_24_LR_DIODEQUARTER'], neff_dict['N4789_24_LR_DIODEQUARTER'], label = r'N4789_24_LR_DIODEQUARTER $N_{eff}$ Data')

plt.plot(annealing_fit_dict['N4789_24_LR_DIODE'], neff_fit_dict['N4789_24_LR_DIODE'])
plt.plot(annealing_fit_dict['N4789_24_LR_DIODEHALF'], neff_fit_dict['N4789_24_LR_DIODEHALF'])
plt.plot(annealing_fit_dict['N4789_24_LR_DIODEQUARTER'], neff_fit_dict['N4789_24_LR_DIODEQUARTER'])

plt.scatter(min_anneal_time_DIODE, min_neff_DIODE, label = 'N4789_24_LR_DIODE Minimum at ' + str(round_sig(min_anneal_time_DIODE)) + ' min')
plt.scatter(min_anneal_time_DIODEHALF, min_neff_DIODEHALF, label = 'N4789_24_LR_DIODEHALF Minimum at ' + str(round_sig(min_anneal_time_DIODEHALF)) + ' min')
plt.scatter(min_anneal_time_DIODEQUARTER, min_neff_DIODEQUARTER, label = 'N4789_24_LR_DIODEQUARTER Minimum at ' + str(round_sig(min_anneal_time_DIODEQUARTER)) + ' min')
plt.xlabel('Annealing (min)', fontsize=22)
plt.ylabel(r'$N_{eff}$  $\propto$ $V_{depletion}$', fontsize=22)
plt.legend(fontsize = 14)

plt.savefig('HGCAL_2.7_Back' + '.png', dpi=900)


#%%
# HGCAL Campaign Round 2.8 Front

# Get Minimum Values
min_neff_DIODE, min_anneal_time_DIODE =  find_minimum(neff_fit_dict, annealing_fit_dict, 'N4789_19_UR_DIODE')
min_neff_DIODEQUARTER, min_anneal_time_DIODEQUARTER =  find_minimum(neff_fit_dict, annealing_fit_dict, 'N4789_19_UR_DIODEQUARTER')

# Plot Fits 
figure(figsize=(12, 7), dpi=100)

plt.suptitle('HGCAL Campaign Round 2.8 Front', fontsize=20)

plt.scatter(annealing_dict['N4789_19_UR_DIODE'], neff_dict['N4789_19_UR_DIODE'], label = r'N4789_19_UR_DIODE $N_{eff}$ Data')
plt.scatter(annealing_dict['N4789_19_UR_DIODEQUARTER'], neff_dict['N4789_19_UR_DIODEQUARTER'], label = r'N4789_19_UR_DIODEQUARTER $N_{eff}$ Data')

plt.plot(annealing_fit_dict['N4789_19_UR_DIODE'], neff_fit_dict['N4789_19_UR_DIODE'])
plt.plot(annealing_fit_dict['N4789_19_UR_DIODEQUARTER'], neff_fit_dict['N4789_19_UR_DIODEQUARTER'])

plt.scatter(min_anneal_time_DIODE, min_neff_DIODE, label = 'N4789_19_UR_DIODE Minimum at ' + str(round_sig(min_anneal_time_DIODE)) + ' min')
plt.scatter(min_anneal_time_DIODEQUARTER, min_neff_DIODEQUARTER, label = 'N4789_19_UR_DIODEQUARTER Minimum at ' + str(round_sig(min_anneal_time_DIODEQUARTER)) + ' min')
plt.xlabel('Annealing (min)', fontsize=22)
plt.ylabel(r'$N_{eff}$  $\propto$ $V_{depletion}$', fontsize=22)

plt.legend(fontsize = 14)

plt.savefig('HGCAL_2.8_Front' + '.png', dpi=900)


#%%
# HGCAL Campaign Round 2.8 Back

# Get Minimum Values
min_neff_DIODE, min_anneal_time_DIODE =  find_minimum(neff_fit_dict, annealing_fit_dict, 'N4789_19_LL_DIODE')
min_neff_DIODEQUARTER, min_anneal_time_DIODEQUARTER =  find_minimum(neff_fit_dict, annealing_fit_dict, 'N4789_19_LL_DIODEQUARTER')

# Plot Fits 
figure(figsize=(12, 7), dpi=100)

plt.suptitle('HGCAL Campaign Round 2.8 Back', fontsize=20)

plt.scatter(annealing_dict['N4789_19_LL_DIODE'], neff_dict['N4789_19_LL_DIODE'], label = r'N4789_19_LL_DIODE $N_{eff}$ Data')
plt.scatter(annealing_dict['N4789_19_LL_DIODEQUARTER'], neff_dict['N4789_19_LL_DIODEQUARTER'], label = r'N4789_19_LL_DIODEQUARTER $N_{eff}$ Data')

plt.plot(annealing_fit_dict['N4789_19_LL_DIODE'], neff_fit_dict['N4789_19_LL_DIODE'])
plt.plot(annealing_fit_dict['N4789_19_LL_DIODEQUARTER'], neff_fit_dict['N4789_19_LL_DIODEQUARTER'])

plt.scatter(min_anneal_time_DIODE, min_neff_DIODE, label = 'N4789_19_LL_DIODE Minimum at ' + str(round_sig(min_anneal_time_DIODE)) + ' min')
plt.scatter(min_anneal_time_DIODEQUARTER, min_neff_DIODEQUARTER, label = 'N4789_19_LL_DIODEQUARTER Minimum at ' + str(round_sig(min_anneal_time_DIODEQUARTER)) + ' min')
plt.xlabel('Annealing (min)', fontsize=22)
plt.ylabel(r'$N_{eff}$  $\propto$ $V_{depletion}$', fontsize=22)

plt.legend(fontsize = 14)

plt.savefig('HGCAL_2.8_Back' + '.png', dpi=900)














