import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from matplotlib.pyplot import figure

#%%

neff_N4789_20_LL_DIODE = []
neff_N4789_20_LL_DIODEQUARTER = []
neff_N6789_24_LL_DIODE = []
neff_N6789_24_LL_DIODEQUARTER = []

df_N4789_20_LL_DIODE = pd.read_csv('N4789_20_LL_DIODE.csv')
df_N4789_20_LL_DIODEQUARTER = pd.read_csv('N4789_20_LL_DIODEQUARTER.csv')
df_N6789_24_LL_DIODE = pd.read_csv('N6789_24_LL_DIODE.csv')
df_N6789_24_LL_DIODEQUARTER = pd.read_csv('N6789_24_LL_DIODEQUARTER.csv')

annealing_N4789_20_LL_DIODE = df_N4789_20_LL_DIODE['Annealing at 60C (min)']
annealing_N4789_20_LL_DIODEQUARTER = df_N4789_20_LL_DIODEQUARTER['Annealing at 60C (min)']
annealing_N6789_24_LL_DIODE = df_N6789_24_LL_DIODE['Annealing at 60C (min)']
annealing_N6789_24_LL_DIODEQUARTER = df_N6789_24_LL_DIODEQUARTER['Annealing at 60C (min)']

neff_N4789_20_LL_DIODE = df_N4789_20_LL_DIODE['Neff']
neff_N4789_20_LL_DIODEQUARTER = df_N4789_20_LL_DIODEQUARTER['Neff']
neff_N6789_24_LL_DIODE = df_N6789_24_LL_DIODE['Neff']
neff_N6789_24_LL_DIODEQUARTER = df_N6789_24_LL_DIODEQUARTER['Neff']

#%%

fluence = 6.5*10**18

guess = np.array([.9646, -3.5*10**18, 1.4, 63, 90])

def Hamburg(x, ga, NC, gy, ta, ty):
    return ga*np.exp(-x/ta)*fluence + gy*(1.-1./(1.+x/ty))*fluence + NC

annealing_N4789_20_LL_DIODE_fit = np.linspace(annealing_N4789_20_LL_DIODE[0], annealing_N4789_20_LL_DIODE[18], 1000)
annealing_N4789_20_LL_DIODEQUARTER_fit = np.linspace(annealing_N4789_20_LL_DIODEQUARTER[0], annealing_N4789_20_LL_DIODEQUARTER[4], 1000)
annealing_N6789_24_LL_DIODE_fit = np.linspace(annealing_N6789_24_LL_DIODE[0], annealing_N6789_24_LL_DIODE[18], 1000)
annealing_N6789_24_LL_DIODEQUARTER_fit = np.linspace(annealing_N6789_24_LL_DIODEQUARTER[0], annealing_N6789_24_LL_DIODEQUARTER[4], 1000)

Params_N4789_20_LL_DIODE, Covariances_N4789_20_LL_DIODE = curve_fit(Hamburg, annealing_N4789_20_LL_DIODE, neff_N4789_20_LL_DIODE, guess)
Params_N4789_20_LL_DIODEQUARTER, Covariances_N4789_20_LL_DIODEQUARTER = curve_fit(Hamburg, annealing_N4789_20_LL_DIODEQUARTER, neff_N4789_20_LL_DIODEQUARTER, guess)
Params_N6789_24_LL_DIODE, Covariances_N6789_24_LL_DIODE = curve_fit(Hamburg, annealing_N6789_24_LL_DIODE, neff_N6789_24_LL_DIODE, guess)
Params_N6789_24_LL_DIODEQUARTER, Covariances_N6789_24_LL_DIODEQUARTER = curve_fit(Hamburg, annealing_N6789_24_LL_DIODEQUARTER, neff_N6789_24_LL_DIODEQUARTER, guess)

neff_N4789_20_LL_DIODE_fit = Params_N4789_20_LL_DIODE[0]*np.exp(-annealing_N4789_20_LL_DIODE_fit/Params_N4789_20_LL_DIODE[3])*fluence + Params_N4789_20_LL_DIODE[2]*(1.-1./(1.+annealing_N4789_20_LL_DIODE_fit/Params_N4789_20_LL_DIODE[4]))*fluence + Params_N4789_20_LL_DIODE[1]
neff_N4789_20_LL_DIODEQUARTER_fit = Params_N4789_20_LL_DIODEQUARTER[0]*np.exp(-annealing_N4789_20_LL_DIODEQUARTER_fit/Params_N4789_20_LL_DIODEQUARTER[3])*fluence + Params_N4789_20_LL_DIODEQUARTER[2]*(1.-1./(1.+annealing_N4789_20_LL_DIODEQUARTER_fit/Params_N4789_20_LL_DIODEQUARTER[4]))*fluence + Params_N4789_20_LL_DIODEQUARTER[1]
neff_N6789_24_LL_DIODE_fit = Params_N6789_24_LL_DIODE[0]*np.exp(-annealing_N6789_24_LL_DIODE_fit/Params_N6789_24_LL_DIODE[3])*fluence + Params_N6789_24_LL_DIODE[2]*(1.-1./(1.+annealing_N6789_24_LL_DIODE_fit/Params_N6789_24_LL_DIODE[4]))*fluence + Params_N6789_24_LL_DIODE[1]
neff_N6789_24_LL_DIODEQUARTER_fit = Params_N6789_24_LL_DIODEQUARTER[0]*np.exp(-annealing_N6789_24_LL_DIODEQUARTER_fit/Params_N6789_24_LL_DIODEQUARTER[3])*fluence + Params_N6789_24_LL_DIODEQUARTER[2]*(1.-1./(1.+annealing_N6789_24_LL_DIODEQUARTER_fit/Params_N6789_24_LL_DIODEQUARTER[4]))*fluence + Params_N6789_24_LL_DIODEQUARTER[1]

#%%

figure(figsize=(10, 10), dpi=100)

plt.suptitle('HGCal Campaign Round 2.3', fontsize=20)
plt.scatter(71.92619261926193,9.11857E+18, label = 'N4789_20_LL_DIODE Minimum at 71.93 min')
plt.scatter(71.92619261926193,9.11857E+18, label = 'N4789_20_LL_DIODEQUARTER Minimum at 71.93 min')
plt.scatter(annealing_N4789_20_LL_DIODE, neff_N4789_20_LL_DIODE, label = r'N4789_20_LL_DIODE $N_{eff}$ Data')
plt.scatter(annealing_N4789_20_LL_DIODEQUARTER, neff_N4789_20_LL_DIODEQUARTER, label = r'N4789_20_LL_DIODEQUARTER $N_{eff}$ Data')
plt.plot(annealing_N4789_20_LL_DIODE_fit, neff_N4789_20_LL_DIODE_fit, label = r'N4789_20_LL_DIODE $N_{eff}$ Fit')
plt.plot(annealing_N4789_20_LL_DIODEQUARTER_fit, neff_N4789_20_LL_DIODEQUARTER_fit, label = r'N4789_20_LL_DIODEQUARTER $N_{eff}$ Fit')
plt.xlabel('Annealing (min)', fontsize=22)
plt.ylabel(r'$N_{eff}$  $\propto$ $V_{depletion}$', fontsize=22)
plt.legend(fontsize = 18)

#%%

figure(figsize=(10, 10), dpi=100)

plt.suptitle('HGCal Campaign Round 2.2', fontsize=20)
plt.scatter(71.92619261926193,9.11857E+18, label = 'N6789_24_LL_DIODE Minimum at 71.93 min')
plt.scatter(71.92619261926193,9.11857E+18, label = 'N6789_24_LL_DIODEQUARTER Minimum at 71.93 min')
plt.scatter(annealing_N6789_24_LL_DIODE, neff_N6789_24_LL_DIODE, label = r'N6789_24_LL_DIODE $N_{eff}$ Data')
plt.scatter(annealing_N6789_24_LL_DIODEQUARTER, neff_N6789_24_LL_DIODEQUARTER, label = r'N6789_24_LL_DIODEQUARTER $N_{eff}$ Data')
plt.plot(annealing_N6789_24_LL_DIODE_fit, neff_N6789_24_LL_DIODE_fit, label = r'N6789_24_LL_DIODE $N_{eff}$ Fit')
plt.plot(annealing_N6789_24_LL_DIODEQUARTER_fit, neff_N6789_24_LL_DIODEQUARTER_fit, label = r'N6789_24_LL_DIODEQUARTER $N_{eff}$ Fit')
plt.xlabel('Annealing (min)', fontsize=22)
plt.ylabel(r'$N_{eff}$  $\propto$ $V_{depletion}$', fontsize=22)
plt.legend(fontsize = 18)


#%%
print(min(neff_N4789_20_LL_DIODE_fit))

value = min(neff_N4789_20_LL_DIODE_fit)

for i in range(0,len(neff_N4789_20_LL_DIODE_fit)):
    if value == neff_N4789_20_LL_DIODE_fit[i]:
        print(i)
        print(annealing_N4789_20_LL_DIODE_fit[i])
        print(neff_N4789_20_LL_DIODE_fit[i])
        
print(Params_N4789_20_LL_DIODE)

#%%
print(min(neff_N4789_20_LL_DIODEQUARTER_fit))

value = min(neff_N4789_20_LL_DIODEQUARTER_fit)

for i in range(0,len(neff_N4789_20_LL_DIODEQUARTER_fit)):
    if value == neff_N4789_20_LL_DIODEQUARTER_fit[i]:
        print(i)
        print(annealing_N4789_20_LL_DIODEQUARTER_fit[i])
        print(neff_N4789_20_LL_DIODEQUARTER_fit[i])
        
print(Params_N4789_20_LL_DIODEQUARTER)

#%%

print(min(neff_N6789_24_LL_DIODE_fit))

value = min(neff_N6789_24_LL_DIODE_fit)

for i in range(0,len(neff_N6789_24_LL_DIODE_fit)):
    if value == neff_N6789_24_LL_DIODE_fit[i]:
        print(i)
        print(annealing_N6789_24_LL_DIODE_fit[i])
        print(neff_N6789_24_LL_DIODE_fit[i])
        
print(Params_N6789_24_LL_DIODE)

#%%

print(min(neff_N6789_24_LL_DIODEQUARTER_fit))

value = min(neff_N6789_24_LL_DIODEQUARTER_fit)

for i in range(0,len(neff_N6789_24_LL_DIODEQUARTER_fit)):
    if value == neff_N6789_24_LL_DIODEQUARTER_fit[i]:
        print(i)
        print(annealing_N6789_24_LL_DIODEQUARTER_fit[i])
        print(neff_N6789_24_LL_DIODEQUARTER_fit[i])
        
print(Params_N6789_24_LL_DIODEQUARTER)

















