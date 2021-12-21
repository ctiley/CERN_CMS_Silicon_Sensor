import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from matplotlib.pyplot import figure

neff_138 = []
neff_139 = []

df_138 = pd.read_csv('CO138_Neff_vs_Annealing.csv')
df_139 = pd.read_csv('CO139_Neff_vs_Annealing.csv')

annealing_138 = df_138['Annealing at 60C (min)']
annealing_139 = df_139['Annealing at 60C (min)']

neff_138 = df_138['Neff']
neff_139 = df_139['Neff']

fluence = 6.5*10**18

guess = np.array([.9646, -3.5*10**18, 1.4, 63, 90])

def Hamburg(x, ga, NC, gy, ta, ty):
    return ga*np.exp(-x/ta)*fluence + gy*(1.-1./(1.+x/ty))*fluence + NC

annealing_138_fit = np.linspace(annealing_138[0], annealing_138[18], 1000)
annealing_139_fit = np.linspace(annealing_139[0], annealing_139[4], 1000)

Params_138, Covariances_138 = curve_fit(Hamburg, annealing_138, neff_138, guess)
#Params_139, Covariances_139 = curve_fit(Hamburg, annealing_139, neff_139, guess)

neff_138_fit = Params_138[0]*np.exp(-annealing_138_fit/Params_138[3])*fluence + Params_138[2]*(1.-1./(1.+annealing_138_fit/Params_138[4]))*fluence + Params_138[1]
#neff_139_fit = Params_139[0]*np.exp(-annealing_139_fit/Params_139[3])*fluence + Params_139[2]*(1.-1./(1.+annealing_139_fit/Params_139[4]))*fluence + Params_139[1]


figure(figsize=(10, 10), dpi=100)

plt.scatter(71.92619261926193,9.11857E+18, label = 'CO138 Minimum at 71.93 min')
plt.scatter(annealing_138, neff_138, label = r'CO138 $N_{eff}$ Data')
plt.scatter(annealing_139, neff_139, label = r'CO139 $N_{eff}$ Data')
plt.plot(annealing_138_fit, neff_138_fit, label = r'CO138 $N_{eff}$ Fit')
#plt.plot(annealing_139_fit, neff_139_fit, label = r'CO139 $N_{eff}$ Fit')
plt.xlabel('Annealing (min)', fontsize=22)
plt.ylabel(r'$N_{eff}$  $\propto$ $V_{depletion}$', fontsize=22)
plt.legend(fontsize = 18)

"""
print(min(neff_138_fit))

value = min(neff_138_fit)

for i in range(0,len(neff_138_fit)):
    if value == neff_138_fit[i]:
        print(i)
        print(annealing_138_fit[i])
        print(neff_138_fit[i])
        
print(Params_138)
print(Params_139)
"""