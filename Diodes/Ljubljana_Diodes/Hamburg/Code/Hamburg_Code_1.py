import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

annealing = []
neff = []

df = pd.read_csv('Hamburg_Model.csv')
annealing = df['Annealing at 60C (min)']
neff = df['Neff']

fluence = 6.5*10**18

guess = np.array([.269, -1.64*10**18, .437, 50.6, 212])

def Hamburg(x, ga, NC, gy, ta, ty):
    return ga*np.exp(-x/ta)*fluence + gy*(1.-1./(1.+x/ty))*fluence + NC

Params, Covariances = curve_fit(Hamburg, annealing, neff, guess)

annealing_fit = np.linspace(annealing[0], annealing[18], 10000)
neff_fit = Params[0]*np.exp(-annealing_fit/Params[3])*fluence + Params[2]*(1.-1./(1.+annealing_fit/Params[4]))*fluence + Params[1]

plt.scatter(annealing, neff)
plt.plot(annealing_fit, neff_fit)

print(min(neff_fit))

value = min(neff_fit)

for i in range(0,len(neff_fit)):
    if value == neff_fit[i]:
        print(i)
        print(annealing_fit[i])