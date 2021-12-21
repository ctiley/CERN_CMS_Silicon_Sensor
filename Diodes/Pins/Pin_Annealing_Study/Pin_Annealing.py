from scipy.optimize import curve_fit
from math import log10, floor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#%%

# Significant Figures Function

def round_sig(x, sig=6, small_value=1.0e-9):
    return round(x, sig - int(floor(log10(max(abs(x), abs(small_value))))) - 1)

#%%

# Load data
all_data_40C = pd.read_csv('All_Data/40C.csv')
all_data_50C = pd.read_csv('All_Data/50C.csv')
all_data_60C = pd.read_csv('All_Data/60C.csv')
all_data_70C = pd.read_csv('All_Data/70C.csv')
all_data_80C = pd.read_csv('All_Data/80C.csv')
all_data_90C = pd.read_csv('All_Data/90C.csv')
all_data_100C = pd.read_csv('All_Data/100C.csv')

Annealing_40C_Shifted_and_Scaled_data = pd.read_csv('Standard_Candles/Data/40C_Annealing_Shifted_and_Scaled.csv')
Annealing_50C_Shifted_and_Scaled_data = pd.read_csv('Standard_Candles/Data/50C_Annealing_Shifted_and_Scaled.csv')
Annealing_60C_Shifted_and_Scaled_data = pd.read_csv('Standard_Candles/Data/60C_Annealing_Shifted_and_Scaled.csv')
Annealing_70C_Shifted_and_Scaled_data = pd.read_csv('Standard_Candles/Data/70C_Annealing_Shifted_and_Scaled.csv')
Annealing_80C_Shifted_and_Scaled_data = pd.read_csv('Standard_Candles/Data/80C_Annealing_Shifted_and_Scaled.csv')
Annealing_90C_Shifted_and_Scaled_data = pd.read_csv('Standard_Candles/Data/90C_Annealing_Shifted_and_Scaled.csv')
Annealing_100C_Shifted_and_Scaled_data = pd.read_csv('Standard_Candles/Data/100C_Annealing_Shifted_and_Scaled.csv')

Annealing_40C_Shifted_data = pd.read_csv('Standard_Candles/Data/40C_Annealing_Shifted.csv')
Annealing_50C_Shifted_data = pd.read_csv('Standard_Candles/Data/50C_Annealing_Shifted.csv')
Annealing_60C_Shifted_data = pd.read_csv('Standard_Candles/Data/60C_Annealing_Shifted.csv')
Annealing_70C_Shifted_data = pd.read_csv('Standard_Candles/Data/70C_Annealing_Shifted.csv')
Annealing_80C_Shifted_data = pd.read_csv('Standard_Candles/Data/80C_Annealing_Shifted.csv')
Annealing_90C_Shifted_data = pd.read_csv('Standard_Candles/Data/90C_Annealing_Shifted.csv')
Annealing_100C_Shifted_data = pd.read_csv('Standard_Candles/Data/100C_Annealing_Shifted.csv')

exponential_scale_data = pd.read_csv('Standard_Candles/Data/Exponential_Scale.csv')

len_40C = len(Annealing_40C_Shifted_and_Scaled_data['Annealing Time (Min)']) - 1
len_50C = len(Annealing_50C_Shifted_and_Scaled_data['Annealing Time (Min)']) - 1
len_60C = len(Annealing_60C_Shifted_and_Scaled_data['Annealing Time (Min)']) - 1
len_70C = len(Annealing_70C_Shifted_and_Scaled_data['Annealing Time (Min)']) - 1
len_80C = len(Annealing_80C_Shifted_and_Scaled_data['Annealing Time (Min)']) - 1
len_90C = len(Annealing_90C_Shifted_and_Scaled_data['Annealing Time (Min)']) - 1
len_100C = len(Annealing_100C_Shifted_and_Scaled_data['Annealing Time (Min)']) -1
len_exponential_scale = len(exponential_scale_data['Annealing Temperature']) - 1

#%%


# Curve Fitting 

# Fitting Fuction for One Exponential
def exponential_1(z, A, g, a, d, H, l, j):
    return  A*np.exp(-g*(z-a)) + d + H*np.exp(-l*(z-j)) 

guess = np.array([ 1,  .001, -10,  0.1, .1,  .1,  10])
guess_40C = np.array([ 2.71577762e-01,  2.84736599e-03, -5.79459958e+00,  6.91630877e-01,  8.06029899e-02,  6.30728123e-02,  6.29289547e-01])
guess_90C = np.array([2.82143883e-01, 2.56956240e-03, 5.07634758e+00, 1.02948841e-01, 2.67226126e-01, 6.12459515e-02, 5.21496481e+00])
guess_100C = np.array([ 9.28731446e-01,  1.06016301e-01,  2.56832666e+00,  2.46872353e-01,  3.58057255e-01,  4.67793265e-03, -5.59542518e+01])

fitted_annealing_40C = np.linspace(Annealing_40C_Shifted_and_Scaled_data['Annealing Time (Min)'][0], Annealing_40C_Shifted_and_Scaled_data['Annealing Time (Min)'][len_40C], 1000)
params_40C, params_covariance_40C = curve_fit(exponential_1, Annealing_40C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_40C_Shifted_and_Scaled_data['Pin107 Vf (10 min R irad)'], guess_40C)
fitted_exponential_40C_raw = params_40C[0]*np.exp(-1*params_40C[1]*(fitted_annealing_40C - params_40C[2])) + params_40C[3] + params_40C[4]*np.exp(-1*params_40C[5]*(fitted_annealing_40C - params_40C[6]))
first_fitted_exponential_40C = params_40C[0]*np.exp(-1*params_40C[1]*(fitted_annealing_40C - params_40C[2])) + params_40C[3] 
second_fitted_exponential_40C =   params_40C[4]*np.exp(-1*params_40C[5]*(fitted_annealing_40C - params_40C[6])) + params_40C[3] 

fitted_annealing_50C = np.linspace(Annealing_50C_Shifted_and_Scaled_data['Annealing Time (Min)'][0], Annealing_50C_Shifted_and_Scaled_data['Annealing Time (Min)'][len_50C], 1000)
params_50C, params_covariance_50C = curve_fit(exponential_1, Annealing_50C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_50C_Shifted_and_Scaled_data['Pin109 Vf (10 min R irad)'])
fitted_exponential_50C_raw = params_50C[0]*np.exp(-1*params_50C[1]*(fitted_annealing_50C - params_50C[2])) + params_50C[3] + params_50C[4]*np.exp(-1*params_50C[5]*(fitted_annealing_50C - params_50C[6]))
first_fitted_exponential_50C = params_50C[0]*np.exp(-1*params_50C[1]*(fitted_annealing_50C - params_50C[2])) + params_50C[3] 
second_fitted_exponential_50C =   params_50C[4]*np.exp(-1*params_50C[5]*(fitted_annealing_50C - params_50C[6])) + params_50C[3] 

fitted_annealing_60C = np.linspace(Annealing_60C_Shifted_and_Scaled_data['Annealing Time (Min)'][0], Annealing_60C_Shifted_and_Scaled_data['Annealing Time (Min)'][len_60C], 1000)
params_60C, params_covariance_60C = curve_fit(exponential_1, Annealing_60C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_60C_Shifted_and_Scaled_data['Pin105 Vf (10 min R irad)'])
fitted_exponential_60C_raw = params_60C[0]*np.exp(-1*params_60C[1]*(fitted_annealing_60C - params_60C[2])) + params_60C[3] + params_60C[4]*np.exp(-1*params_60C[5]*(fitted_annealing_60C - params_60C[6])) 
first_fitted_exponential_60C = params_60C[0]*np.exp(-1*params_60C[1]*(fitted_annealing_60C - params_60C[2])) + params_60C[3] 
second_fitted_exponential_60C = params_60C[3] + params_60C[4]*np.exp(-1*params_60C[5]*(fitted_annealing_60C - params_60C[6])) 

fitted_annealing_70C = np.linspace(Annealing_70C_Shifted_and_Scaled_data['Annealing Time (Min)'][0], Annealing_70C_Shifted_and_Scaled_data['Annealing Time (Min)'][len_70C], 1000)
params_70C, params_covariance_70C = curve_fit(exponential_1, Annealing_70C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_70C_Shifted_and_Scaled_data['Pin110 Vf (10 min R irad)'], guess)
fitted_exponential_70C_raw = params_70C[0]*np.exp(-1*params_70C[1]*(fitted_annealing_70C - params_70C[2])) + params_70C[3] + params_70C[4]*np.exp(-1*params_70C[5]*(fitted_annealing_70C - params_70C[6]))
first_fitted_exponential_70C = params_70C[0]*np.exp(-1*params_70C[1]*(fitted_annealing_70C - params_70C[2])) + params_70C[3] 
second_fitted_exponential_70C =   params_70C[4]*np.exp(-1*params_70C[5]*(fitted_annealing_70C - params_70C[6])) + params_70C[3] 

fitted_annealing_80C = np.linspace(Annealing_80C_Shifted_and_Scaled_data['Annealing Time (Min)'][0], Annealing_80C_Shifted_and_Scaled_data['Annealing Time (Min)'][len_80C], 1000)
params_80C, params_covariance_80C = curve_fit(exponential_1, Annealing_80C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_80C_Shifted_and_Scaled_data['Pin108 Vf (10 min R irad)'])
fitted_exponential_80C_raw = params_80C[0]*np.exp(-1*params_80C[1]*(fitted_annealing_80C - params_80C[2])) + params_80C[3] + params_80C[4]*np.exp(-1*params_80C[5]*(fitted_annealing_80C - params_80C[6]))
first_fitted_exponential_80C = params_80C[0]*np.exp(-1*params_80C[1]*(fitted_annealing_80C - params_80C[2])) + params_80C[3]
second_fitted_exponential_80C = params_80C[3] + params_80C[4]*np.exp(-1*params_80C[5]*(fitted_annealing_80C - params_80C[6]))

fitted_annealing_90C = np.linspace(Annealing_90C_Shifted_and_Scaled_data['Annealing Time (Min)'][0], Annealing_90C_Shifted_and_Scaled_data['Annealing Time (Min)'][len_90C], 1000)
params_90C, params_covariance_90C = curve_fit(exponential_1, Annealing_90C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_90C_Shifted_and_Scaled_data['Pin111 Vf (10 min R irad)'])
fitted_exponential_90C_raw = params_90C[0]*np.exp(-1*params_90C[1]*(fitted_annealing_90C - params_90C[2])) + params_90C[3] + params_90C[4]*np.exp(-1*params_90C[5]*(fitted_annealing_90C - params_90C[6]))
first_fitted_exponential_90C = params_90C[0]*np.exp(-1*params_90C[1]*(fitted_annealing_90C - params_90C[2])) + params_90C[3] 
second_fitted_exponential_90C =   params_90C[4]*np.exp(-1*params_90C[5]*(fitted_annealing_90C - params_90C[6])) + params_90C[3] 

fitted_annealing_100C = np.linspace(Annealing_100C_Shifted_and_Scaled_data['Annealing Time (Min)'][0], Annealing_100C_Shifted_and_Scaled_data['Annealing Time (Min)'][len_100C], 1000)
params_100C, params_covariance_100C = curve_fit(exponential_1, Annealing_100C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_100C_Shifted_and_Scaled_data['Pin106 Vf (10 min R irad)'], guess_100C)
fitted_exponential_100C_raw = params_100C[0]*np.exp(-1*params_100C[1]*(fitted_annealing_100C - params_100C[2])) + params_100C[3] + params_100C[4]*np.exp(-1*params_100C[5]*(fitted_annealing_100C - params_100C[6]))
first_fitted_exponential_100C = params_100C[0]*np.exp(-1*params_100C[1]*(fitted_annealing_100C - params_100C[2])) + params_100C[3] 
second_fitted_exponential_100C = params_100C[3] + params_100C[4]*np.exp(-1*params_100C[5]*(fitted_annealing_100C - params_100C[6]))


#%%
# Plotting For 40C

# Plot
plt.figure(figsize=(6, 4))
x = range(len(Annealing_40C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(Annealing_40C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_40C_Shifted_and_Scaled_data['Pin107 Vf (10 min R irad)'], '.-', label = 'Pin107 Vf (10 min R irad)')
plt.plot(fitted_annealing_40C, fitted_exponential_40C_raw, label = 'Fitted Data ')
plt.plot(fitted_annealing_40C, first_fitted_exponential_40C, label = 'First Exponential of Fitted Data ')
plt.plot(fitted_annealing_40C, second_fitted_exponential_40C, label = 'Second Exponential of Fitted Data ')
plt.xlabel(r'Annealing Time $(min)$')
plt.ylabel(r'Normalized Forward Voltage $V_f$')
plt.title(r'40C Annealing')
plt.legend(loc = 'best')
plt.show()


# Plotting For 50C

# Plot
plt.figure(figsize=(6, 4))
x = range(len(Annealing_50C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(Annealing_50C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_50C_Shifted_and_Scaled_data['Pin109 Vf (10 min R irad)'], '.-', label = 'Pin109 Vf (10 min R irad)')
plt.plot(fitted_annealing_50C, fitted_exponential_50C_raw, label = 'Fitted Data ')
plt.plot(fitted_annealing_50C, first_fitted_exponential_50C, label = 'First Exponential of Fitted Data ')
plt.plot(fitted_annealing_50C, second_fitted_exponential_50C, label = 'Second Exponential of Fitted Data ')
plt.xlabel(r'Annealing Time $(min)$')
plt.ylabel(r'Normalized Forward Voltage $V_f$')
plt.title(r'50C Annealing')
plt.legend(loc = 'best')
plt.show()


# Plotting For 60C

# Plot
plt.figure(figsize=(6, 4))
x = range(len(Annealing_60C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(Annealing_60C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_60C_Shifted_and_Scaled_data['Pin105 Vf (10 min R irad)'], '.-', label = 'Pin105 Vf (10 min R irad)')
plt.plot(fitted_annealing_60C, fitted_exponential_60C_raw, label = 'Fitted Data ')
plt.plot(fitted_annealing_60C, first_fitted_exponential_60C, label = 'First Exponential of Fitted Data ' )
plt.plot(fitted_annealing_60C, second_fitted_exponential_60C, label = 'Second Exponential of Fitted Data ' )
plt.xlabel(r'Annealing Time $(min)$')
plt.ylabel(r'Normalized Forward Voltage $V_f$')
plt.title(r'60C Annealing')
plt.legend(loc = 'best')
plt.show()


# Plotting For 70C

# Plot
plt.figure(figsize=(6, 4))
x = range(len(Annealing_70C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(Annealing_70C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_70C_Shifted_and_Scaled_data['Pin110 Vf (10 min R irad)'], '.-', label = 'Pin110 Vf (10 min R irad)')
plt.plot(fitted_annealing_70C, fitted_exponential_70C_raw, label = 'Fitted Data ')
plt.plot(fitted_annealing_70C, first_fitted_exponential_70C, label = 'First Exponential of Fitted Data ' )
plt.plot(fitted_annealing_70C, second_fitted_exponential_70C, label = 'Second Exponential of Fitted Data ' )
plt.xlabel(r'Annealing Time $(min)$')
plt.ylabel(r'Normalized Forward Voltage $V_f$')
plt.title(r'70C Annealing')
plt.legend(loc = 'best')
plt.show()


# Plotting For 80C

# Plot
plt.figure(figsize=(6, 4))
x = range(len(Annealing_80C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(Annealing_80C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_80C_Shifted_and_Scaled_data['Pin108 Vf (10 min R irad)'], '.-', label = 'Pin108 Vf (10 min R irad)')
plt.plot(fitted_annealing_80C, fitted_exponential_80C_raw, label = 'Fitted Data ')
plt.plot(fitted_annealing_80C, first_fitted_exponential_80C, label = 'First Exponential of Fitted Data ')
plt.plot(fitted_annealing_80C, second_fitted_exponential_80C, label = 'Second Exponential of Fitted Data ' )
plt.xlabel(r'Annealing Time $(min)$')
plt.ylabel(r'Normalized Forward Voltage $V_f$')
plt.title(r'80C Annealing')
plt.legend(loc = 'best')
plt.show()


# Plotting For 90C

# Plot
plt.figure(figsize=(6, 4))
x = range(len(Annealing_90C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(Annealing_90C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_90C_Shifted_and_Scaled_data['Pin111 Vf (10 min R irad)'], '.-', label = 'Pin111 Vf (10 min R irad)')
plt.plot(fitted_annealing_90C, fitted_exponential_90C_raw, label = 'Fitted Data ')
plt.plot(fitted_annealing_90C, first_fitted_exponential_90C, label = 'First Exponential of Fitted Data ' )
plt.plot(fitted_annealing_90C, second_fitted_exponential_90C, label = 'Second Exponential of Fitted Data '  )
plt.xlabel(r'Annealing Time $(min)$')
plt.ylabel(r'Normalized Forward Voltage $V_f$')
plt.title(r'90C Annealing')
plt.legend(loc = 'best')
plt.show()


# Plot
plt.figure(figsize=(6, 4))
x = range(len(Annealing_100C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(Annealing_100C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_100C_Shifted_and_Scaled_data['Pin106 Vf (10 min R irad)'], '.-', label = 'Pin106 Vf (10 min R irad)')
plt.plot(fitted_annealing_100C, fitted_exponential_100C_raw, label = 'Fitted Data ')
plt.plot(fitted_annealing_100C, first_fitted_exponential_100C, label = 'First Exponential of Fitted Data ' )
plt.plot(fitted_annealing_100C, second_fitted_exponential_100C, label = 'Second Exponential of Fitted Data '  )
plt.xlabel(r'Annealing Time $(min)$')
plt.ylabel(r'Normalized Forward Voltage $V_f$')
plt.title(r'100C Annealing')
plt.legend(loc = 'best')
plt.show()


#%%

# Curve Fitting Using 100C Annealing Parameters and Fitted Double Exponential from Above Without Vertical Shift

# 100C Fitting Parameters
param_1 = 8.36816844e-01
param_2 = 1.16056139e-01
param_3 = 3.68615018e+00
param_4 = 2.64798572e-01
param_5 = 3.11230988e-01
param_6 = 6.04407331e-03
param_7 = -1.72259638e+01

# Fitting Fuction for Two Exponential
def exponential_2(t, a, b):
    return  param_1*np.exp(-param_2*(a*(t - b)+param_3)) + param_4 + param_5*np.exp(-param_6*(a*(t - b)-param_7)) 

guess = np.array([1, 0])
guess_90C = np.array([ 1.01394153e-01, -1.53550593e+02])

params_40C, params_covariance_40C = curve_fit(exponential_2, fitted_annealing_40C, fitted_exponential_40C_raw)
fitted_exponential_40C = param_1*np.exp(-param_2*(params_40C[0]*(fitted_annealing_40C - params_40C[1]) + param_3)) + param_4 + param_5*np.exp(-param_6*(params_40C[0]*(fitted_annealing_40C - params_40C[1])-param_7))                                   

params_50C, params_covariance_50C = curve_fit(exponential_2, fitted_annealing_50C, fitted_exponential_50C_raw)
fitted_exponential_50C = param_1*np.exp(-param_2*(params_50C[0]*(fitted_annealing_50C - params_50C[1]) + param_3)) + param_4 + param_5*np.exp(-param_6*(params_50C[0]*(fitted_annealing_50C - params_50C[1])-param_7)) 

params_60C, params_covariance_60C = curve_fit(exponential_2, fitted_annealing_60C, fitted_exponential_60C_raw)
fitted_exponential_60C = param_1*np.exp(-param_2*(params_60C[0]*(fitted_annealing_60C - params_60C[1]) + param_3)) + param_4 + param_5*np.exp(-param_6*(params_60C[0]*(fitted_annealing_60C - params_60C[1])-param_7)) 

params_70C, params_covariance_70C = curve_fit(exponential_2, fitted_annealing_70C, fitted_exponential_70C_raw)
fitted_exponential_70C = param_1*np.exp(-param_2*(params_70C[0]*(fitted_annealing_70C - params_70C[1]) + param_3)) + param_4 + param_5*np.exp(-param_6*(params_70C[0]*(fitted_annealing_70C - params_70C[1])-param_7))

params_80C, params_covariance_80C = curve_fit(exponential_2, fitted_annealing_80C, fitted_exponential_80C_raw)
fitted_exponential_80C = param_1*np.exp(-param_2*(params_80C[0]*(fitted_annealing_80C - params_80C[1]) + param_3)) + param_4 + param_5*np.exp(-param_6*(params_80C[0]*(fitted_annealing_80C - params_80C[1])-param_7)) 

params_90C, params_covariance_90C = curve_fit(exponential_2, fitted_annealing_90C, fitted_exponential_90C_raw, guess_90C)
fitted_exponential_90C = param_1*np.exp(-param_2*(params_90C[0]*(fitted_annealing_90C - params_90C[1]) + param_3)) + param_4 + param_5*np.exp(-param_6*(params_90C[0]*(fitted_annealing_90C - params_90C[1])-param_7))      

params_100C, params_covariance_100C = curve_fit(exponential_2, fitted_annealing_100C, fitted_exponential_100C_raw, guess)
fitted_exponential_100C = param_1*np.exp(-param_2*(params_100C[0]*(fitted_annealing_100C - params_100C[1]) + param_3)) + param_4 + param_5*np.exp(-param_6*(params_100C[0]*(fitted_annealing_100C - params_100C[1])-param_7)) 


#%%
# Curve Fitting for Expoenetial Scaling Inverse Kelvin

# Fitting Fuction for Exponential with Inverse Time
def exponential_inverse(t, a, b):
    return  np.exp(a*(1/(t+273.15)-1/b))

# Fitting Fuction for Exponential
def exponential(t, a, b):
    return  np.exp(a*(t-b)) 

guess = np.array([-13478, 373.15])

fitted_exponential_scale_data = np.linspace(exponential_scale_data['Annealing Temperature'][0], exponential_scale_data['Exponential Scale Factor'][len_exponential_scale], 1000)
params_exponential, params_covariance_exponential = curve_fit(exponential_inverse, exponential_scale_data['Annealing Temperature'], exponential_scale_data['Exponential Scale Factor'], guess)
fitted_exponential =  np.exp(params_exponential[0]*(1/(fitted_exponential_scale_data+273.15) - 1/params_exponential[1])) 


print(params_exponential)



#%%

# Plotting For 40C 

# Plot
plt.figure(figsize=(6, 4))
x = range(len(Annealing_40C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(fitted_annealing_40C, fitted_exponential_40C_raw, label = 'Fitted Data ')
plt.plot(Annealing_40C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_40C_Shifted_and_Scaled_data['Pin107 Vf (10 min R irad)'], '.-', label = 'Pin107 Vf (10 min R irad)')
plt.plot(fitted_annealing_40C, fitted_exponential_40C, label = r'Fitted Data From Function For 100C Fitting' )
plt.xlabel(r'Annealing Time $(min)$')
plt.ylabel(r'Normalized Forward Voltage $V_f$')
plt.title(r'40C Annealing')
plt.legend(loc = 'best')
plt.show()

# Plotting For 50C 

# Plot
plt.figure(figsize=(6, 4))
x = range(len(Annealing_50C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(fitted_annealing_50C, fitted_exponential_50C_raw, label = 'Fitted Data ')
plt.plot(Annealing_50C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_50C_Shifted_and_Scaled_data['Pin109 Vf (10 min R irad)'], '.-', label = 'Pin109 Vf (10 min R irad)')
plt.plot(fitted_annealing_50C, fitted_exponential_50C, label = r'Fitted Data From Function For 100C Fitting' )
plt.xlabel(r'Annealing Time $(min)$')
plt.ylabel(r'Normalized Forward Voltage $V_f$')
plt.title(r'50C Annealing')
plt.legend(loc = 'best')
plt.show()

# Plotting For 60C 

# Plot
plt.figure(figsize=(6, 4))
x = range(len(Annealing_60C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(fitted_annealing_60C, fitted_exponential_60C_raw, label = 'Fitted Data ')
plt.plot(Annealing_60C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_60C_Shifted_and_Scaled_data['Pin105 Vf (10 min R irad)'], '.-', label = 'Pin105 Vf (10 min R irad)')
plt.plot(fitted_annealing_60C, fitted_exponential_60C, label = r'Fitted Data From Function For 100C Fitting' )
plt.xlabel(r'Annealing Time $(min)$')
plt.ylabel(r'Normalized Folding Voltage $V_f$')
plt.title(r'60C Annealing')
plt.legend(loc = 'best')
plt.show()


# Plotting For 70C 

# Plot
plt.figure(figsize=(6, 4))
x = range(len(Annealing_70C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(fitted_annealing_70C, fitted_exponential_70C_raw, label = 'Fitted Data ')
plt.plot(Annealing_70C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_70C_Shifted_and_Scaled_data['Pin110 Vf (10 min R irad)'], '.-', label = 'Pin110 Vf (10 min R irad)')
plt.plot(fitted_annealing_70C, fitted_exponential_70C, label = r'Fitted Data From Function For 100C Fitting' )
plt.xlabel(r'Annealing Time $(min)$')
plt.ylabel(r'Normalized Forward Voltage $V_f$')
plt.title(r'70C Annealing')
plt.legend(loc = 'best')
plt.show()



# Plotting For 80C 

# Plot
plt.figure(figsize=(6, 4))
x = range(len(Annealing_80C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(fitted_annealing_80C, fitted_exponential_80C_raw, label = 'Fitted Data ')
plt.plot(Annealing_80C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_80C_Shifted_and_Scaled_data['Pin108 Vf (10 min R irad)'], '.-', label = 'Pin108 Vf (10 min R irad)')
plt.plot(fitted_annealing_80C, fitted_exponential_80C, label = r'Fitted Data From Function For 100C Fitting' )
plt.xlabel(r'Annealing Time $(min)$')
plt.ylabel(r'Normalized Forward Voltage $V_f$')
plt.title(r'80C Annealing')
plt.legend(loc = 'best')
plt.show()


# Plotting For 90C 

# Plot
plt.figure(figsize=(6, 4))
x = range(len(Annealing_90C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(fitted_annealing_90C, fitted_exponential_90C_raw, label = 'Fitted Data ')
plt.plot(Annealing_90C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_90C_Shifted_and_Scaled_data['Pin111 Vf (10 min R irad)'], '.-', label = 'Pin111 Vf (10 min R irad)')
plt.plot(fitted_annealing_90C, fitted_exponential_90C, label = r'Fitted Data From Function For 100C Fitting' )
plt.xlabel(r'Annealing Time $(min)$')
plt.ylabel(r'Normalized Forward Voltage $V_f$')
plt.title(r'90C Annealing')
plt.legend(loc = 'best')
plt.show()


# Plotting For 100C 

# Plot
plt.figure(figsize=(6, 4))
x = range(len(Annealing_100C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(fitted_annealing_100C, fitted_exponential_100C_raw, label = 'Fitted Data ')
plt.plot(Annealing_100C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_100C_Shifted_and_Scaled_data['Pin106 Vf (10 min R irad)'], '.-', label = 'Pin106 Vf (10 min R irad)')
plt.plot(fitted_annealing_100C, fitted_exponential_100C, label = 'Fitted Data')
plt.xlabel(r'Annealing Time $(min)$')
plt.ylabel(r'Normalized Forward Voltage $V_f$')
plt.title(r'100C Annealing')
plt.legend(loc = 'best')
plt.show()

#%%

# Plotting For Expoential Scaling

# Plot
plt.figure(figsize=(6, 4))
plt.plot(exponential_scale_data['Annealing Temperature'], exponential_scale_data['Exponential Scale Factor'], '.-', label = r'Expoenential Scale Data')
plt.plot(fitted_exponential_scale_data, fitted_exponential, label = 'Fitted Exponential     ' + r'$e^{(-16655.25*(1/(T+237.15) - 1/372.83))} $')
plt.xlabel(r'Annealing Temperature $(C)$')
plt.ylabel(r'Exponential Scaling')
plt.title(r'Expoenential Scale Factor vs. Temperature $(C)$')
plt.legend(loc = 'best')
plt.show()


#%%

print(params_exponential)

#%%
print(params_40C)
print(params_50C)
print(params_60C)
print(params_70C)
print(params_80C)
print(params_90C)
print(params_100C)

#%%

# Plotting For All Fits

# Plot
plt.figure(figsize=(6, 4))
x = range(len(Annealing_100C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(fitted_annealing_100C, fitted_exponential_100C_raw, label = '100C')
plt.plot(fitted_annealing_90C, fitted_exponential_90C_raw, label = '90C')
plt.plot(fitted_annealing_80C, fitted_exponential_80C_raw, label = '80C')
plt.plot(fitted_annealing_70C, fitted_exponential_70C_raw, label = '70C')
plt.plot(fitted_annealing_60C, fitted_exponential_60C_raw, label = '60C')
plt.plot(fitted_annealing_50C, fitted_exponential_50C_raw, label = '50C')
plt.plot(fitted_annealing_40C, fitted_exponential_40C_raw, label = '40C')
plt.xlabel(r'Annealing Time $(min)$')
plt.ylabel(r'Normalized Forward Voltage $V_f$')
plt.title(r'Normalized Fit')
plt.legend(loc = 'best')
plt.show()

#%%

# Plotting For All Fits Log

# Plot
plt.figure(figsize=(6, 4))
x = range(len(Annealing_100C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(fitted_annealing_100C, fitted_exponential_100C_raw, label = '100C')
plt.plot(fitted_annealing_90C, fitted_exponential_90C_raw, label = '90C')
plt.plot(fitted_annealing_80C, fitted_exponential_80C_raw, label = '80C')
plt.plot(fitted_annealing_70C, fitted_exponential_70C_raw, label = '70C')
plt.plot(fitted_annealing_60C, fitted_exponential_60C_raw, label = '60C')
plt.plot(fitted_annealing_50C, fitted_exponential_50C_raw, label = '50C')
plt.plot(fitted_annealing_40C, fitted_exponential_40C_raw, label = '40C')
plt.xlabel(r'Annealing Time $(min)$')
plt.ylabel(r'Normalized Forward Voltage $V_f$')
plt.title(r'Normalized Fit Log')
plt.legend(loc = 'best')
plt.xscale('log')
plt.show()


#%%

# Plotting For All Data

# Plot
plt.figure(figsize=(6, 4))
x = range(len(Annealing_100C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(Annealing_40C_Shifted_data['Annealing Time (Min)'], Annealing_40C_Shifted_data['Pin107 Vf (10 min R irad)'], '.-', label = '40C')
plt.plot(Annealing_50C_Shifted_data['Annealing Time (Min)'], Annealing_50C_Shifted_data['Pin109 Vf (10 min R irad)'], '.-', label = '50C')
plt.plot(Annealing_60C_Shifted_data['Annealing Time (Min)'], Annealing_60C_Shifted_data['Pin105 Vf (10 min R irad)'], '.-', label = '60C')
plt.plot(Annealing_70C_Shifted_data['Annealing Time (Min)'], Annealing_70C_Shifted_data['Pin110 Vf (10 min R irad)'], '.-', label = '70C')
plt.plot(Annealing_80C_Shifted_data['Annealing Time (Min)'], Annealing_80C_Shifted_data['Pin108 Vf (10 min R irad)'], '.-', label = '80C')
plt.plot(Annealing_90C_Shifted_data['Annealing Time (Min)'], Annealing_90C_Shifted_data['Pin111 Vf (10 min R irad)'], '.-', label = '90C')
plt.plot(Annealing_100C_Shifted_data['Annealing Time (Min)'], Annealing_100C_Shifted_data['Pin106 Vf (10 min R irad)'], '.-', label = '100C')
plt.xlabel(r'Annealing Time $(min)$')
plt.ylabel(r'Normalized Forward Voltage $V_f$')
plt.title(r'Raw Data')
plt.legend(loc = 'best')
plt.show()


#%%

# Plotting For All Scaled Data Log

# Plot
plt.figure(figsize=(6, 4))
x = range(len(Annealing_100C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(Annealing_40C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_40C_Shifted_and_Scaled_data['Pin107 Vf (10 min R irad)'], '.-', label = '40C')
plt.plot(Annealing_50C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_50C_Shifted_and_Scaled_data['Pin109 Vf (10 min R irad)'], '.-', label = '50C')
plt.plot(Annealing_60C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_60C_Shifted_and_Scaled_data['Pin105 Vf (10 min R irad)'], '.-', label = '60C')
plt.plot(Annealing_70C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_70C_Shifted_and_Scaled_data['Pin110 Vf (10 min R irad)'], '.-', label = '70C')
plt.plot(Annealing_80C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_80C_Shifted_and_Scaled_data['Pin108 Vf (10 min R irad)'], '.-', label = '80C')
plt.plot(Annealing_90C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_90C_Shifted_and_Scaled_data['Pin111 Vf (10 min R irad)'], '.-', label = '90C')
plt.plot(Annealing_100C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_100C_Shifted_and_Scaled_data['Pin106 Vf (10 min R irad)'], '.-', label = '100C')
plt.xlabel(r'Annealing Time $(min)$')
plt.ylabel(r'Normalized Forward Voltage $V_f$')
plt.title(r'Normalized Raw Data Log')
plt.legend(loc = 'best')
plt.xscale('log')
plt.show()

#%%

# Plotting For All Scaled Data

# Plot
plt.figure(figsize=(6, 4))
x = range(len(Annealing_100C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(Annealing_40C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_40C_Shifted_and_Scaled_data['Pin107 Vf (10 min R irad)'], '.-', label = '40C')
plt.plot(Annealing_50C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_50C_Shifted_and_Scaled_data['Pin109 Vf (10 min R irad)'], '.-', label = '50C')
plt.plot(Annealing_60C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_60C_Shifted_and_Scaled_data['Pin105 Vf (10 min R irad)'], '.-', label = '60C')
plt.plot(Annealing_70C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_70C_Shifted_and_Scaled_data['Pin110 Vf (10 min R irad)'], '.-', label = '70C')
plt.plot(Annealing_80C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_80C_Shifted_and_Scaled_data['Pin108 Vf (10 min R irad)'], '.-', label = '80C')
plt.plot(Annealing_90C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_90C_Shifted_and_Scaled_data['Pin111 Vf (10 min R irad)'], '.-', label = '90C')
plt.plot(Annealing_100C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_100C_Shifted_and_Scaled_data['Pin106 Vf (10 min R irad)'], '.-', label = '100C')
plt.xlabel(r'Annealing Time $(min)$')
plt.ylabel(r'Normalized Forward Voltage $V_f$')
plt.title(r'Normalized Raw Data')
plt.legend(loc = 'best')
plt.show()



#%%

# Plotting For All Shifted Data Log

# Plot
plt.figure(figsize=(6, 4))
x = range(len(Annealing_100C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(Annealing_40C_Shifted_data['Annealing Time (Min)'], Annealing_40C_Shifted_data['Pin107 Vf (10 min R irad)'], '.-', label = '40C')
plt.plot(Annealing_50C_Shifted_data['Annealing Time (Min)'], Annealing_50C_Shifted_data['Pin109 Vf (10 min R irad)'], '.-', label = '50C')
plt.plot(Annealing_60C_Shifted_data['Annealing Time (Min)'], Annealing_60C_Shifted_data['Pin105 Vf (10 min R irad)'], '.-', label = '60C')
plt.plot(Annealing_70C_Shifted_data['Annealing Time (Min)'], Annealing_70C_Shifted_data['Pin110 Vf (10 min R irad)'], '.-', label = '70C')
plt.plot(Annealing_80C_Shifted_data['Annealing Time (Min)'], Annealing_80C_Shifted_data['Pin108 Vf (10 min R irad)'], '.-', label = '80C')
plt.plot(Annealing_90C_Shifted_data['Annealing Time (Min)'], Annealing_90C_Shifted_data['Pin111 Vf (10 min R irad)'], '.-', label = '90C')
plt.plot(Annealing_100C_Shifted_data['Annealing Time (Min)'], Annealing_100C_Shifted_data['Pin106 Vf (10 min R irad)'], '.-', label = '100C')
plt.xlabel(r'Annealing Time $(min)$')
plt.ylabel(r'Forward Voltage $V_f$')
plt.title(r'Raw Data Log')
plt.legend(loc = 'best')
plt.xscale('log')
plt.show()


#%%

# Plotting For All Data 40C

# Plot
plt.figure(figsize=(6, 4))
x = range(len(Annealing_100C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(all_data_40C['Annealing Time (Min) 1'], all_data_40C['Pin2 (Rabbit 20 m)'], '.-', label = 'Pin2 (Rabbit 20 m)')
plt.plot(all_data_40C['Annealing Time (Min) 2'], all_data_40C['Pin4 (BP 10 m)'], '.-', label = 'Pin4 (BP 10 m)')
plt.plot(all_data_40C['Annealing Time (Min) 3'], all_data_40C['Pin6 (BP 180 m)'], '.-', label = 'Pin6 (BP 180 m)')
plt.plot(all_data_40C['Annealing Time (Min) 4'], all_data_40C['Pin12 (Rabbit 102 m)'], '.-', label = 'Pin12 (Rabbit 102 m)')
plt.plot(all_data_40C['Annealing Time (Min) 5'], all_data_40C['Pin26 (BP 38 m)'], '.-', label = 'Pin26 (BP 38 m)')
plt.plot(all_data_40C['Annealing Time (Min) 6'], all_data_40C['Pin33 (Rabbit 25 m)'], '.-', label = 'Pin33 (Rabbit 25 m)')
plt.plot(all_data_40C['Annealing Time (Min) 7'], all_data_40C['Pin47 (Rabbit 139 m)'], '.-', label = 'Pin47 (Rabbit 139 m)')
plt.plot(all_data_40C['Annealing Time (Min) 8'], all_data_40C['Pin107 Vf (10 min R irad)'], '.-', label = 'Pin107 Vf (10 min R irad)')
plt.xlabel(r'Annealing Time $(min)$')
plt.ylabel(r'Forward Voltage $V_f$')
plt.title(r'40C Raw Data')
plt.legend(loc = 'best')
plt.show()

# Plotting For All Data 50C

# Plot
plt.figure(figsize=(6, 4))
x = range(len(Annealing_100C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(all_data_50C['Annealing Time (Min) 1'], all_data_50C['Pin40 Vf (30 min R irad)'], '.-', label = 'Pin40 Vf (30 min R irad)')
plt.plot(all_data_50C['Annealing Time (Min) 2'], all_data_50C['Pin52 Vf (60 min R irad)'], '.-', label = 'Pin52 Vf (60 min R irad)')
plt.plot(all_data_50C['Annealing Time (Min) 3'], all_data_50C['Pin109 Vf (10 min R irad)'], '.-', label = 'Pin109 Vf (10 min R irad)')
plt.xlabel(r'Annealing Time $(min)$')
plt.ylabel(r'Forward Voltage $V_f$')
plt.title(r'50C Raw Data')
plt.legend(loc = 'best')
plt.show()


# Plotting For All Data 60C

# Plot
plt.figure(figsize=(6, 4))
x = range(len(Annealing_100C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(all_data_60C['Annealing Time (Min) 1'], all_data_60C['Pin20 Vf (30 min R irad)'], '.-', label = 'Pin20 Vf (30 min R irad)')
plt.plot(all_data_60C['Annealing Time (Min) 2'], all_data_60C['Pin49 Vf (60 min R irad)'], '.-', label = 'Pin49 Vf (60 min R irad)')
plt.plot(all_data_60C['Annealing Time (Min) 3'], all_data_60C['Pin105 Vf (10 min R irad)'], '.-', label = 'Pin105 Vf (10 min R irad)')
plt.xlabel(r'Annealing Time $(min)$')
plt.ylabel(r'Forward Voltage $V_f$')
plt.title(r'60C Raw Data')
plt.legend(loc = 'best')
plt.show()


# Plotting For All Data 70C

# Plot
plt.figure(figsize=(6, 4))
x = range(len(Annealing_100C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(all_data_70C['Annealing Time (Min) 1'], all_data_70C['Pin42 Vf (30 min R)'], '.-', label = 'Pin42 Vf (30 min R)')
plt.plot(all_data_70C['Annealing Time (Min) 2'], all_data_70C['Pin51 Vf (60 min R)'], '.-', label = 'Pin51 Vf (60 min R)')
plt.plot(all_data_70C['Annealing Time (Min) 3'], all_data_70C['Pin110 Vf (10 min R)'], '.-', label = 'Pin110 Vf (10 min R)')
plt.xlabel(r'Annealing Time $(min)$')
plt.ylabel(r'Forward Voltage $V_f$')
plt.title(r'70C Raw Data')
plt.legend(loc = 'best')
plt.show()


# Plotting For All Data 80C

# Plot
plt.figure(figsize=(6, 4))
x = range(len(Annealing_100C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(all_data_80C['Annealing Time (Min) 1'], all_data_80C['Pin3 Vf (20 min R irad)'], '.-', label = 'Pin3 Vf (20 min R irad)')
plt.plot(all_data_80C['Annealing Time (Min) 2'], all_data_80C['Pin5 Vf (10 min BP irad)'], '.-', label = 'Pin5 Vf (10 min BP irad)')
plt.plot(all_data_80C['Annealing Time (Min) 3'], all_data_80C['Pin7 Vf (180 min BP irad)'], '.-', label = 'Pin7 Vf (180 min BP irad)')
plt.plot(all_data_80C['Annealing Time (Min) 4'], all_data_80C['Pin14 Vf (102 min R irad)'], '.-', label = 'Pin14 Vf (102 min R irad)')
plt.plot(all_data_80C['Annealing Time (Min) 5'], all_data_80C['Pin27 Vf (38 min BP irad)'], '.-', label = 'Pin27 Vf (38 min BP irad)')
plt.plot(all_data_80C['Annealing Time (Min) 6'], all_data_80C['Pin32 Vf (25 min R irad)'], '.-', label = 'Pin32 Vf (25 min R irad)')
plt.plot(all_data_80C['Annealing Time (Min) 7'], all_data_80C['Pin45 Vf (60 min R irad)'], '.-', label = 'Pin45 Vf (60 min R irad)')
plt.plot(all_data_80C['Annealing Time (Min) 8'], all_data_80C['Pin48 Vf (139 min R irad)'], '.-', label = 'Pin48 Vf (139 min R irad)')
plt.plot(all_data_80C['Annealing Time (Min) 9'], all_data_80C['Pin108 Vf (10 min R irad)'], '.-', label = 'Pin108 Vf (10 min R irad)')
plt.xlabel(r'Annealing Time $(min)$')
plt.ylabel(r'Forward Voltage $V_f$')
plt.title(r'80C Raw Data')
plt.legend(loc = 'best')
plt.show()


# Plotting For All Data 90C

# Plot
plt.figure(figsize=(6, 4))
x = range(len(Annealing_100C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(all_data_90C['Annealing Time (Min) 1'], all_data_90C['Pin41 Vf (30 min R irad)'], '.-', label = 'Pin41 Vf (30 min R irad)')
plt.plot(all_data_90C['Annealing Time (Min) 2'], all_data_90C['Pin50 Vf (60 min R irad)'], '.-', label = 'Pin50 Vf (60 min R irad)')
plt.plot(all_data_90C['Annealing Time (Min) 3'], all_data_90C['Pin111 Vf (10 min R irad)'], '.-', label = 'Pin111 Vf (10 min R irad)')
plt.xlabel(r'Annealing Time $(min)$')
plt.ylabel(r'Forward Voltage $V_f$')
plt.title(r'90C Raw Data')
plt.legend(loc = 'best')
plt.show()


# Plotting For All Data 90C

# Plot
plt.figure(figsize=(6, 4))
x = range(len(Annealing_100C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(all_data_100C['Annealing Time (Min) 1'], all_data_100C['Pin20 Vf (30 min R irad)'], '.-', label = 'Pin20 Vf (30 min R irad)')
plt.plot(all_data_100C['Annealing Time (Min) 2'], all_data_100C['Pin49 Vf (60 min R irad)'], '.-', label = 'Pin49 Vf (60 min R irad)')
plt.plot(all_data_100C['Annealing Time (Min) 3'], all_data_100C['Pin106 Vf (10 min R irad)'], '.-', label = 'Pin106 Vf (10 min R irad)')
plt.xlabel(r'Annealing Time $(min)$')
plt.ylabel(r'Forward Voltage $V_f$')
plt.title(r'100C Raw Data')
plt.legend(loc = 'best')
plt.show()
































































































































































































