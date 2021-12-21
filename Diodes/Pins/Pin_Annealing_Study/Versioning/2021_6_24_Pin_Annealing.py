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
len_100C = len(Annealing_100C_Shifted_and_Scaled_data['Annealing Time (Min)']) - 1
len_exponential_scale = len(exponential_scale_data['Annealing Temperature']) - 1

#%%


# Curve Fitting 

# Fitting Fuction for One Exponential
def exponential_1(z, A, g, a, d, H, l, j):
    return  A*np.exp(-g*(z-a)) + d + H*np.exp(-l*(z-j))

guess = np.array([ 1,  .001, -10,  0.1, .1,  .1,  10])
guess_40C = np.array([5.77901281e-01, 5.58476585e-04, 3.19323676e+00, 3.97150926e-01, 6.86790541e-02, 1.57796934e-01, 2.86449386e+00])
#%%
fitted_annealing_40C = np.linspace(Annealing_40C_Shifted_and_Scaled_data['Annealing Time (Min)'][0], Annealing_40C_Shifted_and_Scaled_data['Annealing Time (Min)'][len_40C], 1000)
params_40C, params_covariance_40C = curve_fit(exponential_1, Annealing_40C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_40C_Shifted_and_Scaled_data['Pin107 Vf (10 min R irad)'], guess_40C)
fitted_exponential_40C = params_40C[0]*np.exp(-1*params_40C[1]*(fitted_annealing_40C - params_40C[2])) + params_40C[3] + params_40C[4]*np.exp(-1*params_40C[5]*(fitted_annealing_40C - params_40C[6]))
first_fitted_exponential_40C = params_40C[0]*np.exp(-1*params_40C[1]*(fitted_annealing_40C - params_40C[2])) + params_40C[3] 
second_fitted_exponential_40C =   params_40C[4]*np.exp(-1*params_40C[5]*(fitted_annealing_40C - params_40C[6])) + params_40C[3] 
#%%
fitted_annealing_50C = np.linspace(Annealing_50C_Shifted_and_Scaled_data['Annealing Time (Min)'][0], Annealing_50C_Shifted_and_Scaled_data['Annealing Time (Min)'][len_50C], 1000)
params_50C, params_covariance_50C = curve_fit(exponential_1, Annealing_50C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_50C_Shifted_and_Scaled_data['Pin109 Vf (10 min R irad)'])
fitted_exponential_50C = params_50C[0]*np.exp(-1*params_50C[1]*(fitted_annealing_50C - params_50C[2])) + params_50C[3] + params_50C[4]*np.exp(-1*params_50C[5]*(fitted_annealing_50C - params_50C[6]))
first_fitted_exponential_50C = params_50C[0]*np.exp(-1*params_50C[1]*(fitted_annealing_50C - params_50C[2])) + params_50C[3] 
second_fitted_exponential_50C =   params_50C[4]*np.exp(-1*params_50C[5]*(fitted_annealing_50C - params_50C[6])) + params_50C[3] 

fitted_annealing_60C = np.linspace(Annealing_60C_Shifted_and_Scaled_data['Annealing Time (Min)'][0], Annealing_60C_Shifted_and_Scaled_data['Annealing Time (Min)'][len_60C], 1000)
params_60C, params_covariance_60C = curve_fit(exponential_1, Annealing_60C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_60C_Shifted_and_Scaled_data['Pin105 Vf (10 min R irad)'])
fitted_exponential_60C = params_60C[0]*np.exp(-1*params_60C[1]*(fitted_annealing_60C - params_60C[2])) + params_60C[3] + params_60C[4]*np.exp(-1*params_60C[5]*(fitted_annealing_60C - params_60C[6])) 
first_fitted_exponential_60C = params_60C[0]*np.exp(-1*params_60C[1]*(fitted_annealing_60C - params_60C[2])) + params_60C[3] 
second_fitted_exponential_60C = params_60C[3] + params_60C[4]*np.exp(-1*params_60C[5]*(fitted_annealing_60C - params_60C[6])) 

fitted_annealing_70C = np.linspace(Annealing_70C_Shifted_and_Scaled_data['Annealing Time (Min)'][0], Annealing_70C_Shifted_and_Scaled_data['Annealing Time (Min)'][len_70C], 1000)
params_70C, params_covariance_70C = curve_fit(exponential_1, Annealing_70C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_70C_Shifted_and_Scaled_data['Pin110 Vf (10 min R irad)'], guess)
fitted_exponential_70C = params_70C[0]*np.exp(-1*params_70C[1]*(fitted_annealing_70C - params_70C[2])) + params_70C[3] + params_70C[4]*np.exp(-1*params_70C[5]*(fitted_annealing_70C - params_70C[6]))
first_fitted_exponential_70C = params_70C[0]*np.exp(-1*params_70C[1]*(fitted_annealing_70C - params_70C[2])) + params_70C[3] 
second_fitted_exponential_70C =   params_70C[4]*np.exp(-1*params_70C[5]*(fitted_annealing_70C - params_70C[6])) + params_70C[3] 

fitted_annealing_80C = np.linspace(Annealing_80C_Shifted_and_Scaled_data['Annealing Time (Min)'][0], Annealing_80C_Shifted_and_Scaled_data['Annealing Time (Min)'][len_80C], 1000)
params_80C, params_covariance_80C = curve_fit(exponential_1, Annealing_80C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_80C_Shifted_and_Scaled_data['Pin108 Vf (10 min R irad)'])
fitted_exponential_80C = params_80C[0]*np.exp(-1*params_80C[1]*(fitted_annealing_80C - params_80C[2])) + params_80C[3] + params_80C[4]*np.exp(-1*params_80C[5]*(fitted_annealing_80C - params_80C[6]))
first_fitted_exponential_80C = params_80C[0]*np.exp(-1*params_80C[1]*(fitted_annealing_80C - params_80C[2])) + params_80C[3]
second_fitted_exponential_80C = params_80C[3] + params_80C[4]*np.exp(-1*params_80C[5]*(fitted_annealing_80C - params_80C[6]))

fitted_annealing_90C = np.linspace(Annealing_90C_Shifted_and_Scaled_data['Annealing Time (Min)'][0], Annealing_90C_Shifted_and_Scaled_data['Annealing Time (Min)'][len_90C], 1000)
params_90C, params_covariance_90C = curve_fit(exponential_1, Annealing_90C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_90C_Shifted_and_Scaled_data['Pin111 Vf (10 min R irad)'])
fitted_exponential_90C = params_90C[0]*np.exp(-1*params_90C[1]*(fitted_annealing_90C - params_90C[2])) + params_90C[3] + params_90C[4]*np.exp(-1*params_90C[5]*(fitted_annealing_90C - params_90C[6]))
first_fitted_exponential_90C = params_90C[0]*np.exp(-1*params_90C[1]*(fitted_annealing_90C - params_90C[2])) + params_90C[3] 
second_fitted_exponential_90C =   params_90C[4]*np.exp(-1*params_90C[5]*(fitted_annealing_90C - params_90C[6])) + params_90C[3] 

fitted_annealing_100C = np.linspace(Annealing_100C_Shifted_and_Scaled_data['Annealing Time (Min)'][0], Annealing_100C_Shifted_and_Scaled_data['Annealing Time (Min)'][len_100C], 1000)
params_100C, params_covariance_100C = curve_fit(exponential_1, Annealing_100C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_100C_Shifted_and_Scaled_data['Pin106 Vf (10 min R irad)'], guess)
fitted_exponential_100C = params_100C[0]*np.exp(-1*params_100C[1]*(fitted_annealing_100C - params_100C[2])) + params_100C[3] + params_100C[4]*np.exp(-1*params_100C[5]*(fitted_annealing_100C - params_100C[6]))
first_fitted_exponential_100C = params_100C[0]*np.exp(-1*params_100C[1]*(fitted_annealing_100C - params_100C[2])) + params_100C[3] 
second_fitted_exponential_100C = params_100C[3] + params_100C[4]*np.exp(-1*params_100C[5]*(fitted_annealing_100C - params_100C[6]))


#%%

# Curve Fitting Using 100C Annealing Parameters

# Fitting Fuction for Two Exponential
def exponential_2(t, a, b):
    return  0.231583*np.exp(-0.00604421*(a*(t - b)+31.6815)) + 0.264799 + 0.150671*np.exp(-0.116058*(a*(t - b)-18.459))
guess = np.array([1, 0])

fitted_annealing_40C = np.linspace(Annealing_40C_Shifted_and_Scaled_data['Annealing Time (Min)'][0], Annealing_40C_Shifted_and_Scaled_data['Annealing Time (Min)'][len_40C], 1000)
params_40C, params_covariance_40C = curve_fit(exponential_2, Annealing_40C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_40C_Shifted_and_Scaled_data['Pin107 Vf (10 min R irad)'])
fitted_exponential_40C = 0.231583*np.exp(-0.00604421*(params_40C[0]*(fitted_annealing_40C - params_40C[1]) + 31.6815)) + 0.264799 + 0.150671*np.exp(-0.116058*(params_40C[0]*(fitted_annealing_40C - params_40C[1])-18.459))                                     

fitted_annealing_50C = np.linspace(Annealing_50C_Shifted_and_Scaled_data['Annealing Time (Min)'][0], Annealing_50C_Shifted_and_Scaled_data['Annealing Time (Min)'][len_50C], 1000)
params_50C, params_covariance_50C = curve_fit(exponential_2, Annealing_50C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_50C_Shifted_and_Scaled_data['Pin109 Vf (10 min R irad)'])
fitted_exponential_50C = 0.231583*np.exp(-0.00604421*(params_50C[0]*(fitted_annealing_50C - params_50C[1]) + 31.6815)) + 0.264799 + 0.150671*np.exp(-0.116058*(params_50C[0]*(fitted_annealing_50C - params_50C[1])-18.459))         

fitted_annealing_60C = np.linspace(Annealing_60C_Shifted_and_Scaled_data['Annealing Time (Min)'][0], Annealing_60C_Shifted_and_Scaled_data['Annealing Time (Min)'][len_60C], 1000)
params_60C, params_covariance_60C = curve_fit(exponential_2, Annealing_60C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_60C_Shifted_and_Scaled_data['Pin105 Vf (10 min R irad)'])
fitted_exponential_60C = 0.231583*np.exp(-0.00604421*(params_60C[0]*(fitted_annealing_60C - params_60C[1]) + 31.6815)) + 0.264799 + 0.150671*np.exp(-0.116058*(params_60C[0]*(fitted_annealing_60C - params_60C[1])-18.459))

fitted_annealing_70C = np.linspace(Annealing_70C_Shifted_and_Scaled_data['Annealing Time (Min)'][0], Annealing_70C_Shifted_and_Scaled_data['Annealing Time (Min)'][len_70C], 1000)
params_70C, params_covariance_70C = curve_fit(exponential_2, Annealing_70C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_70C_Shifted_and_Scaled_data['Pin110 Vf (10 min R irad)'])
fitted_exponential_70C = 0.231583*np.exp(-0.00604421*(params_70C[0]*(fitted_annealing_70C - params_70C[1]) + 31.6815)) + 0.264799 + 0.150671*np.exp(-0.116058*(params_70C[0]*(fitted_annealing_70C - params_70C[1])-18.459))         

fitted_annealing_80C = np.linspace(Annealing_80C_Shifted_and_Scaled_data['Annealing Time (Min)'][0], Annealing_80C_Shifted_and_Scaled_data['Annealing Time (Min)'][len_80C], 1000)
params_80C, params_covariance_80C = curve_fit(exponential_2, Annealing_80C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_80C_Shifted_and_Scaled_data['Pin108 Vf (10 min R irad)'])
fitted_exponential_80C = 0.231583*np.exp(-0.00604421*(params_80C[0]*(fitted_annealing_80C - params_80C[1]) + 31.6815)) + 0.264799 + 0.150671*np.exp(-0.116058*(params_80C[0]*(fitted_annealing_80C - params_80C[1])-18.459))

fitted_annealing_90C = np.linspace(Annealing_90C_Shifted_and_Scaled_data['Annealing Time (Min)'][0], Annealing_90C_Shifted_and_Scaled_data['Annealing Time (Min)'][len_90C], 1000)
params_90C, params_covariance_90C = curve_fit(exponential_2, Annealing_90C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_90C_Shifted_and_Scaled_data['Pin111 Vf (10 min R irad)'])
fitted_exponential_90C = 0.231583*np.exp(-0.00604421*(params_90C[0]*(fitted_annealing_90C - params_90C[1]) + 31.6815)) + 0.264799 + 0.150671*np.exp(-0.116058*(params_90C[0]*(fitted_annealing_90C - params_90C[1])-18.459))         

fitted_annealing_100C = np.linspace(Annealing_100C_Shifted_and_Scaled_data['Annealing Time (Min)'][0], Annealing_100C_Shifted_and_Scaled_data['Annealing Time (Min)'][len_100C], 1000)
params_100C, params_covariance_100C = curve_fit(exponential_2, Annealing_100C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_100C_Shifted_and_Scaled_data['Pin106 Vf (10 min R irad)'], guess)
fitted_exponential_100C = 0.231583*np.exp(-0.00604421*(params_100C[0]*(fitted_annealing_100C - params_100C[1]) + 31.6815)) + 0.264799 + 0.150671*np.exp(-0.116058*(params_100C[0]*(fitted_annealing_100C - params_100C[1])-18.459)) 


#%%
# Curve Fitting for Expoenetial Scaling Inverse Kelvin

# Fitting Fuction for Exponential
def exponential(t, a, b):
    return  np.exp(a*(1/(t+273.15)-1/b))

guess = np.array([-13478, 373.15])

fitted_exponential_scale_data = np.linspace(exponential_scale_data['Annealing Temperature'][0], exponential_scale_data['Exponential Scale Factor'][len_exponential_scale], 1000)
params_exponential, params_covariance_exponential = curve_fit(exponential, exponential_scale_data['Annealing Temperature'], exponential_scale_data['Exponential Scale Factor'], guess)
fitted_exponential =  np.exp(params_exponential[0]*(1/(fitted_exponential_scale_data+273.15) - 1/params_exponential[1])) 


print(params_exponential)







#%%
# Curve Fitting for Expoenetial Scaling

# Fitting Fuction for Exponential
def exponential(t, a, b):
    return  np.exp(a*(t-b)) 


fitted_exponential_scale_data = np.linspace(exponential_scale_data['Annealing Temperature'][0], exponential_scale_data['Exponential Scale Factor'][len_exponential_scale], 1000)
params_exponential, params_covariance_exponential = curve_fit(exponential, exponential_scale_data['Annealing Temperature'], exponential_scale_data['Exponential Scale Factor'])
fitted_exponential =  np.exp(params_exponential[0]*(fitted_exponential_scale_data - params_exponential[1])) 


print(params_exponential[1])




#%%
# Plotting For 40C

# Plot
plt.figure(figsize=(11, 8))
x = range(len(Annealing_40C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(Annealing_40C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_40C_Shifted_and_Scaled_data['Pin107 Vf (10 min R irad)'], label = 'Pin107 Vf (10 min R irad)')
plt.plot(fitted_annealing_40C, fitted_exponential_40C, label = 'Fitted Data ')
plt.plot(fitted_annealing_40C, first_fitted_exponential_40C, label = 'First Exponential of Fitted Data ')
plt.plot(fitted_annealing_40C, second_fitted_exponential_40C, label = 'Second Exponential of Fitted Data ')
plt.xlabel(r'Shifted Annealing Time $(min)$')
plt.ylabel(r'Normalized Folding Voltage $V_f$')
plt.title(r'40C Annealing')
plt.legend(loc = 'best')
plt.show()


#%%
# Plotting For 50C

# Plot
plt.figure(figsize=(11, 8))
x = range(len(Annealing_50C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(Annealing_50C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_50C_Shifted_and_Scaled_data['Pin109 Vf (10 min R irad)'], label = 'Pin109 Vf (10 min R irad)')
plt.plot(fitted_annealing_50C, fitted_exponential_50C, label = 'Fitted Data ')
plt.plot(fitted_annealing_50C, first_fitted_exponential_50C, label = 'First Exponential of Fitted Data ')
plt.plot(fitted_annealing_50C, second_fitted_exponential_50C, label = 'Second Exponential of Fitted Data ')
plt.xlabel(r'Shifted Annealing Time $(min)$')
plt.ylabel(r'Normalized Folding Voltage $V_f$')
plt.title(r'50C Annealing')
plt.legend(loc = 'best')
plt.show()


#%%

# Plotting For 60C

# Plot
plt.figure(figsize=(11, 8))
x = range(len(Annealing_60C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(Annealing_60C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_60C_Shifted_and_Scaled_data['Pin105 Vf (10 min R irad)'], label = 'Pin105 Vf (10 min R irad)')
plt.plot(fitted_annealing_60C, fitted_exponential_60C, label = 'Fitted Data ')
plt.plot(fitted_annealing_60C, first_fitted_exponential_60C, label = 'First Exponential of Fitted Data ' )
plt.plot(fitted_annealing_60C, second_fitted_exponential_60C, label = 'Second Exponential of Fitted Data ' )
plt.xlabel(r'Shifted Annealing Time $(min)$')
plt.ylabel(r'Normalized Folding Voltage $V_f$')
plt.title(r'60C Annealing')
plt.legend(loc = 'best')
plt.show()


#%%
# Plotting For 70C

# Plot
plt.figure(figsize=(11, 8))
x = range(len(Annealing_70C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(Annealing_70C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_70C_Shifted_and_Scaled_data['Pin110 Vf (10 min R irad)'], label = 'Pin110 Vf (10 min R irad)')
plt.plot(fitted_annealing_70C, fitted_exponential_70C, label = 'Fitted Data ')
plt.plot(fitted_annealing_70C, first_fitted_exponential_70C, label = 'First Exponential of Fitted Data ' )
plt.plot(fitted_annealing_70C, second_fitted_exponential_70C, label = 'Second Exponential of Fitted Data ' )
plt.xlabel(r'Shifted Annealing Time $(min)$')
plt.ylabel(r'Normalized Folding Voltage $V_f$')
plt.title(r'70C Annealing')
plt.legend(loc = 'best')
plt.show()



#%%

# Plotting For 80C

# Plot
plt.figure(figsize=(11, 8))
x = range(len(Annealing_80C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(Annealing_80C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_80C_Shifted_and_Scaled_data['Pin108 Vf (10 min R irad)'], label = 'Pin108 Vf (10 min R irad)')

plt.plot(fitted_annealing_80C, fitted_exponential_80C, label = 'Fitted Data ')
plt.plot(fitted_annealing_80C, first_fitted_exponential_80C, label = 'First Exponential of Fitted Data ')
plt.plot(fitted_annealing_80C, second_fitted_exponential_80C, label = 'Second Exponential of Fitted Data ' )
plt.xlabel(r'Shifted Annealing Time $(min)$')
plt.ylabel(r'Normalized Folding Voltage $V_f$')
plt.title(r'80C Annealing')
plt.legend(loc = 'best')
plt.show()



#%%
# Plotting For 90C

# Plot
plt.figure(figsize=(11, 8))
x = range(len(Annealing_90C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(Annealing_90C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_90C_Shifted_and_Scaled_data['Pin111 Vf (10 min R irad)'], label = 'Pin111 Vf (10 min R irad)')
plt.plot(fitted_annealing_90C, fitted_exponential_90C, label = 'Fitted Data ')
plt.plot(fitted_annealing_90C, first_fitted_exponential_90C, label = 'First Exponential of Fitted Data ' )
plt.plot(fitted_annealing_90C, second_fitted_exponential_90C, label = 'Second Exponential of Fitted Data '  )
plt.xlabel(r'Shifted Annealing Time $(min)$')
plt.ylabel(r'Normalized Folding Voltage $V_f$')
plt.title(r'90C Annealing')
plt.legend(loc = 'best')
plt.show()



#%%

# Plotting For 100C

# Plot
plt.figure(figsize=(11, 8))
x = range(len(Annealing_100C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(Annealing_100C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_100C_Shifted_and_Scaled_data['Pin106 Vf (10 min R irad)'], label = 'Pin106 Vf (10 min R irad)')
plt.plot(fitted_annealing_100C, fitted_exponential_100C, label = 'Fitted Data ')
plt.plot(fitted_annealing_100C, first_fitted_exponential_100C, label = 'First Exponential of Fitted Data ' )
plt.plot(fitted_annealing_100C, second_fitted_exponential_100C, label = 'Second Exponential of Fitted Data '  )
plt.xlabel(r'Shifted Annealing Time $(min)$')
plt.ylabel(r'Normalized Folding Voltage $V_f$')
plt.title(r'100C Annealing')
plt.legend(loc = 'best')
plt.show()

#%%

# Plotting For 40C 

# Plot
plt.figure(figsize=(11, 8))
x = range(len(Annealing_40C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(Annealing_40C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_40C_Shifted_and_Scaled_data['Pin107 Vf (10 min R irad)'], label = 'Pin107 Vf (10 min R irad)')
plt.plot(fitted_annealing_40C, fitted_exponential_40C, label = r'Fitted Data From Function For of 100C Fitting, Changing $t \rightarrow 9.76*10^{-3}(t + 1.24*10^{3})$' )
plt.xlabel(r'Shifted Annealing Time $(min)$')
plt.ylabel(r'Normalized Folding Voltage $V_f$')
plt.title(r'40C Annealing')
plt.legend(loc = 'best')
plt.show()


#%%

# Plotting For 50C 

# Plot
plt.figure(figsize=(11, 8))
x = range(len(Annealing_50C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(Annealing_50C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_50C_Shifted_and_Scaled_data['Pin109 Vf (10 min R irad)'], label = 'Pin109 Vf (10 min R irad)')
plt.plot(fitted_annealing_50C, fitted_exponential_50C, label = r'Fitted Data From Function For of 100C Fitting, Changing $t \rightarrow 9.76*10^{-3}(t + 1.24*10^{3})$' )
plt.xlabel(r'Shifted Annealing Time $(min)$')
plt.ylabel(r'Normalized Folding Voltage $V_f$')
plt.title(r'50C Annealing')
plt.legend(loc = 'best')
plt.show()

#%%

# Plotting For 60C 

# Plot
plt.figure(figsize=(11, 8))
x = range(len(Annealing_60C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(Annealing_60C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_60C_Shifted_and_Scaled_data['Pin105 Vf (10 min R irad)'], label = 'Pin105 Vf (10 min R irad)')
plt.plot(fitted_annealing_60C, fitted_exponential_60C, label = r'Fitted Data From Function For of 100C Fitting, Changing $t \rightarrow 4.17*10^{-2}(t + 2.12*10^{2})$' )
plt.xlabel(r'Shifted Annealing Time $(min)$')
plt.ylabel(r'Normalized Folding Voltage $V_f$')
plt.title(r'60C Annealing')
plt.legend(loc = 'best')
plt.show()


#%%

# Plotting For 70C 

# Plot
plt.figure(figsize=(11, 8))
x = range(len(Annealing_70C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(Annealing_70C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_70C_Shifted_and_Scaled_data['Pin110 Vf (10 min R irad)'], label = 'Pin110 Vf (10 min R irad)')
plt.plot(fitted_annealing_70C, fitted_exponential_70C, label = r'Fitted Data From Function For of 100C Fitting, Changing $t \rightarrow 9.76*10^{-3}(t + 1.24*10^{3})$' )
plt.xlabel(r'Shifted Annealing Time $(min)$')
plt.ylabel(r'Normalized Folding Voltage $V_f$')
plt.title(r'70C Annealing')
plt.legend(loc = 'best')
plt.show()



#%%

# Plotting For 80C 

# Plot
plt.figure(figsize=(11, 8))
x = range(len(Annealing_80C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(Annealing_80C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_80C_Shifted_and_Scaled_data['Pin108 Vf (10 min R irad)'], label = 'Pin108 Vf (10 min R irad)')
plt.plot(fitted_annealing_80C, fitted_exponential_80C, label = r'Fitted Data From Function For of 100C Fitting, Changing $t \rightarrow 0.1353*(t + 65.898)$' )
plt.xlabel(r'Shifted Annealing Time $(min)$')
plt.ylabel(r'Normalized Folding Voltage $V_f$')
plt.title(r'80C Annealing')
plt.legend(loc = 'best')
plt.show()


#%%

# Plotting For 90C 

# Plot
plt.figure(figsize=(11, 8))
x = range(len(Annealing_90C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(Annealing_90C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_90C_Shifted_and_Scaled_data['Pin111 Vf (10 min R irad)'], label = 'Pin111 Vf (10 min R irad)')
plt.plot(fitted_annealing_90C, fitted_exponential_90C, label = r'Fitted Data From Function For of 100C Fitting, Changing $t \rightarrow 9.76*10^{-3}(t + 1.24*10^{3})$' )
plt.xlabel(r'Shifted Annealing Time $(min)$')
plt.ylabel(r'Normalized Folding Voltage $V_f$')
plt.title(r'90C Annealing')
plt.legend(loc = 'best')
plt.show()


#%%

# Plotting For 100C 

# Plot
plt.figure(figsize=(11, 8))
x = range(len(Annealing_100C_Shifted_and_Scaled_data['Annealing Time (Min)']))
plt.plot(Annealing_100C_Shifted_and_Scaled_data['Annealing Time (Min)'], Annealing_100C_Shifted_and_Scaled_data['Pin106 Vf (10 min R irad)'], label = 'Pin106 Vf (10 min R irad)')
plt.plot(fitted_annealing_100C, fitted_exponential_100C, label = 'Fitted Data')
plt.xlabel(r'Shifted Annealing Time $(min)$')
plt.ylabel(r'Normalized Folding Voltage $V_f$')
plt.title(r'100C Annealing')
plt.legend(loc = 'best')
plt.show()

#%%

# Plotting For Expoential Scaling

# Plot
plt.figure(figsize=(11, 8))
plt.plot(exponential_scale_data['Annealing Temperature'], exponential_scale_data['Exponential Scale Factor'], label = r'Expoenential Scale Data')
plt.plot(fitted_exponential_scale_data, fitted_exponential, label = 'Fitted Exponential     ' + r'$e^{(-38063*(1/(T+237.15) - 1/375.041))} $')
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






