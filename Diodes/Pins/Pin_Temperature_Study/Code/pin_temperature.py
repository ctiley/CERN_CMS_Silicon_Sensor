import numpy  as np
import matplotlib.pyplot as plt
import itertools
from scipy.optimize import curve_fit
import pandas as pd

with open('RT_Baseline_Test_22C/PinDiodeThermomenter_RT_1S.txt') as PinDiodeThermomenter_RT_1S:
    PinDiodeThermomenter_RT_1S = PinDiodeThermomenter_RT_1S.readlines()[12:]

with open('RT_to_-23C/PinDiodeThermometer_RT_to_Freezer_1S_Delay.txt') as RT_to_neg23C:
    RT_to_neg23C = RT_to_neg23C.readlines()[12:]
    
with open('RT_to_40C/PinDiodeThermomenter_RT_to_40C_1S_Delay.txt') as RT_to_40C:
    RT_to_40C = RT_to_40C.readlines()[12:]
    
with open('RT_to_60C/PinDiodeThermomenter_RT_to_60C_1S_Delay.txt') as RT_to_60C:
    RT_to_60C = RT_to_60C.readlines()[12:]
    
with open('RT_to_80C/PinDiodeThermomenter_RT_to_80C_1S_Delay.txt') as RT_to_80C:
    RT_to_80C = RT_to_80C.readlines()[12:]
    
with open('RT_to_100C/PinDiodeThermomenter_RT_to_100C_1S_Delay.txt') as RT_to_100C:
    RT_to_100C = RT_to_100C.readlines()[12:]
    
with open('RT_to_120C/PinDiodeThermomenter_RT_to_120C_1S_Delay.txt') as RT_to_120C:
    RT_to_120C = RT_to_120C.readlines()[12:]

Saturation_Data = pd.read_csv('Saturation_Data/saturation_time_vs_temperature.csv')

# Turning the Text Files into Lists of Floats 
    
data = np.loadtxt(PinDiodeThermomenter_RT_1S)
RT_to_neg23C_data = np.loadtxt(RT_to_neg23C)
RT_to_40C_data = np.loadtxt(RT_to_40C)
RT_to_60C_data = np.loadtxt(RT_to_60C)
RT_to_80C_data = np.loadtxt(RT_to_80C)
RT_to_100C_data = np.loadtxt(RT_to_100C)
RT_to_120C_data = np.loadtxt(RT_to_120C)


#%%

# Length of the Vectors to be Plotted
length_RT_1S = int(len(data[:, 0])/2)
length_RT_to_neg23C = int(len(RT_to_neg23C_data[:, 0])/2)
length_RT_to_40C = int(len(RT_to_40C_data[:, 0])/2)
length_RT_to_60C = int(len(RT_to_60C_data[:, 0])/2)
length_RT_to_80C = int(len(RT_to_80C_data[:, 0])/2)
length_RT_to_100C = int(len(RT_to_100C_data[:, 0])/2)
length_RT_to_120C = int(len(RT_to_120C_data[:, 0])/2)

# Time and Voltage Vectors
time_RT_1S = np.zeros(length_RT_1S)
volt_RT_1S = np.zeros(length_RT_1S)

time_RT_to_neg23C = np.zeros(length_RT_to_neg23C)
volt_RT_to_neg23C = np.zeros(length_RT_to_neg23C)

time_RT_to_40C = np.zeros(length_RT_to_40C)
volt_RT_to_40C = np.zeros(length_RT_to_40C)

time_RT_to_60C = np.zeros(length_RT_to_60C)
volt_RT_to_60C = np.zeros(length_RT_to_60C)

time_RT_to_80C = np.zeros(length_RT_to_80C)
volt_RT_to_80C = np.zeros(length_RT_to_80C)

time_RT_to_100C = np.zeros(length_RT_to_100C)
volt_RT_to_100C = np.zeros(length_RT_to_100C)

time_RT_to_120C = np.zeros(length_RT_to_120C)
volt_RT_to_120C = np.zeros(length_RT_to_120C)


# Picking out the Correct Voltages

for i in range (0,length_RT_1S):
    
    time_RT_1S[i] = data[2* i, 0] - data[0, 0]
    volt_RT_1S[i] = data[2* i, 2]

for i in range (0,length_RT_to_neg23C):
    
    time_RT_to_neg23C[i] = RT_to_neg23C_data[2* i, 0] - RT_to_neg23C_data[0, 0] - 1500 -175
    volt_RT_to_neg23C[i] = RT_to_neg23C_data[2* i, 2]
    
neg23C_Saturation_Voltage = 0.47042
neg23C_Saturation_Time = 3697218893 - RT_to_neg23C_data[0, 0] - 1500 - 175
    
for i in range (0,length_RT_to_40C):
    
    time_RT_to_40C[i] = RT_to_40C_data[2* i, 0] - RT_to_40C_data[0, 0] - 150 - 69
    volt_RT_to_40C[i] = RT_to_40C_data[2* i, 2]
    
RT_40C_Saturation_Voltage = 0.2786
RT_40C_Saturation_Time = 3697136034 - RT_to_40C_data[0, 0] - 150 - 69
    
for i in range (0,length_RT_to_60C):
    
    time_RT_to_60C[i] = RT_to_60C_data[2* i, 0] - RT_to_60C_data[0, 0] - 370
    volt_RT_to_60C[i] = RT_to_60C_data[2* i, 2]
    
RT_60C_Saturation_Voltage = 0.22238
RT_60C_Saturation_Time = 3697137687 - RT_to_60C_data[0, 0] - 370

for i in range (0,length_RT_to_80C):
    
    time_RT_to_80C[i] = RT_to_80C_data[2* i, 0] - RT_to_80C_data[0, 0] - 283
    volt_RT_to_80C[i] = RT_to_80C_data[2* i, 2]
    
RT_80C_Saturation_Voltage = 0.165455
RT_80C_Saturation_Time = 3697139277 - RT_to_80C_data[0, 0] - 283

for i in range (0,length_RT_to_100C):
    
    time_RT_to_100C[i] = RT_to_100C_data[2* i, 0] - RT_to_100C_data[0, 0] - 335
    volt_RT_to_100C[i] = RT_to_100C_data[2* i, 2]
    
RT_100C_Saturation_Voltage = 0.1125305
RT_100C_Saturation_Time = 3697140904 - RT_to_100C_data[0, 0] - 335
    
for i in range (0,length_RT_to_120C):
    
    time_RT_to_120C[i] = RT_to_120C_data[2* i, 0] - RT_to_120C_data[0, 0] - 255
    volt_RT_to_120C[i] = RT_to_120C_data[2* i, 2]
    
RT_120C_Saturation_Voltage = 0.0698275
RT_120C_Saturation_Time = 3697142564 - RT_to_120C_data[0, 0] - 255

    
#%%

# Saturation Vf vs. Temperature

saturation_vf = [0.478, 0.276, 0.2169, 0.157, 0.10129, 0.05635]
temperature = [-23, 40, 60, 80, 100, 120]
"""
saturation_vf = [0.478, 0.276, 0.2169, 0.157, 0.10129]
temperature = [-23, 40, 60, 80, 100]
"""
# Fitting Fuction for Exponential
def linear(x, m, b):
    return   m*x + b

saturation_vf_fit_data = np.linspace(temperature[0], temperature[4], 1000)
params_linear, params_covariance_linear = curve_fit(linear, temperature, saturation_vf)
fitted_Vf_Temperature_relation =  params_linear[0]*saturation_vf_fit_data + params_linear[1]



    

#%%

plt.plot(time_RT_to_neg23C, volt_RT_to_neg23C, '.-')
plt.plot(neg23C_Saturation_Time, neg23C_Saturation_Voltage, '.', label = 'Saturation Point')
plt.title(r'Room Temperature to Freezer (-23C)', fontsize = 22)
plt.xlabel(r'Time (s)', fontsize = 18)
plt.ylabel(r'Voltage (V)', fontsize = 18)
plt.legend(loc = 'best')
plt.show()


#%%

plt.plot(time_RT_to_40C, volt_RT_to_40C, '.-')
plt.plot(RT_40C_Saturation_Time, RT_40C_Saturation_Voltage, '.', label = 'Saturation Point')
plt.title(r'Room Temperature to 40C', fontsize = 22)
plt.xlabel(r'Time (s)', fontsize = 18)
plt.ylabel(r'Voltage (V)', fontsize = 18)
plt.legend(loc = 'best')
plt.show()


#%%

plt.plot(time_RT_to_60C, volt_RT_to_60C, '.-')
plt.plot(RT_60C_Saturation_Time, RT_60C_Saturation_Voltage, '.', label = 'Saturation Point')
plt.title(r'Room Temperature to 60C', fontsize = 22)
plt.xlabel(r'Time (s)', fontsize = 18)
plt.ylabel(r'Voltage (V)', fontsize = 18)
plt.legend(loc = 'best')
plt.show()


#%%

plt.plot(time_RT_to_80C, volt_RT_to_80C, '.-')
plt.plot(RT_80C_Saturation_Time, RT_80C_Saturation_Voltage, '.', label = 'Saturation Point')
plt.title(r'Room Temperature to 80C', fontsize = 22)
plt.xlabel(r'Time (s)', fontsize = 18)
plt.ylabel(r'Voltage (V)', fontsize = 18)
plt.legend(loc = 'best')
plt.show()


#%%

plt.plot(time_RT_to_100C, volt_RT_to_100C, '.-')
plt.plot(RT_100C_Saturation_Time, RT_100C_Saturation_Voltage, '.', label = 'Saturation Point')
plt.title(r'Room Temperature to 100C', fontsize = 22)
plt.xlabel(r'Time (s)', fontsize = 18)
plt.ylabel(r'Voltage (V)', fontsize = 18)
plt.legend(loc = 'best')
plt.show()


#%%

plt.plot(time_RT_to_120C, volt_RT_to_120C, '.-')
plt.plot(RT_120C_Saturation_Time, RT_120C_Saturation_Voltage, '.', label = 'Saturation Point')
plt.title(r'Room Temperature to 120C', fontsize = 22)
plt.xlabel(r'Time (s)', fontsize = 18)
plt.ylabel(r'Voltage (V)', fontsize = 18)
plt.legend(loc = 'best')
plt.show()

#%%

# Plotting For Linear Scaling

# Plot
plt.figure(figsize=(11, 8))
plt.plot(temperature, saturation_vf, '.-', label = r'Raw Data')
plt.plot(saturation_vf_fit_data, fitted_Vf_Temperature_relation, label = 'Fitted Line     ' + r'$-0.00297985 * T + 0.4014960$')
plt.xlabel(r'Temperature $(C)$')
plt.ylabel(r'Saturation Voltage $V_f$ $(V)$')
plt.title(r'Saturation Voltage $V_f$ vs. Temperature $(C)$')
plt.legend(loc = 'best')
plt.show()

print(params_linear)

#%%

# Plotting For Saturation

# Plot
plt.figure(figsize=(11, 8))
plt.plot(Saturation_Data['Temperature (C)'],Saturation_Data['Delta Time (s)'], 'o', label = r'Time at 95% Saturation')
plt.xlabel(r'Temperature $(C)$')
plt.ylabel(r'Saturation Time $(s)$')
plt.title(r'Saturation Time $(s)$ vs. Temperature $(C)$')
plt.legend(loc = 'best')
plt.show()








