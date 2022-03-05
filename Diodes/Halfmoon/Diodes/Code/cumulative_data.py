from scipy.optimize import curve_fit
from math import log10, floor

import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd

import os
import csv

# Current Working Directory
cwd = os.getcwd() 

# Get list of Files    
def get_files(cwd):
    
    files = []
    
    for obj in os.listdir(cwd):
        
       if os.path.isfile(obj):
           
           if '.DS_Store' not in obj:
           
               files.append(obj)
          
    return files

# This Function Gets Only The Names of the Diodes From the IV Files
def get_diode_names(files):
    
    diode_names = []
    
    for i in range(0, len(files)):
        
        if 'IV' or 'CV' in files[i]:
            
            split = files[i].strip('.txt').split("_")
            diode = split[1] + "_" + split[2] + "_" + split[3] + "_" + split[4] + "_" + split[5] + "_" + split[6]
            
            if "GR" in split[7]: diode = diode + "_" + split[7]
            if "IRRADSENSOR" and 'min' in split[7]: diode = diode + "_" + split[7]
            
            diode_names.append(diode)
        
    diode_names = set(diode_names)
    diode_names = sorted(diode_names)
        
    return diode_names

# This Function Returns a DataFrame of Data With The Original Column Names
def file_to_dataframe(file_name):

    lines = []
    
    with open(file_name) as f:
        
        lines = f.readlines()
        
    n = 0
    
    while 'BiasVoltage' not in lines[n]:
        
        n +=1
        
    del lines[:n]  
    
    columns_names = lines[0].split('\t')
    columns_names.pop()
    
    del lines[0] 
    
    diode_array = np.zeros((len(lines), len(columns_names)))
    
    for i in range(0, len(lines)):
        
        temp_row = lines[i].split('\t')
        
        for j in range(0, len(columns_names)):
            
            diode_array[i,j] = float(temp_row[j])
            
    df = pd.DataFrame(diode_array, columns = columns_names)
    
    return df

# This Function Takes File Names and Returns a Dictionary of Dataframes for Processing
def get_iv_cv_dataframes(file_names):
    
    IV = {}
    CV = {}

    for i in range(0, len(file_names)):
        
        if 'IV' in file_names[i]: 
            
            diode_name = get_diode_names([file_names[i]])
            IV[diode_name[0]] = file_to_dataframe(file_names[i])

        if 'CV' in file_names[i]:
            
            diode_name = get_diode_names([file_names[i]])
            CV[diode_name[0]] = file_to_dataframe(file_names[i])
            
    return IV, CV

# Function for Fitting
def linear_fit(x, m, b):
    return  m*x + b

# Get list of Files    
file_names = get_files(cwd)
        
# Get Diode Names
diode_names = get_diode_names(file_names)

# Turn .txt Files into Dataframes
IV, CV = get_iv_cv_dataframes(file_names)

# Put 1/C^2 values into Dataframes
for i in range(0, len(diode_names)):
    
    if 'LCR_Cp_freq1000.0' in CV[diode_names[i]].columns:
    
        CV[diode_names[i]]['1/C^2'] = 1/CV[diode_names[i]]['LCR_Cp_freq1000.0']
    
    if 'LCR_Cp_freq1' in CV[diode_names[i]].columns:
    
        CV[diode_names[i]]['1/C^2'] = 1/CV[diode_names[i]]['LCR_Cp_freq1']


bias_voltage = CV['37399_049_PSS_HM_XX_DIODEHALFPSTOP']['BiasVoltage']
capacitance = CV['37399_049_PSS_HM_XX_DIODEHALFPSTOP']['1/C^2']

# Initialize Plot
plt.figure(figsize = (10, 8))

# General Plot
plt.plot(-1*CV['37399_049_PSS_HM_XX_DIODEHALFPSTOP']['BiasVoltage'], CV['37399_049_PSS_HM_XX_DIODEHALFPSTOP']['1/C^2'])
#plt.scatter(k_par_data, w_k_data, label = 'Data')

# Label Plot
plt.xlabel(r'Bias Voltage   $[V]$', fontsize = 18)
plt.ylabel(r'1/Capacitance   $[1/C^2]$', fontsize = 18)



#annealing_fit_dict[diode_name] = np.linspace(annealing_dict[diode_name][0], annealing_dict[diode_name][len_dict[diode_name]], 1000)














