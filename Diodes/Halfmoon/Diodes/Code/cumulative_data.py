from scipy.optimize import curve_fit
from math import log10, floor

import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd

import os
import csv

# Is NaN Function
def isNaN(num):
    return num != num

# Significant Figures Function
def round_sig(x, sig=3, small_value=1.0e-9):
    if isNaN(x) == True:
        return x
    else:
        return round(x, sig - int(floor(log10(max(abs(x), abs(small_value))))) - 1)

# Get list of Files    
def get_files():
    
    # Current Working Directory
    cwd = os.getcwd() 
    
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
def get_iv_cv_dataframes(file_names, diode_names):
    
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

def fit_endpoint_finder(data, fit_bias_values):

    data_len = len(data)
    fit_stop = [0,0]

    if abs(data[data_len - 1]) >= 1000:
        
        fit_bias_values[1] = data[data_len]

    for i in range(0, data_len):
        
        if abs(data[i]) <= abs(fit_bias_values[0]):
            
            fit_stop[0] = i    
            
        if abs(data[i]) >= abs(fit_bias_values[1]):
            
            fit_stop[1] = i    
           
            return fit_stop

# Put 1/C^2 values into Dataframes
def get_capacitance_squared_values(CV, diode_names):

    for i in range(0, len(diode_names)):
        
        if 'LCR_Cp_freq1000.0' in CV[diode_names[i]].columns:
        
            CV[diode_names[i]]['1/C^2'] = 1/CV[diode_names[i]]['LCR_Cp_freq1000.0']
        
        if 'LCR_Cp_freq1' in CV[diode_names[i]].columns:
        
            CV[diode_names[i]]['1/C^2'] = 1/CV[diode_names[i]]['LCR_Cp_freq1']

    return CV 

# This Function Produces 1/C^2 Values
def find_depletion_voltage(CV, diode_names, plot = True):
    
    for i in range(0, len(diode_names)):

        bias_data = CV[diode_names[i]]['BiasVoltage']
        capacitance = CV[diode_names[i]]['1/C^2']
        
        left_fit_bias = [100, 190] 
        right_fit_bias = [450, 600]
        
        left_fit_stop = fit_endpoint_finder(bias_data, left_fit_bias)
        right_fit_stop = fit_endpoint_finder(bias_data, right_fit_bias)
        
        left_bias_data = bias_data[left_fit_stop[0]:left_fit_stop[1]].to_numpy()
        right_bias_data = bias_data[right_fit_stop[0]:right_fit_stop[1]].to_numpy()
        
        left_capacitance_data = capacitance[left_fit_stop[0]:left_fit_stop[1]].to_numpy()
        right_capacitance_data = capacitance[right_fit_stop[0]:right_fit_stop[1]].to_numpy()
        
        bias_fit_left = np.linspace(bias_data[left_fit_stop[0]], bias_data[left_fit_stop[1]] - 200, 100)
        bias_fit_right = np.linspace(0, bias_data[right_fit_stop[1]], 100)
        
        guess_left = [1378666666.6667, 124266666666.67]
        guess_right = [0, 1e11]
        
        params_left, covariance_left = curve_fit(linear_fit, left_bias_data, left_capacitance_data, guess_left)
        capacitance_left_fit = params_left[0]*bias_fit_left + params_left[1]
        
        params_right, covariance_right = curve_fit(linear_fit, right_bias_data, right_capacitance_data, guess_right)
        capacitance_right_fit = params_right[0]*bias_fit_right + params_right[1]
        
        dep_v = 0
        dep_v_cap = 0
        
        for j in range(0, len(bias_fit_left)):
            
            if capacitance_left_fit[j] <= capacitance_right_fit[j]:
                
                dep_v = bias_fit_left[j]
                dep_v_cap = capacitance_left_fit[j]
                
        if plot == True:
            
            # Initialize Plot
            plt.figure(figsize = (10, 8))

            # General Plot
            plt.scatter(-1*bias_data, capacitance, label = 'Data')
            plt.plot(-1*bias_fit_left, capacitance_left_fit)
            plt.plot(-1*bias_fit_right, capacitance_right_fit)
            plt.scatter(-1*dep_v, dep_v_cap, label = 'Depletion Voltage: ' + str(-1*round_sig(dep_v)))

            # Label Plot
            plt.suptitle(diode_names[i], fontsize=20)
            plt.xlabel(r'Bias Voltage   $[V]$', fontsize = 18)
            plt.ylabel(r'1/Capacitance   $[1/C^2]$', fontsize = 18)
            plt.legend(fontsize = 14)
        
        CV[diode_names[i]]['Dep_V'] = dep_v
    
    return CV 


def main():	 

    # Get list of Files    
    file_names = get_files()
            
    # Get Diode Names
    diode_names = get_diode_names(file_names)
    
    # Turn .txt Files into Dataframes
    IV, CV = get_iv_cv_dataframes(file_names, diode_names)
    
    # Put 1/C^2 values into Dataframes
    CV = get_capacitance_squared_values(CV, diode_names)
    
    # Get Depletion Values
    CV = find_depletion_voltage(CV, diode_names, plot = True)




if __name__ == '__main__':
	main()










