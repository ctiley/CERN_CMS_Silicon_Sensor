from scipy.optimize import curve_fit
from math import log10, floor

import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd

import os


def main():	 
    
    # Current Working Directory
    cwd = os.getcwd() 

    # Left Fit Estimation
    left_fit_bias = [280, 420] 
    right_fit_length = 350

    plot = True

    # Get list of Files    
    file_names = get_files(cwd)
            
    # Get Diode Names
    diode_names = get_diode_names(file_names)

    # Turn .txt Files into Dataframes
    IV, CV = get_iv_cv_dataframes(file_names, diode_names)

    # Put 1/C^2 values into Dataframes
    CV = get_capacitance_squared_values(CV, diode_names)

    # Get Depletion Values
    CV = find_depletion_voltage(CV, diode_names, left_fit_bias, plot, right_fit_length)

    print(diode_names)

    # CV['N4789_21_UR_DIODE_GR_Irradiated_60C_0min']['Dep_V'] = 1000

    # Get Current at Depletion Voltage
    IV = get_current_at_depv(IV, CV, diode_names)

    # Make CumulData.txt File
    make_culum_data_txt(IV, CV, diode_names)
















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
        
        if 'IV' in files[i]:
            
            split = files[i].strip('.txt').split("_")
            diode = split[1] + "_" + split[2] + "_" + split[3] + "_" + split[4] + "_" + split[5] + "_" + split[6]
            
            if "Annealed" in split: diode = diode + "_" + split[8] + "_" + split[9]
            
            diode_names.append(diode)
            
        if 'CV' in files[i]:
            
            split = files[i].strip('.txt').split("_")
            diode = split[1] + "_" + split[2] + "_" + split[3] + "_" + split[4] + "_" + split[5] + "_" + split[6]
            
            if "Annealed" in split: diode = diode + "_" + split[8] + "_" + split[9]
            
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

def fit_endpoint_finder(data, fit_bias_values, right_distance):
    
    data_len = len(data)
    
    left_fit_stop = [0,0]
    right_fit_stop = [0, data_len - 1]
    
    for i in range(0, data_len):
    
        if abs(data[i]) <= abs(data[data_len - 1]) - right_distance:
            
            right_fit_stop[0] = i  

    for i in range(0, data_len):
        
        if abs(data[i]) <= abs(fit_bias_values[0]):
            
            left_fit_stop[0] = i    
            
        if abs(data[i]) >= abs(fit_bias_values[1]):
            
            left_fit_stop[1] = i    
           
            return left_fit_stop, right_fit_stop

# Put 1/C^2 values into Dataframes
def get_capacitance_squared_values(CV, diode_names):

    for i in range(0, len(diode_names)):
        
        if 'LCR_Cp_freq1000.0' in CV[diode_names[i]].columns:
        
            CV[diode_names[i]]['1/C^2'] = 1/CV[diode_names[i]]['LCR_Cp_freq1000.0']
        
        if 'LCR_Cp_freq1' in CV[diode_names[i]].columns:
        
            CV[diode_names[i]]['1/C^2'] = 1/CV[diode_names[i]]['LCR_Cp_freq1']
            
        if 'LCR_Cp_freqNaN' in CV[diode_names[i]].columns:
        
            CV[diode_names[i]]['1/C^2'] = 1/CV[diode_names[i]]['LCR_Cp_freqNaN']

    return CV 

# This Function Produces 1/C^2 Values
def find_depletion_voltage(CV, diode_names, left_fit_bias, plot, right_distance):
    
    extra_left_fit = 250
    
    for i in range(0, len(diode_names)):

        bias_data = abs(CV[diode_names[i]]['BiasVoltage'])
        capacitance = CV[diode_names[i]]['1/C^2']
        
        left_fit_stop, right_fit_stop = fit_endpoint_finder(bias_data, left_fit_bias, right_distance)
        
        left_bias_data = abs(bias_data[left_fit_stop[0]:left_fit_stop[1]].to_numpy())
        right_bias_data = abs(bias_data[right_fit_stop[0]:right_fit_stop[1]].to_numpy())
        
        left_capacitance_data = capacitance[left_fit_stop[0]:left_fit_stop[1]].to_numpy()
        right_capacitance_data = capacitance[right_fit_stop[0]:right_fit_stop[1]].to_numpy()
        
        bias_fit_left = np.linspace(bias_data[left_fit_stop[0]], bias_data[left_fit_stop[1]] + extra_left_fit, 1000)
        bias_fit_right = np.linspace(50, bias_data[right_fit_stop[1]], 1000)
        
        left_last_value = len(left_capacitance_data) - 1
        right_last_value = len(right_capacitance_data) - 1
        
        guess_left = [(left_capacitance_data[left_last_value] + left_capacitance_data[0])/(left_bias_data[left_last_value] + 
                                                                                           left_bias_data[0]), (left_capacitance_data[0] + 
                                                                                                                left_capacitance_data[left_last_value])/2]
        guess_right = [(right_capacitance_data[right_last_value] + right_capacitance_data[0])/(right_bias_data[right_last_value] + 
                                                                                               right_bias_data[0]), right_capacitance_data[0]]
        
        params_left, covariance_left = curve_fit(linear_fit, left_bias_data, left_capacitance_data, guess_left)
        capacitance_left_fit = params_left[0]*bias_fit_left + params_left[1]
        
        params_right, covariance_right = curve_fit(linear_fit, right_bias_data, right_capacitance_data, guess_right)
        capacitance_right_fit = params_right[0]*bias_fit_right + params_right[1]
        
        dep_v = 0
        dep_v_cap = 0
        
        found_dep_V = False
        
        for j in range(0, len(bias_fit_left)):
            
            for k in range(0, len(bias_fit_right)):
            
                if  bias_fit_right[k] > bias_fit_left[j]:
                    
                    if  capacitance_right_fit[k] < capacitance_left_fit[j]:
                    
                        dep_v = bias_fit_left[j]
                        dep_v_cap = capacitance_left_fit[j]
                        
                        found_dep_V = True
                        
                        break
                
                if j == len(bias_fit_left) - 1:
                        
                        dep_v = bias_data.to_numpy()[-1]
                        dep_v_cap = capacitance.to_numpy()[-1]                
                        
            if found_dep_V == True:
                
                break
                        
        if plot == True:
            
            # Initialize Plot
            plt.figure(figsize = (10, 8))

            # General Plot
            plt.scatter(abs(bias_data), capacitance, label = 'Data')
            plt.plot(abs(bias_fit_left), capacitance_left_fit)
            plt.plot(abs(bias_fit_right), capacitance_right_fit)
            plt.scatter(abs(dep_v), dep_v_cap, label = 'Depletion Voltage: ' + str(abs(round_sig(dep_v))))

            # Label Plot
            plt.suptitle(diode_names[i], fontsize=20)
            plt.xlabel(r'Bias Voltage   $[V]$', fontsize = 18)
            plt.ylabel(r'1/Capacitance^2   $[1/C^2]$', fontsize = 18)
            plt.legend(fontsize = 14)
        
        CV[diode_names[i]]['Dep_V'] = dep_v
    
    return CV 

def findYPoint(xa, xb, ya, yb, xc):
    
    m = (ya - yb) / (xa - xb)
    yc = (xc - xb) * m + yb
    
    return yc

def get_current_at_depv(IV, CV, diode_names):
    
    for i in range(0, len(diode_names)):
        
        depV = CV[diode_names[i]]['Dep_V'][0]
        BiasVoltage = abs(IV[diode_names[i]]['BiasVoltage'])
        bias_current_average = IV[diode_names[i]]['Bias Current_Avg']
        
        for j in range(len(BiasVoltage)):
            
            if abs(BiasVoltage[j]) > abs(depV):
                
                xb = BiasVoltage[j]
                yb = bias_current_average[j]
                
                break
            
            xa = BiasVoltage[j]
            ya = bias_current_average[j]
        
        if abs(depV) < 940:
            
            current_depv = findYPoint(xa, xb, ya, yb, depV)
        
        
        else:
            
            current_depv = abs(bias_current_average[len(bias_current_average) - 1])
        
        IV[diode_names[i]]['Current_Dep_V'] = current_depv
        
    return IV
        


# Make CumulData.txt File
def make_culum_data_txt(IV, CV, diode_names):
    
    f = open("CumulData.csv", "w")
    f.write("File,I(600V),I(800V),I(1000V),DepV,I(DepV)\n")
    
    for i in range(0, len(diode_names)):
        
        diode = diode_names[i]
        
        bias_current_average = IV[diode_names[i]]['Bias Current_Avg']
        bias_voltage = IV[diode_names[i]]['BiasVoltage']
        
        i600 = 0
        i800 = 0
        i1000 = 0
        
        for j in range(0, len(bias_current_average)):
            
            if int(abs(bias_voltage[j])) == 600:
                
                i600 = bias_current_average[j]
                
            if abs(bias_voltage[j]) == 800:
                
                i800 = bias_current_average[j]
                
            if abs(bias_voltage[j]) == 1000:
                
                i1000 = bias_current_average[j]
            
        depV = abs(CV[diode_names[i]]['Dep_V'][0])
        current_depv = abs(IV[diode_names[i]]['Current_Dep_V'][0])
        
        f.write(diode+","+str(i600)+","+str(i800)+","+str(i1000)+","+str(depV)+","+str(current_depv)+"\n")







if __name__ == '__main__':
 	main()





