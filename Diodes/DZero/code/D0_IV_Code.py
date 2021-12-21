from operator import itemgetter

import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd

import os

# Array for IV Data
IV_Path = []
D0_Headers = ['BiasVoltage', 'BiasCurrent', 'Keithley237', 'GRCurrent']

# Current Working Directory
cwd = os.getcwd()

def Extract(lst, element_num):
    return list( map(itemgetter(element_num), lst))

# Convert .txt File to .csv File
def txt_to_csv(file):
    temp = file

    if ".txt" in file:
        file = temp[:-4]

    with open(str(file) + '.txt') as fin, open(str(file)+'.csv', 'w') as fout:
        for line in fin:
            fout.write(line.replace('\t', ','))

def plot_D0(data, headers, sensor_name):
    plt.figure(figsize=(10,8))
    
    for i in range(3):
        plt.plot(data[:,0],data[:,i+1], label = D0_Headers[i+1]) 
        
    plt.xlabel('Voltage [V]')
    plt.ylabel('Current [A]')
    plt.title(str(sensor_name))
    plt.legend(loc = 'best')
    plt.savefig(str(sensor_name)+'.png')

def sensor_name(file_name):
    
    nameForSave = file_name.strip('.txt')
    sensor_name = nameForSave.split("_")[1] + "_" + nameForSave.split("_")[2] + "_" + nameForSave.split("_")[3] + "_" + nameForSave.split("_")[4] + "_" + nameForSave.split("_")[5]
    
    return sensor_name

def save_to_xlsx(data, headers, file_name):
    
    df = pd.DataFrame(data)
    df.to_excel(str(file_name) + '.xlsx', header = headers, index=False)

# List all of the Files in the Current Working Directory 
search_path = '.'
root, dirs, files = next(os.walk(search_path), ([],[],[]))

# Collect the Paths for the Data from the .txt files
for i in range(0, len(files)):
    
    if "IV_D0" in files[i] and ".csv" not in files[i]:
                
        IV_Path.append(files[i].replace('.txt', ''))  
        
# Collect the [Voltage, Bias_Current] and [Voltage, GR_Current] Data in Dictionaries
for i in range(0,len(IV_Path)):

    with open(IV_Path[i] + '.txt') as temp:
        
        temp = temp.readlines()[22:]
        
        for j in range(len(temp)):
            
            temp[j] = temp[j].split()
            
        temp_array = np.asarray(temp, dtype=np.float32)
        
        BiasVoltage_len = len(temp_array[:,0])
        BiasCurrent_len = len(temp_array[:,6])
        Keithley237_len = len(temp_array[:,1])
        
        D0_IV_Data = np.zeros((BiasVoltage_len, 4))
        
        for k in range(BiasVoltage_len):
            
            D0_IV_Data[k,0] = temp_array[k,0]
            D0_IV_Data[k,1] = temp_array[k,6]
            D0_IV_Data[k,2] = temp_array[k,1]
            D0_IV_Data[k,3] = temp_array[k,1] - temp_array[k,6]

    save_to_xlsx(D0_IV_Data, D0_Headers, IV_Path[i])
    plot_D0(D0_IV_Data, D0_Headers, sensor_name(IV_Path[i]))
    


        
        