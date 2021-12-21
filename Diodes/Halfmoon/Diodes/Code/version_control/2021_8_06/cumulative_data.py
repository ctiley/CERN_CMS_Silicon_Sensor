import os
import numpy  as np
from operator import itemgetter
#import pandas as pd
#import matplotlib.pyplot as plt

# Python Script Vectors
headers_irradiated = ['BiasVoltage','LCR_Cp_freq1000.0','LCR_Cp_freq10000.0','LCR_Rp_freq1000.0','LCR_Rp_freq10000.0','Temperature,Air_Temp','Humidity','Dewpoint']
headers_unirradiated = ['BiasVoltage','LCR_Cp_freq1000.0','LCR_Rp_freq1000.0','Temperature,Air_Temp','Humidity','Dewpoint']
diode_names = ["QUARTER","GR", "PSTOP", "HALF", "DIODE"]
IVCV = ["IV", "CV"]

# Dicts for IV and CV Data
IV_Path = {}
CV_Path = {}
IV_Data = {}
CV_Data = {}

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

# List all of the Files in the Current Working Directory 
search_path = '.'   # set your path here.
root, dirs, files = next(os.walk(search_path), ([],[],[]))

#%%

# Collect the Paths for the Data from the .txt files
for i in range(0, len(files)):
    
    for j in range(0,3):
            
        if IVCV[0] in files[i]:
            
            if diode_names[j] in files[i]:
                
                IV_Path[diode_names[j]] = files[i]
            
            if diode_names[3] in files[i] and diode_names[2] not in files[i]:
                
                IV_Path[diode_names[3]] = files[i]
                
            if diode_names[4] in files[i] and diode_names[3] not in files[i] and diode_names[0] not in files[i]:
                
                IV_Path[diode_names[4]] = files[i]    
                
            
        elif IVCV[1] in files[i]:
            
            if diode_names[j] in files[i]:
                
                CV_Path[diode_names[j]] = files[i]
            
            if diode_names[3] in files[i] and diode_names[2] not in files[i]:
                
                CV_Path[diode_names[3]] = files[i]
                
            if diode_names[4] in files[i] and diode_names[3] not in files[i] and diode_names[0] not in files[i]:
                
                CV_Path[diode_names[4]] = files[i]                  

# Convert .txt Files to .csv Files for Easier Working 
for i in range(0,len(CV_Path)):
    txt_to_csv(CV_Path[diode_names[i]])
    
for i in range(0,len(IV_Path)):
    txt_to_csv(IV_Path[diode_names[i]])

#%%

# Collect the [Voltage, Capacitance] Data in Dictionaries
for i in range(0,len(CV_Path)):
    CV_temp = CV_Path[diode_names[i]][:-4]

    with open(CV_temp + '.csv') as temp:
        CV_Data[diode_names[i]] = temp.readlines()[22:]

# Collect the [Voltage, Current] Data in Dictionaries
for i in range(0,len(IV_Path)):
    IV_temp = IV_Path[diode_names[i]][:-4]

    with open(IV_temp + '.csv') as temp:
        IV_Data[diode_names[i]] = temp.readlines()[21:]


#%%

temp = IV_Data[diode_names[2]]
print(Extract(temp,0))

# Collect the 1/C^2 Data in Dictionary

            







































                