import os
#import numpy  as np
#import pandas as pd
#import matplotlib.pyplot as plt

# Diode Names for Plotting 
diode_names = ["QUARTER","GR", "PSTOP", "HALF", "DIODE"]
IVCV = ["IV", "CV"]

# Dicts for IV and CV Data
IV_Path = {}
CV_Path = {}
IV_Data = {}
CV_Data = {}

# Current Working Directory
cwd = os.getcwd()

# Convert .txt File to .csv File
def txt_to_csv(file):
    with open(str(file)+'.txt') as fin, open(str(file)+'.csv', 'w') as fout:
        for line in fin:
            fout.write(line.replace('\t', ','))

# List all of the Files in the Current Working Directory 
search_path = '.'   # set your path here.
root, dirs, files = next(os.walk(search_path), ([],[],[]))

# Collect the Data from the .txt files
for i in range(0, len(files)):
    
    for j in range(0,5):
            
        if diode_names[j] and IVCV[0] in files[i]:
            IV_Path[diode_names[j]] = files[i]
            
        elif diode_names[j] and IVCV[1] in files[i]:
            CV_Path[diode_names[j]] = files[i]
            
        else:
            continue
            
print(IV_Path)
print(CV_Path)
            
            
            
            
            
            
            
            
                