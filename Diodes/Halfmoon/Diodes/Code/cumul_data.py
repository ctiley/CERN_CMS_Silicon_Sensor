import sys
import os
sys.path.append('/.../home/espencer')

from datetime import datetime
from scipy.stats import linregress
from forge.utilities import line_intersection
from statistics import mean
from statistics import stdev

# Current Working Directory
cwd = os.getcwd()

def list_files(dir):
    
   files = []
   
   for obj in os.listdir(dir):
       
      if os.path.isfile(obj):
          
         files.append(obj)
         
   return files

def Vuong(bv, cv):

        print ("Vuong Code")

        # Calculate the slopes from the right and the left
        LR2 = 0  
        RR2 = 0
        Right_stats = 0
        Left_stats = 0

        polarity = abs(bv[0])/bv[0]
        step = bv[len(bv)-1] - bv[len(bv)-2]


        for k in bv, cv:

            delta = [cv[i + 1] - cv[i] for i in range(len(cv) - 1)]
            Range = [bv[i + 1] - bv[i] for i in range(len(bv) - 1)]
            Normalization = max(cv) - min(cv)
            RangeNorm = [i * Normalization for i in Range]
            delta3 = [cv[i + 3] - cv[i] for i in range(len(cv) - 3)]
            Range3 = [bv[i + 3] - bv[i] for i in range(len(bv) - 3)]
            bv3 = [bv[i + 3] for i in range(len(bv) - 3)]
            RangeNorm3 = [i * Normalization for i in Range3]

            m = [x / y for x, y in zip(delta, RangeNorm)]  # m is define as normalized SLOPE
            m3 = [x / y for x, y in zip(delta3, RangeNorm3)]  # m is define as normalized SLOPE

            #mMag = map(abs, m)  # converting all data to positive to set the depletion range
            mMag = list(map(lambda x: abs(x), m))  # converting all data to positive to set the depletion range
            mMag.insert(0, mMag[0])  # making all arrays the same length
            mMag3 = list(map(lambda x: abs(x), m3))  # converting all data to positive to set the depletion range
            
            for i in range(3): mMag3.insert(0, mMag3[0])  # making all arrays the same length

            Deplete = []
            i = len(bv) - 1
            Depleted = False
            startDepletion = False
            
            while not Depleted:
                
                if (mMag3[i] < 0.001):
                    
                    Deplete.append(bv[i])
                    
                    if not startDepletion:
                        
                        startDepletion = True
                        
                elif startDepletion:
                    
                    Depleted = True
                    
                i-= 1
                
            Deplete.reverse()

            if len(Deplete) == 0:  # Creating an artificial depleteion array with last couple of values
            
                Deplete.append(bv[-4])
                Deplete.append(bv[-3])
                Deplete.append(bv[-2])
                Deplete.append(bv[-1])
                print("******************************************************************")
                print("************************* NO DEPLETION ***************************")
                print("******************************************************************")
                
            else:
                
                print("******************************************************************")
                print("*********************** Depletion Exist *************************")
                print("******************************************************************")

            if (bv[10] > 0):  # calculating the absolutine minimum point and neighbor points
            
                print("yo4", zip(bv,m3))
                mfilter = [j for i, j in zip(bv3, m3) if (i > 100.0)] # array to find the extrema far from start point
                PosBias = max(mfilter)  # to find the range before depletion in the normalized derivatice
                print("mfilter", mfilter, PosBias, len(m))
                print("m3", m3)
                Rampup = []
                for i in range(len(m3)):
                    if m3[i] == PosBias:
                        print("yo6")
                        Rampup.append(float(bv3[i]))
                        Rampup.append(float(bv3[i - 3]))
                        Rampup.append(float(bv3[i - 5]))
                print("Ramp", Rampup)
                
            else:
                
                print("yo5")
                mfilter = [j for i, j in zip(bv3, m3) if (i < -100.0)]  # array to find the extrema far from start point
                NegBias = min(mfilter)
                print("mfilter", mfilter, NegBias, len(m))
                print("m3", m3)
                Rampup = []
                
                for i in range(len(m3)):
                    
                    if m3[i] == NegBias:
                        
                        Rampup.append(float(bv3[i]))
                        Rampup.append(float(bv3[i - 3]))
                        Rampup.append(float(bv3[i - 5]))

        slopef1min = Rampup[2] * polarity  # If Error due to Out of Range, must
        slopef1max = Rampup[0] * polarity  # manually put in voltage values here!
        flatf1min = Deplete[0] * polarity
        print("yo7", Deplete[0])

        if ((Deplete[0] + 300.0) <= bv[len(bv) - 1]):  # making sure it doesn't surpass the
        
            flatf1max = (Deplete[int(300 / step)])  # maximum bias voltage
            
        else:
            
            flatf1max = Deplete[len(Deplete) - 1]

        print("yo8", polarity, bv)
        print("slopef1min: ", slopef1min, "slopef1max: ", slopef1max, "flatf1min: ", flatf1min, "flatf1max: ", flatf1max)
        LstartIdx = list(bv).index(slopef1min * polarity)
        LendIdx = list(bv).index(slopef1max * polarity)
        RstartIdx = list(bv).index(flatf1min * polarity)
        RendIdx = list(bv).index(flatf1max * polarity)

        print("*************************************************")
        print("****************** DATA POINTS ******************")
        print("*************************************************")
        print("slopef1min: ", slopef1min, "slopef1max: ", slopef1max, "flatf1min: ", flatf1min, "flatf1max: ", flatf1max)
        print("LstartIdx: ", LstartIdx, "LendIdx: ", LendIdx, "RstartIdx: ", RstartIdx, "RendIdx: ", RendIdx)

        slope_left, intercept_left, r_left, lp_value, std_err_left = linregress(bv[LstartIdx:LendIdx], cv[LstartIdx:LendIdx])
        slope_right, intercept_right, r_right, rp_value, std_err_right = linregress(bv[RstartIdx:RendIdx], cv[RstartIdx:RendIdx])
        print ("linregression done")
        LR2 = r_left * r_left
        RR2 = r_right * r_right
        print ("pts", bv[LstartIdx], intercept_left, bv[LendIdx], slope_left * bv[LendIdx] + intercept_left)
        LeftEndPoints = (
            (bv[LstartIdx], slope_left * bv[LstartIdx] + intercept_left),
            (bv[LendIdx], slope_left * bv[LendIdx] + intercept_left)
        )
        RightEndPoints = (
            (bv[RstartIdx], slope_right * bv[RstartIdx] + intercept_right),
            (bv[RendIdx], slope_right * bv[RendIdx] + intercept_right)
        )

        print ("End Points", LeftEndPoints, RightEndPoints)
        Left_stats = (LeftEndPoints, slope_left, intercept_left, r_left, lp_value, std_err_left)
        Right_stats = (RightEndPoints, slope_right, intercept_right, r_right, rp_value, std_err_right)

        print("Return", LR2, Left_stats, RR2, Right_stats)
        full_depletion_voltage = line_intersection(Left_stats[0], Right_stats[0])
        print("Full Depletion Voltage: ", full_depletion_voltage)
        return full_depletion_voltage

filelist = list_files(".")

i600 = []
i800 = []
i1000 = []
depV = []
files = []
sensors = []
diodes = []
depV_idx = []
curr_idx = []

for cviv in ["IV","CV"]:

    for cvivDataFile in filelist:
        
        if ".txt" in cvivDataFile and cviv in cvivDataFile:
            
            print ("Analyzing: ", cvivDataFile)
            nameForSave = cvivDataFile.strip('.txt')
            sensor = nameForSave.split("_")[1] + "_" + nameForSave.split("_")[2] + "_" + nameForSave.split("_")[3] + "_" + nameForSave.split("_")[4] + "_" + nameForSave.split("_")[5]
            diode = sensor + "_" + nameForSave.split("_")[6]
            
            if "GR" in nameForSave.split("_")[7]: diode = diode + "_" + nameForSave.split("_")[7]
            
            if "IRRADSENSOR" and 'min' in nameForSave.split("_")[7]: diode = diode + "_" + nameForSave.split("_")[7]
            
            files.append(nameForSave)
            sensors.append(sensor)
            
            if diode not in diodes:
            
                diodes.append(diode)

            with open(cvivDataFile, "r") as df:
                #Reading Text File Header
                txtLines = [line for line in df]
                Notes = ""
                for line in txtLines:
                    if "Date/Time" in line:
                        TimeStamp = line[11:-2]
                        #datetime_obj = datetime.strptime(TimeStamp, '%m/%d/%Y %I:%M:%S %p')
                        #print (datetime_obj)
                    elif "Sensor Name" in line:
                        SensorName = line[13:-1]
                    elif "Tester" in line:
                        User = line[8:-1]
                    elif "Notes" in line:
                        Notes = line[9:-2]
                #print("Header Info: ", datetime_obj, SensorName, User, Notes)
                print("Header Info: ", SensorName, User, Notes)
                idx = [i for i, line in enumerate(txtLines) if "BiasVoltage" in line][0]
                headers = txtLines[idx].split('\t')

                if "CV" in cvivDataFile:
                    
                    if "LCR_Cp_freq1" in headers:
                        
                        cv1kHz_idx = headers.index("LCR_Cp_freq1")
                        
                    elif "LCR_Cp_freq1000.0" in headers:
                        
                        cv1kHz_idx = headers.index("LCR_Cp_freq1000.0")
                    temp_idx = headers.index("Temperature")
                    rh_idx = headers.index("Humidity")

                    cv1kHz = []
                    c2 = []
                    bv = []
                    temp = []
                    rh = []

                    data = txtLines[idx+1:]
                    
                    for line in data:
                        
                        words = line.split()
                        
                        bv.append(abs(float(words[0])))
                        cv1kHz.append(float(words[cv1kHz_idx]))
                        c2.append(1/(float(words[cv1kHz_idx])*(float(words[cv1kHz_idx]))))
                        #cv10kHz.append(words[cv10kHz_idx])
                        temp.append(float(words[temp_idx]))
                        rh.append(float(words[rh_idx]))

                    depV_idx.append(diode)
                    
                    if 'PreIrrad' in cwd:
                        
                        depV.append(Vuong(bv, c2)[0])
                    
                    for i in range(len(bv)):
                        #bv[i] = bv[i]
                        if temp[i] == 25.0:  temp[i] = 18.0
                        if temp[i] == 0.0:  temp[i] = 22.0
                        if rh[i] < 0: rh[i] = 4.0

                if "IV" in cvivDataFile:
                    iv_idx = headers.index("Bias Current_Avg")
                    temp_idx = headers.index("Temperature")
                    rh_idx = headers.index("Humidity")

                    iv = []
                    bv = []
                    temp = []
                    rh = []

                    data = txtLines[idx+1:]
                    for line in data:
                        words = line.split()
                        bv.append(float(words[0]))
                        iv.append(float(words[iv_idx]))
                        temp.append(float(words[temp_idx]))
                        rh.append(float(words[rh_idx]))

                    if -600 in bv:
                        curr_idx.append(diode)
                        i600.append(iv[bv.index(-600)])
                    if -800 in bv:
                        i800.append(iv[bv.index(-800)])
                    if -1000 in bv:
                        i1000.append(iv[bv.index(-1000)])
                    for i in range(len(bv)):
                        iv[i] = iv[i]*1e9
                        if temp[i] == 25.0:  temp[i] = 18.0
                        if temp[i] == 0.0:  temp[i] = 22.0
                        if rh[i] < 0: rh[i] = 4.0

                    AvTemp = str(sum(temp)/len(temp))
                    AvRH = str(sum(rh)/len(rh))


f = open("CumulData.txt", "w")
f.write("File\tI(600V)\tI(800V)\tI(1000V)\tDepV\n")

for diode in diodes:
    print(diode)
    #print ("File", file)
    print ("Diode", diode)
    iv = curr_idx.index(diode)
    cv = depV_idx.index(diode)
    print(diode)
    
    if 'PostIrrad' in cwd:
        
        f.write(diode+"\t"+str(i600[iv])+"\t"+str(i800[iv])+"\t"+str(i1000[iv])+"\n")
        
    if 'PreIrrad' in cwd:
        
        f.write(diode+"\t"+str(i600[iv])+"\t"+str(i800[iv])+"\t"+str(i1000[iv])+"\t"+str(depV[cv])+"\n")
    
    
    
    















