import pandas as pd

location = '~/Desktop/U/Computation-/Python/goofLQ4.csv'
dataframe = pd.read_csv(location , header=0)
Stokes1 = dataframe['Time Stamp [s]'].values
Stokes2 = dataframe['Stokes 1'].values
Stokes3 = dataframe['Stokes 2'].values

location2 = '~/Desktop/U/Computation-/Python/retardances.csv'
dataframe2 = pd.read_csv(location2, header=0)
rtds = dataframe2['Retardance'].values

result = []

for i in range(0,len(Stokes1)):
    
    dat1 , dat2 , dat3 = abs(Stokes1[i]) , abs(Stokes2[i]) , abs(Stokes3[i])

    if dat1 < 1 and dat1 > 0.95:
        result.append(i)
        
    elif dat2 < 1 and dat2 > 0.95:
        result.append(i)
    
    elif dat3 < 1 and dat3 > 0.95:
        result.append(i)
    else:
        continue
    
inputRtds = open("RetardosValidos.csv","a")
inputRtds.write("Retardance,Interval" + "\n")
for i in range(0,len(rtds[result])):
    inputRtds.write(str(rtds[result][i]) + ",1" + "\n")
inputRtds.close()