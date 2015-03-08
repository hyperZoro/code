# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 19:07:55 2015

@author: zipenghuang
"""

import numpy as np
import csv
from datetime import datetime
from scipy import interp
import matplotlib.pyplot as plt

class YieldObs:
    def __init__(self,X,Y,date):
        self.X = X
        self.Y = Y
        self.date = date
        assert self.X.size == self.Y.size
        IX = self.X.argsort()
        self.X = self.X[IX]
        self.Y = self.Y[IX]




EURYield = [];
csvfile = open('HistoricalMktData_IR_EUR.csv', 'rb')
spamreader = csv.reader(csvfile)
for row in spamreader:
    pairLen = (len(row)-3)/2
    dt = datetime.strptime(row[1], '%d/%m/%Y')
    X = np.array([int(i) for i in row[3:3+pairLen]])
    Y = np.array([float(i) for i in row[3+pairLen:len(row)]])
    EURYield.append(YieldObs(X,Y,dt))
    del pairLen,dt,X,Y,i
del row

ordinateUnion = reduce(np.union1d, (Obs.X for Obs in EURYield))
obsDate = np.array([Obs.date for Obs in EURYield]).T.flatten()

#deltaT = np.array([td.days for td in obsDate[1:obsDate.size]-obsDate[0:obsDate.size-1]])

interpYield = []
for Obs in EURYield:
    x = Obs.X
    y = Obs.Y
    interpYield.append(interp(ordinateUnion, x, y))
    del x,y

logYield = np.log(np.array(interpYield))
logReturn = logYield[1:346,:]-logYield[0:345,:]
del logYield

returnCov = np.cov(logReturn,rowvar=0)
res = np.linalg.eigh(returnCov)

PC1 = res[1][:,1570]
PC2 = res[1][:,1569]
PC3 = res[1][:,1568]

mxPC = res[1][:,1568:1571]

PC =np.doc(logReturn, mxPC)

plt.figure();
plt.plot(ordinateUnion,PC1,color='r');
plt.plot(ordinateUnion,PC2,color='b');
plt.plot(ordinateUnion,PC3,color='g');
plt.show()


#del EURYield