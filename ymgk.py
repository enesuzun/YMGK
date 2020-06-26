# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 02:34:12 2020

@author: eness
"""

import joblib
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")
import seaborn as sns
import datetime
import seaborn as seabornInstance 

#Modeling
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

airQualityDF= pd.read_excel("Zonguldak_Eren_Enerji.xlsx")

airQualityDF["Day"] = [da.day for da in airQualityDF["Tarih"]]
airQualityDF["Month"] = [da.month for da in airQualityDF["Tarih"]]
airQualityDF["Year"] = [da.year for da in airQualityDF["Tarih"]]
airQualityDF["Hour"] = [da.hour for da in airQualityDF["Tarih"]]
airQualityDF["Minute"] = [da.minute for da in airQualityDF["Tarih"]]
airQualityDF["Second"] = [da.second for da in airQualityDF["Tarih"]]
airQualityDF["HG"] = airQualityDF["Tarih"].dt.dayofweek
airQualityDF["Week"] = airQualityDF["Tarih"].dt.week

print(airQualityDF.isnull().values.any())

airQualityDF = airQualityDF.fillna(airQualityDF.mean())

print(airQualityDF.isnull().values.any())

# Six parameter are required for AQI calculation
# so2, no2, pm10, pm2.5, co, o3

def calculateAirQualityIndexSO2(so2):
    soi2=0
    if (so2>=0 and so2<=100):
     soi2= ((50-0)/(100-0))*(so2-0) + 0
    if (so2>=101 and so2<=250):
     soi2= ((100-51)/(250-101))*(so2-101) + 51
    if (so2>=251 and so2<=500):
     soi2= ((150-101)/(500-251))*(so2-251) + 101
    if (so2>=501 and so2<=850):
     soi2= ((200-151)/(850-501))*(so2-501) + 151
    if (so2>=851 and so2<=1100):
     soi2= ((300-201)/(1100-851))*(so2-851) + 201
    if (so2>=1101 and so2<= 1500):
     soi2= ((500-301)/(1500-1101))*(so2-1101) + 301
    return soi2

def calculateAirQualityIndexNo2(no2):
    noi2=0
    if (no2>=0 and no2<=100):
     noi2= ((50-0)/(100-0))*(no2-0) + 0
    if (no2>=101 and no2<=200):
     noi2= ((100-51)/(200-101))*(no2-101) + 51
    if (no2>=201 and no2<=500):
     noi2= ((150-101)/(500-201))*(no2-201) + 101
    if (no2>=501 and no2<=1000):
     noi2= ((200-151)/(1000-501))*(no2-501) + 151
    if (no2>=1001 and no2<=2000):
     noi2= ((300-201)/(2000-1001))*(no2-1001) + 201
    if (no2>=2001 and no2<= 3000):
     noi2= ((500-301)/(3000-2001))*(no2-2001) + 301
    return noi2

def calculateAirQualityIndexPM10(pm10):
    pm10i2=0
    if (pm10>=0 and pm10<=50):
     pm10i2= ((50-0)/(50-0))*(pm10-0) + 0
    if (pm10>=51 and pm10<=100):
     pm10i2= ((100-51)/(100-51))*(pm10-51) + 51
    if (pm10>=101 and pm10<=260):
     pm10i2= ((150-101)/(260-101))*(pm10-101) + 101
    if (pm10>=261 and pm10<=400):
     pm10i2= ((200-151)/(400-261))*(pm10-261) + 151
    if (pm10>=401 and pm10<=520):
     pm10i2= ((300-201)/(520-401))*(pm10-401) + 201
    if (pm10>=521 and pm10<= 620):
     pm10i2= ((500-301)/(620-521))*(pm10-521) + 301
    
    return pm10i2   

def calculateAirQualityIndexPM25(pm25):
    pm25i2=0
    if (pm25>=0 and pm25<=12):
     pm25i2= ((50-0)/(12-0))*(pm25-0) + 0
    if (pm25>=12.1 and pm25<=35.4):
     pm25i2= ((100-51)/(35.4-12.1))*(pm25-12.1) + 51
    if (pm25>=35.5 and pm25<=55.4):
     pm25i2= ((150-101)/(55.4-35.5))*(pm25-35.5) + 101
    if (pm25>=55.5 and pm25<=150.4):
     pm25i2= ((200-151)/(150.4-55.5))*(pm25-55.5) + 151
    if (pm25>=150.5 and pm25<=250.4):
     pm25i2= ((300-201)/(250.4-150.5))*(pm25-150.5) + 201
    if (pm25>=250.5 and pm25<= 350.4):
     pm25i2= ((400-301)/(350.4-250.5))*(pm25-250.5) + 301
    if (pm25>=350.5 and pm25<= 505.4):
     pm25i2= ((500-401)/(505.4-350.5))*(pm25-350.5) + 401
    return pm25i2      

def calculateAirQualityIndexCO(CO):
    coi2=0
    if (CO>=0 and CO<=5500):
     coi2= ((50-0)/(5500-0))*(CO-0) + 0
    if (CO>=5501 and CO<=10000):
     coi2= ((100-51)/(10000-5501))*(CO-5501) + 51
    if (CO>=10001 and CO<=16000):
     coi2= ((150-101)/(16000-10001))*(CO-10001) + 101
    if (CO>=16001 and CO<=24000):
     coi2= ((200-151)/(24000-16001))*(CO-16001) + 151
    if (CO>=24001 and CO<=32000):
     coi2= ((300-201)/(32000-24001))*(CO-24001) + 201
    if (CO>=32001 and CO<=40000):
     coi2= ((500-301)/(40000-32001))*(CO-32001) + 301
    return coi2  

def calculateAirQualityIndexO3(O3):
    o3i2=0
    if (O3>=0 and O3<=120):
      o3i2= ((50-0)/(120-0))*(O3-0) + 0
    if (O3>=121 and O3<=160):
      o3i2= ((100-51)/(160-121))*(O3-121) + 51
    if (O3>=161 and O3<=180):
      o3i2= ((150-101)/(180-161))*(O3-161) + 101
    if (O3>=181 and O3<=240):
      o3i2= ((200-151)/(240-181))*(O3-181) + 151
    if (O3>=241 and O3<=700):
      o3i2= ((300-201)/(700-241))*(O3-241) + 201
    if (O3>=701 and O3<=1700):
      o3i2= ((500-301)/(1700-701))*(O3-701) + 301
    return o3i2  

def calculateHKI():
    listPM10 = list(airQualityDF.filter(like='PM10').columns)
    listSO2 = list(airQualityDF.filter(like='SO2').columns)
    listNO2 = list(airQualityDF.filter(like='NO2').columns)
    listCO = list(airQualityDF.filter(like='CO').columns)
    listO3 = list(airQualityDF.filter(like='O3').columns)
    listPM25 = list(airQualityDF.filter(like='PM 2.5').columns)
    for pm10 in listPM10:
        airQualityDF["HKI"+pm10] = airQualityDF[pm10].apply(calculateAirQualityIndexPM10)
    for so2 in listSO2:
        airQualityDF["HKI"+so2] = airQualityDF[so2].apply(calculateAirQualityIndexSO2)
    for no2 in listNO2:
        airQualityDF["HKI"+no2] = airQualityDF[no2].apply(calculateAirQualityIndexNo2)
    for co in listCO:
        airQualityDF["HKI"+co] = airQualityDF[co].apply(calculateAirQualityIndexCO)
    for O3 in listO3:
        airQualityDF["HKI"+O3] = airQualityDF[O3].apply(calculateAirQualityIndexO3)
    for PM25 in listPM25:
        airQualityDF["HKI"+PM25] = airQualityDF[PM25].apply(calculateAirQualityIndexPM25)

def splitHKIValueAndValue(newList):
    listMaterialName= []
    listHighValue = []
    for dongu in airQualityDF[newList].values:
        listMaterialName.append(newList[dongu.argmax()].split('-')[1])
        listHighValue.append(dongu.max())
    columnName = newList[0].split('HKI')[1].split('-')[0]
    columnType = columnName+'Type'
    airQualityDF['AQI-'+columnName] = listHighValue
    airQualityDF['AQI-'+columnType] = listMaterialName

if __name__ == '__main__':

    calculateHKI()

    hkiSantral = list(airQualityDF.filter(regex = 'HKISantral-').columns)
    hkiLise = list(airQualityDF.filter(regex = 'HKILise-').columns)
    hkiTepeköy = list(airQualityDF.filter(regex = 'HKITepeköy-').columns)
    
    splitHKIValueAndValue(hkiSantral)
    splitHKIValueAndValue(hkiLise)
    splitHKIValueAndValue(hkiTepeköy)
    
# Plot
    santral = airQualityDF.loc[:,[hkiSantral[0], hkiSantral[1], hkiSantral[2], hkiSantral[3], hkiSantral[4], 
                                  hkiSantral[5], 'AQI-Santral','AQI-SantralType']]
    
    lise = airQualityDF.loc[:,[hkiLise[0], hkiLise[1], hkiLise[2], hkiLise[3], hkiLise[4],
                               hkiLise[5], 'AQI-Lise','AQI-LiseType']]
    
    tepeköy = airQualityDF.loc[:,[hkiTepeköy[0], hkiTepeköy[1], hkiTepeköy[2], hkiTepeköy[3], hkiTepeköy[4],
                                  hkiTepeköy[5], 'AQI-Tepeköy','AQI-TepeköyType']]
#AQI-Santral Değişimi

    plt.figure(figsize=(15,10))
    plt.tight_layout()
    
    seabornInstance.distplot(santral['AQI-Santral'])
    
    
    f,ax=plt.subplots(figsize=(15,8))
    plt.plot(airQualityDF['Tarih'], airQualityDF['Santral-NOX'],color='r', label='NOX',alpha=0.8)
    plt.plot(airQualityDF['Tarih'],airQualityDF['Santral-SO2'],color='b', label='SO2',alpha=0.8)
    plt.legend(loc='upper left')
    plt.title('Santral için değişim')
    plt.show()
    
    # f,ax=plt.subplots(figsize=(15,8))
    # plt.plot(airQualityDF['Tarih'], airQualityDF['Lise-NOX'],color='r', label='NOX',alpha=0.8)
    # plt.plot(airQualityDF['Tarih'],airQualityDF['Lise-SO2'],color='b', label='SO2',alpha=0.8)
    # plt.legend(loc='upper left')
    # plt.title('Lise için değişim')
    # plt.show()
    
    # f,ax=plt.subplots(figsize=(15,8))
    # plt.plot(airQualityDF['Tarih'], airQualityDF['Tepeköy-NOX'],color='r', label='NOX',alpha=0.8)
    # plt.plot(airQualityDF['Tarih'],airQualityDF['Tepeköy-SO2'],color='b', label='SO2',alpha=0.8)
    # plt.legend(loc='upper left')
    # plt.title('Tepeköy için değişim')
    # plt.show()
    
    # seabornInstance.distplot(lise['AQI-Lise'])
    # seabornInstance.distplot(tepeköy['AQI-Tepeköy'])
    
    print(airQualityDF['Santral-NOX'].describe().T)
    print(airQualityDF['Santral-SO2'].describe().T)
    
    # print(lise["AQI-Lise"].describe().T)
    # print(tepeköy["AQI-Tepeköy"].describe().T)
    
    # print("santral::", santral["AQI-Santral"])
    # print("lise::", lise["AQI-Lise"])
    # print("tepeköy::", tepeköy["AQI-Tepeköy"])
           
##################################################################    
#Predict
    print("Predict--------------------------------------------------->")
    X = santral.iloc[:, 0:6]
    y = santral.iloc[:, 6]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    linreg = LinearRegression()  
    linreg.fit(X_train, y_train)
    
    print("X_train", len(X_train))
    print("X_test", len(X_test))
    print("y_train", len(y_train))
    print("y_test", len(y_test))
    
    acc_lin_train = round(linreg.score(X_train, y_train)*100,2)
    acc_lin_test = round(linreg.score(X_test, y_test)*100,2)
    print("Training Acc:  %  {}" .format(acc_lin_train))
    print("Test Acc:  %  {}" .format(acc_lin_test))

    y_pred = linreg.predict(X_test)

    y_pred = pd.Series(y_pred)
    df = pd.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': y_pred.values.flatten()})

    df1 = df.head(25)
    df1.plot(kind='bar',figsize=(16,10))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show() 

    


    
    



