# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_stata('ADULT.dta')
data.head()
data.info()
data.describe()
data.index
data.columns

data['ageNBR'] = data.srage_p1.apply(lambda x: int(x[0:2]) + 2)

sns.boxplot(x="srsex", y="ageNBR", hue = "smkcur", data=data, palette="PRGn")

usefulCols = ['hhsize_p1','srage_p1','ageNBR','srsex','racecn_p1','marit2','ad52','ad53','ad54', \
'disable','ae30','ad37w','ad40w','ad41w','ad42w','ae15','ae15a','ad32_p1','smkcur', \
'smoking','numcig','ac32','ac34_p1','ac35_p1','binge12','ac47_p1','hghti_p','wghtp_p',\
'bmi_p','rbmi','ovrwt','ah44','wrkst_p1','ab1']


limData = pd.DataFrame(data, columns = usefulCols)

for col in ['hhsize_p1','ageNBR','ad41w','ad42w','hghti_p','wghtp_p','bmi_p']:
    limData[col] = limData[col].apply(lambda x: float(str(x).strip()))

healthmap = {'EXCELLENT':1.0, 'FAIR':0.25, 'VERY GOOD':0.75, 'GOOD':0.5, 'POOR':0.0}
limData['HealthConditionNBR'] = limData.ab1.apply(lambda x: healthmap.get(x) if x in healthmap else x)

limData.rename(columns={'hhsize_p1': 'HouseHoldSizeNBR', 'srage_p1': 'AgeRangeCD', \
'ageNBR': 'AgeRoundedNBR', 'srsex':'SexCD', 'racecn_p1':'RaceCD', \
'marit2':'MaritalStatusCD', 'ad52':'DifficultyDressingFLG', 'ad53':'DifficultyGoingOutsideFLG', \
'ad54':'DifficultyWorkingFLG', 'disable':'DisabledFLG', 'ae30':'FluVaccineFLG', \
'ad37w':'Walked10MinTransportFLG', 'ad40w':'Walked10MinLeisureFLG', 'ad41w':'TimesWalked10MinNBR',\
 'ad42w':'MinLeisureWalkNBR', 'smkcur':'SmokerFLG', 'numcig':'CigCD', 'ac35_p1':'4DrinksCD',\
 'binge12':'BingeDrinkingCD', 'hghti_p':'HeightInchesNBR', 'wghtp_p':'WeightLBS', \
 'rbmi': 'BMICD', 'bmi_p':'BMINBR', 'ovrwt':'OverweightFLG', 'ah44':'LivingWSpouseFLG',\
 'wrkst_p1':'WorkingStatusCD', 'ab1':'HealthConditionCD'}, inplace=True)

usefulColsNBR=['HouseHoldSizeNBR', 'AgeRoundedNBR', 'TimesWalked10MinNBR','MinLeisureWalkNBR',\
'HeightInchesNBR', 'WeightLBS', 'BMINBR']

limData[usefulColsNBR].info()
limData.MinLeisureWalkNBR.value_counts()
limData.MinLeisureWalkNBR[limData.MinLeisureWalkNBR < 3000].hist()
sns.lmplot(x='MinLeisureWalkNBR', y="HealthConditionNBR", data=limData[limData.MinLeisureWalkNBR < 3000], palette="PRGn", y_jitter=.02)

for row in usefulColsNBR:
    sns.boxplot(x="HealthConditionCD", y=row, data=limData, palette="PRGn")
    plt.show()

for row in usefulColsNBR:
    sns.lmplot(x=row, y="HealthConditionNBR", data=limData, palette="PRGn", y_jitter=.02)
    plt.show()
    

limData.head()

limData.describe()
limData.info()

sns.pairplot(limData[usefulColsNBR])

limData.srage_p1.hist()

data.hhsize_p1.value_counts().plot(kind='bar')
data.srage_p1.plot(kind='bar')
data.rakedw0.hist()


