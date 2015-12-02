# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import seaborn as sns

### Importing and concatenating 2013 and 2014 data ###


data2014 = pd.read_stata('data/ADULT2014.dta')
data2013 = pd.read_stata('data/ADULT2013.dta')
data201112 = pd.read_stata('data/ADULT201112.dta')
data2009 = pd.read_stata('data/ADULT2009.dta')

'''201112 doesn't have: ac46 (sweet drink consumption), ac47 (water consumption),
ac48_p1 (milk), ah102_p (nights in a hospital) - I originally wanted these columns
but it is probably better to have 4 years worth of data compared to 2'''

data201112.rename(columns={'ac31': 'ac31_p1', 'ah102_p': 'ah102_p1', \
'aheduc': 'ahedc_p1', 'ak2':'ak2_p1', 'hhsize_p': 'hhsize_p1', 'ombsrreo':'ombsrr_p1', \
'srage_p':'srage_p1', 'povll':'povll_aca', 'wrkst':'wrkst_p1'}, inplace=True)

data2009.rename(columns={'ac31': 'ac31_p1', 'ah102_p': 'ah102_p1', \
'aheduc': 'ahedc_p1', 'ak2':'ak2_p1', 'hhsize_p': 'hhsize_p1', 'ombsrreo':'ombsrr_p1', \
'srage_p':'srage_p1', 'povll':'povll_aca', 'wrkst':'wrkst_p1'}, inplace=True)

data2014['SurveyYear'] = 2014
data2013['SurveyYear'] = 2013
data201112['SurveyYear'] = 2011
data2009['SurveyYear'] = 2009

#commonColumns = list(set(data2014.columns) & set(data2013.columns))
commonColumnsALL = list(set(data2014.columns) & set(data2013.columns) & set(data201112.columns) & set(data2009.columns) )

categoryColumns = list(data2014[commonColumnsALL].dtypes[data2014[commonColumnsALL].dtypes == 'category'].index)
for column in categoryColumns:
    data2014[column] = data2014[column].astype('object')
    data2013[column] = data2013[column].astype('object')
    data201112[column] = data201112[column].astype('object')
    data2009[column] = data2009[column].astype('object')



data = pd.concat([data2014[commonColumnsALL],data2013[commonColumnsALL],data201112[commonColumnsALL],data2009[commonColumnsALL]], ignore_index=True)
weights = pd.concat([data2014.rakedw0,data2013.rakedw0,data201112.rakedw0,data2009.rakedw0], ignore_index=True)

data.head()
data.info()
data.describe()
data.index
data.columns

### Potential Responses ###
#doctor's visits counts
data.ab34.value_counts()
#heart disease counts
data.ab34.value_counts()
#high blood pressure counts
data.ab29.value_counts()
#diabetes counts
data.ab22.value_counts()

usefulCols = ['acmdnum', 'ab1', 'ab17', 'ab22', 'ab29', 'ab30', 'ab34', 'ac11',\
'ac31_p1', 'ad42w', 'ad50', 'ad51', 'ad52', 'ad53',\
 'ad57', 'ag22', 'ah14','ah33new', 'ahedc_p1', 'aj29', 'ak2_p1', 'ak22_p', \
'astcur', 'binge12', 'bmi_p', 'disable', 'er', 'hghti_p', 'hhsize_p1', 'ins', \
'marit2', 'numcig', 'ombsrr_p1', 'povll_aca', 'smoking', \
'srage_p1', 'srsex', 'srtenr', 'ur_clrt2', 'usual', \
'wghtp_p', 'wrkst_p1']

data = pd.DataFrame(data, columns = usefulCols)

data.rename(columns={'ab1': 'HealthConditionCD', 
'ab17': 'AsthmaHistFLG', 
'ab22': 'DiabetesFLG', 
'ab29': 'HighBPFLG', 
'ab30': 'BloodPressureMedFLG', 
'ab34': 'HeartDiseaseFLG', 
'ac11': 'SodaNBR', 
'ac31_p1': 'FastFoodNBR', 
#'ac46': 'SweetFruitDrinksNBR', 
#'ac47_p1': 'WaterNBR', 
#'ac48_p1': 'MilkNBR', 
'acmdnum': 'DoctorsVisitsNBR', 
'ad42w': 'TimeWalkNBR', 
'ad50': 'VisionHearingProbFLG', 
'ad51': 'DifConcentratingFLG', 
'ad52': 'DifDressingFLG', 
'ad53': 'DifGoingOutsideFLG',
'ad57': 'LimitPhysActivityFLG', 
#'ae30': 'FluVaccineFLG', 
'ag22': 'ArmyFLG', 
#'ah102_p1': 'NightsInHospNBR', 
'ah14': 'PatientHospitalizedFLG',
'ah33new': 'BornUSFLG', 
'ahedc_p1': 'EducationCD', 
'aj29': 'NervousFLG', 
'ak22_p': 'IncomeNBR', 
'ak2_p1': 'NotWorkingReasonCD', 
#'ak23': 'HousingCD', 
'astcur': 'CurrentAsthmaFLG', 
'binge12': 'BingeCD', 
'bmi_p': 'BMINBR', 
'disable': 'DisabledFLG', 
'er': 'ERVisitsFLG', 
'hghti_p': 'HeightNBR', 
'hhsize_p1': 'HouseHoldSizeNBR',
'ins': 'InsuranceFLG', 
'marit2': 'MaritalCD', 
'numcig': 'NumCigCD', 
'ombsrr_p1': 'RaceCD', 
'povll_aca': 'PovertyLevelCD', 
'smoking': 'SmokingCD', 
'srage_p1': 'AgeRangeNBR', 
'srsex': 'SexFLG', 
'srtenr': 'OwnFLG', 
'ur_clrt2': 'RuralFLG', 
'usual': 'MedicalCareAccessFLG', 
'wghtp_p': 'WeightNBR', 
'wrkst_p1': 'WorkingStatusCD'}, inplace=True)

### Transforming specific columns ###
for col in ['HouseHoldSizeNBR', 'TimeWalkNBR','HeightNBR','WeightNBR','BMINBR']:
    data[col] = data[col].apply(lambda x: float(str(x).strip()))

healthmap = {'EXCELLENT':1.0, 'FAIR':0.25, 'VERY GOOD':0.75, 'GOOD':0.5, 'POOR':0.0}
data['HealthConditionNBR'] = data.HealthConditionCD.apply(lambda x: healthmap.get(x) if x in healthmap else x)

DoctorsVisitsNBRmap = {'1 TIMES':1, '1 TIME':1, '2 TIMES':2, '0 TIME':0, '3 TIMES':3, \
'4 TIMES':4, '9-12 TIMES':11, '6 TIMES':6, '5 TIMES':5, '7-8 TIMES':8, \
'13-24 TIMES': 18, '25+ TIMES':30, 'O TIME':0 }
data['DoctorsVisitsNBR'] = data.DoctorsVisitsNBR.apply(lambda x: DoctorsVisitsNBRmap.get(x) if x in DoctorsVisitsNBRmap else x)

EducationCDmap = {'AA OR AS DEGREE':0.4, 'GRADE 12/H.S. DIPLOMA':0.2, \
'NO FORMAL EDUCATION OR GRADE 1-8':0, 'GRADE 1-8':0, 'GRADE 9-11':0.1, 'MA OR MS DEGREE':0.8, \
'BA OR BS DEGREE':0.6, 'VOCATIONAL SCHOOL':0.4, 'SOME COLLEGE':0.3, \
'PH.D. OR EQUIVALENT':1, 'SOME GRAD. SCHOOL':0.7, 'NO FORMAL EDUCATION':0}
data['EducationCD'] = data.EducationCD.apply(lambda x: EducationCDmap.get(x) if x in EducationCDmap else x)

NervousFLGmap = {'A LITTLE OF THE TIME':0.25, 'NOT AT ALL':0, 'SOME OF THE TIME':0.5,
       'MOST OF THE TIME':0.75, 'ALL OF THE TIME':1, -2:0.5}
data['NervousFLG'] = data.NervousFLG.apply(lambda x: NervousFLGmap.get(x) if x in NervousFLGmap else x)

BingeCDmap = {'LESS THAN MONTHLY, MORE THAN ONCE A YEAR':6, \
'NO BINGE DRINKING PAST YEAR':0, 'ONCE A YEAR':1, 'MONTHLY':12, \
'DAILY OR WEEKLY': 100, 'LESS THAN WEEKLY BUT MORE THAN MONTHLY': 25}
data['BingeCD'] = data.BingeCD.apply(lambda x: BingeCDmap.get(x) if x in BingeCDmap else x)

NumCigCDmap = {'NONE':0, '11-19 CIGARETTES':15, '20 OR MORE': 30, \
'<=1 CIGARETTES':1, '2-5 CIGARETTES':3.5, '6-10 CIGARETTES': 8}
data['NumCigCD'] = data.NumCigCD.apply(lambda x: NumCigCDmap.get(x) if x in NumCigCDmap else x)

PovertyLevelCDmap = {'139%-249% FPL':195, '250%-399% FPL':325,'0-138% FPL':69,\
 '400%+ FPL':500, '139-249% FPL':195, '400% FPL AND ABOVE': 500, \
 '250-399% FPL':325, '300% FPL AND ABOVE':500, '100-199% FPL':150, '0-99% FPL':50, \
 '200-299% FPL':250}
data['PovertyLevelCD'] = data.PovertyLevelCD.apply(lambda x: PovertyLevelCDmap.get(x) if x in PovertyLevelCDmap else x)

AgeRangeNBRmap = {'26-29 YEARS':27.5, '70-74 YEARS':72, '35-39 YEARS':37,\
 '40-44 YEARS':42, '45-49 YEARS':47, '55-59 YEARS':57, '60-64 YEARS':62,\
 '65-69 YEARS':67, '75-79 YEARS':77, '30-34 YEARS':32, '50-54 YEARS':52,\
 '18-25 YEARS':21.5, '80-84 YEARS':82, '85+ YEARS':87}
data['AgeRangeNBR'] = data.AgeRangeNBR.apply(lambda x: AgeRangeNBRmap.get(x) if x in AgeRangeNBRmap else x)

RaceCDmap = {'HISPANIC':'HISPANIC', 'ASIAN ONLY, NH': 'ASIAN', \
'WHITE, NON-HISPANIC (NH)': 'WHITE', 'AFRICAN AMERICAN ONLY, NOT HISPANIC':'AA',\
'TWO OR MORE RACES, NH': 'OTHER', 'AMERICAN INDIAN/ALASKAN NATIVE ONLY, NH': 'OTHER',\
 'OTHER, NH':'OTHER', 'NATIVE HAWAIIAN/PACIFIC ISLANDER, NH': 'OTHER'}
data['RaceCD'] = data.RaceCD.apply(lambda x: RaceCDmap.get(x) if x in RaceCDmap else x)

SmokingCDmap = {'NEVER SMOKED REGULARLY':'NEVER', 'QUIT SMOKING':'FORMER',\
'CURRENTLY SMOKES': 'CURRENT'}
data['SmokingCD'] = data.SmokingCD.apply(lambda x: SmokingCDmap.get(x) if x in SmokingCDmap else x)

WorkingStatusCDmap = {'FULL-TIME EMPLOYMENT (21+ HRS/WEEK)':'FULLTIME',\
 'UNEMPLOYED, NOT LOOKING FOR WORK':'UNEMPLOYED',\
 'PART-TIME EMPLOYMENT (0-20 HRS/WEEK)':'PARTTIME', \
 'UNEMPLOYED, LOOKING FOR WORK':'UNEMPLOYED', 'OTHER EMPLOYED':'PARTTIME',\
 'EMPLOYED, NOT AT WORK':'PARTTIME'}
data['WorkingStatusCD'] = data.WorkingStatusCD.apply(lambda x: WorkingStatusCDmap.get(x) if x in WorkingStatusCDmap else x)

MaritalCDmap = {'LIVING W/ PARTNER':'MARRIED', 'MARRIED':'MARRIED',\
'WID/SEP/DIV':'DIVORCEDorWIDOWED', 'NEVER MARRIED':'NEVERMARRIED'}
data['MaritalCD'] = data.MaritalCD.apply(lambda x: MaritalCDmap.get(x) if x in MaritalCDmap else x)

#data.NightsInHospNBR = np.where(data.NightsInHospNBR == -1, 0, data.NightsInHospNBR)
#data.WaterNBR = np.where(data.WaterNBR == 'LESS THAN ONE GLASS', 0.5, data.WaterNBR)
data.BloodPressureMedFLG = np.where(data.BloodPressureMedFLG == -1, 'NO', data.BloodPressureMedFLG)
data.TimeWalkNBR = np.where(data.TimeWalkNBR == -1, 0, data.TimeWalkNBR)
data.BornUSFLG = np.where(data.BornUSFLG == 'BORN IN U.S.', True, False)
data.SexFLG = np.where(data.SexFLG  == 'MALE', True, False)
data.OwnFLG = np.where(data.OwnFLG == 'OWN', True, False)
data.RuralFLG = np.where(data.RuralFLG == 'RURAL', True, False)
data.DiabetesFLG = np.where(data.DiabetesFLG == 'BORDERLINE OR PRE-DIABETES', 'YES', data.DiabetesFLG)
data.HighBPFLG = np.where(data.HighBPFLG == 'BORDERLINE HYPERTENSION', 'YES', data.HighBPFLG)
data.CurrentAsthmaFLG = np.where(data.CurrentAsthmaFLG == 'CURRENT ASTHMA', True, False)
data.DisabledFLG = np.where(data.DisabledFLG == 'DISABLED', True, False)
data.WorkingStatusCD = np.where(data.NotWorkingReasonCD == 'RETIRED', 'RETIRED', data.WorkingStatusCD)
data.WorkingStatusCD = np.where(data.NotWorkingReasonCD == 'GOING TO SCHOOL/STUDENT', 'FULLTIME', data.WorkingStatusCD)
data.WorkingStatusCD = np.where(data.NotWorkingReasonCD == 'TAKING CARE OF HOUSE OR FAMILY', 'FAMILYorVACATION', data.WorkingStatusCD)
data.WorkingStatusCD = np.where(data.NotWorkingReasonCD == 'ON PLANNED VACATION', 'FAMILYorVACATION', data.WorkingStatusCD)
data.PatientHospitalizedFLG = np.where(data.PatientHospitalizedFLG == -1, 'YES', data.PatientHospitalizedFLG)
data.TimeWalkNBR = np.where(data.TimeWalkNBR == 3600, 360, data.TimeWalkNBR)

data.drop(['HealthConditionCD','NotWorkingReasonCD'], axis=1, inplace=True)

YesNoCols = ['AsthmaHistFLG', 'DiabetesFLG', 'HighBPFLG', 'BloodPressureMedFLG',\
 'VisionHearingProbFLG', 'DifConcentratingFLG', 'LimitPhysActivityFLG', 'ArmyFLG'\
 , 'ERVisitsFLG', 'HeartDiseaseFLG', 'MedicalCareAccessFLG', 'InsuranceFLG', \
 'DifDressingFLG', 'DifGoingOutsideFLG', 'PatientHospitalizedFLG']

for col in YesNoCols:
    data[col] = np.where(data[col] == 'YES', True, False)

data = pd.concat([pd.get_dummies(data, columns = ['RaceCD', 'SmokingCD', 'WorkingStatusCD','MaritalCD'], prefix_sep='')], axis = 1)

objectCols = ['SodaNBR','FastFoodNBR','IncomeNBR']
for col in objectCols:
    data[col] = data[col].astype(float)

#data.to_csv('dataAll_transformed.csv', index=False)

data.info()
data.describe()

###############################################################################
### Exploring the data ###

for col in list(data.columns):
    print col
    print data[col].value_counts().iloc[0:10]
    print

data.DoctorsVisitsNBR.hist()

data.DoctorsVisitsNBR.count()

plt.figure()
plt.hist(data.DoctorsVisitsNBR, normed=True, bins=30)
plt.xlabel('Number of Doctors Visits')
plt.ylabel('Relative Frequency')
plt.title('Histogram of of Doctors Visits (relative)')

data.DoctorsVisitsNBR.describe()
data.boxplot(column="DoctorsVisitsNBR")

plt.figure(figsize=(17,10))
sns.heatmap(data.corr())

plt.figure(figsize=(25,25))
sns.pairplot(data)

plt.figure(figsize=(15,15))
sns.pairplot(data[['HealthConditionNBR', 'AgeRangeNBR', 'IncomeNBR','BMINBR',\
'TimeWalkNBR', 'SodaNBR', 'FastFoodNBR', 'HouseHoldSizeNBR',\
'DoctorsVisitsNBR']], hue='DoctorsVisitsNBR')

    
#This takes several minutes
for col in data.columns:
    sns.lmplot(x=col, y="DoctorsVisitsNBR", data=data, x_jitter = 0.2, y_jitter = 0.5)
    plt.show()

#Exploring some of the most correlated fields
for col in ['PatientHospitalizedFLG', 'LimitPhysActivityFLG', 'HealthConditionNBR', 'ERVisitsFLG']:
    sns.lmplot(x=col, y="DoctorsVisitsNBR", data=data, x_jitter = 0.2, y_jitter = 0.5)
    plt.show()

#Exploring some of the most correlated fields
for col in ['PatientHospitalizedFLG', 'LimitPhysActivityFLG', 'HealthConditionNBR', 'ERVisitsFLG']:
    sns.violinplot(x=col, y="DoctorsVisitsNBR", \
    data=data, split=True, inner="quart")
    plt.show()

for col in ['PatientHospitalizedFLG', 'LimitPhysActivityFLG', 'HealthConditionNBR', 'ERVisitsFLG']:
    sns.boxplot(x=col, y="DoctorsVisitsNBR", data=data)
    plt.show()

#bins = [0,25,30,35,40,45,50,55,60,65,70,75,80,85,100]
#data['AgeRangeCD'] = pd.cut(data.AgeRangeNBR, bins = bins)
#data['AgeRangeCD'].unique()


###############################################################################

###############################################################################

##### BELOW CODE PREDICTS IF DoctorsVisitsNBR is Greater than a specific X ####

###############################################################################

###############################################################################

###############################################################################
##### LOGISTIC REGRESSION  WITH > 11 DOCTORS VISITS #####

from sklearn import metrics
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from time import time
from scipy.stats import randint as sp_randint

data.DoctorsVisitsNBR.value_counts(normalize = True, sort=False)
data['DoctorsVisitsGTX'] = np.where(data.DoctorsVisitsNBR > 11, True, False)
'''dummy answer is 6.15% excluding 2009 data and 6.32% with 2009 data'''
1- data.DoctorsVisitsGTX.mean()

#data.ix[:,1:].to_csv('Outputs/dataAll_transformed_classification.csv', index=False)

plt.figure(figsize=(15,15))
sns.pairplot(data[['HealthConditionNBR', 'AgeRangeNBR', 'IncomeNBR','BMINBR',\
'TimeWalkNBR', 'SodaNBR', 'FastFoodNBR', 'HouseHoldSizeNBR','DoctorsVisitsNBR',\
'DoctorsVisitsGTX']], hue='DoctorsVisitsGTX')

features, response = data.ix[:,1:-1], data.DoctorsVisitsGTX
X, y = features, response 

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

'''initial fits below to explore the data'''
from sklearn.linear_model import LogisticRegression

'''running on train and test data'''
logreg = LogisticRegression(C=1e9)
logreg.fit(X_train, y_train)

print 'intercept and coefficient'
print logreg.intercept_ 
print logreg.coef_

y_preds = logreg.predict(X_test)

testResults = pd.DataFrame(zip(y_preds, y_test), columns = ['Prediction','Actual'])
testResults['Correct'] = testResults.Prediction == testResults.Actual 

'''results in 93.91% accuracy'''
print 'accuracy:'
print metrics.accuracy_score(testResults.Actual, testResults.Prediction)
print 'classification report:'
print metrics.classification_report(testResults.Actual, testResults.Prediction)

'''cross-validated accuracy of 93.85% without scale'''
logreg = LogisticRegression(C=1e9)
scores = cross_validation.cross_val_score(logreg, X, y, scoring='accuracy', cv=5)
print scores.mean()

'''cross-validated accuracy of 93.71% with scale (nearly identical result)
   Note: took 10 seconds to run on all data
'''
logreg = LogisticRegression(C=1e9)
start = time()
scores = cross_validation.cross_val_score(logreg, X_scaled, y, scoring='accuracy', cv=5)
print time() - start
print scores.mean()

'''cross-validated accuracy of [__]% with scale (nearly identical result)
   Note: took 10 seconds to run on all data
'''
logreg = LogisticRegression(C=1e9, class_weight = 'balanced')
start = time()
scores = cross_validation.cross_val_score(logreg, X_scaled, y, scoring='accuracy', cv=5)
print time() - start
print scores.mean()

'''train test split:
   Note: took 10 seconds to run on all data
'''
logreg = LogisticRegression(C=1e9)
start = time()
scores = cross_validation.cross_val_score(logreg, X_scaled, y, scoring='roc_auc', cv=5)
print time() - start
print scores.mean()


logreg.fit(X_train_scaled, y_train)

y_preds = logreg.predict(X_test_scaled)

from nltk import ConfusionMatrix
print ConfusionMatrix(list(y_test), list(y_preds))

## ROC CURVES and AUC

# plot ROC curve
probs = logreg.predict_proba(X_test_scaled)[:, 1]
plt.hist(probs)
plt.cla()

fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')

# calculate AUC
print metrics.roc_auc_score(y_test, probs)

probs = logreg.predict_proba(X_test_scaled)[:, 1]
preds = np.where(probs > 0.2, 1, 0)
confusion = ConfusionMatrix(list(y_test), list(preds))
print confusion 
PredictedTrueOfTotal = (confusion[0,1] + float(confusion[1,1])) / len(y_test)
print PredictedTrueOfTotal 
TPR = confusion[1,1] / (confusion[1,0] + float(confusion[1,1]))
print TPR 
Accuracy = (confusion[0,0] + float(confusion[1,1])) / len(y_test)
print Accuracy 

''' Trying to identify for caring for 10% of the population, what % of positive
cases can I cover'''

PredictedTrueOfTotalList = []
TPRList = []
AccuracyList = []
threshold = range(1,100,1) 
threshold = [x / 100.0 for x in threshold]
for i in threshold:
    preds = np.where(probs > i, 1, 0)
    confusion = ConfusionMatrix(list(y_test), list(preds))
    PredictedTrueOfTotal = (confusion[0,1] + float(confusion[1,1])) / len(y_test)
    PredictedTrueOfTotalList.append(PredictedTrueOfTotal)
    TPR = confusion[1,1] / (confusion[1,0] + float(confusion[1,1]))
    TPRList.append(TPR)
    Accuracy = (confusion[0,0] + float(confusion[1,1])) / len(y_test)
    AccuracyList.append(Accuracy)

plt.plot(PredictedTrueOfTotalList, TPRList, label = '% of total predicted positive')
plt.axvline(x=.1, ymin=0, ymax=1, color='r', ls = '--')
plt.xlabel('% of Total Predicted True')
plt.ylabel('True Positive Rate')

#see below which threshold resulted in positive 
pd.DataFrame(zip(PredictedTrueOfTotalList, threshold, TPRList), columns = ['PredictedTrueOfTotalList', 'threshold', 'TPRList'])

''' OLD code to look at threshold versus TPR, accuract, etc.
plt.plot(threshold, PredictedTrueOfTotalList, label = '% of total predicted positive')
plt.plot(threshold, TPRList, label = 'TPR')
plt.plot(threshold, AccuracyList, label = 'Accuracy')
plt.legend()
plt.xlabel('Threshold')
plt.ylabel('Percentage')
'''

###############################################################################
#### Penelized Logistic Regression ####

logreg = LogisticRegression(penalty ='l1', C=1, n_jobs=-1)
start = time()
scores = cross_validation.cross_val_score(logreg, X_scaled, y, scoring='accuracy', cv=5)
print time() - start
print scores.mean()

'''took a few mintues to run'''
from sklearn.grid_search import GridSearchCV
logreg = LogisticRegression(n_jobs=-1)
penalty = ['l1','l2']
c = [0.001, 0.01, 0.1, 1, 10, 100]
param_grid = dict(penalty=penalty, C=c)
grid = GridSearchCV(logreg, param_grid, cv=5, scoring='accuracy')
grid.fit(X_scaled, y)

grid_mean_scoresl1 = [result[1] for result in grid.grid_scores_ if result[0]['penalty'] == 'l1']
grid_mean_scoresl2 = [result[1] for result in grid.grid_scores_ if result[0]['penalty'] == 'l2']

# plot the results
plt.figure()
plt.plot(c, grid_mean_scoresl1, label = 'l1')
plt.plot(c, grid_mean_scoresl2, label = 'l2')
plt.legend()
plt.xscale('log')
plt.yscale('linear')
plt.title('l1 and l2 Regression')

'''best cross validated score is 93.718% with L1 penalty an C of 0.1'''
grid.best_score_     # shows us the best score
grid.best_params_    # shows us the optimal parameters
grid.best_estimator_ # this is the actual model


###############################################################################
#### Naive Bayes ####
from sklearn.naive_bayes  import GaussianNB
nb = GaussianNB()
start = time()
scores = cross_validation.cross_val_score(nb, X_scaled, y, scoring='accuracy', cv=5)
print time() - start
print scores.mean()

nb.fit(X_train_scaled, y_train)

y_preds = nb .predict(X_test_scaled)

from nltk import ConfusionMatrix
print ConfusionMatrix(list(y_test), list(y_preds))

probs = nb.predict_proba(X_test_scaled)[:, 1]
preds = np.where(probs > 0.2, 1, 0)



###############################################################################
#### Decision Trees ####
from sklearn import tree
from sklearn.cross_validation import train_test_split

features, response = data.ix[:,1:-1], data.DoctorsVisitsGTX
X, y = features, response 


X_train, X_test, y_train, y_test = train_test_split(features, response, random_state=1)


from sklearn.tree import DecisionTreeClassifier
ctree = tree.DecisionTreeClassifier(max_depth = 5, random_state=1)
ctree.fit(X_train, y_train)

'''note: lowest cross validated rmse is 56.97% with max_depth of 5'''
from sklearn.cross_validation import cross_val_score
ctree = tree.DecisionTreeClassifier(random_state=1)
cross_val_score(ctree, X_train, y_train, cv=10, scoring='roc_auc').mean()

#gathering importance
ctree = tree.DecisionTreeClassifier(max_depth=5, random_state=1)
ctree.fit(X_train, y_train)

feature_cols = features.columns
pd.DataFrame({'feature':feature_cols, 'importance':ctree.feature_importances_})

from sklearn.tree import export_graphviz
with open("DoctorsVisits_tree.dot", 'wb') as f:
    f = export_graphviz(ctree, out_file=f, feature_names=feature_cols)
#below code didn't work - converted with GVEdit GUI
from os import system
system("dot -Tpng DoctorsVisits_tree.dot -o DoctorsVisits_tree.png")


###############################################################################
#### Random forests ####
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import RandomizedSearchCV

rfclf = RandomForestClassifier(n_estimators=100, max_features='auto', oob_score=True, random_state=1)
rfclf.fit(features, response)

cross_val_score(rfclf, features, response, cv=10, scoring='accuracy')

pd.DataFrame({'feature':feature_cols, 'importance':rfclf.feature_importances_})

float(rfclf.oob_score_ )

### grid search ###
rfclf = RandomForestClassifier(oob_score=True, random_state=1, n_jobs = -1)
n_estimators = range(100, 600, 100)
depth_range = range(5, 30, 2)
param_grid = dict(n_estimators=n_estimators, max_depth=depth_range)
grid = GridSearchCV(rfclf, param_grid, cv=5, scoring='accuracy')
grid.fit(features, response)

grid_mean_scores = [result[1] for result in grid.grid_scores_]

# Plot the results of the grid search
plt.figure()
plt.plot(n_estimators, grid_mean_scores)
plt.hold(True)
plt.grid(True)
plt.plot(grid.best_params_['max_depth'], grid.best_score_, 'ro', markersize=12, markeredgewidth=1.5,
         markerfacecolor='None', markeredgecolor='r')

# Get the best estimator
best = grid.best_estimator_

''' Best parameters from grid search are listed below:
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=29, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=400, n_jobs=-1,
            oob_score=True, random_state=1, verbose=0, warm_start=False
'''

'''ideal feaures results in 93.9% accuracy'''
cross_val_score(best, features, response, cv=5, scoring='accuracy')

'''best features are AgeRangeNBR, IncomeNBR, BMINBR, SodaNBR, FastFoodNBR, 
   TimeWalkNBR, PatientHospitalizedFLG, EducationCD'''

pd.DataFrame({'feature':feature_cols, 'importance':best.feature_importances_})

### Additional Random Forest Parameter Optimization ###
from sklearn.grid_search import RandomizedSearchCV
from time import time
from scipy.stats import randint as sp_randint

# specify parameters and distributions to sample from
param_dist = {"n_estimators": range(200, 800, 100),
              "max_depth": [5, None],
              "max_features": sp_randint(10, 52),
              "min_samples_split": sp_randint(2, 100),
              "min_samples_leaf": sp_randint(1, 50),
              "bootstrap": [True, False]}

# run randomized search
rf = RandomForestClassifier(oob_score=False, random_state=1, criterion = 'gini', \
n_jobs = -1)

n_iter_search = 20
random_search = RandomizedSearchCV(rf, param_distributions=param_dist,\
n_iter=n_iter_search)

start = time()
random_search.fit(X, y)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))

random_search.grid_scores_
random_search.best_score_
random_search.best_estimator_
random_search.best_params_


'''First iteration of 50 searches run on all data except 2009:'''
'''best search resulted in 93.94% accuracy and the following parameters and 
took 4.2 hours to run:

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features=49, max_leaf_nodes=None,
            min_samples_leaf=9, min_samples_split=16,
            min_weight_fraction_leaf=0.0, n_estimators=300, n_jobs=-1,
            oob_score=False, random_state=1, verbose=0, warm_start=False)

'''           

'''First iteration of 20 searches run on all data (including 2009):'''
'''best search resulted in 93.712% accuracy and the following parameters and 
took 1.5 hours to run:

RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
            max_depth=None, max_features=26, max_leaf_nodes=None,
            min_samples_leaf=48, min_samples_split=16,
            min_weight_fraction_leaf=0.0, n_estimators=700, n_jobs=-1,
            oob_score=False, random_state=1, verbose=0, warm_start=False)

'''           


'''run best model (parameters listed above) on test data'''
rf = random_search.best_estimator_
rf.fit(X_scaled, y)

feature_importanceRFR1 = pd.DataFrame({'feature':feature_cols, 'importance':rf.feature_importances_})
feature_importanceRFR1.set_index('feature', inplace=True)
feature_importanceRFR1.sort(columns = 'importance', ascending=True).plot(kind = 'barh', figsize=(8,12))
feature_importanceRFR1.sort()

'''look at ROC_AUC curve'''
rf = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
            max_depth=None, max_features=26, max_leaf_nodes=None,
            min_samples_leaf=48, min_samples_split=16,
            min_weight_fraction_leaf=0.0, n_estimators=700, n_jobs=-1,
            oob_score=False, random_state=1, verbose=0, warm_start=False)
rf.fit(X_train_scaled, y_train)
probs = rf.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')

# calculate AUC
print metrics.roc_auc_score(y_test, probs)

# adjust threshold
preds = np.where(probs > 0.1, 1, 0)
print ConfusionMatrix(list(y_test), list(preds))


''' Trying to identify for caring for 10% of the population, what % of positive
cases can I cover'''
probs = rf.predict_proba(X_test_scaled)[:, 1]
PredictedTrueOfTotalList = []
TPRList = []
AccuracyList = []
threshold = range(1,20,1) 
threshold = [x / 20.0 for x in threshold]
for i in threshold:
    preds = np.where(probs > i, 1, 0)
    confusion = ConfusionMatrix(list(y_test), list(preds))
    PredictedTrueOfTotal = (confusion[0,1] + float(confusion[1,1])) / len(y_test)
    PredictedTrueOfTotalList.append(PredictedTrueOfTotal)
    TPR = confusion[1,1] / (confusion[1,0] + float(confusion[1,1]))
    TPRList.append(TPR)
    Accuracy = (confusion[0,0] + float(confusion[1,1])) / len(y_test)
    AccuracyList.append(Accuracy)

plt.plot(PredictedTrueOfTotalList, TPRList, label = '% of total predicted positive')
plt.axvline(x=.1, ymin=0, ymax=1, color='r', ls = '--')
plt.xlabel('% of Total Predicted True')
plt.ylabel('True Positive Rate')

###############################################################################
############## XGBoost ############
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.cross_validation import KFold, train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.grid_search import GridSearchCV

#first test of xg boost
dtrain = xgb.DMatrix(X, label=y)
                        
param = {'n_estimators':100, 'max_depth':2, 'eta':1, 'silent':True, \
        'objective':'binary:logistic', 'learning_rate':0.025}
num_round = 10

xgb.cv(param, dtrain, num_round, nfold=5,
       metrics={'error'}, seed = 0)

'''
clf = xgb.XGBClassifier(n_estimators=25,
                        nthread=-1,
                        max_depth=10,
                        learning_rate=0.025,
                        silent=True,
                        subsample=0.8,
                        colsample_bytree=0.8)
xgb_model = clf.fit(X, y, eval_metric="auc")
'''

'''grid search below results in best accuracy of 93.740% and took 2 hours to run'''
print("Parameter optimization")
xgb_model = xgb.XGBClassifier()
clf = GridSearchCV(xgb_model,
                   {'max_depth': range(2, 12, 2),
                    'n_estimators': [200, 400],
                    'learning_rate':[0.01, 0.1, .5]}, verbose=1)
clf.fit(X,y)
print(clf.best_score_)
print(clf.best_params_)

'''best paramerters: 
{'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 2
'''

clf = xgb.XGBClassifier(clf.best_params_)
# Or you can use: clf = xgb.XGBClassifier(**param_dist)

clf.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)], 
        eval_metric='error',
        verbose=True)

evals_result = clf.evals_result()


import pickle

# The sklearn API models are picklable
print("Pickling sklearn API models")
# must open in binary format to pickle
pickle.dump(clf, open("best_xgboost.pkl", "wb"))
clf2 = pickle.load(open("best_xgboost.pkl", "rb"))



###############################################################################
############## SVM ############
from sklearn import svm

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
X_scaled = scaler.fit_transform(X)

#### Initial Fit ####

clf = svm.SVC(C=1.0, kernel='rbf') # ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
#clf.fit(X_scaled, y)

'''On all data (including 2009) - average accuracy of 93.68% for rbf; [__]% for linear'''
'''On 2011-2014 dataset - average accuracy of 93.88% for rbf; 93.84% for linear 
   and 93.58% for ploy (3 degrees)'''
crossValScore = cross_val_score(clf, X_scaled, y, cv=3, scoring='accuracy')
crossValScore.mean()


###############################################################################
#########   PCA   ##########
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = decomposition.PCA(n_components=X_scaled.shape[1])
X_r = pca.fit_transform(X_scaled)

# Percentage of variance explained for each components
print('explained variance ratio: %s'
      % str(pca.explained_variance_ratio_))

plt.cla()
plt.plot(pca.explained_variance_ratio_)
plt.title('Variance explained by each principal component')
plt.ylabel(' % Variance Explained')
plt.xlabel('Principal component')

### REPEAT LOGISTIC REGRESSION WITH SCALING AND PCA ###
'''7 components based on elbow rule from PCA chart above '''
pca = decomposition.PCA(n_components=7)
X_r = pca.fit_transform(X_scaled)

'''cross-validated accuracy of 93.78% with 7 components'''
logreg = LogisticRegression(C=1e9)
scores = cross_validation.cross_val_score(logreg, X_r, y, scoring='accuracy', cv=5)
print scores.mean()

###############################################################################
### KNN ### 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
features, response = data.ix[:,1:-1], data.DoctorsVisitsGTX
X, y = features, response 

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
X_scaled = scaler.fit_transform(X)

'''running on train / test data to determine optimal k value'''
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
knn = KNeighborsClassifier(n_neighbors=101)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)

'''took 15 hours to run'''
from sklearn.grid_search import GridSearchCV
knn = KNeighborsClassifier()
k_range = range(31, 200, 5)
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid.fit(X_scaled, y)

grid.grid_scores_
grid_mean_scores = [result[1] for result in grid.grid_scores_]

# plot the results
plt.figure()
plt.plot(k_range, grid_mean_scores)

'''best cross validated score is 93.862% at k of 29.  Also had peak at k of 51 (93.861%)'''
grid.best_score_     # shows us the best score
grid.best_params_    # shows us the optimal parameters
grid.best_estimator_ # this is the actual model

clf = KNeighborsClassifier(n_neighbors=29)

'''On all data (including 2009) - average accuracy of 93.67% '''
crossValScore = cross_val_score(clf, X_scaled, y, cv=3, scoring='accuracy')
crossValScore.mean()


###############################################################################
### SGD ### 
from sklearn import linear_model

features, response = data.ix[:,1:-1], data.DoctorsVisitsGTX
X, y = features, response 

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
X_scaled = scaler.fit_transform(X)


from sklearn import metrics
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from time import time

### Linear SVM (default 'hinge' loss funcation) ### 
'''cross-validated accuracy of 93.3% with scale (nearly identical result)
   Note: took 1.6 seconds to run on all data'''
clf = linear_model.SGDClassifier()
start = time()
scores = cross_validation.cross_val_score(clf, X_scaled, y, scoring='accuracy', cv=5)
print time() - start
print scores.mean()

### Log Regression ### 
'''cross-validated accuracy of 92.9% with scale (lower than with typical linear
    regression)
   Note: took 2 seconds to run on all data'''
clf = linear_model.SGDClassifier(loss='log', weights = weights)
start = time()
scores = cross_validation.cross_val_score(clf, X_scaled, y, scoring='accuracy', cv=5)
print time() - start
print scores.mean()

'''IF I WANTED TO USE SGD, I WOULD NEED TO OPTIMZE LOG REGRESSION AND SVM MODELS'''

''' Trying to identify for caring for 10% of the population, what % of positive
cases can I cover'''
clf = linear_model.SGDClassifier(loss='log')
clf.fit(X_train_scaled, y_train)
probs = clf.predict_proba(X_test_scaled)[:, 1]
PredictedTrueOfTotalList = []
TPRList = []
AccuracyList = []
threshold = range(1,50,1) 
threshold = [x / 50.0 for x in threshold]
for i in threshold:
    preds = np.where(probs > i, 1, 0)
    confusion = ConfusionMatrix(list(y_test), list(preds))
    PredictedTrueOfTotal = (confusion[0,1] + float(confusion[1,1])) / len(y_test)
    PredictedTrueOfTotalList.append(PredictedTrueOfTotal)
    TPR = confusion[1,1] / (confusion[1,0] + float(confusion[1,1]))
    TPRList.append(TPR)
    Accuracy = (confusion[0,0] + float(confusion[1,1])) / len(y_test)
    AccuracyList.append(Accuracy)

plt.plot(PredictedTrueOfTotalList, TPRList, label = '% of total predicted positive')
plt.axvline(x=.1, ymin=0, ymax=1, color='r', ls = '--')
plt.xlabel('% of Total Predicted True')
plt.ylabel('True Positive Rate')

###############################################################################
### Gradient Boosted Trees ### 
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,\
max_depth=1, random_state=0).fit(X_train_scaled, y_train)

clf.score(X_test_scaled, y_test)

### Additional Random Forest Parameter Optimization ###
from sklearn.grid_search import RandomizedSearchCV
from time import time
from scipy.stats import randint as sp_randint

# specify parameters and distributions to sample from
param_dist = {"n_estimators": range(50, 550, 100),
              "max_depth": range(1, 10, 1),
              "learning_rate": [.001,.01,.1,1],
              "min_samples_split": sp_randint(2, 100),
              "min_samples_leaf": sp_randint(1, 50),
              "loss": ["deviance", "exponential"]}

# run randomized search
clf = GradientBoostingClassifier(random_state=1)

n_iter_search = 50
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,\
n_iter=n_iter_search)

start = time()
random_search.fit(X, y)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))

random_search.grid_scores_
random_search.best_score_
random_search.best_estimator_
random_search.best_params_

'''best search resulted in 93.73% accuracy and the following parameters and 
took 6.8 hours to run:

GradientBoostingClassifier(init=None, learning_rate=0.1, loss='deviance',
              max_depth=2, max_features=None, max_leaf_nodes=None,
              min_samples_leaf=33, min_samples_split=85,
              min_weight_fraction_leaf=0.0, n_estimators=250,
              random_state=1, subsample=1.0, verbose=0, warm_start=False)


'''           
'''run best model (parameters listed above) on test data'''
clf = random_search.best_estimator_
clf.fit(X, y)

'''ideal feaures results in 93.737% accuracy'''
cross_val_score(clf, X, y, cv=5, scoring='accuracy')

'''look at ROC_AUC curve'''
clf = GradientBoostingClassifier(init=None, learning_rate=0.1, loss='deviance',
              max_depth=2, max_features=None, max_leaf_nodes=None,
              min_samples_leaf=33, min_samples_split=85,
              min_weight_fraction_leaf=0.0, n_estimators=250,
              random_state=1, subsample=1.0, verbose=0, warm_start=False)
clf.fit(X_train_scaled, y_train)
probs = clf.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')

# calculate AUC
print metrics.roc_auc_score(y_test, probs)

# adjust threshold
preds = np.where(probs > 0.1, 1, 0)
print ConfusionMatrix(list(y_test), list(preds))


''' Trying to identify for caring for 10% of the population, what % of positive
cases can I cover'''
probs = clf.predict_proba(X_test_scaled)[:, 1]
PredictedTrueOfTotalList = []
TPRList = []
AccuracyList = []
threshold = range(1,20,1) 
threshold = [x / 20.0 for x in threshold]
for i in threshold:
    preds = np.where(probs > i, 1, 0)
    confusion = ConfusionMatrix(list(y_test), list(preds))
    PredictedTrueOfTotal = (confusion[0,1] + float(confusion[1,1])) / len(y_test)
    PredictedTrueOfTotalList.append(PredictedTrueOfTotal)
    TPR = confusion[1,1] / (confusion[1,0] + float(confusion[1,1]))
    TPRList.append(TPR)
    Accuracy = (confusion[0,0] + float(confusion[1,1])) / len(y_test)
    AccuracyList.append(Accuracy)

plt.plot(PredictedTrueOfTotalList, TPRList, label = '% of total predicted positive')
plt.axvline(x=.1, ymin=0, ymax=1, color='r', ls = '--')
plt.xlabel('% of Total Predicted True')
plt.ylabel('True Positive Rate')

###############################################################################
### Feed Forward Neural Network ### 

from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from numpy import ravel


# pybrain has its own data sample class that we must add
# our training and test set to
ds = ClassificationDataSet(X_scaled.shape[1], 1 , nb_classes=2)
for k in xrange(X_scaled.shape[0]): 
    ds.addSample(ravel(X_scaled[k]),y[k])

# their equivalent of train test split
test_data, training_data = ds.splitWithProportion( 0.25 )


len(training_data.data['input'][0])

test_data

# pybrain's version of dummy variables '''ASK ABOUT THIS - IS THIS NEEDED?'''
test_data._convertToOneOfMany( )
training_data._convertToOneOfMany( )


training_data['input']
training_data['target']

test_data.indim
test_data.outdim

# instantiate the model
fnn = buildNetwork( training_data.indim, 64, training_data.outdim, outclass=SoftmaxLayer )
trainer = BackpropTrainer( fnn, dataset=training_data, momentum=0.1, learningrate=0.1 , verbose=True, weightdecay=0.01) 

'''93.61% accuracy after 10 epochs'''
# change the number of eopchs to try to get better results!
trainer.trainEpochs (50)
print 'Percent Error on Test dataset: ' , \
        percentError( trainer.testOnClassData (
           dataset=test_data )
           , test_data['class'] )


###############################################################################
### Voting Classifiers ### 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features=49, max_leaf_nodes=None,
            min_samples_leaf=9, min_samples_split=16,
            min_weight_fraction_leaf=0.0, n_estimators=300, n_jobs=-1,
            oob_score=False, random_state=1, verbose=0, warm_start=False)
clf3 = GradientBoostingClassifier(init=None, learning_rate=0.1, loss='deviance',
              max_depth=2, max_features=None, max_leaf_nodes=None,
              min_samples_leaf=33, min_samples_split=85,
              min_weight_fraction_leaf=0.0, n_estimators=250,
              random_state=1, subsample=1.0, verbose=0, warm_start=False)

eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gb', clf3)], voting='hard')

for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'Gradient Boosted Trees', 'Ensemble']):
    scores = cross_validation.cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')
    print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), label))

'''


'''


###############################################################################

###############################################################################

###############################################################################

###############################################################################

### BELOW CODE IS FOR PREDICTING NUMBER OF DOCOTRS VISITS -USING REGRESSION ###

###############################################################################

###############################################################################

###############################################################################

#pd.concat([data[data.HeartDiseaseFLG == False].mean(axis = 0), data[data.HeartDiseaseFLG == True].mean(axis = 0)], axis = 1)

'''base rmse to beat is 5.665'''
from sklearn import metrics
y = np.empty(data.shape[0])
y.fill(data.DoctorsVisitsNBR.mean())
np.sqrt(metrics.mean_squared_error(data.DoctorsVisitsNBR, y))



###############################################################################
### LINEAR REGRESSION ###
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.formula.api as smf
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split

'''initial fits below to explore the data'''
data.head()
features, response = data.ix[:,1:], data.DoctorsVisitsNBR
X, y = data.ix[:,1:], data.DoctorsVisitsNBR

'''running on train and test data'''
### spliting data into train and test set for feature selection ###
def train_test_rmse(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)
    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))

'''all features results in rmse of 4.903'''
train_test_rmse(features, response)


'''cross-validated mean squared error'''
linreg = LinearRegression()
scores = cross_validation.cross_val_score(linreg, features, response, scoring='mean_squared_error', cv=5)
print 'cross-validated root mean squared error - 4.918'
print np.sqrt(-scores.mean())

'''running on all data'''
#SCIKIT
linreg = LinearRegression()
linreg.fit(features, response)

# print the coefficients
print linreg.intercept_
print linreg.coef_

list(features.columns)

#stats model
lm = smf.ols(formula='DoctorsVisitsNBR ~ AsthmaHistFLG + \
DiabetesFLG + HighBPFLG + BloodPressureMedFLG + HeartDiseaseFLG + SodaNBR + \
FastFoodNBR + TimeWalkNBR + \
VisionHearingProbFLG + DifConcentratingFLG + DifDressingFLG + \
DifGoingOutsideFLG + LimitPhysActivityFLG + ArmyFLG + \
PatientHospitalizedFLG + BornUSFLG + EducationCD + \
NervousFLG + IncomeNBR + CurrentAsthmaFLG + BingeCD + BMINBR + \
DisabledFLG + ERVisitsFLG + HeightNBR + HouseHoldSizeNBR + InsuranceFLG + \
NumCigCD + PovertyLevelCD + AgeRangeNBR + SexFLG + OwnFLG + RuralFLG + \
MedicalCareAccessFLG + WeightNBR + RaceCDAA + RaceCDASIAN + RaceCDHISPANIC +\
RaceCDOTHER + RaceCDWHITE + SmokingCDCURRENT + SmokingCDFORMER + \
SmokingCDNEVER + WorkingStatusCDFAMILYorVACATION + WorkingStatusCDFULLTIME + \
WorkingStatusCDPARTTIME + WorkingStatusCDRETIRED + WorkingStatusCDUNEMPLOYED + \
MaritalCDDIVORCEDorWIDOWED + MaritalCDMARRIED + MaritalCDNEVERMARRIED', \
data=data).fit()

# print the p-values for the model coefficients
print lm.summary()
print 'p values of each feature:'
print lm.pvalues
print 'r**2 of model with all features:'
print lm.rsquared

### REPEAT LINEAR REGRESSION WITH SCALING ###
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

'''cross-validated mean squared error'''
linreg = LinearRegression()
scores = cross_validation.cross_val_score(linreg, features_scaled, response, scoring='mean_squared_error', cv=5)
print 'cross-validated root mean squared error - 4.9165'
print np.sqrt(-scores.mean())

###############################################################################
#### RIDGE AND LASSO REGRESSION ####
features, response = data.ix[:,1:], data.DoctorsVisitsNBR
X, y = data.ix[:,1:], data.DoctorsVisitsNBR
X_train, X_test, y_train, y_test = train_test_split(features, response, random_state=1)

# ridge regression (alpha must be positive, larger means more regularization)
from sklearn.linear_model import Ridge
rreg = Ridge(alpha=0.1, normalize=True)
rreg.fit(X_train, y_train)
rreg.coef_
preds = rreg.predict(X_test)
np.sqrt(metrics.mean_squared_error(y_test, preds))
'''alpha of 0.1 results in rmse of 4.9055'''

# use RidgeCV to select best alpha
from sklearn.linear_model import RidgeCV
alpha_range = 10.**np.arange(-2, 3)
rregcv = RidgeCV(normalize=True, scoring='mean_squared_error', alphas=alpha_range,\
store_cv_values = True)
rregcv.fit(X_train, y_train)
rregcv.cv_values_
rregcv.alpha_
preds = rregcv.predict(X_test)
np.sqrt(metrics.mean_squared_error(y_test, preds))
'''alpha of 0.1 results in rmse of 4.9047'''

# lasso (alpha must be positive, larger means more regularization)
from sklearn.linear_model import Lasso
las = Lasso(alpha=0.01, normalize=True)
las.fit(X_train, y_train)
las.coef_
preds = las.predict(X_test)
np.sqrt(metrics.mean_squared_error(y_test, preds))
'''alpha of 0.1 results in rmse of 5.6396'''

# use LassoCV to select best alpha (tries 100 alphas by default)
from sklearn.linear_model import LassoCV
lascv = LassoCV(normalize=True)
lascv.fit(X_train, y_train)
lascv.alpha_
lascv.coef_
preds = lascv.predict(X_test)
np.sqrt(metrics.mean_squared_error(y_test, preds))
'''alpha of 0.1 results in rmse of 4.90478'''

###############################################################################
#### Decision Trees ####
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import train_test_split

features, response = data.ix[:,1:], data.DoctorsVisitsNBR
X, y = data.ix[:,1:], data.DoctorsVisitsNBR


X_train, X_test, y_train, y_test = train_test_split(features, response, random_state=1)


'''note: lowest cross validated rmse is 5.365'''
from sklearn.cross_validation import cross_val_score
treereg = DecisionTreeRegressor(max_depth=10, random_state=1)
scores = cross_val_score(treereg, X_train, y_train, cv=3, scoring='mean_squared_error')
np.mean(np.sqrt(-scores))

'''identifing most important features - appear to be PatientHospitalizedFLG, 
LimitPhysActivityFLG, HealthConditionNBR, ERVisitsFLG, BMINBR'''
treereg = DecisionTreeRegressor(max_depth=10, random_state=1)
treereg.fit(X_train, y_train)

feature_cols = features.columns
pd.DataFrame({'feature':feature_cols, 'importance':treereg.feature_importances_})

from sklearn.tree import export_graphviz
with open("DocVisits_tree.dot", 'wb') as f:
    f = export_graphviz(treereg, out_file=f, feature_names=feature_cols)
#below code didn't work - converted with GVEdit GUI
from os import system
system("dot -Tpng DocVisits_tree.dot -o DocVisits_tree.png")


from sklearn.grid_search import GridSearchCV
treereg = DecisionTreeRegressor(random_state=1)
depth_range = range(1, 40)
#max_feaure_range = range(1,3)
#param_grid = dict(max_depth=depth_range, max_features=max_feaure_range)
param_grid = dict(max_depth=depth_range)
grid = GridSearchCV(treereg, param_grid, cv=5, scoring='mean_squared_error')
grid.fit(features, response)


# Check out the scores of the grid search
grid_mean_scores = [result[1] for result in grid.grid_scores_]

'''best depth is 6'''
# Plot the results of the grid search
plt.figure()
plt.plot(depth_range, grid_mean_scores)
plt.hold(True)
plt.grid(True)
plt.plot(grid.best_params_['max_depth'], grid.best_score_, 'ro', markersize=12, markeredgewidth=1.5,
         markerfacecolor='None', markeredgecolor='r')

# Get the best estimator
best = grid.best_estimator_

'''ideal feaures results in 4.996 rmse'''
np.sqrt(-cross_val_score(best, features, response, cv=10, scoring='mean_squared_error').mean())

### sensitize additional parameters ###
from sklearn.grid_search import GridSearchCV
treereg = DecisionTreeRegressor(random_state=1)
depth_range = range(1, 10)
max_feaure_range = range(1,40)
param_grid = dict(max_depth=depth_range, max_features=max_feaure_range)
#param_grid = dict(max_depth=depth_range)
grid = GridSearchCV(treereg, param_grid, cv=5, scoring='mean_squared_error')
grid.fit(features, response)

# Get the best estimator
best = grid.best_estimator_

'''ideal feaures results in [__] rmse'''
np.sqrt(-cross_val_score(best, features, response, cv=10, scoring='mean_squared_error').mean())

#tree with the best parameters
treereg = best
treereg.fit(X_train, y_train)

feature_cols = features.columns
pd.DataFrame({'feature':feature_cols, 'importance':treereg.feature_importances_})

#tree with the best parameters
treereg = DecisionTreeRegressor(max_depth=5, random_state=1)
treereg.fit(X_train, y_train)

feature_cols = features.columns
pd.DataFrame({'feature':feature_cols, 'importance':treereg.feature_importances_})

from sklearn.tree import export_graphviz
with open("DocVisits_tree2.dot", 'wb') as f:
    f = export_graphviz(treereg, out_file=f, feature_names=feature_cols)

from os import system
system("dot -Tpng DocVisits_tree2.dot -o DocVisits_tree.png")


###############################################################################
#### Random forests ####
from sklearn.ensemble import RandomForestRegressor
X_train, X_test, y_train, y_test = train_test_split(features, response, random_state=1)

rf = RandomForestRegressor(n_estimators=100, max_features='auto', oob_score=True, random_state=1, criterion = 'mse')
rf.fit(X_train, y_train)

pd.DataFrame({'feature':feature_cols, 'importance':rf.feature_importances_})

'''the below oob says the oob score is 0.215 and the rmse is 4.967'''
float(rf.oob_score_)
y_pred = rf.predict(X_test)
print np.sqrt(metrics.mean_squared_error(y_test, y_pred))

### grid search - TAKES A COUPLE HOURS TO RUN ###
rf = RandomForestRegressor(max_features='auto', oob_score=True, random_state=1, criterion = 'mse')
n_estimators = range(50, 600, 50)
param_grid = dict(n_estimators=n_estimators)
#param_grid = dict(max_depth=depth_range)
grid = GridSearchCV(rf, param_grid, cv=3, scoring='mean_squared_error')
grid.fit(X_train, y_train)

grid_mean_scores = [result[1] for result in grid.grid_scores_]

'''250 estimators is best - 4.967 rmse.  RMSE keeps decreasing, so need to go higher to see if there
is a better model'''
# Plot the results of the grid search
plt.figure()
plt.plot(n_estimators, grid_mean_scores)
plt.hold(True)
plt.grid(True)
plt.plot(grid.best_params_['n_estimators'], grid.best_score_, 'ro', markersize=12, markeredgewidth=1.5,
         markerfacecolor='None', markeredgecolor='r')

# Get the best estimator
best = grid.best_estimator_
print np.sqrt(-max(grid_mean_scores))

'''run best model (250 n_estimators) on test data'''
rf = RandomForestRegressor(n_estimators=250, max_features='auto', oob_score=True, random_state=1, criterion = 'mse')
rf.fit(X_train, y_train)

pd.DataFrame({'feature':feature_cols, 'importance':rf.feature_importances_})

'''the below oob says the best rmse is 4.9508'''
float(rf.oob_score_)
y_pred = rf.predict(X_test)
print np.sqrt(metrics.mean_squared_error(y_test, y_pred))

### Additional Random Forest Parameter Optimization ###
from sklearn.grid_search import RandomizedSearchCV
from time import time
from scipy.stats import randint as sp_randint

# specify parameters and distributions to sample from
param_dist = {"n_estimators": range(200, 800, 50),
              "max_depth": [5, None],
              "max_features": sp_randint(3, 52),
              "min_samples_split": sp_randint(2, 100),
              "min_samples_leaf": sp_randint(1, 50),
              "bootstrap": [True, False]}

# run randomized search
rf = RandomForestRegressor(oob_score=False, random_state=1, criterion = 'mse', \
n_jobs = -1)

n_iter_search = 100
random_search = RandomizedSearchCV(rf, param_distributions=param_dist,\
n_iter=n_iter_search)

start = time()
random_search.fit(X, y)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))

random_search.grid_scores_
random_search.best_score_
random_search.best_estimator_
random_search.best_params_

'''best search the first time around resulted in the following parameters and 
took 140 minutes to run:

RandomForestRegressor(bootstrap=False, criterion='mse', max_depth=None,
           max_features=11, max_leaf_nodes=None, min_samples_leaf=5,
           min_samples_split=34, min_weight_fraction_leaf=0.0,
           n_estimators=500, n_jobs=-1, oob_score=False, random_state=1,
           verbose=0, warm_start=False)

'''           
'''run best model (parameters listed above) on test data'''
rf = random_search.best_estimator_
rf.fit(X_train, y_train)

feature_importanceRFR1 = pd.DataFrame({'feature':feature_cols, 'importance':rf.feature_importances_})

'''the first time the best rmse is 4.8400'''
y_pred = rf.predict(X_test)
print np.sqrt(metrics.mean_squared_error(y_test, y_pred))


###############################################################################
#### Principle Component Analysis ####
from sklearn import decomposition

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = decomposition.PCA(n_components=X_scaled.shape[1])
X_r = pca.fit_transform(X_scaled)

# Percentage of variance explained for each components
print('explained variance ratio: %s'
      % str(pca.explained_variance_ratio_))

plt.cla()
plt.plot(pca.explained_variance_ratio_)
plt.title('Variance explained by each principal component')
plt.ylabel(' % Variance Explained')
plt.xlabel('Principal component')

### REPEAT LINEAR REGRESSION WITH SCALING AND PCA ###
'''7 components based on elbow rule from PCA chart above '''
pca = decomposition.PCA(n_components=7)
X_r = pca.fit_transform(X_scaled)

'''cross-validated mean squared error'''
linreg = LinearRegression()
scores = cross_validation.cross_val_score(linreg, X_r, response, scoring='mean_squared_error', cv=5)
print 'cross-validated root mean squared error - 5.08'
print np.sqrt(-scores.mean())

'''testing multiple PCA component numbers - best RMSE is ''' 

n_components = range(X_scaled.shape[1]-1)
rmseList = []
for i in n_components:
    pca = decomposition.PCA(n_components=i+1)
    X_r = pca.fit_transform(X_scaled)
    '''cross-validated mean squared error'''
    linreg = LinearRegression()
    scores = cross_validation.cross_val_score(linreg, X_r, response, scoring='mean_squared_error', cv=5)
    print 'cross-validated root mean squared error for PCA with %s components' % (i+1)
    print np.sqrt(-scores.mean())
    rmseList.append(np.sqrt(-scores.mean()))
    print

'''plot of RMSE for linear regression based on PCA of various N's'''
'''Best rmse is 4.9164'''
plt.figure()
plt.plot(n_components, rmseList)
plt.grid(True)




###############################################################################


###############################################################################

###############################################################################

##### BELOW CODE IS FOR PREDICTING HEART DISEASE - RESULTS WERE NOT GOOD ######

###############################################################################

###############################################################################

###############################################################################
### Messing around with Clustering to explore major clusters ###
'''Note: all of the code below is really more for practice'''
from sklearn.cluster import KMeans
from sklearn import metrics

'''MemoryError Issues'''
#### Exploring varios number of clusters with K Means - TAKES A WHILE TO RUN #####
plt.figure(figsize=(7, 8))  
plt.subplot(211)
plt.title('Using the elbow method to inform k choice')
plt.xlim([1,5])
plt.grid(True)
plt.ylabel('Silhouette Coefficient')
for i in range(1):
    k_rng = range(1,5)
    est = [KMeans(n_clusters = k).fit(data) for k in k_rng]
    silhouette_score = [metrics.silhouette_score(data, e.labels_, metric='euclidean') for e in est[1:]]
    plt.plot(k_rng[1:], silhouette_score, 'b*-', alpha=0.5)
    #plt.plot(4,silhouette_score[2], 'o', markersize=12, markeredgewidth=1.5,
    #markerfacecolor='None', markeredgecolor='r', alpha=0.1)

#ways to see Sum of Squares
plt.figure(figsize=(7, 8))
plt.subplot(211)
plt.title('Sum of Squares')
plt.xlim([1,15])
plt.ylabel('Sum of Squares')
plt.grid(True)
for i in range(2):
    k_rng = range(1,15)
    est = [KMeans(n_clusters = k).fit(data) for k in k_rng]
    within_sum_squares = [e.inertia_ for e in est]
    plt.plot(k_rng, within_sum_squares, 'b*-', alpha=0.5)
    #plt.plot(4,within_sum_squares[3], 'ro', markersize=12, markeredgewidth=1.5,
    #     markerfacecolor='None', markeredgecolor='r', alpha=0.1)

###############################################################################
### KNN ### 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
X, y = data.ix[:,1:], data['HeartDiseaseFLG']
data['HeartDiseaseFLG'].value_counts(normalize=True)
data['HeartDiseaseFLG'].value_counts(normalize=True).plot(kind='bar')

# why is this improper cross-validation on the scaled data?
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

'''running on train / test data to determine optimal k value'''
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
#knn = KNeighborsClassifier(n_neighbors=11)
#knn.fit(X_train, y_train)
#knn.score(X_test, y_test)

from sklearn.grid_search import GridSearchCV
knn = KNeighborsClassifier()
k_range = range(1, 30, 2)
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid.fit(X_scaled, y)

grid.grid_scores_
grid_mean_scores = [result[1] for result in grid.grid_scores_]

# plot the results
plt.figure()
plt.plot(k_range, grid_mean_scores)

print 'best cross validated score'
grid.best_score_     # shows us the best score
grid.best_params_    # shows us the optimal parameters
grid.best_estimator_ # this is the actual model

###############################################################################
### LOGISTIC REGRESSION  ###

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

'''running on train and test data'''
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
logreg = LogisticRegression(C=1e9)
logreg.fit(X_train, y_train)

print 'intercept and coefficient'
print logreg.intercept_ 
print logreg.coef_

y_preds = logreg.predict(X_test)

testResults = pd.DataFrame(zip(y_preds, y_test), columns = ['Prediction','Actual'])
testResults['Correct'] = testResults.Prediction == testResults.Actual 

print 'accuracy:'
print metrics.accuracy_score(testResults.Actual, testResults.Prediction)
print 'classification report:'
print metrics.classification_report(testResults.Actual, testResults.Prediction)

'''cross-validated accuracy'''
scores = cross_validation.cross_val_score(logreg, X, y, scoring='accuracy', cv=5)
'''average accuracy'''
print 'cross-validated accuracy'
print scores.mean()

###############################################################################
#### Decision Trees ####
from sklearn import tree
from sklearn.cross_validation import train_test_split

features = data.ix[:,1:]
features.head()
response = data['HeartDiseaseFLG']
response.head()

X_train, X_test, y_train, y_test = train_test_split(features, response, random_state=1)


from sklearn.tree import DecisionTreeClassifier
ctree = tree.DecisionTreeClassifier(random_state=1)
ctree.fit(X_train, y_train)

'''note: lowest cross validated rmse is [__]'''
from sklearn.cross_validation import cross_val_score
ctree = tree.DecisionTreeClassifier(random_state=1)
cross_val_score(ctree, X_train, y_train, cv=10, scoring='roc_auc').mean()

#gathering importance
ctree = tree.DecisionTreeClassifier(max_depth=5, random_state=1)
ctree.fit(X_train, y_train)

feature_cols = features.columns
pd.DataFrame({'feature':feature_cols, 'importance':ctree.feature_importances_})

from sklearn.tree import export_graphviz
with open("HeartDisease_tree.dot", 'wb') as f:
    f = export_graphviz(ctree, out_file=f, feature_names=feature_cols)
#below code didn't work - converted with GVEdit GUI
from os import system
system("dot -Tpng HeartDisease.dot -o HeartDisease.png")

from sklearn.grid_search import GridSearchCV
ctree = tree.DecisionTreeClassifier(random_state=1)
depth_range = range(1, 20)
#max_feaure_range = range(1,3)
#param_grid = dict(max_depth=depth_range, max_features=max_feaure_range)
param_grid = dict(max_depth=depth_range)
grid = GridSearchCV(ctree, param_grid, cv=5, scoring='roc_auc')
grid.fit(features, response)


# Check out the scores of the grid search
grid_mean_scores = [result[1] for result in grid.grid_scores_]


# Plot the results of the grid search
plt.figure()
plt.plot(depth_range, grid_mean_scores)
plt.hold(True)
plt.grid(True)
plt.plot(grid.best_params_['max_depth'], grid.best_score_, 'ro', markersize=12, markeredgewidth=1.5,
         markerfacecolor='None', markeredgecolor='r')

# Get the best estimator
best = grid.best_estimator_

'''ideal feaures results in 21.904 rmse'''
cross_val_score(best, features, response, cv=10, scoring='roc_auc').mean()

### sensitize additional parameters ###
from sklearn.grid_search import GridSearchCV
ctree = tree.DecisionTreeClassifier(random_state=1)
depth_range = range(1, 20)
max_feaure_range = range(1,20)
param_grid = dict(max_depth=depth_range, max_features=max_feaure_range)
#param_grid = dict(max_depth=depth_range)
grid = GridSearchCV(ctree, param_grid, cv=5, scoring='roc_auc')
grid.fit(features, response)

# Get the best estimator
best = grid.best_estimator_

'''ideal feaures results in [__]'''
cross_val_score(best, features, response, cv=10, scoring='roc_auc')

#tree with the best parameters
ctree = best
ctree.fit(X_train, y_train)

feature_cols = features.columns
pd.DataFrame({'feature':feature_cols, 'importance':ctree.feature_importances_})

from sklearn.tree import export_graphviz
with open("HeartDisease_tree2.dot", 'wb') as f:
    f = export_graphviz(ctree, out_file=f, feature_names=feature_cols)

###############################################################################
#### Random forests ####
from sklearn.ensemble import RandomForestClassifier
rfclf = RandomForestClassifier(n_estimators=100, max_features='auto', oob_score=True, random_state=1)
rfclf.fit(features, response)

pd.DataFrame({'feature':feature_cols, 'importance':rfclf.feature_importances_})

float(rfclf.oob_score_ )

### grid search ###
rfclf = RandomForestClassifier(max_features='auto', oob_score=True, random_state=1)
n_estimators = range(50, 300, 50)
param_grid = dict(n_estimators=n_estimators)
#param_grid = dict(max_depth=depth_range)
grid = GridSearchCV(rfclf, param_grid, cv=5, scoring='accuracy')
grid.fit(features, response)

grid_mean_scores = [result[1] for result in grid.grid_scores_]

# Plot the results of the grid search
plt.figure()
plt.plot(n_estimators, grid_mean_scores)
plt.hold(True)
plt.grid(True)
plt.plot(grid.best_params_['max_depth'], grid.best_score_, 'ro', markersize=12, markeredgewidth=1.5,
         markerfacecolor='None', markeredgecolor='r')

# Get the best estimator
best = grid.best_estimator_

'''ideal feaures results in [__]'''
cross_val_score(best, features, response, cv=10, scoring='accuracy')


