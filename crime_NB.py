#!/usr/bin/env python

"""
Script for predicting categorical crime data from provided CSV datasets on Kaggle.
Multivariate Bernoulli model is used, all inputs must be numerical.
Result is CSV file with incident ID (from test data) and prediction matrix with 
values for each possible crime category. 

Author: Brent Key, 4/11/16
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss


#read training and test files, parse date/time
train = pd.read_csv('train.csv', parse_dates = ['Dates'])
test = pd.read_csv('test.csv', parse_dates = ['Dates'])

#change 'Category' text to normalized numerical value
num_categ = preprocessing.LabelEncoder()

#unique value assigned to each category(0->n-1)
category = num_categ.fit_transform(train.Category)  


#transform days, districts, and hours to unique 'binarized' vectors
#minimizes errors from ML algorithm
hour = train.Dates.dt.hour
hour = pd.get_dummies(hour) 
day = pd.get_dummies(train.DayOfWeek)
district = pd.get_dummies(train.PdDistrict)

#new matrix concatenated column-wise
train_data = pd.concat([day, district, hour], axis = 1)
train_data['Category'] = category
 
#format test data, similar to training data
hour = test.Dates.dt.hour
hour = pd.get_dummies(hour) 
day = pd.get_dummies(test.DayOfWeek)
district = pd.get_dummies(test.PdDistrict)

test_data = pd.concat([day, district, hour], axis = 1)

 
#Prediction inputs, all possible values for hour/day/district
params = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
  'BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION', 'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']
params2 = [x for x in range(0,24)]
params = params + params2


#Fit multi-variate Bernoulli model, predict crime category from test data
model = BernoulliNB()
model = model.fit(train_data[params], train_data['Category'])
predict = model.predict_proba(test_data[params])


#Split training data so results can be validated with prelim test
#Implement the following to test data, results analyzed with log loss
#training, validate = train_test_split(train_data, train_size=.60)
#model = BernoulliNB()
#model = model.fit(training[params], training['Category'])
#predict = np.array(model.predict_proba(validate[params])) #return prediction for each class in the model
#print(log_loss(validate['Category'], predict)) 


#Results to CSV
results = pd.DataFrame(predict, columns = num_categ.classes_)
results.to_csv('predict_results.csv', index = True, index_label = 'ID' )
