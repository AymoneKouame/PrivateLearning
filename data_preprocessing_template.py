# -*- coding: utf-8 -*-
"""
Created on Sun May 27 23:30:46 2018

@author: Aymone
"""

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("C:/Users/Aymone/Desktop/SELF-LEARNING_Programming Languages/UDEMY/Machine Learning A-Z Template Folder/Part 1 - Data Preprocessing/data.csv")

# 1. CREATE/SEPARATE DATASET INTO DEPENDENT AND INDEPENDENT VARIABLES (this step not needed in R)
# create a matrix of independent variables
# and a matrix for the dependent variable or feature
# we take all lines and all columns except the first one which the list of indexes
# we also take all values

X = dataset.iloc[:, :-1].values
# to fix spyder error message: "object arrays are currently not supported
# do " X = pd.DataFrame(X); however when imputting missing values in 2,
# X needs to be array

# our dependent variable is the 4th column so index 3 (Python's index system starts with 0)
Y = dataset.iloc[:,3].values
# same as X; Y = pd.DataFrame(Y)

#---------------------------------------------------------------
# 2. PREP DATA
# NB:  2.1 and 2.2 are not encountered in every DS problem. So can be just informational for now
# 2.1 MISSING DATA TREATMENT; happens a lot in real life

# Choice 1: remove lines with missing data; but dangerous because could contain crucial information/predictors
# choice 2: take the mean of the columns and use it to fill in missing data. We will do choice 2.

from sklearn.preprocessing import Imputer
# sklearn contains amazing libraries for machine learning models

# Create an object of the class (tip: select Imputer and press CTRL+i)
imputer = Imputer("NaN", "mean", axis = 0)

#fit inputer object to our matrices, only to the columns with missing values
imputer = imputer.fit(X[:, 1:3]) # with python the upper bound is excluded but lower bound included, 
                        #so 1:3 means columns 1 and 2 only
                        
# replace missing values in columns 1 and 2 with the means (applies the method)
X[:, 1:3] = imputer.transform(X[:, 1:3])

#these representive numbers are called 'dummy variables'

#-------------------------------------------------------------------------
# 2.2 CATEGORICAL VARIABLE
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# since we cannot include characters in our mathematical prediction that we will do later
# we need to transform the country names and yes/no columns into codes
# Because they are categorical variables
# sklearn can automatically assign codes to each country
labelencoder_X= LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0]) #"0" is the index for country 

# We have now assigned codes to "Counrty" but we need one more thing
# We have France labelled with 0 and Spain labelled with 1
# Python will think that France is > Germany because 0 is > Germany
# That doesn't make sense. These are categorical and nominal (nno intrinsic order)
# (Categorical variables like Small, Med, Large have an intrinsic order 
# these are called categorical ordinal variables)

onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
#X = pd.DataFrame(X)
#the encoder created as many columns as they are categorical variaables. In our case 3
# For France, Germany and Spain

# For Y, since this is the dependent variable, the ML model
# will know that it is categorical and not ordered
labelencoder_Y= LabelEncoder()
Y = labelencoder_X.fit_transform(Y)

#-------------------------------------------------------------------------
# 2.3 SPLITTING DATASET BETWEEN TRAINING AND TESTING SETS
# This shoyuld be done in any ML model; 
# because this is what ML is: the algorithm needs to learn from the datasets patterns and correlations

from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# -------------------------------------------------------------------------
# 2.3 FEATURE SCALING
# Most of the time , the ML algorithm performs feature scaling for you but this is to 
# show how to do it if needed. 

# why it is important? The predictor variables for example are not on the same scale
# Age goes from 27 to 44 and salary goes from 48k to 83k. The ML will not know to see these two in proportion
# But the salary features will dominate because the ML will think they're greater
# So we need to put them on the same scale= feature scaling
# There are several ways to do that: Standardisation or Normalissation (there are mathematical equations behind)

""""import sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) # we need to fit and transform the train set
X_test = sc_X.transform(X_test) #we only need to transform the test set because it's already been fitted?""""
