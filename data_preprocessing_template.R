# IMPORTING THE DATASET
dataset = read.csv("C:/Users/Aymone/Desktop/SELF-LEARNING_Programming Languages/UDEMY/Machine Learning A-Z Template Folder/Part 1 - Data Preprocessing/data.csv")

# PREP DATA

# ---------------------------------------------------------------------------------
# 2.1 and 2.2 are not encountered in every DS problem. So can be just informational for now
# 2.1 Missing value treatment


dataset$Age = ifelse(is.na(dataset$Age), yes = stats::ave(dataset$Age, 
                                              FUN = function(x) mean (x, na.rm = TRUE)),
                                        no = dataset$Age)

dataset$Salary = ifelse(is.na(dataset$Salary), yes = stats::ave(dataset$Salary, 
                                                          FUN = function(x) mean (x, na.rm = TRUE)),
                                               no = dataset$Salary)
# 2.2 Categorical and nominal variable treatment

dataset$Country = factor(dataset$Country, levels = c('France', "Spain", "Germany"),
                         labels = c(1, 2, 3))
# here they don't want the dataset to be labelled by the country name, but by 1,2and 3.
#f.y.i "levels" is optional and if labels not specified, the name of the country would be used

dataset$Purchased = factor(dataset$Purchased, levels = c('No', "Yes"),
                         labels = c(0, 1))

# ---------------------------------------------------------------------------------

# 2.3 SPLITTING DATASET BETWEEN TRAINING AND TESTING SETS (VERY IMPORTANT)
# This should be done in any ML model; 
# because this is what ML is: the algorithm needs to learn from the datasets patterns and correlations

library(caTools) #or you can check the box for the package in the packages list

set.seed(123)
split <- sample.split(dataset$Purchased, SplitRatio = 0.8)
trainingSet = subset(dataset, split == TRUE)
testSet = subset(dataset, split == FALSE)

# 2.3 FEATURE SCALING
# Most of the time , the ML algorithm performs feature scaling for you but this is to 
# show how to do it if needed

# why it is important? The predictor variables for example are not on the same scale
# Age goes from 27 to 44 and salary goes from 48k to 83k. The ML will not know to see these two in proportion
# But the salary features will dominate because the ML will think they're greater
# So we need to put them on the same scale= feature scaling
# There are several ways to do that: Standardisation or Normalissation (there are mathematical equations behind)

# trainingSet[,2:3] = scale(trainingSet[,2:3])
# testSet[,2:3]  = scale(testSet[,2:3] )







