# Kaggle_Titanic
import pandas as pd
import numpy as np
# Import 'tree' from scikit-learn library
from sklearn import tree

import csv as csv
import pylab as plt
from sklearn.cross_validation import cross_val_score

training_set =pd.read_csv('train.csv')
test_set=pd.read_csv("test.csv")


print(training_set.head())
print(test_set.head())

training_set.describe()

#****
#Observations: some ages are missing


#############################################
#     PART 1: Looking for survival rates amongst different features
#
##############################################
    
print(training_set["Survived"].value_counts())  #number of survivors
print(training_set["Survived"].value_counts(normalize = True))  # Propotion
      
# Survival rates among males and females
print(training_set["Survived"][training_set["Sex"]=="male"].value_counts(normalize=True))
print(training_set["Survived"][training_set["Sex"]=="female"].value_counts(normalize=True))     

#Proportion of kids that survived. Lets assume a kid is someone less than 15y old
#let's creat a variable Kid

training_set["Kid"]=float('NaN')
training_set["Kid"][training_set["Age"]<15]=1
training_set["Kid"][training_set["Age"]>15]=0

print(training_set["Kid"])
print(training_set["Survived"][training_set["Age"]<15].value_counts(normalize=True))
print(training_set["Survived"][training_set["Age"]>=15].value_counts(normalize=True))
#Survival rates among kids is higher than among adults

print(training_set["Survived"][training_set["Pclass"]==1].value_counts(normalize=True))
print(training_set["Survived"][training_set["Pclass"]==2].value_counts(normalize=True))
print(training_set["Survived"][training_set["Pclass"]==3].value_counts(normalize=True))

def plot_histogram(data, variable, bins=20):
    survived = data[data.Survived == 1]
    dead = data[data.Survived == 0]
    
    x1 = dead[variable].dropna()
    x2 = survived[variable].dropna()
    plt.hist( [x1,x2], label=['Dead','Survived'], color=['red','blue'], bins=bins)
    plt.legend(loc='upper left')
    plt.show()

plot_histogram(data=training_set, variable='Fare')

#******************************************
#
#  Obersvations: 
# 1) survival rates higher amongst female
# 2) survival rates higher amongst kids
# 3) survival rates is higher amongst passenger in class 1 (62%)than among passenger in class3 (24%)
# 4) The probability of death is higher amongst passengers who paid low fare (fare <50)
#
#************************************

##################
#
#
#  DATA CLEANING
#
#
##############

def cleaning_data (table) :
    
    #put the average age whenever the age is missing
    avg_age= np.mean(table["Age"])
    table["Age"]= table["Age"].fillna(avg_age)
    print(avg_age)
    
    ## Turn male and female to numerical data 
    table["Sex"][table["Sex"]=="male"]=0
    table["Sex"][table["Sex"]=="female"]=1

cleaning_data(training_set)

#print(training_set[["Sex","Age"]]) how to print two variable at the same time
    

##########################
#
#RPEDICTION: DECISION TREE
#
##########################


#decision tree using "Pclass", "Sex", "Age", "Fare" as features as they were spotted above as "relevant"

my_target=training_set["Survived"]
my_features_tree=training_set[["Pclass", "Sex", "Age", "Fare"]]

my_tree = tree.DecisionTreeClassifier()
my_tree = my_tree.fit(my_features_tree, my_target)

###########################
#
# PREDICTION: RANDOM FOREST
#
##################3

# Import the `RandomForestClassifier`
from sklearn.ensemble import RandomForestClassifier

my_features_randfor= training_set[["Pclass", "Age", "Sex", "Fare"]].values

# Building and fitting my_forest
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(my_features_randfor, my_target)

#Comparing decisionTree and random forest
print(my_tree.feature_importances_) # .feature_importances shows you about the importance of the feature
print(my_forest.feature_importances_) #

print(my_tree.score(my_features_tree, my_target)) #
print(my_forest.score(my_features_randfor, my_target)) #


score_tree = cross_val_score(my_tree,my_features_tree, my_target, cv=4)
print (round(np.mean(score_logit)*100, 2) ) 
score_forest = cross_val_score(my_forest,my_features_randfor, my_target, cv=4)
print (round(np.mean(score_forest)*100, 2) ) 

###-> Random forest return a higher cros validation score than Decision tree

#-----------------------------------
#
# Make a prediction on test data using decisionTree
#
#----------------------

test_set.describe()
#->Observation: one fare is missing and several ages ar not inputed

#we clean the test data 1st before predicting
cleaning_data(test_set)

#assign the median fare where a fare is missing
test_set["Fare"]=test_set["Fare"].fillna(test_set["Fare"].median())
test_set.describe()

my_features_test=test_set[["Pclass", "Sex", "Age", "Fare"]]

# Make a prediction
my_prediction = my_forest.predict(my_features_test)
print(my_prediction)

PassengerId =np.array(test_set["PassengerId"]).astype(int)
final_prediction = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
print(final_prediction)

#create a csv file with the final_prediction
final_prediction.to_csv("final_prediction.csv", index_label = ["PassengerId"])


