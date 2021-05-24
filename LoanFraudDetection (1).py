#!/usr/bin/env python
# coding: utf-8

# ### Project Name - Bank Loan Default Case
# 
# The loan default dataset has 8 variables and 850 records, each record being loan
# default status for each customer. Each Applicant was rated as “Defaulted” or
# “Not-Defaulted”. New applicants for loan application can also be evaluated on
# these 8 predictor variables and classified as a default or non-default based on
# predictor variables.

# ####  Import Necessary Libaries 

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot


# ####  Read the data file which we want to use in project

# In[2]:


#setting work directory
os.chdir("C://Users//Monali//OneDrive//Desktop//Case Study-DS//Bank_loan_defaultCase")


# In[3]:


os.getcwd()


# In[4]:


Data = pd.read_csv('bank_loan.csv')


# ### Data Inspection

# In[5]:


Data.head()                  


# Here, we can see that there are 8 variables which are independent and termed as independent variables(predictors) and 1 dependent variable(default) which is output whether are the customer is defaulter(1.0) or non-defaulter(0.0)

# In[6]:


Data.shape


# In[7]:


Data.info()


# In[8]:


Data.describe()


# In[9]:


sns.pairplot(Data, hue = 'default')


# As there is overlapping in data at certain extent, Logistic Regression wont be the great choice.
# KNN might work better

#  ### Data Profiling

# In[10]:


get_ipython().system('pip install pandas-profiling')


# In[11]:


import pandas_profiling
from pandas_profiling import ProfileReport


# In[12]:


profile = ProfileReport(Data, title="Pandas Profiling Report")


# In[13]:


profile


# #### Data Preprocessing 

# In[14]:


# Let rename the name of the columns of the data
Data = Data.rename(columns = {'age':"Customer_Age", "ed":'Education','employ':'Employment_Status','address':'Address','income':'Income','debtinc':'Debt_Payment','creddebt':'Debt_to_Credit_Ratio','othdebt':'Other_debts'})


# In[15]:


Data.head()


# ### Missing Value Analysis
# 

# In[16]:


Data.isnull().sum()


# Above, we see that none of the independent variable conatins null values but only some dependent variable data is not available, thats we need to predict.
# 
# We will seperate on the basis of dependent variable that is values having output variable in one dataset and remaining 150 records in another and we used this as input in model to predict its value. 

# In[17]:


#Visualization of missing values with heatmap
sns.heatmap(Data.isnull(),yticklabels = False,cbar = True, cmap = "Blues",linecolor = "Black")


# In[18]:


#Finding out the distribution of missing value
sns.distplot(Data['default'])


# In[19]:


train_set = Data.iloc[0:700,:]


# In[20]:


test_set = Data.iloc[700:,:8]    # we will use this data as real time data to predict output(test_set)


# In[21]:


train_set.head()


# In[22]:


test_set.head()


# In[23]:


train_set['Education'].value_counts()


# In[24]:


train_set.isnull().sum()


# In[25]:


train_set['Employment_Status'].value_counts()


# In[26]:


train_set['default'].value_counts()


# #### Outliers check

# In[27]:


plt.figure(figsize=(20,10))
for j in list(train_set.columns.values):
    plt.scatter(y=train_set[j],x=[i for i in range(len(train_set[j]))],s=[20])
plt.legend()


# In[28]:


plt.figure(figsize = (20,15))
boxplot = train_set.boxplot(column = ['Customer_Age','Education','Address','Income','Employment_Status','Debt_Payment','Debt_to_Credit_Ratio','Other_debts'])


# Above, there are outliers values in some columns but we cannot remove them beacuse all of them contains important information to predict output

# In[29]:


plt.figure(figsize = (12,7))
boxplot = train_set.boxplot(column = ['Income'], by = 'default')
plt.xlabel('default')
plt.ylabel('values')
plt.show()


# #### Feature Selection analysis
# 
# Finding out the best feature which will contribute and have good relation with target variable.

# In[30]:


train_set.shape


# In[31]:


train_set.columns


# ### Finding Corelation betweeen variables
# 

# In[32]:


f, ax = plt.subplots(figsize = (20,10))
sns.heatmap(train_set.corr(),annot = True)


# So, no value is highly positively or negatively corelated to each other hence none variable to drop
# Hence, this is a correlation plot of all the variables. None of the variable has most postive closer to 1 or most negative closer to -1 so, every variable carries unique information and none of them has to remove

# #### Balancing unbalanced dataset

# In[33]:


train_set['default'].value_counts()


# Here, number of non-defaulters(0's) is more than the number of defaulter and there is a quite mismatch in the data. So, to generate proper results and avoid biasing of our model,we need to handle this by appling resampling technique like oversampling

# In[34]:


# visualize the target variable
g = sns.countplot(train_set['default'])
g.set_xticklabels(['Not Fraud','Fraud'])
plt.show()


# In[35]:


# class count
class_count_0, class_count_1 = train_set['default'].value_counts()


# In[36]:


class_count_0


# In[37]:


class_count_1


# In[38]:


# Separate class
class_0 = train_set[train_set['default'] == 0]
class_1 = train_set[train_set['default'] == 1]         

# print the shape of the class
print('class 0:', class_0.shape)
print('class 1:', class_1.shape)


# In[39]:


get_ipython().system('pip install imblearn')


# In[40]:


#SMOTE- Oversampling technique which is used to equal minority and majority class to aviod biasing
from imblearn.over_sampling import SMOTE


# In[41]:


smote = SMOTE()


# In[42]:


x = train_set.iloc[:,0:8]
y= train_set.iloc[:,-1]


# In[43]:


x.head()


# In[44]:


y.head()


# In[45]:


from collections import Counter


# In[46]:


# fit predictor and target variable
x_smote, y_smote = smote.fit_resample(x, y)

print('Original dataset shape', Counter(x_smote))
print('Resample dataset shape', Counter(y_smote))


# In[47]:


train_set = pd.concat([x_smote, y_smote], axis = 1)


# In[48]:


train_set


# In[49]:


g = sns.countplot(train_set['default'])
g.set_xticklabels(['Not Fraud','Fraud'])
plt.show()


# We have make all the classes equal and now it is more good to model out data and make prediction

# In[50]:


train_set.isnull().sum()


# In[51]:


train_set.shape


# #### Model Development
# 
# 1. Split dataset into train and test set in order to prediction w.r.t X_test
# 2. Import model
# 3. Fit the data
# 4. Predict w.r.t X_test
# 5. In Classification, check accuracy score, accuracy of model, Precison, recall, ROC curve
# 6. Plot graph

# train test split

# In[52]:


#Building a regression model
# Import library
from sklearn.model_selection import train_test_split


# In[53]:


# Split our training set into two parts that is train and validation set(test)
X = train_set.iloc[:,0:8]
Y = train_set.iloc[:,-1]


# In[54]:


X_train, X_test, y_train,y_test = train_test_split(X,Y,test_size=0.2, random_state=42)


# In[55]:


from sklearn.datasets import make_classification


# In[56]:


# Make a two class classification first
XA, yb = make_classification(n_samples=1000, n_classes=2, random_state=1)


# In[57]:


# split into train/test sets
trainX, testX, trainy, testy = train_test_split(XA, yb, test_size=0.5, random_state=2)


# ### Model 1: KNN Algorithm

# In[58]:


#KNN imputation and implementaion for model
from sklearn.neighbors import KNeighborsClassifier


# In[59]:


KNN_model = KNeighborsClassifier(n_neighbors = 9).fit(X_train, y_train)


# In[60]:


#predict test case
KNN_predictions = KNN_model.predict(X_test)


# In[61]:


#let build confusion matrix to check our model accuracy
CM = pd.crosstab(y_test, KNN_predictions)

TP = CM.iloc[0,0]
FP = CM.iloc[1,0]
TN = CM.iloc[1,1]
FN = CM.iloc[0,1] 

#accuracy
knn_acc = ((TP+TN)*100)/(TP + TN + FP+ FN)


# In[62]:


knn_acc


# In[63]:


#FNR
(FN*100)/(FN + TP)


# In[64]:


#Evaluation the model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[65]:


confusion_matrix(y_test, KNN_predictions)


# In[66]:


print(classification_report(y_test, KNN_predictions))


# In[67]:


#Recall
TP/(TP + FN)


# In[68]:


#Precision
TP/(TP+FP)


# In[69]:


import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
probs = KNN_model.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.legend(loc = 'lower right')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[70]:


y_pred_proba = KNN_model.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.title('Area Under Curve(AUC)')
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# In[71]:


# Make a two class classification first
XA, yb = make_classification(n_samples=1000, n_classes=2, random_state=1)


# In[72]:


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[73]:


# fit a model
model = KNeighborsClassifier()
model.fit(trainX, trainy)
# predict probabilities
yhat = model.predict_proba(testX)
# retrieve just the probabilities for the positive class
pos_probs = yhat[:, 1]
# calculate the no fraud line as the proportion of the positive class
no_fraud = len(y[y==1]) / len(y)
# plot the no fraud precision-recall curve
pyplot.plot([0, 1], [no_fraud, no_fraud], linestyle='--', label='No Fraud')
# calculate model precision-recall curve
precision, recall, _ = precision_recall_curve(testy, pos_probs)
# plot the model precision-recall curve
pyplot.plot(recall, precision, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()


# #### Model 2: Decision Tree

# In[74]:


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier


# In[75]:


#decision tree algo -- C5.0 - entropy and cart - gini
#the gini criterion is much faster because it is less computationally expensive.
#On the other hand, the obtained results using the entropy criterion are slightly better.

C50_model = tree.DecisionTreeClassifier(criterion = 'entropy').fit(X_train, y_train)


# In[76]:


#predict new test case
DecisionTree_Prediction = C50_model.predict(X_test)


# In[77]:


train_set.columns[0:8]


# In[78]:


#create dot file to visulize tree   #http://webgraphviz.com
dotfile = open('pt.dot', 'w')
df = tree.export_graphviz(C50_model, out_file=dotfile, feature_names = train_set.columns[0:8])


# In[79]:


#let build confusion matrix to check our model accuracy
CM = pd.crosstab(y_test, DecisionTree_Prediction)

TP = CM.iloc[0,0]
FP = CM.iloc[1,0]
TN = CM.iloc[1,1]
FN = CM.iloc[0,1] 

#accuracy
dt_acc = ((TP+TN)*100)/(TP + TN + FP+ FN)


# In[80]:


#FNR
(FN*100)/(FN + TP)


# In[81]:


confusion_matrix(y_test, DecisionTree_Prediction)


# In[82]:


print(classification_report(y_test, DecisionTree_Prediction))


# In[83]:


#Recall
TP/(TP + FN)


# In[84]:


#Precision
TP/(TP+FP)


# In[85]:


import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
probs = C50_model.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[86]:


y_pred_proba = C50_model.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.title('Area Under Curve(AUC)')
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# In[87]:


# fit a model
model = DecisionTreeClassifier()
model.fit(trainX, trainy)
# predict probabilities
yhat = model.predict_proba(testX)
# retrieve just the probabilities for the positive class
pos_probs = yhat[:, 1]
# calculate the no fraud line as the proportion of the positive class
no_fraud = len(y[y==1]) / len(y)
# plot the no fraud precision-recall curve
pyplot.plot([0, 1], [no_fraud, no_fraud], linestyle='--', label='No fraud')
# calculate model precision-recall curve
precision, recall, _ = precision_recall_curve(testy, pos_probs)
# plot the model precision-recall curve
pyplot.plot(recall, precision, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()


# #### Model 3: Random Forest 

# In[88]:


#import random forest library from sklean and dev model
from sklearn.ensemble import RandomForestClassifier


# In[89]:


RF_model = RandomForestClassifier()


# In[90]:


RF_model.fit(X_train, y_train)


# In[91]:


RF_prediction = RF_model.predict(X_test)


# In[92]:


RF_model.score(X_train, y_train)


# In[93]:


RF_model.score(X_test, y_test)


# In[94]:


metrics.r2_score(y_test, RF_prediction)


# In[95]:


#let build confusion matrix to check our model accuracy
CM = pd.crosstab(y_test, RF_prediction)

TP = CM.iloc[0,0]
FP = CM.iloc[1,0]
TN = CM.iloc[1,1]
FN = CM.iloc[0,1] 

#accuracy
((TP+TN)*100)/(TP + TN + FP+ FN)


# In[96]:


#FNR
(FN*100)/(FN + TP)


# In[97]:


confusion_matrix(y_test, RF_prediction)


# In[98]:


print(classification_report(y_test, RF_prediction))


# In[99]:


#Recall
TP/(TP + FN)


# In[100]:


#Precision
TP/(TP+FP)


# In[101]:


import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
probs = RF_model.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[102]:


y_pred_proba = RF_model.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.title('Area Under Curve(AUC)')
plt.legend(loc=4)
plt.show()


# #### Hyperparameter Tuning in Random Forest

#  - Choose following method for hyperparameter tuning
#      1. RandomizedSearchCV --> Fast
#      2. GridSearchCV
#  - Assign hyperparameters in form of dictionery
#  - Fit the model
#  - Check best paramters and best score

# In[103]:


from sklearn.model_selection import RandomizedSearchCV


# In[104]:


#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# In[105]:


# Create the random grid

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}


# In[106]:


# Random search of parameters, using 5 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = RF_model, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[107]:


rf_random.fit(X_train,y_train)


# In[108]:


rf_random.best_params_


# In[109]:


prediction = rf_random.predict(X_test)


# In[110]:


#let build confusion matrix to check our model accuracy
CM = pd.crosstab(y_test, prediction)

TP = CM.iloc[0,0]
FP = CM.iloc[1,0]
TN = CM.iloc[1,1]
FN = CM.iloc[0,1]  

#accuracy
rf_acc =((TP+TN)*100)/(TP + TN + FP+ FN)


# In[111]:


#FNR
(FN*100)/(FN + TP)


# In[112]:


confusion_matrix(y_test, prediction)


# In[113]:


#Recall
TP/(TP + FN)


# In[114]:


#Precision
TP/(TP+FP)


# In[115]:


print(classification_report(y_test, prediction))


# In[116]:


import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
probs = rf_random.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[117]:


y_pred_proba = rf_random.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.title('Area Under Curve(AUC)')
plt.legend(loc=4)
plt.show()


# In[118]:


# fit a model
model = RandomForestClassifier()
model.fit(trainX, trainy)
# predict probabilities
yhat = model.predict_proba(testX)
# retrieve just the probabilities for the positive class
pos_probs = yhat[:, 1]
# calculate the no fraud line as the proportion of the positive class
no_fraud = len(y[y==1]) / len(y)
# plot the no fraud precision-recall curve
pyplot.plot([0, 1], [no_fraud, no_fraud], linestyle='--', label='No fraud')
# calculate model precision-recall curve
precision, recall, _ = precision_recall_curve(testy, pos_probs)
# plot the model precision-recall curve
pyplot.plot(recall, precision, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()


# #### Model 4: XGBoost

# In[119]:


import xgboost


# In[120]:


from xgboost import XGBClassifier


# In[121]:


# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
XG_Pred = model.predict(X_test)


# In[122]:


#let build confusion matrix to check our model accuracy
CM = pd.crosstab(y_test, XG_Pred)

TP = CM.iloc[0,0]
FP = CM.iloc[1,0]
TN = CM.iloc[1,1]
FN = CM.iloc[0,1] 

#accuracy
xg_acc =((TP+TN)*100)/(TP + TN + FP+ FN)


# In[123]:


xg_acc


# In[124]:


#FNR
(FN*100)/(FN + TP)


# In[125]:


confusion_matrix(y_test, XG_Pred)


# In[126]:


#Recall
TP/(TP + FN)


# In[127]:


#Precision
TP/(TP+FP)


# In[128]:


print(classification_report(y_test, XG_Pred))


# In[129]:


import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
probs = model.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[130]:


y_pred_proba = model.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.title('Area Under Curve(AUC)')
plt.legend(loc=4)
plt.show()


# In[131]:


# fit a model
model = XGBClassifier()
model.fit(trainX, trainy)
# predict probabilities
yhat = model.predict_proba(testX)
# retrieve just the probabilities for the positive class
pos_probs = yhat[:, 1]
# calculate the no fraud line as the proportion of the positive class
no_fraud = len(y[y==1]) / len(y)
# plot the no fraud precision-recall curve
pyplot.plot([0, 1], [no_fraud, no_fraud], linestyle='--', label='No fraud')
# calculate model precision-recall curve
precision, recall, _ = precision_recall_curve(testy, pos_probs)
# plot the model precision-recall curve
pyplot.plot(recall, precision, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()


# #### We will freeze Random forest for our model because it is giving the best accuarcy alongwith FNR, Recall, Precision
# 
# From random forest, we are going to apply our test set which we seperated at starting and we will feed that as an input set and predict the output of it.

# In[132]:


# We will be using complete train set to predict the value for defaulter in test set


# In[133]:


test_prediction = rf_random.predict(test_set)


# In[134]:


test_prediction


# In[135]:


test_default = pd.DataFrame(test_prediction)


# In[136]:


unique, counts = np.unique(test_prediction, return_counts=True)
dict(zip(unique, counts))


# #### Save the model and reuse it again

# In[137]:


import pickle   ## Serialize file
# open a file where we want to store a file
file = open('loanfrauddetection.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)


# ### Here comes the completion of the project and we used many different models to predict accuracy, precision, recall and we comes out with random forest to be the best for this scenario
# 
# ### We can also try with different ways to handle things according to the requirements like we can remove balancing of the data used above and try to calculate values or we can simple remove outliers when we have millions of data or any others ways.
