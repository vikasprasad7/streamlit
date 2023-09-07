import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')

#Loading Dataset

churn = pd.read_csv('D:\Downloads\customer_churn_2.csv')
churn.head()

#Data Exploration

churn.shape

churn.isnull().sum()

churn.describe()

churn.info()

churn.hist(bins=20, figsize=(10,10))
plt.show()

plt.bar(churn['Churn'], churn['Location'])
plt.xlabel('Churn')
plt.ylabel('Location')
plt.show()

#No Missing Data Found

plt.figure(figsize=(12, 12))
sns.heatmap(churn.corr() > 0.7, annot=True, cbar=False)
plt.show()

churn.head()

#Preparing the data for machine learning

#Generating Relevant Features from the dataset

features = churn.drop(['Name','Gender','Location'], axis=1)
target = churn['Churn']

xtrain,xtest,ytrain,ytest = train_test_split(features, target, test_size=0.2, random_state=40)
xtrain.shape, xtest.shape

#Applying Normalization

norm = MinMaxScaler()
xtrain = norm.fit_transform(xtrain)
xtest = norm.transform(xtest)

#Applying machine learning algorithms and training and validating the selected model on training dataset

models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]

for i in range(3):
    models[i].fit(xtrain, ytrain)
    
    print(f'{models[i]} : ')
    print('Training Accuracy : ', metrics.roc_auc_score(ytrain, models[i].predict(xtrain)))
    print('Validation Accuracy : ', metrics.roc_auc_score(ytest, models[i].predict(xtest)))
    print()

#Evaluating model's performance

print(metrics.classification_report(ytest, models[1].predict(xtest)))
