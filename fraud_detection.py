# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 07:22:54 2020

@author: raksh
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split, cross_val_score

data = pd.read_csv(r"D:\Github\Fraudulent_Transactions\creditcard.csv")

X = data[data.columns[1:-2]]
y = data[data.columns[-1]].ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=14, stratify=y)

weights = compute_class_weight(class_weight='balanced',classes=np.unique(y),y=y)
weights = {0:round(weights[0],2), 1:round(weights[1],2)}

# Random Forest Classifier

rfClassifier = RandomForestClassifier(class_weight = weights)


n_estimators = [int(x) for x in np.linspace(2, 50, num=48)]
max_features = ['sqrt','log2']
max_depth = [int(x) for x in np.linspace(5, 50, num = 10)]
min_samples_split = [int(x) for x in np.linspace(2,10, num=5)]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features':max_features,
               'max_depth':max_depth,
               'min_samples_split':min_samples_split,
               'bootstrap':bootstrap
               }

del n_estimators,max_features,max_depth,min_samples_split,bootstrap

rfRandom = RandomizedSearchCV(estimator=rfClassifier,
                              param_distributions=random_grid,
                              n_iter=25,
                              cv=5,
                              verbose=10,
                              random_state=7,
                              n_jobs=-1,
                              scoring='f1'
                              )

rfRandom.fit(X_train, y_train)

best_params = rfRandom.best_params_

rfClassifierOpt = RandomForestClassifier(n_estimators=39, min_samples_split=8, max_features='log2', max_depth=50, bootstrap=False, class_weight = weights)

cross_val_scores = cross_val_score(rfClassifierOpt,X_test,y_test,scoring='f1')

print('Average F1-Score for Random Forest with RandomSearchCV is: {:0.2f}'.format(np.mean(cross_val_scores)*100))

print("Average F1-Score for Random Forest without RandomSearchCV is: {:0.2f} ".format(np.mean(cross_val_score(RandomForestClassifier(), X_test, y_test, scoring='f1')*100)))


# Logistic Regression Classifier

standard_scaler = StandardScaler()

X_train_scaled = pd.DataFrame(standard_scaler.fit_transform(X_train), columns = X_train.columns)
X_test_scaled = pd.DataFrame(standard_scaler.transform(X_test), columns = X_test.columns)

logRegClassifier = LogisticRegression()

param_grid = {'fit_intercept': [True, False]}

logRegGridSearch = GridSearchCV(estimator = logRegClassifier,
                            param_grid = param_grid,
                            scoring = 'f1',
                            n_jobs = -1,
                            cv = 5
                            )

logRegGridSearch.fit(X_train_scaled, y_train)

logRegClassifier = LogisticRegression(fit_intercept = False)

cross_val_scores = cross_val_score(logRegClassifier, X_test_scaled, y_test, scoring='f1')

print('Average F1-Score for Logistic Regression with GridSearchCV is: {:0.2f}'.format(np.mean(cross_val_scores)*100))

print("Average F1-Score for Random Forest without GridSearchCV is: {:0.2f} ".format(np.mean(cross_val_score(LogisticRegression(), X_test_scaled, y_test, scoring='f1')*100)))