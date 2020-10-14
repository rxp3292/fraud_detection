# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 07:22:54 2020

@author: raksh
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.model_selection import cross_val_score

data = pd.read_csv(r"D:\Github\Fraudulent_Transactions\creditcard.csv")

X = data[data.columns[1:-2]]
y = data[data.columns[-1]].ravel()
X_train, X_test = train_test_split(X, test_size=0.25)
y_train, y_test = train_test_split(y, test_size=0.25)

weights = compute_class_weight(class_weight='balanced',classes=np.unique(y),y=y)


rfClassifier = RandomForestClassifier(class_weight={0:weights[0], 1:weights[1]})


n_estimators = [int(x) for x in np.linspace(2, 100, num=50)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 50, num = 10)]
min_samples_split = [int(x) for x in np.linspace(2,10, num=1)]
min_samples_leaf = [int(x) for x in np.linspace(1,10, num=10)]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features':max_features,
               'max_depth':max_depth,
               'min_samples_split':min_samples_split,
               'min_samples_leaf':min_samples_leaf,
               'bootstrap':bootstrap
               }

del n_estimators,max_features,max_depth,min_samples_split,min_samples_leaf,bootstrap

rfRandom = RandomizedSearchCV(estimator=rfClassifier,
                              param_distributions=random_grid,
                              n_iter=50,
                              cv=5,
                              verbose=2,
                              random_state=7,
                              n_jobs=4,
                              )

best_params = rfRandom.cv_results_

rfClassifierOpt = RandomForestClassifier(n_estimators=42, min_samples_split=2, min_samples_leaf=9, max_features='sqrt',max_depth=50, bootstrap=True)

cross_val_scores = cross_val_score(rfClassifierOpt,X_train,y_train,scoring='f1')
