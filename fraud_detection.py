# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 07:22:54 2020

@author: raksh
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score

data = pd.read_csv(r"creditcard.csv")

X = data[data.columns[1:-2]]
y = data[data.columns[-1]].ravel()
X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify = y)

weights = compute_class_weight(class_weight='balanced',classes=np.unique(y),y=y)


rfClassifier = RandomForestClassifier(class_weight={0:weights[0], 1:weights[1]})


n_estimators = [10, 50, 100, 200, 500]
max_features = ['auto']
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
min_samples_split = [2, 4, 8]
min_samples_leaf = []
bootstrap = [False]

random_grid = {'n_estimators': n_estimators,
#               'max_features':max_features,
               'max_depth':max_depth,
               'min_samples_split':min_samples_split
#               'min_samples_leaf':min_samples_leaf,
#               'bootstrap':bootstrap
               }

#del n_estimators,max_features,max_depth,min_samples_split,min_samples_leaf,bootstrap

rfGridSearch = GridSearchCV(estimator=rfClassifier,
                              param_grid=random_grid,
                              scoring = 'f1',
                              cv=3,
                              verbose=20,
                              n_jobs=16,
                              return_train_score=True
                              )
gridSearch = rfGridSearch.fit(X_train, y_train)

results = pd.DataFrame(gridSearch.cv_results_)
bestParams = gridSearch.best_params_

#rfClassifierOpt = RandomForestClassifier(n_estimators=42, min_samples_split=2, min_samples_leaf=9, max_features='sqrt',max_depth=50, bootstrap=True)
#
#cross_val_scores = cross_val_score(rfClassifierOpt,X_train,y_train,scoring='f1')
