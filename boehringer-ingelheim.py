# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 09:44:03 2014

"""

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation as cv
from sklearn import grid_search
import numpy as np

def main():
    # read in  data, parse into training set and target set (labels)
    dataset = np.genfromtxt(open('data/train.csv','r'), delimiter=',', dtype='f8')[1:]    
    target = np.array([x[0] for x in dataset])
    train = np.array([x[1:] for x in dataset])
    # test set
    test = np.genfromtxt(open('Data/test.csv','r'), delimiter=',', dtype='f8')[1:]

    # In this case we'll use a random forest, but this could be any classifier
    # n_estimators = the number of trees in the forest
    # max_features = The number of features to consider when looking for the best split
    # n_jobs = 4. For quad-core CPUs (if -1, is set to the number of cores )
    # bootstrap = True. Sampling with replacement (create datasets from which individual trees are grown)
    # Note: scikit-learn does NOT allow the user to specify the sampling rate
    clRF = RandomForestClassifier(criterion="gini", n_estimators=100, n_jobs=-1) 

    # Simple K-Fold cross validation. 5 folds.
    kfCV = cv.KFold(len(train), n_folds=5, indices=False)

    # Implement a grid search hyperparameter optimization    
    n_features_range = np.arange(5,51,5)
    # Dictionary with parameters names as keys and lists of parameter settings to try as values
    params = dict(max_features = n_features_range)
    
    # Random Forest classifier optimized with Grid Search cross validation
    clf = grid_search.GridSearchCV(clRF, param_grid = params, cv = kfCV)
    # fit the model
    clf.fit(train, target)
    print("The best classifier is: ", clf.best_estimator_)
    
    # Estimate score of the classifier
    scores = cv.cross_val_score(clf.best_estimator_, train, target, cv=5)
    print('Estimated score: %0.5f (+/- %0.5f)' % (scores.mean(), scores.std() / 2))

    # Predict with the best performing Random Forest classifier
    # we could predict the class, but this problem requires a probability as output
    predicted_class = clf.best_estimator_.predict(test)
    predicted_probs = clf.best_estimator_.predict_proba(test)

    predicted_probs = [[index + 1, x[1]] for index, x in enumerate(clf.best_estimator_.predict_proba(test))]

    np.savetxt('data/submission.csv', predicted_probs, delimiter=',', fmt='%d,%f', 
            header='MoleculeId,PredictedProbability', comments = '')

if __name__=="__main__":
    main()