# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.io import mmread
import sklearn.linear_model as lm


def list_duplicates_of(seq,item):
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item,start_at+1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs




data_dir = "./ngrams-deprels-origin.runtime.budget.numscreen.ratings.seasons/"

print "importando X..."
X = csr_matrix( mmread(data_dir+'train.x.mm'))
print "importando y..."
y = np.loadtxt(data_dir+'train.y.dat')

'''
model = lm.LinearRegression(fit_intercept = True)
model.fit (X,y)
print "R2=%f"%model.score(X, y)
'''

models = [ ]
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for i in range(0,10):
    print i
    models.append( lm.Lasso(alpha=alphas[i], max_iter=1000, tol=0.001) )
    models[i].fit (X,y)
    print "R2=%f"%models[i].score(X, y)



print "importando X_val..."
X_validation = csr_matrix( mmread(data_dir+'dev.x.mm'))
print "importando y_val..."
y_validation = np.loadtxt(data_dir+'dev.y.dat')


#testing

# The mean square error
MSEs = [ ]
for i in range(0,10):
    MSEs.append = np.mean((models[i].predict(X_validation) - y_validation) ** 2)
    print("Residual sum of squares Nº" + str(i) + "{0:.2f}".format(MSEs[i]) )

print list_duplicates_of(MSEs, min(MSEs))
selected_index = MSEs.index(min(MSEs))
print MSEs.index(min(MSEs))

print "importando X_test..."
X_test = csr_matrix( mmread(data_dir+'test.x.mm'))
print "importando y_test..."
y_test = np.loadtxt(data_dir+'test.y.dat')

print "R2=%f"%models[selected_index].score(X_test, y_test)
