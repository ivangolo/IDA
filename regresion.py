import pandas as pd
import numpy as np

###################################
#                a                #
###################################
# url = 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/prostate.data'
# df = pd.read_csv(url, sep='\t', header=0)
df = pd.read_csv('prostate.data', sep='\t', header=0)
df = df.drop('Unnamed: 0', axis=1)
istrain_str = df['train']
istrain = np.asarray([True if s == 'T' else False for s in istrain_str])
istest = np.logical_not(istrain)
df = df.drop('train', axis=1)

###################################
#                b                #
###################################
print "Dataset shape: ", df.shape
print df.info()
print df.describe()

###################################
#                c                #
###################################
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df_scaled['lpsa'] = df['lpsa']

###################################
#                d                #
###################################
import sklearn.linear_model as lm
X = df_scaled.ix[:, :-1]
N = X.shape[0]
X.insert(X.shape[1], 'intercept', np.ones(N))
y = df_scaled['lpsa']
print y.describe()
Xtrain = X[istrain]
ytrain = y[istrain]
Xtest = X[np.logical_not(istrain)]
ytest = y[np.logical_not(istrain)]
linreg = lm.LinearRegression(fit_intercept=False)
linreg.fit(Xtrain, ytrain)

###################################
#                e                #
###################################
ypred_train = linreg.predict(Xtrain)
from sklearn.metrics import mean_squared_error
sum_squares = np.power(ytrain - ypred_train, 2)
mse_train = mean_squared_error(ytrain, ypred_train)
n = ytrain.shape[0]
d = Xtrain.shape[1]
# var_est = mse_train
var_est = mse_train * n / (n - d)
diag_values = np.diag(np.linalg.pinv(np.dot(Xtrain.T, Xtrain)))
z_scores = np.divide(linreg.coef_, np.sqrt(np.multiply(var_est, diag_values)))
print "\n", "{:<15}{:<20}{}".format("Attribute", "Weight", "Z_score")
for attribute, weight, z_score in zip(Xtrain.columns.values, linreg.coef_, z_scores):
    print "{:<15}{:<20}{}".format(attribute, weight, z_score)

###################################
#                f                #
###################################
yhat_test = linreg.predict(Xtest)
mse_test = np.mean(np.power(yhat_test - ytest, 2))
print "mse test: ", mse_test
from sklearn import cross_validation
Xm = Xtrain.as_matrix()
ym = ytrain.as_matrix()
print "{:<5}{:<20}".format("k", "mse")
for k in range(5, 11):
    k_fold = cross_validation.KFold(len(Xm), k)
    mse_cv = 0
    for i, (train, val) in enumerate(k_fold):
        linreg = lm.LinearRegression(fit_intercept=False)
        linreg.fit(Xm[train], ym[train])
        yhat_val = linreg.predict(Xm[val])
        mse_fold = np.mean(np.power(yhat_val - ym[val], 2))
        mse_cv += mse_fold
    mse_cv = mse_cv / k
    print "{:<5}{:<20}".format(k, mse_cv)

###################################
#                g                #
###################################
import statsmodels.api as sm
from matplotlib import pyplot as plt
fig = sm.qqplot(ypred_train - ytrain, fit=True, line='45')
plt.show()
