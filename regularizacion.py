# coding=UTF-8
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from matplotlib import pyplot as plt
from sklearn import cross_validation

# Loading the dataset
# url = 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/prostate.data'
# df = pd.read_csv(url, sep='\t', header=0)
df = pd.read_csv('prostate.data', sep='\t', header=0)
df = df.drop('Unnamed: 0', axis=1)
istrain_str = df['train']
istrain = np.asarray([True if s == 'T' else False for s in istrain_str])
istest = np.logical_not(istrain)
df = df.drop('train', axis=1)

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df_scaled['lpsa'] = df['lpsa']
X = df_scaled.ix[:, :-1]
N = X.shape[0]
X.insert(X.shape[1], 'intercept', np.ones(N))
y = df_scaled['lpsa']

###################################
#                a                #
###################################
X = X.drop('intercept', axis=1)
Xtrain = X[istrain]
ytrain = y[istrain]
names_regressors = ["Lcavol", "Lweight", "Age", "Lbph", "Svi", "Lcp", "Gleason", "Pgg45"]
alphas_ = np.logspace(4, -1, base=10)
coefs = []
model = Ridge(fit_intercept=True, solver='svd')

for a in alphas_:
    model.set_params(alpha=a)
    model.fit(Xtrain, ytrain)
    coefs.append(model.coef_)

ax = plt.gca()
for y_arr, label in zip(np.squeeze(coefs).T, names_regressors):
    print alphas_.shape
    print y_arr.shape
    plt.plot(alphas_, y_arr, label=label)

plt.legend()
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Regularization Path RIDGE')
plt.axis('tight')
plt.legend(loc=2)
plt.show()

###################################
#                b                #
###################################
alphas_ = np.logspace(1, -2, base=10)
model = Lasso(fit_intercept=True)
coefs = []
for a in alphas_:
    model.set_params(alpha=a)
    model.fit(Xtrain, ytrain)
    coefs.append(model.coef_)

ax = plt.gca()
for y_arr, label in zip(np.squeeze(coefs).T, names_regressors):
    print alphas_.shape
    print y_arr.shape
    plt.plot(alphas_, y_arr, label=label)

plt.legend()
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Regularization Path Lasso')
plt.axis('tight')
plt.legend(loc=2)
plt.show()

###################################
#                c                #
###################################
Xtest = X[np.logical_not(istrain)]
ytest = y[np.logical_not(istrain)]
alphas_ = np.logspace(2, -2, base=10)
coefs = []
model = Ridge(fit_intercept=True)
mse_test = []
mse_train = []
for a in alphas_:
    model.set_params(alpha=a)
    model.fit(Xtrain, ytrain)
    yhat_train = model.predict(Xtrain)
    yhat_test = model.predict(Xtest)
    mse_train.append(np.mean(np.power(yhat_train - ytrain, 2)))
    mse_test.append(np.mean(np.power(yhat_test - ytest, 2)))

ax = plt.gca()
ax.plot(alphas_, mse_train, label='train error ridge')
ax.plot(alphas_, mse_test, label='test error ridge')
plt.legend(loc=2)
plt.xlabel('alpha')
plt.ylabel('mse')
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.show()

###################################
#                d                #
###################################
alphas_ = np.logspace(1, -2, base=10)
coefs = []
model = Lasso(fit_intercept=True)
mse_test = []
mse_train = []
for a in alphas_:
    model.set_params(alpha=a)
    model.fit(Xtrain, ytrain)
    yhat_train = model.predict(Xtrain)
    yhat_test = model.predict(Xtest)
    mse_train.append(np.mean(np.power(yhat_train - ytrain, 2)))
    mse_test.append(np.mean(np.power(yhat_test - ytest, 2)))

ax = plt.gca()
ax.plot(alphas_, mse_train, label='train error lasso')
ax.plot(alphas_, mse_test, label='test error lasso')
plt.xlabel('alpha')
plt.ylabel('mse')
plt.legend(loc=2)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.show()

###################################
#                e                #
###################################


def MSE(y, yhat): return np.mean(np.power(y-yhat, 2))

def best_parameter(x, y, method, alphas):
    Xm = x.as_matrix()
    ym = y.as_matrix()

    if method == "lasso":
        model = Lasso(fit_intercept=True)
    elif method == "ridge":
        model = Ridge(fit_intercept=True)

    k_fold = cross_validation.KFold(len(Xm), 10)
    best_cv_mse = float("inf")

    for a in alphas:
        model.set_params(alpha=a)
        mse_list_k10 = [
                    MSE(model.fit(Xm[train], ym[train]).predict(Xm[val]), ym[val])
                    for train, val in k_fold]
        if np.mean(mse_list_k10) < best_cv_mse:
            best_cv_mse = np.mean(mse_list_k10)
            best_alpha = a
            print method, "BEST PARAMETER=%f, MSE(CV)=%f" % (best_alpha, best_cv_mse)


best_parameter(Xtrain, ytrain, 'ridge', np.logspace(2, -2, base=10))
best_parameter(Xtrain, ytrain, 'lasso', np.logspace(1, -2, base=10))
