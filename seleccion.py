import pandas as pd
import numpy as np
import sklearn.linear_model as lm
from sklearn.preprocessing import StandardScaler

# Para la modificacion, utilizar el z_score (valor absoluto mas alto)
def fss(x, y, names_x, k=10000):
    p = x.shape[1] - 1
    k = min(p, k)
    names_x = np.array(names_x)
    remaining = range(0, p)
    selected = [p]
    current_score = 0.0
    best_new_score = 0.0
    while remaining and len(selected) <= k:
        score_candidates = []
        for candidate in remaining:
            model = lm.LinearRegression(fit_intercept=False)
            indexes = selected + [candidate]
            x_train = x[:, indexes]
            predictions_train = model.fit(x_train, y).predict(x_train)
            residuals_train = predictions_train - y
            mse_candidate = np.mean(np.power(residuals_train, 2))
            var_est = (mse_candidate * x_train.shape[0]) / (x_train.shape[0] - x_train.shape[1] - 1)
            diag_values = np.diag(np.linalg.pinv(np.dot(x_train.T, x_train)))
            z_scores = np.divide(model.coef_, np.sqrt(np.multiply(var_est, diag_values)))
            z_score_candidate = z_scores[-1]
            score_candidates.append((abs(z_score_candidate), candidate))
            score_candidates.sort()
        best_new_score, best_candidate = score_candidates.pop()
        remaining.remove(best_candidate)
        selected.append(best_candidate)
        print "selected= %s..." % names_x[best_candidate]
        print "totalvars=%d, z_score = %f" % (len(indexes), best_new_score)
    return selected


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
Xtrain = X[istrain]
ytrain = y[istrain]

Xm = Xtrain.as_matrix()
ym = ytrain.as_matrix()
names_regressors = ["lcavol", "lweight",
                    "age", "lbph", "svi", "lcp",
                    "gleason", "pgg45"]

print fss(Xm, ym, names_regressors)
