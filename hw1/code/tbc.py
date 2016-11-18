import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

# i
data = pd.read_csv('../data/TB.csv', sep=',', thousands=',', index_col=0)
data.index.names = ['country']
data.columns.names = ['year']
X = data.ix[:, '1990':'2007'].values
X_std = StandardScaler().fit_transform(X)
print(X_std.shape)

# Computing the d-dimensional mean vector
mean_vec = np.mean(X_std, axis=0)
# Cov matrix
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
cov_mat = np.cov(X_std.T)

# eigenvalues and eigenvectors
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

eig_pairs.sort()
eig_pairs.reverse()

tot = sum(eig_vals)
var_exp = [(eig_val / tot)*100 for eig_val, eig_vec in eig_pairs[:4]]

plt.bar(range(4), var_exp, alpha=0.5, align='center', label='individual explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# Choosing 2 eigenvectors with the largest eigenvalues
matrix_w = np.hstack((
    eig_pairs[0][1].reshape(18, 1),
    eig_pairs[1][1].reshape(18, 1),
    )
)

Y = matrix_w.T.dot(X_std.T)
Y = Y.transpose()
print(Y.shape)  # 207 x 2

data_2d = pd.DataFrame(Y)
data_2d.index = data.index
data_2d.columns = ['PC1', 'PC2']

# e
# 1 scatter plot means
row_means = data.mean(axis=1)
data_2d.plot(kind='scatter', x='PC1', y='PC2', figsize=(16, 8), s=100, c=row_means, cmap='OrRd')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# 2 scatter plot trends
row_trends = data.diff(axis=1).mean(axis=1)
data_2d.plot(kind='scatter', x='PC1', y='PC2', figsize=(16, 8), s=100, c=row_trends, cmap='RdBu')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


# f
# 1 bubble plot means
data_2d.plot(kind='scatter',
             x='PC1', y='PC2',
             figsize=(16, 8), s=0.5*row_means,
             c=row_means, cmap='OrRd')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# 2 bubble plot trends
data_2d.plot(kind='scatter',
             x='PC1', y='PC2',
             figsize=(16, 8), s=10*row_trends,
             c=row_trends, cmap='RdBu')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# g labels
fig, ax = plt.subplots(figsize=(16, 8))
data_2d.plot(kind='scatter',
             x='PC1', y='PC2',
             ax=ax, s=0.15*row_means,
             c=row_means, cmap='OrRd')

Q3_TB_world = data.mean(axis=1).quantile(q=0.75)
TB_country = data.mean(axis=1)
names = data.index
for i, txt in enumerate(names):
    if(TB_country[i] > Q3_TB_world):
        ax.annotate(txt, (data_2d.iloc[i].PC1+0.2, data_2d.iloc[i].PC2))

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
