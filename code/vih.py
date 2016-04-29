import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

# a
data = pd.read_csv('../data/HIV.csv', sep=';', decimal=',', index_col=0)
data = data.drop(data.columns[range(0, 11)], axis=1)
data = data[data['2010'].notnull()]
data = data.dropna()
print(data.shape)  # 139 filas x 22 columnas
data.index.names = ['country']
data.columns.names = ['year']


# b
fig, ax = plt.subplots(figsize=(8, 4))
data.loc[['Chile', 'Argentina', 'Italy', 'Cameroon', 'Congo, Rep,'], '1990':].T.plot(ax=ax)
ax.legend(loc='upper left', bbox_to_anchor=(0.1, 1.1), prop={'size': 'x-small'}, ncol=6)
plt.tight_layout(pad=1.5)
plt.show()


# c
X = data.ix[:, '1990':'2011'].values
X_std = StandardScaler().fit_transform(X)


# d
# Computing the d-dimensional mean vector
X_std = X_std.transpose()  # 22 x 139
mean_vector = np.empty([22, 1])
for i in range(22):
    mean = np.mean(X_std[i, :])
    mean_vector[i] = mean


# Computing the scatter plot
scatter_matrix = np.zeros((22, 22))
for i in range(X_std.shape[1]):
    scatter_matrix += (X_std[:, i].reshape(22, 1) - mean_vector).dot((X_std[:, i].reshape(22, 1) - mean_vector).T)

scatter_matrix = scatter_matrix / (X_std.shape[0] - 1)
eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:, i]) for i in range(len(eig_val_sc))]
eig_pairs.sort()
eig_pairs.reverse()

tot = sum(eig_val_sc)
var_exp = [(eig_val / tot)*100 for eig_val, eig_vec in eig_pairs[:4]]

plt.bar(range(4), var_exp, alpha=0.5, align='center',
        label='individual explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Choosing 2 eigenvectors with the largest eigenvalues
matrix_w = np.hstack((
    eig_pairs[0][1].reshape(22, 1),
    eig_pairs[1][1].reshape(22, 1),
    )
)

# Compute the projection
Y = matrix_w.T.dot(X_std)
Y = Y.transpose()

data_2d = pd.DataFrame(Y)
data_2d.index = data.index
data_2d.columns = ['PC1', 'PC2']

# e
# 1 scatter plot means
row_means = data.mean(axis=1)
print(row_means.mean())
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
             figsize=(16, 8), s=50*row_means,
             c=row_means, cmap='OrRd')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# 2 bubble plot trends
data_2d.plot(kind='scatter',
             x='PC1', y='PC2',
             figsize=(16, 8), s=750*row_trends,
             c=row_means, cmap='RdBu')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# g labels
fig, ax = plt.subplots(figsize=(16,8))
row_means = data.mean(axis=1)
row_trends = data.diff(axis=1).mean(axis=1)
data_2d.plot(kind='scatter', x='PC1', y='PC2', ax=ax, s=10*row_means, c=row_means, \
cmap='RdBu')
Q3_HIV_world = data.mean(axis=1).quantile(q=0.85)
HIV_country = data.mean(axis=1)
names = data.index
for i, txt in enumerate(names):
    if(HIV_country[i]>Q3_HIV_world):
        ax.annotate(txt, (data_2d.iloc[i].PC1+0.2,data_2d.iloc[i].PC2))
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
