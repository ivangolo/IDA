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
# plt.show()


# c
X = data.ix[:, '1990':'2011'].values
X_std = StandardScaler().fit_transform(X)
# print(X_std)


# d
# Computing the d-dimensional mean vector
X_std = X_std.transpose()  # 22 x 139
mean_vector = np.empty([22, 1])
for i in range(22):
    mean = np.mean(X_std[i, :])
    mean_vector[i] = mean

print('Mean Vector:\n', mean_vector)

# Computing the scatter plot
scatter_matrix = np.zeros((22, 22))
for i in range(X_std.shape[1]):
    scatter_matrix += (X_std[:, i].reshape(22, 1) - mean_vector).dot((X_std[:, i].reshape(22, 1) - mean_vector).T)
print('Scatter Matrix:\n', scatter_matrix)

eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:, i]) for i in range(len(eig_val_sc))]
eig_pairs.sort()
eig_pairs.reverse()

# Choosing 4 eigenvectors with the largest eigenvalues
matrix_w = np.hstack((
    eig_pairs[0][1].reshape(22, 1),
    eig_pairs[2][1].reshape(22, 1),
    eig_pairs[3][1].reshape(22, 1),
    eig_pairs[4][1].reshape(22, 1),
    )
)

print('Matrix W:\n', matrix_w)

transformed = matrix_w.T.dot(X_std)
assert transformed.shape == (4, 139), "The matrix is not 2x40 dimensional."

print('Matrix W:\n', matrix_w)



# data_2d = pd.DataFrame(Y_sklearn)
# data_2d.index = data.index
# data_2d.columns = ['PC1', 'PC2']
