import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# a
data = pd.read_csv('../data/HIV.csv', sep=';', decimal=',', index_col=0)
data = data.drop(data.columns[range(0, 11)], axis=1)
data = data[data['2010'].notnull()]
data = data.dropna()
print(data.shape)
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
print(X_std)