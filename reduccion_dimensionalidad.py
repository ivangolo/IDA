# import urllib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# from sklearn.lda import LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# from sklearn.qda import QDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
# import seaborn as sns
import numpy as np
from matplotlib import colors
from collections import Counter

###################################
#                a                #
###################################
train_df = pd.DataFrame.from_csv('train_data.csv', header=0, index_col=0)
test_df = pd.DataFrame.from_csv('test_data.csv', header=0, index_col=0)
# print train_df.describe()
# print test_df.describe()

# print test_df.tail()

###################################
#                b                #
###################################
X = train_df.ix[:, 'x.1':'x.10'].values
y = train_df.ix[:, 'y'].values
X_std = StandardScaler().fit_transform(X)

###################################
#                c                #
###################################
sklearn_pca = PCA(n_components=2)
Xred_pca = sklearn_pca.fit_transform(X_std)
color_list = ['darkorange', 'black', 'silver', 'royalblue', 'red', 'yellow', 'green', 'brown', 'magenta', 'cyan', 'lime']
cmap = colors.ListedColormap(color_list)
# cmap = plt.cm.get_cmap('Set3')
mclasses = (1, 2, 3, 4, 5, 6, 7, 8, 9)
mcolors = [cmap(i) for i in np.linspace(0, 1, 10)]
plt.figure(figsize=(12, 8))
for lab, col in zip(mclasses, mcolors):
    plt.scatter(Xred_pca[y == lab, 0], Xred_pca[y == lab, 1], s=40, label=lab, facecolors='none', edgecolors=col)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
leg = plt.legend(loc='upper right', fancybox=True)
plt.tight_layout()
# plt.show()

###################################
#                d                #
###################################
sklearn_lda = LDA(n_components=2)
Xred_lda = sklearn_lda.fit_transform(X_std, y)
color_list = ['darkorange', 'black', 'grey', 'royalblue', 'red', 'yellow', 'green', 'brown', 'magenta', 'cyan', 'lime']
cmap = colors.ListedColormap(color_list)
# cmap = plt.cm.get_cmap('ChooseAnAppropriatePalette')
mclasses = (1, 2, 3, 4, 5, 6, 7, 8, 9)
mcolors = [cmap(i) for i in np.linspace(0, 1, 10)]
plt.figure(figsize=(12, 8))
for lab, col in zip(mclasses, mcolors):
    plt.scatter(Xred_lda[y == lab, 0], Xred_lda[y == lab, 1], label=lab, s=40, facecolors='none', edgecolors=col)
plt.xlabel('LDA/Fisher Direction 1')
plt.ylabel('LDA/Fisher Direction 2')
leg = plt.legend(loc='upper right', fancybox=True)
plt.tight_layout()
# plt.show()

###################################
#                e                #
###################################
# Done (?)
###################################
#                f                #
###################################

def mega_clasificador(y_train, x):
    classes = np.unique(y_train)  # get classes ids 1 ... 11
    counts = np.array(sorted(Counter(y_train).items()))[:, 1]  # count ocurrences
    probs = counts.astype(float)/len(y_train)
    prediction = np.random.choice(classes, 1, p=probs)
    return prediction[0]

# print mega_clasificador(y, [1, 2, 0, 3])
###################################
#                g                #
###################################
Xtest = test_df.ix[:, 'x.1':'x.10'].values
ytest = test_df.ix[:, 'y'].values
X_std_test = StandardScaler().fit_transform(Xtest)
lda_model = LDA()
lda_model.fit(X_std, y)
print "Score for Linear Discriminant Analisys"
print "Train data: ", lda_model.score(X_std, y)
print "Test data: ", lda_model.score(X_std_test, ytest)
print "\n"
qda_model = QDA()
qda_model.fit(X_std, y)
print "Score for Quadratic Discriminant Analisys"
print "Train data: ", qda_model.score(X_std, y)
print "Test data: ", qda_model.score(X_std_test, ytest)
print "\n"
knn_model = KNeighborsClassifier(n_neighbors=10)
knn_model.fit(X_std, y)
print "Score for K Keighbors Classifier"
print "Train data: ", knn_model.score(X_std, y)
print "Test data: ", knn_model.score(X_std_test, ytest)

knn_model_train_points = []
knn_model_test_points = []
for k in range(1, 100):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_std, y)
    knn_model_train_points.append((k, knn_model.score(X_std, y)))
    knn_model_test_points.append((k, knn_model.score(X_std_test, ytest)))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(*zip(*knn_model_train_points), color="b", linestyle="-", label="Train data")
ax.plot(*zip(*knn_model_test_points), color="r", linestyle="-", label="Test data")
ax.set_xlabel('k', fontsize=20)
ax.set_ylabel('Score', fontsize=20)
ax.legend(loc=1, frameon=False)
ax.tick_params(labelsize=14)
fig.tight_layout()
plt.show()



###################################
#                h                #
###################################

###################################
#                i                #
###################################
