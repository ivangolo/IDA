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
from sklearn.metrics import accuracy_score

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
fig = plt.figure()
ax = fig.add_subplot(111)
for lab, col in zip(mclasses, mcolors):
    ax.scatter(Xred_pca[y == lab, 0], Xred_pca[y == lab, 1], s=40, label=lab, facecolors='none', edgecolors=col)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
leg = ax.legend(loc='upper right', fancybox=True)
fig.tight_layout()
fig.savefig('img/dimred_pca.png')
print "Imagen dimred_pca.png guardada en directorio img"

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
fig = plt.figure()
ax = fig.add_subplot(111)
for lab, col in zip(mclasses, mcolors):
    ax.scatter(Xred_lda[y == lab, 0], Xred_lda[y == lab, 1], label=lab, s=40, facecolors='none', edgecolors=col)
ax.set_xlabel('LDA/Fisher Direction 1')
ax.set_ylabel('LDA/Fisher Direction 2')
leg = ax.legend(loc='upper right', fancybox=True)
fig.tight_layout()
fig.savefig('img/dimred_lda.png')
print "Imagen dimred_lda.png guardada en directorio img"

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
ax.set_xlabel('k', fontsize=16)
ax.set_ylabel('Score', fontsize=16)
ax.legend(loc=1, frameon=False)
ax.tick_params(labelsize=14)
fig.tight_layout()
fig.savefig('img/knn_plot.png')
print "Imagen knn_plot.png guardada en directorio img"

###################################
#                h                #
###################################
models_data = {model: {'train': [], 'test': []} for model in ['LDA', 'QDA', 'KNN']}
# print models_data
for d in range(1, 11):
    pca = PCA(n_components=d)
    Xred_pca = pca.fit_transform(X_std)
    Xred_pca_test = pca.transform(X_std_test)

    # lda
    lda_model = LDA()
    lda_model.fit(Xred_pca, y)
    # qda
    qda_model = QDA()
    qda_model.fit(Xred_pca, y)
    # knn
    knn_model = KNeighborsClassifier(n_neighbors=10)
    knn_model.fit(Xred_pca, y)

    # Train error
    models_data['LDA']['train'].append((d, 1 - accuracy_score(y, lda_model.predict(Xred_pca))))
    models_data['QDA']['train'].append((d, 1 - accuracy_score(y, qda_model.predict(Xred_pca))))
    models_data['KNN']['train'].append((d, 1 - accuracy_score(y, knn_model.predict(Xred_pca))))

    # Test error
    models_data['LDA']['test'].append((d, 1 - accuracy_score(ytest, lda_model.predict(Xred_pca_test))))
    models_data['QDA']['test'].append((d, 1 - accuracy_score(ytest, qda_model.predict(Xred_pca_test))))
    models_data['KNN']['test'].append((d, 1 - accuracy_score(ytest, knn_model.predict(Xred_pca_test))))

for model, data in models_data.items():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    print "Modelo con PCA_{}, min error train: ".format(model), min(data['train'], key=lambda t: t[1])
    print "Modelo con PCA_{}, min error test: ".format(model), min(data['test'], key=lambda t: t[1])
    ax.plot(*zip(*data['train']), color="b", linestyle="-", label="Train data")
    ax.plot(*zip(*data['test']), color="r", linestyle="-", label="Test data")
    ax.set_xlabel('d', fontsize=24)
    ax.set_ylabel('Error', fontsize=24)
    ax.legend(loc=1, prop={'size': 20}, frameon=False)
    ax.tick_params(labelsize=18)
    ax.set_title('PCA y ' + model)
    fig.tight_layout()
    fig.savefig('img/PCA_' + model + '.png')
    print "Imagen PCA_" + model + ".png guardada en directorio img"
###################################
#                i                #
###################################
models_data = {model: {'train': [], 'test': []} for model in ['LDA', 'QDA', 'KNN']}
# print models_data
for d in range(1, 11):
    lda = LDA(n_components=d)
    Xred_lda = lda.fit_transform(X_std, y)
    Xred_lda_test = lda.transform(X_std_test)

    # lda
    lda_model = LDA()
    lda_model.fit(Xred_lda, y)
    # qda
    qda_model = QDA()
    qda_model.fit(Xred_lda, y)
    # knn
    knn_model = KNeighborsClassifier(n_neighbors=10)
    knn_model.fit(Xred_lda, y)

    # Train error
    models_data['LDA']['train'].append((d, 1 - accuracy_score(y, lda_model.predict(Xred_lda))))
    models_data['QDA']['train'].append((d, 1 - accuracy_score(y, qda_model.predict(Xred_lda))))
    models_data['KNN']['train'].append((d, 1 - accuracy_score(y, knn_model.predict(Xred_lda))))

    # Test error
    models_data['LDA']['test'].append((d, 1 - accuracy_score(ytest, lda_model.predict(Xred_lda_test))))
    models_data['QDA']['test'].append((d, 1 - accuracy_score(ytest, qda_model.predict(Xred_lda_test))))
    models_data['KNN']['test'].append((d, 1 - accuracy_score(ytest, knn_model.predict(Xred_lda_test))))

for model, data in models_data.items():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    print "Modelo con LDA_{}, min error train: ".format(model), min(data['train'], key=lambda t: t[1])
    print "Modelo con LDA_{}, min error test: ".format(model), min(data['test'], key=lambda t: t[1])
    ax.plot(*zip(*data['train']), color="b", linestyle="-", label="Train data")
    ax.plot(*zip(*data['test']), color="r", linestyle="-", label="Test data")
    ax.set_xlabel('d', fontsize=24)
    ax.set_ylabel('Error', fontsize=24)
    ax.legend(loc=1, prop={'size': 20}, frameon=False)
    ax.tick_params(labelsize=18)
    ax.set_title('LDA y ' + model)
    fig.tight_layout()
    fig.savefig('img/LDA_' + model + '.png')
    print "Imagen LDA_" + model + ".png guardada en directorio img"
