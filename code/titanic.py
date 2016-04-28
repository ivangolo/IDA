import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
# import seaborn as sns
plt.style.use('default')


# Funcion utilizada para realizar la prediccion
# del item h
def prediction(row):
    sex = row['Sex']
    pclass = row['Pclass']
    p = 0
    if pclass == 1:
        if sex == 'female':
            p = 0.968085
        elif sex == 'male':
            p = 0.368852
    elif pclass == 2:
        if sex == 'female':
            p = 0.921053
        elif sex == 'male':
            p = 0.157407
    elif pclass == 3:
        if sex == 'female':
            p = 0.500000
        elif sex == 'male':
            p = 0.135447

    return bernoulli.rvs(p)


# Funcion utilizada para realizar la prediccion
# del item h
def prediction2(row):
    sex = row['Sex']
    pclass = row['Pclass']
    fare_range = row['fare_range']
    p = 0
    if pclass == 1:
        if sex == 'female':
            if fare_range == '[20,29]':
                p = 0.833333
            elif fare_range == '[30,39]':
                p = 1.000000
            elif fare_range == '[40,+[':
                p = 0.976744
        elif sex == 'male':
            if fare_range == '[0,9]':
                p = 0
            elif fare_range == '[20,29]':
                p = 0.407407
            elif fare_range == '[30,39]':
                p = 0.478261
            elif fare_range == '[40,+[':
                p = 0.348485
    elif pclass == 2:
        if sex == 'female':
            if fare_range == '[10,19]':
                p = 0.909091
            elif fare_range == '[20,29]':
                p = 0.900000
            elif fare_range == '[30,39]':
                p = 1.000000
            elif fare_range == '[40,+[':
                p = 1.000000
        elif sex == 'male':
            if fare_range == '[0,9]':
                p = 0
            elif fare_range == '[10,19]':
                p = 0.158730
            elif fare_range == '[20,29]':
                p = 0.160000
            elif fare_range == '[30,39]':
                p = 0.375000
            elif fare_range == '[40,+[':
                p = 0
    elif pclass == 3:
        if sex == 'female':
            if fare_range == '[0,9]':
                p = 0.625000
            elif fare_range == '[10,19]':
                p = 0.538462
            elif fare_range == '[20,29]':
                p = 0.350000
            elif fare_range == '[30,39]':
                p = 0.200000
            elif fare_range == '[40,+[':
                p = 0.368421
        elif sex == 'male':
            if fare_range == '[0,9]':
                p = 0.110204
            elif fare_range == '[10,19]':
                p = 0.250000
            elif fare_range == '[20,29]':
                p = 0.150000
            elif fare_range == '[30,39]':
                p = 0.200000
            elif fare_range == '[40,+[':
                p = 0.170732

    return bernoulli.rvs(p)


# Determina el intervalo de la variable Fare
def add_interval(row):
    fare = row['Fare']
    interval = ''
    if 0 <= fare <= 9:
        interval = '[0,9]'
    elif 10 <= fare <= 19:
        interval = '[10,19]'
    elif 20 <= fare <= 29:
        interval = '[20,29]'
    elif 30 <= fare <= 39:
        interval = '[30,39]'
    else:
        interval = '[40,+['

    return interval



train = pd.read_csv('../data/titanic-train.csv', sep=';')
train['fare_range'] = train.apply(add_interval, axis=1)


# Esto es para ver la proporcion el porcentaje de sobrevivientes por cada grupo
# y determinar prediction 1
survived = train[train.Survived == 1].groupby(['Pclass', 'Sex']).size()
total = train.groupby(['Pclass', 'Sex']).size()
print('Proporciones para regla de predicción 1:')
print(survived/total)


# Esto es para ver la proporcion el porcentaje de sobrevivientes por cada grupo
# y determinar prediction 2
survived = train[train.Survived == 1].groupby(['Pclass', 'Sex', 'fare_range']).size()
total = train.groupby(['Pclass', 'Sex', 'fare_range']).size()
print('Proporciones para regla de predicción 2:')
print(survived/total)


precision = 0
recall = 0
for i in range(20):
    train['prediction'] = train.apply(prediction, axis=1)
    precision += train[train.prediction == 1][train.Survived == 1].size/float(train[train.prediction == 1].size)
    recall += train[train.prediction == 1][train.Survived == 1].size/float(train[train.Survived == 1].size)

print('Regla de predicción 1 Training data:')
print('precision promedio: ' + str(precision/20))
print('recall promedio: ' + str(recall/20))


precision = 0
recall = 0
for i in range(20):
    train['prediction'] = train.apply(prediction2, axis=1)
    precision += train[train.prediction == 1][train.Survived == 1].size/float(train[train.prediction == 1].size)
    recall += train[train.prediction == 1][train.Survived == 1].size/float(train[train.Survived == 1].size)

print('Regla de predicción 2: Training data:')
print('precision promedio: ' + str(precision/20))
print('recall promedio: ' + str(recall/20))




test = pd.read_csv('../data/titanic-test.csv')
test_survived = pd.read_csv('../data/gendermodel.csv')
test = pd.concat([test, test_survived], axis=1)
test['fare_range'] = test.apply(add_interval, axis=1)

precision = 0
recall = 0
for i in range(20):
    test['prediction'] = test.apply(prediction, axis=1)
    precision += test[test.prediction == 1][test.Survived == 1].size/float(test[test.prediction == 1].size)
    recall += test[test.prediction == 1][test.Survived == 1].size/float(test[test.Survived == 1].size)

print('Regla de predicción 1 Test data:')
print('precision promedio: ' + str(precision/20))
print('recall promedio: ' + str(recall/20))

precision = 0
recall = 0
for i in range(20):
    test['prediction'] = test.apply(prediction2, axis=1)
    precision += test[test.prediction == 1][test.Survived == 1].size/float(test[test.prediction == 1].size)
    recall += test[test.prediction == 1][test.Survived == 1].size/float(test[test.Survived == 1].size)

print('Regla de predicción 2 Test data:')
print('precision promedio: ' + str(precision/20))
print('recall promedio: ' + str(recall/20))