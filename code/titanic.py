import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import seaborn as sns
plt.style.use('default')


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


def prediction2(row):
    sex = row['Sex']
    pclass = row['Pclass']
    fare_range = row['fare_range']
    p = 0
    if pclass == 1:
        if sex == 'female':
            if fare_range == '[0,9]':
                p = 1
            elif fare_range == '[10,19]':
                p = 1
            elif fare_range == '[20,29]':
                p = 0.75
            elif fare_range == '[30,39]':
                p = 1
            elif fare_range == '[40,+[':
                p = 0.9
        elif sex == 'male':
            if fare_range == '[0,9]':
                p = 0.2
            elif fare_range == '[10,19]':
                p = 0.090909
            elif fare_range == '[20,29]':
                p = 0.428571
            elif fare_range == '[30,39]':
                p = 0
            elif fare_range == '[40,+[':
                p = 0.285714
    elif pclass == 2:
        if sex == 'female':
            if fare_range == '[0,9]':
                p = 0.8
            if fare_range == '[10,19]':
                p = 1
            elif fare_range == '[20,29]':
                p = 0.857143
            elif fare_range == '[30,39]':
                p = 1
            elif fare_range == '[40,+[':
                p = 1
        elif sex == 'male':
            if fare_range == '[0,9]':
                p = 0.294118
            elif fare_range == '[10,19]':
                p = 0.25
            elif fare_range == '[20,29]':
                p = 0
            elif fare_range == '[30,39]':
                p = 0
            elif fare_range == '[40,+[':
                p = 0.062500
    elif pclass == 3:
        if sex == 'female':
            if fare_range == '[0,9]':
                p = 0.653846
            elif fare_range == '[10,19]':
                p = 0.562500
            elif fare_range == '[20,29]':
                p = 0.5
            elif fare_range == '[30,39]':
                p = 0.8
            elif fare_range == '[40,+[':
                p = 0.473684
        elif sex == 'male':
            if fare_range == '[0,9]':
                p = 0.111111
            elif fare_range == '[10,19]':
                p = 0.25
            elif fare_range == '[20,29]':
                p = 0.107143
            elif fare_range == '[30,39]':
                p = 0.142857
            elif fare_range == '[40,+[':
                p =  0.153846

    return bernoulli.rvs(p)



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

# Esto es para ver la proporción el porcentaje de sobrevivientes por cada grupo
# y determinar prediction 2
# survived = train[train.Survived == 1].groupby(['Pclass', 'Sex', 'fare_range']).size()
# total = train.groupby(['Pclass', 'Sex', 'fare_range']).size()
# print(survived/total)

train = pd.read_csv('../data/titanic-train.csv', sep=';')
test = pd.read_csv('../data/titanic-test.csv')
test_survived = pd.read_csv('../data/gendermodel.csv')
test = pd.concat([test, test_survived], axis=1)

test['fare_range'] = test.apply(add_interval, axis=1)
print(test.head())

precision = 0
recall = 0
for i in range(20):
    test['prediction'] = test.apply(prediction2, axis=1)
    precision += test[test.prediction == 1][test.Survived == 1].size/float(test[test.prediction == 1].size)
    recall += test[test.prediction == 1][test.Survived == 1].size/float(test[test.Survived == 1].size)

print('precision promedio: ' + str(precision/20))
print('recall promedio: ' + str(recall/20))
