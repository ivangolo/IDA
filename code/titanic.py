# coding=UTF-8
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import seaborn as sns

pd.options.display.mpl_style = 'default'


# a
data = pd.read_csv('../data/titanic-train.csv', sep = ';')


# b
#shape: Return a tuple representing the dimensionality of the DataFrame.
print data.shape
#info: Concise summary of a DataFrame.
print data.info()
#describe: Generate various summary statistics, excluding NaN values.
print data.describe()


# c
#head(n): Returns first n rows
print data[['Sex','Survived']].head(10)
#tail(n): Returns last n rows
print data[['Sex','Survived']].tail(10)
print data[['Sex','Survived']][200:210]

print data['Sex'].value_counts()
grouped_props_1 = data.groupby('Sex')['Survived'].value_counts()
#print grouped_props_1
grouped_props_1 = data.groupby('Sex')['Survived'].value_counts()/data.groupby('Sex').size()
print grouped_props_1

grouped_props_1.unstack().plot(kind='bar',grid=True)

print data.groupby('Survived').size()
grouped_props_2 = data.groupby('Survived')['Sex'].value_counts()
#print grouped_props_2
grouped_props_2 = data.groupby('Survived')['Sex'].value_counts()/data.groupby('Survived').size()
print grouped_props_2

grouped_props_2.unstack().plot(kind='bar', grid=True)


# d
print data['Sex'].value_counts()
grouped_props_1 = data.groupby('Sex')['Survived'].value_counts()
#print grouped_props_1
grouped_props_1 = data.groupby('Sex')['Survived'].value_counts()/data.groupby('Sex').size()
print grouped_props_1

grouped_props_1.unstack().plot(kind='bar',grid=True)

print data.groupby('Survived').size()
grouped_props_2 = data.groupby('Survived')['Sex'].value_counts()
#print grouped_props_2
grouped_props_2 = data.groupby('Survived')['Sex'].value_counts()/data.groupby('Survived').size()
print grouped_props_2

# e
# Histogramas
data.groupby('Survived')['Age'].mean()
data.boxplot(column='Age',by='Survived')
data.hist(column='Age',by='Survived')
sum(data[data.Survived==0]['Age'].isnull())
sum(data[data.Survived==0]['Age'].notnull())
data[data.Age==data['Age'].max()]
data.hist(column='Age',by='Survived')
data.boxplot(column='Age',by='Survived')

# Histogramas y distribuciones de probabilidad
survived_data = data[data.Age.notnull()][data.Survived==1]
died_data = data[data.Age.notnull()][data.Survived==0]
fig, ax = plt.subplots()
sns.distplot(survived_data['Age'])
sns.distplot(died_data['Age'])
plt.show()
pd.options.display.mpl_style = 'default'


# f
#Se usa la edad promedio para imputar las edades
mean_male = data[(data.Age.notnull()) & (data.Sex=='male')]['Age'].mean()
print mean_male
mean_female = data[(data.Age.notnull()) & (data.Sex=='female')]['Age'].mean()
print mean_female
data.loc[(data.Age.isnull()) & (data.Sex=='female'), 'Age'] = mean_female
data.loc[(data.Age.isnull()) & (data.Sex=='male'), 'Age'] = mean_male

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
# del item i
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


# Precision y recall para train data regla de prediccion 1
precision = 0
recall = 0
for i in range(20):
    train['prediction'] = train.apply(prediction, axis=1)
    precision += train[train.prediction == 1][train.Survived == 1].size/float(train[train.prediction == 1].size)
    recall += train[train.prediction == 1][train.Survived == 1].size/float(train[train.Survived == 1].size)

print('Regla de predicción 1 Training data:')
print('precision promedio: ' + str(precision/20))
print('recall promedio: ' + str(recall/20))


# Precision y recall para train data regla de prediccion 2
precision = 0
recall = 0
for i in range(20):
    train['prediction'] = train.apply(prediction2, axis=1)
    precision += train[train.prediction == 1][train.Survived == 1].size/float(train[train.prediction == 1].size)
    recall += train[train.prediction == 1][train.Survived == 1].size/float(train[train.Survived == 1].size)

print('Regla de predicción 2: Training data:')
print('precision promedio: ' + str(precision/20))
print('recall promedio: ' + str(recall/20))



# Importando la data de testing y agregando columna survived
test = pd.read_csv('../data/titanic-test.csv')
test_survived = pd.read_csv('../data/gendermodel.csv')
test = pd.concat([test, test_survived], axis=1)
test['fare_range'] = test.apply(add_interval, axis=1)

# Precision y recall para test data regla de prediccion 1
precision = 0
recall = 0
for i in range(20):
    test['prediction'] = test.apply(prediction, axis=1)
    precision += test[test.prediction == 1][test.Survived == 1].size/float(test[test.prediction == 1].size)
    recall += test[test.prediction == 1][test.Survived == 1].size/float(test[test.Survived == 1].size)

print('Regla de predicción 1 Test data:')
print('precision promedio: ' + str(precision/20))
print('recall promedio: ' + str(recall/20))

# Precision y recall para test data regla de prediccion 2
precision = 0
recall = 0
for i in range(20):
    test['prediction'] = test.apply(prediction2, axis=1)
    precision += test[test.prediction == 1][test.Survived == 1].size/float(test[test.prediction == 1].size)
    recall += test[test.prediction == 1][test.Survived == 1].size/float(test[test.Survived == 1].size)

print('Regla de predicción 2 Test data:')
print('precision promedio: ' + str(precision/20))
print('recall promedio: ' + str(recall/20))
