# coding=UTF-8
import urllib
import pandas as pd
import re
import time
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer, word_tokenize
from nltk.stem.porter import PorterStemmer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import random

###################################
#                a                #
###################################
# train_data_url = "http://www.inf.utfsm.cl/~jnancu/stanford-subset/polarity.train"
# test_data_url = "http://www.inf.utfsm.cl/~jnancu/stanford-subset/polarity.dev"
# train_data_f = urllib.urlretrieve(train_data_url, "train_data.csv")
# test_data_f = urllib.urlretrieve(test_data_url, "test_data.csv")
# ftr = open("train_data.csv", "r")
# fts = open("test_data.csv", "r")
ftr = open("polarity.train", "r")
fts = open("polarity.dev", "r")
rows = [line.split(" ", 1) for line in ftr.readlines()]
train_df = pd.DataFrame(rows, columns=['Sentiment', 'Text'])
train_df['Sentiment'] = pd.to_numeric(train_df['Sentiment'])
rows = [line.split(" ", 1) for line in fts.readlines()]
test_df = pd.DataFrame(rows, columns=['Sentiment', 'Text'])
test_df['Sentiment'] = pd.to_numeric(test_df['Sentiment'])
print train_df.shape
print test_df.shape

###################################
#                b                #
###################################
def word_extractor(text):
  wordstemer = PorterStemmer()
  commonwords = stopwords.words('english')
  text = re.sub(r'([a-z])\1+', r'\1\1',text)#substitute multiple letter by two
  words = ""
  wordtokens = [ wordstemer.stem(word.lower()) for word in word_tokenize(text.decode('utf-8', 'ignore')) ]
  for word in wordtokens:
    if word not in commonwords:
      words+=" "+word
  return words

print "word extractor with Stemmer"
print word_extractor("I love to eat cake")
print word_extractor("I love eating cake")
print word_extractor("I loved eating the cake")
print word_extractor("I do not love eating cake")
print word_extractor("I don't love eating cake") + '\n'

# print "Sin stemming:"
# print word_extractor("I love to eat cake", stemming=False)
# print word_extractor("I love eating cake", stemming=False)
# print word_extractor("I loved eating the cake", stemming=False)
# print word_extractor("I do not love eating cake", stemming=False)
# print word_extractor("I don't love eating cake", stemming=False)

###################################
#                c                #
###################################

def word_extractor2(text, StopWords):
  wordlemmatizer = WordNetLemmatizer()
  if(StopWords):
   commonwords = stopwords.words('english') 
  text = re.sub(r'([a-z])\1+', r'\1\1',text)#substitute multiple letter by two
  words = ""
  wordtokens = [ wordlemmatizer.lemmatize(word.lower()) for word in word_tokenize(text.decode('utf-8', 'ignore')) ]
  for word in wordtokens:
    if StopWords :
      if word not in commonwords:
        words+=" "+word
    else:
      words+=" "+word
  return words

print "word extractor 2 with Lemmatizer"
print word_extractor2("I love to eat cake",True)
print word_extractor2("I love eating cake",True)
print word_extractor2("I loved eating the cake",True)
print word_extractor2("I do not love eating cake",True)
print word_extractor2("I don't love eating cake",True)
print word_extractor2("Those are stupids dogs",True) + '\n'

###################################
#                d                #
###################################
def inicializacion(tipo='lematizador'):
  if (tipo == 'lematizador'):
    texts_train = [word_extractor2(text,True) for text in train_df.Text]
    texts_test = [word_extractor2(text,True) for text in test_df.Text]
  elif tipo == 'stopwords':
    texts_train = [word_extractor2(text,False) for text in train_df.Text]
    texts_test = [word_extractor2(text,False) for text in test_df.Text]
  elif tipo == 'stemming':
    texts_train = [word_extractor(text) for text in train_df.Text]
    texts_test = [word_extractor(text) for text in test_df.Text]

  vectorizer = CountVectorizer(ngram_range=(1, 1), binary='False')
  vectorizer.fit(np.asarray(texts_train))

  features_train = vectorizer.transform(texts_train) #x
  features_test = vectorizer.transform(texts_test) #xt

  labels_train = np.asarray((train_df.Sentiment.astype(float)+1)/2.0) #y
  labels_test = np.asarray((test_df.Sentiment.astype(float)+1)/2.0) #yt

  vocab = vectorizer.get_feature_names()
  dist=list(np.array(features_train.sum(axis=0)).reshape(-1,))
  return vocab, dist, features_train,labels_train,features_test,labels_test

print "Métricas Lematizador\n"
vocab, dist, features_train,labels_train,features_test,labels_test = inicializacion('lematizador')

tags = []
for tag, count in zip(vocab, dist):
  tags.append((count, tag))

tags.sort()
tags[:] = tags[::-1]

print "\nPalabras más frecuentes en el conjunto de Entrenamiento"
for i in range(0,10):
  print "frecuencia = %d, palabra = %s"% (tags[i][0], tags[i][1])

dist=list(np.array(features_test.sum(axis=0)).reshape(-1,))

tags = []
for tag, count in zip(vocab, dist):
  tags.append((count, tag))

tags.sort()
tags[:] = tags[::-1]

print "\nPalabras más frecuentes en el conjunto de Prueba"
for i in range(0,10):
  print "frecuencia = %d, palabra = %s"% (tags[i][0], tags[i][1])


###################################
#                e                #
###################################
def score_the_model(model,x,y,xt,yt,text):
  acc_tr = model.score(x,y)
  acc_test = model.score(xt[:-1],yt[:-1])
  print "\nTraining Accuracy %s: %f"%(text,acc_tr)
  print "Test Accuracy %s: %f"%(text,acc_test)
  print "Detailed Analysis Testing Results ..."
  print(classification_report(yt, model.predict(xt), target_names=['+','-']))
  precision, recall, fscore, support = precision_recall_fscore_support(yt, model.predict(xt))
  
  if text == 'BernoulliNB':
    metricas_bernoulliNB.append(acc_tr)
    metricas_bernoulliNB.append(acc_test)
    metricas_bernoulliNB.append(np.mean(precision))
    metricas_bernoulliNB.append(np.mean(recall))
    metricas_bernoulliNB.append(np.mean(fscore))
  elif text == 'MULTINOMIAL':
    metricas_multinomial.append(acc_tr)
    metricas_multinomial.append(acc_test)
    metricas_multinomial.append(np.mean(precision))
    metricas_multinomial.append(np.mean(recall))
    metricas_multinomial.append(np.mean(fscore))
  elif text == 'LOGISTIC':
    metricas_logit.append(acc_tr)
    metricas_logit.append(acc_test)
    metricas_logit.append(np.mean(precision))
    metricas_logit.append(np.mean(recall))
    metricas_logit.append(np.mean(fscore))
  elif text == 'SVM':
    metricas_svm.append(acc_tr)
    metricas_svm.append(acc_test)
    metricas_svm.append(np.mean(precision))
    metricas_svm.append(np.mean(recall))
    metricas_svm.append(np.mean(fscore))

#Función creada para determinar el mejor parametro de regularización en los clasificadores que necesitan determinarlo
def score_the_model2(model,x,y,xt,yt,text):
  acc_tr = model.score(x,y)
  acc_test = model.score(xt[:-1],yt[:-1])

  return acc_test


###################################
#                f                #
###################################
metricas_bernoulliNB = []
def do_NAIVE_BAYES(x,y,xt,yt):
  model = BernoulliNB()
  model = model.fit(x, y)
  score_the_model(model,x,y,xt,yt,"BernoulliNB")
  return model

model=do_NAIVE_BAYES(features_train,labels_train,features_test,labels_test)
test_pred = model.predict_proba(features_test)

spl = random.sample(xrange(len(test_pred)), 5)
for text, sentiment in zip(test_df.Text[spl], test_pred[spl]):
  print sentiment, text

print "Métricas Lematizador sin Stopwords"
vocab, dist, features_train,labels_train,features_test,labels_test = inicializacion('stopwords')
model=do_NAIVE_BAYES(features_train,labels_train,features_test,labels_test)

print "Métricas Stemming"
vocab, dist, features_train,labels_train,features_test,labels_test = inicializacion('stemming')
model=do_NAIVE_BAYES(features_train,labels_train,features_test,labels_test)

###################################
#                g                #
###################################
metricas_multinomial = []
def do_MULTINOMIAL(x,y,xt,yt):
  model = MultinomialNB()
  model = model.fit(x, y)
  score_the_model(model,x,y,xt,yt,"MULTINOMIAL")
  return model

print "Clasificador Bayesiano Ingenuo Multinomial"

print "Metricas Lematizador"
vocab, dist, features_train,labels_train,features_test,labels_test = inicializacion('lematizador')
model=do_MULTINOMIAL(features_train,labels_train,features_test,labels_test)
test_pred = model.predict_proba(features_test)

spl = random.sample(xrange(len(test_pred)), 5)
for text, sentiment in zip(test_df.Text[spl], test_pred[spl]):
  print sentiment, text

print "Métricas Lematizador sin Stopwords"
vocab, dist, features_train,labels_train,features_test,labels_test = inicializacion('stopwords')
model=do_MULTINOMIAL(features_train,labels_train,features_test,labels_test)

print "Métricas Stemming"
vocab, dist, features_train,labels_train,features_test,labels_test = inicializacion('stemming')
model=do_MULTINOMIAL(features_train,labels_train,features_test,labels_test)

###################################
#                h                #
###################################
metricas_logit = []
def do_LOGIT(x,y,xt,yt,C):
  if C == 0:
    Cs = [0.01,0.1,10,100,1000]
    best_c = 0
    acc_test = []
    for C in Cs:
      #print "Usando C= %f"%C
      model = LogisticRegression(penalty='l2',C=C)
      model = model.fit(x, y)
      acc_test.append(score_the_model2(model,x,y,xt,yt,"LOGISTIC"))

    best_c = Cs[acc_test.index(max(acc_test))]
    return best_c
  else:
    model = LogisticRegression(penalty='l2',C=C)
    model = model.fit(x, y)
    score_the_model(model,x,y,xt,yt,"LOGISTIC")
    return model

vocab, dist, features_train,labels_train,features_test,labels_test = inicializacion('lematizador')
C = do_LOGIT(features_train,labels_train,features_test,labels_test,0)
print "Mejor parametro de regularización es %f"% C

print "Modelo de Regresión Logística Regularizado"

print "Metricas Lematizador con C = %f"% C
model = do_LOGIT(features_train,labels_train,features_test,labels_test,C)
test_pred = model.predict_proba(features_test)

spl = random.sample(xrange(len(test_pred)), 5)
for text, sentiment in zip(test_df.Text[spl], test_pred[spl]):
  print sentiment, text

print "Métricas Lematizador sin Stopwords con C = %f"% C
vocab, dist, features_train,labels_train,features_test,labels_test = inicializacion('stopwords')
model = do_LOGIT(features_train,labels_train,features_test,labels_test,C)

print "Métricas Stemming con C = %f"% C
vocab, dist, features_train,labels_train,features_test,labels_test = inicializacion('stemming')
model = do_LOGIT(features_train,labels_train,features_test,labels_test,C)


###################################
#                i                #
###################################
metricas_svm = []
def do_SVM(x,y,xt,yt,C):
  if C == 0:
    best_c = 0
    acc_test = []
    Cs = [0.01,0.1,10,100,1000]
    for C in Cs:
      #print "El valor de C que se esta probando: %f"%C
      model = LinearSVC(C=C)
      model = model.fit(x, y)
      acc_test.append(score_the_model2(model,x,y,xt,yt,"SVM"))

    best_c = Cs[acc_test.index(max(acc_test))]
    return best_c
  else:
    model = LinearSVC(C=C)
    model = model.fit(x, y)
    score_the_model(model,x,y,xt,yt,"SVM")
    return model

vocab, dist, features_train,labels_train,features_test,labels_test = inicializacion('lematizador')
C = do_SVM(features_train,labels_train,features_test,labels_test, 0)
print "Mejor parametro de regularización es %f"% C

print "Máquina de Vectores de Soporte (SVM) Lineal"

print "Metricas Lematizador con C = %f"% C
model = do_SVM(features_train,labels_train,features_test,labels_test, C)

print "Métricas Lematizador sin Stopwords con C = %f"% C
vocab, dist, features_train,labels_train,features_test,labels_test = inicializacion('stopwords')
model = do_SVM(features_train,labels_train,features_test,labels_test, C)

print "Métricas Stemming con C = %f"% C
vocab, dist, features_train,labels_train,features_test,labels_test = inicializacion('stemming')
model = do_SVM(features_train,labels_train,features_test,labels_test, C)

###################################
#                j                #
###################################
#Comparacion de modelos con lematizacion
# data to plot
n_groups = 4

accuracy_train = []
accuracy_train.append(metricas_bernoulliNB[0])
accuracy_train.append(metricas_multinomial[0])
accuracy_train.append(metricas_logit[0])
accuracy_train.append(metricas_svm[0])

accuracy_test = []
accuracy_test.append(metricas_bernoulliNB[1])
accuracy_test.append(metricas_multinomial[1])
accuracy_test.append(metricas_logit[1])
accuracy_test.append(metricas_svm[1])

precision = []
precision.append(metricas_bernoulliNB[2])
precision.append(metricas_multinomial[2])
precision.append(metricas_logit[2])
precision.append(metricas_svm[2])

recall = []
recall.append(metricas_bernoulliNB[3])
recall.append(metricas_multinomial[3])
recall.append(metricas_logit[3])
recall.append(metricas_svm[3])

fscore = []
fscore.append(metricas_bernoulliNB[4])
fscore.append(metricas_multinomial[4])
fscore.append(metricas_logit[4])
fscore.append(metricas_svm[4])


# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.15
opacity = 0.8
 
rects1 = plt.bar(index, accuracy_train, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Accuracy Train')
 
rects2 = plt.bar(index + bar_width, accuracy_test, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Accuracy Test')

rects3 = plt.bar(index + 2*bar_width, precision, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Precision')

rects4 = plt.bar(index + 3*bar_width, recall, bar_width,
                 alpha=opacity,
                 color='c',
                 label='Recall')

rects5 = plt.bar(index + 4*bar_width, fscore, bar_width,
                 alpha=opacity,
                 color='m',
                 label='F1-Score')
 
plt.xlabel('Metodos de Clasificacion')
plt.ylabel('Scores')
plt.title('Metricas con Lematizacion')
plt.xticks(index + bar_width, ('BernoulliNB', 'MultinomialNB', 'LogisticRegression', 'LinearSVC'))
plt.legend()

fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.15
opacity = 0.8
 
rects1 = plt.bar(index, accuracy_train, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Accuracy Train')
 
rects2 = plt.bar(index + bar_width, accuracy_test, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Accuracy Test')

rects3 = plt.bar(index + 2*bar_width, precision, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Precision')

rects4 = plt.bar(index + 3*bar_width, recall, bar_width,
                 alpha=opacity,
                 color='c',
                 label='Recall')

rects5 = plt.bar(index + 4*bar_width, fscore, bar_width,
                 alpha=opacity,
                 color='m',
                 label='F1-Score')
 
plt.xlabel('Metodos de Clasificacion')
plt.ylabel('Scores')
plt.title('Metricas con Lematizacion (zoom)')
plt.xticks(index + bar_width, ('BernoulliNB', 'MultinomialNB', 'LogisticRegression', 'LinearSVC'))
plt.legend()
plt.ylim(0.7,0.76)

#Comparacion de modelos con stemming
# data to plot
n_groups = 4

accuracy_train = []
accuracy_train.append(metricas_bernoulliNB[10])
accuracy_train.append(metricas_multinomial[10])
accuracy_train.append(metricas_logit[10])
accuracy_train.append(metricas_svm[10])

accuracy_test = []
accuracy_test.append(metricas_bernoulliNB[11])
accuracy_test.append(metricas_multinomial[11])
accuracy_test.append(metricas_logit[11])
accuracy_test.append(metricas_svm[11])

precision = []
precision.append(metricas_bernoulliNB[12])
precision.append(metricas_multinomial[12])
precision.append(metricas_logit[12])
precision.append(metricas_svm[12])

recall = []
recall.append(metricas_bernoulliNB[13])
recall.append(metricas_multinomial[13])
recall.append(metricas_logit[13])
recall.append(metricas_svm[13])

fscore = []
fscore.append(metricas_bernoulliNB[14])
fscore.append(metricas_multinomial[14])
fscore.append(metricas_logit[14])
fscore.append(metricas_svm[14])


# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.15
opacity = 0.8
 
rects1 = plt.bar(index, accuracy_train, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Accuracy Train')
 
rects2 = plt.bar(index + bar_width, accuracy_test, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Accuracy Test')

rects3 = plt.bar(index + 2*bar_width, precision, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Precision')

rects4 = plt.bar(index + 3*bar_width, recall, bar_width,
                 alpha=opacity,
                 color='c',
                 label='Recall')

rects5 = plt.bar(index + 4*bar_width, fscore, bar_width,
                 alpha=opacity,
                 color='m',
                 label='F1-Score')
 
plt.xlabel('Metodos de Clasificacion')
plt.ylabel('Scores')
plt.title('Metricas con Stemming')
plt.xticks(index + bar_width, ('BernoulliNB', 'MultinomialNB', 'LogisticRegression', 'LinearSVC'))
plt.legend()

fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.15
opacity = 0.8
 
rects1 = plt.bar(index, accuracy_train, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Accuracy Train')
 
rects2 = plt.bar(index + bar_width, accuracy_test, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Accuracy Test')

rects3 = plt.bar(index + 2*bar_width, precision, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Precision')

rects4 = plt.bar(index + 3*bar_width, recall, bar_width,
                 alpha=opacity,
                 color='c',
                 label='Recall')

rects5 = plt.bar(index + 4*bar_width, fscore, bar_width,
                 alpha=opacity,
                 color='m',
                 label='F1-Score')
 
plt.xlabel('Metodos de Clasificacion')
plt.ylabel('Scores')
plt.title('Metricas con Stemming (zoom)')
plt.xticks(index + bar_width, ('BernoulliNB', 'MultinomialNB', 'LogisticRegression', 'LinearSVC'))
plt.legend()
plt.ylim(0.7,0.76)
plt.show()