import urllib
import pandas as pd
import re
import time
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer, word_tokenize
from nltk.stem.porter import PorterStemmer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

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
# print train_df.head()
# print test_df.head()

###################################
#                b                #
###################################
# def word_extractor(text):
#     wordlemmatizer = WordNetLemmatizer()
#     commonwords = stopwords.words('english')
#     text = re.sub(r'([a-z])\1+', r'\1\1', text)  # substitute multiple letter by two
#     words = ""
#     wordtokens = [wordlemmatizer.lemmatize(word.lower()) for word in word_tokenize(text.decode('utf-8', 'ignore'))]
#
#     for word in wordtokens:
#         if word not in commonwords:
#             words += " " + word
#
#     return words

def word_extractor(text, stemming=True):
    words = word_tokenize(text.decode('utf-8', 'ignore'))
    commonwords = stopwords.words('english')
    words = [word.lower() for word in words if word.lower() not in commonwords]
    if stemming:
        porter = PorterStemmer()
        words = [porter.stem(word).encode('utf-8') for word in words]

    words = ' '.join(words)
    return words

print "Con stemming:"
print word_extractor("I love to eat cake")
print word_extractor("I love eating cake")
print word_extractor("I loved eating the cake")
print word_extractor("I do not love eating cake")
print word_extractor("I don't love eating cake")
print word_extractor("Those are stupids dogs")

print "Sin stemming:"
print word_extractor("I love to eat cake", stemming=False)
print word_extractor("I love eating cake", stemming=False)
print word_extractor("I loved eating the cake", stemming=False)
print word_extractor("I do not love eating cake", stemming=False)
print word_extractor("I don't love eating cake", stemming=False)

###################################
#                c                #
###################################

def word_extractor2(text):
    wordlemmatizer = WordNetLemmatizer()
    commonwords = stopwords.words('english')
    # text = re.sub(r'([a-z])\1+', r'\1\1', text)  # substitute multiple letter by two
    # print text
    words = word_tokenize(text.decode('utf-8', 'ignore'))
    words = [word.lower() for word in words if word.lower() not in commonwords]
    words = [wordlemmatizer.lemmatize(word) for word in words]
    words = ' '.join(words)
    return words

print word_extractor2("I love to eat cake")
print word_extractor2("I love eating cake")
print word_extractor2("I loved eating the cake")
print word_extractor2("I do not love eating cake")
print word_extractor2("I don't love eating cake")
print word_extractor2("Those are stupids dogs")

###################################
#                d                #
###################################
texts_train = [word_extractor2(text) for text in train_df.Text]
texts_test = [word_extractor2(text) for text in test_df.Text]
vectorizer = CountVectorizer(ngram_range=(1, 1), binary='False')
vectorizer.fit(np.asarray(texts_train))
features_train = vectorizer.transform(texts_train)
features_test = vectorizer.transform(texts_test)
labels_train = np.asarray((train_df.Sentiment.astype(float)+1)/2.0)
labels_test = np.asarray((test_df.Sentiment.astype(float)+1)/2.0)
vocab = vectorizer.get_feature_names()
dist = list(np.array(features_train.sum(axis=0)).reshape(-1,))
# for tag, count in zip(vocab, dist):
#     print count, tag


###################################
#                e                #
###################################
def score_the_model(model, x, y, xt, yt, text):
    acc_tr = model.score(x, y)
    acc_test = model.score(xt[:-1], yt[:-1])
    print "Training Accuracy %s: %f" % (text, acc_tr)
    print "Test Accuracy %s: %f" % (text, acc_test)
    print "Detailed Analysis Testing Results ..."
    print(classification_report(yt, model.predict(xt), target_names=['+', '-']))


# from sklearn import svm
# classifier_rbf = svm.SVC()
# classifier_rbf.fit(features_train, labels_train)
# score_the_model(classifier_rbf, features_train, labels_train, features_test, labels_test, "this movie sucks")


###################################
#                f                #
###################################
def do_NAIVE_BAYES(x, y, xt, yt):
    model = BernoulliNB()
    model = model.fit(x, y)
    score_the_model(model, x, y, xt, yt, "BernoulliNB")
    return model

# model = do_NAIVE_BAYES(features_train, labels_train, features_test, labels_test)
# test_pred = model.predict_proba(features_test)
# spl = random.sample(xrange(len(test_pred)), 15)
# for text, sentiment in zip(test_df.Text[spl], test_pred[spl]):
#     print sentiment, text

###################################
#                g                #
###################################
def do_MULTINOMIAL(x, y, xt, yt):
    model = MultinomialNB()
    model = model.fit(x, y)
    score_the_model(model, x, y, xt, yt, "MULTINOMIAL")
    return model

model = do_MULTINOMIAL(features_train, labels_train, features_test, labels_test)
test_pred = model.predict_proba(features_test)
spl = random.sample(xrange(len(test_pred)), 15)
# for text, sentiment in zip(test_df.Text[spl], test_pred[spl]):
#     print sentiment, text

###################################
#                h                #
###################################
def do_LOGIT(x, y, xt, yt):
    start_t = time.time()
    Cs = [0.01, 0.1, 10, 100, 1000]
    for C in Cs:
        print "Usando C= %f" % C
        model = LogisticRegression(penalty='l2', C=C)
        model = model.fit(x, y)
        score_the_model(model, x, y, xt, yt, "LOGISTIC")

do_LOGIT(features_train, labels_train, features_test, labels_test)


###################################
#                i                #
###################################
def do_SVM(x, y, xt, yt):
    Cs = [0.01, 0.1, 10, 100, 1000]
    for C in Cs:
        print "El valor de C que se esta probando: %f" % C
        model = LinearSVC(C=C)
        model = model.fit(x, y)
        score_the_model(model, x, y, xt, yt, "SVM")


do_SVM(features_train, labels_train, features_test, labels_test)
