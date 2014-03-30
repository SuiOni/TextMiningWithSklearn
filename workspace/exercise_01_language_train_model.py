"""Build a language detector model

The goal of this exercice is to train a linear classifier on text features
that represent sequences of up to 3 consecutive characters so as to be
recognize natural languages by using the frequencies of short character
sequences as 'fingerprints'.

"""
# Author: Olivier Grisel <olivier.grisel@ensta.org>
# License: Simplified BSD

import sys

#from sklearn.feature_extraction import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn import metrics

import sklearn.naive_bayes as nb
import sklearn.pipeline as pp
import sklearn.cross_validation as cv
import sklearn.feature_extraction.text as tx
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
import sklearn.grid_search as gs

###### 
# fetch data
dataset = fetch_20newsgroups()
print dataset.filenames.shape

# Explore the dataset
print sys.getsizeof(dataset)
print len(dataset.data)
print dataset.target_names
print dataset.target

#pl.figure()
#pl.hist(dataset.target, np.unique(dataset.target))
#pl.show()


##### 
# Transform data to count
count_transformer = tx.CountVectorizer()
data_count = count_transformer.fit_transform(dataset.data)

# Obtain tfidf
tfidf_transformer = tx.TfidfTransformer()
data_tfidf = tfidf_transformer.fit_transform(data_count)

# display transformed data info
print data_tfidf.shape
print len(count_transformer.vocabulary_)
print len(count_transformer.stop_words_)


##### 
# Choose classifier
clf = nb.MultinomialNB()


##### 
## Train and Predict, for proof of concept 
#clf.fit(data_tfidf, dataset.target)
#test_count = count_transformer.transform(['God is love', 'intel makes the best processor'])
#test_tfidf = tfidf_transformer.transform(test_count)
#res = clf.predict(test_tfidf)
#for i in res: print dataset.target_names[i]

##### 
# Pipeline the classification procedure
pipeline = pp.Pipeline([('vect', tx.CountVectorizer()),
                        ('tfidf', tx.TfidfTransformer()),
                        ('clf', nb.MultinomialNB()),])

parameters = {
            'tfidf__use_idf': (True, False),
            'clf__alpha': (1e-2,1e-3),
            }

#####
# Grid search for optimal parameter
gs_clf = gs.GridSearchCV(pipeline, parameters, n_jobs=1)
gs_clf.fit(dataset.data, dataset.target)  # remember to use original data as training

# display grid search result
print gs_clf.best_params_
print gs_clf.best_score_


# Predict the new instance
dd = ['God is love', 'intel makes the best processor']
res = gs_clf.predict(dd)

for i in res:
    print dataset.target_names[i]



