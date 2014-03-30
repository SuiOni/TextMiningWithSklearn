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
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
import sklearn.cross_validation as cv
import sklearn.feature_extraction.text as tx
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
import sklearn.grid_search as gs

# The training data folder must be passed as first argument
dataset = fetch_20newsgroups()
print dataset.filenames.shape
# Explore the dataset
print dataset.target_names
#pl.figure()
#pl.hist(dataset.target, np.unique(dataset.target))
#pl.show()


# Transform data
vectorizer = tx.CountVectorizer()
vectors = vectorizer.fit_transform(dataset.data)
print len(vectorizer.vocabulary_)
print len(vectorizer.stop_words_)
## Split the dataset in training and test set:
docs_train, docs_test, y_train, y_test = cv.train_test_split(
    vectors, dataset.target, test_size=0.5)

# cross validation on the entire dataset
clf = MultinomialNB(alpha=.1)
score = cv.cross_val_score(clf, vectors, dataset.target, scoring='f1', cv=5)

# TASK: Chain the vectorizer with a linear classifier into a Pipeline
# instance. Its variable should be named `pipeline`.
pipeline = Pipeline([
    ('vec', vectorizer),
    ('clf', MultinomialNB()),
])

# Grid search
parameters = {'vec__max_n':(1,2),
              'clf__alpha': (1e-2,1e-3),
              }
gs_clf = gs.GridSearchCV(clf, parameters, n_jobs=1)






