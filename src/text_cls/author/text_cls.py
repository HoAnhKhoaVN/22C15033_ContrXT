from sklearn.utils import resample
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import model_selection, preprocessing, linear_model, metrics, svm
from sklearn.model_selection import train_test_split
import gensim
from gensim.models.phrases import Phraser
import spacy
from spacy.util import minibatch, compounding
from nltk.corpus import stopwords
import nltk.tokenize
import nltk
import pandas as pd
import numpy as np
import os
from typing import Iterable, List
import string
from time import time
import re
import itertools
import random
import warnings
import pickle
from tqdm import tqdm
from collections import Counter, defaultdict
# from matplotlib import pyplot as plt
from scipy import stats
import json
import math
import gzip
warnings.filterwarnings("ignore")


class EstimatorSelectionHelper:
    """
    A class used to run a GridSearch CV and a score summary
    ---------------------------------------------------------

    Methods
    -------
    fit(X, y, cv=5, n_jobs=3, verbose=1, scoring=None, refit=False)
        Perform a GridSearchCV and fit for each different model
        Ex. 3 parameters with cv=5 -> total 15 fit

    score_summary(sort_by='mean_score')
        Create a score summary for each model sorted by sort_by
    """

    def __init__(self, models, params):
        """
        Parameters
        ----------
        models : dict
            The model to apply
            ex. {'MultinomialNB': MultinomialNB()}
        params : dict
            The params to apply
            ex. {'MultinomialNB': { 'alpha': [0.1, 0.01, 0.001]}
        """
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError(
                "Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.gridcvs = {}
        self.results = defaultdict(lambda: {'params': None, 'score': 0.0})

    def fit(self, X, y, class_names=None, cv=5, n_jobs=-1, verbose=11, scoring=None, refit=False):
        np.random.seed(42)
        random.seed(42)

        for model in self.models:
            print(f'Starting {model}')
            gcv = GridSearchCV(estimator=self.models[model],
                               param_grid=self.params[model],
                               scoring=scoring,
                               n_jobs=-1,
                               cv=cv,
                               verbose=verbose,
                               refit=refit)
            self.gridcvs[model] = gcv

            gcv.fit(X, y)

            # Results for each hyperparameter set
            for mean, std, params in zip(gcv.cv_results_[f'mean_test_score'],
                                         gcv.cv_results_[f'std_test_score'],
                                         gcv.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
                if mean > self.results[model]['score']:
                    self.results[model] = {'params': params, 'score': mean}

            print(f'Best hyperparameter chosen: {self.results[model]}')

if __name__ == "__main__":
    pass