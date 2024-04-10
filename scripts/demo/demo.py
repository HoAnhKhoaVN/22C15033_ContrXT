import random
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from src.contrxt.contrxt import ContrXT

CSV_1 = 'tests/dataset/df_time_1.csv'
CSV_2 = 'tests/dataset/df_time_2.csv'

def read_dataset()->Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read csv dataset for demo
    """
    # region 1: Read csv
    df_time_1 = pd.read_csv(CSV_1, delimiter= ',')
    df_time_2 = pd.read_csv(CSV_2, delimiter= ',')

    # endregion Read csv

    # region 2. Show dataframe
    print(f'Length D_1: {len(df_time_1)}')
    print(f'Length D_2: {len(df_time_2)}')

    print(f'****** D_1 ******')
    print(df_time_1.head())

    print(f'****** D_2 ******')
    print(df_time_2.head())

    # endregion Show datafram

    return df_time_1, df_time_2

def simple_classifier(
    df_time_1 : pd.DataFrame,
    df_time_2 : pd.DataFrame,
    state: int = 42
)-> Tuple:
    # region 1: Set random state
    np.random.seed(state)
    random.seed(state)

    # endregion

    # region 2: Encode the categorical target
    encoder = preprocessing.LabelEncoder() # Encode the label
    X_t1, X_t2 = df_time_1['corpus'], df_time_2['corpus']
    Y_t1, Y_t2 = (
        encoder.fit_transform(y = df_time_1['category']),
        encoder.fit_transform(y = df_time_2['category']),
    ) 

    # endregion Encode the categorical target

    # region 3: Define vectorizer
    vec_t1 = TfidfVectorizer(
        ngram_range= (1,1),
        max_features= int(21e6) # Maximun feature is 21 milions.
    )

    vec_t2 = TfidfVectorizer(
        max_features= int(21e6),
        ngram_range= (1,1)
    )

    # endregion Define vectorizer

    # region 4: Fit and transform text data ==> Training phrase
    sparse_t1, sparse_t2 = (
        vec_t1.fit_transform(X_t1),
        vec_t2.fit_transform(X_t2),
    )

    # endregion : Fit and transform text data

    # region 5: Train simple Naive Bayes Classifiers
    classiifier_1, classiifier_2 = MultinomialNB(), MultinomialNB()
    classiifier_1.fit(X = sparse_t1, y = Y_t1)
    classiifier_2.fit(X = sparse_t2, y = Y_t2)

    # endregion Train simple Naive Bayes Classifiers

    # region 6: Get class names
    class_names = df_time_1['category'].unique()
    class_names.sort()

    # endregion Get class names

    # region 7: Get model predictions
    prediction_1 = classiifier_1.predict(sparse_t1)
    prediction_2 = classiifier_2.predict(sparse_t2)

    predicted_labels_t1 = [class_names[i] for i in prediction_1]
    predicted_labels_t2 = [class_names[i] for i in prediction_2]

    print(f'predicted_labels_t1 : {predicted_labels_t1[:3]}...')
    print(f'predicted_labels_t2 : {predicted_labels_t2[:3]}...')
    # endregion Get model predictions

    return X_t1, X_t2, predicted_labels_t1, predicted_labels_t2
if __name__ == '__main__':
    df_time_1, df_time_2 = read_dataset()

    X_t1, X_t2, predicted_labels_t1, predicted_labels_t2= simple_classifier(
        df_time_1,
        df_time_2
    )

    print("##### Initialize ContrXT #####")
    exp = ContrXT(
        X_t1,
        predicted_labels_t1,
        X_t2,
        predicted_labels_t2,
        hyperparameters_selection= True,
        save_path = 'results/demo_1',
        save_surrogates = True,
        save_bdds = True
    )

    print("##### Step 1: Trace #####")
    exp.run_trace()

    print("##### Step 2: Run explanation #####")
    exp.run_explain()

    exit(126)

    print("##### Step 3: Binary Decision Diagram to Natural Language #####")
    exp.explain.BDD2Text()