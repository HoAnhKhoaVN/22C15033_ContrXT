# python src/text_cls/demo/classification_of_text_documents_using_sparse_feature.py

from typing import List, Text, Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time

from src.text_cls.constant import LABEL, TEXT

TRAIN_PATH = "src/text_cls/dataset/20newsgroups/orginal/train__split_noun_phrase__remove_outlier_char.csv"
TRAIN_PATH = "src/text_cls/dataset/20newsgroups/orginal/train__split_noun_phrase__remove_outlier_char__remove_word_outlier.csv"
TEST_PATH = "src/text_cls/dataset/20newsgroups/orginal/test__split_noun_phrase.csv"

def size_mb(docs: List[Text])-> float:
    """_summary_

    Args:
        docs (List[Text]): _description_

    Returns:
        float: _description_
    """
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6

def load_dataset(
    verbose : bool = False,
)-> Tuple:
    """_summary_

    Args:
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        Tuple: _description_
    """
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)

    df_train.dropna(axis = 0 ,how = 'any', inplace = True, ignore_index= False)
    df_test.dropna(axis = 0 ,how = 'any', inplace = True, ignore_index= False)

    df_train.reset_index(drop= True, inplace= True)
    df_test.reset_index(drop= True, inplace= True)

    train_data, test_data = df_train[TEXT], df_test[TEXT]
    y_train, y_test = df_train[LABEL], df_test[LABEL]

    # region Extracting features from the training data using a sparese vectorizer
    vectorizer = TfidfVectorizer(
        sublinear_tf= True,
        max_df= 1.0,
        min_df= 5,
        stop_words= 'english'
    )
    t0 = time()
    X_train = vectorizer.fit_transform(train_data)
    duration_train = time() - t0

    t1 = time()
    X_test = vectorizer.transform(raw_documents= test_data)
    duration_test = time() - t1
    
    feature_names = vectorizer.get_feature_names_out()

    target_names = y_train.unique().tolist()
    
    # endregion

    # region show log
    if verbose:
        # region Compute  size of loaded data
        data_train_size_mb = size_mb(train_data)
        data_test_size_mb = size_mb(test_data)
        # endregion

        print(
            f'{len(train_data)} documents - '
            f'{data_train_size_mb:.2f} MB (training set)'
        )

        print(
            f'{len(test_data)} documents - '
            f'{data_test_size_mb:.2f} MB (test set)'
        )

        print(f"{len(target_names)} categories")
        print(
            f"vectorize training done in {duration_train:.3f}s "
            f"at {data_train_size_mb / duration_train:.3f}MB/s"
        )
        print(f"n_samples: {X_train.shape[0]}, n_features: {X_train.shape[1]}")
        print(
            f"vectorize testing done in {duration_test:.3f}s "
            f"at {data_test_size_mb / duration_test:.3f}MB/s"
        )
        print(f"n_samples: {X_test.shape[0]}, n_features: {X_test.shape[1]}")
    # endregion

    return X_train, X_test, y_train, y_test, feature_names, target_names
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names, target_names = load_dataset(verbose=True)
