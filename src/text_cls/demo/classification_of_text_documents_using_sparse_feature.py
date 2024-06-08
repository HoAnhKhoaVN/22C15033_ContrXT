from typing import List, Text, Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

TRAIN_PATH = "src/text_cls/dataset/20newsgroups/orginal/train__split_noun_phrase.csv"
TEST_PATH = "src/text_cls/dataset/20newsgroups/orginal/test__split_noun_phrase.csv"

def size_mb(docs: List[Text])-> float:
    """_summary_

    Args:
        docs (List[Text]): _description_

    Returns:
        float: _description_
    """
    return sum(s.encode('utf-8') for s in docs) / 1e6

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


if __name__ == "__main__":
    pass
