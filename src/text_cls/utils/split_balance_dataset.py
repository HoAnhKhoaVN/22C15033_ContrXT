import pandas as pd
from collections import Counter

from src.text_cls.constant import LABEL


def handle_err_testset(test_path: str)-> None:
    """Handle error in testset. You can write anything in here to address the errors.

    Args:
        test_path (str): Test csv path.
    """
    test_df = pd.read_csv(test_path)

    test_df.drop(index=[1949], inplace = True)
    test_df['label']= test_df['label'].apply(lambda x: 'misc.forsale' if x == 'misc.forsale  ' else x)

    print(f'Length test label: {len(set(test_df["label"]))}')

    test_df.to_csv(test_path, index= False)

if __name__ == "__main__":
    TRAIN_CSV= "src/text_cls/dataset/20newsgroups/split_data/20newgroups/train__preprocess__no_remove_sw__apostrophe__v4_5/train__preprocess__no_remove_sw__apostrophe__v4_5__train.csv"
    train_df = pd.read_csv(TRAIN_CSV)
    Counter(train_df[LABEL])

    print()

