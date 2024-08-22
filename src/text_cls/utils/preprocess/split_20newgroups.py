# python src/text_cls/utils/preprocess/split_20newgroups.py
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, ShuffleSplit
import tqdm

from src.text_cls.constant import RANDOM_STATE, TEXT

FD_PATH = "src/text_cls/dataset/20newsgroups/split_data/20newgroups"


if __name__ == "__main__":
    print("Split 20 new groups")
    PATH = "src/text_cls/dataset/20newsgroups/orginal/train__preprocess__no_remove_sw__apostrophe__v4_5.csv"
    
    # region : Split the datasets and save them
    random.seed(42)
    np.random.seed(42)

    # endregion

    # region 0: Load Dataframe
    df = pd.read_csv(PATH)
    df.dropna(axis = 0 ,how = 'any', inplace = True, ignore_index= False)
    df.reset_index(drop = True, inplace = True)
    df[TEXT] = df[TEXT].apply(lambda x: x.replace("\n", " "))
    # endregion

    # region 1: Get folder path
    fn = os.path.splitext(os.path.basename(PATH))[0]

    fd_path = os.path.join(FD_PATH, fn)
    os.makedirs(name = fd_path, exist_ok= True)

    TEST_PATH = os.path.join(fd_path, f"{fn}__test.csv")
    TRAIN_PATH = os.path.join(fd_path, f"{fn}__train.csv")

    # endregion

    # region 2: Split train - test
    df_train, df_test = train_test_split(
        df,
        test_size= 0.25,
        random_state= RANDOM_STATE
    )

    df_train.to_csv(TRAIN_PATH, index = False)
    df_test.to_csv(TEST_PATH, index = False)
    # endregion

    # region 3: Split validation
    rs = ShuffleSplit(n_splits=5, test_size=.25, random_state=RANDOM_STATE)
    for i, (train_index, test_index) in enumerate(rs.split(df_train)):
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        tmp_df_train : pd.DataFrame = df_train[df_train.index.isin(train_index)]
        csv_tmp_train_path = os.path.join(fd_path, f'train__{fn}__fold_{i}.csv')
        tmp_df_train.to_csv(csv_tmp_train_path, index = False)
        print(f"  Test:  index={test_index}")
        tmp_df_test : pd.DataFrame = df_train[df_train.index.isin(test_index)]
        csv_tmp_test_path = os.path.join(fd_path, f'val__{fn}__fold_{i}.csv')
        tmp_df_test.to_csv(csv_tmp_test_path, index = False)

    # endregion