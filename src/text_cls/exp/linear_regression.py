import os
import pandas as pd

TRAIN_PATH = ""

TEST_PATH = ""

if __name__ == "__name__":
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)