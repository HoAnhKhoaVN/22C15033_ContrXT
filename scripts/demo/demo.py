from typing import Tuple
import pandas as pd

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
    print(df_time_1.head())
    print(df_time_2.head())

    # endregion Show datafram

    return df_time_1, df_time_2







if __name__ == '__main__':
    df_time_1, df_time_2 = read_dataset()