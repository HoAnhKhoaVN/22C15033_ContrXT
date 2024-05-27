import os
import pandas as pd
from sklearn.model_selection import train_test_split

class DataFrameSplitter:
    """
    A class for splitting a pandas DataFrame into multiple subsets with random state control and train-test splits.

    Attributes:
        None

    Methods:
        split_dataframe(df: pd.DataFrame, random_state: int = None) -> tuple[pd.DataFrame, pd.DataFrame]:
            Splits the DataFrame into two equal parts using a specified random state.

        split_train_test(df: pd.DataFrame, test_size: float = 0.3, random_state: int = None) -> tuple[pd.DataFrame, pd.DataFrame]:
            Splits a DataFrame into training and testing sets with a specified test size and random state.
    """

    @staticmethod
    def split_dataframe(df: pd.DataFrame, random_state: int = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits a DataFrame into two equal parts using a specified random state.

        Args:
            df: The DataFrame to split.
            random_state: The random seed for reproducibility (default: None).

        Returns:
            A tuple containing the two resulting DataFrames.
        """
        df_shuffled = df.sample(frac=1, random_state=random_state)
        midpoint = len(df_shuffled) // 2
        return df_shuffled[:midpoint], df_shuffled[midpoint:]

    @staticmethod
    def split_train_test(
        df: pd.DataFrame,
        test_size: float = 0.3,
        random_state: int = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits a DataFrame into training and testing sets with a specified test size and random state.

        Args:
            df: The DataFrame to split.
            test_size: The proportion of the dataset to include in the test split (default: 0.3).
            random_state: The random seed for reproducibility (default: None).

        Returns:
            A tuple containing the training and testing DataFrames.
        """
        return train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            shuffle= False
        )


if __name__ == "__main__":
    splitter = DataFrameSplitter()
    train_df = pd.read_csv("src/text_cls/dataset/VNTC/original/word__remove_stop_words/train__word__remove_stop_words.csv")

    df_train_time1, df_train_time2 = splitter.split_dataframe(train_df, random_state=2103)

    df_time1_train, df_time1_val = splitter.split_train_test(df_train_time1, test_size=0.3, random_state=2103)
    df_time2_train, df_time2_val = splitter.split_train_test(df_train_time2, test_size=0.3, random_state=2103)

    print("TIME 1")
    df_time1_train.to_csv(
        "src/text_cls/dataset/VNTC/split_data/time_1/word__remove_stop_words/train.csv",
        index = False,
    )
    df_time1_val.to_csv(
        "src/text_cls/dataset/VNTC/split_data/time_1/word__remove_stop_words/val.csv",
        index = False,
    )

    os.system(
        command= f'cp src/text_cls/dataset/VNTC/original/word__remove_stop_words/test__word__remove_stop_words.csv src/text_cls/dataset/VNTC/split_data/time_1/word__remove_stop_words/test.csv'
    )

    print("TIME 2")
    df_time2_train.to_csv(
        "src/text_cls/dataset/VNTC/split_data/time_2/word__remove_stop_words/train.csv",
        index = False,
    )
    df_time2_val.to_csv(
        "src/text_cls/dataset/VNTC/split_data/time_2/word__remove_stop_words/val.csv",
        index = False,
    )

    os.system(
        command= f'cp src/text_cls/dataset/VNTC/original/word__remove_stop_words/test__word__remove_stop_words.csv src/text_cls/dataset/VNTC/split_data/time_2/word__remove_stop_words/test.csv'
    )