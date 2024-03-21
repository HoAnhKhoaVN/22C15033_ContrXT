from typing import List, Text
import pandas as pd
from src.contrxt.data.data_manager import DataManager

class TextDataManager(DataManager):

    def __init__(
        self,
        X: pd.Series,
        Y_predicted : List[Text],
        time_label: Text
    )-> None:
        # region 1: Check variable name in pandas series
        X = list(map(lambda x: self.check_var_names(x), X))
        X = pd.Series(X)

        # endregion

        super().__init__(X, Y_predicted, time_label)




if __name__ == '__main__':
    pass