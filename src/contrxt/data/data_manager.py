import pandas as pd
from typing import List, Text
import logging

class DataManager():
    def __init__(
        self,
        X : pd.Series,
        Y_predicted : List[Text],
        time_label: Text
    ) -> None:
        self.logger = build_logger(
            logging.DEBUG,
            __name__,
            'logs/trace.log'
        )

if __name__ == '__main__':
    pass