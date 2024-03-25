import string
import pandas as pd
from typing import Dict, List, Text
import logging
from src.contrxt.utils.logger import build_logger

class DataManager():
    def __init__(
        self,
        X : pd.Series,
        Y_predicted : List[Text],
        time_label: Text,
        random_state : int = 42
    ) -> None:
        self.logger = build_logger(
            logging.DEBUG,
            __name__,
            'logs/trace.log'
        )

        self.X = X
        self.Y_predicted = pd.Series(list(map(lambda y: self.check_column_names(y),Y_predicted)))
        self.Y_predicted_binarized = {}
        self.time_label = time_label
        self.feature_names = None
        self.surrogate_train_data : Dict = {}
        self.random_state = random_state
    
        try:
            if self.X.shape[1]:
                self.df = pd.DataFrame(self.X.copy)
                self.df['Y_predicted'] = self.Y_predicted.copy()
        except IndexError:
            self.df = pd.DataFrame({
                'X': self.X.copy(),
                'Y_predicted': self.Y_predicted.copy()
            })
        self.classes = self.Y_predicted.unique().astype('str')
        self.classes.sort()

    def check_var_names(s: Text)->Text:
        """Check for potential pyeda error and automatically fix."""
        banned_words = ['and', 'or', 'xor', 'not', 'nand', 'nor']
        s_complete = []
        for w in s.split(' '):
            if not w:
                continue
            
            if w.strip().lower() in banned_words:
                continue

            if w in string.digits:
                s_complete.append(f'z{w}')
            
        s = ' '.join(s_complete).title().replace('_', '')
        return s
    
    def filter_classes(
        self,
        classes: List[Text]
    )-> None:
        """Filter the dataframe categories with a list of classes.

        Args:
            classes (List[Text]): List of allowed classes.
        """
        # region 1: Get classes
        self.classes = classes
        self.classes.sort

        # endregion

        # region 2: Filter the class in dataframe
        self.df = self.df[self.df['Y_predicted'].isin(values= self.classes)]

        # endregion



if __name__ == '__main__':
    pass