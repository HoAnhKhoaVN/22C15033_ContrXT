
import logging
from typing import List, Text
import pandas as pd
from src.contrxt.data.text_data_manager import TextDataManager
from src.contrxt.data.tabular_data_manager import TabularDataManager
from src.contrxt.trace import Trace

class ContrXT(object):
    def __init__(
        self,
        X_t1: pd.Series,
        predicted_labels_t1: List[Text],        
        X_t2: pd.Series,
        predicted_labels_t2: List[Text],
        data_type: Text = 'text',
        hyperparameters_selection : bool = True,
        log_level = logging.INFO,
        save_path: Text = 'results',
        graphviz_path : Text = 'C:/Program Files (x86)/Graphviz2.38/bin',
        surrogate_type: Text = 'sklearn',
        save_surrogates : bool = False,
        save_csvs : bool = True,
        save_bdds : bool = False
    ) -> None:
        self.hyperparameters_selection = hyperparameters_selection
        self.log_level = log_level
        self.save_path = save_path
        self.graphviz_path = graphviz_path
        self.surrogate_type = surrogate_type
        self.save_surrogates = save_surrogates
        self.save_csvs = save_csvs
        self.save_bdds = save_bdds

        if data_type == 'text':
            self.data_manager = {
                'time_1': TextDataManager(
                    X_t1,
                    predicted_labels_t1,
                    'time_1'
                ),
                'time_2': TextDataManager(
                    X_t2,
                    predicted_labels_t2,
                    'time_2'
                )
            }
        else:
            self.data_manager = {
                'time_1': TabularDataManager(
                    X_t1,
                    predicted_labels_t1,
                    'time_1'
                ),
                'time_2': TabularDataManager(
                    X_t2,
                    predicted_labels_t2,
                    'time_2'
                )
            }

        self.trace = Trace(
            self.data_manager,
            log_level = self.log_level,
            hyperparameters_selection= self.hyperparameters_selection,
            save_path = self.save_path,
            surrogate_type = self.surrogate_type,
            save_surrogates = self.save_surrogates,
            save_csvs = self.save_csvs
        )

        self.explain = None

    def run_trace(
        self,
        percent_dataset: float = 1.0
    )-> None:
        """Run trace to generate BDD

        Args:
            percent_dataset (float, optional): _description_. Defaults to 1.
        """
        self.trace.run_trace(percent_dataset)

if __name__ == '__main__':
    pass