


from collections import defaultdict
import logging
import os
import time
import traceback
from typing import Text

import numpy as np
from src.contrxt.data.data_manager import DataManager
from src.contrxt.utils.logger import build_logger


class Trace(object):
    """_summary_

    Args:
        object (_type_): _description_
    """
    def __init__(
        self,
        data_manager: DataManager,
        hyperparameters_selection: bool = True,
        log_level : Text = logging.INFO,
        save_path : Text = 'results',
        surrogate_type: Text = 'sklearn',
        save_surrogates : bool = False,
        save_csvs: bool = True
    ) -> None:
        """Initialize class Trace

        Args:
            data_manager (DataManager): _description_
            hyperparameters_selection (bool, optional): _description_. Defaults to True.
            log_level (Text, optional): _description_. Defaults to logging.INFO.
            save_path (Text, optional): _description_. Defaults to 'results'.
            surrogate_type (Text, optional): _description_. Defaults to 'sklearn'.
            save_surrogates (bool, optional): _description_. Defaults to False.
            save_csvs (bool, optional): _description_. Defaults to True.
        """
        self.logger = build_logger(
            log_level= log_level,
            log_name= __name__,
            out_file='logs/trace.log'
        )

        self.data_manager :DataManager = data_manager

        # region Get save path
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        # endregion 
        
        self.surrogate_type = surrogate_type
        self.save_surrogates = save_surrogates
        self.save_csvs = save_csvs
        self.hyperparameters_selection = hyperparameters_selection

        self._assign_classes()
        self._initialize_hyperparamaters()
        self._initialize_result_dicts()
 
    def _assign_classes(self)->None:
        """Get list of common classes.
        """
        # region 1: Get list of class
        classes_t1 = set(self.data_manager['time_1'].classes)
        classes_t2 = set(self.data_manager['time_2'].classes)
        self.classes = np.array(list(classes_t1.intersection(classes_t2)))
        self.classes.sort()
        self.logger.info(f'List of common classes: {self.classes}')

        # endregion

        # region 2: Filter class in each dataset
        self.data_manager['time_1'].filter_classes(self.classes)
        self.data_manager['time_2'].filter_classes(self.classes)
        # endregion

    def _initialize_result_dicts(self)->None:
        """Initialize result dictionaries.
        """
        self.bdds = {'time_1': {}, 'time_2': {}}
        self.paths = {'time_1': {}, 'time_2': {}}
        self.times = {'time_1': {}, 'time_2': {}}
        self.fidelities = {'time_1': {}, 'time_2': {}}

    def _initialize_hyperparamaters(self)->None:
        """Initialize hyperparamaters dictionary.
        """
        if self.surrogate_type == 'fair':
            self.hyperparameters = {
                'time_1': defaultdict(
                    lambda : {
                        'max_depth': 5,
                        'min_samples_split': 0.02,
                        'predict_threshold': 0.5,
                        'sensitive_class': None,
                        'sensitive_feature': None
                    }
                ),
                'time_2': defaultdict(
                    lambda : {
                        'max_depth': 5,
                        'min_samples_split': 0.02,
                        'predict_threshold': 0.5,
                        'sensitive_class': None,
                        'sensitive_feature': None
                    }
                )
            } 

        else: # sklearn
            self.hyperparameters = {
                'time_1': defaultdict(
                    lambda : {
                        'max_depth': 5,
                        'min_samples_split': 0.02,
                        'criterion': 'gini',
                        'min_samples_leaf': 0.01
                    }
                ),
                'time_2': defaultdict(
                    lambda : {
                        'max_depth': 5,
                        'min_samples_split': 0.02,
                        'criterion': 'gini',
                        'min_samples_leaf': 0.01
                    }
                )
            }

    def _generate_tree(
        self,
        time_label : Text,
    )->None:
        """Generate surrogate tree

        Args:
            time_label (Text): Description of paramater `time_label`
        """
        for class_id in self.classes:
            try:
                self.logger.info(f'Starting explanation in {time_label} for class_id {class_id}')
                start_time = time()

                if self.surrogate_type == 'fair':
                    pass
                else: # sklearn
                    surrogate_explainer = SklearnSurrogate(
                       
                    )

            except Exception as e:
                self.logger.debug(nsg = e)
                self.logger.exception(traceback.print_exc())
                break

    def run_trace(
        self,
        percent_dataset : float = 1.0
    )-> None:
        """Run trace to generate BDD

        Args:
            percent_dataset (float, optional): _description_. Defaults to 1.0.
        """

        # region 1: Generate data prediction
        for time_label in ['time_1', 'time_2']:
            self.data_manager[time_label].generate_data_predictions(percent_dataset)
            self._generate_tree(time_label)

        # endregion

        # region 2: save csv file
        if self.save_csvs:
            self._save_results(percent_dataset)
            self._save_surrogate_paths()

        # endregion