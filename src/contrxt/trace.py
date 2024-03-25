


import logging
import os
from typing import Text

import numpy as np
from contrxt.data.data_manager import DataManager
from contrxt.utils.logger import build_logger


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

