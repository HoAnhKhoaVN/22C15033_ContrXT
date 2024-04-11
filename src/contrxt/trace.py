


from collections import defaultdict
import logging
import os
import time
import traceback
from typing import Text

import numpy as np
import pandas as pd
from src.contrxt.surrogate.sklearn_surrogate import SklearnSurrogate
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

    def _save_results(
        self,
        percent_dataset: float = 1
    )-> None:
        """Save results to csv

        Args:
            percent_dataset (float, optional): _description_. Defaults to 1.
        """
        bdd_df = []

        for time_label in ['time_1', 'time_2']:
            for class_id in self.classes:
                class_id = str(class_id)

                try:
                    row = (
                        time_label,
                        class_id,
                        self.bdds[time_label][class_id],
                        self.times[time_label][class_id],
                        percent_dataset,
                        *self.hyperparameters[time_label][class_id].values(),
                        *self.fidelities[time_label][class_id].values()
                    )

                    bdd_df.append(row)

                except KeyError:
                    continue

        hyperparameters_cols = list(self.hyperparameters['time_1'][self.classes[0]].keys())
        fidelity_cols = list(self.fidelities[time_label][self.classes[0]].keys())
        lst_cols = ['time_label', 'class_id', 'bdd_string', 'run_time', 'percent_dataset']
        final_cols = lst_cols + hyperparameters_cols + fidelity_cols
        bdd_df = pd.DataFrame(bdd_df, columns= final_cols)

        self.logger.info(f'Mean fidelity : {round(bdd_df[fidelity_cols[0]].mean(), 3)}')

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        try:
            file_name = f'{self.save_path}/trace.csv'
            self.logger.info(f'Result Trace in file {file_name}')
            bdd_df.to_csv(file_name, sep = ';', decimal= ',')
        except PermissionError:
            self.logger.error('Error: cannot save result, file is use!')

    def _save_surrogate_paths(self):
        """Save surrogate path to csv file
        """
        # region 1: Prepare to save csv file
        cols = ['time_label', 'class_id', 'bdd_string', 'n']
        paths_df = pd.DataFrame([], columns= cols)

        for time_label in ['time_1', 'time_2']:
            for class_id in self.classes:
                class_id = str(class_id)

                try:
                    df_paths_class = pd.DataFrame.from_dict(
                        data= self.paths[time_label][class_id],
                        orient='index',
                    ).reset_index()

                    df_paths_class.columns = ['bdd_string', 'n']

                    df_paths_class['class_id'] = class_id
                    df_paths_class['time_label'] = time_label

                    df_paths_class = df_paths_class[cols]

                    paths_df = pd.concat([paths_df, df_paths_class])
                except KeyError:
                    continue

        # endregion


        # region 2: Save file
        try:
            fn = f'{self.save_path}/surrogate_paths.csv'
            self.logger.info(f'Results the paths that save in {fn}')
            paths_df.to_csv(fn)
        except PermissionError:
            self.logger.error('Error: cannot save result, file is use!')


        # endregion
   
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
                start_time = time.time()

                if self.surrogate_type == 'fair':
                    pass
                else: # sklearn
                    surrogate_explainer = SklearnSurrogate(
                        X = self.data_manager[time_label].surrogate_train_data[class_id],
                        predicted_labels= self.data_manager[time_label].Y_predicted_binarized[class_id],
                        time_label= time_label,
                        class_id= class_id,
                        feature_names= self.data_manager[time_label].feature_names,
                        hyperparameters= self.hyperparameters[time_label][class_id]
                    )
                if self.hyperparameters_selection:
                    surrogate_explainer.hyperparameters_selection()
                    self.hyperparameters[time_label][class_id] = surrogate_explainer.hyperparameters
                
                surrogate_explainer.fit()
                surrogate_explainer.surrogate_to_bdd_string()


                self.bdds[time_label][class_id] = surrogate_explainer.bdd
                self.paths[time_label][class_id] = surrogate_explainer.paths
                self.fidelities[time_label][class_id] = surrogate_explainer.fidelity

                if self.save_surrogates:
                    surrogate_explainer.save_surrogate_image(self.save_path)

                self.times[time_label][class_id] = round(time.time() - start_time, 3)

                break

            except Exception as e:
                self.logger.debug(msg = e)
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