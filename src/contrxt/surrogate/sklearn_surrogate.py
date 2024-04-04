
import random
import time
from typing import Any, Dict, List, Text
import numpy as np
from pandas import Series
from sklearn import tree
from sklearn.model_selection import RandomizedSearchCV
from src.contrxt.surrogate.genetic_surrogate import GeneticSurrogate


class SklearnSurrogate(GeneticSurrogate):
    def __init__(
        self,
        X: Series,
        predicted_labels: List[Text],
        time_label: Text,
        class_id: Text,
        feature_names: Any,
        hyperparameters: Any
    ) -> None:
        super().__init__(
            X,
            predicted_labels,
            time_label,
            class_id,
            feature_names
        )

        self.hyperparameters = hyperparameters
        self.bdd = None
        self.paths = {}
        self.fit_time = None
        self.surrogate_predictions = None
        self.fidelity = None
        self._model = tree.DecisionTreeClassifier(
            splitter= "best",
            criterion= self.hyperparameters['criterion'],
            min_samples_leaf=self.hyperparameters['min_samples_leaf'],
            max_depth=self.hyperparameters['max_depth'],
            min_samples_split=self.hyperparameters['min_samples_split'],
            max_features= None,
            random_state= 42,
            max_leaf_nodes= None,
            class_weight= 'balanced'
        )

    def hyperparameters_selection(
        self,
        grid_para: Dict = None,
        cv : int = 5
    )-> None:
        """Selection hyperparameters for surrogate model.

        Args:
            grid_para (Dict, optional): _description_. Defaults to None.
            cv (int, optional): _description_. Defaults to 5.
        """
        # region 1: Init
        start_time = time()
        np.random.seed(42)
        random.seed(42)
        self.logger.info(f'Begining hyperparameter selection...')

        # endregion Init

        # region 2: Create a grid for search
        default_grid = {
            'criterion' : ['gini', 'entropy'],
            'min_sample_leaf': [0.01, 0.02],
            'max_depth': [3, 5, 7],
            'min_sample_split': [0.01, 0.02, 0.03]
        }
        seach_space = grid_para if grid_para is not None else default_grid

        # endregion

        # region 3: Selection the best parameter
        # Cost function aiming to optimize(Total Cost) = measure of fit + measure of complexity
        # References for pruning:
        # 1. http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        # 2. https://www.coursera.org/lecture/ml-classification/optional-pruning-decision-trees-to-avoid-overfitting-qvf6v
        # Using Randomize Search here to prune the trees to improve readability without
        # comprising on model's performance
        verbose_level = 4 if self.logger.level >= 20 else 0
        random_search_estimator = RandomizedSearchCV(
            estimator= self._model,
            param_distributions= seach_space,
            n_iter= 10,
            scoring= 'f1',
            n_jobs= -1,
            random_state= 42,
            verbose= verbose_level
        )

        # endregion

        # region 4: Train a surrogate model
        random_search_estimator.fit(
            X = self.X,
            y = self.predicted_labels
        )

        # endregion

        # region 5: Access the best estimator
        self.model_ = random_search_estimator.best_estimator_
        self.hyperparameters['max_depth'] = self.model_.max_depth
        self.hyperparameters['min_sample_split'] = self.model_.min_sample_split
        self.logger.info(f'Time for fitting surrogate: {round(time() - start_time, 3)}')
        self.logger.info(f'Best model: {self._model}')

        # endregion

