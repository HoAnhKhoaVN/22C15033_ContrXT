
import os
import random
import time
from typing import Any, Dict, List, Text
import numpy as np
from pandas import Series
from sklearn import tree
from sklearn.model_selection import RandomizedSearchCV
from src.contrxt.surrogate.genetic_surrogate import GeneticSurrogate
from sklearn import metrics
from sklearn.tree._tree import TREE_UNDEFINED
from pydot import graph_from_dot_data

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

    def fit(self):
        """Train model surrogate with dataset for `class_id`.
        """
        # region 1: Prepare
        s = time()
        np.random.seed(42)
        random.seed(42)

        # endregion: Prepare

        # region 2: Train model
        self._model.fit(
            X = self.X,
            y = self.predicted_labels
        )

        # endregion: Train model

        # region 3: Cacl the score
        self.score()

        # endregion

    def score(self):
        """Calculate the score for sklearn model. It is the fidelity score.
        """
        # region 1: Predict on training set
        self.surrogate_predictions = self._model.predict(X = self.X)

        # endregion

        # region 2: Compute the fidelity
        self.fidelity = {
            'f1_binary': metrics.f1_score(
                y_true= self.predicted_labels,
                y_pred= self.surrogate_predictions,
                average= 'binary'
            ),
            'f1_macro': metrics.f1_score(
                y_true= self.predicted_labels,
                y_pred= self.surrogate_predictions,
                average= 'macro'
            ),
            'f1_weighted': metrics.f1_score(
                y_true= self.predicted_labels,
                y_pred= self.surrogate_predictions,
                average= "weighted"
            ),
            'recall_weighted': metrics.recall_score(
                y_true= self.predicted_labels,
                y_pred= self.surrogate_predictions,
                average= 'weighted'
            ),
            'precision_weighted': metrics.precision_score(
                y_true= self.predicted_labels,
                y_pred= self.surrogate_predictions,
                average= 'weighted'
            ),
            'balanced_accuracy': metrics.balanced_accuracy_score(
                y_true= self.predicted_labels,
                y_pred= self.surrogate_predictions,
            ),
        }
        self.fidelity = {k: round(v, 3) for k, v in self.fidelity.item()}

        # endregion

        # region 3: Write the log
        self.logger.debug(self.predicted_labels[:100])
        self.logger.debug(self.surrogate_predictions[:100])
        self.logger.info(f'Fidelity of the surrogate: {self.fidelity}')
        self.logger.info(metrics.classification_report(
            y_true=self.predicted_labels,
            y_pred=self.surrogate_predictions)
        )
        # endregion

    def surrogate_to_bdd_string(self):
        """Transform surrogate to BDD string using depth first search
        """

        self.logger.info(msg= 'Transform surrogate to BDD...')
        stack = []
        self.bdd = []

        def _tree_recurese(
            node: Any
        )->None:
            """Recurese

            Args:
                node (Any): tree
            """

            if self._model.tree_.feature[node] == TREE_UNDEFINED: # Stop condition
                value = np.argmax(self._model.tree_.value[node][0])
                if value == 1: # ????
                    path = ' & '.join(stack)
                    self.bdd.append(path)
                    self.paths[path] = self._model.tree_.n_node_samples[node]
                return
        
            # region Recursion case
            name = self.feature_names[self._model.tree_.features[node]]
            stack.append(f'~{name}')
            self.logger.debug(stack)

            _tree_recurese(self._model.tree_.children_left[node])

            stack.pop()
            self.logger.debug(stack)

            stack.append(name)
            self.logger.debug(stack)

            _tree_recurese(self._model.tree_.children_right[node])

            stack.pop()
            self.logger.debug(stack)            

            # endregion
        
        # region run main code
        _tree_recurese(0)
        self.bdd = ' | '.join(self.bdd)
        self.logger.info(f'BDD String for class {self.class_id}: {self.bdd}')

        # endregion

    def save_surrogate_image(
        self,
        save_path: Text
        )-> None:
        """Save decision tree surrogates to image.

        Args:
            save_path (Text): Path to save 
        """
        # region 1: Prepare path
        folder_path = os.path.join(
            save_path,
            "surrogate_tree"
        )


        if not os.path.exists(folder_path):
            os.mkdir(path = folder_path)

        fname = os.path.join(
            folder_path,
            f'{self.class_id}_{self.time_label}.png'
        )

        # endregion

        # region 2: Export graph viz
        graph_str = tree.export_graphviz(
            decision_tree= self._model,
            class_names= [self.class_id, f'NOT {self.class_id}'],
            feature_names= self.feature_names,
            filled= True
        )

        (graph, )  = graph_from_dot_data(graph_str)
        self.logger.info(f'Saving {fname} to disk')
        graph.write_png(fname)
        # endregion


