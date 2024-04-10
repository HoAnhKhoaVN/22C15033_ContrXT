from typing import List, Text, Any
import pandas as pd
from src.contrxt.utils.logger import build_logger
import logging


class GeneticSurrogate(object):
    """Base class for surrogate
    """
    def __init__(
        self,
        X: pd.Series,
        predicted_labels: List[Text],
        time_label: Text,
        class_id: Text,
        feature_names: Any
    ) -> None:
        """Initalize for genetic surrogate

        Args:
            X (pd.Series): Dataset
            predicted_labels (List[Text]): List of predicted text.
            time_label (Text): _description_
            class_id (Text): _description_
            feature_names (Any): _description_
        """
        self.logger = build_logger(
            log_level= logging.INFO,
            log_name= __name__,
            out_file= 'logs/surrogates.log'
        )

        self.X = X
        self.predicted_labels= predicted_labels
        self.time_label = time_label
        self.class_id = class_id
        self.feature_names = feature_names
    
    def hyperparameters_selection(self):
        """Selection hyperparameters for surrogate model.
        """
        pass

    def fit(self):
        """Train model surrogate with dataset for `class_id`.
        """
        pass

    def surrogate_to_bdd_string(self):
        """Convert surrogate to BDD
        """
        pass

    def save_surrogate_image(self):
        """Save surrogate to image to 
        """
        pass