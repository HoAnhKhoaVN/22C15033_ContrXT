import ast
import re
from typing import Dict, List, Text
import numpy as np
import pandas as pd
from src.contrxt.data.data_manager import DataManager
from sklearn.feature_extraction.text import CountVectorizer

class TextDataManager(DataManager):
    """Handle for pandas dataframe

    Args:
        DataManager (_type_): _description_
    """
    def __init__(
        self,
        X: pd.Series,
        Y_predicted : List[Text],
        time_label: Text,
        random_state : int  = 42
    )-> None:
        # region 1: Check variable name in pandas series
        X = list(map(lambda x: self.check_var_names(x), X))
        X = pd.Series(X)

        # endregion

        super().__init__(X, Y_predicted, time_label, random_state)

    def generate_data_predictions(
        self, 
        percent_dataset: int = 1
    )-> None:
        """Generates the predictions for both dataset.

        Args:
            percent_dataset (int, optional): Describes the percentage of the dataset to use.
            Defaults to 1.
        """
        self.logger.info(f'Sampling dataset with percent: {percent_dataset} and saving labels...')
        self.logger.info(f'Total dataset n is {self.df.shape[0]}.')

        data = self.df.sample(
            frac= percent_dataset,
            replace= False,
            random_state= self.random_state
        )

        self.logger.info(f'N. Samples {self.time_label}: {data.shape[0]}')

        onehot_vectorizer = CountVectorizer(binary= True, lowercase= False)

        onehot_vectorizer.fit(data['X'])

        self.feature_names = onehot_vectorizer.get_feature_names_out()

        for i, class_id in enumerate(self.classes): 
            df_data_class :pd.DataFrame = data.copy()
            class_id = str(class_id)
            n_positive_class = sum(data['Y_predicted'] == class_id)

            # region Get sample
            try:
                df_data_class= pd.concat(
                    [
                        df_data_class[df_data_class['Y_predicted']==class_id],
                        df_data_class[df_data_class['Y_predicted']!=class_id].sample(
                            n = n_positive_class,
                            replace= False,
                            random_state= self.random_state
                        )
                    ]
                )
            except ValueError:
                pass
            
            # region Convert to vector for each document 
            self.surrogate_train_data[class_id] = onehot_vectorizer.transform(df_data_class['X'])
            
            if df_data_class['Y_predicted'].dtype == np.dtype(int) or df_data_class['Y_predicted'].dtype == np.dtype(np.int64):
                # Type : int
                self.Y_predicted_binarized[class_id]  = np.array(
                    [1 if int(x) == i else 0 for x in df_data_class['Y_predicted']]
                )
            else: 
                # Type: String
                self.Y_predicted_binarized[class_id]  = np.array(
                    [1 if x == class_id else 0 for x in df_data_class['Y_predicted']]
                )

            # endregion

        self.logger.info(f'Finished predicting {self.time_label}')

    def count_rule_occurrence(
        self,
        rule: Text
    )-> int:
        """Count the number of occurrences in the corpus for a specific rule.

        Args:
            rule (Text): Rule apply in Binary Decision Diagram

        Returns:
            int: Count of occurrences in the corpus.
        """
        count = 0
        rule_dict :Dict = ast.literal_eval(re.sub(r'(\w+):', r'"\1'), rule)
        contains = [k for k, v in rule_dict.items() if v]
        avoids = [k for k, v in rule_dict.items() if not v]
        for sent in self.X:
            if all(x in sent.split() for x in contains) and\
               all(x not in sent.split() for x in avoids):
                count +=1
        return count
if __name__ == '__main__':
    pass