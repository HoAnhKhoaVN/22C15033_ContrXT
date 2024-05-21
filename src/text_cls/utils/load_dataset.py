from typing import Text
import pandas as pd
from sklearn.model_selection import train_test_split
from src.text_cls.constant import (
    TWENTY_NEW_GROUP_TRAIN_PATH,
    TWENTY_NEW_GROUP_TEST_PATH
)

class TwentyNewGroupLoader:
    """
    A class to load datasets from a CSV file and split it into training and testing sets.
    
    Attributes:
    -----------
    train : pd.DataFrame
        DataFrame containing the training dataset.
    test : pd.DataFrame
        DataFrame containing the testing dataset.

    Examples:
    ```sh
    
    data_loader = TwentyNewGroupLoader(
        train_path = TWENTY_NEW_GROUP_TRAIN_PATH,
        test_path=  TWENTY_NEW_GROUP_TEST_PATH
    )
    train_data = data_loader.get_train_data()
    print(train_data.head())

    test_data = data_loader.get_test_data()
    print(test_data.head())

    val_data = data_loader.get_val_data()
    print(val_data.head())

    ```
    """
    
    def __init__(
        self, 
        train_path: Text,
        test_path: Text,
        random_state: int = None
    ):
        """
        Initializes the DataLoader with the given CSV file.
        
        Parameters:
        -----------
        file_path : str
            The path to the CSV file containing the dataset.
        test_size : float, optional
            The proportion of the dataset to include in the test split. Default is 0.2 (20%).
        random_state : int, optional
            Controls the shuffling applied to the data before splitting. Pass an int for reproducible output.
        """
        self.train_path = train_path
        self.test_path = test_path
        self.random_state = random_state
        
        # Initialize train and test attributes as None
        self.train = None
        self.test = None
        self.val = None
        
        # Load the data from the CSV file
        self.load_data()
        
    def load_data(self):
        """
        Loads data from the CSV file and splits it into training and testing sets.
        
        This method reads the CSV file into a DataFrame, then splits the DataFrame into 
        training and testing sets based on the provided test_size and random_state.
        """
        try:
            # Read the CSV file into a DataFrame
            train = pd.read_csv(self.train_path)
            
            # Split the data into training and testing sets
            self.train, self.val = train_test_split(
                train,
                test_size = 0.3,
                random_state = self.random_state
            )

            self.test = pd.read_csv(self.train_path)
            
            print("Data loaded successfully.")
        except Exception as e:
            print(f"An error occurred while loading the data: {e}")
    
    def get_train_data(self)->pd.DataFrame:
        """
        Returns the training dataset.
        
        Returns:
        --------
        pd.DataFrame
            The DataFrame containing the training dataset.
        """
        return self.train
    
    def get_test_data(self)->pd.DataFrame:
        """
        Returns the testing dataset.
        
        Returns:
        --------
        pd.DataFrame
            The DataFrame containing the testing dataset.
        """
        return self.test
    
    def get_val_data(self)->pd.DataFrame:
        """
        Returns the val dataset.
        
        Returns:
        --------
        pd.DataFrame
            The DataFrame containing the testing dataset.
        """
        return self.val



if __name__ == "__main__":
    data_loader = TwentyNewGroupLoader(
        train_path = TWENTY_NEW_GROUP_TRAIN_PATH,
        test_path=  TWENTY_NEW_GROUP_TEST_PATH
    )
    train_data = data_loader.get_train_data()
    print(train_data.head())

    test_data = data_loader.get_test_data()
    print(test_data.head())

    val_data = data_loader.get_val_data()
    print(val_data.head())
