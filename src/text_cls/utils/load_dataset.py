import os
from typing import Text, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import chardet
from src.text_cls.constant import (
    ID,
    LABEL,
    TEXT,
    TWENTY_NEW_GROUP_TRAIN_PATH,
    TWENTY_NEW_GROUP_TEST_PATH,
    VNTC_TRAIN_PATH,
    VNTC_TEST_PATH,
    VNTC_TRAIN_CSV_PATH,
    VNTC_TEST_CSV_PATH
)

class MyDataLoader:
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
        train_path: Text = TWENTY_NEW_GROUP_TRAIN_PATH,
        test_path: Text = TWENTY_NEW_GROUP_TEST_PATH,
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
        
    def train_val_split(
        self,
        data: np.ndarray,
        target: np.ndarray,
        test_size: float = 0.3,
        random_state: int = 2103
    )-> Tuple:
        """_summary_

        Args:
            data (np.ndarray): _description_
            target (np.ndarray): _description_
            test_sizeLfloat (float, optional): _description_. Defaults to 0.3.

        Ref: https://stackoverflow.com/questions/35472712/how-to-split-data-on-balanced-training-set-and-test-set-on-sklearn

        Returns:
            Tuple: _description_
        """
        # region set the seed to reproducible
        # REF: https://stackoverflow.com/questions/22994423/difference-between-np-random-seed-and-np-random-randomstate
        rng = np.random.RandomState(random_state)

        # endregion
        
        classes = np.unique(target)
        # can give test_size as fraction of input data size of number of samples
        if test_size < 1:
            n_test = np.round(len(target)*test_size)
        else:
            n_test = test_size
        
        n_train = max(0,len(target)-n_test)
        n_train_per_class = max(1,int(np.floor(n_train/len(classes))))
        n_test_per_class = max(1,int(np.floor(n_test/len(classes))))

        ixs = []
        for cl in classes:
            if (n_train_per_class+n_test_per_class) > np.sum(target==cl):
                # if data has too few samples for this class, do upsampling
                # split the data to training and testing before sampling so data points won't be
                #  shared among training and test data
                splitix = int(np.ceil(n_train_per_class/(n_train_per_class+n_test_per_class)*np.sum(target==cl)))
                ixs.append(np.r_[rng.choice(np.nonzero(target==cl)[0][:splitix], n_train_per_class),
                    rng.choice(np.nonzero(target==cl)[0][splitix:], n_test_per_class)])
            else:
                ixs.append(rng.choice(np.nonzero(target==cl)[0], n_train_per_class+n_test_per_class,
                    replace=False))

        # take same num of samples from all classes
        ix_train = np.concatenate([x[:n_train_per_class] for x in ixs])
        ix_test = np.concatenate([x[n_train_per_class:(n_train_per_class+n_test_per_class)] for x in ixs])

        X_train = data[ix_train]
        X_test = data[ix_test]
        y_train = target[ix_train]
        y_test = target[ix_test]

        return X_train, X_test, y_train, y_test, ix_train, ix_test

    def load_data(self):
        """
        Loads data from the CSV file and splits it into training and testing sets.
        
        This method reads the CSV file into a DataFrame, then splits the DataFrame into 
        training and testing sets based on the provided test_size and random_state.
        """
        try:
            # Read the CSV file into a DataFrame
            train = pd.read_csv(self.train_path)

            X = np.array(train[TEXT].to_list())
            y = np.array(train[LABEL].to_list())

            X_train, X_test, y_train, y_test, ix_train, ix_test = self.train_val_split(
                data= X,
                target= y,
                test_size= 0.3
            )

            self.train = pd.DataFrame(
                data = {
                    ID : ix_train,
                    TEXT : X_train,
                    LABEL: y_train
                }
            )

            self.val = pd.DataFrame(
                data = {
                    ID : ix_test,
                    TEXT : X_test,
                    LABEL: y_test
                }
            )

            self.val.drop_duplicates(
                subset=[TEXT],
                inplace= True,
                keep= 'first',
                ignore_index= False
            )
           
            self.test = pd.read_csv(self.test_path)
            
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

class VNTCLoader:
    """
    A class to load text files from specified train and test directories into Pandas DataFrames,
    automatically handling different file encodings.

    Example:
    ```sh
    loader = VNTCLoader()
    train_df, test_df = loader.load_data()
    print(train_df.head())
    print(test_df.head())
    ```
    """
    def __init__(
        self,
        train_dir: Text = VNTC_TRAIN_PATH,
        test_dir: Text = VNTC_TEST_PATH
    )-> None:
        self.train_dir = train_dir
        self.test_dir = test_dir
    
    def detect_encoding(
        self,
        file_path: Text
    )-> Text:
        """
        Detects the character encoding of a given file.
        
        Args:
        file_path (str): The path to the file.
        
        Returns:
        str: The detected encoding of the file.
        """
        with open(file_path, 'rb') as file:
            raw_data = file.read(50000)  # Read first 50,000 bytes to guess the encoding
        result = chardet.detect(raw_data)
        return result['encoding']

    def read_files_in_directory(self, directory_path):
        """
        Recursively reads text files from given directory path, detecting encoding automatically,
        and extracts content and labels.
        
        Args:
        directory_path (str): The path to the directory containing text files organized in subdirectories.
        
        Returns:
        list of tuples: A list where each tuple contains the content of the text file and its label.
        """
        data = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    encoding = self.detect_encoding(file_path)  # Detect file encoding
                    with open(file_path, 'r', encoding=encoding) as file:  # Use the detected encoding
                        content = file.read()
                    label = os.path.basename(root)
                    data.append((content, label))
        return data

    def create_dataframe(self, data):
        """
        Converts a list of tuples into a pandas DataFrame.
        
        Args:
        data (list of tuples): Data to be converted, where each tuple contains text content and label.
        
        Returns:
        pd.DataFrame: A DataFrame with columns 'text' and 'label'.
        """
        return pd.DataFrame(data, columns=['text', 'label'])

    def load_data(self, ):
        """
        Loads text data from train and test directories into two separate DataFrames.
        
        Args:
        train_dir (str): Path to the training data directory.
        test_dir (str): Path to the testing data directory.
        
        Returns:
        tuple of pd.DataFrame: A tuple containing two DataFrames, one for training and one for testing.
        """
        train_data = self.read_files_in_directory(self.train_dir)
        test_data = self.read_files_in_directory(self.test_dir)
        train_df = self.create_dataframe(train_data)
        test_df = self.create_dataframe(test_data)
        return train_df, test_df

if __name__ == "__main__":
    
    # region TwentyNewGroupLoader
    # data_loader = TwentyNewGroupLoader(
    #     train_path = TWENTY_NEW_GROUP_TRAIN_PATH,
    #     test_path=  TWENTY_NEW_GROUP_TEST_PATH
    # )
    # train_data = data_loader.get_train_data()
    # print(train_data.head())

    # test_data = data_loader.get_test_data()
    # print(test_data.head())

    # val_data = data_loader.get_val_data()
    # print(val_data.head())

    # endregion

    # # region VNTC
    # loader = VNTCLoader()
    # train_df, test_df = loader.load_data()
    # print(train_df.head())
    # print(test_df.head())

    # root_path = os.path.dirname(VNTC_TRAIN_PATH)
    # train_df.to_csv(os.path.join(root_path, 'train.csv'))
    # test_df.to_csv(os.path.join(root_path, 'test.csv'))
    # # endregion

    # region Load VNTC CSV
    data_loader = MyDataLoader(
        train_path = "src/text_cls/dataset/20newsgroups/noun_phrase/train__split_noun_phrase.csv",
        test_path= "src/text_cls/dataset/20newsgroups/noun_phrase/test__split_noun_phrase.csv"
    )
    train_data = data_loader.get_train_data()
    train_data.to_csv(
        "src/text_cls/dataset/20newsgroups/full_data/noun_phrase/balance/train.csv",
        index = False
    )

    test_data = data_loader.get_test_data()
    test_data.to_csv(
        "src/text_cls/dataset/20newsgroups/full_data/noun_phrase/balance/test.csv",
        index = False
    )

    val_data = data_loader.get_val_data()
    val_data.to_csv(
        "src/text_cls/dataset/20newsgroups/full_data/noun_phrase/balance/val.csv",
        index = False
    )

    # endregion