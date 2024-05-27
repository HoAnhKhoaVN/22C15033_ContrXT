import os
from typing import Text
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from src.text_cls.constant import LABEL

class EDA(object):
    def __init__(
        self,
        path: Text
    )-> None:
        self.path = path
        self.df = pd.read_csv(self.path)
        self.img_fd = self.get_image_fd_path()

    def get_image_fd_path(self):
        """"""
        fd_path = os.path.dirname(self.path)
        img_fd = os.path.join(fd_path, 'eda')

        if not os.path.exists(img_fd):
            os.makedirs(name = img_fd)
        
        return img_fd
    
    def count_label(self)-> None:
        print(f'Max: {self.df[LABEL].value_counts().max()}')
        print(f'Min: {self.df[LABEL].value_counts().min()}')
        print(f'Mean: {self.df[LABEL].value_counts().mean()}')
        print(f'Mode: {self.df[LABEL].value_counts().mode()[0]}')
        print(f'Median: {self.df[LABEL].value_counts().median()}')
        print(f'Label with max count: {self.df[LABEL].value_counts().idxmax()}')
        print(f'Label with min count: {self.df[LABEL].value_counts().idxmin()}')
        print(f'Sort label by sample')
        print(self.df[LABEL].value_counts())

        # region 2. Draw barchar
        plt.rcParams["figure.figsize"] = [20, 10]
        plt.rcParams["figure.autolayout"] = True
        sns.countplot(
            x = LABEL,
            data=self.df,
            hue = LABEL,
            palette="Dark2_r"
        )
        plt.title('Count label')
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.xticks(rotation=90)

        img_path = os.path.join(self.img_fd, 'count_label.png')
        plt.savefig(img_path)
        print(f'Save image at {img_path}')
        # endregion

    def __call__(self):
        """"""
        print('###################')
        print('### COUNT LABEL ###')
        print('###################')

        self.count_label()





if __name__ == "__main__":
    PATH = "test.csv"
    EDA(
        path = PATH
    )()

