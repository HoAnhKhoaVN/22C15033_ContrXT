# python src/text_cls/demo/preprocess/custom_stop_word.py
import os
import nltk
from typing import Text
import pandas as pd
from nltk.corpus import stopwords
from collections import defaultdict
import pickle

nltk.download('stopwords')

from src.text_cls.constant import TEXT

CSV_PATH = "src/text_cls/dataset/20newsgroups/word/train__word.csv"
PICKLE_FD = "src/text_cls/asset/stop_word/en"

def read_data(csv: Text)-> pd.DataFrame:
    """Read CSV to DataFrame
    
    Returns:
        pd.DataFrame:Output
    """
    df = pd.read_csv(csv)
    df.dropna(axis = 0 ,how = 'any', inplace = True, ignore_index= False)

    return df


if __name__ == "__main__":
    # region 1: Read csv
    df = read_data(csv = CSV_PATH)
    # endregion

    # region 2: Load stop word in English
    stop=set(stopwords.words('english'))

    # endregion

    # region 3: Get stop word in corpus
    corpus=[]
    new= df[TEXT].str.split()
    new=new.values.tolist()
    corpus=[word for i in new for word in i]

    dic=defaultdict(int)
    for word in corpus:
        if word in stop:
            dic[word]+=1

    print(dic)

    # endregion


    # region save piclke file
    pickle_path = os.path.join(
        PICKLE_FD,
        'train_word.plk'
    )
    with open(pickle_path, 'wb') as f:
        pickle.dump(
            obj = dic,
            file = f
        )
    # endregion