# python src/text_cls/eda/tri_gram_analysis.py -p src/text_cls/dataset/20newsgroups/noun_phrase/train__split_noun_phrase.csv -i tri_gram_train__split_noun_phrase.png

import argparse
import os
from typing import Any, Text
from matplotlib import pyplot as plt
import pandas as pd
from src.text_cls.constant import ID, LABEL, TEXT
from collections import Counter
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

def get_top_ngram(
    corpus,
    n : int = None,
    topk : int = 10
    ):
    """_summary_

    Args:
        corpus (_type_): _description_
        n (int, optional): _description_. Defaults to None.
        topk (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
    """
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx])
                  for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[: topk]

def tri_gram_analysis(
    df: pd.DataFrame,
    img_path: Text,
)-> None:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
    """
    # region Get top n-gram
    top_n_trigrams=get_top_ngram(
        corpus= df['text'],
        n = 3,
        topk= 10
    )

    x,y=map(list,zip(*top_n_trigrams))
    sns.barplot(
        x=y,
        y=x,
    )
    plt.title('Top Tri-gram')
    plt.xlabel('Count')
    plt.ylabel('Label')

    plt.savefig(img_path)
    print(f'Save image at {img_path}')
    # endregion

def cli():
    parser = argparse.ArgumentParser(description="Công cụ dòng lệnh ví dụ")

    # Thêm tham số vị trí (positional argument)
    parser.add_argument("-p", '--path' ,help="Path CSV")

    parser.add_argument("-i", '--img_path' ,help="Path image to save")
    # Phân tích các tham số đã cung cấp
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    plt.rcParams["figure.figsize"] = [20, 10]
    plt.rcParams["figure.autolayout"] = True

    args = cli()

    # CSV_PATH = "src/text_cls/dataset/20newsgroups/noun_phrase/train__split_noun_phrase.csv"

    df = pd.read_csv(args.path)
    df.dropna(axis = 0 ,how = 'any', inplace = True, ignore_index= False)

    tri_gram_analysis(
        df,
        args.img_path
    )