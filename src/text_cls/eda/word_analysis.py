# python src/text_cls/eda/word_analysis.py -p src/text_cls/dataset/20newsgroups/orginal/train__split_noun_phrase__remove_outlier_char__remove_word_outlier.csv

# python src/text_cls/eda/word_analysis.py -p src/text_cls/dataset/20newsgroups/orginal/train__split_noun_phrase__remove_outlier_char.csv -r

# python src/text_cls/eda/word_analysis.py -p src/text_cls/dataset/20newsgroups/orginal/train__split_noun_phrase__remove_outlier_char.csv -i box_plot_train__split_noun_phrase__remove_outlier_char.png

# python src/text_cls/eda/word_analysis.py -p src/text_cls/dataset/20newsgroups/orginal/train__word.csv -i box_plot_train__word.png

# python src/text_cls/eda/word_analysis.py -p src/text_cls/dataset/20newsgroups/orginal/train__split_noun_phrase.csv -i box_plot_train__split_noun_phrase.


import argparse
import os
from typing import Text
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.text_cls.constant import TEXT

def draw_box_plot(
    df: pd.DataFrame,
    img_path: Text,
)->None:
    data = df[TEXT].str.split().map(lambda x: len(x))
    print(f'Length data: {len(data)}')
    print(f'Mean: {data.mean()}')
    print(f'Min: {data.min()}')
    print(f'Max: {data.max()}')
    print(f'Std: {data.std()}')
    sns.boxplot(data)
    plt.title(f'Box Plot với độ dài của từng từ trước khi lọc')
    plt.ylabel("Độ dài từ")
    plt.savefig(img_path)

def find_upper(df: pd.DataFrame)-> int:
    """_summary_

    Args:
        df (pd.DataFrame): _description_

    Returns:
        int: _description_
    """
    df.dropna(axis = 0 ,how = 'any', inplace = True, ignore_index= False)
    df.reset_index(inplace= True, drop=True,)
    data = df[TEXT].str.split().map(lambda x: len(x))
    # IQR    
    Q1 = np.percentile(data, 25, method='midpoint')
    Q3 = np.percentile(data, 75, method='midpoint')
    IQR = Q3 - Q1
    upper = int(Q3 + 1.5*IQR)
    print(f'Upper: {upper}')

    return upper

def remove_row_outlier(
    df: pd.DataFrame,
    upper : int
)-> pd.DataFrame:
    df.dropna(axis = 0 ,how = 'any', inplace = True, ignore_index= False)
    df.reset_index(drop= True, inplace= True)
    data = df[TEXT].str.split().map(lambda x: len(x))
    outlier_indices = np.where(data > upper)[0]
    df_no_outliers = df.drop(labels=outlier_indices)
    return df_no_outliers

def removal_word_outlier(
    df: pd.DataFrame,
    upper: int
)-> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    # Create arrays of Boolean values indicating the outlier rows
    df[TEXT] = df[TEXT].str.split().map(lambda x: ' '.join(x[:upper]) if len(x)> upper else ' '.join(x))
    return df
    
def cli():
    parser = argparse.ArgumentParser(description="CLI: ")

    # Thêm tham số vị trí (positional argument)
    parser.add_argument("-p", '--path' ,help="Path CSV")

    parser.add_argument("-i", '--img_path' ,help="Path image to save")

    parser.add_argument(
        "-r",
        "--remove",
        action="store_true",
        help = "Is remove outliers row?"
    )

    parser.add_argument(
        "-u",
        "--upper",
        help = "Max length"
    )
    # Phân tích các tham số đã cung cấp
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = cli()
    df = pd.read_csv(args.path)

    root = os.path.dirname(args.path)
    fn, ext = os.path.splitext(os.path.basename(args.path))
    img_path = os.path.join(
        root,
        f'{fn}__remove_word_outlier.png'
    )
    if args.remove:
        if not args.upper:
            upper = find_upper(df)
            new_df = remove_row_outlier(df, upper)

            
            out_csv = os.path.join(root, f'{fn}__remove_word_outlier.{ext}')
            new_df.to_csv(
                path_or_buf= out_csv,
                index = False
            )
            draw_box_plot(new_df, img_path)
        else:
            upper = int(args.upper)
            new_df = removal_word_outlier(df, upper)

            
            out_csv = os.path.join(root, f'{fn}__remove_word_outlier.{ext}')
            new_df.to_csv(
                path_or_buf= out_csv,
                index = False
            )
            draw_box_plot(new_df, img_path)
    else:
        draw_box_plot(df, img_path)    
