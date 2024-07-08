# python src/text_cls/eda/character_analysis.py -p src/text_cls/dataset/20newsgroups/orginal/train__split_noun_phrase.csv -i box_plot_char_org.png

# python src/text_cls/eda/character_analysis.py -p src/text_cls/dataset/20newsgroups/orginal/train__split_noun_phrase.csv  -i box_plot_char_remove_outlier.png -r -o src/text_cls/dataset/20newsgroups/orginal/train__split_noun_phrase__remove_outlier_char.csv


# python src/text_cls/eda/character_analysis.py -p src/text_cls/dataset/20newsgroups/orginal/train__split_noun_phrase.csv -i box_plot_char_org.png


# python src/text_cls/eda/character_analysis.py -p src/text_cls/dataset/20newsgroups/orginal/train__split_noun_phrase.csv -i box_plot_char_org.png


# python src/text_cls/eda/character_analysis.py -p src/text_cls/dataset/20newsgroups/orginal/train__split_noun_phrase.csv -i box_plot_char_org.png


# python src/text_cls/eda/character_analysis.py -p src/text_cls/dataset/20newsgroups/orginal/train__split_noun_phrase.csv -i box_plot_char_org.png

import argparse
from typing import Text
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.text_cls.constant import TEXT

def draw_box_plot(
    df: pd.DataFrame,
    img_path : Text,
)->None:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
    """
    data = df[TEXT].str.len()
    print(f'Length data: {len(data)}')
    print(f'Mean: {data.mean()}')
    print(f'Min: {data.min()}')
    print(f'Max: {data.max()}')
    print(f'Std: {data.std()}')
    sns.boxplot(data)
    plt.title(f'Box Plot với độ dài của từng mẫu dữ liệu trước khi lọc lại')
    plt.ylabel("Độ dài ký tự")
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
    data = df[TEXT].str.len()
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
    data = df[TEXT].str.len()
    outlier_indices = np.where(data > upper)[0]
    df_no_outliers = df.drop(labels=outlier_indices)
    return df_no_outliers
    
def cli():
    parser = argparse.ArgumentParser(description="CLI: ")

    # Thêm tham số vị trí (positional argument)
    parser.add_argument("-p", '--path' ,help="Path CSV")

    parser.add_argument("-o", '--out' ,help="Path output CSV")

    parser.add_argument("-i", '--img_path' ,help="Path image to save")

    parser.add_argument(
        "-r",
        "--remove",
        action="store_true",
        help = "Is remove outliers row?"
    )
    # Phân tích các tham số đã cung cấp
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = cli()
    df = pd.read_csv(args.path)
    if args.remove:
        upper = find_upper(df)
        new_df = remove_row_outlier(df, upper)
        new_df.to_csv(
            path_or_buf= args.out,
            index = False
        )
        draw_box_plot(new_df, args.img_path)
    else:
        draw_box_plot(df, args.img_path)    

