# python src/text_cls/eda/word_analysis.py -p preprocess.csv -i box_plot_word_org.png
import argparse
from typing import Text
from matplotlib import pyplot as plt
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

def cli():
    parser = argparse.ArgumentParser(description="CLI: ")

    # Thêm tham số vị trí (positional argument)
    parser.add_argument("-p", '--path' ,help="Path CSV")

    parser.add_argument("-i", '--img_path' ,help="Path image to save")
    # Phân tích các tham số đã cung cấp
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = cli()
    df = pd.read_csv(args.path)
    df.dropna(axis = 0 ,how = 'any', inplace = True, ignore_index= False)
    draw_box_plot(df, args.img_path)