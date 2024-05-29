# python src/text_cls/eda/character_analysis.py
from typing import Text
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.text_cls.constant import TEXT

def removal_box_plot(
    df: pd.DataFrame,
    col : Text = TEXT
    ):
    df.reset_index(inplace= True, drop=True,)
    # IQR
    data = df[TEXT].str.len()
    print(f'Length data: {len(data)}')
    print(f'Mean: {data.mean()}')
    print(f'Min: {data.min()}')
    print(f'Max: {data.max()}')
    print(f'Std: {data.std()}')
    # sns.boxplot(data)
    # plt.title(f'Original Box Plot with character length')
    # plt.savefig('boxplot_char.png')
    
    Q1 = np.percentile(data, 25, method='midpoint')
    Q3 = np.percentile(data, 75, method='midpoint')
    IQR = Q3 - Q1
    print(f'IQR: {IQR}')

    lower = Q1 - 1.5*IQR
    print(f'Lower: {lower}')

    upper_value = Q3 + 1.5*IQR
    min_value = data.min()
    upper = max(upper_value, min_value)
    print(f'Upper: {upper}')

    # Create arrays of Boolean values indicating the outlier rows
    upper_array = np.where(data >= upper)[0]
    print(f'Length upper_array: {upper_array}')

    lower_array = np.where(data <= lower)[0]
    print(f'Length lower_array: {lower_array}')
    
    # Removing the outliers
    df.drop(index=upper_array, inplace= True)
    df.drop(index=lower_array, inplace= True)
    df.reset_index(inplace= True, drop=True)

    final_data = df[TEXT].str.len()
    print(len(final_data))
    print(f'Mean: {final_data.mean()}')
    print(f'Min: {final_data.min()}')
    print(f'Max: {final_data.max()}')
    print(f'Std: {final_data.std()}')
    
    sns.boxplot(final_data)
    plt.title(f'Box Plot after remove outlier')
    plt.savefig('boxplot_char__remove_outlier.png')
    return df

if __name__ == "__main__":
    CSV_PATH = "preprocess.csv"

    df = pd.read_csv(CSV_PATH)
    df.dropna(axis = 0 ,how = 'any', inplace = True, ignore_index= False)

    df = removal_box_plot(df)
    df.to_csv('df_remove_outlier_char.csv', index = False)