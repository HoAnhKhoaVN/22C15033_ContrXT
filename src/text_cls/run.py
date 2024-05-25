import yaml
import os
from src.text_cls.utils.preprocess.en import EnglishTextPreprocessor
import pandas as pd
import argparse


def cli():
    parser = argparse.ArgumentParser(description="Công cụ dòng lệnh ví dụ")

    # Thêm tham số vị trí (positional argument)
    parser.add_argument("-p", '--path' ,help="Path CSV")

    # Thêm tham số tùy chọn (optional argument)
    parser.add_argument(
        "-n",
        "--noun_phrase",
        action="store_true",
        help = "Split by noun phrase"
    )

    # Phân tích các tham số đã cung cấp
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    # # region 1. Load YAML file
    # with open('config.yaml', 'r') as file:
    #     config = yaml.safe_load(file)

    # # endregion

    args = cli()

    # region Prepare output
    path = args.path
    dirname = os.path.dirname(path)
    fd_name = os.path.splitext(os.path.basename(path))[0]
    
    if args.noun_phrase:
        fd_name += "__split_noun_phrase.csv"
    else:
        fd_name += "__word.csv"
    
    out_path = os.path.join(
        dirname,
        fd_name
    )
    print(f'out_path: {out_path}')
    # endregion
    
    # region preprocess
    # region INIT
    preprocessor = EnglishTextPreprocessor()
    df = pd.read_csv(path)

    print(df.head(5))
    
    # endregion

    # region PREPROCESS
    if args.noun_phrase:
        noun_phrase = True
    else:
        noun_phrase = False

    preprocessed_df = preprocessor.preprocess_dataframe(
        df,
        noun_phrase = noun_phrase,
    )

    print(preprocessed_df.head(5))


    
    # endregion

    # endregion


    # region SAVE FILE
    preprocessed_df.to_csv(
        out_path,
        index = False
    )

    # endregion