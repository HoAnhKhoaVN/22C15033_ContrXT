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
    # region preprocess
    # region INIT
    preprocessor = EnglishTextPreprocessor()
    path = args.path
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
        'text',
        noun_phrase = noun_phrase,
    )

    print(preprocessed_df.head(5))


    
    # endregion

    # endregion


    # region SAVE FILE
    dirname = os.path.dirname(path)
    basename = os.path.basename(path)

    fd_name = os.path.basename(dirname)
    
    if args.noun_phrase:
        fd_name += "__split_noun_phrase"
    else:
        fd_name += "__word"
    
    out_fd = os.path.join(
        dirname,
        fd_name
    )

    os.makedirs(out_fd)
    
    out_path = os.path.join(
        out_fd,
        basename
    )
    
    preprocessed_df.to_csv(
        out_path,
        index = False
    )

    # endregion