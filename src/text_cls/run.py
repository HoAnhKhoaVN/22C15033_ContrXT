from typing import Text
import yaml
import os
from src.text_cls.constant import EN, VI
from src.text_cls.utils.preprocess.en import EnglishTextPreprocessor
import pandas as pd
import argparse


def cli():
    parser = argparse.ArgumentParser(description="Công cụ dòng lệnh ví dụ")

    # Thêm tham số vị trí (positional argument)
    parser.add_argument("-p", '--path' ,help="Path CSV")

    parser.add_argument(
        "-l",
        '--lang',
        help="Language"
    )
      
    parser.add_argument(
        "-n",
        "--noun_phrase",
        action="store_true",
        help = "Split by noun phrase"
    )


    # Phân tích các tham số đã cung cấp
    args = parser.parse_args()

    return args

def eng_noun_phrase(
    path: Text,
    noun_phrase: bool
    ):
    # region 1. Prepare output
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

    # region 2. INIT Preprocess
    preprocessor = EnglishTextPreprocessor()
    df = pd.read_csv(path)

    print(df.head())
    
    # endregion

    # region 3. PREPROCESS

    preprocessed_df = preprocessor.preprocess_dataframe(df, noun_phrase)
    print(preprocessed_df.head())    

    # endregion

    # region 4. SAVE FILE
    preprocessed_df.to_csv(
        out_path,
        index = False
    )

    # endregion

def vi_noun_phrase(path: Text)-> None:
    """_summary_

    Args:
        path (Text): _description_
    """
    pass

def vi_word(path: Text)-> None:
    """_summary_

    Args:
        path (Text): _description_
    """
    pass

if __name__ == "__main__":
    # # region 1. Load YAML file
    # with open('config.yaml', 'r') as file:
    #     config = yaml.safe_load(file)

    # # endregion

    args = cli()

    if args.lang == EN:
        if args.noun_phrase:
            eng_noun_phrase(args.path, True)
        else:
            eng_noun_phrase(args.path, False)
    elif args.lang == VI:
        if args.noun_phrase:
            vi_noun_phrase(args.path)
        else:
            vi_word(args.path) # After run eng_noun_phrase with CSV file
    else:
        print(f'Language only "{EN}" and "{VI}"')