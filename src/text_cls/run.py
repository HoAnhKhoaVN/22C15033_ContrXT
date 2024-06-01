from typing import Text
import yaml
import os
from src.text_cls.constant import EN, VI
from src.text_cls.utils.preprocess.en import EnglishTextPreprocessor
import pandas as pd
import argparse

from src.text_cls.utils.preprocess.vi import VietnameseTextPreprocessor


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

    parser.add_argument(
        "-r",
        "--remove_stop_words",
        action="store_true",
        help = "Remove stop word"
    )

    parser.add_argument(
        "-t",
        "--train",
        action="store_true",
        help = "Is train?"
    )

    # Phân tích các tham số đã cung cấp
    args = parser.parse_args()

    return args

def eng_noun_phrase(
    path: Text,
    noun_phrase: bool,
    is_train: bool
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

    preprocessed_df = preprocessor.preprocess_dataframe(
        df = df,
        noun_phrase= noun_phrase,
        is_train= is_train
    )
    print(preprocessed_df.head())    

    # endregion

    # region 4. SAVE FILE
    preprocessed_df.to_csv(
        out_path,
        index = False
    )

    # endregion

def vi_noun_phrase(
    path: Text,
    noun_phrase: bool = True,
    remove_stopwords : bool = True
    )-> None:
    """_summary_

    Args:
        path (Text): _description_
    """
    # region 1. Prepare output
    dirname = os.path.dirname(path)
    fd_name = os.path.splitext(os.path.basename(path))[0]

    if args.noun_phrase:
        fd_name += "__noun_phrase"
    else:
        fd_name += "__word"

    if args.remove_stop_words:
        fd_name += '__remove_stop_words'
    else:
        fd_name += '__exist_stop_words'
    
    fd_name += '.csv'
    
    out_path = os.path.join(
        dirname,
        fd_name
    )
    print(f'out_path: {out_path}')
    # endregion

    # region 2. INIT Preprocess
    preprocessor = VietnameseTextPreprocessor()
    df = pd.read_csv(path)

    print(df.head())
    
    # endregion

    # region 3. PREPROCESS

    preprocessed_df = preprocessor.preprocess_dataframe(df, noun_phrase, remove_stopwords)
    print(preprocessed_df.head())    

    # endregion

    # region 4. SAVE FILE
    preprocessed_df.to_csv(
        out_path,
        index = False
    )

    # endregion


if __name__ == "__main__":
    # # region 1. Load YAML file
    # with open('config.yaml', 'r') as file:
    #     config = yaml.safe_load(file)

    # # endregion

    args = cli()

    if args.lang == EN:
        if args.noun_phrase:
            if args.train:
                eng_noun_phrase(
                    path = args.path,
                    noun_phrase= True,
                    is_train= True
                )
            else:
                eng_noun_phrase(
                    path = args.path,
                    noun_phrase= True,
                    is_train= False
                )
                
        else:
            if args.train:
                eng_noun_phrase(
                    path = args.path,
                    noun_phrase= False,
                    is_train= True
                )
            else:
                eng_noun_phrase(
                    path = args.path,
                    noun_phrase= False,
                    is_train= False
                )
    elif args.lang == VI:
        if args.noun_phrase:
            if args.remove_stop_words:
                vi_noun_phrase(args.path, True, True)
            else:
                vi_noun_phrase(args.path, True, False)
        else:
            if args.remove_stop_words:
                vi_noun_phrase(args.path, False, True)
            else:
                vi_noun_phrase(args.path, False, False)
    else:
        print(f'Language only "{EN}" and "{VI}"')