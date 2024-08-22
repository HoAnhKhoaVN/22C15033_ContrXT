# python src/text_cls/utils/preprocess/compare_version.py
from typing import Dict, List, Text
import pandas as pd
import os
from src.text_cls.constant import ID, LABEL, TEXT

ROOT = "src/text_cls/dataset/20newsgroups/orginal"

def read_csv(
        csv_path: Text,
        idx2id: Dict
    )-> List:
    """"""
    df = pd.read_csv(csv_path)
    df.dropna(subset=[TEXT, LABEL], inplace= True)
    df_org.reset_index(drop= True, inplace = True)
    cur_lst_idx = list(map(int,list(df[ID])))

    df_tmp = pd.DataFrame(
        data = {
            ID : list(idx2id.keys()),
            TEXT : [None] * len(idx2id)
        }
    )

    cur_id2idx= dict(zip(cur_lst_idx, list(df.index)))

    # print(f'cur_id2idx: {cur_id2idx}')
    print(f'cur_lst_idx: {cur_lst_idx[:3]}')
    print(f'idx2id value : {list(idx2id.values())[:3]}')

    for _id in list(idx2id.values()):
        if _id in cur_lst_idx:
            _idx = int(id2idx[_id])
            cur_idx = int(cur_id2idx[_id])
            # print(f'_idx: {_idx}')
            # print(f'cur_idx: {cur_idx}')
            df_tmp.loc[_idx, TEXT] =  df[TEXT][cur_idx]
    return list(df_tmp[TEXT])

if __name__ == "__main__":
    # region 1: Read original csv
    ORG_CSV = "src/text_cls/dataset/20newsgroups/orginal/train.csv"
    df_org = pd.read_csv(ORG_CSV)
    df_org.dropna(subset=[TEXT, LABEL], inplace= True)
    df_org.reset_index(drop= True, inplace = True)
    org_index = list(df_org[ID])
    
    idx2id = dict(zip(list(df_org.index), org_index))
    id2idx = dict(zip(org_index, list(df_org.index)))
    # endregion

    INPUT_CSV = [
        "src/text_cls/dataset/20newsgroups/orginal/train__preprocess_v4_3.csv",
        "src/text_cls/dataset/20newsgroups/orginal/train__preprocess__no_remove_sw__v4_4.csv",
        "src/text_cls/dataset/20newsgroups/orginal/train__preprocess__no_remove_sw__apostrophe__v4_5.csv",
        'src/text_cls/dataset/20newsgroups/orginal/train__preprocess__remove_sw__apostrophe__v4_6.csv',
        'src/text_cls/dataset/20newsgroups/orginal/train__preprocess__no_remove_sw__noun_phrase__v4_7.csv',
        'src/text_cls/dataset/20newsgroups/orginal/train__preprocess__remove_sw__noun_phrase__v4_8.csv',
        'src/text_cls/dataset/20newsgroups/orginal/train__preprocess__remove_sw__noun_phrase__v4_9.csv',
    ]

    # region Get text
    lst_texts = []
    for csv_path in INPUT_CSV:
        lst_text = read_csv(csv_path, idx2id)
        print(f'Len {csv_path}: {len(lst_text)}')
        lst_texts.append(lst_text)

    # endregion

    # region Get col name
    col_names = [os.path.splitext(os.path.basename(path))[0] for path in INPUT_CSV]

    # endregion

    # region Get the final csv
    data = {ID: org_index}

    for col_name, lst_text in zip(col_names, lst_texts):
        data[col_name] = lst_text

    df = pd.DataFrame(
        data= data
    )

    # endregion

    # region Export to csv
    OUT_CSV = "train__merge_csv.csv"
    OUT_PATH = os.path.join(ROOT, OUT_CSV)
    df.to_csv(OUT_PATH, index= False)

    # endregion
