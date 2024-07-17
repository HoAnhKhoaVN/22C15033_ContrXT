# python src/text_cls/dataset/20newsgroups/filter_dataset.py

import re
import pandas as pd
from tqdm import tqdm

from src.text_cls.constant import TEXT

def preprocess(text: str, subject=True):
    # -> List[List[str]]:
    """
    Preprocess a text for uni/bigram extraction.
    Returns a lemmatized and cleaned version of the text.
    """

    # Run regex
    removeEmail = re.compile(r'\S*@\S*\s?', re.IGNORECASE)
    text = removeEmail.sub(" ", text)

    remove_phone = re.compile(r'[\+]?[0-9]?[-\s\.]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}', re.IGNORECASE)
    text = remove_phone.sub(" ", text)

    remove_html = re.compile(r'(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])', re.IGNORECASE)
    text = remove_html.sub(" ", text)

    return text
 
    # df_train["text"] = df_train["text"].apply(lambda t: t+' ')
    # df_train_new = pd.DataFrame(
    #     data={
    #         'id': df_train['id'],
    #         'text': df_train['text'],
    #         'label': df_train['label'],
    #     }
    # )
if __name__ == "__main__":
    TRAIN_PATH = "D:/Desktop/xAI/exp/22C15033_ContrXT/src/text_cls/dataset/20newsgroups/orginal/train__preprocess_v4_2.csv"
    df_train = pd.read_csv(TRAIN_PATH)
    # is_train = True
    # df_train.dropna(axis = 0 ,how = 'any', inplace = True, ignore_index= False)
    # if is_train:
    #     df_train.drop(index = [144,1492, 1506, 1541, 2931, 3198, 4495, 4515, 4772, 8213, 8665, 9080, 10275], inplace = True)
    #     df_train.reset_index(drop = True, inplace = True)
    # df_train.dropna(axis = 0 ,how = 'any', inplace = True, ignore_index= False)
    # df_train.reset_index(drop = True, inplace = True)

    # df_train[TEXT] = [preprocess(text )for text in tqdm(df_train[TEXT])]

    # df_train["text"] = df_train["text"].apply(lambda t: t+'\n')
    # df_train_new = pd.DataFrame(
    #     data={
    #         'id': df_train['id'],
    #         'text': df_train['text'],
    #         'label': df_train['label'],
    #     }
    # )
    # df_train.to_csv(
    #     "D:/Desktop/xAI/exp/22C15033_ContrXT/src/text_cls/dataset/20newsgroups/orginal/train__preprocess_v4_1.csv",
    #     index= False
    # )
    print(df_train)


    # TEST_PATH = "D:/Desktop/xAI/exp/22C15033_ContrXT/src/text_cls/dataset/20newsgroups/orginal/test_preproces_v4.csv"
    # df_test = pd.read_csv(TEST_PATH)

    # # df_test.dropna(axis = 0 ,how = 'any', inplace = True, ignore_index= False)
    # # df_test.reset_index(drop = True, inplace = True)

    # # df_test[TEXT] = [preprocess(text )for text in tqdm(df_test[TEXT])]
    # # df_test.to_csv("D:/Desktop/xAI/exp/22C15033_ContrXT/src/text_cls/dataset/20newsgroups/orginal/test__preprocess_v4_1.csv", index= False)
    # print(df_test)