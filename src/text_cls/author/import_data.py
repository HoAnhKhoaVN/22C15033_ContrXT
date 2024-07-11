from collections import Counter
import pickle
import pandas as pd

from src.text_cls.author.const import DATASET_DIR, TRAIN_OR_TEST


if __name__ == "__main__":
    df_time_1 = pd.read_csv(f'data/{DATASET_DIR}/df_time_1_{TRAIN_OR_TEST}.csv')
    df_time_2 = pd.read_csv(f'data/{DATASET_DIR}/df_time_2_{TRAIN_OR_TEST}.csv')

    df_time_1 = df_time_1[~df_time_1['corpus'].isnull()]
    df_time_2 = df_time_2[~df_time_2['corpus'].isnull()]

    # Load vectorizer
    vectorizer_time_1 = pickle.load(
        open(f'model/{DATASET_DIR}/vectorizer_time_1.pickle', 'rb'))
    vectorizer_time_2 = pickle.load(
        open(f'model/{DATASET_DIR}/vectorizer_time_2.pickle', 'rb'))

    X_t1, Y_t1 = df_time_1['corpus'], df_time_1['category']
    X_t2, Y_t2 = df_time_2['corpus'], df_time_2['category']

    print('Classes for time 1: ', Counter(Y_t1), flush=True)
    print('Classes for time 2: ', Counter(Y_t2), flush=True)

    MODEL = 'BIGRU'

    predicted_labels_t1 = pd.read_csv(
        f'data/{DATASET_DIR}/{MODEL}_surrogate_predicted_labels/pred_t1.csv', header=None)[0].values.astype('str')
    predicted_labels_t2 = pd.read_csv(
        f'data/{DATASET_DIR}/{MODEL}_surrogate_predicted_labels/pred_t2.csv', header=None)[0].values.astype('str')
    

    # region Get surrogate predicted labels
    preds = []
    with open(f'data/{DATASET_DIR}/{MODEL}_surrogate_predicted_labels/pred_t1.csv', 'r') as file:
        for line in file.readlines():
            line = line.strip()
            if line.startswith('['):
                elem = line
                continue
            if line.endswith(']'):
                elem += line
                elem = elem.strip('[]').split(' ')
                preds.append(elem)
                continue
            elem += line

    # endregion

    