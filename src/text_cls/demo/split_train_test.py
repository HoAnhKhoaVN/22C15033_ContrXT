import numpy as np
import pandas as pd

from src.text_cls.constant import ID, LABEL, TEXT

def split_balanced(data, target, test_size=0.2, random_state = 2103):
    rng = np.random.RandomState(random_state)
    classes = np.unique(target)
    # can give test_size as fraction of input data size of number of samples
    if test_size<1:
        n_test = np.round(len(target)*test_size)
    else:
        n_test = test_size
    
    n_train = max(0,len(target)-n_test)
    n_train_per_class = max(1,int(np.floor(n_train/len(classes))))
    n_test_per_class = max(1,int(np.floor(n_test/len(classes))))

    ixs = []
    for cl in classes:
        if (n_train_per_class+n_test_per_class) > np.sum(target==cl):
            # if data has too few samples for this class, do upsampling
            # split the data to training and testing before sampling so data points won't be
            #  shared among training and test data
            splitix = int(np.ceil(n_train_per_class/(n_train_per_class+n_test_per_class)*np.sum(target==cl)))
            ixs.append(np.r_[rng.choice(np.nonzero(target==cl)[0][:splitix], n_train_per_class),
                rng.choice(np.nonzero(target==cl)[0][splitix:], n_test_per_class)])
        else:
            ixs.append(rng.choice(np.nonzero(target==cl)[0], n_train_per_class+n_test_per_class,
                replace=False))

    # take same num of samples from all classes
    ix_train = np.concatenate([x[:n_train_per_class] for x in ixs])
    ix_test = np.concatenate([x[n_train_per_class:(n_train_per_class+n_test_per_class)] for x in ixs])

    X_train = data[ix_train]
    X_test = data[ix_test]
    y_train = target[ix_train]
    y_test = target[ix_test]

    return X_train, X_test, y_train, y_test, ix_train, ix_test


if __name__ == "__main__":
    df = pd.read_csv("src/text_cls/dataset/20newsgroups/noun_phrase/train__split_noun_phrase.csv")

    X = np.array(df[TEXT].to_list())
    y = np.array(df[LABEL].to_list())

    X_train, X_test, y_train, y_test, ix_train, ix_test = split_balanced(
        data= X,
        target= y,
        test_size= 0.3
    )

    df_train = pd.DataFrame(
        data = {
            ID : ix_train,
            TEXT : X_train,
            LABEL: y_train
        }
    )

    df_train.to_csv('train.csv', index= False)

    df_test = pd.DataFrame(
        data = {
            ID : ix_test,
            TEXT : X_test,
            LABEL: y_test
        }
    )

    df_test.drop_duplicates(
        subset=[TEXT],
        inplace= True,
        keep= 'first',
        ignore_index= False
    )

    df_test.to_csv('test.csv', index= False)

    print(f"Train IDX: {ix_train}")
    print(f"Test IDX: {ix_test}")



