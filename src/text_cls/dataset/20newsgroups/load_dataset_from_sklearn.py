from typing import Any, Text
from sklearn.datasets import fetch_20newsgroups
from pickle import dump, load

def dump_pickle(
    fn: Text,
    obj: Any
)->None:
    with open(fn, 'wb') as f:
        dump(
            obj= obj,
            file = f
        )

def load_pickle(fn: Text)-> Any:
    with open(fn, 'rb') as f:
        data = load(f)
    
    return data


if __name__ == "__main__":
    # Fetch the training data
    newsgroups_train = fetch_20newsgroups(
        subset='train',
        remove=('header','footer','quotes'),
        shuffle= False
    )

    plk_path = 'newsgroups_train.plk'
    dump_pickle(
        fn = plk_path,
        obj= newsgroups_train
    )

    newsgroups_train_ = load_pickle(plk_path)


    # # Fetch the test data
    # newsgroups_test = fetch_20newsgroups(subset='test')

    # # Print some information about the dataset
    # print("Number of training documents:", len(newsgroups_train.data))
    # print("Number of test documents:", len(newsgroups_test.data))
    # print("Target names (categories):", newsgroups_train.target_names)

    # Print a sample document
    print("\nSample document:\n", newsgroups_train_.data[0])
    print("\nCategory of the sample document:", newsgroups_train_.target_names[newsgroups_train.target_[0]])
