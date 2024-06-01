from sklearn.datasets import fetch_20newsgroups


if __name__ == "__main__":
    # Fetch the training data
    newsgroups_train = fetch_20newsgroups(
        subset='train',
        remove=('header','footer','quotes'),
        categories=['rec.autos'],
        shuffle= False
    )

    # # Fetch the test data
    # newsgroups_test = fetch_20newsgroups(subset='test')

    # # Print some information about the dataset
    # print("Number of training documents:", len(newsgroups_train.data))
    # print("Number of test documents:", len(newsgroups_test.data))
    # print("Target names (categories):", newsgroups_train.target_names)

    # Print a sample document
    print("\nSample document:\n", newsgroups_train.data[0])
    print("\nCategory of the sample document:", newsgroups_train.target_names[newsgroups_train.target[0]])
