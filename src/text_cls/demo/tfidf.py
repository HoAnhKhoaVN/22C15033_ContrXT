from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


if __name__ == "__main__":
    # Define the sample texts
    documents = [
        "machine learning is fascinating",
        "artificial intelligence is a branch of computer science",
        "machine learning and artificial intelligence are closely related fields"
    ]

    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the texts into TF-IDF embeddings
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Get feature names
    feature_names = vectorizer.get_feature_names_out()

    # Create a DataFrame with the TF-IDF values
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    
    print(tfidf_df)
