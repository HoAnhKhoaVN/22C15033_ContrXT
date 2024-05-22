import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class EnglishTextPreprocessor:
    """
    A class for preprocessing text data in English, including common cleaning steps, tokenization, stopword removal, and lemmatization.

    Attributes:
        None

    Methods:
        clean_text(text: str) -> str:
            Cleans a text string by removing special characters, converting to lowercase, and handling whitespace.

        tokenize_text(text: str) -> list[str]:
            Tokenizes a text string into a list of words.

        remove_stopwords(words: list[str]) -> list[str]:
            Removes stopwords from a list of words.

        lemmatize_words(words: list[str]) -> list[str]:
            Lemmatizes a list of words.

        preprocess_text(text: str) -> list[str]:
            Applies a full text preprocessing pipeline to a text string.

        preprocess_dataframe(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
            Applies the full text preprocessing pipeline to a specified text column in a DataFrame.
    """

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

    def stem_text(self, text: str) -> str:
        """
        Applies stemming to reduce words to their root form.
        
        Args:
        text (str): The text to preprocess.
        
        Returns:
        str: The text with words stemmed.
        """
        return ' '.join([self.stemmer.stem(word) for word in text.split()])
    
    def remove_punctuation(self, text: str) -> str:
        """
        Removes punctuation from the text.
        
        Args:
        text (str): The text to preprocess.
        
        Returns:
        str: The text without punctuation.
        """
        return re.sub(r'[^\w\s]', '', text)

    def clean_text(self, text: str) -> str:
        """
        Cleans a text string by removing special characters, converting to lowercase, and handling whitespace.

        Args:
            text: The text string to clean.

        Returns:
            The cleaned text string.
        """
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
        text = text.lower()  # Convert to lowercase
        text = text.strip()  # Remove leading/trailing whitespace
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        return text

    def tokenize_text(self, text: str) -> list[str]:
        """
        Tokenizes a text string into a list of words.

        Args:
            text: The text string to tokenize.

        Returns:
            A list of words.
        """
        return text.split()

    def to_lowercase(self, text: str) -> str:
        """
        Converts all characters in the text to lowercase.
        
        Args:
        text (str): The text to preprocess.
        
        Returns:
        str: The text in lowercase.
        """
        return text.lower()

    def remove_stopwords(self, words: list[str]) -> list[str]:
        """
        Removes stopwords from a list of words.

        Args:
            words: The list of words to process.

        Returns:
            A list of words with stopwords removed.
        """
        return [word for word in words if word not in self.stop_words]

    def lemmatize_words(self, words: list[str]) -> list[str]:
        """
        Lemmatizes a list of words.

        Args:
            words: The list of words to lemmatize.

        Returns:
            A list of lemmatized words.
        """
        return [self.lemmatizer.lemmatize(word) for word in words]

    def join_words(
        self,
        words: list[str],
        separator: str = " "
    ) -> str:
        """
        Joins a list of words into a single string, using a specified separator.

        Args:
            words: A list of strings representing the words to join.
            separator: The string to insert between the words (default: a single space).

        Returns:
            The resulting string with words joined by the separator.

        Example:
            >>> words = ["This", "is", "a", "sentence."]
            >>> joined_text = join_words(words)
            >>> print(joined_text)  
            This is a sentence.
        """

        joined_string = separator.join(words)
        return joined_string


    def preprocess_text(self, text: str) -> list[str]:
        """
        Applies a full text preprocessing pipeline to a text string.

        Args:
            text: The text string to preprocess.

        Returns:
            A list of preprocessed words.
        """
        try:
            text = self.clean_text(text)
            words = self.tokenize_text(text)
            words = self.remove_stopwords(words)
            words = self.lemmatize_words(words)
            result = self.join_words(words)
        except Exception as e:
            print(f"Error: {e}")
            print(f'Text: {text}')

        return result
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Applies the full text preprocessing pipeline to a specified text column in a DataFrame.

        Args:
            df: The DataFrame containing the text data.
            text_column: The name of the column containing the text data.

        Returns:
            The DataFrame with the specified column preprocessed.
        """
        # Create an explicit copy of the DataFrame to avoid SettingWithCopyWarning
        df_processed = df.copy()

        # Drop rows with NaN values in the specified text column
        df_processed = self.remove_nan_rows(df_processed)

        # Preprocess
        df_processed[text_column] = df_processed[text_column].apply(self.preprocess_text)
        return df_processed
    
    def remove_nan_rows(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Removes rows from the DataFrame where the text column is NaN.

        Args:
        df (pd.DataFrame): The DataFrame to clean.

        Returns:
        pd.DataFrame: The DataFrame with NaN text rows removed.
        """
        return df.dropna(subset=['text'])

if __name__ == "__main__":  
    preprocessor = EnglishTextPreprocessor()
    train_df = pd.read_csv("src/text_cls/dataset/20newsgroups/train.csv")
    print(train_df.head(10))
    preprocessed_df = preprocessor.preprocess_dataframe(train_df, 'text')
    print(preprocessed_df.head(10))
