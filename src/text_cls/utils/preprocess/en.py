import re
from typing import List, Text
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from tqdm import tqdm
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from textblob import TextBlob


def is_valid_string(input_string: Text) -> bool:
    """
    Checks if the input string consists entirely of letters (uppercase or lowercase), 
    numbers, hyphens, and spaces.

    Args:
        input_string (str): The string to be checked.

    Returns:
        bool: True if the string is valid, otherwise False.
    """
    # Define the regular expression pattern
    pattern = r'^[a-zA-Z0-9\- ]+$'
    
    # Use re.fullmatch to check if the entire string matches the pattern
    match = re.fullmatch(pattern, input_string)
    
    # Return True if there is a match, otherwise False
    return match is not None


def check_alphabet(
    texts: List[Text]
)-> List[Text]:
    res = []
    for text in texts:
        if is_valid_string(text):
            res.append(text)
    return res

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
        self.idx = 0

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

    def split_sentences(
        self,
        text: Text
    )-> List[Text]:
        """
        Splits a text string into a list of individual sentences.

        This function uses a regular expression to identify sentence boundaries based
        on punctuation and common abbreviations. It handles edge cases like
        sentences ending in abbreviations or questions marks.

        Args:
            text (str): The text string to split into sentences.

        Returns:
            list: A list of strings, where each string is a single sentence.

        Examples:
            >>> split_sentences("This is a sentence. Is this another? It is!")
            ['This is a sentence.', 'Is this another?', 'It is!']

            >>> split_sentences("I love N.Y. It's a great city.")
            ['I love N.Y.', "It's a great city."]

            >>> split_sentences("Dr. Smith said, 'Hello there!'")
            ['Dr', "Smith said, 'Hello there!'"]
        """
        text = text = re.sub("\n"," ",text)
        text = re.sub(r"[!:\.\?\-;]\s",r"__",text)
        text = re.sub("[\t\n\r\x0b\x0c]", '__', text)
        # text = re.sub(r"\"\"", '__', text)
        text = re.sub(r'\s\s+', '__', text)
        sents = text.split('__')
        return sents

    def clean_text(self, text: str) -> str:
        """
        Cleans a text string by removing special characters, converting to lowercase, and handling whitespace.

        Args:
            text: The text string to clean.

        Returns:
            The cleaned text string.
        """
        text = re.sub(r'[^a-zA-Z_\s]', '', text)  # Remove special characters
        text = text.lower()  # Convert to lowercase
        text = text.strip()  # Remove leading/trailing whitespace
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        return text

    def tokenize_text(
        self,
        text: str,
    ) -> list[str]:
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

    def spell_checking(
        self,
        text: str
    ):
        """
        """
        # Spell-checking using TextBlob
        corrected_sentence = TextBlob(text).correct()
    
        return corrected_sentence.string
        
    def add_noun_phrase(
        self,
        text
    )-> str:
        """
        Processes the input sentence by spell-checking and tokenizing it with noun phrases linked together by '_'.
        
        Args:
        input_sentence (str): The sentence to be processed.
        
        Returns:
        str: The processed sentence with noun phrases linked by '_'.    
        """

        # Extracting noun phrases using TextBlob
        blob = TextBlob(text)
        noun_phrases = check_alphabet(blob.noun_phrases)

        # Tokenizing the sentence by replacing spaces in noun phrases with underscores
        processed_sentence = blob.string.replace(" ", "%20")

        # Creating a regular expression pattern for matching noun phrases
        for phrase in noun_phrases:
            # Replace spaces with underscores within each noun phrase
            underscored_phrase = '_'.join(phrase.split())
            # Use regular expression to replace the original noun phrase in the sentence with the underscored version
            pattern = re.escape(phrase.replace(" ", "%20"))
            processed_sentence = re.sub(pattern, underscored_phrase, processed_sentence)

        processed_sentence.replace("%20", "")
        return processed_sentence
    
    def preprocess_text(
        self,
        text: str,
        noun_phrase : bool = False,
    ) -> list[str]:
        """
        Applies a full text preprocessing pipeline to a text string.

        Args:
            text: The text string to preprocess.

        Returns:
            A list of preprocessed words.
        """
        self.idx+=1
        # print(f'Text: {text}')
        # print(f'{self.idx}')
        sents = self.split_sentences(text)

        res = []

        for s in sents:
            try:
                _s = self.spell_checking(s)
                if noun_phrase:
                    _s = self.add_noun_phrase(_s)
                    _s = _s.replace("%20", " ")
                _s = self.clean_text(_s)
                words = self.tokenize_text(_s)
                words = self.remove_stopwords(words)
                words = self.lemmatize_words(words)
                tmp_s = self.join_words(words)

                if tmp_s.strip():
                    res.append(tmp_s)

            except Exception as e:
                print(f"Error: {e}")
                print(f's: {s}')
            
        # print(res)
        return ". ".join(res)
    
    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        noun_phrase: bool = False,    
    ) -> pd.DataFrame:
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
        preprocess_texts = []
        for text in tqdm(df_processed[text_column]):
            new_text = self.preprocess_text(
                text,
                noun_phrase,
            )

            preprocess_texts.append(new_text)
            
        df_processed[text_column] = preprocess_texts

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
    # region INIT
    preprocessor = EnglishTextPreprocessor()
    train_df = pd.read_csv("src/text_cls/dataset/20newsgroups/train.csv")
    print(train_df.head(3))
    
    # endregion

    # region PREPROCESS
    preprocessed_df = preprocessor.preprocess_dataframe(
        train_df[:3],
        'text',
        noun_phrase = True,
    )

    # endregion

    # region TO CSV
    print(preprocessed_df)
    preprocessed_df.to_csv(
        path_or_buf= "preprocessed_df_en.csv",
        index = False
    )

    # endregion
