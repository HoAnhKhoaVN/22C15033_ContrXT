import re
from typing import List, Text
import pandas as pd
from tqdm import tqdm
from underthesea import text_normalize
from underthesea import word_tokenize
from underthesea import sent_tokenize

from src.text_cls.constant import LABEL, TEXT, VIET_STOP_WORD_PATH

class VietnameseTextPreprocessor:
    """
    A class for preprocessing text data in Vietnamese, including common cleaning steps, tokenization, stopword removal, and lemmatization.

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
        self.idx = 0
        self.stop_words = self.load_viet_stop_word()
        self.pattern = r'\b(?:' + '|'.join(re.escape(word) for word in self.stop_words) + r')\b'
        
    def load_viet_stop_word(self)-> List:
        try:
            with open(VIET_STOP_WORD_PATH, "r", encoding="utf-8") as f:
                stopwords = list(set(f.read().splitlines()))
        except FileNotFoundError:
            raise FileNotFoundError(f"Stopwords file not found: {VIET_STOP_WORD_PATH}")

        return stopwords

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

        Args:
            text: The input text to be split.

        Returns:
            A list of strings where each element represents a single sentence from the input text. 

        Notes:
            - Sentences are identified based on common sentence-ending punctuation (period, question mark, exclamation mark).
            - Punctuation marks are included at the end of each sentence.
            - The function handles multiple spaces or leading/trailing spaces around punctuation.
        """
        return sent_tokenize(text)

    def clean_text(self, text: str) -> str:
        """
        Removes non-Vietnamese letters (uppercase or lowercase) and spaces from the given text.

        Args:
            text (str): The input text string.

        Returns:
            str: The modified text with non-Vietnamese letters and spaces removed.
        """

        pattern = r'[^a-zA-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơưẠ-ỹ _]'  # Matches any characters not in the Vietnamese range, plus spaces.

        text = re.sub(pattern, '', text)

        text = text.lower()

        text = text.strip()  # Remove leading/trailing whitespace
        
        text = re.sub(r'\s+', ' ', text)  # Replace multipl
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

    def remove_stopwords(self, text: str) -> str:
        """
        Removes stopwords from a list of words.

        Args:
            words: The list of words to process.

        Returns:
            A list of words with stopwords removed.
        """
        # return [word for word in words if word not in self.stop_words]
        # Use re.sub to replace stop words with an empty string
        result = re.sub(self.pattern, '', text)
        
        # Remove extra spaces that might result from stop word removal
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result
    
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

    def text_normalize(
        self,
        text: str
    ):
        """
        """
        return text_normalize(text)
        
    def word_tokenize(
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
        return word_tokenize(text, format="text")
    
    def preprocess_text(
        self,
        text: str,
        noun_phrase : bool = True,
        remove_stopwords: bool = True,
    ) -> list[str]:
        """
        Applies a full text preprocessing pipeline to a text string.

        Args:
            text: The text string to preprocess.

        Returns:
            A list of preprocessed words.
        """
        self.idx+=1
        sents = self.split_sentences(text)
        res = []
        for _s in sents:
            try:
                # print(f'S BEFORE: {s}')
                
                if remove_stopwords:
                    _s = self.remove_stopwords(_s.lower())
                    # print(f' #### REMOVE STOP WORD ####')
                    # print(_s)

                _s = self.text_normalize(_s)
                # print(f' #### TEXT NORMALIZE ####')
                # print(_s)

                if noun_phrase:
                    _s = self.word_tokenize(_s)
                    # print(f' #### WORD TOKEN ####')
                    # print(_s)
                
                _s = self.clean_text(_s)
                # print(f' #### CLEAN TEXT ####')
                # print(_s)

                # print(f'S AFTER: {_s}')
                if _s.strip():
                    res.append(_s)
                
                # break

            except Exception as e:
                print(f"Error: {e}")
                print(f's: {s}')

        return ". ".join(res)
    
    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
        noun_phrase: bool = True,
        remove_stopwords: bool = True,    
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
        for text in tqdm(df_processed[TEXT]):
            new_text = self.preprocess_text(text, noun_phrase, remove_stopwords)

            preprocess_texts.append(new_text)
            
        df_processed[TEXT] = preprocess_texts

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
        return df.dropna(subset=[TEXT, LABEL])

if __name__ == "__main__":
    # region INIT
    preprocessor = VietnameseTextPreprocessor()
    train_df = pd.read_csv("src/text_cls/dataset/VNTC/sample/sample.csv")

    train_df
    print(train_df.head())
    
    # endregion

    # region PREPROCESS
    preprocessed_df = preprocessor.preprocess_dataframe(
        train_df,
        noun_phrase = True,
        remove_stopwords= True,
    )

    # endregion

    # region TO CSV
    print(preprocessed_df.head())
    preprocessed_df.to_csv(
        path_or_buf= "preprocessed_df_vi.csv",
        index = False
    )

    # endregion
