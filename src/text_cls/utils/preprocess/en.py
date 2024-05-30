import re
from typing import List, Text
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from tqdm import tqdm
import seaborn as sns

from src.text_cls.constant import LABEL, TEXT
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from textblob import TextBlob

_QUOTE_RE = re.compile(
    r"(writes in|writes:|wrote:|says:|said:" r"|^In article|^Quoted from|^\||^>)"
)

CUSTOM_STOP_WORD = [
    'it',
    'm',
    'he',
    'have',
    'as',
    'can',
    'will',
    'haven',
    'in',
    'same',
    'there',
    'more',
    'being',
    'down',
    'out',
    'now',
    'i',
    're',
    'at',
    'don',
    'do',
    'here',
    'an',
    'why',
    'ma',
    'who',
    't',
    'y',
    'few',
    'are',
    's',
    'o',
    'd',
    'no',
    'one',
    "maxaxaxaxaxaxaxaxaxaxaxaxaxaxax",
    'x',
    'one',
    'would',
    'also',
]

CUSTOM_BIGRAM = [
    'db db'
]

CUSTOM_TRIGRAM = [
    'db db db'
]

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
        # print("###### ORIGINAL #######")
        # print(text)

        # # print("#### REMOVE ENTER ####")
        # text = re.sub(r"\s",r" ",text)
        text = re.sub("[\t\n\r\x0b\x0c]", r' ', text)

        # print("#### SPLIT 1 ####")
        text = re.sub(r"[!:\.\?\-;] ",r"__",text)
        # print(text)

        # # print("#### SPLIT 2 ####")
        # text = re.sub("[\t\n\r\x0b\x0c]", '__', text)
        # print(text)

        # print("#### SPLIT 3 ####")
        text = re.sub(r'\s\s+', r' ', text)
        # print(text)

        # print("#### FINAL ####")
        sents = text.split('__')

        res = []
        for sent in sents:
            tmp = sent.strip()
            if tmp:
                res.append(tmp)
        
        # for sent in res:
        #     print(sent)
        return res

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

    def strip_newsgroup_header(
        self,
        text: Text
    )-> Text:
        """
        Given text in "news" format, strip the headers, by removing everything
        before the first blank line.

        Parameters
        ----------
        text : str
            The text from which to remove the signature block.
        """
        _before, _blankline, after = text.partition("\n\n")
        return after
    
    def strip_newsgroup_quoting(text):
        """
        Given text in "news" format, strip lines beginning with the quote
        characters > or |, plus lines that often introduce a quoted section
        (for example, because they contain the string 'writes:'.)

        Parameters
        ----------
        text : str
            The text from which to remove the signature block.
        """
        good_lines = [line for line in text.split("\n") if not _QUOTE_RE.search(line)]
        return "\n".join(good_lines)

    def strip_newsgroup_footer(text):
        """
        Given text in "news" format, attempt to remove a signature block.

        As a rough heuristic, we assume that signatures are set apart by either
        a blank line or a line made of hyphens, and that it is the last such line
        in the file (disregarding blank lines at the end).

        Parameters
        ----------
        text : str
            The text from which to remove the signature block.
        """
        lines = text.strip().split("\n")
        for line_num in range(len(lines) - 1, -1, -1):
            line = lines[line_num]
            if line.strip().strip("-") == "":
                break

        if line_num > 0:
            return "\n".join(lines[:line_num])
        else:
            return text

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
    
    @staticmethod
    def before_remove_word_outlier(
        df: pd.DataFrame
    )->None:
        """_summary_

        Args:
            df (pd.DataFrame): _description_
        """
        data = df[TEXT].str.split().map(lambda x: len(x))
        print(f'Length data: {len(data)}')
        print(f'Mean: {data.mean()}')
        print(f'Min: {data.min()}')
        print(f'Max: {data.max()}')
        print(f'Std: {data.std()}')
        sns.boxplot(data)
        plt.title(f'Original Box Plot with word length')
        plt.savefig('boxplot_word.png')
    
    @staticmethod
    def removal_word_outlier(
        df: pd.DataFrame,
    )-> pd.DataFrame:
        """_summary_

        Args:
            df (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        df.reset_index(inplace= True, drop=True,)
        data = df[TEXT].str.split().map(lambda x: len(x))
        # IQR    
        Q1 = np.percentile(data, 25, method='midpoint')
        Q3 = np.percentile(data, 75, method='midpoint')
        IQR = Q3 - Q1
        print(f'IQR: {IQR}')

        lower = Q1 - 1.5*IQR
        print(f'Lower: {lower}')

        upper_value = Q3 + 1.5*IQR
        min_value = data.min()
        upper = max(upper_value, min_value)
        print(f'Upper: {upper}')

        # Create arrays of Boolean values indicating the outlier rows
        upper_array = np.where(data >= upper)[0]
        print(f'Length upper_array: {upper_array}')

        lower_array = np.where(data <= lower)[0]
        print(f'Length lower_array: {lower_array}')
        
        # Removing the outliers
        df.drop(index=upper_array, inplace= True)
        df.drop(index=lower_array, inplace= True)
        df.reset_index(inplace= True, drop=True)

        final_data = df[TEXT].str.split().map(lambda x: len(x))
        print(len(final_data))
        print(f'Mean: {final_data.mean()}')
        print(f'Min: {final_data.min()}')
        print(f'Max: {final_data.max()}')
        print(f'Std: {final_data.std()}')
        
        sns.boxplot(final_data)
        plt.title(f'Box Plot after remove outlier')
        plt.savefig('boxplot_word__remove_outlier.png')
        return df

    @staticmethod
    def before_remove_char_outlier(
        df: pd.DataFrame
    )->None:
        """_summary_

        Args:
            df (pd.DataFrame): _description_
        """
        data = df[TEXT].str.len()
        print(f'Length data: {len(data)}')
        print(f'Mean: {data.mean()}')
        print(f'Min: {data.min()}')
        print(f'Max: {data.max()}')
        print(f'Std: {data.std()}')
        sns.boxplot(data)
        plt.title(f'Original Box Plot with character length')
        plt.savefig('boxplot_char.png')

    @staticmethod
    def removal_char_outlier(
        df: pd.DataFrame,
        ):
        df.reset_index(inplace= True, drop=True,)
        # IQR
        data = df[TEXT].str.len()
        Q1 = np.percentile(data, 25, method='midpoint')
        Q3 = np.percentile(data, 75, method='midpoint')
        IQR = Q3 - Q1
        print(f'IQR: {IQR}')

        lower = Q1 - 1.5*IQR
        print(f'Lower: {lower}')

        upper_value = Q3 + 1.5*IQR
        min_value = data.min()
        upper = max(upper_value, min_value)
        print(f'Upper: {upper}')

        # Create arrays of Boolean values indicating the outlier rows
        upper_array = np.where(data >= upper)[0]
        print(f'Length upper_array: {upper_array}')

        lower_array = np.where(data <= lower)[0]
        print(f'Length lower_array: {lower_array}')
        
        # Removing the outliers
        df.drop(index=upper_array, inplace= True)
        df.drop(index=lower_array, inplace= True)
        df.reset_index(inplace= True, drop=True)

        final_data = df[TEXT].str.len()
        print(len(final_data))
        print(f'Mean: {final_data.mean()}')
        print(f'Min: {final_data.min()}')
        print(f'Max: {final_data.max()}')
        print(f'Std: {final_data.std()}')
        
        sns.boxplot(final_data)
        plt.title(f'Box Plot after remove outlier')
        plt.savefig('boxplot_char__remove_outlier.png')
        return df

    def preprocess_text(
        self,
        text: str,
        noun_phrase: bool,
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
                # _s = self.spell_checking(s)
                if s.strip():
                    if noun_phrase:
                        _s = self.add_noun_phrase(s)
                        _s = _s.replace("%20", " ")
                        _s = self.clean_text(_s)
                    else:
                        _s = self.clean_text(s)
                        
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
        noun_phrase: bool
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
            new_text = self.preprocess_text(text, noun_phrase)

            preprocess_texts.append(new_text)
            
        df_processed[TEXT] = preprocess_texts


        # region Remove Outlier

            # region Character
        self.before_remove_char_outlier(df_processed)
        df_processed = self.removal_char_outlier(df_processed)
            # endregion

            # region word
        self.before_remove_word_outlier(df_processed)
        df_processed = self.removal_word_outlier(df_processed)

            # endregion

        # endregion

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
    preprocessor = EnglishTextPreprocessor()
    train_df = pd.read_csv("src/text_cls/dataset/20newsgroups/orginal/train.csv")
    print(train_df.head(3))
    
    # endregion

    # region PREPROCESS
    preprocessed_df = preprocessor.preprocess_dataframe(
        train_df[:20],
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
