import re
from typing import List, Text
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from tqdm import tqdm

from src.text_cls.constant import EN_STOP_WORD, LABEL, TEXT, TEXT_CLS_CONTANST
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from textblob import TextBlob

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
    'db db',
    'ax ax'
]

CUSTOM_TRIGRAM = [
    'db db db',
    'ax ax ax'
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
        self.stop_words= self.get_stop_word()
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.idx = 0

    def get_stop_word(self):
        # region Load stop word on NLTK
        nltk_stop_word = set(stopwords.words('english'))

        # endregion
        
        # region Load stop word on Mathematica
        # ref: https://gist.github.com/sebleier/554280?permalink_comment_id=3525942#gistcomment-3525942
        s = """0,1,2,3,4,5,6,7,8,9,a,A,about,above,across,after,again,against,all,almost,alone,along,already,also,although,always,am,among,an,and,another,any,anyone,anything,anywhere,are,aren't,around,as,at,b,B,back,be,became,because,become,becomes,been,before,behind,being,below,between,both,but,by,c,C,can,cannot,can't,could,couldn't,d,D,did,didn't,do,does,doesn't,doing,done,don't,down,during,e,E,each,either,enough,even,ever,every,everyone,everything,everywhere,f,F,few,find,first,for,four,from,full,further,g,G,get,give,go,h,H,had,hadn't,has,hasn't,have,haven't,having,he,he'd,he'll,her,here,here's,hers,herself,he's,him,himself,his,how,however,how's,i,I,i'd,if,i'll,i'm,in,interest,into,is,isn't,it,it's,its,itself,i've,j,J,k,K,keep,l,L,last,least,less,let's,m,M,made,many,may,me,might,more,most,mostly,much,must,mustn't,my,myself,n,N,never,next,no,nobody,noone,nor,not,nothing,now,nowhere,o,O,of,off,often,on,once,one,only,or,other,others,ought,our,ours,ourselves,out,over,own,p,P,part,per,perhaps,put,q,Q,r,R,rather,s,S,same,see,seem,seemed,seeming,seems,several,shan't,she,she'd,she'll,she's,should,shouldn't,show,side,since,so,some,someone,something,somewhere,still,such,t,T,take,than,that,that's,the,their,theirs,them,themselves,then,there,therefore,there's,these,they,they'd,they'll,they're,they've,this,those,though,three,through,thus,to,together,too,toward,two,u,U,under,until,up,upon,us,v,V,very,w,W,was,wasn't,we,we'd,we'll,well,we're,were,weren't,we've,what,what's,when,when's,where,where's,whether,which,while,who,whole,whom,who's,whose,why,why's,will,with,within,without,won't,would,wouldn't,x,X,y,Y,yet,you,you'd,you'll,your,you're,yours,yourself,yourselves,you've,z,Z"""
        stop_word_mathematica = set(s.split(','))

        # endregion

        # region Load full stop word
        # ref: https://github.com/stopwords-iso/stopwords-en/blob/master/stopwords-en.txt
        stop_word_full = set()
        with open(EN_STOP_WORD, 'r', encoding= 'utf-8') as f:
            for line in f:
                stop_word_full.add(line.strip())

        # endregion

        # region Load custom stop word after EDA
        custom_stop_word = set(CUSTOM_STOP_WORD)

        # endregion

        # region merge all stop word
        final_stop_word = set()
        final_stop_word.update(nltk_stop_word)
        final_stop_word.update(stop_word_mathematica)
        final_stop_word.update(stop_word_full)
        final_stop_word.update(custom_stop_word)

        # endregion
        
        print(f'Length stop word: {len(final_stop_word)}')
        return final_stop_word
    
    def stem_text(self, text: str) -> str:
        """
        Applies stemming to reduce words to their root form.
        
        Args:
        text (str): The text to preprocess.
        
        Returns:
        str: The text with words stemmed.
        """
        return ' '.join([self.stemmer.stem(word) for word in text.split()])
    
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
        text = re.sub(r"[^a-zA-Z_\s']", ' ', text)  # Remove special characters
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
    def find_upper(df: pd.DataFrame)-> int:
        """_summary_

        Args:
            df (pd.DataFrame): _description_

        Returns:
            int: _description_
        """
        df.reset_index(inplace= True, drop=True,)
        data = df[TEXT].str.split().map(lambda x: len(x))
        # IQR    
        Q1 = np.percentile(data, 25, method='midpoint')
        Q3 = np.percentile(data, 75, method='midpoint')
        IQR = Q3 - Q1
        upper = int(Q3 + 1.5*IQR)
        print(f'Upper: {upper}')
        with open(TEXT_CLS_CONTANST, 'a', encoding= 'utf-8') as f:
            f.write(f'\nMAX_WORD_LEN= {upper}\n')

        return upper

    @staticmethod
    def removal_word_outlier(
        df: pd.DataFrame,
        upper: int
    )-> pd.DataFrame:
        """_summary_

        Args:
            df (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        # Create arrays of Boolean values indicating the outlier rows
        df[TEXT] = df[TEXT].str.split().map(lambda x: ' '.join(x[:upper]) if len(x)> upper else ' '.join(x))
        # df.to_csv('df_after_trunc.csv', index = False)
        # final_data = df[TEXT].str.split().map(lambda x: len(x))
        # print(len(final_data))
        # print(f'Mean: {final_data.mean()}')
        # print(f'Min: {final_data.min()}')
        # print(f'Max: {final_data.max()}')
        # print(f'Std: {final_data.std()}')
        # sns.boxplot(final_data)
        # plt.title(f'Box Plot after remove outlier')
        # plt.savefig('boxplot_word__remove_outlier.png')
        return df

    def preprocess_text(
        self,
        text: str,
        noun_phrase: bool,
        show_log: bool = False
    ) -> list[str]:
        """
        Applies a full text preprocessing pipeline to a text string.

        Args:
            text: The text string to preprocess.

        Returns:
            A list of preprocessed words.
        """
        self.idx+=1
        if show_log:
            print(f'Text')
            print(text)
        
        sents = self.split_sentences(text)

        if show_log:
            print("After split_sentences")
            print(f'{text}')

        res = []

        for s in sents:
            try:
                # _s = self.spell_checking(s)
                s = s.strip()
                if s:
                    if show_log:
                        print(f'====== {s} =======')
                    if noun_phrase:
                        _s = self.add_noun_phrase(s)
                        _s = _s.replace("%20", " ")
                        if show_log:
                            print(f'add_noun_phrase: {_s}')

                        _s = self.clean_text(_s)

                        if show_log:
                            print(f'clean_text: {_s}')
                    else:
                        _s = self.clean_text(s)
                        if show_log:
                            print(f'clean_text: {_s}')
                        
                    words = self.tokenize_text(_s)
                    if show_log:
                        print(f'tokenize_text: {words}')

                    words = self.remove_stopwords(words)

                    if show_log:
                        print(f'remove_stopwords: {words}')

                    words = self.lemmatize_words(words)
                    if show_log:
                        print(f'lemmatize_words: {words}')

                    tmp_s = self.join_words(words)

                    if show_log:
                        print(f'join_words: {words}')

                    if tmp_s.strip():
                        res.append(f'{tmp_s}.')
                        if show_log:
                            print(f'final: {tmp_s}')

            

            

            except Exception as e:
                print(f"Error: {e}")
                print(f's: {s}')

        final_text = " ".join(res)
        final_text = re.sub(
            pattern= r"[']",
            repl= ' ',
            string = final_text
        )

        if show_log:
            print('Remove quoue char:')
            print(final_text)


        final_text = re.sub(
            pattern= r" _([a-z]+)_ ",
            repl= r'\1',
            string = final_text
        )

        if show_log:
            print('Remove _abc_:')
            print(final_text)

        return final_text
    
    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
        noun_phrase: bool,
        show_log: bool = False,
        is_train: bool = True
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
            new_text = self.preprocess_text(text, noun_phrase, show_log)
            preprocess_texts.append(new_text)
            
        df_processed[TEXT] = preprocess_texts
        df_processed = self.remove_nan_rows(df_processed)

        print(f'df_processed before')
        print(df_processed.head(20))


        # region Remove Outlier . Only for train. Test thì chỉ cần lấy những kết quả của train để mà áp dụng vào.
        # Cụ thể, với xóa cái điểm ngoại vi thì trong tập test không thể xóa. Ta sẽ cắt lấy đi max_word. Mình đâu cần xóa dữ liệu. Chỉ cần trích đi các outlier theo tập train.            
        if is_train:
            # region word
            print(f'Train mode')
            upper = self.find_upper(df)
            df_processed = self.removal_word_outlier(df = df_processed, upper= upper)
            print(df_processed.head(20))

            # endregion
        else: # Test
            from src.text_cls.constant import MAX_WORD_LEN
            print(f'Test mode with max word len is {MAX_WORD_LEN}')
            df_processed = self.removal_word_outlier(df = df_processed, upper= MAX_WORD_LEN)
            print(df_processed.head(20))
        # endregion

        df_processed = self.remove_nan_rows(df = df_processed)

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
    train_df = pd.read_csv("src/text_cls/dataset/20newsgroups/orginal/test_part_1.csv")
    print(train_df.head())
    
    # endregion

    # region PREPROCESS
    preprocessed_df = preprocessor.preprocess_dataframe(
        train_df[:20],
        noun_phrase = True,
        show_log = False,
        is_train= False
    )

    # endregion

    # region TO CSV
    print(preprocessed_df)
    preprocessed_df.to_csv(
        path_or_buf= "test_df_en.csv",
        index = False
    )

    # endregion
