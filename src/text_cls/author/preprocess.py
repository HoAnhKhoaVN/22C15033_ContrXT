# python src/text_cls/author/preprocess.py
import os
import re
import string
import time
from typing import Text
import pandas as pd
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from tqdm import tqdm

from src.text_cls.constant import TEXT
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def lemmatize(nlp, text: str):
    """
    Lemmatize a text.

    Parameters
    ---------
    nlp: spacy.Language
        the lemmatizer to use

    text: str
        the text to lemmatize

    Returns
    -------
    lemmas: List[str]
        a list of the lemmatized words in the same order as the original text
    """
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc]

    return lemmas


def preprocess(nlp, text: str, removeEmail, removeUrl, removeNumbers, removeOthers, language='english', subject=False):
    # -> List[List[str]]:
    """
    Preprocess a text for uni/bigram extraction.
    Returns a lemmatized and cleaned version of the text.
    """

    # Remove anything before Subject:
    if subject:
        text = text[text.find('Subject:') + 9:]

    # Run regex
    textNoEmail = removeEmail.sub("", text)
    textNoUrl = removeUrl.sub("", textNoEmail)
    textNoNumbers = removeNumbers.sub("", textNoUrl)

    # Lemmatize
#     get_lemmas = partial(lemmatize, nlp)
#     lemmatized = " ".join(get_lemmas(textNoNumbers))

    # Also to be removed
    punct = set(string.punctuation)
    stopw = stopwords.words(language)
#     stopw.remove('it')

    # funzioni da applicare in sequenza al singolo testo (str)
    # 0 separa in frasi: sent_tokenize e toglie la punteggiatura
    # 1 separa in parole: word_tokenize, strips newline e altro
    # 2 str.lower e togliamo la punteggiatura e le stop words

    sentences = nltk.sent_tokenize(textNoNumbers, language=language)
    sentencesNoPunct = [
        ''.join([c for c in sentence if c not in punct]) for sentence in sentences]
    words = [nltk.word_tokenize(sentence, language=language)
             for sentence in sentencesNoPunct]
    result = ' '.join([' '.join([str.lower(x)
                      for x in ws if str.lower(x) not in stopw]) for ws in words])
    return result


def setup(df: pd.DataFrame):
    """
    Setup data and perform preprocessing

    Returns
    -------
    df: pandas.DataFrame
    """

    # region Regex compilation
    removeEmail = re.compile(r'\S*@\S*\s?', re.IGNORECASE)
    removeUrl = re.compile(r'^https?:\/\/.*[\r\n]*', re.IGNORECASE)
    removeNumbers = re.compile(r'\d+')
    removeOthers = re.compile(r'^a-zA-Z ')

    # endregion

    # region Preprocess data
    start = time.time()
    print("Begin preprocessing")
    df[TEXT] = [preprocess(None, text, removeEmail, removeUrl, removeNumbers,
                               removeOthers, subject=True) for text in tqdm(df[TEXT])]

    print("Execution time %d" % (time.time()-start))

    # endregion

    return df



if __name__ == "__main__":
    # region 1: Constant
    TRAIN_PATH = "src/text_cls/dataset/20newsgroups/orginal/train_part_1.csv"
    ROOT = "src/text_cls/author"
    is_train = False

    # endregion

    # region 2: Input
    df = pd.read_csv(TRAIN_PATH)
    df.dropna(axis = 0 ,how = 'any', inplace = True, ignore_index= False)
    if is_train:
        df.drop(index = [144,1492, 1506, 1541, 2931, 3198, 4495, 4515, 4772, 8213, 8665, 9080, 10275], inplace = True)
        df.reset_index(drop = True, inplace = True)
    print(df.head())

    # endregion


    # region 3: Preprocess
    df = setup(df)
    print(df.head())

    fn = os.path.basename(TRAIN_PATH).split(".")[0]
    path = os.path.join(ROOT, f'{fn}.csv')
    df.to_csv(path)

    # endregion

