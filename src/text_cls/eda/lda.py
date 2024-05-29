
# python src/text_cls/eda/lda.py
import pandas as pd
import gensim
import pyLDAvis
import pyLDAvis.gensim
from src.text_cls.constant import TEXT
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

def load_corpus(df):
    corpus=[]
    stem=PorterStemmer()
    lem=WordNetLemmatizer()
    for news in df[TEXT]:
        words=word_tokenize(news)

        words=[lem.lemmatize(w) for w in words if len(w)>2]

        corpus.append(words)
    return corpus


if __name__ == "__main__":
    CSV_PATH = "df_remove_outlier_word.csv"

    df = pd.read_csv(CSV_PATH)
    df.dropna(axis = 0 ,how = 'any', inplace = True, ignore_index= False)
    corpus=load_corpus(df)
    with open('corpus.txt', 'w', encoding='utf-8') as f:
        for word in corpus:
            f.write(f'{word}\n')
    dic=gensim.corpora.Dictionary(corpus)
    bow_corpus = [dic.doc2bow(doc) for doc in corpus]

    n_label = len(df['label'].unique())
    lda_model = gensim.models.LdaMulticore(bow_corpus,
                                   num_topics = n_label,
                                   id2word = dic,
                                   passes = 10,
                                   workers = 2)
    print(lda_model.show_topics())
    vis = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dic)
    pyLDAvis.save_html(vis, 'lda_remove_outlier_word.html')

