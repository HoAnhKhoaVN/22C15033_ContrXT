# python src/text_cls/eda/lda.py -p src/text_cls/dataset/20newsgroups/noun_phrase/train__split_noun_phrase__remove_stopword.csv -l lda_train_en.html -c corpus.txt
import argparse
import pandas as pd
import gensim
import pyLDAvis
import pyLDAvis.gensim
from src.text_cls.constant import LABEL, TEXT
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def load_corpus(df):
    corpus=[]
    lem=WordNetLemmatizer()
    for news in df[TEXT]:
        words=word_tokenize(news)

        words=[lem.lemmatize(w) for w in words if len(w)>2]

        corpus.append(words)
    return corpus

def cli():
    parser = argparse.ArgumentParser(description="CLI:")

    # Thêm tham số vị trí (positional argument)
    parser.add_argument("-p", '--path' ,help="Path CSV")

    parser.add_argument("-l", '--html_path' ,help="Path .html to save")

    parser.add_argument("-c", '--corpus_path' ,help="Path corpus to save")
    # Phân tích các tham số đã cung cấp
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = cli()

    df = pd.read_csv(args.path)
    df.dropna(axis = 0 ,how = 'any', inplace = True, ignore_index= False)

    corpus=load_corpus(df)

    with open(args.corpus_path, 'w', encoding='utf-8') as f:
        for word in corpus:
            f.write(f'{word}\n')

    dic=gensim.corpora.Dictionary(corpus)
    bow_corpus = [dic.doc2bow(doc) for doc in corpus]

    n_label = len(df[LABEL].unique())
    lda_model = gensim.models.LdaMulticore(bow_corpus,
                                   num_topics = n_label,
                                   id2word = dic,
                                   passes = 10,
                                   workers = 2)
    print(lda_model.show_topics())
    vis = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dic)
    pyLDAvis.save_html(vis, args.html_path)

