# python src/text_cls/eda/lda.py -p src/text_cls/dataset/20newsgroups/noun_phrase/train__split_noun_phrase__remove_stopword.csv -l lda_train_en_bow.html -c corpus.txt -m bow

# REFERENCE
# https://github.com/clevyclev/Deep-Learning-Projects/blob/master/Latent%20Dirichlet%20Allocation%20-%20Bag%20of%20Words%20and%20TF-IDF/Latent_dirichlet_allocation.py#L301
# https://www.kaggle.com/code/gibborathod/topic-modeling-lda-tf-idf#Latent-Dirichlet-Allocation
# 

import argparse
import pandas as pd
import gensim
import pyLDAvis
import pyLDAvis.gensim
from src.text_cls.constant import LABEL, TEXT
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim import corpora, models
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

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

    parser.add_argument("-m", '--mode' ,help="Path corpus to save")
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


    '''
    OPTIONAL STEP
    Remove very rare and very common words:

    - words appearing less than 15 times
    - words appearing in more than 10% of all documents
    '''
    # TODO: apply dictionary.filter_extremes() with the parameters mentioned above
    dic.filter_extremes(no_below=15, no_above=0.1)

    '''
    Create the Bag-of-words model for each document i.e for each document we create a dictionary reporting how many
    words and how many times those words appear. Save this to 'bow_corpus'
    '''
    bow_corpus = [dic.doc2bow(doc) for doc in corpus]



# ## Step 3.2: TF-IDF on our document set ##
# 
# While performing TF-IDF on the corpus is not necessary for LDA implemention using the gensim model, it is recemmended. TF-IDF expects a bag-of-words (integer values) training corpus during initialization. During transformation, it will take a vector and return another vector of the same dimensionality.
# 
# *Please note: The author of Gensim dictates the standard procedure for LDA to be using the Bag of Words model.*

# ** TF-IDF stands for "Term Frequency, Inverse Document Frequency".**
# 
# * It is a way to score the importance of words (or "terms") in a document based on how frequently they appear across multiple documents.
# * If a word appears frequently in a document, it's important. Give the word a high score. But if a word appears in many documents, it's not a unique identifier. Give the word a low score.
# * Therefore, common words like "the" and "for", which appear in many documents, will be scaled down. Words that appear frequently in a single document will be scaled up.
# 
# In other words:
# 
# * TF(w) = `(Number of times term w appears in a document) / (Total number of terms in the document)`.
# * IDF(w) = `log_e(Total number of documents / Number of documents with term w in it)`.
# 
# ** For example **
# 
# * Consider a document containing `100` words wherein the word 'tiger' appears 3 times. 
# * The term frequency (i.e., tf) for 'tiger' is then: 
#     - `TF = (3 / 100) = 0.03`. 
# 
# * Now, assume we have `10 million` documents and the word 'tiger' appears in `1000` of these. Then, the inverse document frequency (i.e., idf) is calculated as:
#     - `IDF = log(10,000,000 / 1,000) = 4`. 
# 
# * Thus, the Tf-idf weight is the product of these quantities: 
#     - `TF-IDF = 0.03 * 4 = 0.12`.

# ## Step 4.1: Running LDA using Bag of Words ##
# 
# We are going for 10 topics in the document corpus.
# 
# ** We will be running LDA using all CPU cores to parallelize and speed up model training.**
# 
# Some of the parameters we will be tweaking are:
# 
# * **num_topics** is the number of requested latent topics to be extracted from the training corpus.
# * **id2word** is a mapping from word ids (integers) to words (strings). It is used to determine the vocabulary size, as well as for debugging and topic printing.
# * **workers** is the number of extra processes to use for parallelization. Uses all available cores by default.
# * **alpha** and **eta** are hyperparameters that affect sparsity of the document-topic (theta) and topic-word (lambda) distributions. We will let these be the default values for now(default value is `1/num_topics`)
#     - Alpha is the per document topic distribution.
#         * High alpha: Every document has a mixture of all topics(documents appear similar to each other).
#         * Low alpha: Every document has a mixture of very few topics
# 
#     - Eta is the per topic word distribution.
#         * High eta: Each topic has a mixture of most words(topics appear similar to each other).
#         * Low eta: Each topic has a mixture of few words.
# 
# * ** passes ** is the number of training passes through the corpus. For  example, if the training corpus has 50,000 documents, chunksize is  10,000, passes is 2, then online training is done in 10 updates: 
#     * `#1 documents 0-9,999 `
#     * `#2 documents 10,000-19,999 `
#     * `#3 documents 20,000-29,999 `
#     * `#4 documents 30,000-39,999 `
#     * `#5 documents 40,000-49,999 `
#     * `#6 documents 0-9,999 `
#     * `#7 documents 10,000-19,999 `
#     * `#8 documents 20,000-29,999 `
#     * `#9 documents 30,000-39,999 `
#     * `#10 documents 40,000-49,999` 

# LDA mono-core -- fallback code in case LdaMulticore throws an error on your machine
# lda_model = gensim.models.LdaModel(bow_corpus, 
#                                    num_topics = 10, 
#                                    id2word = dictionary,                                    
#                                    passes = 50)

    n_label = len(df[LABEL].unique())
    if args.mode =='bow':
        
        lda_model = gensim.models.LdaMulticore(bow_corpus,
                                    num_topics = n_label,
                                    id2word = dic,
                                    passes = 10,
                                    workers = 2)
    elif args.mode == 'tfidf':
        tfidf = models.TfidfModel(bow_corpus)
        # Apply transformation to the entire corpus and call it 'corpus_tfidf'
        corpus_tfidf = tfidf[bow_corpus]

        # Preview TF-IDF scores for our first document --> --> (token_id, tfidf score)
        for doc in corpus_tfidf:
            pprint(doc)
            break
    

        lda_model = gensim.models.LdaMulticore(
            corpus_tfidf,
            num_topics= n_label,
            id2word=dic,
            passes= 10,
            workers= 2
        )
    else:
        raise "Only mode is `bow` or `tfidf`"
    
    '''
    For each topic, we will explore the words occuring in that topic and its relative weight
    '''
    for idx, topic in lda_model.print_topics(-1):
        print("Topic: {} \nWords: {}".format(topic, idx ))
        print("\n")
    # print(lda_model.show_topics())

    # Our test document is document number 4310
    for index, score in sorted(lda_model[bow_corpus[0]], key=lambda tup: -1*tup[1]):
        print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))

    vis = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dic)
    pyLDAvis.save_html(vis, args.html_path)

    df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=df)

    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

    # Show
    df_dominant_topic.head(10)

