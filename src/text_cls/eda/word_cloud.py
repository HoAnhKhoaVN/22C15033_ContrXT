import pandas as pd
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)
import matplotlib.pyplot as plt
from src.text_cls.constant import TEXT

def load_corpus(df):
    corpus=[]
    for text in df[TEXT]:
        words = []
        for w in text.split(' '):
            if w != '.':
                words.append(w)
        corpus.extend(words)
    return corpus

def show_wordcloud(data):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=100,
        max_font_size=30,
        scale=3,
        random_state=1)

    wordcloud=wordcloud.generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    # fig.axis('off')
    fig.savefig('word_cloud.png')
if __name__ == "__main__":
    
    CSV_PATH = "df_remove_outlier_char.csv"

    df = pd.read_csv(CSV_PATH)
    df.dropna(axis = 0 ,how = 'any', inplace = True, ignore_index= False)

    corpus=load_corpus(df)

    show_wordcloud(corpus)