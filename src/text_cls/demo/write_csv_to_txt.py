import pandas as pd
IDX = 0

if __name__ == "__main__":
    df = pd.read_csv("src/text_cls/dataset/sample.csv")

    with open('plantext.txt', 'w', encoding= 'utf-8') as f:
        f.write(df['text'][IDX])

