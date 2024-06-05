# python src/text_cls/demo/preprocess/en.py
import re
from typing import Any, Text
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from src.text_cls.constant import EN_STOP_WORD, ID, LABEL, TEXT
nltk.download('stopwords')

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

class PreprocessEnglish(object):
    def __init__(
        self,
        csv: Text,
        out: Text
    )-> None:
        self.csv = csv
        self.out = out
        self.df = self.read_data()
        self.stop_word= self.get_stop_word()

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
    
    def read_data(self)-> pd.DataFrame:
        """Read CSV to DataFrame
        
        Returns:
            pd.DataFrame:Output
        """
        df = pd.read_csv(self.csv)
        df.dropna(axis = 0 ,how = 'any', inplace = True, ignore_index= False)

        return df

    def remove_custom_stop_word(self)-> pd.DataFrame:
        corpus=[]
        bi_pattern = r'\b(?:' + '|'.join(re.escape(word) for word in CUSTOM_BIGRAM) + r')\b'
        tri_pattern = r'\b(?:' + '|'.join(re.escape(word) for word in CUSTOM_TRIGRAM) + r')\b'
        for text in self.df[TEXT]:
            sent=' '.join([w for w in word_tokenize(text) if (w not in self.stop_word)])
            sent = re.sub(bi_pattern, '', sent)
            sent = re.sub(tri_pattern, '', sent)

            corpus.append(sent)

        new_df = pd.DataFrame(
            data = {
                ID: self.df[ID],
                TEXT: corpus,
                LABEL: self.df[LABEL]
            }
        )

        return new_df
    
    def __call__(self) -> Any:
        new_df : pd.DataFrame = self.remove_custom_stop_word()

        new_df.to_csv(
            path_or_buf= self.out,
            index = False
        )

if __name__ == "__main__":
    PreprocessEnglish(
        csv = 'src/text_cls/dataset/20newsgroups/noun_phrase/train__split_noun_phrase.csv',
        out = 'src/text_cls/dataset/20newsgroups/noun_phrase/train__split_noun_phrase__remove_stopword.csv'
    )()