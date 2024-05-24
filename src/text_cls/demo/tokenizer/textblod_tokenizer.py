from textblob import TextBlob
import re

def process_text(input_sentence: str) -> str:
    """
    Processes the input sentence by spell-checking and tokenizing it with noun phrases linked together by '_'.
    
    Args:
    input_sentence (str): The sentence to be processed.
    
    Returns:
    str: The processed sentence with noun phrases linked by '_'.
    """
    
    # Spell-checking using TextBlob
    corrected_sentence = TextBlob(input_sentence).correct()

    # Extracting noun phrases using TextBlob
    noun_phrases = corrected_sentence.noun_phrases

    # Tokenizing the sentence by replacing spaces in noun phrases with underscores
    processed_sentence = corrected_sentence.string

    # Creating a regular expression pattern for matching noun phrases
    for phrase in noun_phrases:
        # Replace spaces with underscores within each noun phrase
        underscored_phrase = '_'.join(phrase.split())
        # Use regular expression to replace the original noun phrase in the sentence with the underscored version
        processed_sentence = re.sub(r'\b' + re.escape(phrase) + r'\b', underscored_phrase, processed_sentence)

    
    # region Split sentences
    sentences = TextBlob(processed_sentence).sentences
    print(sentences)
    # res = []
    # for sent in sentences:
    #     print(sent)


    # endregion
    
    return sentences

if __name__ == "__main__":
    text = """
well folks, my mac plus finally gave up the ghost this weekend after
starting life as a 512k way back in 1985.  sooo, i'm in the market for a
new machine a bit sooner than i intended to be...

i'm looking into picking up a powerbook 160 or maybe 180 and have a bunch
of questions that (hopefully) somebody can answer:

* does anybody know any dirt on when the next round of powerbook
introductions are expected?  i'd heard the 185c was supposed to make an
appearence ""this summer"" but haven't heard anymore on it - and since i
don't have access to macleak, i was wondering if anybody out there had
more info...

* has anybody heard rumors about price drops to the powerbook line like the
ones the duo's just went through recently?

* what's the impression of the display on the 180?  i could probably swing
a 180 if i got the 80Mb disk rather than the 120, but i don't really have
a feel for how much ""better"" the display is (yea, it looks great in the
store, but is that all ""wow"" or is it really that good?).  could i solicit
some opinions of people who use the 160 and 180 day-to-day on if its worth
taking the disk size and money hit to get the active display?  (i realize
this is a real subjective question, but i've only played around with the
machines in a computer store breifly and figured the opinions of somebody
who actually uses the machine daily might prove helpful).

* how well does hellcats perform?  ;)

thanks a bunch in advance for any info - if you could email, i'll post a
summary (news reading time is at a premium with finals just around the
corner... :( )
--
Tom Willis  /  twillis@ecn.purdue.edu    /    Purdue Electrical Engineering
"""
    processed_sentence = process_text(text)
    print(processed_sentence)


    # blob_obj = TextBlob(
    #     text = text,
    #     tokenizer= None,
    #     np_extractor= None,
    #     pos_tagger= None
    # )

    # # region Tokenizer
    # print(f"noun_phrases: {blob_obj.noun_phrases}")
    # print(f"ngrams: {blob_obj.ngrams(n = 2)}")
    # print(f"tokenize: {blob_obj.tokenize()}")
    # print(f"words: {blob_obj.words}")
    # print(f"tokenizer: {blob_obj.tokenizer}")
    # print(f"words: {blob_obj.words}")
    # print(dir(blob_obj))

    # noun_phrases = blob_obj.noun_phrases
    # print(f"noun_phrases: {noun_phrases}")

    # endregion
