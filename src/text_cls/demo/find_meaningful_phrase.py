import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

def find_meaningful_phrases(words):
    """
    This function takes a list of words and groups them into meaningful phrases based on bigram collocations.

    Args:
        words (list of str): A list of words from which to form phrases.

    Returns:
        list of tuples: A list of tuples where each tuple contains words that form a meaningful phrase.

    Example:
        >>> find_meaningful_phrases(["deep", "learning", "artificial", "intelligence", "machine", "learning"])
        [('artificial', 'intelligence'), ('machine', 'learning'), ('deep', 'learning')]
    """
    # Tokenize the list of words into a single string for processing
    text = ' '.join(words)
    tokens = word_tokenize(text)

    # Find bigram collocations in the text
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(tokens)
    scored = finder.score_ngrams(bigram_measures.raw_freq)

    # Sort scored bigrams by their frequency
    sorted_bigram = sorted(scored, key=lambda x: -x[1])

    # Return only the bigram phrases
    res = []
    for bigram, score in sorted_bigram:
        print(f'bigram: {bigram} - score: {score}')
        if score > 0.01: # Adjust the threshold as necessary
            res.append(bigram)

    return res 

if __name__ == "__main__":
    """TODO: https://www.nltk.org/howto/collocations.html"""
    words = ['I', 'learn',"deep", "learning",'and' ,"artificial", "intelligence",'I', 'very', 'love', "machine", "learning"]
    phrases = find_meaningful_phrases(words)
    print(phrases)
