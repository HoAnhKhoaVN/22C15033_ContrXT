
import re
from src.text_cls.constant import VIET_STOP_WORD_PATH


def remove_vietnamese_stopwords(text: str, stopwords_file: str = VIET_STOP_WORD_PATH ) -> str:
    """
    Removes Vietnamese stopwords from a given text string.

    Args:
        text (str): The input text string to be processed.
        stopwords_file (str, optional): Path to a file containing Vietnamese stopwords, one per line. Defaults to "vietnamese-stopwords.txt".

    Returns:
        str: The text string with Vietnamese stopwords removed.

    Raises:
        FileNotFoundError: If the specified stopwords file is not found.
    """

    try:
        with open(stopwords_file, "r", encoding="utf-8") as f:
            stop_words = list(set(f.read().splitlines()))
    except FileNotFoundError:
        raise FileNotFoundError(f"Stopwords file not found: {stopwords_file}")
    
    # Create a regular expression pattern to match stop words
    pattern = r'\b(?:' + '|'.join(re.escape(word) for word in stop_words) + r')\b'
    
    print(f'Langth regrex: {len(pattern)}')

    # Use re.sub to replace stop words with an empty string
    result = re.sub(pattern, '', text)
    
    # Remove extra spaces that might result from stop word removal
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result

if __name__ == "__main__":
    text = "à à Đây là một ví dụ về văn bản tiếng Việt có chứa từ dừng."
    text = text.lower()
    # text = 'Tôi và bạn đang học Python để cải thiện kỹ năng lập trình của mình.'
    filtered_text = remove_vietnamese_stopwords(text)
    print(filtered_text)  
