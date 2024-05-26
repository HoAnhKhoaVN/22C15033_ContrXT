import re

def remove_non_vietnamese_letters(text: str) -> str:
    """
    Removes non-Vietnamese letters (uppercase or lowercase) and spaces from the given text.

    Args:
        text (str): The input text string.

    Returns:
        str: The modified text with non-Vietnamese letters and spaces removed.
    """

    pattern = r'[^a-zA-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơưẠ-ỹ _]'  # Matches any characters not in the Vietnamese range, plus spaces.

    modified_text = re.sub(pattern, '', text)

    return modified_text

if __name__ == "__main__":
    TEXT = """Hàu là con vật nhuyễn thể thuộc dòng "nghêu sò ốc hến" sống ở ven biển nước mặn."""
    print(remove_non_vietnamese_letters(TEXT))