import re
from typing import Text

def is_valid_string(input_string: Text) -> bool:
    """
    Checks if the input string consists entirely of letters (uppercase or lowercase), 
    numbers, hyphens, and spaces.

    Args:
        input_string (str): The string to be checked.

    Returns:
        bool: True if the string is valid, otherwise False.
    """
    # Define the regular expression pattern
    pattern = r'^[a-zA-Z0-9\- ]+$'
    
    # Use re.fullmatch to check if the entire string matches the pattern
    match = re.fullmatch(pattern, input_string)
    
    # Return True if there is a match, otherwise False
    return match is not None

if __name__ == "__main__":
    print(is_valid_string("Hello World 123"))  # Should return True
    print(is_valid_string("Hello@World!"))     # Should return False
    print(is_valid_string("Just-Text"))        # Should return True
    print(is_valid_string("12345-6789"))       # Should return True
    print(is_valid_string("Invalid_Chars!"))   # Should return False

