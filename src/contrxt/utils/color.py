import re

def visualize_text(text):
    # Regular expression to match ANSI escape codes
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    # Function to remove ANSI escape codes
    def remove_escape_codes(s):
        return ansi_escape.sub('', s)

    # Replace arrow characters with actual arrows
    text = text.replace('←', '←')

    # Print the text without ANSI escape codes
    clean_text = remove_escape_codes(text)
    print(clean_text)

if __name__ == "__main__":
    text = """
    ←[48;5;155mThe model now uses the following classification rules for this class:←[0m
    This class has 1 added classification rule.

     - ←[38;5;10mHaving←[0m ←[1m←[38;5;10mAtheist←[0m.

    ←[48;5;1mThe model is not using the following classification rules anymore:←[0m
    This class has 2 deleted classification rules, but only 1 is used to classify the 80%
    of the items.

     - ←[38;5;10mHaving←[0m ←[1m←[38;5;10mGod←[0m, and ←[1m←[38;5;10mAlso←[0m but ←[4m←[38;5;9mnot←[0m ←[1m←[38;5;9mPoliticalAtheists←[0m.

    There are no '←[48;5;220munchanged←[0m' classification rules.
    """

    visualize_text(text)
