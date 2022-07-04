import string

def clean_text(text: str) -> str:
    return text.translate(str.maketrans('', '', string.punctuation)).lower()