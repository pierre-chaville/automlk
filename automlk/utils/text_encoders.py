import string


TABLE_TRANS = str.maketrans({key: ' ' for key in string.punctuation})


def clean_text(s, first_words):
    # transforms sentence for word processing into a list a words (only first words
    words = s.lower().translate(TABLE_TRANS).split()
    if first_words != 0:
        words = words[:first_words]
    return " ".join(words)
