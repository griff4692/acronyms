import re
import string


def remove_punctuation(str):
    punctuation_regex = re.compile('[%s]' % re.escape(string.punctuation))
    return punctuation_regex.sub('', str)

def standardize_upper(str):
    return remove_punctuation(str).strip().upper()


def standardize_lower(str):
    return remove_punctuation(str).strip().lower()

