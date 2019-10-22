import re

from .numbers import normalize_numbers

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')


def expand_numbers(text):
    return normalize_numbers(text)


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def chinese_cleaners(text):
    text = expand_numbers(text)
    text = collapse_whitespace(text)
    return text
