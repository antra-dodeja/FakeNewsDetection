"""
utils.py — Text Preprocessing Utilities
=========================================
Helper functions for cleaning and preparing raw news text
before it is fed into the TF-IDF vectorizer.

No external NLP libraries (NLTK, spaCy) are required — this
keeps the dependency list minimal and the code easy to understand.
"""

import re
import string


# Simple but comprehensive stopword list (no NLTK needed)
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "was", "are", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall",
    "this", "that", "these", "those", "it", "its", "he", "she",
    "they", "we", "you", "i", "me", "him", "her", "us", "them",
    "my", "your", "his", "our", "their", "what", "which", "who",
    "when", "where", "why", "how", "not", "no", "so", "if", "as",
    "up", "out", "about", "into", "than", "more", "also", "just",
    "said", "say", "says", "after", "before", "while", "there",
    "one", "two", "new", "can", "now", "then", "through", "over",
    "even", "back", "any", "good", "way", "well", "since", "most",
    "however", "both", "each", "between", "during", "without", "again",
    "further", "once", "only", "own", "same", "other", "very", "per",
}


def preprocess_text(text: str) -> str:
    """
    Clean and normalise raw news text for TF-IDF vectorisation.

    Processing pipeline
    -------------------
    1. Lowercase all characters
    2. Remove URLs  (http:// and www.)
    3. Remove email addresses
    4. Remove HTML tags (if any)
    5. Remove punctuation
    6. Remove standalone digits
    7. Collapse extra whitespace
    8. Remove stopwords and very short tokens (≤2 chars)

    Parameters
    ----------
    text : str
        Raw news article text or headline.

    Returns
    -------
    str
        Cleaned, stopword-filtered text ready for vectorisation.

    Examples
    --------
    >>> preprocess_text("BREAKING: Scientists at MIT say 'AI cures cancer'!")
    'breaking scientists mit ai cures cancer'
    """
    if not isinstance(text, str):
        text = str(text)

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # 3. Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", "", text)

    # 4. Strip HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # 5. Remove punctuation (keep only letters and spaces)
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 6. Remove standalone digits / number-only tokens
    text = re.sub(r"\b\d+\b", "", text)

    # 7. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 8. Filter stopwords and short tokens
    tokens = [
        word for word in text.split()
        if word not in STOPWORDS and len(word) > 2
    ]

    return " ".join(tokens)


def get_word_count(text: str) -> int:
    """Return the number of words in raw text."""
    return len(text.split()) if text.strip() else 0


def truncate_preview(text: str, max_chars: int = 80) -> str:
    """Return a truncated preview string for display purposes."""
    text = text.strip()
    return text[:max_chars] + "..." if len(text) > max_chars else text
