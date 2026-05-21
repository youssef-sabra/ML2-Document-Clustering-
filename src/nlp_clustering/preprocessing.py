import re
from typing import List, Iterable, Optional

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize


_URL_RE = re.compile(r"http[s]?://\S+|www\.\S+", flags=re.IGNORECASE)
_EMAIL_RE = re.compile(r"\S+@\S+")
_DATE_RE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})\b",
    flags=re.IGNORECASE,
)
_MIXED_TOKEN_RE = re.compile(r"\b(?=\w*\d)\w+\b")
_NUM_RE = re.compile(r"\b\d+\b")
_UNDERSCORE_RE = re.compile(r"_+")
_SYMBOL_RUN_RE = re.compile(r"[=\-*#@~`^]{2,}|\.{2,}|/\\{2,}")
_PUNC_RE = re.compile(r"[^\w\s]")
_NLTK_RESOURCES_READY = False
_EXTRA_NEWSGROUP_STOPWORDS = {
    'also', 'could', 'did', 'does', 'dont', 'even', 'get', 'going', 'got',
    'know', 'like', 'make', 'maybe', 'one', 'people', 'post', 'right', 'say',
    'thing', 'think', 'time', 'use', 'used', 'want', 'way', 'would', 'yes',
}


def _ensure_nltk_resources():
    """Download required NLTK resources if missing."""
    global _NLTK_RESOURCES_READY
    if _NLTK_RESOURCES_READY:
        return
    resources = [
        ("punkt", "tokenizers/punkt"),
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
        ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
        ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
    ]
    for name, path in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)
    _NLTK_RESOURCES_READY = True


def _get_wordnet_pos(treebank_tag: str):
    """Map Treebank POS tags to WordNet POS names for lemmatization."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    if treebank_tag.startswith('V'):
        return wordnet.VERB
    if treebank_tag.startswith('N'):
        return wordnet.NOUN
    if treebank_tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN


def clean_text(text: str,
               lowercase: bool = True,
               remove_urls: bool = True,
               remove_emails: bool = True,
               remove_dates: bool = True,
               remove_numbers: bool = True,
               remove_mixed_tokens: bool = True,
               remove_repeated_symbols: bool = True,
               remove_punctuation: bool = True,
               collapse_whitespace: bool = True) -> str:
    """Basic string-level cleaning.

    Steps:
    - lowercase (optional)
    - remove URLs
    - remove emails, dates, and mixed alphanumeric tokens
    - remove numbers
    - remove repeated underscores and symbol runs
    - remove punctuation
    - collapse repeated whitespace
    """
    if not isinstance(text, str):
        text = str(text)
    if lowercase:
        text = text.lower()
    if remove_urls:
        text = _URL_RE.sub(' ', text)
    if remove_emails:
        text = _EMAIL_RE.sub(' ', text)
    if remove_dates:
        text = _DATE_RE.sub(' ', text)
    if remove_mixed_tokens:
        text = _MIXED_TOKEN_RE.sub(' ', text)
    if remove_numbers:
        text = _NUM_RE.sub(' ', text)
    if remove_repeated_symbols:
        text = _UNDERSCORE_RE.sub(' ', text)
        text = _SYMBOL_RUN_RE.sub(' ', text)
    if remove_punctuation:
        text = _PUNC_RE.sub(' ', text)
    if collapse_whitespace:
        text = re.sub(r"\s+", ' ', text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Tokenize using NLTK's word_tokenize (requires `punkt`)."""
    _ensure_nltk_resources()
    return word_tokenize(text)


def remove_stopwords(tokens: Iterable[str], language: str = 'english') -> List[str]:
    """Remove stopwords using NLTK stopword list."""
    _ensure_nltk_resources()
    stops = set(stopwords.words(language)) | _EXTRA_NEWSGROUP_STOPWORDS
    return [t for t in tokens if t.lower() not in stops]


def lemmatize_tokens(tokens: Iterable[str]) -> List[str]:
    """Lemmatize tokens using NLTK WordNetLemmatizer with POS tags for accuracy."""
    _ensure_nltk_resources()
    lemmatizer = WordNetLemmatizer()
    tagged = pos_tag(list(tokens))
    lemmas = [lemmatizer.lemmatize(tok, _get_wordnet_pos(tag)) for tok, tag in tagged]
    return lemmas


def preprocess_texts(texts: List[str],
                     lowercase: bool = True,
                     remove_urls: bool = True,
                     remove_numbers: bool = True,
                     remove_punctuation: bool = True,
                     remove_stopwords_flag: bool = True,
                     lemmatize: bool = True,
                     join_tokens: bool = True) -> List[str]:
    """Run a configurable preprocessing pipeline over a list of texts.

    Returns a list of processed strings (joined tokens) by default. Set `join_tokens=False`
    to get token lists instead.
    """
    out = []
    for text in texts:
        s = clean_text(text,
                       lowercase=lowercase,
                       remove_urls=remove_urls,
                       remove_numbers=remove_numbers,
                       remove_punctuation=remove_punctuation)
        toks = tokenize(s)
        toks = [tok for tok in toks if tok.isalpha() and len(tok) >= 3]
        if remove_stopwords_flag:
            toks = remove_stopwords(toks)
        if lemmatize:
            toks = lemmatize_tokens(toks)
        toks = [tok for tok in toks if tok.isalpha() and len(tok) >= 3]
        if remove_stopwords_flag:
            toks = remove_stopwords(toks)
        if join_tokens:
            out.append(' '.join(toks))
        else:
            out.append(toks)
    return out


if __name__ == '__main__':
    # quick smoke test
    samples = [
        'This is a test: visiting https://example.com and numbers 12345.',
        'Running NLP preprocessing, like lemmatization and tokenization!'
    ]
    print(preprocess_texts(samples) )
