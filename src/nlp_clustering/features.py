"""TF-IDF feature extraction utilities.

Provides reusable functions to build and apply TF-IDF vectorizers with
configurable n-gram ranges, vocabulary size and document-frequency thresholds.

Also includes simple cosine-similarity helpers.

Notes / explanations:
- Sparse matrices: TF-IDF transforms produce large, mostly-empty matrices
  (documents x vocabulary). These are stored in sparse formats (scipy.sparse)
  which store only nonzero entries to save memory and speed linear-algebra ops.
- Vectorization: converting raw text into numeric feature vectors (here TF-IDF)
  that machine learning models can consume. TF-IDF downweights very frequent
  terms and upweights terms that are distinctive for a document.
- Cosine similarity: measures the cosine of the angle between two vectors; for
  TF-IDF vectors it is a common way to measure document similarity while being
  invariant to vector length.
"""

from typing import List, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Optional
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer


def build_tfidf_vectorizer(max_features: int = 20000,
                           ngram_range: Tuple[int, int] = (1, 1),
                           min_df=1,
                           max_df=1.0,
                           stop_words: str = 'english') -> TfidfVectorizer:
    """Create a configured TfidfVectorizer (not yet fitted).

    Args:
        max_features: maximum vocabulary size (keeps top features by TF-IDF).
        ngram_range: tuple (min_n, max_n) for n-grams (e.g., (1,2) for unigrams+biagrams).
        min_df: ignore terms that appear in fewer than `min_df` documents (int) or fraction (float).
        max_df: ignore terms that appear in more than `max_df` documents (int) or fraction (float).
        stop_words: stopword handling passed to TfidfVectorizer.

    Returns:
        Unfitted `TfidfVectorizer`.
    """
    return TfidfVectorizer(max_features=max_features,
                           ngram_range=ngram_range,
                           min_df=min_df,
                           max_df=max_df,
                           stop_words=stop_words)


def fit_transform_tfidf(texts: List[str],
                        max_features: int = 20000,
                        ngram_range: Tuple[int, int] = (1, 1),
                        min_df=1,
                        max_df=1.0,
                        stop_words: str = 'english'):
    """Fit a TF-IDF vectorizer and transform texts.

    Returns (vectorizer, X) where X is a scipy sparse matrix (documents x features).
    """
    vec = build_tfidf_vectorizer(max_features=max_features,
                                 ngram_range=ngram_range,
                                 min_df=min_df,
                                 max_df=max_df,
                                 stop_words=stop_words)
    X = vec.fit_transform(texts)
    return vec, X


def vectorize_ngrams(texts: List[str],
                     ngram_ranges: List[Tuple[int, int]] = [(1, 1), (1, 2)],
                     max_features: int = 20000,
                     min_df=1,
                     max_df=1.0,
                     stop_words: str = 'english') -> Dict[Tuple[int, int], Tuple[TfidfVectorizer, object]]:
    """Create TF-IDF representations for multiple n-gram ranges.

    Returns a dict mapping ngram_range -> (vectorizer, X).
    """
    out = {}
    for rng in ngram_ranges:
        vec, X = fit_transform_tfidf(texts, max_features=max_features, ngram_range=rng,
                                     min_df=min_df, max_df=max_df, stop_words=stop_words)
        out[rng] = (vec, X)
    return out


def top_terms(vectorizer: TfidfVectorizer, top_n: int = 20):
    """Return the top `top_n` terms from the vectorizer vocabulary by idf (most distinctive).

    This is a simple utility to inspect the vocabulary; it helps compare unigram vs bigram
    TF-IDF feature sets.
    """
    idf = getattr(vectorizer, 'idf_', None)
    if idf is None:
        return []
    terms = vectorizer.get_feature_names_out()
    order = np.argsort(idf)[::-1]
    top_idx = order[:top_n]
    return [terms[i] for i in top_idx]


def cosine_similarity_matrix(X, dense: bool = False):
    """Compute cosine similarity between rows of X.

    Args:
        X: sparse or dense document-term matrix (documents x features).
        dense: if True, return a dense numpy array; otherwise return the result (may still be dense).

    Warning: the full pairwise similarity matrix is O(n^2) memory; for large corpora use
    approximate or a top-k similarity approach.
    """
    S = cosine_similarity(X)
    if dense:
        return np.array(S)
    return S


def build_vectorizer(max_features: int = 20000,
                     ngram_range: Tuple[int, int] = (1, 1),
                     min_df=1,
                     max_df=1.0,
                     stop_words: str = 'english') -> TfidfVectorizer:
    """Compatibility wrapper: build a TfidfVectorizer with common defaults."""
    return build_tfidf_vectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        stop_words=stop_words,
    )


def vectorize(texts: List[str],
              max_features: int = 20000,
              n_components: Optional[int] = None,
              ngram_range: Tuple[int, int] = (1, 1),
              min_df=1,
              max_df=1.0,
              stop_words: str = 'english',
              random_state: int = 42):
    """Return TF-IDF matrix and optionally SVD-reduced matrix.

    Returns: (X_raw, X_used, vectorizer, svd_model)
    """
    vectorizer = build_vectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        stop_words=stop_words,
    )
    X_raw = vectorizer.fit_transform(texts)
    svd_model = None
    if n_components is not None and n_components > 0 and n_components < min(X_raw.shape[0], X_raw.shape[1]):
        svd_model = TruncatedSVD(n_components=n_components, random_state=random_state)
        X_reduced = svd_model.fit_transform(X_raw)
        return X_raw, X_reduced, vectorizer, svd_model
    return X_raw, X_raw, vectorizer, svd_model


def vectorize_for_clustering(texts: List[str],
                             max_features: int = 20000,
                             n_components: int = 100,
                             ngram_range: Tuple[int, int] = (1, 1),
                             min_df=1,
                             max_df=1.0,
                             stop_words: str = 'english',
                             random_state: int = 42):
    """Build a simple TF-IDF -> TruncatedSVD -> Normalizer clustering matrix.

    Returns (X_raw, X_reduced, X_cluster, vectorizer, svd_model, normalizer).
    """
    vectorizer = build_vectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        stop_words=stop_words,
    )
    X_raw = vectorizer.fit_transform(texts)

    svd_model = None
    X_reduced = X_raw
    if n_components is not None and n_components > 0 and X_raw.shape[1] > 1:
        n_components = min(n_components, X_raw.shape[0] - 1, X_raw.shape[1] - 1)
        if n_components >= 2:
            svd_model = TruncatedSVD(n_components=n_components, random_state=random_state)
            X_reduced = svd_model.fit_transform(X_raw)

    normalizer = Normalizer(copy=False)
    X_cluster = normalizer.fit_transform(X_reduced)
    return X_raw, X_reduced, X_cluster, vectorizer, svd_model, normalizer
