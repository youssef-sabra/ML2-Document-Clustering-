from __future__ import annotations

import os
import re
from types import SimpleNamespace
from pathlib import Path
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING
from collections import Counter
import logging

try:
    from wordcloud import WordCloud
    _HAS_WORDCLOUD = True
except Exception:
    _HAS_WORDCLOUD = False

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_20NEWSGROUPS_CATEGORIES = [
    'alt.atheism',
    'comp.graphics',
    'sci.space',
    'talk.religion.misc',
]

if TYPE_CHECKING:
    import pandas as pd


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def resolve_data_path(filename: str, data_dir: Optional[str] = None) -> Path:
    """Resolve a dataset file from data/raw first, then data/ for backward compatibility."""
    candidates = []
    if data_dir:
        candidates.append(Path(data_dir) / filename)
    candidates.extend([
        PROJECT_ROOT / 'data' / 'raw' / filename,
        PROJECT_ROOT / 'data' / filename,
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'Could not find {filename} in data/raw or data/')


def load_dataset(filename: str = 'people_wiki.csv', data_dir: Optional[str] = None) -> Tuple[List[str], List[str], 'pd.DataFrame']:
    """Load the local People Wikipedia CSV from data/raw.

    Returns (texts, names, dataframe).
    """
    import pandas as pd

    path = resolve_data_path(filename, data_dir=data_dir)
    df = pd.read_csv(path)

    text_col = 'text' if 'text' in df.columns else df.columns[-1]
    name_col = 'name' if 'name' in df.columns else (df.columns[1] if len(df.columns) > 1 else None)

    texts = df[text_col].fillna('').astype(str).tolist()
    names = df[name_col].fillna('').astype(str).tolist() if name_col else [f'doc_{i}' for i in range(len(df))]
    return texts, names, df


def fetch_20newsgroups(
    categories: Optional[List[str]] = None,
    data_home: Optional[str] = None,
    remove: tuple[str, ...] = ('headers', 'footers', 'quotes'),
):
    """Fetch 20 Newsgroups with header/footer/quote removal.

    Tries scikit-learn first, then falls back to the local corpus if the built-in
    downloader is not available.
    """
    try:
        from sklearn.datasets import fetch_20newsgroups as sklearn_fetch_20newsgroups

        return sklearn_fetch_20newsgroups(
            subset='all',
            categories=categories,
            remove=remove,
            data_home=data_home,
            download_if_missing=False,
        )
    except Exception:
        texts, labels, df = _load_local_20newsgroups_dataset(categories=categories)
        target_names = categories if categories else sorted(df['category'].unique().tolist())
        target_lookup = {name: idx for idx, name in enumerate(target_names)}
        targets = [target_lookup.get(label, -1) for label in labels]
        return SimpleNamespace(data=texts, target=targets, target_names=target_names)


def _split_newsgroup_documents(raw_text: str) -> List[str]:
    documents: List[str] = []
    current: List[str] = []

    for line in raw_text.splitlines():
        if line.startswith('Newsgroup:') and current:
            block = '\n'.join(current).strip()
            if block:
                documents.append(block)
            current = [line]
        else:
            if current or line.strip():
                current.append(line)

    if current:
        block = '\n'.join(current).strip()
        if block:
            documents.append(block)

    return documents


def _parse_newsgroup_document(block: str, fallback_category: str, fallback_index: int) -> Dict[str, str]:
    lines = block.splitlines()
    category = fallback_category
    document_id = f'{fallback_category}_{fallback_index}'
    body_start = 0

    for idx, line in enumerate(lines[:10]):
        if line.startswith('Newsgroup:'):
            category = line.split(':', 1)[1].strip() or fallback_category
        elif line.startswith('document_id:'):
            document_id = line.split(':', 1)[1].strip() or document_id

    for idx, line in enumerate(lines):
        if line.strip() == '':
            body_start = idx + 1
            break

    body_lines = []
    for line in lines[body_start:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('>'):
            continue
        if stripped.startswith('--') or stripped.startswith('____'):
            continue
        body_lines.append(line)

    text = '\n'.join(body_lines).strip() if body_lines else block.strip()
    return {
        'text': text,
        'category': category,
        'document_id': document_id,
    }


def _load_local_20newsgroups_dataset(categories: Optional[List[str]] = None) -> Tuple[List[str], List[str], 'pd.DataFrame']:
    import pandas as pd

    corpus_dir = PROJECT_ROOT / 'data' / '20-newsgroups'
    if not corpus_dir.exists():
        raise FileNotFoundError(f'Could not find 20 Newsgroups data directory: {corpus_dir}')

    category_filter = set(categories) if categories else None
    rows: List[Dict[str, str]] = []

    for file_path in sorted(corpus_dir.glob('*.txt')):
        category = file_path.stem
        if category_filter and category not in category_filter:
            continue

        raw_text = file_path.read_text(encoding='utf-8', errors='ignore')
        blocks = _split_newsgroup_documents(raw_text)
        for idx, block in enumerate(blocks):
            parsed = _parse_newsgroup_document(block, fallback_category=category, fallback_index=idx)
            if category_filter and parsed['category'] not in category_filter:
                continue
            parsed['source_file'] = file_path.name
            rows.append(parsed)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f'No 20 Newsgroups documents were found in {corpus_dir}')

    texts = df['text'].fillna('').astype(str).tolist()
    labels = df['category'].fillna('').astype(str).tolist()
    return texts, labels, df


def load_20newsgroups_dataset(
    data_dir: Optional[str] = None,
    categories: Optional[List[str]] = None,
    max_docs_per_category: Optional[int] = 250,
) -> Tuple[List[str], List[str], 'pd.DataFrame']:
    """Load the local 20 Newsgroups corpus stored under data/20-newsgroups.

    Returns (texts, labels, dataframe).
    """
    import pandas as pd

    if data_dir:
        dataset_texts, labels, df = _load_local_20newsgroups_dataset(categories=categories)
    else:
        fetched = fetch_20newsgroups(categories=categories)
        dataset_texts = list(fetched.data)
        labels = [fetched.target_names[target] if target >= 0 else 'unknown' for target in fetched.target]
        df = pd.DataFrame(
            {
                'text': dataset_texts,
                'category': labels,
                'document_id': [f'{label}_{idx}' for idx, label in enumerate(labels)],
                'source_file': 'sklearn.fetch_20newsgroups',
            }
        )

    if max_docs_per_category is not None:
        df = (
            df.groupby('category', group_keys=False)
            .head(max_docs_per_category)
            .reset_index(drop=True)
        )

    texts = df['text'].fillna('').astype(str).tolist()
    labels = df['category'].fillna('').astype(str).tolist()
    return texts, labels, df


def to_dataframe(texts: List[str], labels: Optional[List[int]] = None, source: str = 'unknown') -> 'pd.DataFrame':
    import pandas as pd

    df = pd.DataFrame({'text': texts})
    if labels is not None:
        df['label'] = labels
    df['source'] = source
    return df


def missing_stats(df: 'pd.DataFrame') -> 'pd.DataFrame':
    """Return missing value counts and percentages per column."""
    import pandas as pd

    total = len(df)
    miss = df.isnull().sum()
    pct = miss / total * 100
    out = pd.DataFrame({'missing_count': miss, 'missing_percent': pct})
    return out


def text_length_stats(texts: List[str]) -> Dict[str, float]:
    import numpy as np

    lens = [len(t) for t in texts]
    return {
        'count': len(lens),
        'min': int(np.min(lens)) if lens else 0,
        'max': int(np.max(lens)) if lens else 0,
        'median': float(np.median(lens)) if lens else 0.0,
        'mean': float(np.mean(lens)) if lens else 0.0,
        'std': float(np.std(lens)) if lens else 0.0,
    }


def tokenize(text: str, lower: bool = True, remove_stopwords: bool = True) -> List[str]:
    if lower:
        text = text.lower()
    tokens = word_tokenize(text)
    if remove_stopwords:
        try:
            stops = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            stops = set(stopwords.words('english'))
        tokens = [t for t in tokens if t.isalpha() and t not in stops]
    else:
        tokens = [t for t in tokens if t.isalpha()]
    return tokens


def compute_word_freqs(texts: List[str], top_n: int = 30) -> List[Tuple[str, int]]:
    all_tokens = []
    for t in texts:
        toks = tokenize(t)
        all_tokens.extend(toks)
    c = Counter(all_tokens)
    return c.most_common(top_n)


def plot_length_distribution(texts: List[str], out_path: Optional[str] = None, bins: int = 50, title: Optional[str] = None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    lens = [len(t) for t in texts]
    plt.figure(figsize=(8, 4))
    sns.histplot(lens, bins=bins, kde=True)
    if title:
        plt.title(title)
    else:
        plt.title('Document length distribution (characters)')
    plt.xlabel('Length')
    if out_path:
        _ensure_dir(out_path)
        plt.savefig(out_path, bbox_inches='tight')
    plt.show()


def plot_top_words(freqs: List[Tuple[str, int]], out_path: Optional[str] = None, top_n: int = 20, title: Optional[str] = None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    words, counts = zip(*freqs[:top_n]) if freqs else ([], [])
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(counts)[::-1], y=list(words)[::-1])
    if title:
        plt.title(title)
    else:
        plt.title('Top words')
    if out_path:
        _ensure_dir(out_path)
        plt.savefig(out_path, bbox_inches='tight')
    plt.show()


def plot_wordcloud_from_freqs(freqs: List[Tuple[str, int]], out_path: Optional[str] = None, width: int = 800, height: int = 400, title: Optional[str] = None):
    if not _HAS_WORDCLOUD:
        raise RuntimeError('wordcloud package not available. Install with `pip install wordcloud`')
    import matplotlib.pyplot as plt

    freq_dict = dict(freqs)
    wc = WordCloud(width=width, height=height, background_color='white')
    img = wc.generate_from_frequencies(freq_dict)
    plt.figure(figsize=(12, 6))
    plt.imshow(img, interpolation='bilinear')
    plt.axis('off')
    if title:
        # Wordclouds typically have no axes, but we can optionally place a title above
        plt.title(title)
    if out_path:
        _ensure_dir(out_path)
        img.to_file(out_path)
    plt.show()
