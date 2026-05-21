from nlp_clustering.eda_utils import DEFAULT_20NEWSGROUPS_CATEGORIES, load_20newsgroups_dataset


def test_load_20newsgroups_dataset_returns_documents():
    texts, labels, df = load_20newsgroups_dataset(
        categories=DEFAULT_20NEWSGROUPS_CATEGORIES,
        max_docs_per_category=5,
    )

    assert len(texts) == len(labels) == len(df)
    assert len(texts) > 0
    assert set(labels).issubset(set(DEFAULT_20NEWSGROUPS_CATEGORIES))
    assert {'text', 'category', 'document_id', 'source_file'}.issubset(df.columns)