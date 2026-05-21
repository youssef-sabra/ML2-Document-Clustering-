from nlp_clustering import features


def test_vectorize_basic():
    texts = ["This is a test.", "Another test document." ]
    X_raw, X_used, vec, svd = features.vectorize(texts, max_features=50, n_components=None)
    assert X_raw.shape[0] == 2
    assert vec is not None


def test_vectorize_with_svd():
    texts = ["Dog cat mouse.", "Cat and dog.", "Mouse loves cheese."]
    X_raw, X_used, vec, svd = features.vectorize(texts, max_features=50, n_components=2)
    # when n_components < min(n_samples, n_features) we expect an svd model and reduced array
    assert svd is not None
    assert hasattr(X_used, 'shape')
