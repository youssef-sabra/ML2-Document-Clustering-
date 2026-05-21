from nlp_clustering.preprocessing import preprocess_texts


def test_preprocess_basic():
    texts = ['Hello, WORLD!!!', 'NLP-testing 123']
    out = preprocess_texts(texts)
    assert out[0].islower()
    assert 'nlp' in out[1]
