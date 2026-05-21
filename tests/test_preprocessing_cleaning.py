from nlp_clustering.preprocessing import clean_text, preprocess_texts


def test_clean_text_removes_metadata_noise():
    text = 'Email test@example.com on 1993apr10 at 13h with 24x ___ and 32bis.'
    cleaned = clean_text(text)

    assert 'test@example.com' not in cleaned
    assert '1993apr10' not in cleaned
    assert '13h' not in cleaned
    assert '24x' not in cleaned
    assert '32bis' not in cleaned
    assert '___' not in cleaned


def test_preprocess_texts_keeps_only_words():
    out = preprocess_texts(['Graphics 1993apr10 24x example text!'])
    assert out[0].split() == ['graphic', 'example', 'text'] or out[0].split() == ['graphics', 'example', 'text']