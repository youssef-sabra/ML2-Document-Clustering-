"""Simple script to download required NLTK resources for the project."""
import nltk

REQUIRED = [
    'punkt',
    'stopwords',
    'wordnet',
    'averaged_perceptron_tagger'
]

if __name__ == '__main__':
    for r in REQUIRED:
        try:
            nltk.data.find(r)
            print(f'{r} already available')
        except LookupError:
            print(f'Downloading {r}...')
            nltk.download(r)
    print('NLTK resource download complete.')
