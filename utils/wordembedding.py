import numpy as np
from nltk import sent_tokenize
from utils.text_preprocess import LemmaTokenizer


# review_text: str
# model: trained word2vec model
# return: list of numpy arrays (average word embeddings for each sentence)
def generate_word_embeddings(review_text, model):
    sentences = sent_tokenize(review_text)
    sentence_embeddings = [_average_word_embeddings(sentence, model) for sentence in sentences]
    return sentence_embeddings


def _average_word_embeddings(sentence, model):
    words = LemmaTokenizer()(sentence)
    word_vectors = [model.wv[word] for word in words if word in model.wv]

    if len(word_vectors) == 0:
        return np.zeros(model.vector_size)

    return np.mean(word_vectors, axis=0)


