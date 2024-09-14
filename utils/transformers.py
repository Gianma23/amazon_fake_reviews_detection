import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class SentenceEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X should be a list of numpy arrays (each array representing sentence embeddings)
        transformed = []
        for embeddings in X:
            # If embeddings is a list, convert it to a numpy array
            if isinstance(embeddings, list):
                embeddings = np.array(embeddings)

            if embeddings.size > 0:
                # Return the average embedding of the sentences
                transformed.append(np.mean(embeddings, axis=0))
            else:
                # Handle case where there are no embeddings
                transformed.append(np.zeros(embeddings.shape[1]))

        return np.array(transformed)


class DenseTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.toarray() if hasattr(X, 'toarray') else X