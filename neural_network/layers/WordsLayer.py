import theano
import numpy as np
import theano.tensor as T


class WordsLayer(object):
    def __init__(self, layer_name, importance_vector_init):
        self.layer_name = layer_name

        if type(importance_vector_init) == dict:
            n_row = importance_vector_init.itervalues().next().shape[0]
            n_col = len(importance_vector_init)
            importance_matrix_init = np.empty((n_row, n_col), dtype=np.float32)
            for i, (_, importance_vector_value) in enumerate(importance_vector_init.iteritems()):
                importance_matrix_init[:, i] = importance_vector_value
        else:
            importance_matrix_init = importance_vector_init

        self.importance_matrix = theano.shared(value=importance_matrix_init,
                                               name='{}_importance_matrix'.format(layer_name),
                                               borrow=True)
        self.output_vector_size = importance_matrix_init.shape[1] * 2

    def get_forward_pass_expr(self, words_matrix):
        words_vectors_norm = T.sqrt(T.sum(T.sqr(words_matrix), axis=1, keepdims=True))
        _words_vectors_norm = T.switch(T.eq(words_vectors_norm, 0.), 1.0, words_vectors_norm)
        normalized_words_matrix = words_matrix / _words_vectors_norm
        normalized_importance_matrix = self.importance_matrix / T.sqrt(T.sum(T.sqr(self.importance_matrix), axis=0))
        words_importances = T.sqr(T.dot(normalized_words_matrix, normalized_importance_matrix))
        return T.concatenate([T.mean(words_importances, axis=0), T.max(words_importances, axis=0)])

    def get_parameters(self):
        return [self.importance_matrix]