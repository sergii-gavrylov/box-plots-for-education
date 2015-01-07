import numpy as np
from numpy import random
from itertools import izip, count
from neural_network.data_iterators import WordVectorsContainer
from preprocessing import text_columns, float_columns, multi_labels


class InfiniteRandomDataIterator(WordVectorsContainer):
    def __init__(self, label_name, data_frame, word_vectors_path, word_dropout_prob, random_seed):
        self.label_name = label_name
        self.idx_to_label = multi_labels[label_name]
        self.label_to_idx = dict(izip(self.idx_to_label, count()))
        self.num_classes = len(multi_labels[label_name])
        self.data_frame = data_frame
        self.word_to_idx = self._get_word_to_idx(word_vectors_path)
        self.idx_to_vector = self._get_idx_to_vector(word_vectors_path)
        self.word_vector_dim = len(self.idx_to_vector[0])
        self.keep_word = lambda: self.r.choice([True, False], p=[1.-word_dropout_prob, word_dropout_prob])
        self.r = random.RandomState(seed=random_seed)
        self.indexes = self.data_frame.index.tolist()

    def get_infinite_iterator(self):
        while True:
            self.r.shuffle(self.indexes)
            for idx in self.indexes:
                label_value = self.data_frame[self.label_name][idx]
                words_matrices = {}
                for column_name in text_columns:
                    text = self.data_frame[column_name][idx]
                    text = [word for word in text if (self.keep_word() and word in self.word_to_idx)]
                    if text:
                        matrix = np.empty(shape=(len(text), self.word_vector_dim), dtype=np.float32)
                    else:
                        matrix = np.zeros(shape=(1, self.word_vector_dim), dtype=np.float32)
                    for row_idx, word in enumerate(text):
                        matrix[row_idx] = self.idx_to_vector[self.word_to_idx[word]]
                    words_matrices[column_name] = matrix
                float_scalars = {}
                for column_name in float_columns:
                    float_scalars[column_name] = np.float32(self.data_frame[column_name][idx])
                yield words_matrices, float_scalars, self.label_to_idx[label_value]