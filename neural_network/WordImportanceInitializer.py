import numpy as np
from numpy import random
from collections import Counter
from preprocessing import multi_labels
from data_iterators import WordVectorsContainer


class WordImportanceInitializer(WordVectorsContainer):
    def __init__(self, data_frame, label_name, word_vectors_path, random_seed, logger):
        self.data_frame = data_frame
        self.label_name = label_name
        self.word_to_idx = self._get_word_to_idx(word_vectors_path)
        self.idx_to_vector = self._get_idx_to_vector(word_vectors_path)
        self.word_vector_dim = len(self.idx_to_vector[0])
        self.r = random.RandomState(seed=random_seed)
        self.logger = logger

    def get_importances_vectors_init_values(self, text_column_name):
        importance_vector_init = {}
        self.logger.info('{} {} {}'.format(20*'=--=', text_column_name, 20*'=--='))
        for label_value in multi_labels[self.label_name]:
            counter = Counter()
            index = self.data_frame.index[self.data_frame[self.label_name] == label_value]
            data_frame = self.data_frame.loc[index]
            for text in data_frame[text_column_name]:
                counter.update(text)
            importance_vector_value = np.zeros(shape=(self.word_vector_dim, ), dtype=np.float64)
            total_freq = 0
            self.logger.info('{} {} {}'.format(20*'=', label_value, 20*'='))
            for word, freq in counter.most_common():
                self.logger.info((word, freq))
                if word in self.word_to_idx:
                    importance_vector_value += freq * self.idx_to_vector[self.word_to_idx[word]]
                    total_freq += freq
            if total_freq != 0:
                importance_vector_value /= total_freq
                importance_vector_init[label_value] = importance_vector_value.astype(dtype=np.float32)
        return importance_vector_init


class SerializationWordImportanceInitializer(object):
    def __init__(self, words_layers):
        self.importance_vector_init = {}
        for words_layer in words_layers:
            self.importance_vector_init[words_layer.layer_name] = words_layer.importance_matrix.get_value()

    def get_importances_vectors_init_values(self, text_column_name):
        return self.importance_vector_init[text_column_name]