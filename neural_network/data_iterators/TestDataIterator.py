import numpy as np
from preprocessing import text_columns, float_columns
from neural_network.data_iterators import WordVectorsContainer


class TestDataIterator(WordVectorsContainer):
    def __init__(self, data_frame, word_vectors_path):
        self.data_frame = data_frame
        self.word_to_idx = self._get_word_to_idx(word_vectors_path)
        self.idx_to_vector = self._get_idx_to_vector(word_vectors_path)
        self.word_vector_dim = len(self.idx_to_vector[0])

    def __iter__(self):
        for idx in self.data_frame.index:
            words_matrices = {}
            for column_name in text_columns:
                text = self.data_frame[column_name][idx]
                text = [word for word in text if word in self.word_to_idx]
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
            yield words_matrices, float_scalars