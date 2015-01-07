import h5py
import numpy as np


class ContinuousBagOfWordsTransformer(object):
    def __init__(self, word_vectors_path, text_columns):
        with h5py.File(word_vectors_path, 'r') as h5_file:
            self.word_to_idx = dict((word.decode('utf-8'), i) for i, word in enumerate(h5_file['idx_to_word'][...]))
            self.idx_to_vector = h5_file['idx_to_vector'][...]
        self.vector_dim = len(self.idx_to_vector[0])
        self.text_columns = text_columns

    def transform(self, data_frame, data_array=None):
        if data_array is not None:
            cbow = data_array
        else:
            cbow = np.zeros(shape=(data_frame.shape[0], len(self.text_columns) * self.vector_dim), dtype=np.float32)

        offset = 0
        for column_name in self.text_columns:
            for row_idx, text in enumerate(data_frame[column_name]):
                n = 0
                for word in text:
                    if word in self.word_to_idx:
                        cbow[row_idx, offset:offset+self.vector_dim] += self.idx_to_vector[self.word_to_idx[word]]
                        n += 1
                if n != 0:
                    cbow[row_idx, offset:offset+self.vector_dim] /= n
            offset += self.vector_dim
        return cbow

    def _get_array_shape(self, data_frame):
        return data_frame.shape[0], len(self.text_columns) * self.vector_dim