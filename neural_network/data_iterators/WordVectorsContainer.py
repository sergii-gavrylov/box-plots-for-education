import h5py


class WordVectorsContainer(object):
    __word_vectors = {}

    @staticmethod
    def __load_word_vectors(word_vectors_path):
        with h5py.File(word_vectors_path, 'r') as h5_file:
            word_to_idx = dict((word.decode('utf-8'), i) for i, word in enumerate(h5_file['idx_to_word'][...]))
            idx_to_vector = h5_file['idx_to_vector'][...]
        WordVectorsContainer.__word_vectors[word_vectors_path] = {'word_to_idx': word_to_idx,
                                                                  'idx_to_vector': idx_to_vector}
        return word_to_idx, idx_to_vector

    def _get_word_to_idx(self, word_vectors_path):
        if word_vectors_path not in WordVectorsContainer.__word_vectors:
            self.__load_word_vectors(word_vectors_path)
        return WordVectorsContainer.__word_vectors[word_vectors_path]['word_to_idx']

    def _get_idx_to_vector(self, word_vectors_path):
        if word_vectors_path not in WordVectorsContainer.__word_vectors:
            self.__load_word_vectors(word_vectors_path)
        return WordVectorsContainer.__word_vectors[word_vectors_path]['idx_to_vector']