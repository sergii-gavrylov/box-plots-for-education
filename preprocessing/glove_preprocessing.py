import h5py
import numpy as np


if __name__ == '__main__':
    with open('data/word_vectors/glove.6B.50d.txt') as model_txt_file:
        skipping_lines = set()
        idx_to_word = []
        for i, line in enumerate(model_txt_file):
            try:
                line = line.decode('utf-8')
                space_idx = line.find(u' ')
                idx_to_word.append(line[:space_idx].encode('utf-8'))
                dim = len(line.split()) - 1
            except UnicodeDecodeError:
                skipping_lines.add(i)

        idx_to_vector = np.empty((len(idx_to_word), dim), dtype=np.float32)
        model_txt_file.seek(0)
        line_generator = (line.decode('utf-8') for i, line in enumerate(model_txt_file) if i not in skipping_lines)
        for k, line in enumerate(line_generator):
            idx_to_vector[k] = [np.float64(number) for number in line.split(u' ')[1:]]

    with h5py.File('data/word_vectors/glove_model_50d.h5', 'w') as h5_file:
        h5_file['idx_to_vector'] = idx_to_vector
        h5_file['idx_to_word'] = idx_to_word