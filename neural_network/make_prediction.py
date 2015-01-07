import os
import sys
import json
import time
import cPickle
import numpy as np
import pandas as pd
import preprocessing
from pandas import DataFrame
from neural_network.layers import NeuralNetwork
from neural_network.transformers import transform_data
from neural_network.data_iterators import TestDataIterator


with open(sys.argv[1]) as f:
    conf = json.load(f)
test_data_file_path = conf['test_data_file_path']
classifiers_paths = conf['classifiers_paths']
word_vectors_path = conf['word_vectors_path']
transformers_dir = conf['transformers_dir']
submission_file_path = conf['submission_file_path']


def load_transformers(label_name):
    with open('{}/{}.pckl'.format(transformers_dir, label_name)) as f:
        d = cPickle.load(f)
    return d['softmax_normalizers'], d['nan_replacers']


if __name__ == '__main__':
    columns = []
    predictions = []

    for label_name, label_values in preprocessing.multi_labels.iteritems():
        test_data_frame = pd.read_pickle(test_data_file_path)
        softmax_normalizers, nan_replacers = load_transformers(label_name)
        test_data_frame = transform_data(test_data_frame, softmax_normalizers, nan_replacers)
        test_data_iterator = TestDataIterator(test_data_frame, word_vectors_path)
        nn = NeuralNetwork.load(classifiers_paths[label_name])
        nn.set_test_mode()

        t = time.time()
        print label_name
        pred_matrix = np.empty(shape=(test_data_frame.shape[0], len(label_values)), dtype=np.float32)
        for i, (words_matrices, float_scalars) in enumerate(test_data_iterator):
            if i % 2500 == 0:
                print i
            pred_matrix[i] = nn.predict(words_matrices, float_scalars)
        predictions.append(pred_matrix)
        columns.extend('{}__{}'.format(label_name, label_value) for label_value in label_values)
        print time.time() - t
        print 'done'
    predictions = np.column_stack(predictions)

    dir_path = os.path.dirname(submission_file_path)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    test_data_frame = pd.read_pickle(test_data_file_path)
    DataFrame(data=predictions, index=test_data_frame.index, columns=columns).to_csv(submission_file_path)