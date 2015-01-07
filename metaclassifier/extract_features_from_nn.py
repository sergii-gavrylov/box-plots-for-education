import os
import sys
import json
import time
import cPickle
import hickle
import numpy as np
import pandas as pd
import preprocessing
from collections import defaultdict
from neural_network.layers import NeuralNetwork
from neural_network.transformers import transform_data
from neural_network.data_iterators import ValidationDataIterator, TestDataIterator


with open(sys.argv[1]) as f:
    conf = json.load(f)
data_file_path = conf['data_file_path']
transformers_dir = conf['transformers_dir']
word_vectors_path = conf['word_vectors_path']
classifiers_paths = conf['classifiers_paths']
nan_substitute = conf['nan_substitute']
features_file_path = conf['features_file_path']


if __name__ == '__main__':
    features = []
    data_frame = pd.read_pickle(data_file_path['test'])
    index = data_frame.index.tolist()
    fte_features = data_frame['FTE'].values.copy()
    total_features = data_frame['Total'].values.copy()
    fte_features[np.isnan(fte_features)] = nan_substitute['FTE']
    total_features[np.isnan(total_features)] = nan_substitute['Total']
    features.append(fte_features)
    features.append(total_features)


    for label_name, label_values in preprocessing.multi_labels.iteritems():
        data_frame = pd.read_pickle(data_file_path['test'])
        with open('{}/{}.pckl'.format(transformers_dir, label_name)) as f:
            d = cPickle.load(f)
        data_frame = transform_data(data_frame, d['softmax_normalizers'], d['nan_replacers'])
        nn = NeuralNetwork.load(classifiers_paths[label_name])
        nn.set_test_mode()
        test_data_iterator = TestDataIterator(data_frame, word_vectors_path)

        t = time.time()
        print 'Start extracting features for label {}'.format(label_name)
        prediction_matrix = np.empty(shape=(data_frame.shape[0], len(label_values)), dtype=np.float32)
        text_features_n_col = sum(words_layer.output_vector_size for words_layer in nn.words_layers)
        text_features = np.empty(shape=(data_frame.shape[0], text_features_n_col), dtype=np.float32)

        for i, (words_matrices, float_scalars) in enumerate(test_data_iterator):
            if i % 2500 == 0:
                print i
            prediction_matrix[i], text_features[i] = nn.predict_and_get_text_features(words_matrices, float_scalars)
        features.append(prediction_matrix)
        features.append(text_features)
        print 'done', time.time() - t
    features = np.column_stack(features)

    dir_path = os.path.dirname(features_file_path['test'])
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    with open(features_file_path['test'], 'w') as f:
        hickle.dump({'features':features, 'index': index}, f)

    # ==================================================================================================================

    features = []
    data_frame = pd.read_pickle(data_file_path['train'])
    fte_features = data_frame['FTE'].values.copy()
    total_features = data_frame['Total'].values.copy()
    fte_features[np.isnan(fte_features)] = nan_substitute['FTE']
    total_features[np.isnan(total_features)] = nan_substitute['Total']
    features.append(fte_features)
    features.append(total_features)

    labels = defaultdict(list)
    for label_name, label_values in preprocessing.multi_labels.iteritems():
        data_frame = pd.read_pickle(data_file_path['train'])
        with open('{}/{}.pckl'.format(transformers_dir, label_name)) as f:
            d = cPickle.load(f)
        data_frame = transform_data(data_frame, d['softmax_normalizers'], d['nan_replacers'])
        nn = NeuralNetwork.load(classifiers_paths[label_name])
        nn.set_test_mode()
        test_data_iterator = ValidationDataIterator(label_name, data_frame, word_vectors_path, enable_matrices_cache=False)

        t = time.time()
        print 'Start extracting features for label {}'.format(label_name)
        prediction_matrix = np.empty(shape=(data_frame.shape[0], len(label_values)), dtype=np.float32)
        text_features_n_col = sum(words_layer.output_vector_size for words_layer in nn.words_layers)
        text_features = np.empty(shape=(data_frame.shape[0], text_features_n_col), dtype=np.float32)
        float_features = np.empty(shape=(data_frame.shape[0], 2))
        for i, (words_matrices, float_scalars, label_idx) in enumerate(test_data_iterator):
            if i % 2500 == 0:
                print i
            prediction_matrix[i], text_features[i] = nn.predict_and_get_text_features(words_matrices, float_scalars)
            labels[label_name].append(label_idx)
        features.append(prediction_matrix)
        features.append(text_features)
        print 'done', time.time() - t
    features = np.column_stack(features)

    dir_path = os.path.dirname(features_file_path['train'])
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    with open(features_file_path['train'], 'w') as f:
        hickle.dump({'features': features, 'labels': labels}, f)