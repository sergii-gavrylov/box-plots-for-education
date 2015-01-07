import os
import sys
import json
import logging
import cPickle
import numpy as np
import pandas as pd
import theano.tensor as T
from neural_network.NnTrainer import train_nn
from neural_network.layers import NeuralNetwork
from neural_network import WordImportanceInitializer
from sklearn.cross_validation import StratifiedShuffleSplit
from preprocessing import text_columns, float_columns, multi_labels
from neural_network.transformers import SoftmaxNormalizer, NanReplacer, fit_transform_data, transform_data
from neural_network.data_iterators import InfiniteRandomDataIterator, ValidationDataIterator


with open(sys.argv[1]) as f:
    conf = json.load(f)
train_data_file_path = conf['train_data_file_path']
word_vectors_path = conf['word_vectors_path']
transformers_dir = conf['transformers_dir']
classifiers_dir = conf['classifiers_dir']
logs_dir = conf['logs_dir']
nan_substitute = conf['nan_substitute']
save_freq = conf['save_freq']
valid_size = conf.get('valid_size')
valid_freq = conf['valid_freq']
models = conf['models']
for label_name, params in models.iteritems():
    models[label_name]['learning_rate_schedule']= dict((int(k), np.float32(v)) for k, v in params['learning_rate_schedule'].iteritems())
    models[label_name]['nag_momentum_schedule'] = dict((int(k), np.float32(v)) for k, v in params['nag_momentum_schedule'].iteritems())
random_seed = conf['random_seed']


def train_valid_split(data_frame, label_name):
    labels = data_frame[label_name].tolist()
    stratified_splitter = StratifiedShuffleSplit(labels, n_iter=1, test_size=valid_size, random_state=random_seed)
    train_indexes, valid_indexes = stratified_splitter.__iter__().next()
    return data_frame.iloc[train_indexes], data_frame.iloc[valid_indexes]


def create_transformers():
    softmax_normalizers = []
    for column_name in float_columns:
        softmax_normalizers.append(SoftmaxNormalizer(column_name))
    nan_replacers = []
    for column_name in float_columns:
        nan_replacers.append(NanReplacer(nan_substitute[column_name], column_name))
    return softmax_normalizers, nan_replacers


def save_transformers(softmax_normalizers, nan_replacers, label_name):
    with open('{}/{}.pckl'.format(transformers_dir, label_name), 'wb') as f:
        cPickle.dump({'softmax_normalizers': softmax_normalizers,
                      'nan_replacers': nan_replacers}, f)


def get_logger(label_name):
    logger = logging.getLogger(label_name)
    handler = logging.FileHandler('{}/{}.log'.format(logs_dir, label_name), 'w', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', '%d-%m-%Y %H:%M:%S'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def create_neural_network(num_classes, word_importance_initializer, softmax_layer_dropout_prob):
    words_matrices = {}
    for column_name in text_columns:
        words_matrices[column_name] = T.fmatrix(column_name)
    float_scalars = {}
    for column_name in float_columns:
        float_scalars[column_name] = T.fscalar(column_name)
    true_label = T.iscalar('true_label')

    return NeuralNetwork(num_classes=num_classes,
                         words_matrices=words_matrices,
                         float_scalars=float_scalars,
                         true_label=true_label,
                         word_importance_initializer=word_importance_initializer,
                         softmax_layer_dropout_prob=softmax_layer_dropout_prob)


if __name__ == '__main__':
    if not os.path.isdir(transformers_dir):
        os.makedirs(transformers_dir)
    if not os.path.isdir(classifiers_dir):
        os.makedirs(classifiers_dir)
    if not os.path.isdir(logs_dir):
        os.makedirs(logs_dir)

    for label_name, params in models.iteritems():
        data_frame = pd.read_pickle(train_data_file_path)
        if valid_size:
            train_data_frame, valid_data_frame = train_valid_split(data_frame, label_name)
        else:
            train_data_frame = data_frame

        softmax_normalizers, nan_replacers = create_transformers()
        train_data_frame = fit_transform_data(train_data_frame, softmax_normalizers, nan_replacers)
        save_transformers(softmax_normalizers, nan_replacers, label_name)
        if valid_size:
            valid_data_frame = transform_data(valid_data_frame, softmax_normalizers, nan_replacers)

        train_iterator = InfiniteRandomDataIterator(label_name, train_data_frame, word_vectors_path, params['word_dropout_prob'], random_seed)
        if valid_size:
            valid_iterator = ValidationDataIterator(label_name, valid_data_frame, word_vectors_path)
        else:
            valid_iterator = None
        logger = get_logger(label_name)

        word_importance_initializer = WordImportanceInitializer(train_data_frame, label_name, word_vectors_path, random_seed, logger)
        nn = create_neural_network(num_classes=len(multi_labels[label_name]),
                                   word_importance_initializer=word_importance_initializer,
                                   softmax_layer_dropout_prob=params['softmax_layer_dropout_prob'])

        train_nn(nn, train_iterator, valid_iterator, save_freq, valid_freq, params['max_iters'],
                 params['learning_rate_schedule'], params['nag_momentum_schedule'], classifiers_dir, logger)