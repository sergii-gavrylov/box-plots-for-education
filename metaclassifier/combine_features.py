import os
import sys
import json
import numpy as np
import hickle as hkl


with open(sys.argv[1]) as f:
    conf = json.load(f)
features_file_paths = conf['features_file_paths']
combined_features_file_path = conf['combined_features_file_path']


if __name__ == '__main__':
    shapes = []
    for train_features_file_path in features_file_paths['train']:
        with open(train_features_file_path) as f:
            d = hkl.load(f)
            train_features = d['features']
        num_instances = train_features.shape[0]
        shapes.append(train_features.shape[1])
    train_features_comb = np.empty(shape=(num_instances, sum(shapes)), dtype=np.float32)
    offset = 0
    for i, train_features_file_path in enumerate(features_file_paths['train']):
        with open(train_features_file_path) as f:
            d = hkl.load(f)
            train_features = d['features']
            labels = d['labels']
        train_features_comb[:, offset:offset+shapes[i]] = train_features
        offset += shapes[i]

    dir_path = os.path.dirname(combined_features_file_path['train'])
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    with open(combined_features_file_path['train'], 'w') as f:
        hkl.dump({'features': train_features_comb, 'labels': labels}, f)

    # ==================================================================================================================

    shapes = []
    for test_features_file_path in features_file_paths['test']:
        with open(test_features_file_path) as f:
            d = hkl.load(f)
            test_features = d['features']
            num_instances = test_features.shape[0]
            shapes.append(test_features.shape[1])
    test_features_comb = np.empty(shape=(num_instances, sum(shapes)), dtype=np.float32)
    offset = 0
    for i, test_features_file_path in enumerate(features_file_paths['test']):
        with open(test_features_file_path) as f:
            d = hkl.load(f)
            test_features = d['features']
            index = d['index']
        test_features_comb[:, offset:offset+shapes[i]] = test_features
        offset += shapes[i]

    dir_path = os.path.dirname(combined_features_file_path['test'])
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    with open(combined_features_file_path['test'], 'w') as f:
        hkl.dump({'features': test_features_comb, 'index': index}, f)