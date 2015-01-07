import os
import sys
import json
import pandas as pd
import hickle as hkl
import preprocessing
from continuous_bag_of_words.feature_extraction import NanReplacer, FeaturesUnion, LabelEncoder
from continuous_bag_of_words.feature_extraction.ContinuousBagOfWordsTransformer import ContinuousBagOfWordsTransformer


with open(sys.argv[1]) as f:
    conf = json.load(f)
word_vectors_path = conf['word_vectors_path']
nan_substitute = conf['nan_substitute']
data_file_path = conf['data_file_path']
features_file_path = conf['features_file_path']


if __name__ == '__main__':
    transformers = [ContinuousBagOfWordsTransformer(word_vectors_path=word_vectors_path, text_columns=preprocessing.text_columns)]
    for column_name in preprocessing.float_columns:
        transformers.append(NanReplacer(nan_substitute[column_name], column_name))
    features_union = FeaturesUnion(transformers)

    for dataset_type in ['train', 'test']:
        data = pd.read_pickle(data_file_path[dataset_type])
        features = features_union.transform(data)

        print dataset_type
        print features.shape

        if dataset_type == 'train':
            labels = {}
            for label_name, label_values in preprocessing.multi_labels.iteritems():
                labels[label_name] = LabelEncoder(label_name, label_values).transform(data)
            features = {'features': features, 'labels': labels}
        else:
            features = {'features': features, 'index': data.index.values}

        dir_path = os.path.dirname(features_file_path[dataset_type])
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        with open(features_file_path[dataset_type], 'w') as f:
            hkl.dump(features, f)