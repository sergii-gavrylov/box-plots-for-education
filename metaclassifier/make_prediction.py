import os
import sys
import json
import hickle
import cPickle
import numpy as np
import preprocessing
from pandas import DataFrame


with open(sys.argv[1]) as f:
    conf = json.load(f)
test_features_file_path = conf['test_features_file_path']
classifiers_dir = conf['classifiers_dir']
submission_file_path = conf['submission_file_path']


if __name__ == '__main__':
    with open(test_features_file_path) as f:
        d = hickle.load(f)
        test_features = d['features']
        index = d['index']

    columns = []
    predictions = []
    for label_name, label_values in preprocessing.multi_labels.iteritems():
        print label_name
        with open('{}/{}.clf'.format(classifiers_dir, label_name)) as f:
            clf = cPickle.load(f)
            predictions.append(clf.predict_proba(test_features))
        columns.extend(label_name + '__' + label_value for label_value in label_values)
    predictions = np.column_stack(predictions)

    dir_path = os.path.dirname(submission_file_path)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    DataFrame(data=predictions, index=index, columns=columns).to_csv(submission_file_path)