import os
import sys
import json
import cPickle
import hickle as hkl
from sklearn.ensemble import RandomForestClassifier


with open(sys.argv[1]) as f:
    conf = json.load(f)
classifiers_dir = conf['classifiers_dir']
train_features_file_path = conf['train_features_file_path']


if __name__ == '__main__':
    if not os.path.isdir(classifiers_dir):
        os.makedirs(classifiers_dir)

    with open(train_features_file_path) as f:
        d = hkl.load(f)
        train_features = d['features']
        labels = d['labels']

    for label_name, y in labels.iteritems():
        # clf = RandomForestClassifier(n_estimators=15,
        #                              criterion='entropy',
        #                              max_features=0.55,
        #                              max_depth=10,
        #                              min_samples_split=30,
        #                              oob_score=True,
        #                              n_jobs=-1,
        #                              random_state=42,
        #                              verbose=0)
        clf = RandomForestClassifier(n_estimators=30,
                                     criterion='entropy',
                                     max_features=0.4,
                                     max_depth=12,
                                     min_samples_split=10,
                                     oob_score=True,
                                     n_jobs=-1,
                                     random_state=42,
                                     verbose=0)
        print 'start fit', label_name
        clf.fit(train_features, y)
        print 'fit is done'
        print label_name, clf.oob_score_

        with open('{}/{}.clf'.format(classifiers_dir, label_name), 'wb') as f:
            cPickle.dump(clf, f)