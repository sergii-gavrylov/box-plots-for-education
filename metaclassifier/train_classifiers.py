import os
import sys
import json
import hickle
import cPickle
from sklearn.ensemble import RandomForestClassifier


with open(sys.argv[1]) as f:
    conf = json.load(f)
train_features_file_path = conf['train_features_file_path']
classifiers_dir = conf['classifiers_dir']


if __name__ == '__main__':
    if not os.path.isdir(classifiers_dir):
        os.makedirs(classifiers_dir)

    with open(train_features_file_path) as f:
        d = hickle.load(f)
        train_features, labels = d['features'], d['labels']

    for label_name, y in labels.iteritems():
        # clf = RandomForestClassifier(n_estimators=15,
        #                              criterion='entropy',
        #                              max_features=0.5,
        #                              max_depth=10,
        #                              min_samples_split=25,
        #                              oob_score=True,
        #                              n_jobs=-1,
        #                              random_state=42,
        #                              verbose=2)
        clf = RandomForestClassifier(n_estimators=40,
                                     criterion='entropy',
                                     max_depth=13,
                                     min_samples_split=13,
                                     oob_score=True,
                                     n_jobs=-1,
                                     random_state=42,
                                     verbose=2)
        print 'start fit', label_name
        clf.fit(train_features, y)
        print 'fit is done'
        print label_name, clf.oob_score_

        with open('{}/{}.clf'.format(classifiers_dir, label_name), 'wb') as f:
            cPickle.dump(clf, f)