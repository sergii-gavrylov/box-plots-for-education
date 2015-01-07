import numpy as np
from itertools import izip, count


class LabelEncoder(object):
    def __init__(self, column_name, labels):
        self.column_name = column_name
        self.idx_to_label = labels
        self.label_to_idx = dict(izip(labels, count()))

    def transform(self, data_frame):
        encoded_labels = np.empty(shape=(data_frame.shape[0], ), dtype=np.int32)
        for idx, label in enumerate(data_frame[self.column_name]):
            encoded_labels[idx] = self.label_to_idx[label]
        return encoded_labels