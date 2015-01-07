import numpy as np


class SoftmaxNormalizer(object):
    """ Inplace softmax normalizer """
    def __init__(self, column_name):
        self.column_name = column_name
        self.mean = None
        self.std = None

    def fit(self, data_frame):
        values = data_frame[self.column_name].values
        values = values[np.logical_not(np.isnan(values))]
        self.mean = values.mean()
        self.std = values.std()

    def transform(self, data_frame):
        values = data_frame[self.column_name].values
        values = 1./(1.+np.exp(-(values-self.mean)/self.std))
        data_frame[self.column_name] = values
        return data_frame

    def fit_transform(self, data_frame):
        self.fit(data_frame)
        return self.transform(data_frame)