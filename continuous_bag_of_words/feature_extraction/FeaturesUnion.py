import numpy as np
from itertools import izip


class FeaturesUnion(object):
    def __init__(self, transformers):
        self.transformers = transformers

    def transform(self, data_frame):
        shapes = [t._get_array_shape(data_frame) for t in self.transformers]
        n_row = None
        n_col = 0
        for shape in shapes:
            if n_row and n_row != shape[0]:
                raise ValueError('Transformers have different n_row!')
            else:
                n_row = shape[0]
            n_col += shape[1]

        data_array = np.zeros(shape=(n_row, n_col), dtype=np.float32)
        col_offset = 0
        for t, shape in izip(self.transformers, shapes):
            t.transform(data_frame, data_array[:, col_offset:col_offset+shape[1]])
            col_offset += shape[1]
        return data_array