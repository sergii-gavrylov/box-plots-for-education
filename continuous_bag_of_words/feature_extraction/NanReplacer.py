import numpy as np


class NanReplacer(object):
    def __init__(self, nan_substitute, column_name):
        self.nan_substitute = nan_substitute
        self.column_name = column_name

    def transform(self, data_frame, data_array=None):
        column_data = data_frame[self.column_name].values
        column_data = np.reshape(column_data, newshape=(column_data.shape[0], 1))
        if data_array is not None:
            np.copyto(dst=data_array, src=column_data)
        else:
            data_array = column_data.copy()
        data_array[np.isnan(data_array)] = self.nan_substitute
        return data_array

    def _get_array_shape(self, data_frame):
        return data_frame.shape[0], 1