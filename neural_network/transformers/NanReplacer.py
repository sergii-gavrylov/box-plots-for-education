class NanReplacer(object):
    """ Replace all nan value inplace """
    def __init__(self, nan_substitute, column_name):
        self.nan_substitute = nan_substitute
        self.column_name = column_name

    def transform(self, data_frame):
        data_frame.fillna(self.nan_substitute, inplace=True)
        return data_frame