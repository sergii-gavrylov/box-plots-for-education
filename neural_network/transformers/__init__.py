from neural_network.transformers.NanReplacer import NanReplacer
from neural_network.transformers.SoftmaxNormalizer import SoftmaxNormalizer


def fit_transform_data(data_frame, softmax_normalizers, nan_replacers):
    for s_n in softmax_normalizers:
        data_frame = s_n.fit_transform(data_frame)
    for n_r in nan_replacers:
        data_frame = n_r.transform(data_frame)
    return data_frame


def transform_data(data_frame, softmax_normalizers, nan_replacers):
    for s_n in softmax_normalizers:
        data_frame = s_n.transform(data_frame)
    for n_r in nan_replacers:
        data_frame = n_r.transform(data_frame)
    return data_frame