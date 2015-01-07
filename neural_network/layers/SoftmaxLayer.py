import theano
import numpy as np
import theano.tensor as T
from theano.ifelse import ifelse


class SoftmaxLayer(object):
    def __init__(self, n_in, num_classes, dropout_prob, layer_name):
        self.W = theano.shared(value=np.zeros((n_in, num_classes), dtype=np.float32), name=layer_name+'_W', borrow=True)
        self.b = theano.shared(value=np.zeros((num_classes, ), dtype=np.float32), name=layer_name+'_b', borrow=True)
        self.train_mode = theano.shared(value=1, name='train_mode')
        self.keep_prob = 1. - dropout_prob

    def set_train_mode(self):
        self.train_mode.set_value(1)

    def set_test_mode(self):
        self.train_mode.set_value(0)

    def get_cross_entropy_loss_expr(self, input_vector, true_label):
        label_prob_dist = self.get_predict_expr(input_vector)
        return -T.log(label_prob_dist[true_label] + 1e-13)

    def get_predict_expr(self, input_vector):
        r_stream = T.shared_randomstreams.RandomStreams(1984)
        mask = r_stream.binomial(size=input_vector.shape, p=self.keep_prob, dtype='float32')
        dropout_input_vector = ifelse(condition=T.eq(self.train_mode, 1),
                                      then_branch=mask * input_vector,
                                      else_branch=self.keep_prob * input_vector)
        lin = T.dot(dropout_input_vector, self.W) + self.b
        exp_lin = T.exp(lin - lin.max())
        return exp_lin / exp_lin.sum()

    def get_parameters(self):
        return [self.W, self.b]