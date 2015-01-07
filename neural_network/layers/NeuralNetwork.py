import theano
import cPickle
import theano.tensor as T
from neural_network.layers import WordsLayer, SoftmaxLayer
from neural_network.WordImportanceInitializer import SerializationWordImportanceInitializer


class NeuralNetwork(object):
    def __init__(self, num_classes, words_matrices, float_scalars, true_label, word_importance_initializer, softmax_layer_dropout_prob):
        self.words_layers = []
        words_layers_output_exprs = [T.stack(*float_scalars.values())]
        text_output_expr = []
        for column_name, words_matrix in words_matrices.iteritems():
            importance_vector_init = word_importance_initializer.get_importances_vectors_init_values(column_name)
            words_layer = WordsLayer(column_name, importance_vector_init)
            output_expr = words_layer.get_forward_pass_expr(words_matrix)
            words_layers_output_exprs.append(output_expr)
            text_output_expr.append(output_expr)
            self.words_layers.append(words_layer)
        words_layers_output_expr = T.concatenate(words_layers_output_exprs)
        text_output_expr = T.concatenate(text_output_expr)

        self.softmax_layer = SoftmaxLayer(n_in=sum(words_layer.output_vector_size for words_layer in self.words_layers) + len(float_scalars),
                                          num_classes=num_classes,
                                          dropout_prob=softmax_layer_dropout_prob,
                                          layer_name='soft_max')

        self.cross_entropy_loss_expr = self.softmax_layer.get_cross_entropy_loss_expr(words_layers_output_expr, true_label)
        self.__cross_entropy = theano.function(inputs=words_matrices.values() + float_scalars.values() + [true_label],
                                               outputs=self.cross_entropy_loss_expr)

        predict_expr = self.softmax_layer.get_predict_expr(words_layers_output_expr)
        self.__predict = theano.function(inputs=words_matrices.values() + float_scalars.values(),
                                         outputs=predict_expr)

        self.__predict_and_get_text_features = theano.function(inputs=words_matrices.values() + float_scalars.values(),
                                                               outputs=[predict_expr, text_output_expr])

        self.words_matrices = words_matrices
        self.float_scalars = float_scalars
        self.true_label = true_label

    def set_train_mode(self):
        self.softmax_layer.set_train_mode()

    def set_test_mode(self):
        self.softmax_layer.set_test_mode()

    def get_cross_entropy_loss(self, words_matrices, float_scalars, true_label):
        return self.__cross_entropy(*(words_matrices.values() + float_scalars.values() + [true_label]))

    def predict(self, words_matrices, float_scalars):
        return self.__predict(*(words_matrices.values() + float_scalars.values()))

    def predict_and_get_text_features(self, words_matrices, float_scalars):
        return self.__predict_and_get_text_features(*(words_matrices.values() + float_scalars.values()))

    def get_parameters(self):
        params = []
        for words_layer in self.words_layers:
            params.extend(words_layer.get_parameters())
        params.extend(self.softmax_layer.get_parameters())
        return params

    def save(self, file_path):
        params = dict()
        params['word_importance_initializer'] = SerializationWordImportanceInitializer(self.words_layers)
        params[self.softmax_layer.W.name] = self.softmax_layer.W.get_value()
        params[self.softmax_layer.b.name] = self.softmax_layer.b.get_value()
        params['softmax_layer_dropout_prob'] = 1. - self.softmax_layer.keep_prob
        params['num_classes'] = self.softmax_layer.b.get_value().shape[0]

        with open(file_path, 'wb') as f:
            cPickle.dump(params, f)

    @staticmethod
    def load(model_path):
        from preprocessing import text_columns, float_columns
        words_matrices = {}
        for column_name in text_columns:
            words_matrices[column_name] = T.fmatrix(column_name)
        float_scalars = {}
        for column_name in float_columns:
            float_scalars[column_name] = T.fscalar(column_name)
        true_label = T.iscalar('true_label')

        with open(model_path) as f:
            params = cPickle.load(f)

        nn = NeuralNetwork(params['num_classes'], words_matrices, float_scalars, true_label, params['word_importance_initializer'], params['softmax_layer_dropout_prob'])
        nn.softmax_layer.W.set_value(params[nn.softmax_layer.W.name])
        nn.softmax_layer.b.set_value(params[nn.softmax_layer.b.name])
        nn.set_test_mode()
        return nn