import theano
import numpy as np
import theano.tensor as T
from itertools import izip
from itertools import count
from collections import defaultdict


# https://github.com/lisa-lab/pylearn2/pull/136
def get_nesterov_momentum_updates(loss_expr, parameters, learning_rate, momentum):
    grads = T.grad(cost=loss_expr, wrt=parameters)

    updates = []
    for param, grad in izip(parameters, grads):
        velocity = theano.shared(np.zeros_like(param.get_value(), dtype=np.float32), name='velocity_'+param.name)
        v = momentum * velocity - learning_rate * grad
        p = param + momentum * v - learning_rate * grad
        updates.append((velocity, v))
        updates.append((param, p))

    return updates


def train_nn(nn, train_iterator, valid_iterator, save_freq, valid_freq, max_iters, learning_rate_schedule, momentum_schedule, classifiers_dir, logger):
    train_data_generator = train_iterator.get_infinite_iterator()
    learning_rate = theano.shared(value=np.float32(0.0), name='learning_rate')
    momentum = theano.shared(value=np.float32(0.0), name='momentum')
    logger.info(nn.get_parameters())
    nesterov_momentum_updates = get_nesterov_momentum_updates(loss_expr=nn.cross_entropy_loss_expr,
                                                              parameters=nn.get_parameters(),
                                                              learning_rate=learning_rate,
                                                              momentum=momentum)
    update_params = theano.function(inputs=nn.words_matrices.values() + nn.float_scalars.values() + [nn.true_label],
                                    outputs=nn.cross_entropy_loss_expr,
                                    updates=nesterov_momentum_updates)

    update_nn_params = lambda w_matrices, f_scalars, t_label: update_params(*(w_matrices.values() + f_scalars.values() + [t_label]))

    nn.set_train_mode()
    train_cross_entropy = defaultdict(list)
    for iter_num in count():
        if iter_num in learning_rate_schedule:
            learning_rate.set_value(np.float32(learning_rate_schedule[iter_num]))
            logger.info('iter_num: {} learning rate: {}'.format(iter_num, learning_rate_schedule[iter_num]))
        if iter_num in momentum_schedule:
            momentum.set_value(np.float32(momentum_schedule[iter_num]))
            logger.info('iter_num: {} momentum: {}'.format(iter_num, momentum_schedule[iter_num]))

        words_matrices, float_scalars, true_label = train_data_generator.next()
        loss = update_nn_params(words_matrices, float_scalars, true_label)
        train_cross_entropy[train_iterator.idx_to_label[true_label]].append(loss)

        if iter_num % valid_freq == 0 and iter_num != 0:
            logger.info('iter_num: {}'.format(iter_num))
            logger.info('===train error===')
            total_loss = []
            for label_value in train_iterator.idx_to_label:
                if label_value in train_cross_entropy:
                    total_loss += train_cross_entropy[label_value]
                    mean_loss = np.mean(train_cross_entropy[label_value])
                else:
                    mean_loss = np.nan
                logger.info('{:50s}: {:1.10f}'.format(label_value, mean_loss))
            logger.info('{:50s}: {:1.10f}'.format('total_loss', np.mean(total_loss)))
            train_cross_entropy = defaultdict(list)
            if valid_iterator:
                logger.info('===valid error===')
                nn.set_test_mode()
                valid_cross_entropy = defaultdict(list)
                total_loss = []
                for words_matrices, float_scalars, true_label in valid_iterator:
                    loss = nn.get_cross_entropy_loss(words_matrices, float_scalars, true_label)
                    valid_cross_entropy[valid_iterator.idx_to_label[true_label]].append(loss)
                    total_loss.append(loss)

                for label_value in valid_iterator.idx_to_label:
                    mean_loss = np.mean(valid_cross_entropy[label_value]) if label_value in valid_cross_entropy else np.nan
                    logger.info('{:50s}: {:1.10f}'.format(label_value, mean_loss))
                logger.info('{:50s}: {:1.10f}'.format('total_loss', np.mean(total_loss)))
                nn.set_train_mode()


        if iter_num % save_freq == 0 and iter_num != 0:
           logger.info('iter_num: {}'.format(iter_num))
           logger.info('saving model...')
           nn.save('{}/{}_{}.clf'.format(classifiers_dir, train_iterator.label_name, iter_num))
           logger.info('done')

        if iter_num >= max_iters:
            logger.info('iter_num: {}'.format(iter_num))
            logger.info('max iters limit!!!')
            logger.info('saving model...')
            nn.save('{}/{}_{}.clf'.format(classifiers_dir, train_iterator.label_name, iter_num))
            logger.info('done')
            break