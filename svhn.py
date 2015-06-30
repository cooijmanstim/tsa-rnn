import numpy as np

import theano
import theano.tensor as T

from blocks.initialization import Orthogonal, Constant
from blocks.bricks import MLP, Softmax
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate

from fuel.transformers import Mapping
from fuel.datasets.svhn import SVHN
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme

import util

def fix_target_representation(data):
    x, y = data
    # use zero to represent zero
    y[y == 10] = 0
    return x, y

class DigitTask(object):
    def __init__(self, batch_size, hidden_dim, hyperparameters, shrink_dataset_by=1, **kwargs):
        self.n_classes = 10
        self.n_channels = 3
        hyperparameters["n_channels"] = self.n_channels
        self.datasets = dict(
            train=SVHN(which_sets=["train"], which_format=2, subset=slice(None, 50000)),
            valid=SVHN(which_sets=["train"], which_format=2, subset=slice(50000, None)),
            test=SVHN(which_sets=["test"], which_format=2))
        self.datastreams = dict(
            (which,
             self.get_stream(which,
                             ShuffledScheme(dataset.num_examples / shrink_dataset_by,
                                            batch_size)))
            for which, dataset in self.datasets.items())

    def get_stream(self, which_set, scheme):
        return Mapping(
            data_stream=DataStream.default_stream(
                dataset=self.datasets[which_set],
                iteration_scheme=scheme),
            mapping=fix_target_representation)

    def get_variables(self):
        # shape (batch, channel, height, width)
        x = T.tensor4('features', dtype=theano.config.floatX)
        # shape (batch_size, n_classes)
        y = T.lmatrix('targets')

        theano.config.compute_test_value = 'warn'
        x.tag.test_value = np.random.random((7, self.n_channels, 32, 32)).astype("float32")
        y.tag.test_value = np.random.random_integers(0, 9, (7, 1)).astype("int64")

        return x, y

    def get_emitter(self, hidden_dim, **kwargs):
        return MLP(activations=[Softmax()],
                   dims=[hidden_dim, self.n_classes],
                   weights_init=Orthogonal(),
                   biases_init=Constant(0))

    def compute(self, x, hs, yhats, y):
        yhat = yhats[:, -1, :]
        cross_entropy = util.named(CategoricalCrossEntropy().apply(y.flatten(), yhat),
                                   "cross_entropy")
        error_rate = util.named(MisclassificationRate().apply(y.flatten(), yhat),
                                "error_rate")
        monitor_channels = [cross_entropy, error_rate]
        plot_channels = [["%s_%s" % (which_set, name) for which in task.datasets.keys()]
                         for name in "cross_entropy error_rate".split()]
        return cross_entropy, monitor_channels, plot_channels

