import os
import itertools as it

import numpy as np

import theano
import theano.tensor as T

from blocks.initialization import Orthogonal, Constant
from blocks.bricks import MLP, Softmax, Initializable, Linear
from blocks.bricks.base import application
from blocks.filter import VariableFilter
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate

from fuel.transformers import Mapping
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme

import util

from fuel.datasets import H5PYDataset

class SVHN(H5PYDataset):
    def __init__(self, **kwargs):
        kwargs.setdefault('load_in_memory', True)
        self.filename = 'dataset_64x64.h5'
        super(SVHN, self).__init__(self.data_path, **kwargs)

    @property
    def data_path(self):
        path = os.path.join("/data/lisatmp3/cooijmat/datasets/svhn", self.filename)
        return path

def fix_representation(data):
    x, y = data

    # grayscale
    x = x.mean(axis=3, keepdims=True)
    x = np.rollaxis(x, 3, 1)
    # standardize and center on zero (TODO: subtract mean)
    x = (x / 255.0) * 2 - 1

    y = np.array(y, copy=True)
    # use zero to represent zero
    y[y == 10] = 0
    lengths = (y >= 0).sum(axis=1)
    y[y < 0] = 0
    # pretend there are no examples with length > 5 (there are too few to care about)
    lengths = np.clip(lengths, 0, 5)
    # repurpose the last column to store 0-based lenghts
    y[:, -1] = lengths - 1
    return x, y

class Emitter(Initializable):
    def __init__(self, hidden_dim, n_classes, **kwargs):
        super(Emitter, self).__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        # TODO: use TensorLinear or some such
        self.emitters = [Linear(input_dim=hidden_dim,
                                output_dim=n,
                                name="linear_%i" % i,
                                weights_init=Orthogonal(),
                                biases_init=Constant(0))
                         for i, n in enumerate(self.n_classes)]
        self.softmax = Softmax()

        self.children = self.emitters + [self.softmax]

    # some day: @application(...) def feedback(self, h)

    @application(inputs=['h', 'y'])
    def cost(self, h, y):
        max_length = len(self.n_classes) - 1
        _length_masks = theano.shared(
            np.tril(np.ones((max_length, max_length), dtype='int8')),
            name='shared_length_masks')
        lengths = y[:, -1]
        length_masks = _length_masks[lengths]

        energies = [emitter.apply(h) for emitter in self.emitters]
        cross_entropies = [util.named(self.softmax.categorical_cross_entropy(y[:, i], energy)
                                      # to avoid punishing predictions of nonexistent digits:
                                      * (length_masks[:, i] if i < max_length else 1),
                                      "cross_entropy_%i" % i)
                           for i, energy in enumerate(energies)]
        cross_entropy = util.named(sum(cross_entropies).mean(), "cross_entropy")
        errors = [util.named(T.neq(y[:, i], energy.argmax(axis=1))
                             # to avoid punishing predictions of nonexistent digits:
                             * (length_masks[:, i] if i < max_length else 1),
                             "error_%i" % i)
                  for i, energy in enumerate(energies)]
        error_rate = util.named(T.stack(*errors).any(axis=0).mean(), "error_rate")
        for variable in it.chain([cross_entropy, error_rate],
                                 cross_entropies, errors):
            self.add_auxiliary_variable(variable, name=variable.name)
        return cross_entropy

class NumberTask(object):
    def __init__(self, batch_size, hidden_dim, hyperparameters, shrink_dataset_by=1, **kwargs):
        self.max_length = 5
        self.n_classes = [10,] * self.max_length + [self.max_length]
        self.n_channels = 1
        hyperparameters["n_channels"] = self.n_channels
        self.datasets = dict(
            train=SVHN(which_sets=["train"]),
            valid=SVHN(which_sets=["valid"]),
            test=SVHN(which_sets=["test"]))
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
            mapping=fix_representation)

    def get_variables(self):
        # shape (batch, channel, height, width)
        x = T.tensor4('features', dtype=theano.config.floatX)
        # shape (batch_size, n_classes)
        y = T.lmatrix('targets')

        theano.config.compute_test_value = 'warn'
        test_batch_size = 7
        x.tag.test_value = np.random.random((test_batch_size, self.n_channels, 64, 64)).astype("float32")
        y.tag.test_value = np.concatenate([np.random.random_integers(0, n-1, (test_batch_size, 1))
                                           for n in self.n_classes],
                                          axis=1)

        return x, y

    def get_emitter(self, hidden_dim, **kwargs):
        return Emitter(hidden_dim, self.n_classes)

    def monitor_channels(self, graph):
        return [VariableFilter(name=name)(graph.auxiliary_variables)[0]
                for name in "cross_entropy error_rate".split()]

    def plot_channels(self):
        return [["%s_%s" % (which_set, name) for which_set in self.datasets.keys()]
                for name in "cross_entropy error_rate".split()]
