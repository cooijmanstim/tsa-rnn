import os

import numpy as np

import theano
import theano.tensor as T

from blocks.initialization import Orthogonal, Constant
from blocks.bricks import MLP, Softmax, Initializable, Rectifier, Identity
from blocks.bricks.base import application
from blocks.filter import VariableFilter

from fuel.transformers import Mapping
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme

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

    x /= 255.0
    x = x.mean(axis=3, keepdims=True) # grayscale
    x = np.rollaxis(x, 3, 1)

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
        self.emitters = [MLP(activations=[Rectifier(), Identity()],
                             dims=[hidden_dim, hidden_dim/2, n],
                             name="mlp_%i" % i,
                             weights_init=Orthogonal(),
                             biases_init=Constant(0))
                         for i, n in enumerate(self.n_classes)]
        self.softmax = Softmax()

        self.children = self.emitters + [self.softmax]

    # some day: @application(...) def feedback(self, h)

    @application(inputs=['cs', 'y'], outputs=['cost'])
    def cost(self, cs, y, n_patches):
        max_length = len(self.n_classes) - 1
        _length_masks = theano.shared(
            np.tril(np.ones((max_length, max_length), dtype='int8')),
            name='shared_length_masks')
        lengths = y[:, -1]
        length_masks = _length_masks[lengths]

        mean_cross_entropies = []
        error_rates = []
        for t in xrange(n_patches):
            energies = [emitter.apply(cs[:, t, :]) for emitter in self.emitters]
            mean_cross_entropies.append(
                sum(self.softmax.categorical_cross_entropy(y[:, i], energy)
                    # to avoid punishing predictions of nonexistent digits:
                    * (length_masks[:, i] if i < max_length else 1)
                    for i, energy in enumerate(energies)).mean())
            # FIXME: do proper logprob-minimizing prediction of length
            error_rates.append(
                T.stack(*[T.neq(y[:, i], energy.argmax(axis=1))
                          # to avoid punishing predictions of nonexistent digits:
                          * (length_masks[:, i] if i < max_length else 1)
                          for i, energy in enumerate(energies)]).any(axis=0).mean())

        self.add_auxiliary_variable(mean_cross_entropies[-1], name="cross_entropy")
        self.add_auxiliary_variable(error_rates[-1], name="error_rate")

        # minimize the mean cross entropy over time and over batch
        cost = T.stack(*mean_cross_entropies).mean()
        return cost

class NumberTask(object):
    def __init__(self, batch_size, hidden_dim, hyperparameters, shrink_dataset_by=1, **kwargs):
        self.shrink_dataset_by = shrink_dataset_by
        self.batch_size = batch_size
        self.max_length = 5
        self.n_classes = [10,] * self.max_length + [self.max_length]
        self.n_channels = 1
        hyperparameters["n_channels"] = self.n_channels
        self.datasets = dict(
            train=SVHN(which_sets=["train"]),
            valid=SVHN(which_sets=["valid"]),
            test=SVHN(which_sets=["test"]))

    def get_stream(self, which_set, scheme=None):
        if not scheme:
            scheme = ShuffledScheme(
                self.datasets[which_set].num_examples
                / self.shrink_dataset_by,
                self.batch_size)
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

    def preprocess(self, x):
        print "taking mean"
        mean = 0
        n = 0
        for batch in self.get_stream("train").get_epoch_iterator(as_dict=True):
            batch_sum = batch["features"].sum(axis=0, keepdims=True)
            k = batch["features"].shape[0]
            mean = n/float(n+k) * mean + 1/float(n+k) * batch_sum
            n += k
        print "mean taken"
        return x - mean
