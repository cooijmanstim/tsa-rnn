import logging

import numpy as np

import theano
import theano.tensor as T

from blocks.filter import VariableFilter

from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme

import emitters

logger = logging.getLogger(__name__)

class Classification(object):
    def __init__(self, batch_size, hidden_dim, shrink_dataset_by=1, **kwargs):
        self.shrink_dataset_by = shrink_dataset_by
        self.batch_size = batch_size
        self.datasets = self.load_datasets()

    def load_datasets(self):
        raise NotImplementedError()

    def get_stream(self, which_set, scheme=None):
        if not scheme:
            scheme = ShuffledScheme(
                self.datasets[which_set].num_examples
                / self.shrink_dataset_by,
                self.batch_size)
        return DataStream.default_stream(
            dataset=self.datasets[which_set],
            iteration_scheme=scheme)

    def get_variables(self):
        test_batch = self.get_stream("valid").get_epoch_iterator(as_dict=True).next()

        broadcastable = [False]*test_batch["features"].ndim
        # shape (batch, channel, [time,] height, width)
        x = T.TensorType(broadcastable=broadcastable,
                         dtype=theano.config.floatX)("features")
        # shape (batch_size, n_classes)
        broadcastable = [False]*test_batch["targets"].ndim
        y = T.TensorType(broadcastable=broadcastable,
                         dtype="uint8")('targets')

        # remove the singleton from mnist and svhn targets
        if test_batch["targets"].ndim == 2 and test_batch["targets"].shape[1] == 1:
            y = y.flatten()

        theano.config.compute_test_value = 'warn'
        x.tag.test_value = test_batch["features"]
        y.tag.test_value = test_batch["targets"]

        return x, y

    def get_emitter(self, hidden_dim, **kwargs):
        return emitters.SingleSoftmax(hidden_dim, self.n_classes)

    def monitor_channels(self, graph):
        return [VariableFilter(name=name)(graph.auxiliary_variables)[0]
                for name in "cross_entropy error_rate".split()]

    def plot_channels(self):
        return [["%s_%s" % (which_set, name) for which_set in self.datasets.keys()]
                for name in "cross_entropy error_rate".split()]

    def preprocess(self, x):
        cache = "/data/lisatmp3/cooijmat/preprocess-cache/%s.npz" % self.name
        try:
            data = np.load(cache)
            mean = data["mean"]
        except IOError:
            print "taking mean"
            mean = 0
            n = 0
            for batch in self.get_stream("train").get_epoch_iterator(as_dict=True):
                batch_sum = batch["features"].sum(axis=0, keepdims=True)
                k = batch["features"].shape[0]
                mean = n/float(n+k) * mean + 1/float(n+k) * batch_sum
                n += k
            print "mean taken"
            try:
                np.savez(cache, mean=mean)
            except IOError:
                logger.error("couldn't save preprocessing cache")
                import ipdb; ipdb.set_trace()
        return x - mean
