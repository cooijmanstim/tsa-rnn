import os
import logging

import numpy as np

import theano
import theano.tensor as T

from blocks.filter import VariableFilter

from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme, SequentialScheme

import emitters

logger = logging.getLogger(__name__)

class Classification(object):
    def __init__(self, batch_size, hidden_dim, shrink_dataset_by=1, **kwargs):
        self.shrink_dataset_by = shrink_dataset_by
        self.batch_size = batch_size
        self.datasets = self.load_datasets()

    def load_datasets(self):
        raise NotImplementedError()

    def get_stream_num_examples(self, which_set, monitor):
        return (self.datasets[which_set].num_examples
                / self.shrink_dataset_by)

    def get_stream(self, which_set, shuffle=True, monitor=False, num_examples=None):
        scheme_klass = ShuffledScheme if shuffle else SequentialScheme
        if num_examples is None:
            num_examples = self.get_stream_num_examples(which_set, monitor=monitor)
        scheme = scheme_klass(num_examples, self.batch_size)
        return DataStream.default_stream(
            dataset=self.datasets[which_set],
            iteration_scheme=scheme)

    def get_variables(self):
        test_batch = self.get_stream("valid").get_epoch_iterator(as_dict=True).next()

        broadcastable = [False]*test_batch["features"].ndim
        # shape (batch, channel, [time,] height, width)
        x = T.TensorType(broadcastable=broadcastable,
                         dtype=theano.config.floatX)("features")
        # shape (batch_size, 1)
        broadcastable = [False]*test_batch["targets"].ndim
        y = T.TensorType(broadcastable=broadcastable,
                         dtype="uint8")('targets')

        theano.config.compute_test_value = 'warn'
        x.tag.test_value = test_batch["features"][:11, ...]
        y.tag.test_value = test_batch["targets"][:11, ...]

        # remove the singleton from mnist and svhn targets
        if test_batch["targets"].ndim == 2 and test_batch["targets"].shape[1] == 1:
            y = y.flatten()

        return x, y

    def get_emitter(self, hidden_dim, batch_normalize, **kwargs):
        return emitters.SingleSoftmax(hidden_dim, self.n_classes,
                                      batch_normalize=batch_normalize)

    def monitor_channels(self, graph):
        return [VariableFilter(name=name)(graph.auxiliary_variables)[0]
                for name in "cross_entropy error_rate".split()]

    def plot_channels(self):
        return [["%s_%s" % (which_set, name) for which_set in self.datasets.keys()]
                for name in "cross_entropy error_rate".split()]

    def preprocess(self, x):
        cache_dir = os.environ["PREPROCESS_CACHE"]
        try:
            os.mkdir(cache_dir)
        except OSError:
            # directory already exists. surely the end of the world.
            pass
        cache = os.path.join(cache_dir, "%s.npz" % self.name)
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
            mean = mean.astype(np.float32)
            print "mean taken"
            try:
                np.savez(cache, mean=mean)
            except IOError, e:
                logger.error("couldn't save preprocessing cache: %s" % e)
                import ipdb; ipdb.set_trace()
        return x - mean
