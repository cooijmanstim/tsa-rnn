# SVHN number transcription as in http://arxiv.org/pdf/1312.6082v4.pdf
import numpy as np

import theano
import theano.tensor as T

from blocks.bricks.base import application
from blocks.filter import VariableFilter

from fuel.transformers import Mapping
from fuel.datasets import H5PYDataset

import bricks
import initialization

import tasks
import masonry

class SVHN(H5PYDataset):
    def __init__(self, **kwargs):
        kwargs.setdefault('load_in_memory', True)
        super(SVHN, self).__init__(
            "/data/lisatmp3/cooijmat/datasets/svhn/dataset_64_gray.h5",
            **kwargs)

def fix_representation(data):
    x, y = data

    x /= 255.0
    x = x.mean(axis=3, keepdims=True) # grayscale
    # move channel axis forward
    x = np.rollaxis(x, 3, 1)

    # crop images randomly
    assert(x.shape[2] == x.shape[3])
    image_size = x.shape[2]
    crop_size = 54
    a = np.random.randint(0, image_size - crop_size, size=(2,))
    b = a + crop_size
    x = x[:, :, a[0]:b[0], a[1]:b[1]]

    y = np.array(y, copy=True)
    # use zero to represent zero
    y[y == 10] = 0
    lengths = (y >= 0).sum(axis=1)
    y[y < 0] = 0
    # pretend there are no examples with length > 5 (there are too few to care about)
    lengths = np.clip(lengths, 0, 5)
    # repurpose the last column to store 0-based lenghts
    y[:, -1] = lengths - 1
    y = y.astype(np.uint8)
    return x, y

class Emitter(bricks.Initializable):
    def __init__(self, hidden_dim, n_classes, batch_normalize, **kwargs):
        super(Emitter, self).__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        # TODO: use TensorLinear or some such
        self.emitters = [
            masonry.construct_mlp(
                activations=[None, bricks.Identity()],
                input_dim=hidden_dim,
                hidden_dims=[hidden_dim/2, n],
                name="mlp_%i" % i,
                batch_normalize=batch_normalize,
                initargs=dict(weights_init=initialization.Orthogonal(),
                              biases_init=initialization.Constant(0)))
            for i, n in enumerate(self.n_classes)]
        self.softmax = bricks.Softmax()

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

        def compute_yhat(logprobs):
            digits_logprobs = T.stack(*logprobs[:-1]) # (#positions, batch, #classes)
            length_logprobs = logprobs[-1]           # (batch, #classes)
            # predict digits independently
            digits_hat = digits_logprobs.argmax(axis=2) # (#positions, batch)
            # likelihood of prediction
            digits_logprob = digits_logprobs.max(axis=2)
            # logprobs of resulting number given length
            number_logprobs = T.extra_ops.cumsum(digits_logprob, axis=0) # (#positions, batch)
            # choose length to minimize length_logprob + number_logprob
            length_hat = (length_logprobs.T + number_logprobs).argmax(axis=0, keepdims=True) # (1, batch)
            yhat = T.concatenate([digits_hat, length_hat], axis=0).T
            return yhat # shape (batch, #positions + 1)

        def compute_mean_cross_entropy(y, logprobs):
            return sum(self.softmax.categorical_cross_entropy(y[:, i], logprob)
                       # to avoid punishing predictions of nonexistent digits:
                       * (length_masks[:, i] if i < max_length else 1)
                       for i, logprob in enumerate(logprobs)).mean()
        def compute_error_rate(y, logprobs):
            yhat = compute_yhat(logprobs)
            return T.stack(*[T.neq(y[:, i], yhat[:, i])
                             # to avoid punishing predictions of nonexistent digits:
                             * (length_masks[:, i] if i < max_length else 1)
                             for i, logprob in enumerate(logprobs)]).any(axis=0).mean()

        mean_cross_entropies = []
        error_rates = []
        for t in xrange(n_patches):
            logprobs = [self.softmax.log_probabilities(emitter.apply(cs[:, t, :]))
                        for emitter in self.emitters]
            mean_cross_entropies.append(compute_mean_cross_entropy(y, logprobs))
            error_rates.append(compute_error_rate(y, logprobs))

        self.add_auxiliary_variable(mean_cross_entropies[-1], name="cross_entropy")
        self.add_auxiliary_variable(error_rates[-1], name="error_rate")

        # minimize the mean cross entropy over time and over batch
        cost = mean_cross_entropies[-1]
        return cost

class NumberTask(tasks.Classification):
    name = "svhn_number"

    def __init__(self, *args, **kwargs):
        super(NumberTask, self).__init__(*args, **kwargs)
        self.max_length = 5
        self.n_classes = [10,] * self.max_length + [self.max_length]
        self.n_channels = 1

    def load_datasets(self):
        return dict(
            train=SVHN(which_sets=["train"]),
            valid=SVHN(which_sets=["valid"]),
            test=SVHN(which_sets=["test"]))

    def get_stream_num_examples(self, which_set, monitor):
        if monitor and which_set == "train":
            return 10000
        return super(NumberTask, self).get_stream_num_examples(which_set, monitor)

    def get_stream(self, *args, **kwargs):
        return Mapping(super(NumberTask, self).get_stream(*args, **kwargs),
                       mapping=fix_representation)

    def get_emitter(self, hidden_dim, batch_normalize, **kwargs):
        return Emitter(hidden_dim, self.n_classes,
                       batch_normalize=batch_normalize)

    def monitor_channels(self, graph):
        return [VariableFilter(name=name)(graph.auxiliary_variables)[0]
                for name in "cross_entropy error_rate".split()]

    def plot_channels(self):
        return [["%s_%s" % (which_set, name) for which_set in self.datasets.keys()]
                for name in "cross_entropy error_rate".split()]
