import theano.tensor as T

from blocks.bricks.base import application

import bricks
import initialization

import util
import masonry

class SingleSoftmax(bricks.Initializable):
    def __init__(self, hidden_dim, n_classes, batch_normalize, **kwargs):
        super(SingleSoftmax, self).__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        self.mlp = masonry.construct_mlp(
            name="mlp",
            activations=[None, bricks.Identity()],
            input_dim=hidden_dim,
            hidden_dims=[hidden_dim/2, self.n_classes],
            batch_normalize=batch_normalize,
            initargs=dict(weights_init=initialization.Orthogonal(),
                          biases_init=initialization.Constant(0)))
        self.softmax = bricks.Softmax()

        self.children = [self.mlp, self.softmax]

    # some day: @application(...) def feedback(self, h)

    @application(inputs=['cs', 'y'], outputs=['cost'])
    def cost(self, cs, y, n_patches):
        energies = [self.mlp.apply(cs[:, t, :])
                    for t in xrange(n_patches)]
        cross_entropies = [
            self.softmax.categorical_cross_entropy(
                y.flatten(), energy).mean(axis=0)
            for energy in energies]
        error_rates = [
            T.neq(y, energy.argmax(axis=1)).mean(axis=0)
            for energy in energies]
        # train on final prediction
        cost = util.named(cross_entropies[-1], "cost")
        # monitor final prediction
        self.add_auxiliary_variable(cross_entropies[-1], name="cross_entropy")
        self.add_auxiliary_variable(error_rates[-1], name="error_rate")
        return cost
