import theano.tensor as T

from blocks.bricks import Initializable, MLP, Softmax, Rectifier
from blocks.bricks.base import application
from blocks.initialization import Orthogonal, Constant

import util

class SingleSoftmax(Initializable):
    def __init__(self, hidden_dim, n_classes, **kwargs):
        super(SingleSoftmax, self).__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        self.mlp = MLP(activations=[Rectifier(), Softmax()],
                       dims=[hidden_dim, hidden_dim/2, self.n_classes],
                       weights_init=Orthogonal(),
                       biases_init=Constant(0))
        self.softmax = Softmax()

        self.children = [self.mlp, self.softmax]

    # some day: @application(...) def feedback(self, h)

    @application(inputs=['cs', 'y'], outputs=['cost'])
    def cost(self, cs, y, n_patches):
        energies = [self.mlp.apply(cs[:, t, :])
                    for t in xrange(n_patches)]
        cross_entropies = [self.softmax.categorical_cross_entropy(y.flatten(), energy)
                           for energy in energies]
        error_rates = [T.neq(y.flatten(), energy.argmax(axis=1)).mean(axis=0)
                       for energy in energies]
        # train on all predictions
        cost = util.named(T.stack(*cross_entropies).mean(), "cost")
        # monitor final prediction
        self.add_auxiliary_variable(cross_entropies[-1], name="cross_entropy")
        self.add_auxiliary_variable(error_rates[-1], name="error_rate")
        return cost
