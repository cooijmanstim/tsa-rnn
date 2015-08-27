import theano.tensor as T

from blocks.bricks import Initializable, MLP, Softmax, Rectifier, Identity
from blocks.bricks.base import application
from blocks.initialization import Orthogonal, Constant

import util
import masonry

class SingleSoftmax(Initializable):
    def __init__(self, hidden_dim, n_classes, batch_normalize, **kwargs):
        super(SingleSoftmax, self).__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        self.mlp = masonry.construct_mlp(
            activations=[None, Identity()],
            input_dim=hidden_dim,
            hidden_dims=[hidden_dim/2, self.n_classes],
            batch_normalize=batch_normalize,
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
        error_rates = [T.neq(y, energy.argmax(axis=1)).mean(axis=0)
                       for energy in energies]
        # train on final prediction
        cost = util.named(cross_entropies[-1], "cost")
        # monitor final prediction
        self.add_auxiliary_variable(cross_entropies[-1], name="cross_entropy")
        self.add_auxiliary_variable(error_rates[-1], name="error_rate")
        return cost
