import logging
logger = logging.getLogger(__name__)

import theano, theano.tensor as T
from blocks.bricks.base import application
import bricks, initialization, util, masonry, graph

class SingleSoftmax(object):
    def __init__(self, input_dim, n_classes, batch_normalize):
        self.input_dim = input_dim
        self.n_classes = n_classes

        self.mlp = masonry.construct_mlp(
            name="mlp",
            activations=[None, bricks.Identity()],
            input_dim=input_dim,
            hidden_dims=[input_dim/2, self.n_classes],
            batch_normalize=batch_normalize,
            weights_init=initialization.Orthogonal(),
            biases_init=initialization.Constant(0))
        self.softmax = bricks.Softmax()

        self.children = [self.mlp, self.softmax]

    def initialize(self):
        for child in self.children:
            child.initialize()

    def emit(self, x, y):
        return self.emit_costs(targets=y, **self.emit_distribution(x))

    def emit_distribution(self, x):
        scope = util.Scope()
        scope.energies = self.mlp.apply(x)
        scope.log_probabilities = self.softmax.log_probabilities(scope.energies)
        scope.probabilities = T.exp(scope.log_probabilities)
        return scope

    def emit_costs(self, targets, energies=None,
                   log_probabilities=None, probabilities=None,
                   **kwargs):
        if not energies:
            if probabilities and not log_probabilities:
                log_probabilities = T.log(probabilities)
            if log_probabilities:
                energies = log_probabilities
            assert energies

        cross_entropy = self.softmax.categorical_cross_entropy(
            targets.flatten(), energies).mean(axis=0)
        error_rate = T.neq(targets, energies.argmax(axis=1)).mean(axis=0)
        return util.Scope(
            cost=cross_entropy.copy(name="cost"),
            cross_entropy=cross_entropy.copy(name="cross_entropy"),
            error_rate=error_rate.copy(name="error_rate"))

    def tag_dropout(self, variables, rng=None, **hyperparameters):
        from blocks.roles import INPUT
        from blocks.filter import VariableFilter
        bricks_ = [brick for brick in util.all_bricks([self.mlp])
                   if isinstance(brick, bricks.Linear)]
        variables = (VariableFilter(roles=[INPUT], bricks=bricks_)
                     (theano.gof.graph.ancestors(variables)))
        graph.add_transform(
            variables,
            graph.DropoutTransform("classifier_dropout", rng=rng),
            reason="regularization")
