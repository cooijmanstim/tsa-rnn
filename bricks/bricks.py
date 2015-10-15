import operator, logging
from collections import OrderedDict

import theano
import theano.tensor as T

from blocks.bricks.base import application, lazy
from blocks.roles import add_role, WEIGHT, BIAS, VariableRole
from blocks.utils import shared_floatx_nans

import blocks.bricks as bricks
from blocks.bricks.conv import Flattener

import initialization, graph, util

logger = logging.getLogger(__name__)

class NormalizedActivation(bricks.Initializable, bricks.Feedforward):
    @lazy(allocation="shape broadcastable".split())
    def __init__(self, shape, broadcastable, activation=None, batch_normalize=False, **kwargs):
        super(NormalizedActivation, self).__init__(**kwargs)
        self.shape = shape
        self.broadcastable = broadcastable
        self.activation = activation or bricks.Rectifier()
        self.batch_normalize = batch_normalize

    @property
    def broadcastable(self):
        return self._broadcastable or [False]*len(self.shape)

    @broadcastable.setter
    def broadcastable(self, broadcastable):
        self._broadcastable = broadcastable

    def _allocate(self):
        arghs = dict(shape=self.shape,
                     broadcastable=self.broadcastable)
        self.sequence = []
        if self.batch_normalize:
            self.sequence.append(BatchNormalization(**arghs))
        else:
            self.sequence.append(SharedShift(
                biases_init=initialization.Constant(0),
                **arghs))
        self.sequence.append(self.activation)
        self.children = list(self.sequence)

    @application
    def apply(self, x):
        for brick in self.sequence:
            x = brick.apply(x)
        return x

    def get_dim(self, name):
        try:
            return dict(input_=self.shape,
                        output=self.shape)
        except:
            return super(NormalizedActivation, self).get_dim(name)

class FeedforwardFlattener(Flattener, bricks.Feedforward):
    def __init__(self, input_shape, **kwargs):
        super(FeedforwardFlattener, self).__init__(**kwargs)
        self.input_shape = input_shape

    @property
    def input_dim(self):
        return reduce(operator.mul, self.input_shape)

    @property
    def output_dim(self):
        return reduce(operator.mul, self.input_shape)

class FeedforwardIdentity(bricks.Feedforward):
    def __init__(self, dim, **kwargs):
        super(FeedforwardIdentity, self).__init__(**kwargs)
        self.dim = dim

    @property
    def input_dim(self):
        return self.dim

    @property
    def output_dim(self):
        return self.dim

    @application(inputs=["x"], outputs=["x"])
    def apply(self, x):
        return x

class SharedScale(bricks.Initializable, bricks.Feedforward):
    """
    Element-wise scaling with optional parameter-sharing across axes.
    """
    @lazy(allocation="shape broadcastable".split())
    def __init__(self, shape, broadcastable, **kwargs):
        super(SharedScale, self).__init__(**kwargs)
        self.shape = shape
        self.broadcastable = broadcastable

    def _allocate(self):
        parameter_shape = [1 if broadcast else dim
                           for dim, broadcast in zip(self.shape, self.broadcastable)]
        self.gamma = shared_floatx_nans(parameter_shape, name='gamma')
        add_role(self.gamma, WEIGHT)
        self.parameters.append(self.gamma)
        self.add_auxiliary_variable(self.gamma.norm(2), name='gamma_norm')

    def _initialize(self):
        self.weights_init.initialize(self.gamma, self.rng)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return input_ * T.patternbroadcast(self.gamma, self.broadcastable)

    def get_dim(self, name):
        if name == 'input_':
            return self.shape
        if name == 'output':
            return self.shape
        return super(SharedScale, self).get_dim(name)

class SharedShift(bricks.Initializable, bricks.Feedforward):
    """
    Element-wise bias with optional parameter-sharing across axes.
    """
    @lazy(allocation="shape broadcastable".split())
    def __init__(self, shape, broadcastable, **kwargs):
        super(SharedShift, self).__init__(**kwargs)
        self.shape = shape
        self.broadcastable = broadcastable

    def _allocate(self):
        parameter_shape = [1 if broadcast else dim
                           for dim, broadcast in zip(self.shape, self.broadcastable)]
        self.beta = shared_floatx_nans(parameter_shape, name='beta')
        add_role(self.beta, BIAS)
        self.parameters.append(self.beta)
        self.add_auxiliary_variable(self.beta.norm(2), name='beta_norm')

    def _initialize(self):
        self.biases_init.initialize(self.beta, self.rng)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return input_ + T.patternbroadcast(self.beta, self.broadcastable)

    def get_dim(self, name):
        if name == 'input_':
            return self.shape
        if name == 'output':
            return self.shape
        return super(SharedShift, self).get_dim(name)

class BatchMeanRole(VariableRole):
    pass
class BatchVarRole(VariableRole):
    pass

class BatchNormalization(bricks.Initializable, bricks.Feedforward):
    stats = "mean var".split()
    roles = dict(
        mean=BatchMeanRole(),
        var=BatchVarRole())

    def __init__(self, shape, broadcastable, alpha=1e-2, **kwargs):
        super(BatchNormalization, self).__init__(**kwargs)
        self.shape = shape
        self.broadcastable = list(broadcastable)
        self.alpha = alpha

    def _allocate(self):
        parameter_shape = [1] + [1 if broadcast else dim for dim, broadcast
                                 in zip(self.shape, self.broadcastable)]
        self.population_stats = dict(
            (stat, shared_floatx_nans(parameter_shape,
                                      name="population_%s" % stat))
            for stat in self.stats)
        self.gamma = shared_floatx_nans(parameter_shape, name='gamma')
        self.beta = shared_floatx_nans(parameter_shape, name='beta')
        add_role(self.gamma, WEIGHT)
        add_role(self.beta, BIAS)
        self.parameters.append(self.gamma)
        self.parameters.append(self.beta)

    def _initialize(self):
        zero, one = initialization.Constant(0.), initialization.Constant(1.)
        zero.initialize(self.population_stats["mean"], rng=self.rng)
        one.initialize(self.population_stats["var"], rng=self.rng)
        one.initialize(self.gamma, rng=self.rng)
        zero.initialize(self.beta, rng=self.rng)

    @application(inputs=["input_"], outputs=["output"])
    def apply(self, input_):
        aggregate_axes = [0] + [1 + i for i, b in enumerate(self.broadcastable) if b]
        self.batch_stats = dict(
            (stat, getattr(input_, stat)(axis=aggregate_axes,
                                         keepdims=True))
            for stat in self.stats)

        for stat, role in self.roles.items():
            graph.add_transform([self.batch_stats[stat]],
                                graph.ConstantTransform(
                                    # adding zero to ensure it's a TensorType(float32, row)
                                    # just like the corresponding batch_stat, rather than a
                                    # CudaNdarray(float32, row).  -__-
                                    0 + T.patternbroadcast(
                                        self.population_stats[stat],
                                        [True] + self.broadcastable)),
                                reason="population_normalization")

            # make the batch statistics identifiable to get_updates() below
            add_role(self.batch_stats[stat], self.roles[stat])
            self.batch_stats[stat].tag.batch_normalization_brick = self

        gamma = T.patternbroadcast(self.gamma, [True] + self.broadcastable)
        beta = T.patternbroadcast(self.beta, [True] + self.broadcastable)
        return theano.tensor.nnet.bn.batch_normalization(
            inputs=input_, gamma=gamma, beta=beta,
            mean=self.batch_stats["mean"],
            std=T.sqrt(self.batch_stats["var"]) + 1e-8)

    @staticmethod
    def get_updates(variables):
        # this is fugly because we must get the batch stats from the
        # graph so we get the ones that are *actually being used in
        # the computation* after graph transforms have been applied
        updates = []
        variables = theano.gof.graph.ancestors(variables)
        for stat, role in BatchNormalization.roles.items():
            from blocks.filter import VariableFilter
            batch_stats = VariableFilter(roles=[role])(variables)
            batch_stats = util.dedup(batch_stats, equal=util.equal_computations)

            batch_stats_by_brick = OrderedDict()
            for batch_stat in batch_stats:
                brick = batch_stat.tag.batch_normalization_brick
                population_stat = brick.population_stats[stat]
                batch_stats_by_brick.setdefault(brick, []).append(batch_stat)

            for brick, batch_stats in batch_stats_by_brick.items():
                population_stat = brick.population_stats[stat]
                if len(batch_stats) > 1:
                    # makes sense for recurrent structures
                    logger.warning("averaging multiple population statistic estimates to update %s: %s"
                                   % (util.get_path(population_stat), batch_stats))
                batch_stat = T.stack(batch_stats).mean(axis=0)
                updates.append((population_stat,
                                (1 - brick.alpha) * population_stat
                                + brick.alpha * batch_stat))
        return updates
