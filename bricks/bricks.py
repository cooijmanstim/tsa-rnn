# -*- coding: utf-8 -*-
import operator, logging, collections
from collections import OrderedDict

import numpy
import theano
import theano.tensor as T

from blocks.bricks.base import application, lazy
from blocks.graph import add_annotation
from blocks.roles import add_role, WEIGHT, BIAS, INITIAL_STATE, VariableRole
from blocks.utils import shared_floatx_nans, shared_floatx_zeros

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
        # batch_normalize may be a bool or a dict of kwargs to the BatchNormalization constructor
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
            aarghs = dict(arghs)
            if isinstance(self.batch_normalize, collections.Mapping):
                aarghs.update(self.batch_normalize)
            self.sequence.append(BatchNormalization(**aarghs))
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

    def __init__(self, shape, broadcastable, alpha=1e-2, epsilon=1e-8, **kwargs):
        super(BatchNormalization, self).__init__(**kwargs)
        self.shape = shape
        self.broadcastable = list(broadcastable)
        self.alpha = alpha
        self.epsilon = epsilon

    def _allocate(self):
        parameter_shape = [1] + [1 if broadcast else dim for dim, broadcast
                                 in zip(self.shape, self.broadcastable)]
        self.population_stats = dict(
            (stat, self.annotated_statistic(
                shared_floatx_nans(parameter_shape,
                                   name="population_%s" % stat)))
            for stat in self.stats)
        self.gamma = shared_floatx_nans(parameter_shape, name='gamma')
        self.beta = shared_floatx_nans(parameter_shape, name='beta')
        add_role(self.gamma, WEIGHT)
        add_role(self.beta, BIAS)
        self.parameters.append(self.gamma)
        self.parameters.append(self.beta)

    def annotated_statistic(self, var):
        add_annotation(var, self)
        var.tag.batch_normalization_brick = self
        return var

    def _initialize(self):
        zero, one = initialization.Constant(0.), initialization.Constant(1.)
        zero.initialize(self.population_stats["mean"], rng=self.rng)
        one.initialize(self.population_stats["var"], rng=self.rng)
        one.initialize(self.gamma, rng=self.rng)
        zero.initialize(self.beta, rng=self.rng)

    @application(inputs=["input_"], outputs=["output"])
    def apply(self, input_):
        aggregate_axes = [0] + [1 + i for i, b in enumerate(self.broadcastable) if b]
        # NOTE: don't put batch_stats on self because apply may be
        # called multiple times
        batch_stats = dict(
            (stat, getattr(input_, stat)(axis=aggregate_axes,
                                         keepdims=True))
            for stat in self.stats)

        for stat, role in self.roles.items():
            graph.add_transform([batch_stats[stat]],
                                graph.ConstantTransform(
                                    # adding zero to ensure it's a TensorType(float32, row)
                                    # just like the corresponding batch_stat, rather than a
                                    # CudaNdarray(float32, row).  -__-
                                    0 + T.patternbroadcast(
                                        self.population_stats[stat],
                                        [True] + self.broadcastable)),
                                reason="population_normalization")

            # make the batch statistics identifiable to get_updates() below
            add_role(batch_stats[stat], self.roles[stat])
            batch_stats[stat] = self.annotated_statistic(batch_stats[stat])

        gamma = T.patternbroadcast(self.gamma, [True] + self.broadcastable)
        beta = T.patternbroadcast(self.beta, [True] + self.broadcastable)
        return theano.tensor.nnet.bn.batch_normalization(
            inputs=input_, gamma=gamma, beta=beta,
            mean=batch_stats["mean"],
            std=T.sqrt(batch_stats["var"] + self.epsilon))

    @staticmethod
    def get_updates(variables):
        # this is fugly because we must get the batch stats from the
        # graph so we get the ones that are *actually being used in
        # the computation* after graph transforms have been applied
        updates = []
        variables = graph.deep_ancestors(variables)
        for stat, role in BatchNormalization.roles.items():
            from blocks.roles import has_roles
            batch_stats = [var for var in variables if has_roles(var, [role])]
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

from theano import tensor
from blocks.bricks import Logistic, Tanh, Initializable
from blocks.bricks.recurrent import BaseRecurrent, recurrent
class GatedRecurrent(BaseRecurrent, Initializable):
    u"""Gated recurrent neural network.

    Gated recurrent neural network (GRNN) as introduced in [CvMG14]_. Every
    unit of a GRNN is equipped with update and reset gates that facilitate
    better gradient propagation.

    Parameters
    ----------
    dim : int
        The dimension of the hidden state.
    activation : :class:`.Brick` or None
        The brick to apply as activation. If ``None`` a
        :class:`.Tanh` brick is used.
    gate_activation : :class:`.Brick` or None
        The brick to apply as activation for gates. If ``None`` a
        :class:`.Logistic` brick is used.

    Notes
    -----
    See :class:`.Initializable` for initialization parameters.

    .. [CvMG14] Kyunghyun Cho, Bart van Merriënboer, Çağlar Gülçehre,
        Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua
        Bengio, *Learning Phrase Representations using RNN Encoder-Decoder
        for Statistical Machine Translation*, EMNLP (2014), pp. 1724-1734.

    """
    @lazy(allocation=['dim'])
    def __init__(self, dim, activation=None, gate_activation=None,
                 **kwargs):
        super(GatedRecurrent, self).__init__(**kwargs)
        self.dim = dim

        if not activation:
            activation = Tanh()
        if not gate_activation:
            gate_activation = Logistic()
        self.activation = activation
        self.gate_activation = gate_activation

        self.children = [activation, gate_activation]

    @property
    def state_to_state(self):
        return self.parameters[0]

    @property
    def state_to_gates(self):
        return self.parameters[1]

    def get_dim(self, name):
        if name == 'mask':
            return 0
        if name in ['inputs', 'states']:
            return self.dim
        if name == 'gate_inputs':
            return 2 * self.dim
        return super(GatedRecurrent, self).get_dim(name)

    def _allocate(self):
        self.parameters.append(shared_floatx_nans((self.dim, self.dim),
                               name='state_to_state'))
        self.parameters.append(shared_floatx_nans((self.dim, 2 * self.dim),
                               name='state_to_gates'))
        self.parameters.append(shared_floatx_zeros((self.dim,),
                               name="initial_state"))
        for i in range(2):
            if self.parameters[i]:
                add_role(self.parameters[i], WEIGHT)
        add_role(self.parameters[2], INITIAL_STATE)

    def _initialize(self):
        self.weights_init.initialize(self.state_to_state, self.rng)
        state_to_update = self.weights_init.generate(
            self.rng, (self.dim, self.dim))
        state_to_reset = self.weights_init.generate(
            self.rng, (self.dim, self.dim))
        self.state_to_gates.set_value(
            numpy.hstack([state_to_update, state_to_reset]))

    @recurrent(sequences=['mask', 'inputs', 'gate_inputs'],
               states=['states'], outputs=['states'], contexts=[])
    def apply(self, inputs, gate_inputs, states, mask=None):
        """Apply the gated recurrent transition.

        Parameters
        ----------
        states : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of current states in the shape
            (batch_size, dim). Required for `one_step` usage.
        inputs : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of inputs in the shape (batch_size,
            dim)
        gate_inputs : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of inputs to the gates in the
            shape (batch_size, 2 * dim).
        mask : :class:`~tensor.TensorVariable`
            A 1D binary array in the shape (batch,) which is 1 if there is
            data available, 0 if not. Assumed to be 1-s only if not given.

        Returns
        -------
        output : :class:`~tensor.TensorVariable`
            Next states of the network.

        """
        # use wrapped self.state_to_state so we can replace it
        # without the brick losing reference to the parameter
        state_to_state = self.state_to_state.copy(name="weight_noise_goes_here")

        gate_values = self.gate_activation.apply(
            states.dot(self.state_to_gates) + gate_inputs)
        update_values = gate_values[:, :self.dim]
        reset_values = gate_values[:, self.dim:]
        states_reset = states * reset_values
        next_states = self.activation.apply(
            states_reset.dot(state_to_state) + inputs)
        next_states = (next_states * update_values +
                       states * (1 - update_values))
        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
        return next_states

    @application(outputs=apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        return [tensor.repeat(self.parameters[2][None, :], batch_size, 0)]


class LSTM(BaseRecurrent, Initializable):
    u"""Long Short Term Memory.

    Every unit of an LSTM is equipped with input, forget and output gates.
    This implementation is based on code by Mohammad Pezeshki that
    implements the architecture used in [GSS03]_ and [Grav13]_. It aims to
    do as many computations in parallel as possible and expects the last
    dimension of the input to be four times the output dimension.

    Unlike a vanilla LSTM as described in [HS97]_, this model has peephole
    connections from the cells to the gates. The output gates receive
    information about the cells at the current time step, while the other
    gates only receive information about the cells at the previous time
    step. All 'peephole' weight matrices are diagonal.

    .. [GSS03] Gers, Felix A., Nicol N. Schraudolph, and Jürgen
        Schmidhuber, *Learning precise timing with LSTM recurrent
        networks*, Journal of Machine Learning Research 3 (2003),
        pp. 115-143.
    .. [Grav13] Graves, Alex, *Generating sequences with recurrent neural
        networks*, arXiv preprint arXiv:1308.0850 (2013).
    .. [HS97] Sepp Hochreiter, and Jürgen Schmidhuber, *Long Short-Term
        Memory*, Neural Computation 9(8) (1997), pp. 1735-1780.

    Parameters
    ----------
    dim : int
        The dimension of the hidden state.
    activation : :class:`.Brick`, optional
        The activation function. The default and by far the most popular
        is :class:`.Tanh`.

    Notes
    -----
    See :class:`.Initializable` for initialization parameters.

    """
    @lazy(allocation=['dim'])
    def __init__(self, dim, activation=None, **kwargs):
        super(LSTM, self).__init__(**kwargs)
        self.dim = dim

        if not activation:
            activation = Tanh()
        self.children = [activation]

    def get_dim(self, name):
        if name == 'inputs':
            return self.dim * 4
        if name in ['states', 'cells']:
            return self.dim
        if name == 'mask':
            return 0
        return super(LSTM, self).get_dim(name)

    def _allocate(self):
        self.W_state = shared_floatx_nans((self.dim, 4*self.dim),
                                          name='W_state')
        self.W_cell_to_in = shared_floatx_nans((self.dim,),
                                               name='W_cell_to_in')
        self.W_cell_to_forget = shared_floatx_nans((self.dim,),
                                                   name='W_cell_to_forget')
        self.W_cell_to_out = shared_floatx_nans((self.dim,),
                                                name='W_cell_to_out')
        # The underscore is required to prevent collision with
        # the `initial_state` application method
        self.initial_state_ = shared_floatx_zeros((self.dim,),
                                                  name="initial_state")
        self.initial_cells = shared_floatx_zeros((self.dim,),
                                                 name="initial_cells")
        add_role(self.W_state, WEIGHT)
        add_role(self.W_cell_to_in, WEIGHT)
        add_role(self.W_cell_to_forget, WEIGHT)
        add_role(self.W_cell_to_out, WEIGHT)
        add_role(self.initial_state_, INITIAL_STATE)
        add_role(self.initial_cells, INITIAL_STATE)

        self.parameters = [
            self.W_state, self.W_cell_to_in, self.W_cell_to_forget,
            self.W_cell_to_out, self.initial_state_, self.initial_cells]

    def _initialize(self):
        for weights in self.parameters[:4]:
            self.weights_init.initialize(weights, self.rng)

    @recurrent(sequences=['inputs', 'mask'], states=['states', 'cells'],
               contexts=[], outputs=['states', 'cells'])
    def apply(self, inputs, states, cells, mask=None):
        """Apply the Long Short Term Memory transition.

        Parameters
        ----------
        states : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of current states in the shape
            (batch_size, features). Required for `one_step` usage.
        cells : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of current cells in the shape
            (batch_size, features). Required for `one_step` usage.
        inputs : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of inputs in the shape (batch_size,
            features * 4). The `inputs` needs to be four times the
            dimension of the LSTM brick to insure each four gates receive
            different transformations of the input. See [Grav13]_
            equations 7 to 10 for more details.
        mask : :class:`~tensor.TensorVariable`
            A 1D binary array in the shape (batch,) which is 1 if there is
            data available, 0 if not. Assumed to be 1-s only if not given.

        .. [Grav13] Graves, Alex, *Generating sequences with recurrent
            neural networks*, arXiv preprint arXiv:1308.0850 (2013).

        Returns
        -------
        states : :class:`~tensor.TensorVariable`
            Next states of the network.
        cells : :class:`~tensor.TensorVariable`
            Next cell activations of the network.

        """
        
        # use wrapped self.state_to_state so we can replace it
        # without the brick losing reference to the parameter
        W_state = self.W_state.copy(name="weight_noise_goes_here")
        
        def slice_last(x, no):
            return x[:, no*self.dim: (no+1)*self.dim]

        nonlinearity = self.children[0].apply

        activation = tensor.dot(states, W_state) + inputs
        in_gate = tensor.nnet.sigmoid(slice_last(activation, 0) +
                                      cells * self.W_cell_to_in)
        forget_gate = tensor.nnet.sigmoid(slice_last(activation, 1) +
                                          cells * self.W_cell_to_forget +
                                          # forget gate bias
                                          1)
        next_cells = (forget_gate * cells +
                      in_gate * nonlinearity(slice_last(activation, 2)))
        out_gate = tensor.nnet.sigmoid(slice_last(activation, 3) +
                                       next_cells * self.W_cell_to_out)
        next_states = out_gate * nonlinearity(next_cells)

        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
            next_cells = (mask[:, None] * next_cells +
                          (1 - mask[:, None]) * cells)

        return next_states, next_cells

    @application(outputs=apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        return [tensor.repeat(self.initial_state_[None, :], batch_size, 0),
                tensor.repeat(self.initial_cells[None, :], batch_size, 0)]
