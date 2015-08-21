import operator
import logging

logger = logging.getLogger(__name__)

import numpy as np

import theano
import theano.tensor as T

from blocks.bricks.base import application, Brick, lazy
from blocks.roles import add_role, WEIGHT, BIAS
from blocks.bricks.parallel import Fork, Merge, Parallel
from blocks.bricks.recurrent import BaseRecurrent, recurrent, LSTM
from blocks.bricks import Linear, Rectifier, Initializable, MLP, FeedforwardSequence, Feedforward, Bias, Activation
from blocks.initialization import Constant, IsotropicGaussian
from blocks.utils import shared_floatx_nans
from blocks.bricks.conv import Flattener

import blocks.bricks.conv as conv2d
import conv3d

from util import NormalizedInitialization

floatX = theano.config.floatX

class Merger(Initializable):
    def __init__(self, area_transform, patch_transform, response_transform,
                 n_spatial_dims, batch_normalize, whatwhere_interaction="additive",
                 **kwargs):
        super(Merger, self).__init__(**kwargs)

        self.patch_transform = patch_transform
        self.area_transform = area_transform

        self.whatwhere_interaction = whatwhere_interaction
        self.response_merge = Parallel(
            input_names="area patch".split(),
            input_dims=[area_transform.brick.output_dim,
                        patch_transform.brick.output_dim],
            output_dims=2*[response_transform.brick.input_dim],
            prototype=Linear(use_bias=False),
            child_prefix="response_merge")
        self.response_merge_activation = NormalizedActivation(
            shape=[response_transform.brick.input_dim],
            name="response_merge_activation",
            batch_normalize=batch_normalize)
        self.response_transform = response_transform

        self.children = [self.response_merge_activation,
                         self.response_merge,
                         patch_transform.brick,
                         area_transform.brick,
                         response_transform.brick]

    @application(inputs="patch location scale".split(),
                 outputs=['response'])
    def apply(self, patch, location, scale):
        # don't backpropagate through these to avoid the model using
        # the location/scale as merely additional hidden units
        #location, scale = list(map(theano.gradient.disconnected_grad, (location, scale)))
        patch = self.patch_transform(patch)
        area = self.area_transform(T.concatenate([location, scale], axis=1))
        parts = self.response_merge.apply(area, patch)
        if self.whatwhere_interaction == "additive":
            response = sum(parts)
        elif self.whatwhere_interaction == "multiplicative":
            response = reduce(operator.mul, parts)
        response = self.response_merge_activation.apply(response)
        response = self.response_transform(response)
        return response

class Locator(Initializable):
    def __init__(self, input_dim, n_spatial_dims, area_transform,
                 weights_init, biases_init, **kwargs):
        super(Locator, self).__init__(**kwargs)

        self.n_spatial_dims = n_spatial_dims
        self.area_transform = area_transform

        self.locationscale = Linear(
            input_dim=area_transform.brick.output_dim,
            output_dim=2*n_spatial_dims,
            # these are huge reductions in dimensionality, so use
            # normalized initialization to avoid huge values.
            weights_init=NormalizedInitialization(IsotropicGaussian(std=1e-3)),
            biases_init=Constant(0),
            name="locationscale")

        self.children = [self.area_transform.brick, self.locationscale]

    @application(inputs=['h'], outputs=['location', 'scale'])
    def apply(self, h):
        area = self.area_transform(h)
        locationscale = self.locationscale.apply(area)
        return (locationscale[:, :self.n_spatial_dims],
                locationscale[:, self.n_spatial_dims:])

# this belongs on SpatialAttention as a static method, but that breaks pickling
def static_map_to_input_space(location, scale, patch_shape, image_shape):
    # linearly map locations from (-1, 1) to image index space
    location = (location + 1) / 2 * image_shape
    # disallow negative scale
    scale *= scale > 0
    # translate scale such that scale = 0 corresponds to shrinking the
    # full image to fit into the patch, and the model can only zoom in
    # beyond that.  i.e. by default the model looks at a very coarse
    # version of the image, and can choose to selectively refine
    # regions
    scale += patch_shape / image_shape
    return location, scale

class SpatialAttention(Brick):
    def __init__(self, locator, cropper, merger, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)

        self.locator = locator
        self.cropper = cropper
        self.merger = merger

        self.children = [self.locator, self.cropper, self.merger]

    def map_to_input_space(self, location, scale):
        return static_map_to_input_space(
            location, scale,
            T.cast(self.cropper.patch_shape, floatX),
            T.cast(self.cropper.image_shape, floatX))

    def compute_initial_location_scale(self, x):
        location = T.alloc(T.cast(0.0, floatX),
                           x.shape[0], self.cropper.n_spatial_dims)
        scale = T.zeros_like(location)
        return location, scale

    @application(inputs=['x', 'h', 'location_noise', 'scale_noise'], outputs=['u', 'location', 'scale', 'patch', 'mean_savings'])
    def apply(self, x, h, location_noise, scale_noise):
        location, scale = self.locator.apply(h)
        location += location_noise
        scale += scale_noise
        patch, mean_savings = self.crop(x, location, scale)
        u = self.merger.apply(patch, location, scale)
        return u, location, scale, patch, mean_savings

    def crop(self, x, location, scale):
        true_location, true_scale = self.map_to_input_space(location, scale)
        patch, mean_savings = self.cropper.apply(x, true_location, true_scale)
        return patch, mean_savings

    @application(inputs=['x'], outputs="u0 location0 scale0 patch0 mean_savings0".split())
    def compute_initial_input(self, x):
        location, scale = self.compute_initial_location_scale(x)
        patch, mean_savings = self.crop(x, location, scale)
        u = self.merger.apply(patch, location, scale)
        return u, location, scale, patch, mean_savings

class RecurrentAttentionModel(BaseRecurrent):
    def __init__(self, rnn, attention, emitter, **kwargs):
        super(RecurrentAttentionModel, self).__init__(**kwargs)

        # life's too short to try to reconcile the differences between LSTM and plain RNN interfaces
        assert isinstance(rnn, LSTM)

        self.rnn = rnn
        self.attention = attention
        self.emitter = emitter

        self.children = [self.rnn, self.attention, self.emitter]

    def get_dim(self, name):
        try:
            return dict(h=self.rnn.get_dim("states"),
                        c=self.rnn.get_dim("cells"))[name]
        except KeyError:
            return super(RecurrentAttentionModel, self).get_dim(name)

    @recurrent(sequences=["location_noises", "scale_noises"], contexts=['x'], states="h c".split(),
               outputs="h c location scale patch mean_savings".split())
    def apply(self, x, h, c, location_noises, scale_noises):
        u, location, scale, patch, mean_savings = self.attention.apply(x, c, location_noises, scale_noises)
        h, c = self.rnn.apply(inputs=u, iterate=False, states=h, cells=c)
        return h, c, location, scale, patch, mean_savings

    @application(inputs=['x'], outputs="h0 c0 location0 scale0 patch0 mean_savings0".split())
    def compute_initial_state(self, x):
        u, location, scale, patch, mean_savings = self.attention.compute_initial_input(x)
        h, c = self.rnn.initial_states(x.shape[0])
        h, c = self.rnn.apply(inputs=u, iterate=False, states=h, cells=c)
        return h, c, location, scale, patch, mean_savings

def construct_cnn_layer(name, layer_spec, conv_module, ndim, batch_normalize):
    type_ = layer_spec.pop("type", "conv")
    if type_ == "pool":
        layer = conv_module.MaxPooling(
            name=name,
            pooling_size=layer_spec.pop("size", (1,) * ndim),
            step=layer_spec.pop("step", (1,) * ndim))
    elif type_ == "conv":
        border_mode = layer_spec.pop("border_mode", (0,) * ndim)
        if not isinstance(border_mode, basestring):
            # conv bricks barf on list-type shape arguments :/
            border_mode = tuple(border_mode)
        activation = NormalizedActivation(
            name="activation",
            batch_normalize=batch_normalize)
        layer = conv_module.ConvolutionalActivation(
            name=name,
            activation=activation.apply,
            # our activation function will handle the bias
            use_bias=False,
            filter_size=tuple(layer_spec.pop("size", (1,) * ndim)),
            step=tuple(layer_spec.pop("step", (1,) * ndim)),
            num_filters=layer_spec.pop("num_filters", 1),
            border_mode=border_mode)
    if layer_spec:
        logger.warn("ignoring unknown layer specification keys [%s]"
                    % " ".join(layer_spec.keys()))
    return layer

def construct_cnn(name, layer_specs, n_channels, input_shape, batch_normalize):
    ndim = len(input_shape)
    conv_module = {
        2: conv2d,
        3: conv3d,
    }[ndim]
    cnn = conv_module.ConvolutionalSequence(
        name=name,
        layers=[construct_cnn_layer("patch_conv_%i" % i,
                                    layer_spec, ndim=ndim,
                                    conv_module=conv_module,
                                    batch_normalize=batch_normalize)
                for i, layer_spec in enumerate(layer_specs)],
        num_channels=n_channels,
        image_size=tuple(input_shape))
    # ensure output dim is determined
    cnn.push_allocation_config()
    # variance-preserving initialization
    prev_num_filters = n_channels
    for layer in cnn.layers:
        if not hasattr(layer, "filter_size"):
            continue
        layer.weights_init = IsotropicGaussian(
            std=np.sqrt(2./(np.prod(layer.filter_size) * prev_num_filters)))
        layer.biases_init = Constant(0)
        prev_num_filters = layer.num_filters
    # tell the activations what shapes they'll be dealing with
    for layer in cnn.layers:
        # woe is me
        try:
            activation = layer.application_methods[-1].brick
        except:
            continue
        if isinstance(activation, NormalizedActivation):
            activation.shape = layer.get_dim("output")
            activation.broadcastable = [False] + len(input_shape)*[True]
    cnn.initialize()
    return cnn

def construct_mlp(name, hidden_dims, input_dim, initargs, batch_normalize):
    if not hidden_dims:
        return FeedforwardIdentity(dim=input_dim)
    dims = [input_dim] + hidden_dims
    activations = [
        NormalizedActivation(
            shape=[hidden_dim],
            name="activation_%i" % i,
            batch_normalize=batch_normalize)
        for i, hidden_dim in enumerate(hidden_dims)]
    mlp = MLP(name=name,
              activations=activations,
              dims=dims,
              **initargs)
    # biases are handled by our activation function
    for layer in mlp.linear_transformations:
        layer.use_bias = False
    return mlp

class NormalizedActivation(Initializable, Feedforward):
    @lazy(allocation="shape broadcastable".split())
    def __init__(self, shape, broadcastable, activation=None, batch_normalize=False, **kwargs):
        super(NormalizedActivation, self).__init__(**kwargs)
        self.shape = shape
        self.broadcastable = broadcastable
        self.activation = activation or Rectifier()
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
        sequence = []
        if self.batch_normalize:
            sequence.append(Standardization(**arghs))
            sequence.append(SharedScale(
                weights_init=Constant(1),
                **arghs))
        sequence.append(SharedShift(
            biases_init=Constant(0),
            **arghs))
        sequence.append(self.activation)
        self.sequence = FeedforwardSequence([
            brick.apply for brick in sequence
        ], name="ffs")
        self.children = [self.sequence]

    @application(inputs=["input_"], outputs=["output"])
    def apply(self, input_):
        return self.sequence.apply(input_)

    def get_dim(self, name):
        try:
            return dict(input_=self.shape,
                        output=self.shape)
        except:
            return super(NormalizedActivation, self).get_dim(name)

class FeedforwardFlattener(Flattener, Feedforward):
    def __init__(self, input_shape, **kwargs):
        super(FeedforwardFlattener, self).__init__(**kwargs)
        self.input_shape = input_shape

    @property
    def input_dim(self):
        return reduce(operator.mul, self.input_shape)

    @property
    def output_dim(self):
        return reduce(operator.mul, self.input_shape)

class FeedforwardIdentity(Feedforward):
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

class SharedScale(Initializable, Feedforward):
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
        self.w = shared_floatx_nans(parameter_shape, name='w')
        add_role(self.w, WEIGHT)
        self.parameters.append(self.w)
        self.add_auxiliary_variable(self.w.norm(2), name='w_norm')

    def _initialize(self):
        self.weights_init.initialize(self.w, self.rng)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return input_ * T.patternbroadcast(self.w, self.broadcastable)

    def get_dim(self, name):
        if name == 'input_':
            return self.shape
        if name == 'output':
            return self.shape
        return super(SharedScale, self).get_dim(name)

class SharedShift(Initializable, Feedforward):
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
        self.b = shared_floatx_nans(parameter_shape, name='b')
        add_role(self.b, BIAS)
        self.parameters.append(self.b)
        self.add_auxiliary_variable(self.b.norm(2), name='b_norm')

    def _initialize(self):
        self.biases_init.initialize(self.b, self.rng)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return input_ + T.patternbroadcast(self.b, self.broadcastable)

    def get_dim(self, name):
        if name == 'input_':
            return self.shape
        if name == 'output':
            return self.shape
        return super(SharedShift, self).get_dim(name)

# TODO: replacement of batch/population statistics by annotations
# TODO: depends on replacements inside scan
class Standardization(Initializable, Feedforward):
    stats = "mean var".split()

    def __init__(self, shape, broadcastable, alpha=1e-2, **kwargs):
        super(Standardization, self).__init__(**kwargs)
        self.shape = shape
        self.broadcastable = broadcastable
        self.alpha = alpha

    def _allocate(self):
        parameter_shape = [1 if broadcast else dim
                           for dim, broadcast in zip(self.shape, self.broadcastable)]
        self.population_stats = dict(
            (stat, shared_floatx_nans(parameter_shape,
                                      name="population_%s" % stat))
            for stat in self.stats)

    def _initialize(self):
        for stat, initialization in (("mean", 0), ("var",  1)):
            self.population_stats[stat].get_value().fill(initialization)

    @application
    def apply(self, input_):
        aggregate_axes = [0] + [1 + i for i, b in enumerate(self.broadcastable) if b]
        self.batch_stats = dict(
            (stat, getattr(input_, stat)(axis=aggregate_axes,
                                         keepdims=True)[0])
            for stat in self.stats)

        # NOTE: these are unused for now
        self._updates = [(self.population_stats[stat],
                          (1 - self.alpha)*self.population_stats[stat]
                          + self.alpha*self.batch_stats[stat])
                         for stat in self.stats]
        self._replacements = [(self.batch_stats[stat], self.population_stats[stat])
                              for stat in self.stats]

        return ((input_ - self.batch_stats["mean"])
                / (T.sqrt(self.batch_stats["var"] + 1e-8)))
