import operator
import logging

logger = logging.getLogger(__name__)

import numpy as np

import theano
import theano.tensor as T

from blocks.bricks.base import application, Brick
from blocks.bricks.parallel import Fork, Merge, Parallel
from blocks.bricks.recurrent import BaseRecurrent, recurrent, LSTM
from blocks.bricks import Linear, Rectifier, Initializable, MLP, FeedforwardSequence, Feedforward
from blocks.bricks.conv import ConvolutionalSequence, ConvolutionalActivation, MaxPooling, Flattener
from blocks.initialization import Constant, IsotropicGaussian

from util import NormalizedInitialization

floatX = theano.config.floatX

class Merger(Initializable):
    def __init__(self, area_transform, patch_transform, response_transform,
                 n_spatial_dims, whatwhere_interaction="additive", **kwargs):
        super(Merger, self).__init__(**kwargs)

        self.rectifier = Rectifier()

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
        if self.whatwhere_interaction == "additive":
            self.response_merge.children[0].use_bias = True
        self.response_transform = response_transform

        self.children = [self.rectifier, self.response_merge,
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
            response = self.rectifier.apply(sum(parts))
        elif self.whatwhere_interaction == "multiplicative":
            response = reduce(operator.mul, parts)
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

def construct_cnn_layer(name, layer_spec):
    type_ = layer_spec.pop("type", "conv")
    if type_ == "pool":
        layer = MaxPooling(
            name=name,
            pooling_size=layer_spec.pop("size", (1, 1)),
            step=layer_spec.pop("step", (1, 1)))
    elif type_ == "conv":
        layer = ConvolutionalActivation(
            name=name,
            activation=Rectifier().apply,
            filter_size=layer_spec.pop("size", (1, 1)),
            step=layer_spec.pop("step", (1, 1)),
            num_filters=layer_spec.pop("num_filters", 1),
            border_mode=layer_spec.pop("border_mode", (0, 0)))
    if layer_spec:
        logger.warn("ignoring unknown layer specification keys [%s]"
                    % " ".join(layer_spec.keys()))
    return layer

def construct_cnn(name, layer_specs, n_channels, input_shape):
    cnn = ConvolutionalSequence(
        name=name,
        layers=[construct_cnn_layer("patch_conv_%i" % i,
                                    layer_spec)
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
    cnn.initialize()
    return cnn

def construct_mlp(name, hidden_dims, input_dim, initargs):
    if not hidden_dims:
        return FeedforwardIdentity(dim=input_dim)
    dims = [input_dim] + hidden_dims
    activations = [Rectifier() for i in xrange(len(hidden_dims))]
    mlp = MLP(name=name,
              activations=activations,
              dims=dims,
              **initargs)
    return mlp

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
