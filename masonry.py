import numpy as np

import theano
import theano.tensor as T

from blocks.bricks.base import lazy, application, Brick
from blocks.bricks.parallel import Fork, Merge
from blocks.bricks.recurrent import BaseRecurrent, recurrent
from blocks.bricks import Linear, Tanh, Rectifier, Initializable, MLP, Sequence, FeedforwardSequence
from blocks.bricks.conv import Flattener
from blocks.initialization import Constant, IsotropicGaussian

from util import NormalizedInitialization

floatX = theano.config.floatX

class Merger(Initializable):
    def __init__(self, n_spatial_dims, patch_postdim, response_dim,
                 patch_posttransform=None,
                 response_pretransform=None,
                 response_posttransform=None,
                 **kwargs):
        super(Merger, self).__init__(**kwargs)

        self.patch_posttransform = FeedforwardSequence([patch_posttransform, Flattener().apply])

        self.response = Merge(input_names="location scale patch".split(),
                              input_dims=[n_spatial_dims,
                                          n_spatial_dims,
                                          patch_postdim],
                              output_dim=response_dim,
                              prototype=response_pretransform,
                              child_prefix="merger_response")
        self.response.children[0].use_bias = True
        self.response_posttransform = response_posttransform

        self.children = [self.response, self.patch_posttransform, self.response_posttransform]

    @application(inputs="patch location scale".split(),
                 outputs=['response'])
    def apply(self, patch, location, scale):
        # don't backpropagate through these to avoid the model using
        # the location/scale as merely additional hidden units
        #location, scale = list(map(theano.gradient.disconnected_grad, (location, scale)))
        patch = self.patch_posttransform.apply(patch)
        response = self.response.apply(location, scale, patch)
        response = self.response_posttransform.apply(response)
        return response

class Locator(Initializable):
    def __init__(self, input_dim, n_spatial_dims, **kwargs):
        super(Locator, self).__init__(**kwargs)

        # these are huge reductions in dimensionality, so use
        # normalized initialization to avoid huge values.
        prototype = Linear(weights_init=NormalizedInitialization(IsotropicGaussian(std=1e-3)),
                           biases_init=Constant(0))
        self.fork = Fork(output_names=['raw_location', 'raw_scale'],
                         input_dim=input_dim,
                         output_dims=[n_spatial_dims, n_spatial_dims],
                         prototype=prototype,
                         child_prefix="locator_location_scale")

        self.children = [self.fork]

    @application(inputs=['h'], outputs=['location', 'scale'])
    def apply(self, h):
        return self.fork.apply(h)

# this belongs on SpatialAttention as a static method, but that breaks pickling
def static_map_to_image_space(location, scale, patch_shape, image_shape):
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

class SpatialAttention(Initializable):
    def __init__(self, locator, cropper, merger, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)

        self.locator = locator
        self.cropper = cropper
        self.merger = merger

        self.children = [self.locator, self.cropper, self.merger]

    def map_to_image_space(self, location, scale):
        return static_map_to_image_space(
            location, scale,
            T.cast(self.cropper.patch_shape, floatX),
            T.cast(self.cropper.image_shape, floatX))

    def compute_initial_location_scale(self, x):
        location = T.alloc(T.cast(0.0, floatX),
                           x.shape[0], self.cropper.n_spatial_dims)
        scale = T.zeros_like(location)
        return location, scale

    @application(inputs=['x', 'h'], outputs=['u', 'location', 'scale', 'patch', 'mean_savings'])
    def apply(self, x, h):
        location, scale = self.locator.apply(h)
        patch, mean_savings = self.crop(x, location, scale)
        u = self.merger.apply(patch, location, scale)
        return u, location, scale, patch, mean_savings

    def crop(self, x, location, scale):
        true_location, true_scale = self.map_to_image_space(location, scale)
        patch, mean_savings = self.cropper.apply(x, true_location, true_scale)
        return patch, mean_savings

    @application(inputs=['x'], outputs="u0 location0 scale0 patch0 mean_savings0".split())
    def compute_initial_input(self, x):
        location, scale = self.compute_initial_location_scale(x)
        patch, mean_savings = self.crop(x, location, scale)
        u = self.merger.apply(patch, location, scale)
        return u, location, scale, patch, mean_savings

class RecurrentAttentionModel(BaseRecurrent, Initializable):
    def __init__(self, rnn, attention, emitter, **kwargs):
        super(RecurrentAttentionModel, self).__init__(**kwargs)

        self.rnn = rnn
        self.attention = attention
        self.emitter = emitter

        self.children = [self.rnn, self.attention, self.emitter]

    def get_dim(self, name):
        try:
            return dict(h=self.rnn.get_dim("states"))[name]
        except KeyError:
            return super(RecurrentAttentionModel, self).get_dim(name)

    @recurrent(sequences=[''], contexts=['x'], states=['h'], outputs=['h', 'location', 'scale', 'patch', 'mean_savings'])
    def apply(self, x, h):
        u, location, scale, patch, mean_savings = self.attention.apply(x, h)
        h = self.rnn.apply(states=h, inputs=u, iterate=False)
        return h, location, scale, patch, mean_savings

    @application(inputs=['x'], outputs=['h0', 'location0', 'scale0', 'patch0', 'mean_savings0'])
    def compute_initial_state(self, x):
        u, location, scale, patch, mean_savings = self.attention.compute_initial_input(x)
        h = self.rnn.apply(states=self.rnn.initial_states(state_name="states", batch_size=x.shape[0]),
                           inputs=u, iterate=False)
        return h, location, scale, patch, mean_savings
