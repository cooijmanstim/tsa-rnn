import operator
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

import theano
import theano.tensor as T

from blocks.bricks.base import application, Brick
from blocks.bricks import Initializable

import bricks
import initialization

import masonry

floatX = theano.config.floatX

class Merger(Initializable):
    def __init__(self, area_transform, patch_transform, response_transform,
                 n_spatial_dims, batch_normalize, whatwhere_interaction="additive",
                 **kwargs):
        super(Merger, self).__init__(**kwargs)

        self.patch_transform = patch_transform
        self.area_transform = area_transform

        self.whatwhere_interaction = whatwhere_interaction
        self.response_merge = bricks.Parallel(
            input_names="area patch".split(),
            input_dims=[area_transform.brick.output_dim,
                        patch_transform.brick.output_dim],
            output_dims=2*[response_transform.brick.input_dim],
            prototype=bricks.Linear(use_bias=False),
            child_prefix="response_merge")
        self.response_merge_activation = bricks.NormalizedActivation(
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
                 weights_init, biases_init, location_std, scale_std, **kwargs):
        super(Locator, self).__init__(**kwargs)

        self.n_spatial_dims = n_spatial_dims
        self.area_transform = area_transform

        self.locationscale = bricks.Linear(
            input_dim=area_transform.brick.output_dim,
            output_dim=2*n_spatial_dims,
            # these are huge reductions in dimensionality, so use
            # normalized initialization to avoid huge values.
            weights_init=initialization.NormalizedInitialization(
                initialization.IsotropicGaussian(std=1e-3)),
            biases_init=initialization.Constant(0),
            name="locationscale")

        self.T_rng = theano.sandbox.rng_mrg.MRG_RandomStreams(12345)
        self.location_std = location_std
        self.scale_std = scale_std

        self.children = [self.area_transform.brick, self.locationscale]

    @application(inputs=['h'], outputs=['location', 'scale'])
    def apply(self, h):
        area = self.area_transform(h)
        locationscale = self.locationscale.apply(area)
        location, scale = (locationscale[:, :self.n_spatial_dims],
                           locationscale[:, self.n_spatial_dims:])
        location += self.T_rng.normal(location.shape, std=self.location_std)
        scale += self.T_rng.normal(scale.shape, std=self.scale_std)
        return location, scale

# this belongs on RecurrentAttentionModel as a static method, but that breaks pickling
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

class RecurrentAttentionModel(bricks.BaseRecurrent):
    def __init__(self, rnn, locator, cropper, merger, emitter, batch_normalize, attention_state_name, h2h_transforms, **kwargs):
        super(RecurrentAttentionModel, self).__init__(**kwargs)

        self.locator = locator
        self.cropper = cropper
        self.merger = merger
        self.emitter = emitter

        self.rnn = rnn

        # a dict mapping recurrentstack state names to applications
        self.h2h_transforms = dict(h2h_transforms)
        self.identity = bricks.Identity()

        # name of the RNN state that determines the parameters of the next glimpse
        self.attention_state_name = attention_state_name

        self.children = ([self.rnn,
                          self.locator, self.cropper, self.merger,
                          self.emitter, self.identity]
                         + list(self.h2h_transforms.values()))

        # states aren't known until now
        self.apply.outputs = self.rnn.apply.outputs
        self.compute_initial_state.outputs = self.rnn.apply.outputs

    def get_dim(self, name):
        try:
            return self.rnn.get_dim(name)
        except:
            return super(RecurrentAttentionModel, self).get_dim(name)

    @application
    def apply(self, x, **states):
        location, scale = self.locator.apply(states[self.attention_state_name])
        patch = self.crop(x, location, scale)
        u = self.merger.apply(patch, location, scale)
        states = OrderedDict((key, self.h2h_transforms.get(key, self.identity).apply(value))
                             for key, value in states.items())
        states = self.rnn.apply(inputs=u, iterate=False, as_dict=True, **states)
        return tuple(states.values())

    @application
    def compute_initial_state(self, x):
        initial_states = self.rnn.initial_states(x.shape[0], as_dict=True)
        # condition on initial shrink-to-fit patch
        location = T.alloc(T.cast(0.0, floatX),
                           x.shape[0], self.cropper.n_spatial_dims)
        scale = T.zeros_like(location)
        patch = self.crop(x, location, scale)
        u = self.merger.apply(patch, location, scale)
        conditioned_states = self.rnn.apply(as_dict=True, inputs=u, iterate=False, **initial_states)
        return tuple(conditioned_states.values())

    def crop(self, x, location, scale):
        true_location, true_scale = self.map_to_input_space(location, scale)
        patch = self.cropper.apply(x, true_location, true_scale)
        self.add_auxiliary_variable(location, name="location")
        self.add_auxiliary_variable(scale, name="scale")
        self.add_auxiliary_variable(true_location, name="true_location")
        self.add_auxiliary_variable(true_scale, name="true_scale")
        self.add_auxiliary_variable(patch, name="patch")
        return patch

    def map_to_input_space(self, location, scale):
        return static_map_to_input_space(
            location, scale,
            T.cast(self.cropper.patch_shape, floatX),
            T.cast(self.cropper.image_shape, floatX))
