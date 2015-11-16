import logging
import numpy as np
import theano, theano.tensor as T
from blocks.bricks.base import application
import blocks.bricks.conv as conv2d, conv3d
import util, bricks, initialization, masonry, graph

logger = logging.getLogger(__name__)
floatX = theano.config.floatX

# this belongs on RecurrentAttentionModel as a static method, but that breaks pickling
def static_map_to_input_space(location, scale, patch_shape, image_shape, max_coverage=1.):
    # linearly map locations from (-1, 1) to image index space
    location = (location + 1) / 2 * image_shape
    # disallow negative scale
    exp = np.exp if isinstance(scale, np.ndarray) else T.exp
    scale = exp(scale)
    # translate scale such that scale = 0 corresponds to shrinking the
    # full image to fit into the patch, and the model can only zoom in
    # beyond that.  i.e. by default the model looks at a very coarse
    # version of the image, and can choose to selectively refine
    # regions
    prior_scale = patch_shape / image_shape / max_coverage
    scale *= prior_scale
    return location, scale

class RecurrentAttentionModel(object):
    def __init__(self, hidden_dim, cropper,
                 attention_state_name, hyperparameters, **kwargs):
        # we're no longer a brick, but we still need to make sure we
        # initialize everything
        self.children = []

        self.rnn = bricks.RecurrentStack(
            [bricks.GatedRecurrent(activation=bricks.Tanh(), dim=hidden_dim),
             bricks.GatedRecurrent(activation=bricks.Tanh(), dim=hidden_dim)],
            weights_init=initialization.NormalizedInitialization(
                initialization.IsotropicGaussian()),
            biases_init=initialization.Constant(0))

        # name of the RNN state that determines the parameters of the next glimpse
        self.attention_state_name = attention_state_name

        self.cropper = cropper
        self.construct_locator(**hyperparameters)
        self.construct_merger(**hyperparameters)

        self.embedder = bricks.Linear(
            name="embedder",
            input_dim=self.response_mlp.output_dim,
            output_dim=self.rnn.get_dim("inputs") + self.rnn.get_dim("gate_inputs"),
            use_bias=True,
            weights_init=initialization.Orthogonal(),
            biases_init=initialization.Constant(0))

        self.children.extend([self.rnn, self.cropper, self.embedder])

    def initialize(self):
        for child in self.children:
            child.initialize()
        for gru in self.rnn.transitions:
            initialization.Identity().initialize(gru.state_to_state, gru.rng)

    @util.checkargs
    def construct_merger(self, n_spatial_dims, n_channels,
                         patch_shape, patch_cnn_spec, patch_mlp_spec,
                         merge_mlp_spec, response_mlp_spec,
                         batch_normalize, batch_normalize_patch,
                         task_name,
                         hyperparameters, **kwargs):
        # construct patch interpretation network
        if task_name == "featurelevel_ucf101":
            n_channels = 512 + 4096
            assert patch_shape[1:] == [1, 1]
            patch_shape = (patch_shape[0], 1)
        patch_transforms = []
        if patch_cnn_spec == "pretrained":
            import pretrained
            patch_transforms.append(pretrained.get_patch_transform(**hyperparameters))
            shape = patch_transforms[-1].get_dim("output")
        elif patch_cnn_spec:
            patch_transforms.append(masonry.construct_cnn(
                name="patch_cnn",
                layer_specs=patch_cnn_spec,
                input_shape=patch_shape,
                n_channels=n_channels,
                batch_normalize=batch_normalize_patch))
            shape = patch_transforms[-1].get_dim("output")
        patch_transforms.append(bricks.FeedforwardFlattener(input_shape=shape))
        if patch_mlp_spec:
            patch_transforms.append(masonry.construct_mlp(
                name="patch_mlp",
                hidden_dims=patch_mlp_spec,
                input_dim=patch_transforms[-1].output_dim,
                weights_init=initialization.Orthogonal(),
                biases_init=initialization.Constant(0),
                batch_normalize=batch_normalize_patch))
        self.patch_transform = bricks.FeedforwardSequence(
            [brick.apply for brick in patch_transforms], name="ffs")

        # construct theta interpretation network
        self.merge_mlp = masonry.construct_mlp(
            name="merge_mlp",
            input_dim=2*n_spatial_dims,
            hidden_dims=merge_mlp_spec,
            weights_init=initialization.Orthogonal(),
            biases_init=initialization.Constant(0),
            batch_normalize=batch_normalize)

        self.response_mlp = masonry.construct_mlp(
            name="response_mlp",
            hidden_dims=response_mlp_spec,
            input_dim=self.patch_transform.output_dim + self.merge_mlp.output_dim,
            weights_init=initialization.Orthogonal(),
            biases_init=initialization.Constant(0),
            batch_normalize=batch_normalize)

        self.children.extend([
            self.patch_transform,
            self.merge_mlp,
            self.response_mlp])

    @util.checkargs
    def construct_locator(self, locate_mlp_spec, n_spatial_dims,
                          batch_normalize, **kwargs):
        self.n_spatial_dims = n_spatial_dims

        self.locate_mlp = masonry.construct_mlp(
            name="locate_mlp",
            input_dim=self.get_dim(self.attention_state_name),
            hidden_dims=locate_mlp_spec,
            weights_init=initialization.Orthogonal(),
            biases_init=initialization.Constant(0),
            batch_normalize=batch_normalize)
        self.theta_from_area = bricks.Linear(
            input_dim=self.locate_mlp.output_dim,
            output_dim=2*n_spatial_dims,
            name="theta_from_area",
            # normalize columns because the fan-in is large
            weights_init=initialization.NormalizedInitialization(
                initialization.IsotropicGaussian()),
            # initialize location biases to zero and scale biases to a
            # positive value so the model will zoom in by default
            biases_init=initialization.Constant(np.array(
                [0.] * n_spatial_dims + [.1] * n_spatial_dims)))

        self.children.extend([
            self.locate_mlp,
            self.theta_from_area])

    def get_dim(self, name):
        return self.rnn.get_dim(name)

    def apply(self, scope, initial=False):
        if initial:
            if isinstance(scope.x_shape, tuple):
                # for featurelevel_UCF101
                batch_size = scope.x_shape[0].shape[0]
            else:
                batch_size = scope.x_shape.shape[0]
            # condition on initial shrink-to-fit patch
            scope.raw_location = T.alloc(T.cast(0.0, floatX),
                                         batch_size, self.cropper.n_spatial_dims)
            scope.raw_scale = T.zeros_like(scope.raw_location)
            scope.previous_states = self.rnn.initial_states(batch_size, as_dict=True)
        else:
            self.locate(scope)
        self.map_to_input_space(scope)
        scope.patch, scope.savings = self.cropper.apply(scope.x, scope.x_shape, scope.true_location, scope.true_scale)
        graph.add_transform([scope.patch],
                            graph.WhiteNoiseTransform("patch_std"),
                            reason="regularization")
        scope.response = self.response_mlp.apply(
            T.concatenate([
                self.patch_transform.apply(scope.patch),
                self.merge_mlp.apply(
                    T.concatenate([
                        scope.raw_location,
                        scope.raw_scale
                    ], axis=1)),
            ], axis=1))
        embedding = self.embedder.apply(scope.response)
        hidden_dim = self.rnn.get_dim("inputs")
        scope.rnn_inputs = dict(
            inputs=embedding[:, :hidden_dim],
            gate_inputs=embedding[:, hidden_dim:],
            **scope.previous_states)
        scope.rnn_outputs = self.rnn.apply(iterate=False, as_dict=True,
                                           **scope.rnn_inputs)
        return scope

    def locate(self, scope, initial=False):
        scope.theta = self.theta_from_area.apply(
            self.locate_mlp.apply(
                scope.previous_states[self.attention_state_name]))
        location, scale = (scope.theta[:, :self.n_spatial_dims],
                           scope.theta[:, self.n_spatial_dims:])
        graph.add_transform([location],
                            graph.WhiteNoiseTransform("location_std"),
                            reason="regularization")
        graph.add_transform([scale],
                            graph.WhiteNoiseTransform("scale_std"),
                            reason="regularization")
        scope.raw_location = location.copy()
        scope.raw_scale = scale.copy()

    def map_to_input_space(self, scope):
        patch_shape = T.cast(self.cropper.patch_shape, floatX)
        image_shape = scope.x_shape
        if isinstance(image_shape, tuple):
            # for featurelevel_UCF101
            image_shape = image_shape[1][:, 1:]
        scope.true_location, scope.true_scale = static_map_to_input_space(
            scope.raw_location, scope.raw_scale,
            patch_shape, image_shape)
        # measure the extent to which raw_location is outside the image
        scope.excursion = util.rectify(scope.raw_location**2 - 1)

    def tag_attention_dropout(self, variables, rng=None, **hyperparameters):
        from blocks.roles import INPUT, has_roles
        bricks_ = [brick for brick in
                   util.all_bricks([self.patch_transform])
                   if isinstance(brick, (bricks.Linear,
                                         conv2d.Convolutional,
                                         conv3d.Convolutional))]
        variables = [var for var in graph.deep_ancestors(variables)
                     if (has_roles(var, [INPUT]) and
                         any(brick in var.tag.annotations for brick in bricks_))]
        graph.add_transform(
            variables,
            graph.DropoutTransform("attention_dropout", rng=rng),
            reason="regularization")

    def tag_recurrent_weight_noise(self, variables, rng=None, **hyperparameters):
        variables = [var for var in graph.deep_ancestors(variables)
                     if var.name == "weight_noise_goes_here"]
        graph.add_transform(
            variables,
            graph.WhiteNoiseTransform("recurrent_weight_noise", rng=rng),
            reason="regularization")

    def tag_recurrent_dropout(self, variables, recurrent_dropout,
                              rng=None, **hyperparameters):
        from blocks.roles import OUTPUT, has_roles
        ancestors = graph.deep_ancestors(variables)
        for lstm in self.rnn.transitions:
            variables = [var for var in ancestors
                         if (has_roles(var, [OUTPUT]) and
                             lstm in var.tag.annotations and
                             var.name.endswith("states"))]

            # get one dropout mask for all time steps.  use the very
            # first state to get the hidden state shape, else we get
            # graph cycles.
            initial_state = util.the([var for var in variables
                                      if "initial_state" in var.name])
            mask = util.get_dropout_mask(
                initial_state.shape, recurrent_dropout, rng=rng)

            subsequent_states = [var for var in variables
                                 if "initial_state" not in var.name]
            graph.add_transform(
                subsequent_states,
                graph.DropoutTransform("recurrent_dropout", mask=mask),
                reason="regularization")
