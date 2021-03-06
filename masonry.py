import logging, collections
import util, bricks, initialization
import blocks.bricks.conv as conv2d
import conv3d

logger = logging.getLogger(__name__)

@util.checkargs
def construct_cnn_layer(name, layer_spec, conv_module, ndim, batch_normalize):
    if "type" in layer_spec:
        raise ValueError("conv layer spec error: `pool` is part of `conv` now"
                         " (keys `pooling_size`, `pooling_step`) and `type` is"
                         " no longer needed")
    border_mode = layer_spec.pop("border_mode", "valid")
    activation = bricks.NormalizedActivation(
        name="activation",
        batch_normalize=batch_normalize)
    klass = conv_module.ConvolutionalActivation
    kwargs = dict(
        name=name,
        activation=activation.apply,
        filter_size=tuple(layer_spec.pop("size", (1,) * ndim)),
        step=tuple(layer_spec.pop("step", (1,) * ndim)),
        num_filters=layer_spec.pop("num_filters", 1),
        border_mode=border_mode,
        # our activation function will handle the bias
        use_bias=False)
    if "pooling_size" in layer_spec or "pooling_step" in layer_spec:
        klass = conv_module.ConvolutionalLayer
        kwargs["pooling_size"] = layer_spec.pop("pooling_size", None)
        kwargs["pooling_step"] = layer_spec.pop("pooling_step", None)
        # zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
        kwargs["conv_step"] = kwargs.pop("step")
    layer = klass(**kwargs)
    if layer_spec:
        logger.warn("ignoring unknown layer specification keys [%s]"
                    % " ".join(layer_spec.keys()))
    return layer

@util.checkargs
def construct_cnn(name, layer_specs, n_channels, input_shape, batch_normalize):
    ndim = len(input_shape)
    conv_module = {
        2: conv2d,
        3: conv3d,
    }[ndim]
    cnn = conv_module.ConvolutionalSequence(
        name=name,
        layers=[construct_cnn_layer(name="%s_%i" % (name, i),
                                    layer_spec=layer_spec,
                                    ndim=ndim,
                                    conv_module=conv_module,
                                    batch_normalize=batch_normalize)
                for i, layer_spec in enumerate(layer_specs)],
        num_channels=n_channels,
        image_size=tuple(input_shape),
        weights_init=initialization.ConvolutionalInitialization(
            initialization.Orthogonal()),
        # our activation function will handle the bias
        use_bias=False)
    # ensure output dim is determined
    cnn.push_allocation_config()
    # tell the activations what shapes they'll be dealing with
    for layer in cnn.layers:
        activation = util.get_conv_activation(layer, conv_module)
        assert isinstance(activation, bricks.NormalizedActivation)
        activation.shape = layer.get_dim("output")
        activation.broadcastable = [False] + ndim*[True]
    cnn.initialize()
    return cnn

@util.checkargs
def construct_mlp(name, hidden_dims, input_dim, batch_normalize,
                  activations=None, weights_init=None, biases_init=None):
    if not hidden_dims:
        return bricks.FeedforwardIdentity(dim=input_dim)

    if not activations:
        activations = [bricks.Rectifier() for dim in hidden_dims]
    elif not isinstance(activations, collections.Iterable):
        activations = [activations] * len(hidden_dims)
    assert len(activations) == len(hidden_dims)

    if not weights_init:
        weights_init = initialization.Orthogonal()
    if not biases_init:
        biases_init = initialization.Constant(0)

    dims = [input_dim] + hidden_dims
    wrapped_activations = [
        bricks.NormalizedActivation(
            shape=[hidden_dim],
            name="activation_%i" % i,
            batch_normalize=batch_normalize,
            activation=activation)
        for i, (hidden_dim, activation)
        in enumerate(zip(hidden_dims, activations))]
    mlp = bricks.MLP(name=name,
                     activations=wrapped_activations,
                     # biases are handled by our activation function
                     use_bias=False,
                     dims=dims,
                     weights_init=weights_init,
                     biases_init=biases_init)
    return mlp
