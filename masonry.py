import collections
import logging

logger = logging.getLogger(__name__)

import numpy as np

import blocks.bricks.conv as conv2d
import conv3d

import bricks
import initialization

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
        activation = bricks.NormalizedActivation(
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
        layer.weights_init = initialization.IsotropicGaussian(
            std=np.sqrt(2./(np.prod(layer.filter_size) * prev_num_filters)))
        layer.biases_init = initialization.Constant(0)
        prev_num_filters = layer.num_filters
    # tell the activations what shapes they'll be dealing with
    for layer in cnn.layers:
        # woe is me
        try:
            activation = layer.application_methods[-1].brick
        except:
            continue
        if isinstance(activation, bricks.NormalizedActivation):
            activation.shape = layer.get_dim("output")
            activation.broadcastable = [False] + len(input_shape)*[True]
    cnn.initialize()
    return cnn

def construct_mlp(name, hidden_dims, input_dim, batch_normalize, initargs=None, activations=None):
    if not hidden_dims:
        return bricks.FeedforwardIdentity(dim=input_dim)

    initargs = initargs or dict()

    if not activations:
        activations = [bricks.Rectifier() for dim in hidden_dims]
    elif not isinstance(activations, collections.Iterable):
        activations = [activations] * len(hidden_dims)
    assert len(activations) == len(hidden_dims)

    dims = [input_dim] + hidden_dims
    wrapped_activations = [
        bricks.NormalizedActivation(
            shape=[hidden_dim],
            name="activation_%i" % i,
            # biases are handled by our activation function
            use_bias=False,
            batch_normalize=batch_normalize,
            activation=activation)
        for i, (hidden_dim, activation)
        in enumerate(zip(hidden_dims, activations))]
    mlp = bricks.MLP(name=name,
                     activations=wrapped_activations,
                     dims=dims,
                     **initargs)
    return mlp
