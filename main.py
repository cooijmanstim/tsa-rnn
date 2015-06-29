import yaml
import math
import sys
import os
import operator as op
from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T

from fuel.datasets import MNIST
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream

from blocks.initialization import IsotropicGaussian, Constant, NdarrayInitialization, Orthogonal, Identity
from blocks.utils import named_copy
from blocks.model import Model
from blocks.algorithms import GradientDescent, RMSProp
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing, ProgressBar
from blocks.roles import PARAMETER, OUTPUT, INPUT, DROPOUT
from blocks.bricks import Softmax, Rectifier, Brick, application, MLP, FeedforwardSequence
from blocks.bricks.recurrent import LSTM, SimpleRecurrent
from blocks.graph import ComputationGraph
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.main_loop import MainLoop
from blocks.filter import VariableFilter
from blocks.extras.extensions.plot import Plot
from blocks.bricks.conv import ConvolutionalSequence, ConvolutionalLayer, Flattener

import masonry
import crop
import util
from patchmonitor import PatchMonitoring
import mnist

floatX = theano.config.floatX

class Ram(object):
    def __init__(self, image_shape, patch_shape, patch_transform,
                 patch_postdim, hidden_dim, area_dim, n_spatial_dims,
                 batched_window, n_classes, initargs, **kwargs):
        self.locator = masonry.Locator(hidden_dim, area_dim, n_spatial_dims)
        self.cropper = crop.LocallySoftRectangularCropper(
            n_spatial_dims, image_shape, patch_shape,
            crop.gaussian,
            batched_window=batched_window)
        self.merger = masonry.Merger(
            n_spatial_dims, patch_postdim, area_dim, hidden_dim,
            patch_posttransform=patch_transform.apply,
            area_posttransform=Rectifier(),
            response_posttransform=Rectifier(),
            **initargs)
        self.attention = masonry.SpatialAttention(self.locator, self.cropper, self.merger)
        self.emitter = MLP(activations=[Softmax()],
                           dims=[hidden_dim, n_classes],
                           **initargs)
        self.rnn = SimpleRecurrent(activation=Rectifier(),
                                   dim=hidden_dim,
                                   weights_init=Identity(),
                                   biases_init=Constant(0))
        self.model = masonry.RecurrentAttentionModel(
            self.rnn, self.attention, self.emitter,
            **initargs)

    def initialize(self):
        self.model.initialize()

    def compute(self, x, n_patches):
        initial_outputs = self.model.compute_initial_state(x)
        step_outputs = self.model.apply(x=x, h=initial_outputs[1], n_steps=n_patches - 1, batch_size=x.shape[0])
        # prepend initial values
        step_outputs = [T.concatenate([T.shape_padleft(initial_output), step_output], axis=0)
                        for initial_output, step_output in zip(initial_outputs, step_outputs)]
        # move batch axis in front of RNN time axis
        step_outputs = [step_output.dimshuffle(1, 0, *range(step_output.ndim)[2:])
                        for step_output in step_outputs]
        yhats, hs, locations, scales, patches = step_outputs
        return step_outputs

def construct_model(convolutional, patch_shape, initargs, hyperparameters, **kwargs):
    patch_dim = reduce(op.mul, patch_shape)

    if convolutional:
        patch_transform = ConvolutionalSequence(
            layers=[ConvolutionalLayer(activation=Rectifier().apply,
                                       filter_size=(3, 3),
                                       pooling_size=(2, 2),
                                       num_filters=patch_dim*(i+1),
                                       name="patch_conv_%i" % i)
                    for i in xrange(2)],
            num_channels=n_channels,
            image_size=patch_shape,
            weights_init=IsotropicGaussian(std=1e-8),
            biases_init=Constant(0))
        patch_transform.push_allocation_config()
        # ConvolutionalSequence doesn't provide output_dim
        patch_postdim = reduce(op.mul, patch_transform.get_dim("output"))
    else:
        patch_postdim = 128
        patch_transform = FeedforwardSequence([Flattener().apply,
                                               MLP(activations=[Rectifier()],
                                                   dims=[patch_dim, patch_postdim],
                                                   **initargs).apply])

    return Ram(patch_postdim=patch_postdim,
               patch_transform=patch_transform,
               **hyperparameters)

def construct_monitors(datasets, datastreams, cross_entropy,
                       error_rate, n_patches, x, hs, locations,
                       scales, patches, graph, **kwargs):
    channels = util.Channels()
    channels.add(cross_entropy)
    channels.add(error_rate)
    for i in xrange(n_patches):
        channels.add(hs[:, i].max(), "h%i_max" % i)
    #for activation in VariableFilter(roles=[OUTPUT])(graph.variables):
    #    quantity = activation.mean()
    #    quantity.name = "%s_mean" % activation.name
    #    channels.add(quantity)

    monitors = OrderedDict()
    monitors["train"] = TrainingDataMonitoring(channels.get_channels(),
                                               prefix="train",
                                               after_epoch=True)
    for which in "valid test".split():
        monitors[which] = DataStreamMonitoring(
            channels.get_channels(),
            data_stream=datastreams[which],
            prefix=which,
            after_epoch=True)

    patch_monitoring_datastream = DataStream.default_stream(
        datasets["valid"],
        iteration_scheme=SequentialScheme(5, 5))
    patch_monitoring = PatchMonitoring(patch_monitoring_datastream,
                                       theano.function([x], [locations, scales, patches]))
    patch_monitoring.save_patches("test.png")

    return list(monitors.values()) + [patch_monitoring]

def construct_main_loop(name, convolutional, patch_shape, batch_size,
                        n_spatial_dims, n_patches, n_channels,
                        n_epochs, learning_rate, hyperparameters,
                        **kwargs):
    # shape (batch, channel, height, width)
    x = T.tensor4('features', dtype=floatX)
    # shape (batch_size, ntargets)
    y = T.lmatrix('targets')

    theano.config.compute_test_value = 'warn'
    x.tag.test_value = np.random.random((batch_size, n_channels, 28, 28)).astype("float32")
    y.tag.test_value = np.random.random_integers(0, 9, (batch_size, 1)).astype("int64")

    hyperparameters["image_shape"] = x.shape[-n_spatial_dims:]

    model = construct_model(**hyperparameters)
    model.initialize()
    yhats, hs, locations, scales, patches = model.compute(x, n_patches)
    yhat = yhats[:, -1, :]

    cross_entropy = util.named(CategoricalCrossEntropy().apply(y.flatten(), yhat),
                               "cross_entropy")
    error_rate = util.named(MisclassificationRate().apply(y.flatten(), yhat),
                            "error_rate")

    graph = ComputationGraph(cross_entropy)

    print "setting up main loop..."
    algorithm = GradientDescent(cost=cross_entropy,
                                params=graph.parameters,
                                step_rule=RMSProp(learning_rate=learning_rate))
    datasets, datastreams = mnist.load(**hyperparameters)
    monitors = construct_monitors(x=x, y=y, yhats=yhats, hs=hs,
                                  locations=locations, scales=scales, patches=patches,
                                  cross_entropy=cross_entropy, error_rate=error_rate,
                                  datasets=datasets, datastreams=datastreams, graph=graph,
                                  **hyperparameters)
    main_loop = MainLoop(data_stream=datastreams["train"],
                         algorithm=algorithm,
                         extensions=(monitors +
                                     [FinishAfter(after_n_epochs=n_epochs),
                                      ProgressBar(),
                                      Printing(),
                                      Plot(name,
                                           channels=[["%s_cross_entropy" % which for which in datasets.keys()],
                                                     ["%s_error_rate"    % which for which in datasets.keys()]],
                                           after_epoch=True)]),
                         model=Model(cross_entropy))
    return main_loop

if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), "defaults.yaml")) as f:
        hyperparameters = yaml.load(f)
    hyperparameters["n_spatial_dims"] = len(hyperparameters["patch_shape"])
    hyperparameters["initargs"] = dict(weights_init=Orthogonal(),
                                       biases_init=Constant(0))
    hyperparameters["hyperparameters"] = hyperparameters
    main_loop = construct_main_loop(**hyperparameters)
    print "training..."
    main_loop.run()
