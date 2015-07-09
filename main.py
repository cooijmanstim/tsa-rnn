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
from blocks.theano_expressions import l2_norm
from blocks.serialization import load_parameter_values
from blocks.model import Model
from blocks.algorithms import GradientDescent, RMSProp
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing, ProgressBar
from blocks.roles import PARAMETER, OUTPUT, INPUT, DROPOUT
from blocks.bricks import Softmax, Rectifier, Brick, application, MLP, FeedforwardSequence, Tanh
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
import svhn
import goodfellow_svhn

from dump import Dump, DumpMinimum

floatX = theano.config.floatX

class Ram(object):
    def __init__(self, image_shape, patch_shape, patch_transform,
                 patch_postdim, hidden_dim, area_dim, n_spatial_dims,
                 batched_window, initargs, emitter, **kwargs):
        self.locator = masonry.Locator(hidden_dim, area_dim, n_spatial_dims)
        self.cropper = crop.LocallySoftRectangularCropper(
            n_spatial_dims, image_shape, patch_shape,
            crop.Gaussian(),
            batched_window=batched_window)
        self.merger = masonry.Merger(
            n_spatial_dims, patch_postdim, area_dim, hidden_dim,
            patch_posttransform=patch_transform.apply,
            area_posttransform=Rectifier(),
            response_posttransform=Rectifier(),
            **initargs)
        self.attention = masonry.SpatialAttention(self.locator, self.cropper, self.merger)
        self.emitter = emitter
        self.rnn = SimpleRecurrent(activation=Tanh(),
                                   dim=hidden_dim,
                                   weights_init=Identity(),
                                   biases_init=Constant(0),
                                   name="recurrent")
        self.model = masonry.RecurrentAttentionModel(
            self.rnn, self.attention, self.emitter, **initargs)

    def initialize(self):
        self.model.initialize()

    def compute(self, x, n_patches):
        initial_outputs = self.model.compute_initial_state(x)
        step_outputs = self.model.apply(x=x, h=initial_outputs[0], n_steps=n_patches - 1, batch_size=x.shape[0])
        # prepend initial values
        step_outputs = [T.concatenate([T.shape_padleft(initial_output), step_output], axis=0)
                        for initial_output, step_output in zip(initial_outputs, step_outputs)]
        # mean_savings is special; it has no batch axis
        mean_savings = step_outputs.pop()
        # move batch axis in front of RNN time axis
        step_outputs = [step_output.dimshuffle(1, 0, *range(step_output.ndim)[2:])
                        for step_output in step_outputs]
        step_outputs.append(mean_savings)
        return step_outputs

def get_task(task_name, hyperparameters, **kwargs):
    klass = dict(mnist=mnist.Task,
                 svhn_digit=svhn.DigitTask,
                 svhn_number=goodfellow_svhn.NumberTask)[task_name]
    return klass(**hyperparameters)

def construct_model(task, convolutional, patch_shape, initargs,
                    n_channels, hyperparameters, **kwargs):
    patch_dim = n_channels * reduce(op.mul, patch_shape)

    if convolutional:
        patch_transform = ConvolutionalSequence(
            layers=[ConvolutionalLayer(activation=Rectifier().apply,
                                       filter_size=(3, 3),
                                       pooling_size=(2, 2),
                                       num_filters=patch_dim*(i+1),
                                       name="patch_conv_%i" % i)
                    for i in xrange(2)],
            num_channels=n_channels,
            image_size=tuple(patch_shape),
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

    emitter = task.get_emitter(**hyperparameters)

    return Ram(patch_postdim=patch_postdim,
               patch_transform=patch_transform,
               emitter=emitter,
               **hyperparameters)

def construct_monitors(algorithm, task, task_channels, task_plots,
                       n_patches, x, x_uncentered, hs, locations,
                       scales, patches, mean_savings, graph, plot_url,
                       name, model, **kwargs):
    channels = util.Channels()
    channels.append(util.named(mean_savings.mean(), "mean_savings"))
    channels.extend(task_channels)
    for i in xrange(n_patches):
        channels.append(hs[:, i].max(), "h%i_max" % i)

    step_norms = util.Channels()
    step_norms.extend(util.named(l2_norm([algorithm.steps[param]]),
                                 "step_norm_%s" % name)
                      for name, param in model.params.items())
    step_channels = step_norms.get_channels()
    #for activation in VariableFilter(roles=[OUTPUT])(graph.variables):
    #    quantity = activation.mean()
    #    quantity.name = "%s_mean" % activation.name
    #    channels.append(quantity)

    monitors = OrderedDict()
    monitors["train"] = TrainingDataMonitoring(
        (channels.get_channels() + step_channels),
        prefix="train", after_epoch=True)
    for which in "valid test".split():
        monitors[which] = DataStreamMonitoring(
            channels.get_channels(),
            data_stream=task.datastreams[which],
            prefix=which, after_epoch=True)

    patch_monitoring = PatchMonitoring(
        task.get_stream("valid", SequentialScheme(5, 5)),
        extractor=theano.function([x_uncentered], [locations, scales, patches]),
        map_to_image_space=masonry.static_map_to_image_space)
    patch_monitoring.save_patches("test.png")

    step_plots = [["train_%s" % step_channel.name for step_channel in step_channels]]
    plotter = Plot(name,
                   channels=(task_plots + step_plots),
                   after_epoch=True,
                   server_url=plot_url)

    return list(monitors.values()) + [patch_monitoring, plotter]

def construct_main_loop(name, convolutional, patch_shape, batch_size,
                        n_spatial_dims, n_patches, n_epochs,
                        learning_rate, hyperparameters, **kwargs):
    task = get_task(**hyperparameters)
    x_uncentered, y = task.get_variables()

    x = task.preprocess(x_uncentered)

    # this is a theano variable; it may depend on the batch
    hyperparameters["image_shape"] = x.shape[-n_spatial_dims:]

    model = construct_model(task=task, **hyperparameters)
    model.initialize()

    hs, locations, scales, patches, mean_savings = model.compute(x, n_patches)
    cost = model.emitter.cost(hs[:, -1, :], y)

    # get patches from original (uncentered) images
    patches = T.stack(*[model.attention.crop(x_uncentered, locations[:, i, :], scales[:, i, :])[0]
                        for i in xrange(n_patches)])
    # zzz
    patches = patches.dimshuffle(1, 0, *range(2, patches.ndim))

    print "setting up main loop..."
    graph = ComputationGraph(cost)
    task_channels = task.monitor_channels(graph)
    task_plots = task.plot_channels()
    uselessflunky = Model(cost)
    algorithm = GradientDescent(cost=cost,
                                params=graph.parameters,
                                step_rule=RMSProp(learning_rate=learning_rate))
    monitors = construct_monitors(
        x=x, x_uncentered=x_uncentered, y=y, hs=hs,
        locations=locations, scales=scales, patches=patches,
        mean_savings=mean_savings, algorithm=algorithm, task=task,
        task_channels=task_channels, task_plots=task_plots,
        model=uselessflunky, graph=graph, **hyperparameters)
    main_loop = MainLoop(data_stream=task.datastreams["train"],
                         algorithm=algorithm,
                         extensions=(monitors +
                                     [FinishAfter(after_n_epochs=n_epochs),
                                      DumpMinimum(name+'_best', channel_name='valid_error_rate'),
                                      Dump(name+'_dump', every_n_epochs=1),
                                      Checkpoint(name+'_checkpoint.pkl', every_n_epochs=1, on_interrupt=False),
                                      ProgressBar(),
                                      Printing()]),
                         model=Model(cost))
    return main_loop

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hyperparameters", help="YAML file from which to load hyperparameters")
    parser.add_argument("--parameters", help="pickle file from which to load parameters")

    args = parser.parse_args()

    with open(os.path.join(os.path.dirname(__file__), "defaults.yaml"), "rb") as f:
        hyperparameters = yaml.load(f)
    if args.hyperparameters:
        with open(args.hyperparameters, "rb") as f:
            hyperparameters.update(yaml.load(f))

    hyperparameters["n_spatial_dims"] = len(hyperparameters["patch_shape"])
    hyperparameters["initargs"] = dict(weights_init=Orthogonal(),
                                       biases_init=Constant(0))
    hyperparameters["hyperparameters"] = hyperparameters

    main_loop = construct_main_loop(**hyperparameters)

    if args.parameters:
        # pickle made with blocks.serialization.dump(model.params)
        params = load_parameter_values(args.parameters)
        main_loop.model.set_param_values(params)

    print "training..."
    main_loop.run()
