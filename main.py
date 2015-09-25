import yaml
import os
import logging

import numpy as np

import theano
import theano.tensor as T
import theano.sandbox.rng_mrg

from blocks.model import Model
from blocks.algorithms import GradientDescent, RMSProp, Adam
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing, ProgressBar, Timing
from blocks.roles import OUTPUT
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter

import bricks
import initialization

import masonry
import attention
import crop
import util
from patchmonitor import PatchMonitoring, VideoPatchMonitoring

import mnist
import cluttered_mnist_video
import svhn
import goodfellow_svhn

from dump import Dump, DumpMinimum, PrintingTo, load_model_parameters

floatX = theano.config.floatX

def get_task(task_name, hyperparameters, **kwargs):
    klass = dict(mnist=mnist.Task,
                 cluttered_mnist_video=cluttered_mnist_video.Task,
                 svhn_digit=svhn.DigitTask,
                 svhn_number=goodfellow_svhn.NumberTask)[task_name]
    return klass(**hyperparameters)

def construct_model(patch_shape, hidden_dim, task,
                    hyperparameters, **kwargs):
    cropper = crop.LocallySoftRectangularCropper(
        name="cropper", kernel=crop.Gaussian(),
        patch_shape=patch_shape, hyperparameters=hyperparameters)
    emitter = task.get_emitter(**hyperparameters)
    return attention.RecurrentAttentionModel(
        hidden_dim=hidden_dim, cropper=cropper, emitter=emitter,
        hyperparameters=hyperparameters,
        # attend based on upper RNN states
        attention_state_name="states#1",
        name="ram",
        # blocks' push_initialization_config stinks; it overwrites everything
        # that comes in its way, leaving us with no good way to specialize
        # initialization for e.g. convolution layers deep inside the model.
        # so we are doomed to use one initializer that "works" for every
        # situation.
        weights_init=initialization.IsotropicGaussian(std=1e-3),
        biases_init=initialization.Constant(0))

def construct_monitors(algorithm, task, n_patches, x, x_shape, hs, 
                       graph, name, ram, model, cost,
                       n_spatial_dims, plot_url, patchmonitor_interval=100, **kwargs):
    location, scale, savings = util.get_recurrent_auxiliaries(
        "location scale savings".split(), graph, n_patches)

    channels = util.Channels()
    channels.extend(task.monitor_channels(graph))

    #for i in xrange(n_patches):
    #    channels.append(hs[:, i].mean(), "h%i.mean" % i)

    channels.append(util.named(savings.mean(), "savings.mean"))

    for variable_name in "location scale".split():
        variable = locals()[variable_name]
        channels.append(variable.var(axis=0).mean(),
                        "%s.batch_variance" % variable_name)
        channels.append(variable.var(axis=1).mean(),
                        "%s.time_variance" % variable_name)

    #step_norms = util.Channels()
    #step_norms.extend(util.named(l2_norm([algorithm.steps[param]]),
    #                             "%s.step_norm" % name)
    #                  for name, param in model.get_parameter_dict().items())
    #step_channels = step_norms.get_channels()

    #for activation in VariableFilter(roles=[OUTPUT])(graph.variables):
    #    quantity = activation.mean()
    #    quantity.name = "%s.mean" % util.get_path(activation)
    #    channels.append(quantity)

    data_independent_channels = util.Channels()
    for parameter in graph.parameters:
        if parameter.name in "gamma beta".split():
            quantity = parameter.mean()
            quantity.name = "%s.mean" % util.get_path(parameter)
            data_independent_channels.append(quantity)

    extensions = []

    #extensions.append(TrainingDataMonitoring(
    #    step_channels,
    #    prefix="train", after_epoch=True))

    extensions.append(DataStreamMonitoring(data_independent_channels.get_channels(),
                                           data_stream=None, after_epoch=True))
    extensions.extend(DataStreamMonitoring((channels.get_channels() + [cost]),
                                           data_stream=task.get_stream(which, monitor=True),
                                           prefix=which, after_epoch=True)
                      for which in "train valid test".split())

    patchmonitor = None
    if n_spatial_dims == 2:
        patchmonitor_klass = PatchMonitoring
    elif n_spatial_dims == 3:
        patchmonitor_klass = VideoPatchMonitoring

    if patchmonitor_klass:
        patch = T.stack(*[
            ram.crop(x, x_shape, location[:, i, :], scale[:, i, :])
            for i in xrange(n_patches)])
        patch = patch.dimshuffle(1, 0, *range(2, patch.ndim))
        patch_extractor = theano.function([x, x_shape],
                                          [location, scale, patch])

        for which in "train valid".split():
            patchmonitor = patchmonitor_klass(
                save_to="%s_patches_%s" % (name, which),
                data_stream=task.get_stream(which, shuffle=False, num_examples=5),
                every_n_batches=patchmonitor_interval,
                extractor=patch_extractor,
                map_to_input_space=attention.static_map_to_input_space)
            patchmonitor.save_patches("patchmonitor_test.png")
            extensions.append(patchmonitor)

    if plot_url:
        plot_channels = []
        plot_channels.extend(task.plot_channels())
        plot_channels.append(["train_cost"])
        #plot_channels.append(["train_%s" % step_channel.name for step_channel in step_channels])

        from blocks.extras.extensions.plot import Plot
        extensions.append(Plot(name, channels=plot_channels,
                            after_epoch=True, server_url=plot_url))

    return extensions

def construct_main_loop(name, task_name, patch_shape, batch_size,
                        n_spatial_dims, n_patches, n_epochs,
                        learning_rate, hyperparameters, **kwargs):
    name = "%s_%s" % (name, task_name)
    hyperparameters["name"] = name

    task = get_task(**hyperparameters)
    hyperparameters["n_channels"] = task.n_channels

    theano.config.compute_test_value = "warn"

    x, x_shape, y = task.get_variables()

    ram = construct_model(task=task, **hyperparameters)
    ram.initialize()

    states = []
    states.append(ram.compute_initial_state(x, x_shape, as_dict=True))
    n_steps = n_patches - 1
    for i in xrange(n_steps):
        states.append(ram.apply(x, x_shape, as_dict=True, **states[-1]))
    hs = T.concatenate([state["states"][:, np.newaxis, :]
                        for state in states],
                       axis=1)

    cost = ram.emitter.cost(hs, y, n_patches)
    cost.name = "cost"

    print "setting up main loop..."
    graph = ComputationGraph(cost)
    uselessflunky = Model(cost)
    algorithm = GradientDescent(cost=cost,
                                parameters=graph.parameters,
                                step_rule=Adam(learning_rate=learning_rate))
    monitors = construct_monitors(
        x=x, x_shape=x_shape, y=y, hs=hs, cost=cost,
        algorithm=algorithm, task=task, model=uselessflunky, ram=ram,
        graph=graph, **hyperparameters)
    main_loop = MainLoop(data_stream=task.get_stream("train"),
                         algorithm=algorithm,
                         extensions=(monitors +
                                     [FinishAfter(after_n_epochs=n_epochs),
                                      DumpMinimum(name+'_best', channel_name='valid_error_rate'),
                                      Dump(name+'_dump', every_n_epochs=10),
                                      #Checkpoint(name+'_checkpoint.pkl', every_n_epochs=10, on_interrupt=False),
                                      ProgressBar(),
                                      Timing(),
                                      Printing(),
                                      PrintingTo(name+"_log")]),
                         model=uselessflunky)
    return main_loop

if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger(__name__)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hyperparameters", help="YAML file from which to load hyperparameters")
    parser.add_argument("--parameters", help="npy/npz file from which to load parameters")

    args = parser.parse_args()

    with open(os.path.join(os.path.dirname(__file__), "defaults.yaml"), "rb") as f:
        hyperparameters = yaml.load(f)
    if args.hyperparameters:
        with open(args.hyperparameters, "rb") as f:
            hyperparameters.update(yaml.load(f))

    hyperparameters["n_spatial_dims"] = len(hyperparameters["patch_shape"])
    hyperparameters["initargs"] = dict(weights_init=initialization.Orthogonal(),
                                       biases_init=initialization.Constant(0))
    hyperparameters["hyperparameters"] = hyperparameters

    main_loop = construct_main_loop(**hyperparameters)

    if args.parameters:
        load_model_parameters(args.parameters, main_loop.model)

    print "training..."
    main_loop.run()
