import yaml
import os
import logging

import numpy as np

import theano
import theano.tensor as T
import theano.sandbox.rng_mrg

from blocks.model import Model
from blocks.algorithms import GradientDescent, RMSProp, Adam, CompositeRule, StepClipping
from blocks.extensions.training import SharedVariableModifier
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing, ProgressBar, Timing
from blocks.roles import OUTPUT
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.theano_expressions import l2_norm

import util
import bricks
import initialization
import masonry
import attention
import crop
import tasks
from patchmonitor import PatchMonitoring, VideoPatchMonitoring
from dump import Dump, DumpMinimum, PrintingTo, load_model_parameters

floatX = theano.config.floatX

def construct_model(patch_shape, hidden_dim, hyperparameters, **kwargs):
    cropper = crop.LocallySoftRectangularCropper(
        name="cropper", kernel=crop.Gaussian(),
        patch_shape=patch_shape, hyperparameters=hyperparameters)
    return attention.RecurrentAttentionModel(
        hidden_dim=hidden_dim, cropper=cropper,
        hyperparameters=hyperparameters,
        # attend based on upper RNN states
        attention_state_name="states#1",
        name="ram")

@util.checkargs
def construct_monitors(algorithm, task, n_patches, x, x_shape, graphs,
                       name, ram, model, n_spatial_dims, plot_url,
                       hyperparameters, patchmonitor_interval=100,
                       **kwargs):
    extensions = []

    if True:
        step_norms = util.Channels()
        step_norms.extend(util.named(l2_norm([algorithm.steps[param]]),
                                     "%s.step_norm" % name)
                          for name, param in model.get_parameter_dict().items())
        step_channels = step_norms.get_channels()

        extensions.append(TrainingDataMonitoring(
            step_channels, prefix="train", after_epoch=True))

    if True:
        data_independent_channels = util.Channels()
        for parameter in graphs["train"].parameters:
            if parameter.name in "gamma beta".split():
                quantity = parameter.mean()
                quantity.name = "%s.mean" % util.get_path(parameter)
                data_independent_channels.append(quantity)
        for key in "location_std scale_std".split():
            data_independent_channels.append(hyperparameters[key].copy(name=key))

        extensions.append(DataStreamMonitoring(
            data_independent_channels.get_channels(),
            data_stream=None, after_epoch=True))

    for which_set in "train valid test".split():
        graph = graphs[which_set]

        channels = util.Channels()
        channels.extend(task.monitor_channels(graph))

        (location, scale,
         true_location, true_scale,
         savings) = util.get_recurrent_auxiliaries(
            "location scale true_location true_scale savings".split(),
             graph, n_patches)

        channels.append(util.named(savings.mean(), "savings.mean"))

        for variable_name in "location scale".split():
            variable = locals()[variable_name]
            channels.append(variable.mean(axis=0).T,
                            "%s.mean" % variable_name)
            channels.append(variable.var(axis=0).T,
                            "%s.variance" % variable_name)

        if which_set == "train":
            channels.append(algorithm.total_gradient_norm,
                            "total_gradient_norm")

        extensions.append(DataStreamMonitoring(
            (channels.get_channels() + graph.outputs[0]),
            data_stream=task.get_stream(which_set, monitor=True),
            prefix=which_set, after_epoch=True))

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

def get_training_graph(cost, emitter, dropout, **kwargs):
    [cost] = util.replace_by_tags(
        [cost], "location_noise scale_noise".split())
    graph = ComputationGraph(cost)
    graph = emitter.apply_dropout(graph, dropout)
    return graph

def get_inference_graph(cost, **kwargs):
    return ComputationGraph(cost)

def construct_main_loop(name, task_name, patch_shape, batch_size,
                        n_spatial_dims, n_patches, n_epochs,
                        learning_rate, hyperparameters, **kwargs):
    name = "%s_%s" % (name, task_name)
    hyperparameters["name"] = name

    task = tasks.get_task(**hyperparameters)
    hyperparameters["n_channels"] = task.n_channels

    extensions = []

    # let theta noise decay as training progresses
    for key in "location_std scale_std".split():
        hyperparameters[key] = theano.shared(hyperparameters[key], name=key)
        rate = hyperparameters["%s_decay" % key]
        extensions.append(SharedVariableModifier(
            hyperparameters[key],
            lambda i, x: rate * x))

    theano.config.compute_test_value = "warn"

    x, x_shape, y = task.get_variables()

    ram = construct_model(task=task, **hyperparameters)
    ram.initialize()

    states = []
    states.append(ram.compute_initial_state(x, x_shape, as_dict=True))
    n_steps = n_patches - 1
    for i in xrange(n_steps):
        states.append(ram.apply(x, x_shape, as_dict=True, **states[-1]))

    emitter = task.get_emitter(
        input_dim=ram.get_dim("states"),
        **hyperparameters)
    emitter.initialize()
    cost = emitter.cost(states[-1]["states"], y, n_patches)
    cost.name = "cost"

    print "setting up main loop..."
    graphs = OrderedDict()
    graphs["train"] = get_training_graph(cost, emitter=emitter, **hyperparameters)
    graphs["test"] = get_inference_graph(cost, **hyperparameters)
    graphs["valid"] = graphs["test"]

    uselessflunky = Model(cost)
    algorithm = GradientDescent(
        cost=graphs["train"].outputs[0],
        parameters=graphs["train"].parameters,
        step_rule=CompositeRule([StepClipping(1e2),
                                 Adam(learning_rate=learning_rate)]))
    extensions.extend(construct_monitors(
        x=x, x_shape=x_shape,
        algorithm=algorithm, task=task, model=uselessflunky, ram=ram,
        graphs=graphs, **hyperparameters))
    extensions.extend([
        FinishAfter(after_n_epochs=n_epochs),
        DumpMinimum(name+'_best', channel_name='valid_error_rate'),
        Dump(name+'_dump', every_n_epochs=10),
        #Checkpoint(name+'_checkpoint.pkl', every_n_epochs=10, on_interrupt=False),
        ProgressBar(),
        Timing(),
        Printing(),
        PrintingTo(name+"_log")])
    main_loop = MainLoop(data_stream=task.get_stream("train"),
                         algorithm=algorithm,
                         extensions=extensions,
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

    hyperparameters_path = getattr(
        args, "hyperparameters",
        os.path.join(os.path.dirname(__file__), "defaults.yaml"))

    with open(hyperparameters_path, "rb") as f:
        hyperparameters = yaml.load(f)

    hyperparameters["n_spatial_dims"] = len(hyperparameters["patch_shape"])
    hyperparameters["hyperparameters"] = hyperparameters

    main_loop = construct_main_loop(**hyperparameters)

    if args.parameters:
        load_model_parameters(args.parameters, main_loop.model)

    print "training..."
    main_loop.run()
