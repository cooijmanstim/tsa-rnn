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

class Ram(object):
    def __init__(self, image_shape, patch_shape, hidden_dim,
                 n_spatial_dims, whatwhere_interaction, prefork_area_transform,
                 postmerge_area_transform, patch_transform, batch_normalize,
                 response_transform, location_std, scale_std, cutoff,
                 batched_window, initargs, emitter, **kwargs):
        self.rnn = bricks.RecurrentStack(
            [bricks.LSTM(activation=bricks.Tanh(), dim=hidden_dim),
             bricks.LSTM(activation=bricks.Tanh(), dim=hidden_dim)],
            weights_init=initialization.IsotropicGaussian(1e-4),
            biases_init=initialization.Constant(0))

        self.cropper = crop.LocallySoftRectangularCropper(
            n_spatial_dims=n_spatial_dims,
            image_shape=image_shape, patch_shape=patch_shape,
            kernel=crop.Gaussian(), cutoff=cutoff,
            batched_window=batched_window)
        self.emitter = emitter
        self.model = attention.RecurrentAttentionModel(
            self.rnn, self.cropper, self.emitter,
            # attend based on upper RNN states
            attention_state_name="states#1",
            n_spatial_dims=n_spatial_dims,
            location_std=location_std,
            scale_std=scale_std,
            prefork_area_transform=prefork_area_transform,
            patch_transform=patch_transform,
            postmerge_area_transform=postmerge_area_transform,
            response_transform=response_transform,
            whatwhere_interaction=whatwhere_interaction,
            batch_normalize=batch_normalize,
            name="ram")

    def initialize(self):
        self.model.initialize()
        for rnn in self.rnn.transitions:
            initialization.Identity().initialize(rnn.W_state, rnn.rng)

    def compute(self, x, n_patches):
        states = []
        states.append(self.model.compute_initial_state(x, as_dict=True))
        n_steps = n_patches - 1
        for i in xrange(n_steps):
            states.append(self.model.apply(x=x, as_dict=True, **states[-1]))
        outputs = T.concatenate([state["states"][:, np.newaxis, :]
                                 for state in states],
                                axis=1)
        return outputs

def get_task(task_name, hyperparameters, **kwargs):
    klass = dict(mnist=mnist.Task,
                 cluttered_mnist_video=cluttered_mnist_video.Task,
                 svhn_digit=svhn.DigitTask,
                 svhn_number=goodfellow_svhn.NumberTask)[task_name]
    return klass(**hyperparameters)

def construct_model(task, patch_shape, initargs, n_channels, n_spatial_dims, hidden_dim,
                    batch_normalize,
                    hyperparameters, patch_cnn_spec=None, patch_mlp_spec=None,
                    prefork_area_mlp_spec=[], postmerge_area_mlp_spec=[], response_mlp_spec=[],
                    **kwargs):
    patch_transforms = []
    if patch_cnn_spec:
        patch_transforms.append(masonry.construct_cnn(
            name="patch_cnn",
            layer_specs=patch_cnn_spec,
            input_shape=patch_shape,
            n_channels=n_channels,
            batch_normalize=batch_normalize).apply)
        shape = patch_transforms[-1].brick.get_dim("output")
    else:
        shape = (n_channels,) + tuple(patch_shape)
    patch_transforms.append(bricks.FeedforwardFlattener(input_shape=shape).apply)
    if patch_mlp_spec:
        patch_transforms.append(masonry.construct_mlp(
            name="patch_mlp",
            hidden_dims=patch_mlp_spec,
            input_dim=patch_transforms[-1].brick.output_dim,
            batch_normalize=batch_normalize,
            initargs=initargs).apply)
    patch_transform = bricks.FeedforwardSequence(patch_transforms, name="ffs")

    prefork_area_transform = masonry.construct_mlp(
        name="prefork_area_mlp",
        input_dim=hidden_dim,
        hidden_dims=prefork_area_mlp_spec,
        batch_normalize=batch_normalize,
        initargs=initargs)

    postmerge_area_transform = masonry.construct_mlp(
        name="postmerge_area_mlp",
        input_dim=2*n_spatial_dims,
        hidden_dims=postmerge_area_mlp_spec,
        batch_normalize=batch_normalize,
        initargs=initargs)

    # LSTM requires the input to have dim=4*hidden_dim
    response_mlp_activations = [None for dim in response_mlp_spec[1:]]
    response_mlp_spec.append(4*hidden_dim)
    response_mlp_activations.append(bricks.Identity())
    response_transform = masonry.construct_mlp(
        name="response_mlp",
        hidden_dims=response_mlp_spec[1:],
        input_dim=response_mlp_spec[0],
        batch_normalize=batch_normalize,
        activations=response_mlp_activations,
        initargs=initargs)

    emitter = task.get_emitter(**hyperparameters)

    return Ram(patch_transform=patch_transform.apply,
               prefork_area_transform=prefork_area_transform.apply,
               postmerge_area_transform=postmerge_area_transform.apply,
               response_transform=response_transform.apply,
               emitter=emitter,
               **hyperparameters)

def construct_monitors(algorithm, task, n_patches, x, x_uncentered, hs, 
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
        # get patches from original (uncentered) images
        patch = T.stack(*[ram.attention.crop(x_uncentered, location[:, i, :], scale[:, i, :])
                          for i in xrange(n_patches)])
        patch = patch.dimshuffle(1, 0, *range(2, patch.ndim))
        patch_extractor = theano.function([x_uncentered], [location, scale, patch])

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

    x_uncentered, y = task.get_variables()

    x = task.preprocess(x_uncentered)

    # this is a theano variable; it may depend on the batch
    hyperparameters["image_shape"] = x.shape[-n_spatial_dims:]

    ram = construct_model(task=task, **hyperparameters)
    ram.initialize()

    hs = ram.compute(x, n_patches)
    cost = ram.emitter.cost(hs, y, n_patches)
    cost.name = "cost"

    print "setting up main loop..."
    graph = ComputationGraph(cost)
    uselessflunky = Model(cost)
    algorithm = GradientDescent(cost=cost,
                                parameters=graph.parameters,
                                step_rule=Adam(learning_rate=learning_rate))
    monitors = construct_monitors(
        x=x, x_uncentered=x_uncentered, y=y, hs=hs, cost=cost,
        algorithm=algorithm, task=task, model=uselessflunky,
        ram=ram, graph=graph, **hyperparameters)
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
