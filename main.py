import yaml
import os
import operator as op
from collections import OrderedDict

import theano
import theano.tensor as T
import theano.sandbox.rng_mrg

from fuel.schemes import SequentialScheme

from blocks.initialization import IsotropicGaussian, Constant, Orthogonal, Identity
from blocks.theano_expressions import l2_norm
from blocks.serialization import load_parameter_values
from blocks.model import Model
from blocks.algorithms import GradientDescent, RMSProp
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing, ProgressBar
from blocks.bricks import Rectifier, Tanh
from blocks.bricks.recurrent import LSTM
from blocks.graph import ComputationGraph
from blocks.extras.extensions.plot import Plot

import masonry
import crop
import util
from patchmonitor import PatchMonitoring

import mnist
import svhn
import goodfellow_svhn

from dump import Dump, DumpMinimum, PrintingTo

floatX = theano.config.floatX

class Ram(object):
    def __init__(self, image_shape, patch_shape, patch_transform,
                 patch_postdim, hidden_dim, area_dim, n_spatial_dims,
                 location_std, scale_std, cutoff, batched_window,
                 initargs, emitter, **kwargs):
        self.rng = theano.sandbox.rng_mrg.MRG_RandomStreams(12345)
        self.location_std = location_std
        self.scale_std = scale_std
        self.rnn = LSTM(activation=Tanh(),
                        dim=hidden_dim,
                        name="recurrent",
                        weights_init=IsotropicGaussian(1e-4),
                        biases_init=Constant(0))
        self.locator = masonry.Locator(hidden_dim, area_dim, n_spatial_dims,
                                       **initargs)
        self.cropper = crop.LocallySoftRectangularCropper(
            n_spatial_dims=n_spatial_dims,
            image_shape=image_shape, patch_shape=patch_shape,
            kernel=crop.Gaussian(), cutoff=cutoff,
            batched_window=batched_window)
        self.merger = masonry.Merger(
            n_spatial_dims, patch_postdim, area_dim, response_dim=self.rnn.get_dim("inputs"),
            patch_posttransform=patch_transform.apply,
            area_posttransform=Rectifier(),
            response_posttransform=Rectifier(),
            **initargs)
        self.attention = masonry.SpatialAttention(self.locator, self.cropper, self.merger)
        self.emitter = emitter
        self.model = masonry.RecurrentAttentionModel(
            self.rnn, self.attention, self.emitter)

    def initialize(self):
        self.model.initialize()
        Identity().initialize(self.rnn.W_state, self.rnn.rng)

    def compute(self, x, n_patches):
        initial_outputs = self.model.compute_initial_state(x)
        n_steps = n_patches - 1
        location_noises = self.rng.normal(
            [n_steps, initial_outputs[2].shape[0], initial_outputs[2].shape[1]],
            std=self.location_std)
        scale_noises = self.rng.normal(
            [n_steps, initial_outputs[3].shape[0], initial_outputs[3].shape[1]],
            std=self.scale_std)
        step_outputs = self.model.apply(x=x,
                                        h=initial_outputs[0],
                                        c=initial_outputs[1],
                                        location_noises=location_noises,
                                        scale_noises=scale_noises)
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

def construct_model(task, patch_transform_spec,
                    patch_shape, initargs, n_channels,
                    hyperparameters, **kwargs):
    if "cnn" in patch_transform_spec:
        patch_transform, patch_postdim = masonry.construct_cnn(
            patch_transform_spec["cnn"],
            input_shape=patch_shape,
            **hyperparameters)
    elif "mlp" in patch_transform_spec:
        patch_transform, patch_postdim = masonry.construct_mlp(
            patch_transform_spec["mlp"],
            input_dim=n_channels * reduce(op.mul, patch_shape),
            **hyperparameters)

    emitter = task.get_emitter(**hyperparameters)

    return Ram(patch_postdim=patch_postdim,
               patch_transform=patch_transform,
               emitter=emitter,
               **hyperparameters)

def construct_monitors(algorithm, task, n_patches, x, x_uncentered,
                       hs, cs, locations, scales, patches, mean_savings,
                       graph, plot_url, name, model, cost,
                       patchmonitor_interval=100, **kwargs):
    channels = util.Channels()
    channels.append(util.named(mean_savings.mean(), "mean_savings"))
    channels.extend(task.monitor_channels(graph))
    for i in xrange(n_patches):
        channels.append(hs[:, i].mean(), "h%i_mean" % i)
    for i in xrange(n_patches):
        channels.append(cs[:, i].mean(), "c%i_mean" % i)

    for variable_name in "locations scales".split():
        variable = locals()[variable_name]
        channels.append(variable.var(axis=0).mean(),
                        "%s_variance_across_batch" % variable_name)
        channels.append(variable.var(axis=1).mean(),
                        "%s_variance_across_time" % variable_name)

    step_norms = util.Channels()
    step_norms.extend(util.named(l2_norm([algorithm.steps[param]]),
                                 "step_norm_%s" % name)
                      for name, param in model.get_parameter_dict().items())
    step_channels = step_norms.get_channels()
    #for activation in VariableFilter(roles=[OUTPUT])(graph.variables):
    #    quantity = activation.mean()
    #    quantity.name = "%s_mean" % activation.name
    #    channels.append(quantity)

    monitors = OrderedDict()
    monitors["train"] = TrainingDataMonitoring(
        step_channels,
        prefix="train", after_epoch=True)
    for which in "train valid test".split():
        monitors[which] = DataStreamMonitoring(
            (channels.get_channels() + [cost]),
            data_stream=task.datastreams[which],
            prefix=which, after_epoch=True)

    patch_monitoring = PatchMonitoring(
        task.get_stream("valid", SequentialScheme(5, 5)),
        every_n_batches=patchmonitor_interval,
        extractor=theano.function([x_uncentered], [locations, scales, patches]),
        map_to_image_space=masonry.static_map_to_image_space)
    patch_monitoring.save_patches("test.png")

    step_plots = [["train_%s" % step_channel.name for step_channel in step_channels]]
    plotter = Plot(name,
                   channels=(task.plot_channels() + [['train_cost']] + step_plots),
                   after_epoch=True,
                   server_url=plot_url)

    return list(monitors.values()) + [patch_monitoring, plotter]

def construct_main_loop(name, task_name, patch_shape, batch_size,
                        n_spatial_dims, n_patches, n_epochs,
                        learning_rate, hyperparameters, **kwargs):
    name = "%s_%s" % (name, task_name)
    hyperparameters["name"] = name

    task = get_task(**hyperparameters)
    x_uncentered, y = task.get_variables()

    x = task.preprocess(x_uncentered)

    # this is a theano variable; it may depend on the batch
    hyperparameters["image_shape"] = x.shape[-n_spatial_dims:]

    model = construct_model(task=task, **hyperparameters)
    model.initialize()

    hs, cs, locations, scales, patches, mean_savings = model.compute(x, n_patches)
    cost = model.emitter.cost(cs, y, n_patches)
    cost.name = "cost"

    # get patches from original (uncentered) images
    patches = T.stack(*[model.attention.crop(x_uncentered, locations[:, i, :], scales[:, i, :])[0]
                        for i in xrange(n_patches)])
    # zzz
    patches = patches.dimshuffle(1, 0, *range(2, patches.ndim))

    print "setting up main loop..."
    graph = ComputationGraph(cost)
    uselessflunky = Model(cost)
    algorithm = GradientDescent(cost=cost,
                                parameters=graph.parameters,
                                step_rule=RMSProp(learning_rate=learning_rate))
    monitors = construct_monitors(
        x=x, x_uncentered=x_uncentered, y=y, hs=hs, cs=cs, cost=cost,
        locations=locations, scales=scales, patches=patches, mean_savings=mean_savings,
        algorithm=algorithm, task=task, model=uselessflunky,
        graph=graph, **hyperparameters)
    main_loop = MainLoop(data_stream=task.datastreams["train"],
                         algorithm=algorithm,
                         extensions=(monitors +
                                     [FinishAfter(after_n_epochs=n_epochs),
                                      DumpMinimum(name+'_best', channel_name='valid_error_rate'),
                                      Dump(name+'_dump', every_n_epochs=10),
                                      Checkpoint(name+'_checkpoint.pkl', every_n_epochs=10, on_interrupt=False),
                                      ProgressBar(),
                                      Printing(),
                                      PrintingTo(name+"_log")]),
                         model=uselessflunky)
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
        # pickle made with blocks.serialization.dump(model.get_parameter_dict())
        parameters = load_parameter_values(args.parameters)
        main_loop.model.set_parameter_values(parameters)

    print "training..."
    main_loop.run()
