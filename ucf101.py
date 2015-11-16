import os, logging, yaml
from collections import OrderedDict
import numpy as np
import theano
import theano.tensor as T
from blocks.graph import ComputationGraph
import util, attention, crop, tasks, graph, bricks, masonry
from crop.brick import Cropper

# disable cached constants. this keeps the graph from ballooning with
# map_variables.
T.constant.enable = False

floatX = theano.config.floatX

class UCF101Cropper(object):
    def __init__(self, patch_shape, kernel, hyperparameters):
        self.cropper1d = Cropper(patch_shape[:1], kernel, hyperparameters, name="cropper1d")
        self.cropper3d = Cropper(patch_shape    , kernel, hyperparameters, name="cropper3d")
        self.patch_shape = patch_shape
        self.n_spatial_dims = len(patch_shape)

#        self.fc_conv = masonry.construct_cnn(
#            name="fc_conv",
#            layer_specs=[
#            ],
#            input_shape=(patch_shape[0], 1),
#            n_channels=4096,
#            batch_normalize=hyperparameters["batch_normalize_patch"])
        self.conv_conv = masonry.construct_cnn(
            name="fc_conv",
            layer_specs=[
                dict(size=(5, 1, 1), num_filters=512, pooling_size=(2, 1, 1), pooling_step=(2, 1, 1)),
                dict(size=(5, 1, 1), num_filters=512, pooling_size=(2, 1, 1), pooling_step=(2, 1, 1)),
            ],
            input_shape=patch_shape,
            n_channels=512,
            batch_normalize=hyperparameters["batch_normalize_patch"])

    def initialize(self):
        #self.fc_conv.initialize()
        self.conv_conv.initialize()

    def apply(self, image, image_shape, location, scale):
        # image is secretly two variables; conv and fc features
        fc, conv = image
        fc_shape, conv_shape = image_shape
        # (batch, 4096, 16, 1)
        fc_patch = T.shape_padright(self.cropper1d.apply(
            fc, fc_shape[:, 1:],
            location[:, 0, np.newaxis],
            scale[:, 0, np.newaxis],
        )[0])
        # (batch, 512, 16, 1, 1)
        conv_patch = self.cropper3d.apply(
            conv, conv_shape[:, 1:],
            location, scale,
        )[0]
        fc_repr = fc_patch
        #fc_repr = self.fc_conv.apply(fc_patch)
        conv_repr = self.conv_conv.apply(conv_patch)
        # global average pooling
        fc_repr = fc_repr.mean(axis=range(2, fc_repr.ndim))
        conv_repr = conv_repr.mean(axis=range(2, conv_repr.ndim))
        patch = T.concatenate([fc_repr, conv_repr], axis=1)
        return patch, 0.

    @property
    def output_shape(self):
        return (4096 + 512,)

@util.checkargs
def construct_model(patch_shape, hidden_dim, hyperparameters, **kwargs):
    cropper = UCF101Cropper(
        kernel=crop.Gaussian(),
        patch_shape=patch_shape, hyperparameters=hyperparameters)
    return attention.RecurrentAttentionModel(
        hidden_dim=hidden_dim, cropper=cropper,
        hyperparameters=hyperparameters,
        # attend based on upper RNN states
        attention_state_name="states#1")

@util.checkargs
def construct_monitors(algorithm, task, model, graphs, outputs,
                       updates, monitor_options, n_spatial_dims,
                       hyperparameters, **kwargs):
    from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring

    extensions = []

    if "steps" in monitor_options:
        step_channels = []
        step_channels.extend([
            algorithm.steps[param].norm(2).copy(name="step_norm:%s" % name)
            for name, param in model.get_parameter_dict().items()])
        step_channels.append(algorithm.total_step_norm.copy(name="total_step_norm"))
        step_channels.append(algorithm.total_gradient_norm.copy(name="total_gradient_norm"))
        logger.warning("constructing training data monitor")
        extensions.append(TrainingDataMonitoring(
            step_channels, prefix="train", after_epoch=True))

    if "parameters" in monitor_options:
        data_independent_channels = []
        for parameter in graphs["train"].parameters:
            if parameter.name in "gamma beta W b".split():
                quantity = parameter.norm(2)
                quantity.name = "parameter.norm:%s" % util.get_path(parameter)
                data_independent_channels.append(quantity)
        for key in "location_std scale_std".split():
            data_independent_channels.append(hyperparameters[key].copy(name="parameter:%s" % key))
        extensions.append(DataStreamMonitoring(
            data_independent_channels, data_stream=None, after_epoch=True))

    for which_set in "train test".split():
        channels = []
        channels.extend(outputs[which_set][key] for key in
                        "cost emitter_cost excursion_cost".split())
        channels.extend(outputs[which_set][key] for key in
                        task.monitor_outputs())
        channels.append(outputs[which_set]["savings"]
                        .mean().copy(name="mean_savings"))

        if "theta" in monitor_options:
            for key in "raw_location raw_scale".split():
                for stat in "mean var".split():
                    channels.append(getattr(outputs[which_set][key], stat)(axis=1)
                                    .copy(name="%s.%s" % (key, stat)))
        if which_set == "train":
            if "activations" in monitor_options:
                from blocks.roles import has_roles, OUTPUT
                cnn_outputs = OrderedDict()
                for var in theano.gof.graph.ancestors(graphs[which_set].outputs):
                    if (has_roles(var, [OUTPUT]) and util.annotated_by_a(
                            util.get_convolution_classes(), var)):
                        cnn_outputs.setdefault(util.get_path(var), []).append(var)
                for path, vars in cnn_outputs.items():
                    vars = util.dedup(vars, equal=util.equal_computations)
                    for i, var in enumerate(vars):
                        channels.append(var.mean().copy(
                            name="activation[%i].mean:%s" % (i, path)))

        if "batch_normalization" in monitor_options:
            errors = []
            for population_stat, update in updates[which_set]:
                if population_stat.name.startswith("population"):
                    # this is a super robust way to get the
                    # corresponding batch statistic from the
                    # exponential moving average expression
                    batch_stat = update.owner.inputs[1].owner.inputs[1]
                    errors.append(((population_stat - batch_stat)**2).mean())
            if errors:
                channels.append(T.stack(errors).mean().copy(name="population_statistic_mse"))

        logger.warning("constructing %s monitor" % which_set)
        extensions.append(DataStreamMonitoring(
            channels, prefix=which_set, after_epoch=True,
            data_stream=task.get_stream(which_set, monitor=True)))

    return extensions

@util.checkargs
def prepare_mode(mode, outputs, ram, emitter, hyperparameters, **kwargs):
    if mode == "training":
        hyperparameters["rng"] = util.get_rng(seed=1)
        emitter.tag_dropout(outputs, **hyperparameters)
        ram.tag_attention_dropout(outputs, **hyperparameters)
        ram.tag_recurrent_weight_noise(outputs, **hyperparameters)
        ram.tag_recurrent_dropout(outputs, **hyperparameters)
        logger.warning("%i variables in %s graph" % (graph.graph_size(outputs), mode))
        outputs = graph.apply_transforms(outputs, reason="regularization",
                                         hyperparameters=hyperparameters)
        logger.warning("%i variables in %s graph" % (graph.graph_size(outputs), mode))

        updates = bricks.BatchNormalization.get_updates(outputs)
        print "batch normalization updates:", updates

        return outputs, updates
    elif mode == "inference":
        logger.warning("%i variables in %s graph" % (graph.graph_size(outputs), mode))
        outputs = graph.apply_transforms(
            outputs, reason="population_normalization",
            hyperparameters=hyperparameters)
        logger.warning("%i variables in %s graph" % (graph.graph_size(outputs), mode))
        return outputs, []

@util.checkargs
def construct_graphs(task, n_patches, hyperparameters, **kwargs):
    x, x_shape, y = task.get_variables()

    ram = construct_model(**hyperparameters)
    ram.initialize()

    scopes = []
    scopes.append(ram.apply(util.Scope(x=x, x_shape=x_shape), initial=True))
    n_steps = n_patches - 1
    for i in xrange(n_steps):
        scopes.append(ram.apply(util.Scope(
            x=x, x_shape=x_shape,
            previous_states=scopes[-1].rnn_outputs)))

    emitter = task.get_emitter(
        input_dim=ram.get_dim("states"),
        **hyperparameters)
    emitter.initialize()

    emitter_outputs = emitter.emit(scopes[-1].rnn_outputs["states"], y)
    emitter_cost = emitter_outputs.cost.copy(name="emitter_cost")
    excursion_cost = (T.stack([scope.excursion for scope in scopes])
                      .mean().copy(name="excursion_cost"))
    cost = (emitter_cost + excursion_cost).copy(name="cost")

    # gather all the outputs we could possibly care about for training
    # *and* monitoring; prepare_graphs will do graph transformations
    # after which we may *only* use these to access *any* variables.
    outputs_by_name = OrderedDict()
    for key in "emitter_cost excursion_cost cost".split():
        outputs_by_name[key] = locals()[key]
    for key in task.monitor_outputs():
        outputs_by_name[key] = emitter_outputs[key]
    for key in "raw_location raw_scale patch savings".split():
        outputs_by_name[key] = T.stack([scope[key] for scope in scopes])
    outputs = list(outputs_by_name.values())

    # construct training and inference graphs
    mode_by_set = OrderedDict([
        ("train", "training"),
        ("test", "inference")])
    outputs_by_mode, updates_by_mode = OrderedDict(), OrderedDict()
    for mode in "training inference".split():
        (outputs_by_mode[mode],
         updates_by_mode[mode]) = prepare_mode(
             mode, outputs, ram=ram, emitter=emitter, **hyperparameters)
    # inference updates may make sense at some point but don't know
    # where to put them now
    assert not updates_by_mode["inference"]

    # assign by set for convenience
    graphs_by_set = OrderedDict([
        (which_set, ComputationGraph(outputs_by_mode[mode]))
        for which_set, mode in mode_by_set.items()])
    outputs_by_set = OrderedDict([
        (which_set, OrderedDict(util.equizip(outputs_by_name.keys(),
                                             outputs_by_mode[mode])))
        for which_set, mode in mode_by_set.items()])
    updates_by_set = OrderedDict([
        (which_set, updates_by_mode[mode])
        for which_set, mode in mode_by_set.items()])

    return graphs_by_set, outputs_by_set, updates_by_set

@util.checkargs
def construct_main_loop(name, task_name, patch_shape, batch_size,
                        n_spatial_dims, n_patches, max_epochs,
                        patience_epochs, learning_rate,
                        hyperparameters, **kwargs):
    task = tasks.get_task(**hyperparameters)
    hyperparameters["n_channels"] = task.n_channels

    extensions = []

    # let theta noise decay as training progresses
    from blocks.extensions.training import SharedVariableModifier
    for key in "location_std scale_std".split():
        hyperparameters[key] = theano.shared(hyperparameters[key], name=key)
        extensions.append(util.ExponentialDecay(
            hyperparameters[key],
            hyperparameters["%s_decay" % key],
            after_batch=True))

    print "constructing graphs..."
    graphs, outputs, updates = construct_graphs(task=task, **hyperparameters)

    print "setting up main loop..."

    from blocks.model import Model
    model = Model(outputs["train"]["cost"])

    from blocks.algorithms import GradientDescent, CompositeRule, StepClipping, Adam
    algorithm = GradientDescent(
        cost=outputs["train"]["cost"],
        parameters=graphs["train"].parameters,
        step_rule=CompositeRule([StepClipping(1e2), Adam(learning_rate=learning_rate)]))
    algorithm.add_updates(updates["train"])

    extensions.extend(construct_monitors(
        algorithm=algorithm, task=task, model=model, graphs=graphs,
        outputs=outputs, updates=updates, **hyperparameters))

    from blocks.extensions import FinishAfter, Printing, ProgressBar, Timing
    from blocks.extensions.stopping import FinishIfNoImprovementAfter
    from blocks.extensions.training import TrackTheBest
    from blocks.extensions.saveload import Checkpoint
    from dump import DumpBest, LightCheckpoint, PrintingTo, DumpGraph
    extensions.extend([
        FinishAfter(after_n_epochs=max_epochs),
        Checkpoint(hyperparameters["checkpoint_save_path"],
                   on_interrupt=False, every_n_epochs=5,
                   use_cpickle=True, save_separately=["log"]),
        ProgressBar(),
        Timing(),
        Printing(), PrintingTo(name+"_log"),
        DumpGraph(name+"_grad_graph")])

    from blocks.main_loop import MainLoop
    main_loop = MainLoop(data_stream=task.get_stream("train"),
                         algorithm=algorithm,
                         extensions=extensions,
                         model=model)

    from tabulate import tabulate
    print "parameter sizes:"
    print tabulate((key, "x".join(map(str, value.get_value().shape)), value.get_value().size)
                   for key, value in main_loop.model.get_parameter_dict().items())

    return main_loop

if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger(__name__)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hyperparameters", help="YAML file from which to load hyperparameters")
    parser.add_argument("--checkpoint", help="Checkpoint file from which to resume training")
    parser.add_argument("--autoresume", action="store_true", help="Resume from default checkpoint path or start training if it does not exist")

    args = parser.parse_args()

    hyperparameters_path = getattr(
        args, "hyperparameters",
        os.path.join(os.path.dirname(__file__), "defaults.yaml"))

    with open(hyperparameters_path, "rb") as f:
        hyperparameters = yaml.load(f)

    hyperparameters["n_spatial_dims"] = len(hyperparameters["patch_shape"])
    hyperparameters["hyperparameters"] = hyperparameters
    hyperparameters["name"] += "_" + hyperparameters["task_name"]
    hyperparameters["checkpoint_save_path"] = hyperparameters["name"] + "_checkpoint.zip"

    checkpoint_path = None
    if args.autoresume and os.path.exists(hyperparameters["checkpoint_save_path"]):
        checkpoint_path = hyperparameters["checkpoint_save_path"]
    elif args.checkpoint:
        checkpoint_path = args.checkpoint
    if checkpoint_path:
        from blocks.serialization import load
        main_loop = load(checkpoint_path)
    else:
        main_loop = construct_main_loop(**hyperparameters)

    if not (args.autoresume and main_loop.log.current_row.get("training_finish_requested", False)):
        print "training..."
        main_loop.run()
