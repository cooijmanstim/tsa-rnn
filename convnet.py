import os, logging, yaml
from collections import OrderedDict
import numpy as np
import theano
import theano.tensor as T
from blocks.graph import ComputationGraph
import util, tasks, dump, graph, bricks, masonry

# disable cached constants. this keeps the graph from ballooning with
# map_variables.
T.constant.enable = False

floatX = theano.config.floatX

@util.checkargs
def construct_model(convnet_spec, n_channels, video_shape,
                    batch_normalize, hyperparameters, **kwargs):
    return masonry.construct_cnn(
        name="convnet",
        input_shape=video_shape,
        layer_specs=convnet_spec,
        n_channels=n_channels,
        batch_normalize=batch_normalize)

@util.checkargs
def construct_monitors(algorithm, task, model, graphs, outputs,
                       plot_url, hyperparameters, **kwargs):
    from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
    from patchmonitor import PatchMonitoring, VideoPatchMonitoring

    extensions = []

    if True:
        extensions.append(TrainingDataMonitoring(
            [algorithm.steps[param].norm(2).copy(name="step_norm:%s" % name)
             for name, param in model.get_parameter_dict().items()],
            prefix="train", after_epoch=True))

    if True:
        data_independent_channels = []
        for parameter in graphs["train"].parameters:
            if parameter.name in "gamma beta W b".split():
                quantity = parameter.norm(2)
                quantity.name = "parameter.norm:%s" % util.get_path(parameter)
                data_independent_channels.append(quantity)
        extensions.append(DataStreamMonitoring(
            data_independent_channels, data_stream=None, after_epoch=True))

    for which_set in "train valid test".split():
        channels = []
        channels.extend(outputs[which_set][key] for key in
                        "cost".split())
        channels.extend(outputs[which_set][key] for key in
                        task.monitor_outputs())
        if which_set == "train":
            if True:
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

            channels.append(algorithm.total_gradient_norm.copy(name="total_gradient_norm"))
        extensions.append(DataStreamMonitoring(
            channels, prefix=which_set, after_epoch=True,
            data_stream=task.get_stream(which_set, monitor=True)))

    if plot_url:
        plot_channels = []
        plot_channels.extend(task.plot_channels())
        plot_channels.append(["train_cost"])
        #plot_channels.append(["train_%s" % step_channel.name for step_channel in step_channels])

        from blocks.extras.extensions.plot import Plot
        extensions.append(Plot(name, channels=plot_channels,
                            after_epoch=True, server_url=plot_url))

    return extensions

def tag_convnet_dropout(outputs, rng=None, **kwargs):
    from blocks.roles import has_roles, OUTPUT
    cnn_outputs = OrderedDict()
    for var in theano.gof.graph.ancestors(outputs):
        if (has_roles(var, [OUTPUT]) and util.annotated_by_a(
                util.get_convolution_classes(), var)):
            cnn_outputs.setdefault(util.get_path(var), []).append(var)
    unique_outputs = []
    for path, vars in cnn_outputs.items():
        vars = util.dedup(vars, equal=util.equal_computations)
        unique_outputs.append(util.the(vars))
    graph.add_transform(
        unique_outputs,
        graph.DropoutTransform("convnet_dropout", rng=rng),
        reason="regularization")

@util.checkargs
def prepare_mode(mode, outputs, emitter, hyperparameters, **kwargs):
    if mode == "training":
        hyperparameters["rng"] = util.get_rng(seed=1)
        emitter.tag_dropout(outputs, **hyperparameters)
        tag_convnet_dropout(outputs, **hyperparameters)
        logger.warning("%i variables in %s graph" % (util.graph_size(outputs), mode))
        outputs = graph.apply_transforms(outputs, reason="regularization",
                                         hyperparameters=hyperparameters)
        logger.warning("%i variables in %s graph" % (util.graph_size(outputs), mode))

        updates = bricks.BatchNormalization.get_updates(outputs)
        print "batch normalization updates:", updates

        return outputs, updates
    elif mode == "inference":
        logger.warning("%i variables in %s graph" % (util.graph_size(outputs), mode))
        outputs = graph.apply_transforms(
            outputs, reason="population_normalization",
            hyperparameters=hyperparameters)
        logger.warning("%i variables in %s graph" % (util.graph_size(outputs), mode))
        return outputs, []

@util.checkargs
def construct_graphs(task, hyperparameters, **kwargs):
    x, x_shape, y = task.get_variables()

    convnet = construct_model(task=task, **hyperparameters)
    convnet.initialize()

    h = convnet.apply(x)
    h = h.flatten(ndim=2)

    emitter = task.get_emitter(
        input_dim=np.prod(convnet.get_dim("output")),
        **hyperparameters)
    emitter.initialize()

    emitter_outputs = emitter.emit(h, y)
    cost = emitter_outputs.cost.copy(name="cost")

    # gather all the outputs we could possibly care about for training
    # *and* monitoring; prepare_graphs will do graph transformations
    # after which we may *only* use these to access *any* variables.
    outputs_by_name = OrderedDict()
    for key in "x x_shape cost".split():
        outputs_by_name[key] = locals()[key]
    for key in task.monitor_outputs():
        outputs_by_name[key] = emitter_outputs[key]
    outputs = list(outputs_by_name.values())

    # construct training and inference graphs
    mode_by_set = OrderedDict([
        ("train", "training"),
        ("valid", "inference"),
        ("test", "inference")])
    outputs_by_mode, updates_by_mode = OrderedDict(), OrderedDict()
    for mode in "training inference".split():
        (outputs_by_mode[mode],
         updates_by_mode[mode]) = prepare_mode(
             mode, outputs, convnet=convnet, emitter=emitter, **hyperparameters)
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
def construct_main_loop(name, task_name, batch_size, max_epochs,
                        patience_epochs, learning_rate,
                        hyperparameters, **kwargs):
    task = tasks.get_task(**hyperparameters)
    hyperparameters["n_channels"] = task.n_channels

    extensions = []

    print "constructing graphs..."
    graphs, outputs, updates = construct_graphs(task=task, **hyperparameters)

    print "setting up main loop..."

    from blocks.model import Model
    model = Model(outputs["train"]["cost"])

    from blocks.algorithms import GradientDescent, CompositeRule, StepClipping, Adam
    algorithm = GradientDescent(
        cost=outputs["train"]["cost"],
        parameters=graphs["train"].parameters,
        step_rule=CompositeRule([StepClipping(1e1),
                                 Adam(learning_rate=learning_rate),
                                 StepClipping(1e2)]),
        on_unused_sources="warn")
    algorithm.add_updates(updates["train"])

    extensions.extend(construct_monitors(
        algorithm=algorithm, task=task, model=model, graphs=graphs,
        outputs=outputs, **hyperparameters))

    from blocks.extensions import FinishAfter, Printing, ProgressBar, Timing
    from blocks.extensions.stopping import FinishIfNoImprovementAfter
    from blocks.extensions.training import TrackTheBest
    from blocks.extensions.saveload import Checkpoint
    from dump import DumpBest, LightCheckpoint, PrintingTo
    extensions.extend([
        TrackTheBest("valid_error_rate", "best_valid_error_rate"),
        FinishIfNoImprovementAfter("best_valid_error_rate", epochs=patience_epochs),
        FinishAfter(after_n_epochs=max_epochs),
        DumpBest("best_valid_error_rate", name+"_best.zip"),
        Checkpoint(hyperparameters["checkpoint_save_path"],
                   on_interrupt=False, every_n_epochs=5,
                   before_training=True, use_cpickle=True),
        ProgressBar(), Timing(), Printing(), PrintingTo(name+"_log")])

    from blocks.main_loop import MainLoop
    main_loop = MainLoop(data_stream=task.get_stream("train"),
                         algorithm=algorithm,
                         extensions=extensions,
                         model=model)

    # note blocks will crash and burn because it cannot deal with an
    # already-initialized Algorithm, so this should be enabled only for
    # debugging
    if False:
        with open("graph", "w") as graphfile:
            algorithm.initialize()
            theano.printing.debugprint(algorithm._function, file=graphfile)

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
    parser.add_argument("--checkpoint", help="LightCheckpoint zipfile from which to resume training")
    parser.add_argument("--autoresume", action="store_true", help="Resume from default checkpoint path or start training if it does not exist")

    args = parser.parse_args()

    hyperparameters_path = getattr(
        args, "hyperparameters",
        os.path.join(os.path.dirname(__file__), "defaults.yaml"))

    with open(hyperparameters_path, "rb") as f:
        hyperparameters = yaml.load(f)

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
