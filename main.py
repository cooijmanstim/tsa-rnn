import os, logging, yaml
from collections import OrderedDict
import theano
import theano.tensor as T
from blocks.graph import ComputationGraph
import util, attention, crop, tasks, dump, graph, bricks

# disable cached constants. this keeps the graph from ballooning with
# map_variables.
T.constant.enable = False

floatX = theano.config.floatX

@util.checkargs
def construct_model(patch_shape, hidden_dim, hyperparameters, **kwargs):
    cropper = crop.LocallySoftRectangularCropper(
        name="cropper", kernel=crop.Gaussian(),
        patch_shape=patch_shape, hyperparameters=hyperparameters)
    return attention.RecurrentAttentionModel(
        hidden_dim=hidden_dim, cropper=cropper,
        hyperparameters=hyperparameters,
        # attend based on upper RNN states
        attention_state_name="states#1")

@util.checkargs
def construct_monitors(algorithm, task, model, graphs, outputs,
                       n_spatial_dims, plot_url, hyperparameters,
                       patchmonitor_interval, **kwargs):
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
            if parameter.name in "gamma beta".split():
                quantity = parameter.mean()
                quantity.name = "parameter:%s.mean" % util.get_path(parameter)
                data_independent_channels.append(quantity)
        for key in "location_std scale_std".split():
            data_independent_channels.append(hyperparameters[key].copy(name="parameter:%s" % key))
        extensions.append(DataStreamMonitoring(
            data_independent_channels, data_stream=None, after_epoch=True))

    for which_set in "train valid test".split():
        channels = []
        channels.extend(outputs[which_set][key] for key in
                        "cost emitter_cost excursion_cost".split())
        channels.extend(outputs[which_set][key] for key in
                        task.monitor_outputs())
        channels.append(outputs[which_set]["savings"]
                        .mean().copy(name="mean_savings"))
        for key in "raw_location raw_scale".split():
            for stat in "mean var".split():
                channels.append(getattr(outputs[which_set][key], stat)(axis=1)
                                .copy(name="%s.%s" % (key, stat)))
        if which_set == "train":
            channels.append(algorithm.total_gradient_norm.copy(name="total_gradient_norm"))
        extensions.append(DataStreamMonitoring(
            channels, prefix=which_set, after_epoch=True,
            data_stream=task.get_stream(which_set, monitor=True)))

    patchmonitor = None
    if n_spatial_dims == 2:
        patchmonitor_klass = PatchMonitoring
    elif n_spatial_dims == 3:
        patchmonitor_klass = VideoPatchMonitoring

    if patchmonitor_klass:
        for which in "train valid".split():
            patch = outputs[which]["patch"]
            patch = patch.dimshuffle(1, 0, *range(2, patch.ndim))
            patch_extractor = theano.function(
                [outputs[which][key] for key in "x x_shape".split()],
                [outputs[which][key] for key in "raw_location raw_scale".split()] + [patch])

            patchmonitor = patchmonitor_klass(
                save_to="%s_patches_%s" % (hyperparameters["name"], which),
                data_stream=task.get_stream(which, shuffle=False, num_examples=10),
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

@util.checkargs
def prepare_mode(mode, outputs, ram, emitter, hyperparameters, **kwargs):
    if mode == "training":
        hyperparameters["rng"] = util.get_rng(seed=1)
        emitter.tag_dropout(outputs, **hyperparameters)
        ram.tag_attention_dropout(outputs, **hyperparameters)
        ram.tag_recurrent_weight_noise(outputs, **hyperparameters)
        ram.tag_recurrent_dropout(outputs, **hyperparameters)
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
def construct_graphs(task, n_patches, hyperparameters, **kwargs):
    x, x_shape, y = task.get_variables()

    ram = construct_model(task=task, **hyperparameters)
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

    emitter_outputs = emitter.emit(scopes[-1].rnn_outputs["states"], y, n_patches)
    emitter_cost = emitter_outputs.cost.copy(name="emitter_cost")
    # take abs to avoid sudden huge gradient
    excursion_cost = sum(scope.excursion for scope in scopes).sum().copy(name="excursion_cost")
    cost = (emitter_cost + excursion_cost).copy(name="cost")

    # gather all the outputs we could possibly care about for training
    # *and* monitoring; prepare_graphs will do graph transformations
    # after which we may *only* use these to access *any* variables.
    outputs_by_name = OrderedDict()
    for key in "x x_shape emitter_cost excursion_cost cost".split():
        outputs_by_name[key] = locals()[key]
    for key in task.monitor_outputs():
        outputs_by_name[key] = emitter_outputs[key]
    for key in "raw_location raw_scale patch savings".split():
        outputs_by_name[key] = T.stack([scope[key] for scope in scopes])
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
    name = "%s_%s" % (name, task_name)
    hyperparameters["name"] = name

    task = tasks.get_task(**hyperparameters)
    hyperparameters["n_channels"] = task.n_channels

    extensions = []

    # let theta noise decay as training progresses
    from blocks.extensions.training import SharedVariableModifier
    for key in "location_std scale_std".split():
        hyperparameters[key] = theano.shared(hyperparameters[key], name=key)
        rate = hyperparameters["%s_decay" % key]
        extensions.append(SharedVariableModifier(
            hyperparameters[key],
            lambda i, x: rate * x))

    theano.config.compute_test_value = "warn"

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
                                 StepClipping(1e2)]))
    algorithm.add_updates(updates["train"])

    extensions.extend(construct_monitors(
        algorithm=algorithm, task=task, model=model, graphs=graphs,
        outputs=outputs, **hyperparameters))

    from blocks.extensions import FinishAfter, Printing, ProgressBar, Timing
    from blocks.extensions.stopping import FinishIfNoImprovementAfter
    from blocks.extensions.training import TrackTheBest
    from dump import DumpBest, LightCheckpoint, PrintingTo
    extensions.extend([
        TrackTheBest("valid_error_rate", "best_valid_error_rate"),
        FinishIfNoImprovementAfter("best_valid_error_rate", epochs=patience_epochs),
        FinishAfter(after_n_epochs=max_epochs),
        DumpBest("best_valid_error_rate", name+"_best.zip"),
        LightCheckpoint(name+"_checkpoint.zip", on_interrupt=False),
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
    print tabulate((key, value.get_value().size) for key, value in
                   main_loop.model.get_parameter_dict().items())

    return main_loop

if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger(__name__)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hyperparameters", help="YAML file from which to load hyperparameters")
    parser.add_argument("--checkpoint", help="LightCheckpoint zipfile from which to resume training")

    args = parser.parse_args()

    hyperparameters_path = getattr(
        args, "hyperparameters",
        os.path.join(os.path.dirname(__file__), "defaults.yaml"))

    with open(hyperparameters_path, "rb") as f:
        hyperparameters = yaml.load(f)

    hyperparameters["n_spatial_dims"] = len(hyperparameters["patch_shape"])
    hyperparameters["hyperparameters"] = hyperparameters

    main_loop = construct_main_loop(**hyperparameters)

    if args.checkpoint:
        dump.load_main_loop(main_loop, args.checkpoint)

    print "training..."
    main_loop.run()
