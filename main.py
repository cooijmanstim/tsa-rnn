import math
import sys
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
from blocks.bricks import Softmax, Rectifier, Brick, application, MLP
from blocks.bricks.recurrent import LSTM, SimpleRecurrent
from blocks.graph import ComputationGraph
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.main_loop import MainLoop
from blocks.filter import VariableFilter
from blocks.extras.extensions.plot import Plot

import masonry
import crop
import util
from patchmonitor import PatchMonitoring

name = "attention_rnn"
n_epochs = 100
batch_size = 100
hidden_dim = 1024
n_steps = 8
patch_shape = (8, 8)
n_spatial_dims = len(patch_shape)
patch_dim = reduce(op.mul, patch_shape)
area_dim = 512
n_classes = 10

initargs = dict(weights_init=Orthogonal(),
                biases_init=Constant(0))

floatX = theano.config.floatX

datasets = dict(
    train=MNIST(which_sets=["train"], subset=slice(None, 50000)),
    valid=MNIST(which_sets=["train"], subset=slice(50000, None)),
    test=MNIST(which_sets=["test"]))
datastreams = dict(
    (which,
     DataStream.default_stream(
        dataset,
        iteration_scheme=ShuffledScheme(dataset.num_examples, batch_size)))
    for which, dataset in datasets.items())

# shape (batch, channel, height, width)
x = T.tensor4('features', dtype=floatX)
# shape (batch_size, ntargets)
y = T.lmatrix('targets')

theano.config.compute_test_value = 'warn'
x.tag.test_value = np.random.random((batch_size, 1, 28, 28)).astype("float32")
y.tag.test_value = np.random.random_integers(0, 9, (batch_size, 1)).astype("int64")

locator = masonry.Locator(hidden_dim, area_dim, n_spatial_dims)
cropper = crop.LocallySoftRectangularCropper(
    n_spatial_dims, x.shape[-n_spatial_dims:], patch_shape,
    crop.gaussian,
    batched_window=True)
merger = masonry.Merger(patch_shape, area_dim, hidden_dim,
                        area_posttransform=Rectifier(),
                        response_posttransform=Rectifier(),
                        **initargs)
attention = masonry.SpatialAttention(locator, cropper, merger)
emitter = MLP(activations=[Softmax()],
              dims=[hidden_dim, n_classes],
              **initargs)
rnn = SimpleRecurrent(activation=Rectifier(),
                      dim=hidden_dim,
                      weights_init=Identity(),
                      biases_init=initargs["biases_init"])
model = masonry.RecurrentAttentionModel(rnn, attention, emitter,
                                        **initargs)

model.initialize()

initial_outputs = model.compute_initial_state(x)
step_outputs = model.apply(x=x, h=initial_outputs[1], n_steps=n_steps, batch_size=x.shape[0])
# prepend initial values
step_outputs = [T.concatenate([T.shape_padleft(initial_output), step_output], axis=0)
                for initial_output, step_output in zip(initial_outputs, step_outputs)]
# move batch axis in front of RNN time axis
step_outputs = [step_output.dimshuffle(1, 0, *range(step_output.ndim)[2:])
                for step_output in step_outputs]
yhats, hs, locations, scales, patches = step_outputs
yhat = yhats[:, -1, :]

cross_entropy = CategoricalCrossEntropy().apply(y.flatten(), yhat)
cross_entropy.name = "cross_entropy"
error_rate = MisclassificationRate().apply(y.flatten(), yhat)
error_rate.name = "error_rate"

graph = ComputationGraph(cross_entropy)

#import theano.printing
#theano.printing.pydotprint(theano.function([x, y], cross_entropy), outfile='graph.png', format='png', scan_graphs=True)
#sys.exit(0)

print "setting up main loop..."
algorithm = GradientDescent(cost=cross_entropy,
                            params=graph.parameters,
                            step_rule=RMSProp(learning_rate=1e-4))

channels = util.Channels()
channels.add(cross_entropy)
channels.add(error_rate)
for i in xrange(n_steps):
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

model = Model(cross_entropy)
main_loop = MainLoop(data_stream=datastreams["train"],
                     algorithm=algorithm,
                     extensions=(list(monitors.values()) +
                                 [FinishAfter(after_n_epochs=n_epochs),
                                  ProgressBar(),
                                  Printing(),
                                  Plot(name,
                                       channels=[["%s_cross_entropy" % which for which in monitors.keys()],
                                                 ["%s_error_rate"    % which for which in monitors.keys()]],
                                       after_epoch=True),
                                  patch_monitoring]),
                     model=model)
print "training..."

main_loop.run()
