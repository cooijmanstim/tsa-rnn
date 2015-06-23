import math
import sys
import operator as op

import numpy as np
import theano
import theano.tensor as T

from fuel.datasets import MNIST
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream

from blocks.initialization import IsotropicGaussian, Constant, NdarrayInitialization, Orthogonal
from blocks.utils import named_copy
from blocks.model import Model
from blocks.algorithms import GradientDescent, RMSProp
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing, ProgressBar
from blocks.bricks import Softmax, Rectifier, Brick, application, MLP
from blocks.bricks.recurrent import LSTM, SimpleRecurrent
from blocks.graph import ComputationGraph
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.main_loop import MainLoop
from blocks.filter import VariableFilter

import masonry
import crop
from patchmonitor import PatchMonitoring

n_epochs = 100
batch_size = 100
hidden_dim = 1024
n_steps = 8
patch_shape = (8, 8)
n_spatial_dims = len(patch_shape)
patch_dim = reduce(op.mul, patch_shape)
area_dim = 512

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
n_classes = 10

# shape (batch, channel, height, width)
x = T.tensor4('features', dtype=floatX)
# shape (batch_size, ntargets)
y = T.lmatrix('targets')

locator = masonry.Locator(hidden_dim, area_dim, n_spatial_dims)
cropper = crop.RectangularCropper(n_spatial_dims, x.shape[-n_spatial_dims:], patch_shape, crop.gaussian)
merger = masonry.Merger(patch_shape, area_dim, hidden_dim,
                        area_posttransform=Rectifier(),
                        response_posttransform=Rectifier(),
                        **initargs)
attention = masonry.SpatialAttention(locator, cropper, merger)
emitter = MLP(activations=[Softmax()],
              dims=[hidden_dim, n_classes],
              **initargs)
rnn = SimpleRecurrent(activation=Rectifier(),
                      dim=hidden_dim, **initargs)
model = masonry.RecurrentAttentionModel(rnn, attention, emitter,
                                        **initargs)

model.initialize()

step_outputs = model.apply(x, n_steps=n_steps, batch_size=x.shape[0])
# move batch axis in front of RNN time axis
step_outputs = [step_output.dimshuffle(1, 0, *range(step_output.ndim)[2:])
                for step_output in step_outputs]
yhats, hs, locations, scales, patches = step_outputs
yhat = yhats[:, -1, :]

cost = CategoricalCrossEntropy().apply(y.flatten(), yhat)
error = MisclassificationRate().apply(y.flatten(), yhat)

graph = ComputationGraph(cost)

#import theano.printing
#theano.printing.pydotprint(theano.function([x, y], cost), outfile='graph.png', format='png', scan_graphs=True)
#sys.exit(0)

print "setting up main loop..."
algorithm = GradientDescent(cost=cost,
                            params=graph.parameters,
                            step_rule=RMSProp(learning_rate=1e-4))

monitors = [TrainingDataMonitoring([cost, error], prefix="train", after_epoch=True)]
for which in "valid test".split():
    monitors.append(DataStreamMonitoring(
        [cost, error],
        data_stream=datastreams[which],
        prefix=which,
        after_epoch=True))

patch_monitoring_datastream = DataStream.default_stream(
    datasets["valid"],
    iteration_scheme=ShuffledScheme(5, 5))
patch_monitoring = PatchMonitoring(patch_monitoring_datastream,
                                   theano.function([x], [locations, scales, patches]))
patch_monitoring.save_patches("test.png")

model = Model(cost)
main_loop = MainLoop(data_stream=datastreams["train"],
                     algorithm=algorithm,
                     extensions=(monitors +
                                 [FinishAfter(after_n_epochs=n_epochs),
                                  ProgressBar(),
                                  Printing(),
                                  patch_monitoring]),
                     model=model)
print "training..."

main_loop.run()
