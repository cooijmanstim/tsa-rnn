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

class RectangularCropper(Brick):
    def __init__(self, n_spatial_dims, image_shape, patch_shape, kernel, **kwargs):
        super(RectangularCropper, self).__init__(**kwargs)
        self.patch_shape = patch_shape
        self.image_shape = image_shape
        self.kernel = kernel
        self.n_spatial_dims = n_spatial_dims
        self.precompute()

    def precompute(self):
        # compute most of the stuff that deals with indices outside of
        # the scan function to avoid gpu/host transfers due to the use
        # of integers.  basically, if our scan body deals with
        # integers, the whole scan loop will move onto the cpu.
        self.ImJns = []
        for axis in xrange(self.n_spatial_dims):
            m = T.cast(self.image_shape[axis], 'float32')
            n = T.cast(self.patch_shape[axis], 'float32')
            I = T.arange(m).dimshuffle('x', 0, 'x') # (1, image_dim, 1)
            J = T.arange(n).dimshuffle('x', 'x', 0) # (1, 1, patch_dim)
            self.ImJns.append((I, m, J, n))

    def compute_crop_matrices(self, locations, scales):
        Ws = []
        for axis, (I, m, J, n) in enumerate(self.ImJns):
            location = locations[:, axis].dimshuffle(0, 'x', 'x')   # (batch_size, 1, 1)
            scale    = scales   [:, axis].dimshuffle(0, 'x', 'x')   # (batch_size, 1, 1)

            # linearly map locations in [-1, 1] into image index space
            location = (location + 1)/2 * m                         # (batch_size, 1, 1)

            # map patch index into image index space
            J = (J - 0.5*n) / scale + location                      # (batch_size, 1, patch_dim)

            # compute squared distances between image index and patch
            # index in the current dimension:
            #   dx**2 = (i - j)*(i - j)
            #               where i is image index
            #                     j is patch index mapped into image space
            #         = i**2 + j**2 -2ij
            #         = I**2 + J**2 -2IJ'  for all i,j in one swoop

            IJ = I * J                # (batch_size, image_dim, patch_dim)
            dx2 = I**2 + J**2 - 2*IJ  # (batch_size, image_dim, patch_dim)

            Ws.append(self.kernel(dx2, scale))
        return Ws

    @application(inputs=['image', 'location', 'scale'], outputs=['patch'])
    def apply(self, image, location, scale):
        matrices = self.compute_crop_matrices(location, scale)
        patch = image
        for axis, matrix in enumerate(matrices):
            patch = T.batched_tensordot(patch, matrix, [[2], [1]])
        return patch

def gaussian(x2, scale=1):
    sigma = 0.5 / scale
    volume = T.sqrt(2*math.pi)*sigma
    return T.exp(-0.5*x2/(sigma**2)) / volume

# shape (batch, channel, height, width)
x = T.tensor4('features', dtype=floatX)
# shape (batch_size, ntargets)
y = T.lmatrix('targets')

locator = masonry.Locator(hidden_dim, area_dim, n_spatial_dims)
cropper = RectangularCropper(n_spatial_dims, x.shape[-n_spatial_dims:], patch_shape, gaussian)
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
