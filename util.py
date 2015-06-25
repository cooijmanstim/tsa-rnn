import itertools as it
import numbers
from theano.compile import ViewOp

import theano.tensor as T

def broadcast_index(index, axes, ndim):
    dimshuffle_args = ['x'] * ndim
    if isinstance(axes, numbers.Integral):
        axes = [axes]
    for i, axis in enumerate(axes):
        dimshuffle_args[axis] = i
    return index.dimshuffle(*dimshuffle_args)

def broadcast_indices(index_specs, ndim):
    indices = []
    for index, axes in index_specs:
        indices.append(broadcast_index(index, axes, ndim))
    return indices

def subtensor(x, index_specs):
    indices = broadcast_indices(index_specs, x.ndim)
    return x[tuple(indices)]

class WithDifferentiableApproximation(ViewOp):
    __props__ = ()

    def make_node(self, fprop_output, bprop_output):
        # avoid theano wasting time computing the gradient of fprop_output
        fprop_output = theano.gradient.disconnected_grad(fprop_output)
        return gof.Apply(self, [fprop_output, bprop_output], [f.type()])

    def grad(self, wrt, input_gradients):
        import pdb; pdb.set_trace()
        # check that we need input_gradients[1] rather than input_gradients[:][1]
        return input_gradients[1]

def with_differentiable_approximation(fprop_output, bprop_output):
    return WithDifferentiableApproximation()(fprop_output, bprop_output)

