import sys, operator, logging, collections, itertools
import numbers

from collections import OrderedDict
from cStringIO import StringIO

import theano
import theano.tensor.basic
import theano.sandbox.cuda.blas
import theano.printing
import theano.scan_module.scan_utils
import theano.tensor as T

from blocks.filter import VariableFilter

logger = logging.getLogger(__name__)

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

# from http://stackoverflow.com/a/16571630
class StdoutLines(list):
    def __enter__(self):
        self._stringio = StringIO()
        self._stdout = sys.stdout
        sys.stdout = self._stringio
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout


def batched_tensordot(a, b, axes=2):
    return theano.tensor.basic._tensordot_as_dot(
        a, b, axes,
        dot=theano.sandbox.cuda.blas.batched_dot,
        batched=True)

def dedup(xs, equal=operator.is_):
    ys = []
    for x in xs:
        if not any(equal(x, y) for y in ys):
            ys.append(x)
    return ys

# for use with dedup
def equal_computations(a, b):
    return theano.scan_module.scan_utils.equal_computations([a], [b])

from blocks.bricks.base import Brick, ApplicationCall

# attempt to fully qualify an annotated variable
def get_path(x):
    if isinstance(x, (T.TensorVariable,
                      # zzzzzzzzzzzzzzzzzzzzzzzzzzz
                      T.sharedvar.TensorSharedVariable,
                      T.compile.sharedvalue.SharedVariable)):
        paths = list(set(map(get_path, getattr(x.tag, "annotations", []))))
        name = getattr(x.tag, "name", x.name)
        if len(paths) > 1:
            logger.warning(
                "get_path: variable %s has multiple possible origins, using first of [%s]"
                % (name, " ".join(paths)))
        if len(paths) < 1:
            return name
        return paths[0] + "/" + name
    elif isinstance(x, Brick):
        if x.parents:
            paths = list(set(map(get_path, x.parents)))
            if len(paths) > 1:
                logger.warning(
                    "get_path: brick %s has multiple parents, using first of [%s]"
                    % (x.name, " ".join(paths)))
            return paths[0] + "/" + x.name
        else:
            return "/" + x.name
    elif isinstance(x, ApplicationCall):
        return get_path(x.application.brick)
    else:
        raise TypeError()

# decorator to improve python's terrible argument error reporting
def checkargs(f):
    def g(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except TypeError as e:
            type, value, traceback = sys.exc_info()
            if "takes exactly" in e.message:
                import inspect
                argspec = inspect.getargspec(f)
                required_args = argspec.args
                if argspec.defaults:
                    required_args = required_args[:-len(argspec.defaults)]
                required_args = [arg for arg in required_args
                                 if arg not in kwargs]
                missing_args = required_args[len(args):]
                if missing_args:
                    # reraise with more informative message
                    message = ("%s. not given: %s" %
                               (e.message, ", ".join(missing_args)))
                    raise type, (message,), traceback
            raise
    return g

def rectify(x):
    return (x > 0)*x

def all_bricks(bricks):
    fringe = collections.deque(bricks)
    bricks = []
    while fringe:
        brick = fringe.popleft()
        bricks.append(brick)
        fringe.extend(brick.children)
    return bricks

def graph_size(variable_list):
    return len(set(theano.gof.graph.ancestors(variable_list)))

def get_dropout_mask(shape, probability, **rng_args):
    rng = get_rng(**rng_args)
    return (rng.binomial(shape, p=1 - probability,
                         dtype=theano.config.floatX)
            / (1 - probability))

@checkargs
def get_rng(rng=None, seed=None):
    return rng or theano.sandbox.rng_mrg.MRG_RandomStreams(
        1 if seed is None else seed)

def toposort(variables):
    inputs = theano.gof.graph.inputs(variables)
    nodes = theano.gof.graph.io_toposort(inputs, variables)
    outputs = itertools.chain.from_iterable(node.outputs for node in nodes)
    return [output for output in outputs if output in variables]

def equizip(a, b):
    a, b = list(a), list(b)
    assert len(a) == len(b)
    return zip(a, b)

class Scope(object):
    def __init__(self, **kwargs):
        self._dikt = OrderedDict(**kwargs)

    def __getattr__(self, key):
        if key[0] == "_":
            return self.__dict__[key]
        else:
            return self._dikt[key]

    def __setattr__(self, key, value):
        if key[0] == "_":
            self.__dict__[key] = value
        else:
            self._dikt[key] = value

    def __getitem__(self, key):
        return self._dikt[key]

    def __setitem__(self, key, value):
        self._dikt[key] = value

    def __len__(self):
        return len(self._dikt)

    def __iter__(self):
        return iter(self._dikt)

def the(xs):
    assert len(xs) == 1
    return xs[0]

def annotated_by_a(klass, var):
    return any(isinstance(annotation, klass) or
               (hasattr(annotation, "brick") and
                isinstance(annotation.brick, klass))
               for annotation in getattr(var.tag, "annotations", []))

def get_convolution_classes():
    from blocks.bricks import conv as conv2d
    import conv3d
    return (conv2d.Convolutional,
            conv3d.Convolutional)
