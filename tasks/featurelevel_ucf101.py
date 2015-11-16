import os, logging, cPickle, zlib, h5py, functools
from StringIO import StringIO
import numpy as np
import theano, theano.tensor as T
import fuel.transformers

import tasks
import transformers

logger = logging.getLogger(__name__)

max_duration = 100
def bound_duration(sources, augment=False):
    duration = sources[0].shape[1]
    crop_duration = duration
    if augment:
        crop_duration = duration / 2
    crop_duration = min(crop_duration, max_duration)
    if duration > crop_duration:
        sources = list(sources)
        offset = np.random.randint(0, duration - crop_duration)
        for i, source in enumerate(sources):
            sources[i] = source[:, offset:(offset + crop_duration), ...]
    return sources

class FeaturelevelUCF101Dataset(fuel.datasets.H5PYDataset):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("load_in_memory", True)
        path = os.environ["FEATURELEVEL_UCF101_HDF5"]
        super(FeaturelevelUCF101Dataset, self).__init__(path, *args, **kwargs)

    def get_data(self, *args, **kwargs):
        sources = list(super(FeaturelevelUCF101Dataset, self).get_data(*args, **kwargs))
        for i in range(2):
            sources[i] = list(map(cPickle.load, map(StringIO, map(zlib.decompress, sources[i]))))
            # move channel axis before time axis
            sources[i] = [np.rollaxis(x, 1, 0) for x in sources[i]]
        sources[:2] = list(zip(*list(map(
            functools.partial(bound_duration,
                              augment=self.which_sets[0] == "train"),
            zip(*sources[:2])))))
        # so i accidentally mixed up the two when generating the dataset
        sources[0], sources[1] = sources[1], sources[0]
        # flatten the degenerate spatial dimensions on the fc features
        sources[0] = [np.reshape(x, (x.shape[0], -1))
                      for x in sources[0]]
        # so targets are 1-based -_-
        sources[2] -= 1
        return sources

def _canonicalize(self, data):
    fc, fc_shapes, conv, conv_shapes, targets = data
    return (fc.astype(theano.config.floatX),
            fc_shapes.astype(theano.config.floatX),
            conv.astype(theano.config.floatX),
            conv_shapes.astype(theano.config.floatX),
            targets.astype(np.uint8))

def _center(self, data):
    return data

class Task(tasks.Classification):
    name = "featurelevel_ucf101"
    canonicalize = _canonicalize
    center = _center

    def __init__(self, *args, **kwargs):
        super(Task, self).__init__(*args, **kwargs)
        self.n_classes = 101
        self.n_channels = None # should be unused

    def get_variables(self):
        fc = T.tensor3("fc")
        conv = T.TensorType(broadcastable=[False]*5,
                                     dtype=theano.config.floatX)("conv")
        fc_shapes = T.matrix("fc_shapes")
        conv_shapes = T.matrix("conv_shapes")

        targets = T.ivector("targets")

        test_batch = self.get_stream("train").get_epoch_iterator(as_dict=True).next()
        for key, value in test_batch.items():
            locals()[key].tag.test_value = value[:11]

        # x is secretly a tuple of these two variables; UCF101's cropper thingamajig knows about this
        x = (fc, conv)
        x_shape = (fc_shapes, conv_shapes)
        y = targets
        return x, x_shape, y

    def load_datasets(self):
        return dict(
            train=FeaturelevelUCF101Dataset(which_sets=["train"]),
            test= FeaturelevelUCF101Dataset(which_sets=["test"]))

    def get_stream_num_examples(self, which_set, monitor):
        num_examples = super(Task, self).get_stream_num_examples(which_set, monitor)
        if monitor and which_set == "train":
            return min(1000, num_examples)
        return num_examples

    def apply_default_transformers(self, stream, monitor):
        stream = transformers.PaddingShape(
            stream, shape_sources="fc conv".split())
        return stream

    def get_stream(self, *args, **kwargs):
        stream = super(Task, self).get_stream(*args, **kwargs)
        return FeaturelevelSources(stream)

class FeaturelevelSources(fuel.transformers.AgnosticTransformer):
    def __init__(self, stream):
        super(FeaturelevelSources, self).__init__(
            stream, stream.produces_examples)
        self.sources = tuple("fc fc_shapes conv conv_shapes targets".split())

    def transform_any(self, data):
        return data
