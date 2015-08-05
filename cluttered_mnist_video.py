import numpy as np

from fuel.transformers import Mapping
from fuel.datasets import H5PYDataset

import tasks

class ClutteredMNISTVideo(H5PYDataset):
    def __init__(self, which_sets, **kwargs):
        kwargs.setdefault('load_in_memory', False)
        super(ClutteredMNISTVideo, self).__init__(
            "/data/lisatmp3/cooijmat/datasets/cluttered-mnist-video/cluttered-mnist-video.hdf5",
            which_sets, **kwargs)

def mapping(data):
    x, y = data
    # move channel just after batch axis
    x = np.rollaxis(x, x.ndim - 1, 1)
    return x, y

class Task(tasks.Classification):
    name = "cluttered_mnist_video"

    def __init__(self, *args, **kwargs):
        super(Task, self).__init__(*args, **kwargs)
        self.n_channels = 1
        self.n_classes = 10

    def load_datasets(self):
        return dict(
            train=ClutteredMNISTVideo(which_sets=["train"]),
            valid=ClutteredMNISTVideo(which_sets=["valid"]),
            test=ClutteredMNISTVideo(which_sets=["test"]))

    def get_stream(self, *args, **kwargs):
        return Mapping(super(Task, self).get_stream(*args, **kwargs),
                       mapping=mapping)
