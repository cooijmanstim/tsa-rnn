import os
from fuel.datasets import H5PYDataset
import tasks

class MNISTCluttered(H5PYDataset):
    filename = 'mnist-cluttered.hdf5'

    def __init__(self, which_set, **kwargs):
        kwargs.setdefault('load_in_memory', True)
        super(MNISTCluttered, self).__init__(self.data_path,
                                             which_set, **kwargs)

    @property
    def data_path(self):
        return os.path.join(os.environ['MNIST_CLUTTERED'],
                            self.filename)

class Task(tasks.Classification):
    def __init__(self, *args, **kwargs):
        super(Task, self).__init__(*args, **kwargs)
        self.n_channels = 1
        self.n_classes = 10

    def load_datasets(self):
        return dict(
            train=MNISTCluttered(which_sets=["train"], subset=slice(None, 50000)),
            valid=MNISTCluttered(which_sets=["train"], subset=slice(50000, None)),
            test=MNISTCluttered(which_sets=["test"]))
