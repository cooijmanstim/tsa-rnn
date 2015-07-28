from fuel.datasets.mnist import MNIST
import tasks

class Task(tasks.Classification):
    def __init__(self, *args, **kwargs):
        super(Task, self).__init__(*args, **kwargs)
        self.n_classes = 10
        self.n_channels = 1

    def load_datasets(self):
        return dict(
            train=MNIST(which_sets=["train"], subset=slice(None, 50000)),
            valid=MNIST(which_sets=["train"], subset=slice(50000, None)),
            test=MNIST(which_sets=["test"]))
