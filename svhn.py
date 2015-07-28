from fuel.transformers import Mapping
from fuel.datasets.svhn import SVHN
import tasks

def fix_target_representation(data):
    x, y = data
    # use zero to represent zero
    y[y == 10] = 0
    return x, y

class DigitTask(tasks.Classification):
    def __init__(self, *args, **kwargs):
        super(DigitTask, self).__init__(*args, **kwargs)
        self.n_classes = 10
        self.n_channels = 1

    def load_datasets(self):
        return dict(
            train=SVHN(which_sets=["train"], which_format=2, subset=slice(None, 50000)),
            valid=SVHN(which_sets=["train"], which_format=2, subset=slice(50000, None)),
            test=SVHN(which_sets=["test"], which_format=2))

    def get_stream(self, *args, **kwargs):
        return Mapping(super(DigitTask, self).get_stream(*args, **kwargs),
                       mapping=fix_target_representation)
