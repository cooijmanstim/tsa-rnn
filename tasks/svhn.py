from fuel.transformers import Mapping
from fuel.datasets.svhn import SVHN
import tasks

class DigitTask(tasks.Classification):
    name = "svhn_digit"

    def __init__(self, *args, **kwargs):
        super(DigitTask, self).__init__(*args, **kwargs)
        self.n_classes = 10
        self.n_channels = 1

    def load_datasets(self):
        return dict(
            train=SVHN(which_sets=["train"], which_format=2, subset=slice(None, 50000)),
            valid=SVHN(which_sets=["train"], which_format=2, subset=slice(50000, None)),
            test=SVHN(which_sets=["test"], which_format=2))

    def get_stream_num_examples(self, which_set, monitor):
        if monitor and which_set == "train":
            return 10000
        return super(DigitTask, self).get_stream_num_examples(which_set, monitor)

    def preprocess(self, data):
        x, y = data
        # remove bogus singleton dimension
        y = y.flatten()
        y[y == 10] = 0
        x_shape = np.tile([x.shape[2:]], (x.shape[0], 1))
        return (x.astype(np.float32),
                x_shape.astype(np.float32),
                y.astype(np.uint8))

class NumberTask(tasks.Classification):
    name = "svhn_number"

    def __init__(self, *args, **kwargs):
        super(NumberTask, self).__init__(*args, **kwargs)
        self.n_classes = 10
        self.n_channels = 1

    def load_datasets(self):
        return dict(
#            train=SVHN(which_sets="train extra".split(), which_format=1, subset=slice(None, 50000)),
#            valid=SVHN(which_sets=["extra"], which_format=1, subset=slice(0, None)),
            test=SVHN(which_sets=["test"], which_format=1))

    def get_stream_num_examples(self, which_set, monitor):
        if monitor and which_set == "train":
            return 10000
        return super(NumberTask, self).get_stream_num_examples(which_set, monitor)

    def preprocess(self, data):
        x, y = data
        # remove bogus singleton dimension
        y = y.flatten()
        y[y == 10] = 0
        x_shape = np.tile([x.shape[2:]], (x.shape[0], 1))
        return (x.astype(np.float32),
                x_shape.astype(np.float32),
                y.astype(np.uint8))

if __name__ == "__main__":
    from fuel import schemes, streams, transformers, datasets
    dataset = datasets.SVHN(which_format=1,
                            which_sets=["test"],
                            subset=slice(11))
    stream = streams.DataStream.default_stream(
        dataset=dataset,
        iteration_scheme=schemes.SequentialScheme(11, 11))
    batch = stream.get_epoch_iterator(as_dict=True).next()
    import pdb; pdb.set_trace()
