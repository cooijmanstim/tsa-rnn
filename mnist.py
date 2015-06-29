from fuel.datasets.mnist import MNIST
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme

def load(batch_size, shrink_dataset_by=1, **kwargs):
    datasets = dict(
        train=MNIST(which_sets=["train"], subset=slice(None, 50000)),
        valid=MNIST(which_sets=["train"], subset=slice(50000, None)),
        test=MNIST(which_sets=["test"]))
    datastreams = dict(
        (which,
         DataStream.default_stream(
             dataset,
             iteration_scheme=ShuffledScheme(dataset.num_examples / shrink_dataset_by,
                                             batch_size)))
        for which, dataset in datasets.items())
    return datasets, datastreams
