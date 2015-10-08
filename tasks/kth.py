import os
import numpy as np
import fuel.transformers

import tasks
import transformers
import datasets

rng = np.random.RandomState(0)
def augment((videos, targets)):
    mintime = min(len(video) for video in videos)
    crop_shape = np.array([mintime, 100, 140])
    offsets = [[rng.randint(0, dim + 1) for dim in video.shape - crop_shape]
               for video in videos]
    videos = [(video
               [tuple(slice(i, i + k)
                      for i, k in zip(offset, crop_shape))]
               # flip horizontal dimension half the time
               [:, :, ::np.random.choice([-1, 1])])
              for video, offset in zip(videos, offsets)]
    return videos, targets

class Task(tasks.Classification):
    name = "kth"

    def __init__(self, *args, **kwargs):
        super(Task, self).__init__(*args, **kwargs)
        self.n_channels = 1
        self.n_classes = 6

    def load_datasets(self):
        return dict(
            (which_set, datasets.JpegVideoDataset(
                path=os.environ["KTH_JPEG_HDF5"],
                which_set=which_set))
            for which_set in "train valid test".split())

    def apply_default_transformers(self, stream):
        # FIXME: don't augment on valid/test
        stream = fuel.transformers.Mapping(
            stream, mapping=augment)
        stream = transformers.PaddingShape(
            stream, shape_sources=["videos"])
        return stream

    def get_stream_num_examples(self, which_set, monitor):
        if monitor and which_set == "train":
            return 300
        return super(Task, self).get_stream_num_examples(which_set, monitor)

    def compute_batch_mean(self, x, x_shape):
        # average over time first
        time = 2
        mean_frame = x.sum(axis=time, keepdims=True)
        mean_frame /= x_shape[:, np.newaxis, [time], np.newaxis, np.newaxis]
        return mean_frame.mean(axis=0, keepdims=True)

    def preprocess(self, data):
        x, x_shape, y = data
        # introduce channel axis
        x = x[:, np.newaxis, ...]
        return (x.astype(np.float32),
                x_shape.astype(np.float32),
                y.astype(np.uint8))
