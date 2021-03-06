import os
import numpy as np

import tasks, util, datasets

def _canonicalize(self, data):
    x, y = data
    x = np.asarray(x)
    # introduce channel axis
    x = np.expand_dims(x, axis=1)
    x_shape = np.tile([x.shape[2:]], (x.shape[0], 1))
    return (x.astype(np.float32),
            x_shape.astype(np.float32),
            y.astype(np.uint8))

def _center(self, data):
    return data

class Task(tasks.Classification):
    name = "cmv"
    canonicalize = _canonicalize
    center = _center

    @util.checkargs
    def __init__(self, video_shape, **kwargs):
        self.n_channels = 1
        self.n_classes = 10
        self.video_shape = video_shape
        super(Task, self).__init__(**kwargs)

    def load_datasets(self):
        filename = "cmv%s_png.hdf5" % "x".join(map(str, self.video_shape))
        return dict(
            (which_set, datasets.PngVideoDataset(
                path=os.path.join(os.environ["CMV_DATADIR"], filename),
                which_set=which_set))
            for which_set in "train valid test".split())

    def get_stream_num_examples(self, which_set, monitor):
        if monitor and which_set == "train":
            return 1000
        return super(Task, self).get_stream_num_examples(which_set, monitor)

    def compute_batch_mean(self, x, x_shape):
        # average over time first
        time = 2
        mean_frame = x.sum(axis=time, keepdims=True)
        mean_frame /= x_shape[:, np.newaxis, [time], np.newaxis, np.newaxis]
        return mean_frame.mean(axis=0, keepdims=True)
