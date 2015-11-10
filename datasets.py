import h5py
import numpy as np
from StringIO import StringIO
from PIL import Image
import fuel.datasets

import operator, functools

np_float32 = np.float32
nparray_from_image = functools.partial(np.array, dtype=np_float32)
image_open = Image.open
tostring = operator.methodcaller("tostring")
asarray = np.asarray

class FramewiseCompressedVideoDataset(fuel.datasets.H5PYDataset):
    def __init__(self, path, which_set):
        file = h5py.File(path, "r")
        super(FramewiseCompressedVideoDataset, self).__init__(
            file, sources=tuple("videos targets".split()),
            which_sets=(which_set,), load_in_memory=True)
        # TODO: find a way to deal with `which_sets`, especially when
        # they can be discontiguous and when `subset` is provided, and
        # when all the video ranges need to be adjusted to account for this
        self.frames = np.array(file["frames"][which_set])
        file.close()

    def get_data(self, *args, **kwargs):
        video_ranges, targets = super(FramewiseCompressedVideoDataset, self).get_data(*args, **kwargs)
        videos = list(map(self.video_from_frames, video_ranges))
        return videos, targets

    def video_from_frames(self, video_range):
        # we need to do a bunch of things to each frame;
        # try to avoid python overhead by using builtins
        return asarray(
            map(nparray_from_image,
                map(image_open,
                    map(StringIO,
                        map(tostring,
                            self.frames[video_range[0]:video_range[1]])))),
            dtype=np_float32) / 255.0

class JpegVideoDataset(FramewiseCompressedVideoDataset):
    pass

class PngVideoDataset(FramewiseCompressedVideoDataset):
    pass
