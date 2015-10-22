import h5py
import numpy as np
from StringIO import StringIO
from PIL import Image
import fuel.datasets

class JpegVideoDataset(fuel.datasets.H5PYDataset):
    def __init__(self, path, which_set):
        file = h5py.File(path, "r")
        super(JpegVideoDataset, self).__init__(
            file, sources=tuple("videos targets".split()),
            which_sets=(which_set,), load_in_memory=True)
        # TODO: find a way to deal with `which_sets`, especially when
        # they can be discontiguous and when `subset` is provided, and
        # when all the video ranges need to be adjusted to account for this
        self.frames = np.array(file["frames"][which_set])
        file.close()

    def get_data(self, *args, **kwargs):
        video_ranges, targets = super(JpegVideoDataset, self).get_data(*args, **kwargs)
        videos = list(map(self.video_from_jpegs, video_ranges))
        return videos, targets

    def video_from_jpegs(self, video_range):
        frames = self.frames[video_range[0]:video_range[1]]
        video = np.array(map(self.load_frame, frames))
        return video

    def load_frame(self, jpeg):
        image = Image.open(StringIO(jpeg.tostring()))
        image = (np.array(image.getdata(), dtype=np.float32)
                 .reshape((image.size[1], image.size[0])))
        image /= 255.0
        return image
