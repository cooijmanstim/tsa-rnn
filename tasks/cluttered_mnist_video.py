import numpy as np
import fuel.datasets
from base import Classification

class Task(Classification):
    name = "cluttered_mnist_video"

    def __init__(self, *args, **kwargs):
        self.parameters = dict(
            video_shape=(20, 100, 100),
            n_distractors=8,
            distractor_shape=(8, 8),
            occlusion_radius=4,
            occlusion_step=12)
        self.n_channels = 1
        self.n_classes = 10
        super(Task, self).__init__(*args, **kwargs)

    def load_datasets(self):
        return dict(
            train=ClutteredMNISTVideo(self.parameters, which_sets=["train"], subset=slice(0, 50000)),
            # TODO: ensure valid/test set constant!
            valid=ClutteredMNISTVideo(self.parameters, which_sets=["train"], subset=slice(50000, 60000)),
            test=ClutteredMNISTVideo(self.parameters, which_sets=["test"]))

    def get_stream_num_examples(self, which_set, monitor):
        if monitor and which_set == "train":
            return 10000
        return super(Task, self).get_stream_num_examples(which_set, monitor)

    def preprocess(self, data):
        x, y = data
        # remove bogus singleton dimension
        y = y.flatten()
        # introduce channel axis
        x = x[:, np.newaxis, ...]
        x_shape = np.tile([x.shape[2:]], (x.shape[0], 1))
        return (x.astype(np.float32),
                x_shape.astype(np.float32),
                y.astype(np.uint8))

class ClutteredMNISTVideo(fuel.datasets.MNIST):
    def __init__(self, parameters, **kwargs):
        super(ClutteredMNISTVideo, self).__init__(**kwargs)
        self.parameters = parameters

    def apply_default_transformers(self, data_stream):
        return ClutteredMNISTVideoTransformer(
            data_stream,
            mnist=self,
            **self.parameters)

class ClutteredMNISTVideoTransformer(fuel.transformers.Transformer):
    def __init__(self, data_stream, mnist, video_shape,
                 n_distractors=8, distractor_shape=(8, 8),
                 occlusion_radius=4, occlusion_step=12,
                 **kwargs):
        super(ClutteredMNISTVideoTransformer, self).__init__(
            data_stream, data_stream.produces_examples, **kwargs)
        self.video_shape = np.array(video_shape)
        self.n_distractors = n_distractors
        self.distractor_shape = np.array(distractor_shape)
        self.occlusion_radius = occlusion_radius
        self.occlusion_step = occlusion_step
        self.mnist = mnist
        self.rng = np.random.RandomState(0)

    @property
    def sources(self):
        return self.data_stream.sources

    def transform_example(self, example):
        image, target = example
        video = self.transform_image(image)
        return video

    def transform_batch(self, batch):
        images, targets = batch
        videos = np.array(map(self.transform_image, images))
        return videos, targets

    def transform_image(self, digit):
        # NOTE: we use the digit's channel axis as a batch axis
        digit_trajectory = self.sample_trajectories(digit)
        video = np.zeros(self.video_shape, dtype=np.float32)
        video = self.render_trajectories(video, *digit_trajectory)
        video = self.render_occlusions(video, *digit_trajectory)
        video = self.render_trajectories(
            video, *self.sample_trajectories(
                self.sample_distractors(self.n_distractors)))
        video = np.clip(video, 0.0, 1.0)
        return video

    def sample_trajectories(self, patches):
        n = patches.shape[0]
        # choose a time t at which the patch must be fully inside the image
        t = self.rng.randint(self.video_shape[0], size=(n,))
        # choose a location for time t
        xmax = self.video_shape[1:] - patches.shape[1:]
        xt = np.round(self.rng.random_sample((n, 2)) * xmax).astype(np.float32)
        # choose a velocity
        v = self.rng.normal(size=(n, 2), scale=2.).astype(np.float32)
        return t, xt, v, patches

    def render_trajectories(self, video, *trajectories):
        scratch = video*0 # allocate once
        for i, (t, xt, v, patch) in enumerate(zip(*trajectories)):
            scratch.fill(0)
            self.place(scratch[t], xt, patch)
            for k in xrange(scratch.shape[0]):
                if k == t:
                    continue
                dx = (k - t)*v
                self.roll_into(scratch[k], scratch[t], dx)
            video += scratch
        return video

    def render_occlusions(self, video, *digit_trajectory):
        radius, step = self.occlusion_radius, self.occlusion_step
        for i, (t, xt, v, patch) in enumerate(zip(*digit_trajectory)):
            # use axis-aligned bars for occlusion, perpendicular to the
            # velocity of the digit
            dim = v.argmax()
            # move the occlusions in the opposite direction to maximize
            # the movement of the occlusions across the digit
            vo = -1 / (1 + v[dim])
            index = list(map(slice, video.shape[1:]))
            for k in xrange(video.shape[0]):
                offset = int(k*vo) % step
                for j in xrange(radius):
                    index[dim] = slice(offset + j, None, step)
                    video[(k,) + tuple(index)] = 0
        return video

    def sample_distractors(self, n):
        images, _ = self.mnist.get_data(
            request=self.rng.randint(self.mnist.num_examples, size=(n,)))
        images = np.squeeze(images, axis=1)
        # randint doesn't support high=some_shape, so use random_sample
        maxoffset = images.shape[1:] - self.distractor_shape
        offset = np.round(self.rng.random_sample((n, 2)) * maxoffset).astype(int)
        y = offset[:, 0, np.newaxis] + np.arange(self.distractor_shape[0])[np.newaxis, ...]
        x = offset[:, 1, np.newaxis] + np.arange(self.distractor_shape[1])[np.newaxis, ...]
        return images[np.arange(images.shape[0])[:, np.newaxis, np.newaxis],
                      y[:, :, np.newaxis], x[:, np.newaxis, :]]

    def place(self, frame, x, patch):
        # place the patch into the frame at continuous location x
        x = np.round(x)
        # XXX: can use frame.shape?
        frame_shape = self.video_shape[1:]
        pa = np.clip(            - x, (0, 0), patch.shape)
        pb = np.clip(frame_shape - x, (0, 0), patch.shape)
        fa = np.clip(x              , (0, 0), frame_shape)
        fb = np.clip(x + patch.shape, (0, 0), frame_shape)
        frame[fa[0]:fb[0], fa[1]:fb[1]] += patch[pa[0]:pb[0], pa[1]:pb[1]]

    def roll_into(self, dest, src, ds):
        # like np.roll but multidimensional and inplace
        #assert dest.shape == src.shape
        #assert np.allclose(0, dest)
        try:
            dest_indices, src_indices = zip(*(
                self._roll_into_dimslice(int(d), dim)
                for d, dim in zip(ds, dest.shape)))
            dest[dest_indices] = src[src_indices]
        except Nevermind:
            pass

    def _roll_into_dimslice(self, d, n):
        if abs(d) >= n:
            # believe it or not, checking in here and raising a
            # control-flow exception is much faster than checking before
            # the loop in roll_into
            raise Nevermind()
        leftright = slice(0, n - abs(d)), slice(abs(d), n)
        if d >= 0:
            return leftright[1], leftright[0]
        else:
            return leftright

class Nevermind(Exception):
    pass
