import math

import theano
import theano.tensor as T

floatX = theano.config.floatX

from blocks.bricks import Softmax, Rectifier, Brick, application, MLP

import util

class LocallySoftRectangularCropper(Brick):
    def __init__(self, n_spatial_dims, image_shape, patch_shape, kernel, batched_window=False, **kwargs):
        super(LocallySoftRectangularCropper, self).__init__(**kwargs)
        self.image_shape = T.cast(image_shape, 'int16')
        self.patch_shape = patch_shape
        self.kernel = kernel
        self.n_spatial_dims = n_spatial_dims
        self.batched_window = batched_window

    def compute_crop_matrices(self, locations, scales, Is):
        Ws = []
        for axis in xrange(self.n_spatial_dims):
            m = T.cast(self.image_shape[axis], 'float32')
            n = T.cast(self.patch_shape[axis], 'float32')
            I = Is[axis].dimshuffle('x', 0, 'x')    # (1, hardcrop_dim, 1)
            J = T.arange(n).dimshuffle('x', 'x', 0) # (1, 1, patch_dim)

            location = locations[:, axis].dimshuffle(0, 'x', 'x')   # (batch_size, 1, 1)
            scale    = scales   [:, axis].dimshuffle(0, 'x', 'x')   # (batch_size, 1, 1)

            # map patch index into image index space
            J = (J - 0.5*n) / scale + location                      # (batch_size, 1, patch_dim)

            # compute squared distances between image index and patch
            # index in the current dimension:
            #   dx**2 = (i - j)*(i - j)
            #               where i is image index
            #                     j is patch index mapped into image space
            #         = i**2 + j**2 -2ij
            #         = I**2 + J**2 -2IJ'  for all i,j in one swoop

            IJ = I * J                # (batch_size, hardcrop_dim, patch_dim)
            dx2 = I**2 + J**2 - 2*IJ  # (batch_size, hardcrop_dim, patch_dim)

            Ws.append(self.kernel(dx2, scale))
        return Ws

    def compute_hard_windows(self, location, scale):
        # find topleft(front) and bottomright(back) corners for each patch
        a = location - 0.5 * (self.patch_shape / scale)
        b = location + 0.5 * (self.patch_shape / scale)

        # grow by three patch pixels
        # TODO: choose expansion to capture a given proportion of kernel volume (e.g. 2 sigma)
        a -= 3 / scale
        b += 3 / scale

        if self.batched_window:
            # take the bounding box of all windows; now the slices
            # will have the same length for each sample and scan can
            # be avoided.  comes at the cost of typically selecting
            # more of the input.
            a = a.min(axis=0, keepdims=True)
            b = b.max(axis=0, keepdims=True)

        # make integer
        a = T.cast(T.floor(a), 'int16')
        b = T.cast(T.ceil(b), 'int16')

        # clip to fit inside image and have nonempty window
        a = T.clip(a, 0, self.image_shape - 1)
        b = T.clip(b, a + 1, self.image_shape)

        return a, b

    @application(inputs=['image', 'location', 'scale'], outputs=['patch', 'mean_savings'])
    def apply(self, image, location, scale):
        a, b = self.compute_hard_windows(location, scale)

        if self.batched_window:
            patch = self.apply_inner(image, location, scale, a[0], b[0])
        else:
            def map_fn(image, a, b, location, scale):
                # apply_inner expects a batch axis
                image = T.shape_padleft(image)
                location = T.shape_padleft(location)
                scale = T.shape_padleft(scale)

                patch = self.apply_inner(image, location, scale, a, b)

                # return without batch axis
                return patch[0]

            patch, _ = theano.map(map_fn,
                                  sequences=[image, a, b, location, scale])

        mean_savings = (1 - T.cast((b - a).prod(axis=1), floatX) / self.image_shape.prod()).mean(axis=0)
        self.add_auxiliary_variable(mean_savings, name="mean_savings")

        return patch, mean_savings

    def apply_inner(self, image, location, scale, a, b):
        slices = [theano.gradient.disconnected_grad(T.arange(a[i], b[i]))
                  for i in xrange(self.n_spatial_dims)]
        hardcrop = util.subtensor(
            image,
            [(T.arange(image.shape[0]), 0),
             (T.arange(image.shape[1]), 1)]
             + [(slice, 2 + i) for i, slice in enumerate(slices)])
        matrices = self.compute_crop_matrices(location, scale, slices)
        patch = hardcrop
        for axis, matrix in enumerate(matrices):
            patch = T.batched_tensordot(patch, matrix, [[2], [1]])
        return patch

class SoftRectangularCropper(Brick):
    def __init__(self, n_spatial_dims, image_shape, patch_shape, kernel, **kwargs):
        super(SoftRectangularCropper, self).__init__(**kwargs)
        self.patch_shape = patch_shape
        self.image_shape = image_shape
        self.kernel = kernel
        self.n_spatial_dims = n_spatial_dims
        self.precompute()

    def precompute(self):
        # compute most of the stuff that deals with indices outside of
        # the scan function to avoid gpu/host transfers due to the use
        # of integers.  basically, if our scan body deals with
        # integers, the whole scan loop will move onto the cpu.
        self.ImJns = []
        for axis in xrange(self.n_spatial_dims):
            m = T.cast(self.image_shape[axis], 'float32')
            n = T.cast(self.patch_shape[axis], 'float32')
            I = T.arange(m).dimshuffle('x', 0, 'x') # (1, image_dim, 1)
            J = T.arange(n).dimshuffle('x', 'x', 0) # (1, 1, patch_dim)
            self.ImJns.append((I, m, J, n))

    def compute_crop_matrices(self, locations, scales):
        Ws = []
        for axis, (I, m, J, n) in enumerate(self.ImJns):
            location = locations[:, axis].dimshuffle(0, 'x', 'x')   # (batch_size, 1, 1)
            scale    = scales   [:, axis].dimshuffle(0, 'x', 'x')   # (batch_size, 1, 1)

            # linearly map locations in [-1, 1] into image index space
            location = (location + 1)/2 * m                         # (batch_size, 1, 1)

            # map patch index into image index space
            J = (J - 0.5*n) / scale + location                      # (batch_size, 1, patch_dim)

            # compute squared distances between image index and patch
            # index in the current dimension:
            #   dx**2 = (i - j)*(i - j)
            #               where i is image index
            #                     j is patch index mapped into image space
            #         = i**2 + j**2 -2ij
            #         = I**2 + J**2 -2IJ'  for all i,j in one swoop

            IJ = I * J                # (batch_size, image_dim, patch_dim)
            dx2 = I**2 + J**2 - 2*IJ  # (batch_size, image_dim, patch_dim)

            Ws.append(self.kernel(dx2, scale))
        return Ws

    @application(inputs=['image', 'location', 'scale'], outputs=['patch', 'mean_savings'])
    def apply(self, image, location, scale):
        matrices = self.compute_crop_matrices(location, scale)
        patch = image
        for axis, matrix in enumerate(matrices):
            patch = T.batched_tensordot(patch, matrix, [[2], [1]])
        return patch, 0

def gaussian(x2, scale=1):
    sigma = 0.5 / scale
    volume = T.sqrt(2*math.pi)*sigma
    return T.exp(-0.5*x2/(sigma**2)) / volume
