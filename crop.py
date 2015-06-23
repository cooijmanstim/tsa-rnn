import math

import theano.tensor as T

from blocks.bricks import Softmax, Rectifier, Brick, application, MLP

class RectangularCropper(Brick):
    def __init__(self, n_spatial_dims, image_shape, patch_shape, kernel, **kwargs):
        super(RectangularCropper, self).__init__(**kwargs)
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

    @application(inputs=['image', 'location', 'scale'], outputs=['patch'])
    def apply(self, image, location, scale):
        matrices = self.compute_crop_matrices(location, scale)
        patch = image
        for axis, matrix in enumerate(matrices):
            patch = T.batched_tensordot(patch, matrix, [[2], [1]])
        return patch

def gaussian(x2, scale=1):
    sigma = 0.5 / scale
    volume = T.sqrt(2*math.pi)*sigma
    return T.exp(-0.5*x2/(sigma**2)) / volume
