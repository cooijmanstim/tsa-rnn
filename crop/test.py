import numpy as np
import theano
import theano.tensor as T
import h5py
from StringIO import StringIO
from PIL import Image

from op import TimCropperOp
from grad import TimCropperGradOp

if __name__ == "__main__":
    import theano, theano.tensor as T
    patch_shape = (8, 8)
    ndim_spatial = len(patch_shape)
    crop = TimCropperOp(patch_shape)

    x = T.tensor4("x")
    a, b = T.imatrix("a"), T.imatrix("b")
    l, s = T.matrix("l"), T.matrix("s")
    y = crop(x,
             T.cast(a, theano.config.floatX),
             T.cast(b, theano.config.floatX),
             l, s)
    f = theano.function([x, a, b, l, s], y)

    def load_frame(bytes):
        image = Image.open(StringIO(bytes.tostring()))
        image = (np.array(image.getdata(), dtype=np.float32)
                 .reshape((image.size[1], image.size[0], 3)))
        image /= 255.0
        return image

    np.random.seed(1)
    with h5py.File("/data/lisatmp3/ballasn/datasets/UCF101/jpeg_data.hdf5", "r") as ucf101:
        images = ucf101["images"][[128, 256, 512, 1024, 2048, 4096, 8192]]
        np_x = np.asarray(list(map(load_frame, images)))
    np_x = np.rollaxis(np_x, np_x.ndim - 1, 1)

    # less
    np_x = np_x[:, :, 100:116, 100:116]

    input_shape = np_x.shape[2:]
    print input_shape
    np_l = np.random.rand(np_x.shape[0], ndim_spatial) * input_shape
    np_s = 0.1 + 3 * np.random.rand(np_x.shape[0], ndim_spatial)
    np_a, np_b = (0 * np_l).astype(np.int32), (0 * np_l + input_shape).astype(np.int32)
    np_l, np_s = np_l.astype(theano.config.floatX), np_s.astype(theano.config.floatX)

    np_y = np.asarray(f(np_x, np_a, np_b, np_l, np_s))

    np_gys = theano.function([x, a, b, l, s], T.grad(y.mean(), [l, s]))(np_x, np_a, np_b, np_l, np_s);
    print [(np_gy.shape, np_gy) for np_gy in np_gys]

    if True:
        print "verifying grad"
        T.verify_grad(
            lambda l, s: crop(
                T.constant(np_x),
                T.constant(np_a.astype(theano.config.floatX)),
                T.constant(np_b.astype(theano.config.floatX)),
                l, s),
            [np_l, np_s],
            rng=np.random)
        print "grad verified"

    if False:
        for image, patch, location, scale in itertools.izip(np_x, np_y, np_l, np_s):
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(np.rollaxis(image, 0, image.ndim), interpolation="nearest")
            plt.figure()
            plt.imshow(np.rollaxis(patch, 0, patch.ndim), interpolation="nearest")
            plt.title("l %s s %s" % (location, scale))
            plt.show()
