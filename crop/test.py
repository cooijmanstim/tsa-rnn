import itertools
import numpy as np
import theano
import theano.tensor as T
import h5py
from StringIO import StringIO
from PIL import Image

from op import TimCropperOp
from grad import TimCropperGradOp
from oldcropper import LocallySoftRectangularCropper, Gaussian

def get_data():
    def load_frame(bytes):
        image = Image.open(StringIO(bytes.tostring()))
        image = (np.array(image.getdata(), dtype=np.float32)
                 .reshape((image.size[1], image.size[0], 3)))
        image /= 255.0
        return image

    np.random.seed(1)
    with h5py.File("/data/lisatmp3/ballasn/datasets/UCF101/jpeg_data.hdf5", "r") as ucf101:
        images = ucf101["images"][[128, 256, 512, 1024, 2048, 4096, 8192]]
        x = np.asarray(list(map(load_frame, images)))
    x = np.rollaxis(x, x.ndim - 1, 1)

    # less
    x = x[:, :, 100:116, 100:116]

    input_shape = x.shape[2:]
    print input_shape
    l = np.random.rand(x.shape[0], ndim_spatial) * input_shape
    s = 0.1 + 3 * np.random.rand(x.shape[0], ndim_spatial)
    a, b = (0 * l).astype(np.int32), (0 * l + input_shape).astype(np.int32)
    l, s = l.astype(theano.config.floatX), s.astype(theano.config.floatX)

    return dict(x=x, a=a, b=b, l=l, s=s)

if __name__ == "__main__":
    import theano, theano.tensor as T
    patch_shape = (8, 8)
    ndim_spatial = len(patch_shape)
    crop = TimCropperOp(patch_shape)
    oldcropper = LocallySoftRectangularCropper(
        patch_shape, Gaussian(),
        dict(cutoff=100000, batched_window=True,
             batch_size=7, batch_size_constant=False))

    x = T.tensor4("x")
    a, b = T.imatrix("a"), T.imatrix("b")
    l, s = T.matrix("l"), T.matrix("s")
    ynew = crop(x,
                T.cast(a, theano.config.floatX),
                T.cast(b, theano.config.floatX),
                l, s)
    yold = oldcropper.apply_aaargh(x, l, s, a=a, b=b)

    fnew = theano.function([x, a, b, l, s], ynew)
    fold = theano.function([x, a, b, l, s], yold)

    gnew = theano.function([x, a, b, l, s], T.grad(ynew.mean(), [l, s]))
    gold = theano.function([x, a, b, l, s], T.grad(yold.mean(), [l, s]))

    data = get_data()

    np_ynew = np.asarray(fnew(**data))
    np_yold = np.asarray(fold(**data))

    if not np.allclose(np_ynew, np_yold):
        print "new patch differs from old patch!"

    np_dynew = np.asarray(gnew(**data))
    np_dyold = np.asarray(gold(**data))

    if not np.allclose(np_dynew, np_dyold):
        print "new grad differs from old grad!"

    if False:
        print "verifying grad"
        T.verify_grad(
            lambda l, s: crop(
                T.constant(data["x"]),
                T.constant(data["a"].astype(theano.config.floatX)),
                T.constant(data["b"].astype(theano.config.floatX)),
                l, s),
            [data["l"], data["s"]],
            rng=np.random)
        print "grad verified"

    if True:
        for image, patch, oldpatch, location, scale in itertools.izip(data["x"], np_ynew, np_yold, data["l"], data["s"]):
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(np.rollaxis(image, 0, image.ndim), interpolation="nearest")
            plt.figure()
            plt.imshow(np.rollaxis(patch, 0, patch.ndim), interpolation="nearest")
            plt.title("l %s s %s" % (location, scale))
            #plt.figure()
            #plt.imshow(np.rollaxis(oldpatch, 0, patch.ndim), interpolation="nearest")
            #plt.title("old patch")
            plt.show()
