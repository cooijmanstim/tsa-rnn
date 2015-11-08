import itertools
import numpy as np
import theano
import theano.tensor as T
import h5py
from StringIO import StringIO
from PIL import Image
import matplotlib.pyplot as plt

from op import TimCropperOp
from grad import TimCropperGradOp
from oldcropper import LocallySoftRectangularCropper, Gaussian

def load_frame(bytes):
    image = Image.open(StringIO(bytes.tostring()))
    image = (np.array(image.getdata(), dtype=np.float32)
                .reshape((image.size[1], image.size[0], 3)))
    image /= 255.0
    return image

np.random.seed(1)
ucf101 = h5py.File("/data/lisatmp3/ballasn/datasets/UCF101/jpeg_data.hdf5", "r")

def get_data():
    n = 10
    ndim_spatial = len(patch_shape)

    sample = np.random.choice(len(ucf101["images"]), size=(n,))
    images = ucf101["images"][list(sorted(sample))]
    x = np.asarray(list(map(load_frame, images)))
    x = np.rollaxis(x, x.ndim - 1, 1)

    input_shape = x.shape[2:]
    l = np.random.rand(x.shape[0], ndim_spatial) * input_shape
    s = 0.25 + 2 * np.random.rand(x.shape[0], ndim_spatial)
    l, s = l.astype(theano.config.floatX), s.astype(theano.config.floatX)
    return dict(x=x, l=l, s=s)

if __name__ == "__main__":
    import theano, theano.tensor as T
    patch_shape = (32, 32)
    ndim_spatial = len(patch_shape)
    crop = TimCropperOp(patch_shape)
    oldcropper = LocallySoftRectangularCropper(
        patch_shape, Gaussian(),
        dict(cutoff=3, batched_window=True, scan=False))
    newcropper = LocallySoftRectangularCropper(
        patch_shape, Gaussian(),
        dict(cutoff=3, batched_window=False, scan=False))

    x = T.tensor4("x")
    l, s = T.matrix("l"), T.matrix("s")
    x_shape = 0 * l + (240, 320)
    a, b = oldcropper.compute_hard_windows(x_shape, l, s)
    ynew, _ = newcropper.apply(x, x_shape, l, s)
    yold, _ = oldcropper.apply(x, x_shape, l, s)

    fnew = theano.function([x, l, s], ynew)
    fold = theano.function([x, l, s], yold)

    # random projection
    p = T.tensor4("p");
    gnew = theano.function([x, l, s, p], T.grad((p * ynew).sum(), [l, s]))
    gold = theano.function([x, l, s, p], T.grad((p * yold).sum(), [l, s]))

    while True:
        data = get_data()

        np_ynew = np.asarray(fnew(**data))
        np_yold = np.asarray(fold(**data))

        if not np.allclose(np_ynew, np_yold):
            print "new patch differs from old patch! squared error: %s" % ((np_ynew - np_yold)**2).sum()
            import pdb; pdb.set_trace()
        else:
            print "patches match"

        data["p"] = np.random.random(np_ynew.shape).astype(theano.config.floatX)
        data["p"] /= (data["p"]**2).sum()
        np_dynew = np.asarray(gnew(**data))
        np_dyold = np.asarray(gold(**data))

        if not np.allclose(np_dynew, np_dyold):
            print "new grad differs from old grad! squared error: %s" % ((np_dynew - np_dyold)**2).sum()
            def show():
                plt.figure()
                plt.hist(np_dyold[0].ravel())
                plt.title("np_dyold[0]")
                plt.matshow(np_dyold[0]);
                plt.title("np_dyold[0]")
                plt.figure()
                plt.hist(np_dyold[1].ravel())
                plt.title("np_dyold[1]")
                plt.matshow(np_dyold[1]);
                plt.title("np_dyold[1]")
                plt.figure()
                plt.hist(np_dynew[0].ravel())
                plt.title("np_dynew[0]")
                plt.matshow(np_dynew[0]);
                plt.title("np_dynew[0]")
                plt.figure()
                plt.hist(np_dynew[1].ravel())
                plt.title("np_dynew[1]")
                plt.matshow(np_dynew[1]);
                plt.title("np_dynew[1]")
                plt.show()
            import pdb; pdb.set_trace()
        else:
            print "grads match"

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

    if False:
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
