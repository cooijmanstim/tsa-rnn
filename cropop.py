import itertools
import theano
from theano import Apply
from theano import tensor
from six.moves import StringIO, reduce
from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda import GpuOp
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           gpu_contiguous, host_from_gpu)
from theano.tensor import as_tensor_variable

class TimCropperOp(GpuOp):
    def __init__(self, output_shape, grad=False):
        # NOTE: output_shape specifies spatial dimensions only
        self.output_shape = tuple(output_shape)
        self.grad = grad

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.grad == other.grad and
                self.output_shape == other.output_shape)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.grad) ^ hash(self.output_shape)

    def __str__(self):
        return '%s{%s%s}' % (self.__class__.__name__,
                             self.output_shape,
                             ",grad" if self.grad else "")

    def make_node(self, x, a, b, l, s):
        for input, ndim in ((x, 2 + len(self.output_shape)),
                            (a, 2), (b, 2), (l, 2), (s, 2)):
            if not input.type.ndim == ndim:
                raise TypeError()
        # NOTE: cast a, b to floatX just before making the node
        # NOTE: can handle discontiguous x
        x, a, b, l, s = tuple(map(gpu_contiguous, (x, a, b, l, s)))
        inputs = list(map(as_cuda_ndarray_variable, (x, a, b, l, s)))
        patch_type = inputs[0].type()
        if self.grad:
            # return derivatives wrt l, s. the rest is zero.
            outputs = [patch_type, patch_type]
        else:
            outputs = [patch_type]
        self.output_names = ("ddl dds" if self.grad else "y").split()
        return Apply(self, inputs, outputs)

    # TODO
    # def perform(self, node, input_storage, output_storage):
        #raise NotImplementedError('only C is implemented')

    def grad(self, inp, grads):
        x, a, b, l, s = inp
        y, = grads
        ddl, dds = TimCropper(self.output_shape, grad=True)(*inp)
        return [
            T.zeros_like(x),
            T.zeros_like(a),
            T.zeros_like(b),
            y * ddl,
            y * dds]

    def c_code_cache_version(self):
        return (0)

    def _grid_block_definition(self):
        dims = self.output_shape
        ndim_spatial = len(dims)
        if ndim_spatial == 2:
            return """
            dim3 block(1, 32, 32);
            dim3 grid(ydims[0] * ydims[1],
                      (ydims[2] + block.y - 1) / block.y,
                      (ydims[3] + block.z - 1) / block.z);
            """
        elif ndim_spatial == 3:
            return """
            dim3 block(1, 32, 32);
            dim3 grid(ydims[0] * ydims[1],
                      (ydims[2] + block.y - 1) / block.y,
                      (ydims[3] * ydims[4] + block.z - 1) / block.z);
            """
        else:
            raise NotImplementedError()

    def _thread_pixel_index(self):
        ndim_spatial = len(self.output_shape)
        if ndim_spatial == 2:
            return """
            // assuming C order
            int i0 = blockIdx.x / n1v,
                i1 = blockIdx.x % n1v,
                i2v = blockIdx.y * blockDim.y + threadIdx.y,
                i3v = blockIdx.z * blockDim.z + threadIdx.z;
            """
        elif ndim_spatial == 3:
            return """
            // assuming C order
            int i0 = blockIdx.x / n1v,
                i1 = blockIdx.x % n1v,
                i2v = blockIdx.y * blockDim.y + threadIdx.y;
            int i34v = blockIdx.z * blockDim.z + threadIdx.z;
            int i3v = i34v / n4v,
                i4v = i34v % n4v;
            """
        else:
            raise NotImplementedError()

    def c_code(self, node, nodename, inp, out, sub):
        x, a, b, l, s = inp
        if self.grad:
            ddl, dds = out
        else:
            y, = out
        fail = sub['fail']
        function_name = "TimCropper_%(nodename)s" % locals()
        ndim_spatial = len(self.output_shape)
        ndim_total = 2 + ndim_spatial
        grid_block_definition = self._grid_block_definition()
        strings = []

        # check inputs
        strings.append("""
        if ($x->nd != $ndim_total)
        {
            PyErr_SetString(PyExc_ValueError,
                            "TimCropper: first input must have $ndim_total dimensions");
            $fail;
        }
        """)
        for i, var in enumerate((a, b, l, s)):
            strings.append("""
            if (%(var)s->nd != 2) {
                PyErr_SetString(PyExc_ValueError,
                                "TimCropper: %(i)sth input must have 2 dimensions");
                $fail;
            }
            if (CudaNdarray_HOST_DIMS(%(var)s)[0] != CudaNdarray_HOST_DIMS($x)[0]) {
                PyErr_SetString(PyExc_ValueError,
                                "TimCropper: %(i)sth input must have shape[0] equal to batch size");
                $fail;
            }
            if (CudaNdarray_HOST_DIMS(%(var)s)[1] != $ndim_spatial) {
                PyErr_SetString(PyExc_ValueError,
                                "TimCropper: %(i)sth input must have shape[1] equal to number of spatial dimensions ($ndim_spatial)");
                $fail;
            }
            """ % dict(var=var, i=1 + i))

        # allocate outputs
        strings.append("""
        int ydims[$ndim_total];
        """)
        for i in (0, 1):
            strings.append("ydims[%i] = CudaNdarray_HOST_DIMS($x)[%i];" % (i, i))
        for i, dim in enumerate(self.output_shape):
            strings.append("ydims[2 + %i] = %i;" % (i, dim))
        for var in self.output_names:
            strings.append("maybe_allocate_output($%s, ydims);" % var)

        # launch kernel
        arguments = []
        for var in ["x", self.output_names[0]]:
            arguments.append("CudaNdarray_DEV_DIMS($%s)" % var)
            arguments.append("CudaNdarray_SIZE($%s)" % var)
        for var in ["x"] + self.output_names:
            arguments.append("CudaNdarray_DEV_DATA($%s)" % var)
            arguments.append("CudaNdarray_DEV_STRIDES($%s)" % var)
        arguments.extend("CudaNdarray_DEV_DATA($%s)" % var for var in "abls")
        arguments = ", \n".join(arguments)
        strings.append("""
        {
            $grid_block_definition
            $function_name<<<grid, block>>>($arguments)
            CNDA_THREAD_SYNC;
            cudaError_t err = cudaGetLastError();
            if( cudaSuccess != err)
            {
                PyErr_Format(PyExc_RuntimeError,
                    "Cuda error: %s: %s. (grid: %i x %i;"
                    " block: %i x %i x %i)\\n",
                    "$function_name",
                    cudaGetErrorString(err),
                    grid.x, grid.y,
                    block.x, block.y, block.z);
                $fail;
            }
        }""")
 
        from string import Template
        return Template("\n".join(strings)).substitute(locals())

    def c_support_code_apply(self, node, nodename):
        function_name = "TimCropper_%s" % nodename
        ndim_spatial = len(self.output_shape)
        ndim_total = 2 + ndim_spatial
        import math
        sqrt2pi = math.sqrt(2*math.pi)

        strings = []
        strings.append("""
        #include <stdio.h>
        
        void maybe_allocate_output(CudaNdarray* &y, int* ydims) {
            if ((NULL == y) ||
            """ +
            " || ".join(
                "(CudaNdarray_HOST_DIMS(y)[%i] != ydims[%i])" % (i, i) for i in range(ndim_total))
            + """
                )
            {
                Py_XDECREF(y);
                y = (CudaNdarray*)CudaNdarray_New();
                if ((NULL == y)
                    || CudaNdarray_alloc_contiguous(y, $ndim_total, ydims))
                {
                    Py_XDECREF(y);
                    y = NULL;
                    PyErr_SetString(PyExc_ValueError,
                                    "TimCropper: output allocation failed");
                    $fail;
                }
            }
        }

        __device__ void TimCropper_weight(float &w, int nv, int iv, int iV, float l, float s) {
            const float prior_sigma = 0.5;
            float xV = iV;
            float div = (iv - nv/2);
            float xv = div / s + l;
            float delta = (xV - xv);
            // bound the influence of scale on sigma to avoid the kernels
            // becoming too narrow when zooming in.
            float sigma = prior_sigma / min(s, .9);
            float delta2 = delta * delta, sigma2 = sigma * sigma, s2 = s * s;
            float delta2_sigma2 = delta2 / sigma2;
            float g = exp(-0.5 * delta2_sigma2) / sqrt(2*M_PI) / sigma;
            w = g;
            float dw_dl = g * -delta / sigma2;
            float dw_ds = g * (div * delta / sigma2 - s * (delta2_sigma2 - 1)) / s2;
        }
        """)

        arguments = []
        for side in "Vv":
            arguments.append("const int* n%s" % side)
            arguments.append("const size_t %s_size" % side)
        for var in ["x"] + output_names:
            arguments.append("%sfloat* %s" % ("const " if var == "x" else "", var))
            arguments.append("const int* %s_strides" % var)
        # a, b, l, s are contiguous
        arguments.extend("const float* %s" % var for var in "abls")
        arguments = ", \n".join(arguments)

        strings.append("""
        __global__ void $function_name($arguments) {
        """)
        # compute output pixel index i0, i1, i*v
        strings.append(self._thread_pixel_index())

        # compute output memory locations and initialize to zero
        for var in self.output_names:
            strings.append("size_t %(var)s_index = i0 * %(var)s_strides[0] + i1 * %(var)s_strides[1] + %(rest);"
                           % dict(var=var,
                                  rest=" + ".join("i%iv * %(var)s_strides[%i]"
                                                  % dict(var=var, i=2 + 1)
                                                  for i in range(ndim_spatial))))
            strings.append("assert(0 <= %(var)s_index); assert(%(var)s_index < v_size);")
            strings.append("%(var)s[%(var)s_index] = 0.0f;")

        for i in range(ndim_spatial):
            strings.append("""
            int a%(i2)s = __float2int_rz(a[i0 * $ndim_spatial + %(i)s]),
                b%(i2)s = __float2int_rd(b[i0 * $ndim_spatial + %(i)s]);
            float l%(i2)s = l[i0 * $ndim_spatial + %(i)s],
                  s%(i2)s = s[i0 * $ndim_spatial + %(i)s];
            assert(0 <= a%(i2)s); assert(a%(i2)s <= b%(i2)s); assert(b%(i2)s <= n%(i2)sV);
            float w%(i2)s = 0;
            """ % dict(i=i, i2=2 + i))

        # loop over input pixel indices i{2, 3, ...}V
        for i in range(ndim_spatial):
            strings.append("""
            for (int i%(i)sV = a%(i)s; i%(i)sV < b%(i)s; ++i%(i)sV) {
                TimCropper_weight(w%(i)s, n%(i)sv, i%(i)sv, i%(i)sV, l%(i)s, s%(i)s);
            """ % dict(i=2 + i))

        # compute input memory location
        strings.append("size_t x_index = i0 * x_strides[0] + i1 * x_strides[1] + %s;"
                        % " + ".join("i%iv * x_strides[%i]" % dict(i=2 + 1) for i in range(ndim_spatial)))
        strings.append("assert(0 <= x_index); assert(x_index < V_size);")

        # compute weight and contribution
        strings.append("float w = %s;" % " * ".join("w%i" % (2 + i) for i in range(ndim_spatial)))
        for var in self.output_names:
            strings.append("%s[%s_index] += w * x[x_index];" % (var, var))

        strings.extend("}" * (ndim_spatial + 1))

        from string import Template
        return Template("\n".join(strings)).substitute(locals())

#dv/dl[0]: v[i0, i1, izv, iyv, ixv] += dw/dl *  yw *  xw * V[i0, i1, izV, iyV, ixV]
#dv/dl[1]: v[i0, i1, izv, iyv, ixv] +=  zw * dw/dl *  xw * V[i0, i1, izV, iyV, ixV]
#dv/dl[2]: v[i0, i1, izv, iyv, ixv] +=  zw *  yw * dw/dl * V[i0, i1, izV, iyV, ixV]

#similarly ../ds

if __name__ == "__main__":
    import theano, theano.tensor as T
    output_shape = (64, 64)
    ndim_spatial = len(output_shape)
    crop = TimCropperOp(output_shape)
    x = T.tensor4("x")
    a, b = T.imatrix("a"), T.imatrix("b")
    l, s = T.matrix("l"), T.matrix("s")
    y = crop(x,
             T.cast(a, theano.config.floatX),
             T.cast(b, theano.config.floatX),
             l, s)
    f = theano.function([x, a, b, l, s], host_from_gpu(y))

    def load_frame(bytes):
        from PIL import Image
        image = Image.open(StringIO(bytes.tostring()))
        image = (np.array(image.getdata(), dtype=np.float32)
                 .reshape((image.size[1], image.size[0], 3)))
        image /= 255.0
        return image

    import numpy as np
    import h5py
    with h5py.File("/data/lisatmp3/ballasn/datasets/UCF101/jpeg_data.hdf5", "r") as ucf101:
        images = ucf101["images"][[128, 256, 512, 1024, 2048, 4096, 8192]]
        np_x = np.asarray(list(map(load_frame, images)))
    np_x = np.rollaxis(np_x, np_x.ndim - 1, 1)

    input_shape = np_x.shape[2:]
    print input_shape
    np_l = np.random.rand(np_x.shape[0], ndim_spatial) * input_shape
    np_s = 0.1 + 3 * np.random.rand(np_x.shape[0], ndim_spatial)
    np_a, np_b = (0 * np_l).astype(np.int32), (0 * np_l + input_shape).astype(np.int32)
    np_l, np_s = np_l.astype(theano.config.floatX), np_s.astype(theano.config.floatX)

    np_y = np.asarray(f(np_x, np_a, np_b, np_l, np_s))

    for image, patch, location, scale in itertools.izip(np_x, np_y, np_l, np_s):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(np.rollaxis(image, 0, image.ndim), interpolation="nearest")
        plt.figure()
        plt.imshow(np.rollaxis(patch, 0, patch.ndim), interpolation="nearest")
        plt.title("l %s s %s" % (location, scale))
        plt.show()
