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

class TimCropperGradOp(GpuOp):
    def __init__(self, patch_shape):
        # NOTE: patch_shape specifies spatial dimensions only
        self.patch_shape = tuple(patch_shape)

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.patch_shape == other.patch_shape)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.patch_shape)

    def __str__(self):
        return '%s{%s}' % (self.__class__.__name__, self.patch_shape)

    def make_node(self, dCdy, x, a, b, l, s):
        for input, ndim in ((dCdy, 2 + len(self.patch_shape)),
                            (x, 2 + len(self.patch_shape)),
                            (a, 2), (b, 2), (l, 2), (s, 2)):
            if not input.type.ndim == ndim:
                raise TypeError()
        # NOTE: can handle discontiguous dCdy, x
        dCdy, x, a, b, l, s = tuple(map(gpu_contiguous, (dCdy, x, a, b, l, s)))
        inputs = list(map(as_cuda_ndarray_variable, (dCdy, x, a, b, l, s)))

        # we could return the much smaller dCdl, dCds but that
        # gives us very little room to parallelize (e.g. with batch
        # size 100 and 3 spatial dimensions we have only 600
        # independently computable output elements).
        output_type = CudaNdarrayType(
            broadcastable=list(inputs[0].type.broadcastable) + [False],
            dtype=inputs[0].type.dtype)
        dydl = output_type()
        dyds = output_type()
        return Apply(self, inputs, [dydl, dyds])

    # TODO
    # def perform(self, node, input_storage, output_storage):
        #raise NotImplementedError('only C is implemented')

    def c_code_cache_version(self):
        return (0)

    def _grid_block_definition(self):
        dims = self.patch_shape
        ndim_spatial = len(dims)
        if ndim_spatial == 2:
            return """
            dim3 block(1, 32, 32);
            dim3 grid(outdims[0] * outdims[1],
                      (outdims[2] + block.y - 1) / block.y,
                      (outdims[3] + block.z - 1) / block.z);
            """
        elif ndim_spatial == 4:
            return """
            dim3 block(1, 32, 32);
            dim3 grid(outdims[0] * outdims[1],
                      (outdims[2] + block.y - 1) / block.y,
                      (outdims[3] * outdims[4] + block.z - 1) / block.z);
            """
        else:
            raise NotImplementedError()

    def _thread_pixel_index(self):
        ndim_spatial = len(self.patch_shape)
        if ndim_spatial == 2:
            return """
            // assuming C order
            int i0 = blockIdx.x / dydl_dims[1];
            int i1 = blockIdx.x % dydl_dims[1];
            int i2v = blockIdx.y * blockDim.y + threadIdx.y,
                i3v = blockIdx.z * blockDim.z + threadIdx.z;
            // do nothing if out of bounds
            if (i0 >= dydl_dims[0] || i1 >= dydl_dims[1] || i2v >= dydl_dims[2] || i3v >= dydl_dims[3])
                return;
            """
        elif ndim_spatial == 3:
            return """
            // assuming C order
            int i0 = blockIdx.x / dydl_dims[1];
            int i1 = blockIdx.x % dydl_dims[1];
            int i2v = blockIdx.y * blockDim.y + threadIdx.y;
            int i34v = blockIdx.z * blockDim.z + threadIdx.z;
            int i3v = i34v / dydl_dims[4],
                i4v = i34v % dydl_dims[4];
            // do nothing if out of bounds
            if (i0 >= dydl_dims[0] || i1 >= dydl_dims[1] || i2v >= dydl_dims[2] || i3v >= dydl_dims[3] || i4v >= dydl_dims[4])
                return;
            """
        else:
            raise NotImplementedError()

    def c_code(self, node, nodename, inp, out, sub):
        dCdy, x, a, b, l, s = inp
        dydl, dyds = out
        fail = sub['fail']
        function_name = "TimCropperGrad_%(nodename)s" % locals()
        ndim_spatial = len(self.patch_shape)
        ndim_total = 2 + ndim_spatial
        grid_block_definition = self._grid_block_definition()
        strings = []

        # check inputs
        for i, var in enumerate([dCdy, x]):
            strings.append("""
            if (%(var)s->nd != $ndim_total)
            {
                PyErr_SetString(PyExc_ValueError,
                                "TimCropper: %(i)sth input must have $ndim_total dimensions");
                $fail;
            }
            """ % dict(var=var, i=i))
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
            """ % dict(var=var, i=2 + i))

        # allocate outputs
        strings.append("""
        int outdims[$ndim_total + 1];
        outdims[$ndim_total] = $ndim_spatial;
        """)
        for i in range(ndim_total):
            strings.append("outdims[%i] = CudaNdarray_HOST_DIMS($dCdy)[%i];" % (i, i));
        for var in "ls":
            strings.append(("""
            if ((NULL == %(dname)s) || """ + " || ".join(
                "(CudaNdarray_HOST_DIMS(%%(dname)s)[%i] != outdims[%i])" % (i, i) for i in range(ndim_total + 1))
            + """)
            {
                Py_XDECREF(%(dname)s);
                %(dname)s = (CudaNdarray*)CudaNdarray_New();
                if ((NULL == %(dname)s)
                    || CudaNdarray_alloc_contiguous(%(dname)s, $ndim_total + 1, outdims))
                {
                    Py_XDECREF(%(dname)s);
                    %(dname)s = NULL;
                    PyErr_SetString(PyExc_ValueError,
                                    "TimCropperGrad: allocation of output %(dlabel)s failed");
                    $fail;
                }
            }
            """) % dict(name=locals()[var], dname=locals()["dyd" + var], dlabel="dyd" + var))

        # launch kernel
        arguments = []
        for var in "dydl dyds dCdy x".split():
            arguments.append("CudaNdarray_SIZE($%s)" % var)
            arguments.append("CudaNdarray_DEV_DATA($%s)" % var)
            arguments.append("CudaNdarray_DEV_DIMS($%s)" % var)
            arguments.append("CudaNdarray_DEV_STRIDES($%s)" % var)
        arguments.extend("CudaNdarray_DEV_DATA($%s)" % var for var in "abls")
        strings.append("""
        {
            printf("enter grad op\\n");
            $grid_block_definition
            $function_name<<<grid, block>>>(""" + ", \n".join(arguments) + """)
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
            printf("exit grad op\\n");
        }""")

        from string import Template
        return Template("\n".join(strings)).substitute(locals())

    def c_support_code_apply(self, node, nodename):
        function_name = "TimCropperGrad_%s" % nodename
        ndim_spatial = len(self.patch_shape)
        ndim_total = 2 + ndim_spatial
        import math
        sqrt2pi = math.sqrt(2*math.pi)

        strings = []
        strings.append("""
        #include <stdio.h>

        __device__ void TimCropperGrad_weight(float &w, float &dw_dl, float &dw_ds, int nv, int iv, int iV, float l, float s) {
            const float prior_sigma = 0.5;
            float xV = iV;
            float div = (iv - nv/2);
            float xv = div / s + l;
            float delta = (xV - xv);
            // bound the influence of scale on sigma to avoid the kernels
            // becoming too narrow when zooming in.
            //const float s_bound = .9;
            //float sigma = prior_sigma / min(s, s_bound);
            float sigma = prior_sigma / s;
            float delta2 = delta * delta, sigma2 = sigma * sigma, s2 = s * s;
            float delta2_sigma2 = delta2 / sigma2;
            float g = exp(-0.5 * delta2_sigma2) / sqrt(2*M_PI) / sigma;
            w = g;
            dw_dl = g * -delta / sigma2;
            // FIXME: rederive for bounded s
            dw_ds = g * (div * delta / sigma2 - s * (delta2_sigma2 - 1)) / s2;
        }
        """)

        arguments = []
        for var in "dydl dyds dCdy x".split():
            arguments.append("const size_t %s_size" % var)
            arguments.append("%sfloat* %s" % ("const " if var in "x dCdy".split() else "", var))
            arguments.append("const int* %s_dims" % var)
            arguments.append("const int* %s_strides" % var)
        # a, b, l, s are contiguous
        arguments.extend("const float* %s" % var for var in "abls")
        arguments = ", \n".join(arguments)

        strings.append("""
        __global__ void $function_name($arguments) {
        """)
        # compute output index i0, i1, i*v
        strings.append(self._thread_pixel_index())

        # compute output memory locations and initialize to zero
        for var in "dydl dyds".split():
            strings.append("size_t %(var)s_indexish = i0 * %(var)s_strides[0] + i1 * %(var)s_strides[1] + %(rest)s;"
                           % dict(var=var,
                                  rest=" + ".join("i%(i)sv * %(var)s_strides[%(i)s]"
                                                  % dict(var=var, i=2 + i) for i in range(ndim_spatial))))
            for i in range(ndim_spatial):
                strings.append("""
                size_t %(var)s_index%(i)s = %(var)s_indexish + %(i)s * %(var)s_strides[$ndim_total];
                assert(%(var)s_index%(i)s < %(var)s_size);
                %(var)s[%(var)s_index%(i)s] = 0.0f;
                """ % dict(var=var, i=i))

        for i in range(ndim_spatial):
            strings.append("""
            const int a%(i2)s = a[i0 * $ndim_spatial + %(i)s],
                      b%(i2)s = b[i0 * $ndim_spatial + %(i)s];
            const float l%(i2)s = l[i0 * $ndim_spatial + %(i)s],
                        s%(i2)s = s[i0 * $ndim_spatial + %(i)s];
            assert(0 <= a%(i2)s); assert(a%(i2)s <= b%(i2)s); assert(b%(i2)s <= x_dims[%(i2)s]);
            float w%(i2)s = 0, dw_dl%(i2)s = 0, dw_ds%(i2)s = 0;
            """ % dict(i=i, i2=2 + i))

        # loop over input pixel indices i{2, 3, ...}V
        # TODO: compute x_index progressively; x_index += relevant_stride
        for i, patch_dim in enumerate(self.patch_shape):
            strings.append("""
            for (int i%(i)sV = a%(i)s; i%(i)sV < b%(i)s; ++i%(i)sV) {
                TimCropperGrad_weight(w%(i)s, dw_dl%(i)s, dw_ds%(i)s, %(patch_dim)s, i%(i)sv, i%(i)sV, l%(i)s, s%(i)s);
            """ % dict(i=2 + i, patch_dim=patch_dim))

        # compute input memory location
        strings.append("size_t x_index = i0 * x_strides[0] + i1 * x_strides[1] + %s;"
                        % " + ".join("i%(i)sV * x_strides[%(i)s]"
                                     % dict(i=2 + i) for i in range(ndim_spatial)))
        strings.append("assert(x_index < x_size);")

        # compute contribution
        # for dy/dl[0], weight = dw_dl0 * w1 * w2
        # for dy/dl[1], weight = w0 * dw_dl1 * w2
        # etc. and similarly dy/ds
        for var in "ls":
            for j in range(ndim_spatial):
                weight = " * ".join(
                    ("dw_d%s%%i" % var if i == j else "w%i")
                    % (2 + i) for i in range(ndim_spatial))
                strings.append("""
                %(dvar)s[%(dvar)s_index%(j)s] += %(weight)s * x[x_index];
                """ % dict(dvar="dyd" + var, weight=weight, j=j))

        strings.extend("}" * (ndim_spatial + 1))

        from string import Template
        return Template("\n".join(strings)).substitute(locals())
