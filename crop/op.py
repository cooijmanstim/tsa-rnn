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

from grad import TimCropperGradOp
import common

class TimCropperOp(GpuOp):
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

    def make_node(self, x, a, b, l, s):
        for input, ndim in ((x, 2 + len(self.patch_shape)),
                            (a, 2), (b, 2), (l, 2), (s, 2)):
            if not input.type.ndim == ndim:
                raise TypeError()
        # NOTE: cast a, b to floatX just before making the node
        # NOTE: can handle discontiguous x
        x, a, b, l, s = tuple(map(gpu_contiguous, (x, a, b, l, s)))
        inputs = list(map(as_cuda_ndarray_variable, (x, a, b, l, s)))
        return Apply(self, inputs, [inputs[0].type()])

    # TODO
    # def perform(self, node, input_storage, output_storage):
        #raise NotImplementedError('only C is implemented')

    def grad(self, inp, grads):
        x, a, b, l, s = inp
        dCdy, = grads
        dydl, dyds = TimCropperGradOp(self.patch_shape)(dCdy, x, a, b, l, s)

        # for some reason we are given dCdy with an extra broadcastable first dimension?
        dCdy = dCdy[0]

        # a, b are not differentiable, and we don't care about backpropping through x for now
        rval = [theano.gradient.disconnected_type() for i in range(3)]
        # compute dCdl, dCds by summing the product over channels and spatial locations
        # (a simple case for which we don't need batched_tensordot)
        rval.extend(
            (tensor.shape_padright(dCdy) * dy).sum(range(1, dCdy.ndim + 1))
            for dy in (dydl, dyds))
        return rval

    def c_code_cache_version(self):
        return (0)

    def c_code(self, node, nodename, inp, out, sub):
        x, a, b, l, s = inp
        y, = out
        fail = sub['fail']
        function_name = "TimCropper_%(nodename)s" % locals()
        ndim_spatial = len(self.patch_shape)
        ndim_total = 2 + ndim_spatial
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

        # allocate output
        strings.append("""
        int ydims[$ndim_total];
        """)
        for i in (0, 1):
            strings.append("ydims[%i] = CudaNdarray_HOST_DIMS($x)[%i];" % (i, i))
        for i, dim in enumerate(self.patch_shape):
            strings.append("ydims[2 + %i] = %i;" % (i, dim))
        strings.append("""
        if ((NULL == $y) || """ + " || ".join(
            "(CudaNdarray_HOST_DIMS($y)[%i] != ydims[%i])" % (i, i) for i in range(ndim_total))
        + """)
        {
            Py_XDECREF($y);
            $y = (CudaNdarray*)CudaNdarray_New();
            if ((NULL == $y)
                || CudaNdarray_alloc_contiguous($y, $ndim_total, ydims))
            {
                Py_XDECREF($y);
                $y = NULL;
                PyErr_SetString(PyExc_ValueError,
                                "TimCropper: output allocation failed");
                $fail;
            }
        }
        """)

        # launch kernel
        arguments = []
        for var in "x y".split():
            arguments.append("CudaNdarray_SIZE($%s)" % var)
            arguments.append("CudaNdarray_DEV_DATA($%s)" % var)
            arguments.append("CudaNdarray_DEV_DIMS($%s)" % var)
            arguments.append("CudaNdarray_DEV_STRIDES($%s)" % var)
        arguments.extend("CudaNdarray_DEV_DATA($%s)" % var for var in "abls")
        gridblock = common.gridblock(ndim_spatial, "ydims")
        strings.append("""
        {
            printf("enter op\\n");
            $gridblock
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
            printf("exit op\\n");
        }""")

        from string import Template
        return Template("\n".join(strings)).substitute(locals())

    def c_support_code_apply(self, node, nodename):
        function_name = "TimCropper_%s" % nodename
        weight_function_name = "%s_weight" % function_name

        ndim_spatial = len(self.patch_shape)
        ndim_total = 2 + ndim_spatial
        import math
        sqrt2pi = math.sqrt(2*math.pi)

        strings = []
        strings.append("""
        #include <stdio.h>
        """)
        strings.append(common.weightfunction(
            name=weight_function_name, grad=False))

        arguments = []
        for var in "x y".split():
            arguments.append("const size_t %s_size" % var)
            arguments.append("%sfloat* %s" % ("const " if var == "x" else "", var))
            arguments.append("const int* %s_dims" % var)
            arguments.append("const int* %s_strides" % var)
        # a, b, l, s are contiguous
        arguments.extend("const float* %s" % var for var in "abls")
        arguments = ", \n".join(arguments)

        threadindex = common.threadindex(ndim_spatial, "y_dims")

        strings.append("""
        __global__ void $function_name($arguments) {
            $threadindex
        """)
        # compute output memory locations and initialize to zero
        strings.append("size_t y_index = i0 * y_strides[0] + i1 * y_strides[1] + %s;"
                       % " + ".join("i%(i)sv * y_strides[%(i)s]"
                                    % dict(i=2 + i) for i in range(ndim_spatial)))
        strings.append("assert(y_index < y_size);")
#        strings.append("""
#            printf("block %3i %3i %3i thread %3i %3i %3i pixel %3i %3i %3i %3i y_index %3i y_strides %3i %3i %3i %3i\\n",
#                   blockIdx.x, blockIdx.y, blockIdx.z,
#                   threadIdx.x, threadIdx.y, threadIdx.z,
#                   i0, i1, i2v, i3v, y_index,
#                   y_strides[0],
#                   y_strides[1],
#                   y_strides[2],
#                   y_strides[3]
#                   );
#        """)
        strings.append("y[y_index] = 0.0f;")

        for i in range(ndim_spatial):
            strings.append("""
            int a%(i2)s = a[i0 * $ndim_spatial + %(i)s],
                b%(i2)s = b[i0 * $ndim_spatial + %(i)s];
            float l%(i2)s = l[i0 * $ndim_spatial + %(i)s],
                  s%(i2)s = s[i0 * $ndim_spatial + %(i)s];
            assert(0 <= a%(i2)s); assert(a%(i2)s <= b%(i2)s); assert(b%(i2)s <= x_dims[%(i2)s]);
            float w%(i2)s = 0;
            """ % dict(i=i, i2=2 + i))

        # loop over input pixel indices i{2, 3, ...}V
        for i, patch_dim in enumerate(self.patch_shape):
            strings.append("""
            for (int i%(i)sV = a%(i)s; i%(i)sV < b%(i)s; ++i%(i)sV) {
                $weight_function_name(w%(i)s, %(patch_dim)s, i%(i)sv, i%(i)sV, l%(i)s, s%(i)s);
            """ % dict(i=2 + i, patch_dim=patch_dim))
        # compute input memory location
        strings.append("size_t x_index = i0 * x_strides[0] + i1 * x_strides[1] + %s;"
                       % " + ".join("i%(i)sV * x_strides[%(i)s]"
                                    % dict(i=2 + i) for i in range(ndim_spatial)))
        strings.append("assert(x_index < x_size);")

        # compute contribution
        weight = " * ".join("w%i" % (2 + i) for i in range(ndim_spatial))
        strings.append("y[y_index] += %s * x[x_index];" % weight)

        strings.extend("}" * (ndim_spatial + 1))

        from string import Template
        return Template("\n".join(strings)).substitute(locals())
