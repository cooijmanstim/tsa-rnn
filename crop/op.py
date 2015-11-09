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

    # NOTE: cast a, b to floatX just before making the node
    def make_node(self, x, a, b, l, s):
        for input, ndim in ((x, 2 + len(self.patch_shape)),
                            (a, 2), (b, 2), (l, 2), (s, 2)):
            if not input.type.ndim == ndim:
                raise TypeError()
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

        # due to separability we need to compute weights only for patch
        # row/image row and patch col/image col pairs, instead of for
        # the full cartesian product of patch pixel/image pixel pairs.
        # we precompute the weights in a separate pass.
        weightpass_call = common.weightpass_call(
            nodename, patch_shape=self.patch_shape, V=x, l=l, s=s, fail=fail, grad=False)

        # launch kernel
        W = "W" # so call_arguments knows its name
        arguments = common.call_arguments("x y W".split())
        gridblock = common.gridblock(ndim_spatial, "ydims")
        strings.append("""
        {
            $weightpass_call

            $gridblock
            $function_name<<<grid, block>>>(""" + arguments + """)
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

            // free weight storage
            Py_XDECREF(W);
            W = NULL;
        }""")

        from string import Template
        return Template("\n".join(strings)).substitute(locals())

    def c_support_code_apply(self, node, nodename):
        function_name = "TimCropper_%s" % nodename

        ndim_spatial = len(self.patch_shape)
        ndim_total = 2 + ndim_spatial
        import math
        sqrt2pi = math.sqrt(2*math.pi)

        arguments = common.defn_arguments("x y W".split())
        threadindex = common.threadindex(ndim_spatial, "y_dims")
        weightpass_defn = common.weightpass_defn(nodename, self.patch_shape, grad=False)

        strings = []
        strings.append("""
        #include <stdio.h>
        #include <sys/time.h>

        $weightpass_defn

        __global__ void $function_name($arguments) {
            $threadindex
        """)

        for i in range(ndim_spatial):
            strings.append("""
            int a%(i2)s = a[i0 * $ndim_spatial + %(i)s],
                b%(i2)s = b[i0 * $ndim_spatial + %(i)s];
            assert(0 <= a%(i2)s); assert(a%(i2)s <= b%(i2)s); assert(b%(i2)s <= x_dims[%(i2)s]);
            """ % dict(i=i, i2=2 + i))

        # loop over input pixel indices i{2, 3, ...}V
        # compute start of input memory
        # NOTE: assumes x, W contiguous
        # FIXME: the a, b windows are not contiguous, so this is all wrong. must adjust for a, b in both x and W indices
        strings.append("const float* x_pointer = x + i0 * x_strides[0] + i1 * x_strides[1];")
        strings.append("const float* W_pointer = W + i0 * W_strides[0] + i1 * W_strides[1];")
        strings.append("float result = 0.0f;")

        for i, patch_dim in enumerate(self.patch_shape):
            strings.append("""
            x_pointer += a%(i)s;
            for (int i%(i)sV = a%(i)s; i%(i)sV < b%(i)s; ++i%(i)sV) {
            """ % dict(i=2 + i, patch_dim=patch_dim))

        weight = " * ".join("*(W + %i)" % j for j in range(ndim_spatial))
        strings.append("result += $weight * (*x_pointer);")

        strings.append("++x_pointer;")
        strings.append("W_pointer += $ndim_spatial;")

        strings.extend("}" * ndim_spatial)

        # store result at output memory location
        strings.append("size_t y_index = i0 * y_strides[0] + i1 * y_strides[1] + %s;"
                       % " + ".join("i%(i)sv * y_strides[%(i)s]"
                                    % dict(i=2 + i) for i in range(ndim_spatial)))
        strings.append("assert(y_index < y_size);")
        strings.append("y[y_index] = result;")

        strings.append("}")
        from string import Template
        return Template("\n".join(strings)).substitute(locals())
