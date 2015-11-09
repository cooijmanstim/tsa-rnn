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

import common

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

    def c_code(self, node, nodename, inp, out, sub):
        dCdy, x, a, b, l, s = inp
        dydl, dyds = out
        fail = sub['fail']
        function_name = "TimCropperGrad_%s" % nodename
        ndim_spatial = len(self.patch_shape)
        ndim_total = 2 + ndim_spatial
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
        gridblock = common.gridblock(ndim_spatial, "CudaNdarray_HOST_DIMS(%s)" % dCdy)
        strings.append("""
        {
            printf("enter grad op\\n");
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
            printf("exit grad op\\n");
        }""")

        from string import Template
        return Template("\n".join(strings)).substitute(locals())

    def c_support_code_apply(self, node, nodename):
        function_name = "TimCropperGrad_%s" % nodename
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
            name=weight_function_name,
            grad=True))

        arguments = []
        for var in "dydl dyds dCdy x".split():
            arguments.append("const size_t %s_size" % var)
            arguments.append("%sfloat* %s" % ("const " if var in "x dCdy".split() else "", var))
            arguments.append("const int* %s_dims" % var)
            arguments.append("const int* %s_strides" % var)
        # a, b, l, s are contiguous
        arguments.extend("const float* %s" % var for var in "abls")
        arguments = ", \n".join(arguments)

        threadindex = common.threadindex(ndim_spatial, "dCdy_dims")

        # generate code that does output initialization and computation. this is
        # best kept together in the python code to reduce confusion.
        initializations = []
        computations = []
        for var in "ls":
            dy = "dyd" + var
            dw = "dwd" + var

            # patch pixel index; we need to compute dydl[0]..dydl[ndim_spatial]
            # for each pixel, so one more subscript is needed to fully qualify
            # the output element.
            output_indexish = (
                "i0 * %(dy)s_strides[0] + i1 * %(dy)s_strides[1] + %(rest)s"
                % dict(dy=dy, rest=" + ".join(
                    "i%(i)sv * %(dy)s_strides[%(i)s]"
                    % dict(dy=dy, i=2 + i) for i in range(ndim_spatial))))

            for j in range(ndim_spatial):
                index = ("%(output_indexish)s + %(j)s * %(dy)s_strides[$ndim_total]"
                        % dict(output_indexish=output_indexish, dy=dy, j=j))

                # initialize to zero
                initializations.append("""
                assert(%(index)s < %(dy)s_size);
                %(dy)s[%(index)s] = 0.0f;
                """ % dict(dy=dy, index=index))

                # compute contribution
                # for dy/dl[0], weight = dw/dl0 * w1 * w2
                # for dy/dl[1], weight = w0 * dw/dl1 * w2
                # for dy/dl[2], weight = w0 * w1 * dw/dl2
                # etc. and similarly dy/ds
                weight = " * ".join(
                    (dw if i == j else "w") + str(2 + i)
                    for i in range(ndim_spatial))
                computations.append("""
                assert(%(index)s < %(dy)s_size);
                %(dy)s[%(index)s] += %(weight)s * x[x_index];
                """ % dict(dy=dy, weight=weight, index=index))

        strings.append("""
        __global__ void $function_name($arguments) {
            $threadindex
        """)
        strings.append("\n".join(initializations))

        for i in range(ndim_spatial):
            strings.append("""
            const int a%(i2)s = a[i0 * $ndim_spatial + %(i)s],
                      b%(i2)s = b[i0 * $ndim_spatial + %(i)s];
            const float l%(i2)s = l[i0 * $ndim_spatial + %(i)s],
                        s%(i2)s = s[i0 * $ndim_spatial + %(i)s];
            assert(0 <= a%(i2)s); assert(a%(i2)s <= b%(i2)s); assert(b%(i2)s <= x_dims[%(i2)s]);
            float w%(i2)s = 0, dwdl%(i2)s = 0, dwds%(i2)s = 0;
            """ % dict(i=i, i2=2 + i))

        # loop over input pixel indices i{2, 3, ...}V
        for i, patch_dim in enumerate(self.patch_shape):
            strings.append("""
            for (int i%(i)sV = a%(i)s; i%(i)sV < b%(i)s; ++i%(i)sV) {
                $weight_function_name(w%(i)s, dwdl%(i)s, dwds%(i)s, %(patch_dim)s, i%(i)sv, i%(i)sV, l%(i)s, s%(i)s);
            """ % dict(i=2 + i, patch_dim=patch_dim))
        # compute input memory location
        strings.append("size_t x_index = i0 * x_strides[0] + i1 * x_strides[1] + %s;"
                        % " + ".join("i%(i)sV * x_strides[%(i)s]"
                                     % dict(i=2 + i) for i in range(ndim_spatial)))
        strings.append("assert(x_index < x_size);")
        strings.append("\n".join(computations))
        strings.extend("}" * (ndim_spatial + 1))

        from string import Template
        return Template("\n".join(strings)).substitute(locals())
