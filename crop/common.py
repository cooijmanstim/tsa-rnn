import math
from string import Template

def gridblock(ndim_spatial, patch_dims):
    # NOTE: make sure block.x is 1 so we don't parallelize over the batch,
    # otherwise the threads' loops break at different points (because
    # different shape windows)
    strings = []
    if ndim_spatial == 2:
        strings.append("""
        dim3 block(1, 32, 32);
        dim3 grid(($patch_dims[0] * $patch_dims[1] + block.x - 1) / block.x,
                  ($patch_dims[2] + block.y - 1) / block.y,
                  ($patch_dims[3] + block.z - 1) / block.z);
        """)
    elif ndim_spatial == 3:
        strings.append("""
        dim3 block(1, 32, 32);
        dim3 grid(($patch_dims[0] * $patch_dims[1] + block.x - 1) / block.x,
                  ($patch_dims[2] + block.y - 1) / block.y,
                  ($patch_dims[3] * $patch_dims[4] + block.z - 1) / block.z);
        """)
    else:
        raise NotImplementedError()
    return Template("\n".join(strings)).substitute(locals())

def threadindex(ndim_spatial, patch_dims):
    strings = []
    if ndim_spatial == 2:
        strings.append("""
        // assuming C order
        const int i0 = blockIdx.x / $patch_dims[1],
                  i1 = blockIdx.x % $patch_dims[1],
                  i2v = blockIdx.y * blockDim.y + threadIdx.y,
                  i3v = blockIdx.z * blockDim.z + threadIdx.z;
        """)
    elif ndim_spatial == 3:
        strings.append("""
        // assuming C order
        const int i0 = blockIdx.x / $patch_dims[1],
                  i1 = blockIdx.x % $patch_dims[1],
                  i2v = blockIdx.y * blockDim.y + threadIdx.y;
        const int i34v = blockIdx.z * blockDim.z + threadIdx.z;
        const int i3v = i34v / $patch_dims[4],
                  i4v = i34v % $patch_dims[4];
        """)
    else:
        raise NotImplementedError()
    strings.append("""
    // do nothing if out of bounds
    if (i0 >= $patch_dims[0] || i1 >= $patch_dims[1] || %s) { return; }
    """ % " || ".join("i%(i)sv >= $patch_dims[%(i)s]"
                      % dict(i=2 + i) for i in range(ndim_spatial)))
    return Template("\n".join(strings)).substitute(locals())

def weightfunction(name, grad=False):
    sqrt2pi = math.sqrt(math.pi)
    grad_arguments = "float &dwdl, float &dwds," if grad else ""
    grad_assignments = """
        dwdl = w * -delta / sigma / sigma;
        dwds = w * (
            // through effect on delta
            (iv - nv/2) * delta / sigma -
            // through effect on sigma
            (s < 1) * prior_sigma * (delta * delta / sigma /sigma - 1)
        ) / sigma / s / s;
    """ if grad else ""
    return Template("""
    #define prior_sigma 0.5
    // bound the influence of scale on sigma to avoid the kernels
    // becoming too narrow when zooming in.
    #define s_bound 1.
    __forceinline__ __device__ void $name(
            float &w, $grad_arguments
            const int nv, const int iv, const int iV,
            const float l, const float s) {
        const float delta = (iv - nv/2) / s + l - iV;
        const float sigma = prior_sigma / min(s, s_bound);
        w = __expf(-0.5 * delta * delta / sigma / sigma) / $sqrt2pi / sigma;
        $grad_assignments
    }
    """).substitute(locals())

# `big_names` name big arguments (that we want to support discontiguity for)
def call_arguments(big_names):
    arguments = []
    for var in big_names:
        arguments.append("CudaNdarray_SIZE($%s)" % var)
        arguments.append("CudaNdarray_DEV_DATA($%s)" % var)
        arguments.append("CudaNdarray_DEV_DIMS($%s)" % var)
        arguments.append("CudaNdarray_DEV_STRIDES($%s)" % var)
    arguments.extend("CudaNdarray_DEV_DATA($%s)" % var for var in "abls")
    return ", \n".join(arguments)

def defn_arguments(big_names):
    arguments = []
    for var in big_names:
        arguments.append("const size_t %s_size" % var)
        arguments.append("%sfloat* %s" % ("const " if var == "x" else "", var))
        arguments.append("const int* %s_dims" % var)
        arguments.append("const int* %s_strides" % var)
    # a, b, l, s are contiguous
    arguments.extend("const float* %s" % var for var in "abls")
    return ", \n".join(arguments)

def strided_index(strides, indices):
    return " + ".join("%s[%i] * %s" % (strides, j, index) for j, index in enumerate(indices))

def weightpass_call(name, patch_shape, V, l, s, fail, grad=False):
    assert not grad # FIXME
    name += "_weightpass"
    W_ndim = 4
    ndim_spatial = len(patch_shape)
    max_nv = max(patch_shape)
    max_nV = (", ".join("max(CudaNdarray_HOST_DIMS(%s)[%i]"
                        % (V, 2 + j) for j in range(ndim_spatial))
                     + ", 0" + ")" * ndim_spatial)
    batch_size = "CudaNdarray_HOST_DIMS(%s)[0]" % V
    return Template("""
    int W_dims[$W_ndim];
    W_dims[0] = $batch_size;
    W_dims[1] = $max_nv;
    W_dims[2] = $max_nV;
    W_dims[3] = $ndim_spatial;
    CudaNdarray* W = (CudaNdarray*)CudaNdarray_New();
    if (W == NULL || CudaNdarray_alloc_contiguous(W, $W_ndim, W_dims)) {
        Py_XDECREF(W);
        W = NULL;
        PyErr_SetString(PyExc_ValueError,
                        "TimCropper: weight tensor allocation failed");
        $fail;
    }

    {
        dim3 block(16, 16);
        dim3 grid((W_dims[0] * W_dims[1] + block.x - 1) / block.x,
                  (W_dims[2] * W_dims[3] + block.y - 1) / block.y);
        $name<<<grid, block>>>(CudaNdarray_DEV_DATA(W),
                            CudaNdarray_DEV_DIMS(W),
                            CudaNdarray_DEV_STRIDES(W),
                            CudaNdarray_DEV_DATA($l),
                            CudaNdarray_DEV_DATA($s));
        CNDA_THREAD_SYNC;
        cudaError_t err = cudaGetLastError();
        if( cudaSuccess != err)
        {
            PyErr_Format(PyExc_RuntimeError,
                "Cuda error: %s: %s. (grid: %i x %i x %i;"
                " block: %i x %i x %i)\\n",
                "$name",
                cudaGetErrorString(err),
                grid.x, grid.y, grid.z,
                block.x, block.y, block.z);
            $fail;
        }
    }
    """).substitute(locals())

def weightpass_defn(name, patch_shape, grad=False):
    assert not grad #FIXME
    name += "_weightpass"
    weight_function_name = name + "_weightfun"
    ndim_spatial = len(patch_shape)
    weightfun_defn = weightfunction(weight_function_name, grad=grad)
    W_index = strided_index("W_strides", "ib iv iV j".split())
    nv = "+".join("(j == %i) * %i" % (j, dim) for j, dim in enumerate(patch_shape))
    from string import Template
    return Template("""
    $weightfun_defn

    __global__ void $name(float* W,
                          const int* W_strides,
                          const int* W_dims,
                          const float* l,
                          const float* s) {
        const int ib = blockIdx.x / W_dims[1],
                  iv = blockIdx.x % W_dims[1],
                  iV = blockIdx.y / W_dims[3],
                  j  = blockIdx.y % W_dims[3];
        if (ib < W_dims[0] && iv < W_dims[1] && iV < W_dims[2] && j < W_dims[3])
            $weight_function_name(
                W[$W_index],
                $nv,
                iv, iV,
                l[ib * $ndim_spatial + j],
                s[ib * $ndim_spatial + j]);
    }
    """).substitute(locals())
