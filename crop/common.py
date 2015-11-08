from string import Template

def gridblock(ndim_spatial, patch_dims):
    strings = []
    if ndim_spatial == 2:
        strings.append("""
        dim3 block(1, 32, 32);
        dim3 grid($patch_dims[0] * $patch_dims[1],
                  ($patch_dims[2] + block.y - 1) / block.y,
                  ($patch_dims[3] + block.z - 1) / block.z);
        """)
    elif ndim_spatial == 3:
        strings.append("""
        dim3 block(1, 32, 32);
        dim3 grid($patch_dims[0] * $patch_dims[1],
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
    grad_arguments = "float &dwdl, float &dwds," if grad else ""
    grad_assignments = """
        dwdl = w * -delta / sigma / sigma;
        // FIXME: rederive for bounded s
        dwds = w * ((iv - nv/2) * delta / sigma / sigma - s * (delta * delta / sigma / sigma - 1)) / s / s;
    """ if grad else ""
    return Template("""
    #define prior_sigma 0.5
    // bound the influence of scale on sigma to avoid the kernels
    // becoming too narrow when zooming in.
    #define s_bound 1.
    __device__ void $name(
            float &w, $grad_arguments
            const int nv, const int iv, const int iV,
            const float l, const float s) {
        const float delta = (iv - nv/2) / s + l - iV;
        const float sigma = prior_sigma / min(s, s_bound);

        w = exp(-0.5 * delta * delta / sigma / sigma) / sqrt(2*M_PI) / sigma;
        $grad_assignments
    }
    """).substitute(locals())
