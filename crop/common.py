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
        int i0 = blockIdx.x / $patch_dims[1],
            i1 = blockIdx.x % $patch_dims[1],
            i2v = blockIdx.y * blockDim.y + threadIdx.y,
            i3v = blockIdx.z * blockDim.z + threadIdx.z;
        """)
    elif ndim_spatial == 3:
        strings.append("""
        // assuming C order
        int i0 = blockIdx.x / $patch_dims[1],
            i1 = blockIdx.x % $patch_dims[1],
            i2v = blockIdx.y * blockDim.y + threadIdx.y;
        int i34v = blockIdx.z * blockDim.z + threadIdx.z;
        int i3v = i34v / $patch_dims[4],
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
    grad_arguments = "float &dw_dl, float &dw_ds," if grad else ""
    grad_assignments = """
        dw_dl = w * -delta / sigma2;
        // FIXME: rederive for bounded s
        dw_ds = w * (div * delta / sigma2 - s * (delta2_sigma2 - 1)) / s2;
    """ if grad else ""
    return Template("""__device__ void $name(float &w, $grad_arguments int nv, int iv, int iV, float l, float s) {
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
        w = exp(-0.5 * delta2_sigma2) / sqrt(2*M_PI) / sigma;
        $grad_assignments
    }
    """).substitute(locals())
