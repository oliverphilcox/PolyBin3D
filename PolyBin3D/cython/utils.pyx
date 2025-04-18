#cython: language_level=3

from __future__ import print_function
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, cos
from cython.parallel import prange

cdef extern from "complex.h" nogil:
    double complex cpow(double complex, double complex)
    double creal(double complex)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double bin_integrate(double complex[:,:,::1] spec, double[:,:,::1] modk, double kmin, double kmax, int nthreads):
    """Compute Sum_k |d(k)|^2 across all k in range."""
    cdef long ik1, ik2, ik3, nk1=modk.shape[0], nk2=modk.shape[1], nk3=modk.shape[2]
    cdef double out = 0.

    # Iterate over k grid  
    with nogil:
        for ik1 in prange(nk1, schedule='static', num_threads=nthreads):
            for ik2 in xrange(nk2):
                for ik3 in xrange(nk3):
                    if (modk[ik1,ik2,ik3]>=kmin) and (modk[ik1,ik2,ik3]<kmax):
                        out += creal(spec[ik1,ik2,ik3]*spec[ik1,ik2,ik3].conjugate())
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double bin_integrate_cross(double complex[:,:,::1] spec1, double complex[:,:,::1] spec2, double[:,:,::1] modk, double kmin, double kmax, int nthreads):
    """Compute Sum_k |d(k)|^2 across all k in range."""
    cdef long ik1, ik2, ik3, nk1=modk.shape[0], nk2=modk.shape[1], nk3=modk.shape[2]
    cdef double out = 0.

    # Iterate over k grid  
    for ik1 in prange(nk1, nogil=True, schedule='static', num_threads=nthreads):
        for ik2 in xrange(nk2):
            for ik3 in xrange(nk3):
                if (modk[ik1,ik2,ik3]>=kmin) and (modk[ik1,ik2,ik3]<kmax):
                    out += creal(spec1[ik1,ik2,ik3]*spec2[ik1,ik2,ik3].conjugate())
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=1] bin_integrate_cross_all(double complex[:,:,::1] spec1, double complex[:,:,::1] spec2, double[:,:,::1] modk, double[:] k_edges, int nthreads):
    """Compute Sum_k d1(k)d2(-k) across all k in range."""
    cdef long ik1, ik2, ik3, ibin, nk1=modk.shape[0], nk2=modk.shape[1], nk3=modk.shape[2], nbin=k_edges.shape[0]-1
    cdef np.ndarray[np.float64_t,ndim=1] out = np.zeros(nbin, dtype=np.float64)
    cdef double tmp

    # Iterate over k grid  
    for ibin in prange(nbin, nogil=True, schedule='static', num_threads=nthreads):
        tmp = 0.
        for ik1 in xrange(nk1):
            for ik2 in xrange(nk2):
                for ik3 in xrange(nk3):
                    if (modk[ik1,ik2,ik3]>=k_edges[ibin]) and (modk[ik1,ik2,ik3]<k_edges[ibin+1]):
                        tmp = tmp + creal(spec1[ik1,ik2,ik3]*spec2[ik1,ik2,ik3].conjugate())
        out[ibin] = tmp
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=1] bin_integrate_cross_coeff_all(double complex[:] coeff, double complex[:,:,::1] spec1, double complex[:,:,::1] spec2, double[:,:,::1] modk, double[:] k_edges, int nthreads):
    """Compute Sum_k d1(k)d2(-k) across all k in range, weighted by coefficients."""
    cdef long ik1, ik2, ik3, ibin, nk1=modk.shape[0], nk2=modk.shape[1], nk3=modk.shape[2], nbin=k_edges.shape[0]-1
    cdef np.ndarray[np.float64_t,ndim=1] out = np.zeros(nbin, dtype=np.float64)
    cdef double tmp

    # Iterate over k grid  
    for ibin in prange(nbin, nogil=True, schedule='static', num_threads=nthreads):
        tmp = 0.
        for ik1 in xrange(nk1):
            for ik2 in xrange(nk2):
                for ik3 in xrange(nk3):
                    if (modk[ik1,ik2,ik3]>=k_edges[ibin]) and (modk[ik1,ik2,ik3]<k_edges[ibin+1]):
                        tmp = tmp + creal(coeff[ibin]*spec1[ik1,ik2,ik3]*spec2[ik1,ik2,ik3].conjugate())
        out[ibin] = tmp
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double bin_integrate_l(double complex[:,:,::1] spec, double[:,:,::1] modk, double[:,:,::1] muk, int l, double kmin, double kmax, int nthreads):
    """Compute Sum_k |d(k)|^2 L_l(k.n) across all k in range."""
    cdef long ik1, ik2, ik3, nk1=modk.shape[0], nk2=modk.shape[1], nk3=modk.shape[2]
    cdef double out = 0.
    
    # Iterate over k grid  
    with nogil:
        for ik1 in prange(nk1, schedule='static', num_threads=nthreads):
            for ik2 in xrange(nk2):
                for ik3 in xrange(nk3):
                    if (modk[ik1,ik2,ik3]>=kmin) and (modk[ik1,ik2,ik3]<kmax):
                        out += creal(spec[ik1,ik2,ik3]*spec[ik1,ik2,ik3].conjugate())*legendre(l, muk[ik1,ik2,ik3])
    return out

cdef inline double legendre(int l, double mu) noexcept nogil:
    """Compute the Legendre multipole L_l(mu). We use the explicit forms for speed."""
    if l==0:
        return 1.
    elif l==1:
        return mu
    elif l==2:
        return 1./2.*(3*mu**2-1)
    elif l==3:
        return (-3*mu + 5*mu**3)/2.
    elif l==4:
        return (3 - 30*mu**2 + 35*mu**4)/8.
    elif l==5:
        return (15*mu - 70*mu**3 + 63*mu**5)/8.
    elif l==6: 
        return (-5 + 105*mu**2 - 315*mu**4 + 231*mu**6)/16.               

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double bin_integrate_real_ll(double[:,:,::1] spec, double[:,:,::1] modk, double[:,:,::1] muk, int l1, int l2, double kmin, double kmax, int nthreads):
    """Compute Sum_k d(k)^2 L_l1(k.n)L_l2(k.n) across all k in range for real d(k)."""
    cdef long ik1, ik2, ik3, nk1=modk.shape[0], nk2=modk.shape[1], nk3=modk.shape[2]
    cdef double out = 0.
    
    # Iterate over k grid  
    with nogil:
        for ik1 in prange(nk1, schedule='static', num_threads=nthreads):
            for ik2 in xrange(nk2):
                for ik3 in xrange(nk3):
                    if (modk[ik1,ik2,ik3]>=kmin) and (modk[ik1,ik2,ik3]<kmax):
                        out += spec[ik1,ik2,ik3]*spec[ik1,ik2,ik3]*legendre(l1, muk[ik1,ik2,ik3])*legendre(l2, muk[ik1,ik2,ik3])
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=1] bin_integrate_real_ll_vec(double[:,:,::1] spec, double[:,:,::1] modk, double[:,:,::1] muk, int l1, int l2, double[:] k_edges, int nthreads):
    """Compute Sum_k d(k)^2 L_l1(k.n)L_l2(k.n) across all k in range for real d(k)."""
    cdef long ik1, ik2, ik3, nk1=modk.shape[0], nk2=modk.shape[1], nk3=modk.shape[2]
    cdef long il, Nl = k_edges.shape[0]-1
    cdef double tmp
    cdef np.ndarray[np.float64_t,ndim=1] out = np.zeros(Nl, dtype=np.float64)
    
    # Iterate over k grid  
    with nogil:
        for il in prange(Nl, schedule='static', num_threads=nthreads):
            tmp = 0.
            for ik1 in xrange(nk1):
                for ik2 in xrange(nk2):
                    for ik3 in xrange(nk3):
                        if (modk[ik1,ik2,ik3]>=k_edges[il]) and (modk[ik1,ik2,ik3]<k_edges[il+1]):
                            tmp = tmp + spec[ik1,ik2,ik3]**2*legendre(l1, muk[ik1,ik2,ik3])*legendre(l2, muk[ik1,ik2,ik3])
            out[il] = tmp
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=1] bin_integrate_real_vec(double[:,:,::1] spec, double[:,:,::1] modk, double[:] k_edges, int nthreads):
    """Compute Sum_k d(k)^2 across all k in range for real d(k)."""
    cdef long ik1, ik2, ik3, nk1=modk.shape[0], nk2=modk.shape[1], nk3=modk.shape[2]
    cdef long il, Nl = k_edges.shape[0]-1
    cdef double tmp
    cdef np.ndarray[np.float64_t,ndim=1] out = np.zeros(Nl, dtype=np.float64)
    
    # Iterate over k grid  
    with nogil:
        for il in prange(Nl, schedule='static', num_threads=nthreads):
            tmp = 0.
            for ik1 in xrange(nk1):
                for ik2 in xrange(nk2):
                    for ik3 in xrange(nk3):
                        if (modk[ik1,ik2,ik3]>=k_edges[il]) and (modk[ik1,ik2,ik3]<k_edges[il+1]):
                            tmp = tmp + spec[ik1,ik2,ik3]**2
            out[il] = tmp
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double bin_integrate_real(double[:,:,::1] spec, double[:,:,::1] modk, double kmin, double kmax, int nthreads):
    """Compute Sum_k d(k)^2 across all k in range for real d(k)."""
    cdef long ik1, ik2, ik3, nk1=modk.shape[0], nk2=modk.shape[1], nk3=modk.shape[2]
    cdef double out = 0.
    
    # Iterate over k grid  
    with nogil:
        for ik1 in prange(nk1, schedule='static', num_threads=nthreads):
            for ik2 in xrange(nk2):
                for ik3 in xrange(nk3):
                    if (modk[ik1,ik2,ik3]>=kmin) and (modk[ik1,ik2,ik3]<kmax):
                        out += spec[ik1,ik2,ik3]*spec[ik1,ik2,ik3]
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double bin_integrate_cross_l(double complex[:,:,::1] spec1, double complex[:,:,::1] spec2, double[:,:,::1] modk, double[:,:,::1] muk, int l, double kmin, double kmax, int nthreads):
    """Compute Sum_k d1(k)Conj[d2(k)] L_l(k.n) across all k in range."""
    cdef long ik1, ik2, ik3, nk1=modk.shape[0], nk2=modk.shape[1], nk3=modk.shape[2]
    cdef double out = 0.
    
    # Iterate over k grid  
    with nogil:
        for ik1 in prange(nk1, schedule='static', num_threads=nthreads):
            for ik2 in xrange(nk2):
                for ik3 in xrange(nk3):
                    if (modk[ik1,ik2,ik3]>=kmin) and (modk[ik1,ik2,ik3]<kmax):
                        out += creal(spec1[ik1,ik2,ik3]*spec2[ik1,ik2,ik3].conjugate())*legendre(l, muk[ik1,ik2,ik3])
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=1] bin_integrate_cross_all_l(double complex[:,:,::1] spec1, double complex[:,:,::1] spec2, double[:,:,::1] modk, double[:,:,::1] muk, int l, double[:] k_edges, int nthreads):
    """Compute Sum_k d1(k)d2(-k) L_l(k.n) across all k in range."""
    cdef long ik1, ik2, ik3, ibin, nk1=modk.shape[0], nk2=modk.shape[1], nk3=modk.shape[2], nbin=k_edges.shape[0]-1
    cdef np.ndarray[np.float64_t,ndim=1] out = np.zeros(nbin, dtype=np.float64)
    cdef double tmp

    # Iterate over k grid  
    for ibin in prange(nbin, nogil=True, schedule='static', num_threads=nthreads):
        tmp = 0.
        for ik1 in xrange(nk1):
            for ik2 in xrange(nk2):
                for ik3 in xrange(nk3):
                    if (modk[ik1,ik2,ik3]>=k_edges[ibin]) and (modk[ik1,ik2,ik3]<k_edges[ibin+1]):
                        tmp = tmp + creal(spec1[ik1,ik2,ik3]*spec2[ik1,ik2,ik3].conjugate())*legendre(l, muk[ik1,ik2,ik3])
        out[ibin] = tmp
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.complex128_t,ndim=3] prod_map(double complex[:,:,::1] map1, double[:,:,::1] map2, int nthreads):
    """Compute map1(x)map2(x)."""
    cdef long ix1, ix2, ix3, nx1 = map1.shape[0], nx2 = map1.shape[1], nx3 = map1.shape[2]
    cdef np.ndarray[np.complex128_t,ndim=3] out = np.empty((nx1,nx2,nx3),dtype=np.complex128)
    assert nx1==map2.shape[0]
    assert nx2==map2.shape[1]
    assert nx3==map2.shape[2]

    with nogil:
        for ix1 in prange(nx1, schedule='static', num_threads=nthreads):
            for ix2 in xrange(nx2):
                for ix3 in xrange(nx3):
                    out[ix1,ix2,ix3] = map1[ix1,ix2,ix3]*map2[ix1,ix2,ix3]
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.complex128_t,ndim=3] divide_map(double complex[:,:,::1] map1, double[:,:,::1] map2, int nthreads):
    """Compute map1(x)/map2(x)."""
    cdef long ix1, ix2, ix3, nx1 = map1.shape[0], nx2 = map1.shape[1], nx3 = map1.shape[2]
    cdef np.ndarray[np.complex128_t,ndim=3] out = np.empty((nx1,nx2,nx3),dtype=np.complex128)
    assert nx1==map2.shape[0]
    assert nx2==map2.shape[1]
    assert nx3==map2.shape[2]

    with nogil:
        for ix1 in prange(nx1, schedule='static', num_threads=nthreads):
            for ix2 in xrange(nx2):
                for ix3 in xrange(nx3):
                    if map2[ix1,ix2,ix3]==0:
                        out[ix1,ix2,ix3] = 0.
                    else:
                        out[ix1,ix2,ix3] = map1[ix1,ix2,ix3]/map2[ix1,ix2,ix3]
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=3] divide_map_real(double[:,:,::1] map1, double[:,:,::1] map2, int nthreads):
    """Compute map1(x)/map2(x)."""
    cdef long ix1, ix2, ix3, nx1 = map1.shape[0], nx2 = map1.shape[1], nx3 = map1.shape[2]
    cdef np.ndarray[np.float64_t,ndim=3] out = np.empty((nx1,nx2,nx3),dtype=np.float64)
    assert nx1==map2.shape[0]
    assert nx2==map2.shape[1]
    assert nx3==map2.shape[2]

    with nogil:
        for ix1 in prange(nx1, schedule='static', num_threads=nthreads):
            for ix2 in xrange(nx2):
                for ix3 in xrange(nx3):
                    if map2[ix1,ix2,ix3]==0:
                        out[ix1,ix2,ix3] = 0.
                    else:
                        out[ix1,ix2,ix3] = map1[ix1,ix2,ix3]/map2[ix1,ix2,ix3]
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=3] prod_map_real(double[:,:,::1] map1, double[:,:,::1] map2, int nthreads):
    """Compute map1(x)map2(x)."""
    cdef long ix1, ix2, ix3, nx1 = map1.shape[0], nx2 = map1.shape[1], nx3 = map1.shape[2]
    cdef np.ndarray[np.float64_t,ndim=3] out = np.empty((nx1,nx2,nx3),dtype=np.float64)
    assert nx1==map2.shape[0]
    assert nx2==map2.shape[1]
    assert nx3==map2.shape[2]

    with nogil:
        for ix1 in prange(nx1, schedule='static', num_threads=nthreads):
            for ix2 in xrange(nx2):
                for ix3 in xrange(nx3):
                    out[ix1,ix2,ix3] = map1[ix1,ix2,ix3]*map2[ix1,ix2,ix3]
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=3] prod_map_real_diff(double[:,:,::1] map1, double[:,:,::1] map2, double[:,:,::1] map3, double[:,:,::1] map4, int nthreads):
    """Compute map1(x)map2(x)."""
    cdef long ix1, ix2, ix3, nx1 = map1.shape[0], nx2 = map1.shape[1], nx3 = map1.shape[2]
    cdef np.ndarray[np.float64_t,ndim=3] out = np.empty((nx1,nx2,nx3),dtype=np.float64)
    assert nx1==map2.shape[0]
    assert nx2==map2.shape[1]
    assert nx3==map2.shape[2]

    with nogil:
        for ix1 in prange(nx1, schedule='static', num_threads=nthreads):
            for ix2 in xrange(nx2):
                for ix3 in xrange(nx3):
                    out[ix1,ix2,ix3] = map1[ix1,ix2,ix3]*map2[ix1,ix2,ix3]-map3[ix1,ix2,ix3]*map4[ix1,ix2,ix3]
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void prod_map_real_sum(double[:,:,::1] map1, double[:,:,::1] map2, double[:,:,::1] out, int nthreads):
    """Compute map1(x)map2(x) and add to output."""
    cdef long ix1, ix2, ix3, nx1 = map1.shape[0], nx2 = map1.shape[1], nx3 = map1.shape[2]
    assert nx1==map2.shape[0]
    assert nx2==map2.shape[1]
    assert nx3==map2.shape[2]

    with nogil:
        for ix1 in prange(nx1, schedule='static', num_threads=nthreads):
            for ix2 in xrange(nx2):
                for ix3 in xrange(nx3):
                    out[ix1,ix2,ix3] += map1[ix1,ix2,ix3]*map2[ix1,ix2,ix3]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=3] pointing_GIC(double[:,:,::1] map1, double[:,:,::1] mask, double[:,:,::1] mask_GIC, int nthreads):
    """Compute pointing matrix with global integral constraint."""
    cdef long ix1, ix2, ix3, nx1 = map1.shape[0], nx2 = map1.shape[1], nx3 = map1.shape[2]
    cdef np.ndarray[np.float64_t,ndim=3] out = np.empty((nx1,nx2,nx3),dtype=np.float64)
    cdef double av_map=0, av_mask=0
    with nogil:
        for ix1 in prange(nx1, schedule='static', num_threads=nthreads):
            for ix2 in xrange(nx2):
                for ix3 in xrange(nx3):
                    av_map += map1[ix1,ix2,ix3]*mask_GIC[ix1,ix2,ix3]
                    av_mask += mask_GIC[ix1,ix2,ix3]
    av_map = av_map/av_mask

    with nogil:
        for ix1 in prange(nx1, schedule='static', num_threads=nthreads):
            for ix2 in xrange(nx2):
                for ix3 in xrange(nx3):
                    out[ix1,ix2,ix3] = mask[ix1,ix2,ix3]*(map1[ix1,ix2,ix3] - av_map)
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=3] pointing_GIC_transpose(double[:,:,::1] map1, double[:,:,::1] mask, double[:,:,::1] mask_GIC, int nthreads):
    """Compute pointing matrix transpose with global integral constraint."""
    cdef long ix1, ix2, ix3, nx1 = map1.shape[0], nx2 = map1.shape[1], nx3 = map1.shape[2]
    cdef np.ndarray[np.float64_t,ndim=3] out = np.empty((nx1,nx2,nx3),dtype=np.float64)
    cdef double av_map=0, av_mask=0
    with nogil:
        for ix1 in prange(nx1, schedule='static', num_threads=nthreads):
            for ix2 in xrange(nx2):
                for ix3 in xrange(nx3):
                    av_map += map1[ix1,ix2,ix3]*mask[ix1,ix2,ix3]
                    av_mask += mask_GIC[ix1,ix2,ix3]
    av_map = av_map/av_mask

    with nogil:
        for ix1 in prange(nx1, schedule='static', num_threads=nthreads):
            for ix2 in xrange(nx2):
                for ix3 in xrange(nx3):
                    out[ix1,ix2,ix3] = mask[ix1,ix2,ix3]*map1[ix1,ix2,ix3] - mask_GIC[ix1,ix2,ix3]*av_map
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=3] pointing_RIC(double[:,:,::1] map1, double[:,:,::1] mask, double[:,:,::1] mask_GIC, double[:] radial_edges, double[:,:,::1] radial_grid, int nthreads):
    """Compute pointing matrix with radial integral constraint."""
    cdef long ix1, ix2, ix3, iz, nx1 = map1.shape[0], nx2 = map1.shape[1], nx3 = map1.shape[2], nz = radial_edges.shape[0]-1
    cdef np.ndarray[np.float64_t,ndim=3] out = np.empty((nx1,nx2,nx3),dtype=np.float64)
    cdef double[:] av_map = np.zeros(nz, dtype=np.float64)
    cdef double[:] av_mask = np.zeros(nz, dtype=np.float64)
    
    # Compute averages of data in each radial bin
    with nogil:
        for ix1 in prange(nx1, schedule='static', num_threads=nthreads):
            for ix2 in xrange(nx2):
                for ix3 in xrange(nx3):
                    for iz in xrange(nz):
                        if (radial_grid[ix1,ix2,ix3]>=radial_edges[iz]) and (radial_grid[ix1,ix2,ix3]<radial_edges[iz+1]):
                            av_map[iz] += map1[ix1,ix2,ix3]*mask_GIC[ix1,ix2,ix3]
                            av_mask[iz] += mask_GIC[ix1,ix2,ix3]
    
    # Normalize
    for iz in xrange(nz):
        av_map[iz] = av_map[iz]/av_mask[iz]

    # Combine to form pointing matrix
    with nogil:
        for ix1 in prange(nx1, schedule='static', num_threads=nthreads):
            for ix2 in xrange(nx2):
                for ix3 in xrange(nx3):
                    for iz in xrange(nz):
                        if (radial_grid[ix1,ix2,ix3]>=radial_edges[iz]) and (radial_grid[ix1,ix2,ix3]<radial_edges[iz+1]):
                            out[ix1,ix2,ix3] = mask[ix1,ix2,ix3]*(map1[ix1,ix2,ix3] - av_map[iz])
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=3] pointing_RIC_transpose(double[:,:,::1] map1, double[:,:,::1] mask, double[:,:,::1] mask_GIC, double[:] radial_edges, double[:,:,::1] radial_grid, int nthreads):
    """Compute pointing matrix transpose with radial integral constraint."""
    cdef long ix1, ix2, ix3, iz, nx1 = map1.shape[0], nx2 = map1.shape[1], nx3 = map1.shape[2], nz = radial_edges.shape[0]-1
    cdef np.ndarray[np.float64_t,ndim=3] out = np.empty((nx1,nx2,nx3),dtype=np.float64)
    cdef double[:] av_map = np.zeros(nz, dtype=np.float64)
    cdef double[:] av_mask = np.zeros(nz, dtype=np.float64)
    
    # Compute averages of data in each radial bin
    with nogil:
        for ix1 in prange(nx1, schedule='static', num_threads=nthreads):
            for ix2 in xrange(nx2):
                for ix3 in xrange(nx3):
                    for iz in xrange(nz):
                        if (radial_grid[ix1,ix2,ix3]>=radial_edges[iz]) and (radial_grid[ix1,ix2,ix3]<radial_edges[iz+1]):
                            av_map[iz] += map1[ix1,ix2,ix3]*mask[ix1,ix2,ix3]
                            av_mask[iz] += mask_GIC[ix1,ix2,ix3]
    
    # Normalize
    for iz in xrange(nz):
        av_map[iz] = av_map[iz]/av_mask[iz]

    # Combine to form pointing matrix
    with nogil:
        for ix1 in prange(nx1, schedule='static', num_threads=nthreads):
            for ix2 in xrange(nx2):
                for ix3 in xrange(nx3):
                    for iz in xrange(nz):
                        if (radial_grid[ix1,ix2,ix3]>=radial_edges[iz]) and (radial_grid[ix1,ix2,ix3]<radial_edges[iz+1]):
                            out[ix1,ix2,ix3] = mask[ix1,ix2,ix3]*map1[ix1,ix2,ix3] - mask_GIC[ix1,ix2,ix3]*av_map[iz]
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.complex128_t,ndim=3] filt_map_full(double complex[:,:,::1] kmap, double[:,:,::1] modk, double kmin, double kmax, int nthreads):
    """Compute map1(x)map2(x)."""
    cdef long ix1, ix2, ix3, nx1 = kmap.shape[0], nx2 = kmap.shape[1], nx3 = kmap.shape[2]
    cdef np.ndarray[np.complex128_t,ndim=3] out = np.zeros((nx1,nx2,nx3),dtype=np.complex128)

    with nogil:
        for ix1 in prange(nx1, schedule='static', num_threads=nthreads):
            for ix2 in xrange(nx2):
                for ix3 in xrange(nx3):
                    if (modk[ix1,ix2,ix3]>=kmin) and (modk[ix1,ix2,ix3]<kmax):
                        out[ix1,ix2,ix3] = kmap[ix1,ix2,ix3]
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void filt_map_full_sum(double complex[:,:,::1] kmap, double[:,:,::1] modk, double kmin, double kmax, double complex[:,:,::1] out, int nthreads):
    """Compute map1(x)map2(x) and add to output."""
    cdef long ix1, ix2, ix3, nx1 = kmap.shape[0], nx2 = kmap.shape[1], nx3 = kmap.shape[2]
    
    with nogil:
        for ix1 in prange(nx1, schedule='static', num_threads=nthreads):
            for ix2 in xrange(nx2):
                for ix3 in xrange(nx3):
                    if (modk[ix1,ix2,ix3]>=kmin) and (modk[ix1,ix2,ix3]<kmax):
                        out[ix1,ix2,ix3] += kmap[ix1,ix2,ix3]
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.complex128_t,ndim=3] filt_map_full_l(double complex[:,:,::1] kmap, double[:,:,::1] modk, double[:,:,::1] muk, int l, double kmin, double kmax, int nthreads):
    """Compute map1(x)map2(x)Legendre(ell)."""
    cdef long ix1, ix2, ix3, nx1 = kmap.shape[0], nx2 = kmap.shape[1], nx3 = kmap.shape[2]
    cdef np.ndarray[np.complex128_t,ndim=3] out = np.zeros((nx1,nx2,nx3),dtype=np.complex128)
    
    with nogil:
        for ix1 in prange(nx1, schedule='static', num_threads=nthreads):
            for ix2 in xrange(nx2):
                for ix3 in xrange(nx3):
                    if (modk[ix1,ix2,ix3]>=kmin) and (modk[ix1,ix2,ix3]<kmax):
                        out[ix1,ix2,ix3] = kmap[ix1,ix2,ix3]*legendre(l, muk[ix1,ix2,ix3])
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.complex128_t,ndim=3] filt_map_full_ll(double complex[:,:,::1] kmap, double[:,:,::1] modk, double[:,:,::1] muk, int l, int lp, double kmin, double kmax, int nthreads):
    """Compute map1(x)map2(x)Legendre(ell)Legendre(ell')."""
    cdef long ix1, ix2, ix3, nx1 = kmap.shape[0], nx2 = kmap.shape[1], nx3 = kmap.shape[2]
    cdef np.ndarray[np.complex128_t,ndim=3] out = np.zeros((nx1,nx2,nx3),dtype=np.complex128)
    
    with nogil:
        for ix1 in prange(nx1, schedule='static', num_threads=nthreads):
            for ix2 in xrange(nx2):
                for ix3 in xrange(nx3):
                    if (modk[ix1,ix2,ix3]>=kmin) and (modk[ix1,ix2,ix3]<kmax):
                        out[ix1,ix2,ix3] = kmap[ix1,ix2,ix3]*legendre(l, muk[ix1,ix2,ix3])*legendre(lp, muk[ix1,ix2,ix3])
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void prod_map_sum(double complex[:,:,::1] map1, double[:,:,::1] map2, double complex[:,:,::1] out, int nthreads):
    """Compute map1(x)map2(x) and add to existing array."""
    cdef long ix1, ix2, ix3, nx1 = map1.shape[0], nx2 = map1.shape[1], nx3 = map1.shape[2]
    assert nx1==map2.shape[0]
    assert nx2==map2.shape[1]
    assert nx3==map2.shape[2]
    assert nx1==out.shape[0]
    assert nx2==out.shape[1]
    assert nx3==out.shape[2]
    
    for ix1 in prange(nx1, nogil=True, schedule='static', num_threads=nthreads):
        for ix2 in xrange(nx2):
            for ix3 in xrange(nx3):
                out[ix1,ix2,ix3] += map1[ix1,ix2,ix3]*map2[ix1,ix2,ix3]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void prod_map_sum_real(double[:,:,::1] map1, double[:,:,::1] map2, double[:,:,::1] out, int nthreads):
    """Compute map1(x)map2(x) and add to existing array."""
    cdef long ix1, ix2, ix3, nx1 = map1.shape[0], nx2 = map1.shape[1], nx3 = map1.shape[2]
    assert nx1==map2.shape[0]
    assert nx2==map2.shape[1]
    assert nx3==map2.shape[2]
    assert nx1==out.shape[0]
    assert nx2==out.shape[1]
    assert nx3==out.shape[2]
    
    with nogil:
        for ix1 in prange(nx1, schedule='static', num_threads=nthreads):
            for ix2 in xrange(nx2):
                for ix3 in xrange(nx3):
                    out[ix1,ix2,ix3] += map1[ix1,ix2,ix3]*map2[ix1,ix2,ix3]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef dict compute_real_harmonics(double[:,:,:,::1] coordinates, int lmax, bint odd_l, int nthreads):
    """Compute the real spherical harmonics on the coordinate grid."""
    cdef dict Ylms = {}
    cdef double norm, xh, yh, zh
    cdef int i1,i2,i3,n1=coordinates.shape[1],n2=coordinates.shape[2],n3=coordinates.shape[3]
    cdef double[:,:,:,::1] norm_coordinates = np.empty((3,n1,n2,n3),dtype=np.float64)
    cdef double[:,:,:,::1] ylm1, ylm2, ylm3, ylm4

    assert lmax >= 1
    
    # Define coordinates (with unit norm)
    for i1 in prange(n1, nogil=True, schedule='static', num_threads=nthreads):
        for i2 in xrange(n2):
            for i3 in xrange(n3):
                norm = sqrt(coordinates[0, i1, i2, i3]**2 + coordinates[1, i1, i2, i3]**2 + coordinates[2, i1, i2, i3]**2)

                # Safe divide
                if norm==0:
                    norm_coordinates[0,i1,i2,i3] = 0.0
                    norm_coordinates[1,i1,i2,i3] = 0.0
                    norm_coordinates[2,i1,i2,i3] = 0.0
                else:
                    norm_coordinates[0,i1,i2,i3] = coordinates[0,i1,i2,i3]/norm
                    norm_coordinates[1,i1,i2,i3] = coordinates[1,i1,i2,i3]/norm
                    norm_coordinates[2,i1,i2,i3] = coordinates[2,i1,i2,i3]/norm 

    # Compute spherical harmonics
    if odd_l and lmax >= 1:
        ylm1 = np.empty((3,n1,n2,n3),dtype=np.float64)
        for i1 in prange(n1, nogil=True, schedule='static', num_threads=nthreads):
            for i2 in xrange(n2):
                for i3 in xrange(n3):
                    xh = norm_coordinates[0,i1,i2,i3]
                    yh = norm_coordinates[1,i1,i2,i3]
                    zh = norm_coordinates[2,i1,i2,i3]
                    ylm1[0,i1,i2,i3] = yh
                    ylm1[1,i1,i2,i3] = zh
                    ylm1[2,i1,i2,i3] = xh
        Ylms[1] = ylm1
        
    if lmax >= 2:
        ylm2 = np.empty((5, n1, n2, n3), dtype=np.float64)
        for i1 in prange(n1, nogil=True, schedule='static', num_threads=nthreads):
            for i2 in xrange(n2):
                for i3 in xrange(n3):
                    xh = norm_coordinates[0,i1,i2,i3]
                    yh = norm_coordinates[1,i1,i2,i3]
                    zh = norm_coordinates[2,i1,i2,i3]
                    ylm2[0, i1, i2, i3] = 6. * xh * yh * sqrt(1. / 12.)
                    ylm2[1, i1, i2, i3] =   3. * yh * zh * sqrt(1. / 3.)
                    ylm2[2, i1, i2, i3] =  (zh**2 - xh**2 / 2. - yh**2 / 2.)
                    ylm2[3, i1, i2, i3] =  3. * xh * zh * sqrt(1. / 3.)
                    ylm2[4, i1, i2, i3] =  (3 * xh**2 - 3 * yh**2) * sqrt(1. / 12.)
        Ylms[2] = ylm2

    if odd_l and lmax >= 3:
        ylm3 = np.empty((7, n1, n2, n3), dtype=np.float64)
        for i1 in prange(n1, nogil=True, schedule='static', num_threads=nthreads):
                for i2 in xrange(n2):
                    for i3 in xrange(n3):
                        xh = norm_coordinates[0,i1,i2,i3]
                        yh = norm_coordinates[1,i1,i2,i3]
                        zh = norm_coordinates[2,i1,i2,i3]
                        ylm3[0, i1, i2, i3] = (45. * xh**2 * yh - 15. * yh**3) * sqrt(1. / 360.)
                        ylm3[1, i1, i2, i3] = (30. * xh * yh * zh) * sqrt(1. / 60.)
                        ylm3[2, i1, i2, i3] = (-1.5 * xh**2 * yh - 1.5 * yh**3 + 6. * yh * zh**2) * sqrt(1. / 6.)
                        ylm3[3, i1, i2, i3] = (-1.5 * xh**2 * zh - 1.5 * yh**2 * zh + zh**3)
                        ylm3[4, i1, i2, i3] = (-1.5 * xh**3 - 1.5 * xh * yh**2 + 6. * xh * zh**2) * sqrt(1. / 6.)
                        ylm3[5, i1, i2, i3] = (15. * xh**2 * zh - 15. * yh**2 * zh) * sqrt(1. / 60.)
                        ylm3[6, i1, i2, i3] = (15. * xh**3 - 45. * xh * yh**2) * sqrt(1. / 360.)
        Ylms[3] = ylm3

    if lmax >= 4:
        ylm4 = np.empty((9, n1, n2, n3), dtype=np.float64)
        for i1 in prange(n1, nogil=True, schedule='static', num_threads=nthreads):
            for i2 in xrange(n2):
                for i3 in xrange(n3):
                    xh = norm_coordinates[0,i1,i2,i3]
                    yh = norm_coordinates[1,i1,i2,i3]
                    zh = norm_coordinates[2,i1,i2,i3]
                    ylm4[0, i1, i2, i3] = (420. * xh**3 * yh - 420. * xh * yh**3) * sqrt(1. / 20160.)
                    ylm4[1, i1, i2, i3] = (315. * xh**2 * yh * zh - 105. * yh**3 * zh) * sqrt(1. / 2520.)
                    ylm4[2, i1, i2, i3] = (-15. * xh**3 * yh - 15. * xh * yh**3 + 90. * xh * yh * zh**2) * sqrt(1. / 180.)
                    ylm4[3, i1, i2, i3] = (-7.5 * xh**2 * yh * zh - 7.5 * yh**3 * zh + 10. * yh * zh**3) * sqrt(1. / 10.)
                    ylm4[4, i1, i2, i3] = 35. / 8. * zh**4 - 15. / 4. * zh**2 + 3. / 8.
                    ylm4[5, i1, i2, i3] = (-15. / 2. * xh**3 * zh - 15. * xh * yh**2 * zh / 2. + 10. * xh * zh**3) * sqrt(1. / 10.)
                    ylm4[6, i1, i2, i3] = (-15. / 2. * xh**4 + 45. * xh**2 * zh**2 + 15. / 2. * yh**4 - 45. * yh**2 * zh**2) * sqrt(1. / 180.)
                    ylm4[7, i1, i2, i3] = (105. * xh**3 * zh - 315. * xh * yh**2 * zh) * sqrt(1. / 2520.)
                    ylm4[8, i1, i2, i3] = (105. * xh**4 - 630. * xh**2 * yh**2 + 105. * yh**4) * sqrt(1. / 20160.)
        Ylms[4] = ylm4

    return Ylms

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.complex128_t,ndim=1] assemble_fish_filt(double complex[:,:,::1] kmap, double complex[:,:,:,::1] Q_array, double[:,:,::1] modk, double kmin, double kmax, int nthreads):
    """Filter a map and add it to the Fisher matrix"""
    cdef long i_out, i1, i2, i3, n_out = Q_array.shape[0], n1 = Q_array.shape[1], n2 = Q_array.shape[2], n3 = Q_array.shape[3]
    cdef np.ndarray[np.float64_t,ndim=1] out = np.empty(n_out, dtype=np.float64)
    cdef double tmp=0.

    with nogil:
        for i_out in prange(n_out, schedule='static', num_threads=nthreads):
            tmp = 0
            for i1 in xrange(n1):
                for i2 in xrange(n2):
                    for i3 in xrange(n3):
                        if (modk[i1,i2,i3]>=kmin) and (modk[i1,i2,i3]<kmax):
                            tmp = tmp + creal(kmap[i1,i2,i3].conjugate()*Q_array[i_out,i1,i2,i3])
            out[i_out] = tmp
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=1] assemble_b3_all(double[:,:,:,:,::1] g_maps, long[:,::1] all_bins, int nthreads):
    """Assemble the bispectrum numerator from a set of g_lb(x) maps."""
    cdef long i1, i2, i3, n1 = g_maps.shape[2], n2 = g_maps.shape[3], n3 = g_maps.shape[4]
    cdef long index, l, bin1, bin2, bin3, n_bins = len(all_bins)
    cdef double tmp
    cdef np.ndarray[np.float64_t,ndim=1] out=np.zeros(n_bins, dtype=np.float64)

    for index in prange(n_bins,nogil=True, schedule='static', num_threads=nthreads):
        # Define bins
        bin1 = all_bins[index][0]
        bin2 = all_bins[index][1]
        bin3 = all_bins[index][2]
        l = all_bins[index][3]
        
        # Sum over maps
        tmp = 0
        for i1 in xrange(n1):
            for i2 in xrange(n2):
                for i3 in xrange(n3):
                    tmp = tmp+g_maps[0][bin1][i1,i2,i3]*g_maps[0][bin2][i1,i2,i3]*g_maps[l//2][bin3][i1,i2,i3]
        out[index] = tmp
    
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=1] assemble_bshot_all(double[:,:,:,:,::1] gA_maps, double[:,:,:,:,::1] gB_maps, double[:,:,:,:,::1] gC_maps, long[:,::1] all_bins, int nthreads):
    """Assemble the bispectrum numerator from a set of g_lb(x) maps."""
    cdef long i1, i2, i3, n1 = gA_maps.shape[2], n2 = gA_maps.shape[3], n3 = gA_maps.shape[4]
    cdef long index, l, bin1, bin2, bin3, n_bins = len(all_bins)
    cdef double tmp
    cdef np.ndarray[np.float64_t,ndim=1] out=np.zeros(n_bins, dtype=np.float64)

    for index in prange(n_bins,nogil=True, schedule='static', num_threads=nthreads):
        # Define bins
        bin1 = all_bins[index][0]
        bin2 = all_bins[index][1]
        bin3 = all_bins[index][2]
        l = all_bins[index][3]
        
        # Sum over maps
        tmp = 0
        for i1 in xrange(n1):
            for i2 in xrange(n2):
                for i3 in xrange(n3):
                    tmp = tmp+(gA_maps[0][bin1][i1,i2,i3]*gB_maps[0][bin2][i1,i2,i3]*gC_maps[l//2][bin3][i1,i2,i3]+gA_maps[0][bin1][i1,i2,i3]*gC_maps[0][bin2][i1,i2,i3]*gB_maps[l//2][bin3][i1,i2,i3]+gB_maps[0][bin1][i1,i2,i3]*gA_maps[0][bin2][i1,i2,i3]*gC_maps[l//2][bin3][i1,i2,i3]+gB_maps[0][bin1][i1,i2,i3]*gC_maps[0][bin2][i1,i2,i3]*gA_maps[l//2][bin3][i1,i2,i3]+gC_maps[0][bin1][i1,i2,i3]*gA_maps[0][bin2][i1,i2,i3]*gB_maps[l//2][bin3][i1,i2,i3]+gC_maps[0][bin1][i1,i2,i3]*gB_maps[0][bin2][i1,i2,i3]*gA_maps[l//2][bin3][i1,i2,i3])
        out[index] = tmp
    
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=1] assemble_b1_all(double[:,:,:,:,::1] g_maps, double[:,:,:,:,::1] simg_maps, long[:,::1] all_bins, int nthreads):
    """Assemble the bispectrum numerator from a set of g_lb(x) maps."""
    cdef long i1, i2, i3, n1 = g_maps.shape[2], n2 = g_maps.shape[3], n3 = g_maps.shape[4]
    cdef long index, l, bin1, bin2, bin3, n_bins = len(all_bins)
    cdef double tmp
    cdef np.ndarray[np.float64_t,ndim=1] out=np.zeros(n_bins, dtype=np.float64)

    for index in prange(n_bins,nogil=True, schedule='static', num_threads=nthreads):
        # Define bins
        bin1 = all_bins[index][0]
        bin2 = all_bins[index][1]
        bin3 = all_bins[index][2]
        l = all_bins[index][3]
        
        # Sum over maps
        tmp = 0
        for i1 in xrange(n1):
            for i2 in xrange(n2):
                for i3 in xrange(n3):
                    tmp = tmp+g_maps[0][bin1][i1,i2,i3]*simg_maps[0][bin2][i1,i2,i3]*simg_maps[l//2][bin3][i1,i2,i3]
                    tmp = tmp+simg_maps[0][bin1][i1,i2,i3]*g_maps[0][bin2][i1,i2,i3]*simg_maps[l//2][bin3][i1,i2,i3]
                    tmp = tmp+simg_maps[0][bin1][i1,i2,i3]*simg_maps[0][bin2][i1,i2,i3]*g_maps[l//2][bin3][i1,i2,i3]
        out[index] = tmp
    
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double sum_prod(double[:,:,::1] map1, double[:,:,::1] map2, int nthreads):
    """Assemble the bispectrum numerator from a set of g_lb(x) maps."""
    cdef long i1, i2, i3, n1 = map1.shape[0], n2 = map1.shape[1], n3 = map1.shape[2]
    cdef double out=0

    for i1 in prange(n1, nogil=True, schedule='static', num_threads=nthreads):
        for i2 in xrange(n2):
            for i3 in xrange(n3):
                out += map1[i1,i2,i3]*map2[i1,i2,i3]
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double sum_prod3(double[:,:,::1] map1, double[:,:,::1] map2, double[:,:,::1] map3, int nthreads):
    """Assemble the bispectrum numerator from a set of g_lb(x) maps."""
    cdef long i1, i2, i3, n1 = map1.shape[0], n2 = map1.shape[1], n3 = map1.shape[2]
    cdef double out=0

    for i1 in prange(n1, nogil=True, schedule='static', num_threads=nthreads):
        for i2 in xrange(n2):
            for i3 in xrange(n3):
                out += map1[i1,i2,i3]*map2[i1,i2,i3]*map3[i1,i2,i3]
    return out        