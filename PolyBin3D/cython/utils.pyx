#cython: language_level=3

from __future__ import print_function
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt
from cython.parallel import prange

cdef extern from "complex.h" nogil:
    double complex cpow(double complex, double complex)
    double creal(double complex)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef class IntegrationUtils:

    # Define local memviews
    cdef int nthreads
    cdef bint real_fft
    cdef long nk1, nk2, nk3
    cdef double[:,:,::1] modk
    cdef double[:,:,::1] muk
    cdef double[:,:,::1] degk

    def __init__(self, modk, real_fft, nthreads, degk=None, muk=None):
        """Class to perform integrals of pairs of maps across all k-bins. We can use either complex-to-complex or real-to-complex FFTs here.
        There are various functions integrating slightly different things."""

        # Define local attributes
        self.modk = modk
        self.nk1 = modk.shape[0]
        self.nk2 = modk.shape[1]
        self.nk3 = modk.shape[2]
        self.nthreads = nthreads
        self.real_fft = real_fft

        # Load degeneracy factor
        if degk is not None:
            assert self.real_fft
            self.degk = degk

        # Load k angle
        if muk is not None:
            self.muk = muk

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef np.ndarray[np.float64_t,ndim=1] cross_integrate(self, double complex[:,:,::1] spec1, double complex[:,:,::1] spec2, double[:] k_edges, int l):
        """Compute Sum_k d1(k)d2(-k)Legendre(l, k) across all k in range."""
        cdef long ik1, ik2, ik3, ibin, nbin=k_edges.shape[0]-1
        cdef np.ndarray[np.float64_t,ndim=1] out = np.zeros(nbin, dtype=np.float64)
        cdef double tmp, leg

        # Iterate over k grid  
        for ibin in prange(nbin, schedule='static', num_threads=self.nthreads, nogil=True):
            tmp = 0.
            for ik1 in xrange(self.nk1):
                for ik2 in xrange(self.nk2):
                    for ik3 in xrange(self.nk3):
                        if (self.modk[ik1,ik2,ik3]>=k_edges[ibin]) and (self.modk[ik1,ik2,ik3]<k_edges[ibin+1]):
                            # Define Legendre polynomial
                            if l!=0:
                                leg = legendre(l, self.muk[ik1,ik2,ik3])
                            else:
                                leg = 1
                            # Add to bin    
                            if self.real_fft:
                                tmp = tmp + self.degk[ik1,ik2,ik3]*creal(spec1[ik1,ik2,ik3]*spec2[ik1,ik2,ik3].conjugate())*leg
                            else:
                                tmp = tmp + creal(spec1[ik1,ik2,ik3]*spec2[ik1,ik2,ik3].conjugate())*leg
            out[ibin] = tmp
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef np.ndarray[np.float64_t,ndim=1] integrate(self, double complex[:,:,::1] spec, double[:] k_edges, int l):
        """Compute Sum_k |d(k)|^2 Legendre(l, k) across all k in range."""
        cdef long ik1, ik2, ik3, ibin, nbin=k_edges.shape[0]-1
        cdef np.ndarray[np.float64_t,ndim=1] out = np.zeros(nbin, dtype=np.float64)
        cdef double tmp, leg

        # Iterate over k grid  
        for ibin in prange(nbin, schedule='static', num_threads=self.nthreads, nogil=True):
            tmp = 0.
            for ik1 in xrange(self.nk1):
                for ik2 in xrange(self.nk2):
                    for ik3 in xrange(self.nk3):
                        if (self.modk[ik1,ik2,ik3]>=k_edges[ibin]) and (self.modk[ik1,ik2,ik3]<k_edges[ibin+1]):
                            # Define Legendre polynomial
                            if l!=0:
                                leg = legendre(l, self.muk[ik1,ik2,ik3])
                            else:
                                leg = 1
                            # Add to bin    
                            if self.real_fft:
                                tmp = tmp + self.degk[ik1,ik2,ik3]*creal(spec[ik1,ik2,ik3]*spec[ik1,ik2,ik3].conjugate())*leg
                            else:
                                tmp = tmp + creal(spec[ik1,ik2,ik3]*spec[ik1,ik2,ik3].conjugate())*leg                    
            out[ibin] = tmp
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef np.ndarray[np.float64_t,ndim=1] real_integrate_double(self, double[:,:,::1] spec, int l1, int l2, double[:] k_edges):
        """Compute Sum_k d(k)^2 L_l1(k.n)L_l2(k.n) across all k in range for real d(k)."""
        cdef long ik1, ik2, ik3, il, Nl = k_edges.shape[0]-1
        cdef np.ndarray[np.float64_t,ndim=1] out = np.zeros(Nl, dtype=np.float64)
        cdef double tmp, leg
        
        # Iterate over k grid  
        with nogil:
            for il in prange(Nl, schedule='static', num_threads=self.nthreads):
                tmp = 0.
                for ik1 in xrange(self.nk1):
                    for ik2 in xrange(self.nk2):
                        for ik3 in xrange(self.nk3):
                            # Compute Legendre polynomial
                            if l1==0 and l2==0:
                                leg = 1
                            else:
                                leg = legendre(l1, self.muk[ik1,ik2,ik3])*legendre(l2, self.muk[ik1,ik2,ik3])
                            # Add to bin
                            if (self.modk[ik1,ik2,ik3]>=k_edges[il]) and (self.modk[ik1,ik2,ik3]<k_edges[il+1]):
                                if self.real_fft:
                                    tmp = tmp + self.degk[ik1,ik2,ik3]*spec[ik1,ik2,ik3]**2*leg
                                else:
                                    tmp = tmp + spec[ik1,ik2,ik3]**2*leg
                out[il] = tmp
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef np.ndarray[np.float64_t,ndim=1] cross_integrate_wcoeff(self, double complex[:] coeff, double complex[:,:,::1] spec1, double complex[:,:,::1] spec2, double[:] k_edges):
        """Compute Sum_k d1(k)d2(-k) across all k in range, weighted by complex coefficients."""
        cdef long ik1, ik2, ik3, ibin, nbin=k_edges.shape[0]-1
        cdef np.ndarray[np.float64_t,ndim=1] out = np.zeros(nbin, dtype=np.float64)
        cdef double tmp

        # Iterate over k grid  
        for ibin in prange(nbin, schedule='static', num_threads=self.nthreads, nogil=True):
            tmp = 0.
            for ik1 in xrange(self.nk1):
                for ik2 in xrange(self.nk2):
                    for ik3 in xrange(self.nk3):
                        if (self.modk[ik1,ik2,ik3]>=k_edges[ibin]) and (self.modk[ik1,ik2,ik3]<k_edges[ibin+1]):
                            if self.real_fft:
                                tmp = tmp + self.degk[ik1,ik2,ik3]*creal(coeff[ibin]*spec1[ik1,ik2,ik3]*spec2[ik1,ik2,ik3].conjugate())
                            else:
                                tmp = tmp + creal(coeff[ibin]*spec1[ik1,ik2,ik3]*spec2[ik1,ik2,ik3].conjugate())
            out[ibin] = tmp
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef np.ndarray[np.complex128_t,ndim=1] integrate_row(self, double complex[:,:,::1] input_map, double complex[:,:,:,::1] input_array, double kmin, double kmax):
        """Compute Sum_k d(k)e_i(k)* across all k in [kmin, kmax)."""
        cdef long i_out, ik1, ik2, ik3, n_out = input_array.shape[0]
        cdef np.ndarray[np.float64_t,ndim=1] out = np.zeros(n_out, dtype=np.float64)
        cdef double tmp=0.

        for i_out in prange(n_out, schedule='static', num_threads=self.nthreads, nogil=True):
            tmp = 0
            for ik1 in xrange(self.nk1):
                for ik2 in xrange(self.nk2):
                    for ik3 in xrange(self.nk3):
                        if (self.modk[ik1,ik2,ik3]>=kmin) and (self.modk[ik1,ik2,ik3]<kmax):
                            if self.real_fft:
                                tmp = tmp + self.degk[ik1,ik2,ik3]*creal(input_map[ik1,ik2,ik3].conjugate()*input_array[i_out,ik1,ik2,ik3])
                            else:
                                tmp = tmp + creal(input_map[ik1,ik2,ik3].conjugate()*input_array[i_out,ik1,ik2,ik3])
            out[i_out] = tmp
        return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef class MapUtils:

    # Define local memviews
    cdef int nthreads
    cdef double[:,:,::1] modk
    cdef double[:,:,::1] muk
    cdef int nk1, nk2, nk3
    cdef int nx1, nx2, nx3

    def __init__(self, modk, gridsize, nthreads, muk=None):
        """Class to perform various low-level map related functions."""

        # Define local memviews
        self.nthreads = nthreads
        self.modk = modk
        self.nk1 = modk.shape[0]
        self.nk2 = modk.shape[1]
        self.nk3 = modk.shape[2]
        self.nx1 = gridsize[0]
        self.nx2 = gridsize[1]
        self.nx3 = gridsize[2]
        
        # Define optional attributes
        if muk is not None:
            self.muk = muk

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef np.ndarray[np.complex128_t,ndim=3] prod_fourier(self, double complex[:,:,::1] map1, double[:,:,::1] map2):
        """Compute map1(k)map2(k) in Fourier-space (where map2 is real)."""
        cdef long ik1, ik2, ik3
        cdef np.ndarray[np.complex128_t,ndim=3] out = np.empty((self.nk1,self.nk2,self.nk3),dtype=np.complex128)
        assert self.nk1==map2.shape[0]
        assert self.nk2==map2.shape[1]
        assert self.nk3==map2.shape[2]

        for ik1 in prange(self.nk1, schedule='static', num_threads=self.nthreads, nogil=True):
            for ik2 in xrange(self.nk2):
                for ik3 in xrange(self.nk3):
                    out[ik1,ik2,ik3] = map1[ik1,ik2,ik3]*map2[ik1,ik2,ik3]
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef np.ndarray[np.complex128_t,ndim=3] div_fourier(self, double complex[:,:,::1] map1, double[:,:,::1] map2):
        """Compute map1(k)/map2(k) in Fourier-space (where map2 is real)."""
        cdef long ik1, ik2, ik3
        cdef np.ndarray[np.complex128_t,ndim=3] out = np.zeros((self.nk1,self.nk2,self.nk3),dtype=np.complex128)
        assert self.nk1==map2.shape[0]
        assert self.nk2==map2.shape[1]
        assert self.nk3==map2.shape[2]

        for ik1 in prange(self.nk1, schedule='static', num_threads=self.nthreads, nogil=True):
            for ik2 in xrange(self.nk2):
                for ik3 in xrange(self.nk3):
                    if map2[ik1,ik2,ik3]!=0:
                        out[ik1,ik2,ik3] = map1[ik1,ik2,ik3]/map2[ik1,ik2,ik3]
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef np.ndarray[np.float64_t,ndim=3] prod_real(self, double[:,:,::1] map1, double[:,:,::1] map2):
        """Compute map1(x)map2(x) in real-space."""
        cdef long ix1, ix2, ix3
        cdef np.ndarray[np.float64_t,ndim=3] out = np.empty((self.nx1,self.nx2,self.nx3),dtype=np.float64)
        assert self.nx1==map2.shape[0]
        assert self.nx2==map2.shape[1]
        assert self.nx3==map2.shape[2]
        
        for ix1 in prange(self.nx1, schedule='static', num_threads=self.nthreads, nogil=True):
            for ix2 in xrange(self.nx2):
                for ix3 in xrange(self.nx3):
                    out[ix1,ix2,ix3] = map1[ix1,ix2,ix3]*map2[ix1,ix2,ix3]
        return out
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef np.ndarray[np.float64_t,ndim=3] div_real(self, double[:,:,::1] map1, double[:,:,::1] map2):
        """Compute map1(x)/map2(x) in real-space."""
        cdef long ix1, ix2, ix3
        cdef np.ndarray[np.float64_t,ndim=3] out = np.zeros((self.nx1,self.nx2,self.nx3),dtype=np.float64)
        assert self.nx1==map2.shape[0]
        assert self.nx2==map2.shape[1]
        assert self.nx3==map2.shape[2]
        
        for ix1 in prange(self.nx1, schedule='static', num_threads=self.nthreads, nogil=True):
            for ix2 in xrange(self.nx2):
                for ix3 in xrange(self.nx3):
                    if map2[ix1,ix2,ix3]!=0:
                        out[ix1,ix2,ix3] = map1[ix1,ix2,ix3]/map2[ix1,ix2,ix3]
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef void prod_fourier_sum(self, double complex[:,:,::1] map1, double[:,:,::1] map2, double complex[:,:,::1] out):
        """Compute map1(k)map2(k) in Fourier-space (where map2 is real) and add to existing array."""
        cdef long ik1, ik2, ik3
        assert self.nk1==map2.shape[0]
        assert self.nk2==map2.shape[1]
        assert self.nk3==map2.shape[2]
        assert self.nk1==out.shape[0]
        assert self.nk2==out.shape[1]
        assert self.nk3==out.shape[2]
        
        for ik1 in prange(self.nk1, schedule='static', num_threads=self.nthreads, nogil=True):
            for ik2 in xrange(self.nk2):
                for ik3 in xrange(self.nk3):
                    out[ik1,ik2,ik3] += map1[ik1,ik2,ik3]*map2[ik1,ik2,ik3]
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef void prod_real_sum(self, double[:,:,::1] map1, double[:,:,::1] map2, double[:,:,::1] out):
        """Compute map1(x)map2(x) in real-space and add to existing array."""
        cdef long ix1, ix2, ix3
        assert self.nx1==map2.shape[0]
        assert self.nx2==map2.shape[1]
        assert self.nx3==map2.shape[2]
        assert self.nx1==out.shape[0]
        assert self.nx2==out.shape[1]
        assert self.nx3==out.shape[2]
        
        for ix1 in prange(self.nx1, schedule='static', num_threads=self.nthreads, nogil=True):
            for ix2 in xrange(self.nx2):
                for ix3 in xrange(self.nx3):
                    out[ix1,ix2,ix3] += map1[ix1,ix2,ix3]*map2[ix1,ix2,ix3]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef np.ndarray[np.complex128_t,ndim=3] fourier_filter(self, double complex[:,:,::1] kmap, int l, double kmin, double kmax):
        """Filter f(k) to Fourier modes with k in [kmin, kmax), and multiply by a Legendre polynomial."""
        cdef long ik1, ik2, ik3
        cdef double leg
        cdef np.ndarray[np.complex128_t,ndim=3] out = np.zeros((self.nk1,self.nk2,self.nk3),dtype=np.complex128)

        for ik1 in prange(self.nk1, schedule='static', num_threads=self.nthreads, nogil=True):
            for ik2 in xrange(self.nk2):
                for ik3 in xrange(self.nk3):
                    if (self.modk[ik1,ik2,ik3]>=kmin) and (self.modk[ik1,ik2,ik3]<kmax):
                        # Compute legendre polynomial
                        if l==0:
                            leg = 1
                        else:
                            leg = legendre(l, self.muk[ik1,ik2,ik3])
                        out[ik1,ik2,ik3] = kmap[ik1,ik2,ik3]*leg
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef np.ndarray[np.complex128_t,ndim=3] fourier_filter_ll(self, double complex[:,:,::1] kmap, int l1, int l2, double kmin, double kmax):
        """Filter f(k) to Fourier modes with k in [kmin, kmax), and multiply by a Legendre polynomial."""
        cdef long ik1, ik2, ik3
        cdef double leg
        cdef np.ndarray[np.complex128_t,ndim=3] out = np.zeros((self.nk1,self.nk2,self.nk3),dtype=np.complex128)

        for ik1 in prange(self.nk1, schedule='static', num_threads=self.nthreads, nogil=True):
            for ik2 in xrange(self.nk2):
                for ik3 in xrange(self.nk3):
                    if (self.modk[ik1,ik2,ik3]>=kmin) and (self.modk[ik1,ik2,ik3]<kmax):
                        # Compute legendre polynomial
                        if l1==0 and l2==0:
                            leg = 1
                        else:
                            leg = legendre(l1, self.muk[ik1,ik2,ik3])*legendre(l2, self.muk[ik1,ik2,ik3])
                        out[ik1,ik2,ik3] = kmap[ik1,ik2,ik3]*leg
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef void fourier_filter_sum(self, double complex[:,:,::1] kmap, double complex[:,:,::1] out, int l, double kmin, double kmax):
        """Filter f(k) to Fourier modes with k in [kmin, kmax), and multiply by a Legendre polynomial. This is added to the output map."""
        cdef long ik1, ik2, ik3
        cdef double leg
        
        for ik1 in prange(self.nk1, schedule='static', num_threads=self.nthreads, nogil=True):
            for ik2 in xrange(self.nk2):
                for ik3 in xrange(self.nk3):
                    if (self.modk[ik1,ik2,ik3]>=kmin) and (self.modk[ik1,ik2,ik3]<kmax):
                        # Compute legendre polynomial
                        if l==0:
                            leg = 1
                        else:
                            leg = legendre(l, self.muk[ik1,ik2,ik3])
                        out[ik1,ik2,ik3] += kmap[ik1,ik2,ik3]*leg
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef np.ndarray[np.complex128_t,ndim=3] prod_fourier_filter(self, double complex[:,:,::1] map1, double[:,:,::1] map2, double kmin, double kmax):
        """Compute map1(k)map2(k) in Fourier-space (where map2 is real), filtering to Fourier-modes with k in [kmin, kmax)."""
        cdef long ik1, ik2, ik3
        cdef np.ndarray[np.complex128_t,ndim=3] out = np.zeros((self.nk1,self.nk2,self.nk3),dtype=np.complex128)
        assert self.nk1==map2.shape[0]
        assert self.nk2==map2.shape[1]
        assert self.nk3==map2.shape[2]

        for ik1 in prange(self.nk1, schedule='static', num_threads=self.nthreads, nogil=True):
            for ik2 in xrange(self.nk2):
                for ik3 in xrange(self.nk3):
                    if (self.modk[ik1,ik2,ik3]>=kmin) and (self.modk[ik1,ik2,ik3]<kmax):
                        out[ik1,ik2,ik3] = map1[ik1,ik2,ik3]*map2[ik1,ik2,ik3]
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef double sum_pair(self, double[:,:,::1] map1, double[:,:,::1] map2):
        """Sum over a pair of real-space maps."""
        cdef long ix1, ix2, ix3
        cdef double out=0

        for ix1 in prange(self.nx1, schedule='static', num_threads=self.nthreads, nogil=True):
            for ix2 in xrange(self.nx2):
                for ix3 in xrange(self.nx3):
                    out += map1[ix1,ix2,ix3]*map2[ix1,ix2,ix3]
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef double sum_triplet(self, double[:,:,::1] map1, double[:,:,::1] map2, double[:,:,::1] map3):
        """Sum over a triplet of real-space maps."""
        cdef long ix1, ix2, ix3
        cdef double out=0

        for ix1 in prange(self.nx1, schedule='static', num_threads=self.nthreads, nogil=True):
            for ix2 in xrange(self.nx2):
                for ix3 in xrange(self.nx3):
                    out += map1[ix1,ix2,ix3]*map2[ix1,ix2,ix3]*map3[ix1,ix2,ix3]
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef np.ndarray[np.float64_t,ndim=3] prod_real_diff(self, double[:,:,::1] map1, double[:,:,::1] map2, double[:,:,::1] map3, double[:,:,::1] map4):
        """Compute the difference of map1(x)map2(x) and map3(x)map4(x) (all real)."""
        cdef long ix1, ix2, ix3
        cdef np.ndarray[np.float64_t,ndim=3] out = np.empty((self.nx1,self.nx2,self.nx3),dtype=np.float64)
        assert self.nx1==map2.shape[0]
        assert self.nx2==map2.shape[1]
        assert self.nx3==map2.shape[2]

        for ix1 in prange(self.nx1, schedule='static', num_threads=self.nthreads, nogil=True):
            for ix2 in xrange(self.nx2):
                for ix3 in xrange(self.nx3):
                    out[ix1,ix2,ix3] = map1[ix1,ix2,ix3]*map2[ix1,ix2,ix3]-map3[ix1,ix2,ix3]*map4[ix1,ix2,ix3]
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
    cdef np.ndarray[np.float64_t,ndim=3] out = np.zeros((nx1,nx2,nx3),dtype=np.float64)
    cdef double[:] av_map = np.zeros(nz, dtype=np.float64)
    cdef double[:] av_mask = np.zeros(nz, dtype=np.float64)
    
    # Compute averages of data in each radial bin
    with nogil:
        for iz in prange(nz, schedule='static', num_threads=nthreads):
            for ix1 in xrange(nx1):
                for ix2 in xrange(nx2):
                    for ix3 in xrange(nx3):
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
    cdef np.ndarray[np.float64_t,ndim=3] out = np.zeros((nx1,nx2,nx3),dtype=np.float64)
    cdef double[:] av_map = np.zeros(nz, dtype=np.float64)
    cdef double[:] av_mask = np.zeros(nz, dtype=np.float64)
    
    # Compute averages of data in each radial bin
    with nogil:
        for iz in prange(nz, schedule='static', num_threads=nthreads):
            for ix1 in xrange(nx1):
                for ix2 in xrange(nx2):
                    for ix3 in xrange(nx3):
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
cpdef np.ndarray[np.float64_t,ndim=1] assemble_b3(double[:,:,:,:,::1] g_maps, long[:,::1] all_bins, int nthreads):
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
cpdef np.ndarray[np.float64_t,ndim=1] assemble_bshot(double[:,:,:,:,::1] gA_maps, double[:,:,:,:,::1] gB_maps, double[:,:,:,:,::1] gC_maps, long[:,::1] all_bins, int nthreads):
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
cpdef np.ndarray[np.float64_t,ndim=1] assemble_b1(double[:,:,:,:,::1] g_maps, double[:,:,:,:,::1] simg_maps, long[:,::1] all_bins, int nthreads):
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