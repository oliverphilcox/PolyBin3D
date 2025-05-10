### Code for ideal and unwindowed binned polyspectrum estimation in 3D. Author: Oliver Philcox (2023)
## This module contains the basic code

import numpy as np, os, time, scipy
from .cython import utils

class PolyBin3D():
    """Base class for PolyBin3D.
    
    Inputs:
    - boxsize: 3D box-size (with 1 or 3 components)
    - gridsize: Number of Fourier modes per dimension (with 1 or 3 components)
    - Pk (optional): Fiducial power spectrum monopole (including noise), in ordering {k, P_0(k)}. This is used for the ideal estimators and for creating synthetic realizations for the optimal estimators. If unset, unit power will be assumed.
    - boxcenter (optional): Center of the 3D box (with 1 or 3 components). Default: None ( = [0,0,0])
    - pixel_window (optional): Which voxel window function to deconvolve from the data. See self.pixel_windows for a list of options (including "none"). Default: "none"
    - backend (optional): Which backend to use to compute FFTs. Options: "mkl" [requires mkl_fft].   
    - nthreads (optional): How many CPUs to use for the CPU calculations. Default: maximum available.
    - sightline (optional): Whether to assume local or global line-of-sight. Options: "local" [relative to each pair], "global" [relative to z-axis]. Default: "global"
    """
    def __init__(self, boxsize, gridsize, Pk=None, boxcenter=None, pixel_window='none', backend='mkl', nthreads=None, sightline='global', real_fft=False):
        
        self.real_fft = real_fft
        print('remove real_fft!')
        
        # Load attributes
        self.backend = backend
        self.nthreads = nthreads
        self.sightline = sightline
        self.pixel_window = pixel_window
        if self.nthreads is None:
            self.nthreads = os.cpu_count()
        
        # Check and load boxsize
        if type(boxsize)==float:
            self.boxsize = np.asarray([boxsize,boxsize,boxsize])
        else:
            assert len(boxsize)==3, "Need to specify x, y, z boxsize"
            self.boxsize = np.asarray(boxsize)
        
        # Check and load box center
        if boxcenter is None:
            self.boxcenter = np.zeros(3)
        elif type(boxcenter)==float:
            self.boxcenter = np.asarray([boxcenter,boxcenter,boxcenter])
        else:
            assert len(boxcenter)==3, "Need to specify x, y, z boxcenter"
            self.boxcenter = np.asarray(boxcenter)
        
        # Check and load gridsize
        if type(gridsize)==int:
            self.gridsize = np.asarray([gridsize,gridsize,gridsize],dtype=int)
        else:
            assert len(gridsize)==3, "Need to specify n_x, n_y, n_z gridsize"
            self.gridsize = np.asarray(gridsize,dtype=int)

        assert self.sightline in ['local','global'], "Sight-line must be 'local' or 'global'"
        if self.sightline=='global':
            assert np.abs(self.boxcenter).sum()==0., "Global sightlines should only be used for simulation data!"
        if self.sightline=='local':
            assert np.abs(self.boxcenter).sum()!=0., "Local sightlines should only be used for lightcone data!"
        
        # Derived parameters
        self.volume = self.boxsize.prod()
        self.kF = 2.*np.pi/self.boxsize
        self.kNy = self.gridsize*self.kF*0.5 # Nyquist
        
        # Load fiducial spectra
        if (np.asarray(Pk)==None).all():
            k_tmp = np.arange(0.,np.max(self.kNy)*2,np.min(self.kF)/10.)
            self.Pfid = np.asarray([k_tmp, 1.+0.*k_tmp])
        else:
            assert len(Pk)==2, "Pk should contain k and P_0(k) columns"
            self.Pfid = np.asarray(Pk)
        self.kmin, self.kmax = np.min(self.Pfid[0,0]), np.max(self.Pfid[0,-1])

        # Compute the real-space coordinate grid
        if self.sightline=='local': # not used else!
            offset = self.boxcenter #+0.5*self.boxsize/self.gridsize # removing this for consistency with pypower
            r_arrs = [np.fft.fftshift(np.arange(-self.gridsize[i]//2,self.gridsize[i]//2))*self.boxsize[i]/self.gridsize[i]+offset[i] for i in range(3)]
            self.r_grids = np.meshgrid(*r_arrs,indexing='ij')
        print("\n# Dimensions: [%.2e, %.2e, %.2e] Mpc/h"%(self.boxsize[0],self.boxsize[1],self.boxsize[2]))
        print("# Center: [%.2e, %.2e, %.2e] Mpc/h"%(self.boxcenter[0],self.boxcenter[1],self.boxcenter[2]))
        print("# Line-of-sight: %s"%self.sightline)
        
        # Compute the Fourier-space coordinate grid
        if self.real_fft:
            k_arrs = [np.fft.fftshift(np.arange(-self.gridsize[i]//2,self.gridsize[i]//2))*self.kF[i] for i in range(2)]
            k_arrs += [np.arange(0,self.gridsize[2]//2+1)*self.kF[2]]
        else:
            k_arrs = [np.fft.fftshift(np.arange(-self.gridsize[i]//2,self.gridsize[i]//2))*self.kF[i] for i in range(3)]
        self.k_grids = np.meshgrid(*k_arrs,indexing='ij')
        self.modk_grid = np.sqrt(self.k_grids[0]**2+self.k_grids[1]**2+self.k_grids[2]**2)
        if self.real_fft:
            self.degeneracy_factor = np.asarray(1.+(self.k_grids[2]>0), dtype=np.float64)
        else:
            self.degeneracy_factor = None
        
        print("# Cartesian grid: [%d, %d, %d]"%(self.gridsize[0],self.gridsize[1],self.gridsize[2])) 
        print("# Fundamental frequency: [%.4f, %.4f, %.4f] h/Mpc"%(self.kF[0],self.kF[1],self.kF[2]))
        print("# Nyquist frequency: [%.3f, %.3f, %.3f] h/Mpc"%(self.kNy[0],self.kNy[1],self.kNy[2]))

        # Account for pixel window if necessary
        self.pixel_windows = ['none','cic','tsc','pcs','interlaced-cic','interlaced-tsc','interlaced-pcs']
        assert self.pixel_window in self.pixel_windows, "Unknown pixel window '%s' supplied!"%pixel_window
        print("# Pixel window: %s"%self.pixel_window)
        if self.pixel_window!='none':
            windows_1d = [self._pixel_window_1d(np.pi/self.gridsize[i]*k_arrs[i]/self.kF[i]) for i in range(3)]
            self.pixel_window_grid = np.asarray(np.meshgrid(*windows_1d,indexing='ij')).prod(axis=0)
        
        # Define angles polynomials [for generating grids and weighting]
        if self.sightline=='global':
            sight_vector = np.asarray([0.,0.,1.])
        else:
            sight_vector = self.boxcenter/np.sqrt(np.sum(self.boxcenter**2))
        self.muk_grid = np.zeros(self.modk_grid.shape)
        self.muk_grid[self.modk_grid!=0] = np.sum(sight_vector[:,None]*np.asarray(self.k_grids)[:,self.modk_grid!=0],axis=0)/self.modk_grid[self.modk_grid!=0]
        
        # Set-up the FFT calculations
        self._fft_setup()
        
        # Apply Pk to grid
        self.Pk0_grid = scipy.interpolate.interp1d(self.Pfid[0], self.Pfid[1], bounds_error=False, fill_value=0.)(self.modk_grid)
        
        # Invert Pk
        self.invPk0_grid = np.zeros(self.modk_grid.shape, dtype=np.float64)
        good_filter = (self.Pk0_grid!=0)&np.isfinite(self.Pk0_grid)
        self.invPk0_grid[good_filter] = 1./self.Pk0_grid[good_filter] 
        self.Pk0_grid[~np.isfinite(self.Pk0_grid)] = 0.
        # Keep zero mode
        self.invPk0_grid[self.modk_grid==0] = 1.
        self.Pk0_grid[self.modk_grid==0] = 1.
        
        # Counter for FFTs
        self.n_FFTs_forward = 0
        self.n_FFTs_reverse = 0
        
        # Define cython utility class
        self.integrator = utils.IntegrationUtils(self.modk_grid, self.real_fft, self.nthreads, degk=self.degeneracy_factor, muk=self.muk_grid)
        self.map_utils = utils.MapUtils(self.modk_grid, self.gridsize, self.nthreads, muk=self.muk_grid)
        
    # Basic FFTs
    def to_fourier(self, _input_rmap):
        """Transform from real- to Fourier-space."""
        self.n_FFTs_forward += 1
        assert type(_input_rmap[0,0,0])==np.float64

        if self.backend=='fftw':
            # Load-in grid
            np.copyto(self.fftw_in, _input_rmap)
            # Perform FFT
            self.fftw_plan(self.fftw_in,self.fftw_out)
            # Return output
            return self.fftw_out.copy()
        
        elif self.backend=='mkl':
            if self.real_fft:
                return self.mkl_fft.rfftn(_input_rmap, axes=(0,1,2))        
            else:
                return self.mkl_fft.fftn(_input_rmap, axes=(0,1,2))        

        elif self.backend=='jax':
            if self.real_fft: raise Exception()
            arr = self.np.fft.fftn(_input_rmap,axes=(-3,-2,-1))
            return arr
        
    def to_real(self, _input_kmap):
        """Transform from Fourier- to real-space."""
        self.n_FFTs_reverse += 1
        assert type(_input_kmap[0,0,0])==np.complex128

        if self.backend=='fftw':
            # Load-in grid
            np.copyto(self.fftw_out, _input_kmap)
            # Perform FFT
            self.ifftw_plan(self.fftw_out,self.fftw_in)
            # Return output
            if self.real_fft:
                return self.fftw_in.copy()
            else:
                return np.asarray(self.fftw_in.copy().real, order='C')
            
        elif self.backend=='mkl':
            if self.real_fft:
                return self.mkl_fft.irfftn(_input_kmap, axes=(0,1,2))
            else:
                return np.asarray(self.mkl_fft.ifftn(_input_kmap, axes=(0,1,2)).real, order='C')     

        elif self.backend=='jax':
            arr = self.np.fft.ifftn(_input_kmap,axes=(-3,-2,-1))
            return self.np.asarray(arr.real, order='C')

    def _pixel_window_1d(self, k):
        """Pixel window functions including first-order alias corrections (copied from nbodykit)"""
        if self.pixel_window == 'cic':
            s = np.sin(k)**2.
            v = (1.-2./3.*s)**0.5
        elif self.pixel_window == 'tsc':
            s = np.sin(k)**2.
            v = (1.-s+2./15*s**2.)**0.5
        elif self.pixel_window=='pcs':
            s = np.sin(k)**2.
            v = (1.-4./3.*s+2./5.*s**2-4./315.*s**3)**0.5
        elif self.pixel_window == 'interlaced-cic':
            s = np.sinc(k/np.pi)
            v = np.ones_like(s)
            v[s!=0.] = s[s!=0.]**2.
        elif self.pixel_window == 'interlaced-tsc':
            s = np.sinc(k/np.pi)
            v = s**3.
        elif self.pixel_window == 'interlaced-pcs':
            s = np.sinc(k/np.pi)
            v = s**4.
        else:
            raise Exception("Unknown window function '%s' supplied!"%self.pixel_window)
        return v

    def _fft_setup(self, verb=True):
        """Set-up the relevant FFT modules"""
        
        # Set up relevant FFT modules
        if self.backend=='fftw':
            import pyfftw

            # Set-up FFT arrays
            if self.real_fft:
                self.fftw_in  = pyfftw.empty_aligned(self.gridsize,dtype='float64')
                self.fftw_out = pyfftw.empty_aligned([self.gridsize[0],self.gridsize[1],self.gridsize[2]//2+1],dtype='complex128')
            else:
                self.fftw_in  = pyfftw.empty_aligned(self.gridsize,dtype='complex128')
                self.fftw_out = pyfftw.empty_aligned(self.gridsize,dtype='complex128')

            # Preallocate memory-aligned arrays.
            self.fftw_plan = pyfftw.FFTW(self.fftw_in, self.fftw_out, axes=(0, 1, 2), direction='FFTW_FORWARD', flags=('FFTW_MEASURE',), threads=self.nthreads)
            self.ifftw_plan = pyfftw.FFTW(self.fftw_out, self.fftw_in, axes=(0, 1, 2), direction='FFTW_BACKWARD', flags=('FFTW_MEASURE',), threads=self.nthreads)

            # Run a warm-up transform to gain wisdom
            if self.real_fft:
                self.fftw_in[:] = np.ones(self.gridsize,dtype='float64')
            else:
                self.fftw_in[:] = np.ones(self.gridsize,dtype='complex128')
            self.fftw_plan()
            self.ifftw_plan()
            
            self.np = np
            if verb: print('# Using FFTW backend')
        
        elif self.backend=='mkl':
            import mkl_fft
            self.mkl_fft = mkl_fft
            
            self.np = np
            if verb: print('# Using MKL backend')
            
        elif self.backend=='jax':
            import jax
            import jax.numpy as jnp
            jax.config.update("jax_enable_x64", False)
            self.np = jnp
            self.jax = jax
            if verb: print('# Using JAX backend')
            
        else:
            raise Exception("Only 'fftw' and 'jax' backends are currently implemented!")

    # Gaussian random field routines
    def generate_data(self, seed=None, Pk_input=[], output_type='real', include_pixel_window=True):
        """
        Generate a Gaussian periodic map with a given set of P(k) multipoles (or the fiducial power spectrum monopole, if unspecified). 
        The input Pk are expected to by in the form {k, P_0, [P_2], [P_4]}, where P_2, P_4 are optional.

        No mask is added at this stage (except for a pixelation window, unless include_pixel_window=False), and the output can be in real- or Fourier-space.
        """
        assert output_type in ['fourier','real'], "Valid output types are 'fourier' and 'real' only!"
        
        # Define seed
        if seed!=None:
            np.random.seed(seed)
            
        if self.real_fft:
            print('fix this!')
            # Generate the power spectrum on a complex grid then make real afterwards

            k_arrs = [np.fft.fftshift(np.arange(-self.gridsize[i]//2,self.gridsize[i]//2))*self.kF[i] for i in range(3)]
            k_grids = np.meshgrid(*k_arrs,indexing='ij')
            modk_grid_all = np.sqrt(k_grids[0]**2+k_grids[1]**2+k_grids[2]**2)     
            if self.sightline=='global':
                sight_vector = np.asarray([0.,0.,1.])
            else:
                sight_vector = self.boxcenter/np.sqrt(np.sum(self.boxcenter**2))
            muk_grid_all = np.zeros(modk_grid_all.shape)
            muk_grid_all[modk_grid_all!=0] = np.sum(sight_vector[:,None]*np.asarray(k_grids)[:,modk_grid_all!=0],axis=0)/modk_grid_all[modk_grid_all!=0]
                    
            # Define input power spectrum on the k-space grid
            if len(Pk_input)==0:
                Pk_grid = scipy.interpolate.interp1d(self.Pfid[0], self.Pfid[1], bounds_error=False, fill_value=0.)(modk_grid_all)
            else:
                assert len(np.asarray(Pk_input).shape)==2, "Pk should contain k and P_0, (and optionally P_2, P_4) columns"
                assert len(Pk_input) in [2,3,4], "Pk should contain k and P_0, (and optionally P_2, P_4) columns"
                
                Pk_grid = scipy.interpolate.interp1d(Pk_input[0], Pk_input[1], bounds_error=False, fill_value=0.)(modk_grid_all)
                if len(Pk_input)>2:
                    Pk_grid += scipy.special.legendre(2)(muk_grid_all)*scipy.interpolate.interp1d(Pk_input[0], Pk_input[2], bounds_error=False, fill_value=0.)(modk_grid_all)
                if len(Pk_input)>3:
                    Pk_grid += scipy.special.legendre(4)(muk_grid_all)*scipy.interpolate.interp1d(Pk_input[0], Pk_input[3], bounds_error=False, fill_value=0.)(modk_grid_all)
            
            # Generate random Gaussian maps with input P(k)
            rand_fourier = (np.random.randn(*self.gridsize)+1.0j*np.random.randn(*self.gridsize))*np.sqrt(Pk_grid)
            rand_fourier[modk_grid_all==0] = 0.
            
            # Add pixel window function to delta(k)
            if self.pixel_window!='none' and include_pixel_window:
                windows_1d = [self._pixel_window_1d(np.pi/self.gridsize[i]*k_arrs[i]/self.kF[i]) for i in range(3)]
                window_grid = np.asarray(np.meshgrid(*windows_1d,indexing='ij')).prod(axis=0)
                rand_fourier *= window_grid
                
            # Force map to be real and normalize
            self.real_fft = False
            self._fft_setup(verb=False)
            rand_real = self.to_real(rand_fourier)
            self.real_fft = True
            self._fft_setup(verb=False)
            rand_real *= self.gridsize.prod()/np.sqrt(self.volume)
        else:
            # Define input power spectrum on the k-space grid
            if len(Pk_input)==0:
                Pk_grid = self.Pk0_grid
            else:
                assert len(np.asarray(Pk_input).shape)==2, "Pk should contain k and P_0, (and optionally P_2, P_4) columns"
                assert len(Pk_input) in [2,3,4], "Pk should contain k and P_0, (and optionally P_2, P_4) columns"
                
                Pk_grid = scipy.interpolate.interp1d(Pk_input[0], Pk_input[1], bounds_error=False, fill_value=0.)(self.modk_grid)
                if len(Pk_input)>2:
                    Pk_grid += scipy.special.legendre(2)(self.muk_grid)*scipy.interpolate.interp1d(Pk_input[0], Pk_input[2], bounds_error=False, fill_value=0.)(self.modk_grid)
                if len(Pk_input)>3:
                    Pk_grid += scipy.special.legendre(4)(self.muk_grid)*scipy.interpolate.interp1d(Pk_input[0], Pk_input[3], bounds_error=False, fill_value=0.)(self.modk_grid)
            
            # Generate random Gaussian maps with input P(k)
            rand_fourier = (np.random.randn(*self.gridsize)+1.0j*np.random.randn(*self.gridsize))*np.sqrt(Pk_grid)
            rand_fourier[self.modk_grid==0] = 0.
            
            # Add pixel window function to delta(k)
            if self.pixel_window!='none' and include_pixel_window:
                rand_fourier = self.map_utils.prod_fourier(rand_fourier, self.pixel_window_grid)
            
            # Force map to be real and normalize
            rand_real = self.to_real(rand_fourier)
            rand_real *= self.gridsize.prod()/np.sqrt(self.volume)
        
        if output_type=='real': return rand_real
        else: return self.to_fourier(rand_real)
        
    def applyAinv(self, input_data, invPk0_grid=None, input_type='real', output_type='real'):
        """Apply the exact inverse weighting A^{-1} to a map. This assumes a diagonal-in-ell C_l^{XY} weighting, as produced by generate_data.
        
        Note that the code has two input and output options: "fourier" or "real", to avoid unnecessary transforms.
        """
    
        assert input_type in ['fourier','real'], "Valid input types are 'fourier' and 'real' only!"
        assert output_type in ['fourier','real'], "Valid output types are 'fourier' and 'real' only!"

        if type(invPk0_grid)==type(None): 
            invPk0_grid = self.invPk0_grid

        # Transform to harmonic space, if necessary
        if input_type=='real': input_fourier = self.to_fourier(input_data)
        else: input_fourier = input_data.copy()
        
        # Divide by covariance
        Sinv_fourier = self.map_utils.prod_fourier(input_fourier, invPk0_grid)
        
        # Optionally divide by pixel window
        if self.pixel_window!='none':
            Sinv_fourier = self.map_utils.div_fourier(self.map_utils.div_fourier(Sinv_fourier, self.pixel_window_grid), self.pixel_window_grid)
        
        # Optionally return to map-space
        if output_type=='real': return self.to_real(Sinv_fourier)
        else: return Sinv_fourier