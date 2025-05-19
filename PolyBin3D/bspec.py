### Code for ideal and unwindowed binned polyspectrum estimation in 3D. Author: Oliver Philcox (2023)
## This module contains the bispectrum estimation code

import numpy as np
import multiprocessing as mp
import tqdm
from scipy.special import legendre
from scipy.interpolate import interp1d
import time
from .cython import utils

class BSpec():
    """Class containing the bispectrum estimators.
    
    Inputs:
    - base: PolyBin base class
    - k_bins: array of one-dimensional k bin edges (e.g., [0.01, 0.02, 0.03] would give two bins: [0.01,0.02] and [0.02,0.03]).
    - lmax: (optional) maximum Legendre multipole, default: 0.
    - mask: (optional) 3D mask, specifying the background density n(x,y,z).
    - applySinv: (optional) function which weights the data field. This is only used in unwindowed estimators.
    - k_bins_squeeze: (optional) array of one-dimensional squeezed bin edges. These can be used to compute squeezed triangles, giving a different kmax for short and long sides.
    - include_partial_triangles: (optional) whether to include triangles whose centers don't satisfy the triangle conditions, default: False
    - add_GIC: (optional) whether to include the global integral constraint, default: False
    - add_RIC: (optional) whether to include the radial integral constraint, default: False
    - mask_IC: (optional) unweighted 3D mask used to model integral constraints.
    - radial_bins_RIC: (optional) radial bins used to define n(z) [used for the RIC correction].
    - Pk_fid: (optional) fiducial power spectrum to use in ideal Fisher matrix and symmetry factors. If set to 'default', we use the same as the base class. Default: None.
    - mask_shot: (optional) triply weighted 3D field used to define the shot-noise numerator.
    - applySinv_transpose: (optional) transpose of function which weights the data field. This is only used in unwindowed estimators.
    """
    def __init__(self, base, k_bins, lmax=0, mask=None, applySinv=None, k_bins_squeeze=None, include_partial_triangles=False, add_GIC=False, mask_IC=None, add_RIC=False, radial_bins_RIC=[], Pk_fid=None, mask_shot=None, applySinv_transpose=None):
        # Read in attributes
        self.base = base
        self.applySinv = applySinv
        self.k_bins = np.asarray(k_bins)
        self.lmax = lmax
        self.Nk = len(k_bins)-1
        self.Nl = self.lmax//2+1
        self.include_partial_triangles = include_partial_triangles
        self.add_GIC = add_GIC
        self.add_RIC = add_RIC
        self.Pk_fid = Pk_fid
        self.applySinv_transpose = applySinv_transpose
        
        print("")
        assert self.lmax%2==0, "l-max must be even!"
        assert np.max(self.k_bins)<np.min(base.kNy), "k_max must be less than k_Nyquist!"
        assert np.min(self.k_bins)>=np.max(base.kF), "k_min must be at least the k_fundamental!"
        assert np.max(self.k_bins)<=base.Pfid[0][-1], "Fiducial power spectrum should extend up to k_max!"
        assert np.min(self.k_bins)>=base.Pfid[0][0], "Fiducial power spectrum should extend down to k_min!"
        print("Binning: %d bins in [%.3f, %.3f] h/Mpc"%(self.Nk,np.min(self.k_bins),np.max(self.k_bins)))

        # Add squeezed binning
        if type(k_bins_squeeze)!=type(None):
            self.k_bins_squeeze = np.asarray(k_bins_squeeze)
            self.Nk_squeeze = len(k_bins_squeeze)-1
            assert (self.k_bins_squeeze[:self.Nk+1]==self.k_bins).all(), "k_bins_squeeze must contain all non-squeezed bins!"
            assert np.max(self.k_bins_squeeze)<np.min(base.kNy), "k_max must be less than k_Nyquist!"
            assert np.min(self.k_bins_squeeze)>=np.max(base.kF), "k_min must be at least the k_fundamental!"
            assert np.max(self.k_bins_squeeze)<=base.Pfid[0][-1], "Fiducial power spectrum should extend up to k_max!"
            assert np.min(self.k_bins_squeeze)>=base.Pfid[0][0], "Fiducial power spectrum should extend down to k_min!"
            print("Squezed binning: %d bins in [%.3f, %.3f] h/Mpc"%(self.Nk_squeeze,np.min(self.k_bins_squeeze),np.max(self.k_bins_squeeze)))
        else:
            self.k_bins_squeeze = self.k_bins.copy()
            self.Nk_squeeze = self.Nk

        # Define binning
        all_bins = []
        for l in range(0, self.lmax+1, 2):
            for bin1 in range(self.Nk):
                for bin2 in range(bin1, self.Nk_squeeze):
                    for bin3 in range(bin2, self.Nk_squeeze):
                        if not self._check_bin(bin1, bin2, bin3): continue
                        all_bins.append((bin1,bin2,bin3,l))
        self.all_bins = np.asarray(all_bins, dtype=np.int64)
        
        # Compute number of bins
        self.N_bins = len(self.all_bins)
        self.N3 = self.N_bins//self.Nl
        print("l-max: %d"%(self.lmax))
        print("N_bins: %d"%self.N_bins)
        
        assert type(self.lmax)==int, "l-max must be an integer!"
        assert self.lmax<=4, "Only l-max<=4 is currently supported!"
        
        # Read-in mask
        if type(mask)==type(None):
            self.const_mask = True
            self.mask_mean = 1.
            self.cube_mask_mean = 1.
            assert not self.add_GIC, "Global integral constraint cannot be used without a mask!"
            assert not self.add_RIC, "Radial integral constraint cannot be used without a mask!"
        else:
            if type(mask)!=np.ndarray:
                mask = np.array(mask)
            self.mask = mask.astype(np.float64)
            if type(mask_IC)!=type(None):
                self.mask_IC = mask_IC.astype(np.float64)
            # Check if window is uniform
            if np.std(self.mask)<1e-12:
                self.const_mask = True
                self.mask_mean = np.mean(self.mask)
            else:
                self.const_mask = False
            # Compute the mean of the cubed mask
            self.cube_mask_mean = np.mean(self.mask**3)
        if self.const_mask:
            print("Mask: constant")
        else:
            print("Mask: spatially varying")
        
        # Read-in shot-noise mask
        if type(mask_shot)!=type(None):
            self.mask_shot = mask_shot.astype(np.float64)
        
        # Check S^-1 functions
        if self.applySinv is None:
            self.applySinv = self.base.applySinv_trivial
        else:
            if type(self.applySinv_transpose)!=type(lambda x: x):
                self.applySinv_transpose = self.applySinv
                print("## Caution: assuming S^-1 is Hermitian!")
        if self.applySinv_transpose is None:
            self.applySinv_transpose = self.base.applySinv_trivial
        
        # Check integral constraints
        if self.add_GIC:
            print("Accounting for global integral constraint")
            assert hasattr(self,'mask_IC'), "Need to supply mask_IC!"
        if self.add_RIC:
            assert not self.add_GIC, "Radial integral constraint imposes global integral constraint automatically!"
            assert hasattr(self,'mask_IC'), "Need to supply mask_IC!"
            assert len(radial_bins_RIC)>0, "Radial bins need to be supplied for radial integral constraint!"
            print("Accounting for radial integral constraint acrosss %d bins"%(len(radial_bins_RIC)-1))
            self.base.modr_grid = np.sqrt(self.base.r_grids[0]**2.+self.base.r_grids[1]**2.+self.base.r_grids[2]**2.)
            self.radial_bins_RIC = radial_bins_RIC
          
        # Define fiducial power spectrum
        if self.Pk_fid!='default':
            if (np.asarray(self.Pk_fid)==None).all():
                k_tmp = np.arange(np.min(self.base.kF)/10.,np.max(self.base.kNy)*2,np.min(self.base.kF)/10.)
                Pfid = np.asarray([k_tmp, 1.+0.*k_tmp])
            else:
                assert len(self.Pk_fid)==2, "Pk should contain k and P_0(k) columns"
                Pfid = np.asarray(self.Pk_fid)
            
            # Apply Pk to grid
            Pk0_grid = interp1d(Pfid[0], Pfid[1], bounds_error=False, fill_value=0.)(self.base.modk_grid)
            
            # Invert Pk
            self.invPk0 = np.zeros(Pk0_grid.shape, dtype=np.float64)
            self.invPk0[Pk0_grid!=0] = 1./Pk0_grid[Pk0_grid!=0] 
            del Pk0_grid
            
        else:
            self.invPk0 = self.base.invPk0_grid
                                                 
        # Define spherical harmonics in real-space [for computing bispectrum multipoles]
        if self.base.sightline=='local' and self.lmax>0:
            print("Generating spherical harmonics")
            self.Ylm_real = self.base.utils.compute_real_harmonics(np.asarray(self.base.r_grids), self.lmax, False, self.base.nthreads)
        else:
            self.Ylm_real = None
        
        # Define spherical harmonics in Fourier-space [needed for normalizations]
        if self.lmax>0:
            self.Ylm_fourier = self.base.utils.compute_real_harmonics(np.asarray(self.base.k_grids), self.lmax, False, self.base.nthreads)
        else:
            self.Ylm_fourier = None
            
    def _check_bin(self, bin1, bin2, bin3):
        """Return one if modes in the bin satisfy the triangle conditions, or zero else.

        This is used either for all triangles in the bin, or just the center of the bin.
        """
        if self.include_partial_triangles:
            # Maximum possible k3
            k3_hi = self.k_bins_squeeze[bin1+1]+self.k_bins_squeeze[bin2+1]
            # Minimum possible k3
            if bin1>bin2:
                k3_lo = self.k_bins_squeeze[bin1]-self.k_bins_squeeze[bin2+1]
            elif bin1==bin2:
                k3_lo = 0.
            else:
                k3_lo = self.k_bins_squeeze[bin2]-self.k_bins_squeeze[bin1+1]
            # Test bin, checking for numerical overlaps
            if self.k_bins_squeeze[bin3]>=k3_hi-1e-10 or self.k_bins_squeeze[bin3+1]<=k3_lo+1e-10:
                return 0
            else:
                return 1
        else:
            # Test center of bin
            k1 = 0.5*(self.k_bins[bin1]+self.k_bins[bin1+1])
            k2 = 0.5*(self.k_bins_squeeze[bin2]+self.k_bins_squeeze[bin2+1])
            k3 = 0.5*(self.k_bins_squeeze[bin3]+self.k_bins_squeeze[bin3+1])
            if k3<abs(k1-k2) or k3>k1+k2:
                return 0
            else:
                return 1
    
    def get_ks(self):
        """Return an array with the central k1, k2, k3 values for each bispectrum bin.
        """
        k1s, k2s, k3s = [],[],[]
        for bin1 in range(self.Nk):
            k1 = 0.5*(self.k_bins[bin1]+self.k_bins[bin1+1])
            for bin2 in range(bin1, self.Nk_squeeze):
                k2 = 0.5*(self.k_bins_squeeze[bin2]+self.k_bins_squeeze[bin2+1])
                for bin3 in range(bin2, self.Nk_squeeze):
                    k3 = 0.5*(self.k_bins_squeeze[bin3]+self.k_bins_squeeze[bin3+1])
            
                    # skip bins outside the triangle conditions
                    if not self._check_bin(bin1,bin2,bin3): continue
                    
                    # Add to output array
                    k1s.append(k1)
                    k2s.append(k2)
                    k3s.append(k3)
        
        return np.asarray([k1s,k2s,k3s])
    
    def _compute_symmetry_factor(self):
        """
        Compute symmetry factor giving the degeneracy of each bin. For l>0 this is computed using FFTs.
        """
        print("Computing degeneracy factor")
        self.sym_factor = []

        # First compute ell=0 elements
        for bin1 in range(self.Nk):
            for bin2 in range(bin1,self.Nk_squeeze):
                for bin3 in range(bin2,self.Nk_squeeze):
                    if not self._check_bin(bin1,bin2,bin3): continue

                    if bin1==bin2 and bin1==bin3 and bin1==bin3:
                        self.sym_factor.append(6.)

                    elif bin1==bin2 or bin1==bin3 or bin2==bin3:
                        self.sym_factor.append(2.)

                    else:
                        self.sym_factor.append(1.)

        # Now compute ell>0 with FFTs
        if self.lmax>0:
            
            # Define discrete binning functions
            bins = self.base.real_zeros(self.Nk_squeeze, numpy=True)
            for b in range(self.Nk_squeeze):
                bins[b] = self.base.to_real(self.base.map_utils.fourier_filter(self.invPk0.astype(np.complex128), 0, self.k_bins_squeeze[b], self.k_bins_squeeze[b+1]))
            
            for l in range(2,self.lmax+1,2):

                # Define Legendre-weighted binning functions
                bin23_l = self.base.real_zeros(self.Nk_squeeze, numpy=True)
                for b in range(self.Nk_squeeze):
                    for lm_ind in range(len(self.Ylm_fourier[l])): 
                        bin23_l[b] += self.base.to_real(self.base.map_utils.prod_fourier_filter(self.invPk0.astype(np.complex128), self.Ylm_fourier[l][lm_ind], self.k_bins_squeeze[b], self.k_bins_squeeze[b+1]))**2.
                    
                # Iterate over bins
                for bin1 in range(self.Nk):
                    for bin2 in range(bin1,self.Nk_squeeze):
                        for bin3 in range(bin2,self.Nk_squeeze):
                            if not self._check_bin(bin1,bin2,bin3): continue

                            if bin1!=bin2 and bin2!=bin3:
                                self.sym_factor.append(1.)

                            elif bin1==bin2 and bin2!=bin3:
                                self.sym_factor.append(2.)

                            elif bin1!=bin2 and bin2==bin3:
                                self.sym_factor.append(1.+self.base.map_utils.sum_pair(bins[bin1],bin23_l[bin3])/self.base.map_utils.sum_triplet(bins[bin1],bins[bin3],bins[bin3]))

                            elif bin1==bin2 and bin1==bin3:
                                self.sym_factor.append(2.*(1.+2.*self.base.map_utils.sum_pair(bins[bin3],bin23_l[bin3])/self.base.map_utils.sum_triplet(bins[bin3],bins[bin3],bins[bin3])))

        self.sym_factor = np.asarray(self.sym_factor)
        assert len(self.sym_factor)==self.N_bins
        
    def _process_sim(self, sim, input_type='real'):
        """
        Process a single input simulation. This is used for the linear term of the bispectrum estimator.

        We return g_{b,l}(r) maps for this simulation.
        """

        if self.base.pixel_window!='none':
            if input_type=='real':
                sim_fourier = self.base.to_fourier(sim)
            else:
                sim_fourier = sim.copy()
            
            sim_fourier /= self.base.pixel_window_grid

            # Apply S^-1 to simulation and transform to Fourier space
            Sinv_sim_fourier = self.applySinv(sim_fourier, input_type='fourier', output_type='fourier')
        else:
            # Apply S^-1 to data and transform to Fourier space
            Sinv_sim_fourier = self.applySinv(sim, input_type=input_type, output_type='fourier')
            
        # Store the real-space map if necessary
        if (self.base.sightline=='local' and self.lmax>0):
            Sinv_sim_real = self.base.to_real(Sinv_sim_fourier)

        # Compute g_{b,l} maps
        g_bl_maps = self.base.real_zeros((self.Nl,self.Nk_squeeze), numpy=True)
        for l in range(0,self.lmax+1,2):

            if l==0:
                # Compute monopole
                for b in range(self.Nk_squeeze):
                    g_bl_maps[0,b] = self.base.to_real(self.base.map_utils.fourier_filter(Sinv_sim_fourier, 0, self.k_bins_squeeze[b], self.k_bins_squeeze[b+1]))
                
            elif self.base.sightline=='global':
                # Compute higher multipoles, adding L_l(k.n) factor
                for b in range(self.Nk_squeeze):
                    g_bl_maps[l//2,b] = self.base.to_real(self.base.map_utils.fourier_filter(Sinv_sim_fourier, l, self.k_bins_squeeze[b], self.k_bins_squeeze[b+1]))
    
            else:
                # Compute Legendre map using spherical harmonics
                leg_map = self.base.apply_fourier_harmonics(Sinv_sim_real, self.Ylm_real[l], self.Ylm_fourier[l])
                for b in range(self.Nk_squeeze):
                    g_bl_maps[l//2,b] = self.base.to_real(self.base.map_utils.fourier_filter(leg_map, 0, self.k_bins_squeeze[b], self.k_bins_squeeze[b+1]))
        
        return g_bl_maps

    def load_sims(self, load_sim, N_sims, verb=False, input_type='real', preload=True):
        """
        Load in and preprocess N_sim Monte Carlo simulations used in the linear term of the bispectrum estimator. 
        These can alternatively be generated with a fiducial spectrum using the generate_sims script.

        The input is a function which loads the simulations in real- or Fourier-space given an index (0 to N_sims-1).

        If preload=False, the simulation products will not be stored in memory, but instead accessed when necessary. This greatly reduces memory usage, but is less CPU efficient if many datasets are analyzed together.
        """
        
        self.N_it = N_sims
        print("Using %d Monte Carlo simulations"%self.N_it)
        
        if preload:
            self.preload = True

            # Define lists of maps
            self.g_bl_maps = self.base.real_zeros((self.N_it,self.Nl,self.Nk_squeeze), numpy=True)
        
            # Iterate over simulations and preprocess appropriately
            for ii in range(self.N_it):
                if verb: print("Loading bias simulation %d of %d"%(ii+1,self.N_it))    
                this_sim = load_sim(ii)

                # Process simulation
                self.g_bl_maps[ii] = self._process_sim(this_sim, input_type=input_type)

        else:
            self.preload = False
            if verb: print("No preloading; simulations will be loaded and accessed at runtime.")
            
            # Simply save iterator and continue (simulations will be processed in serial later) 
            self.load_sim_data = lambda ii: self._process_sim(load_sim(ii), input_type=input_type)
           
    def generate_sims(self, N_sim, Pk_input=[], verb=False, preload=True):
        """
        Generate N_sim Monte Carlo simulations used in the linear term of the bispectrum generator. 
        These are pure GRFs simulations, generated with the pointing matrix and pixel-window-function, as well as the power spectrum defined in the base class (or Pk_input).
        
        If preload=True, we create N_it such simulations and store the relevant transformations into memory.
        If preload=False, we store only the function used to generate the sims, which will be processed later. This is cheaper on memory, but less CPU efficient if many datasets are analyzed together.
        
        We can alternatively load custom simulations using the load_sims script.
        """
        
        self.N_it = N_sim
        print("Using %d Monte Carlo simulations"%self.N_it)
        
        def load_sim_data(ii, Pk_input):
            """Generate a single GRF simulation and produce g_bl maps"""

            # Generate simulation and Fourier transform
            if self.const_mask:
                sim_fourier = self.mask_mean*self.base.generate_data(int(1e5)+ii, Pk_input=Pk_input, output_type='fourier')
                return self._process_sim(sim_fourier, input_type='fourier')
            else:
                # Generate base simulation
                sim_real = self.apply_pointing(self.base.generate_data(int(1e5)+ii, Pk_input=Pk_input, output_type='real', include_pixel_window=False))

                # Apply pixel window and compute g_{bl}(r) maps
                if self.base.pixel_window!='none':
                    sim_fourier = self.base.to_fourier(sim_real)*self.base.pixel_window_grid
                    return self._process_sim(sim_fourier, input_type='fourier')
                else:
                    return self._process_sim(sim_real, input_type='real')

        if preload:
            self.preload = True

            # Define list of maps
            self.g_bl_maps = self.base.real_zeros((self.N_it,self.Nl,self.Nk_squeeze), numpy=True)

            # Iterate over simulations and preprocess appropriately
            for ii in range(self.N_it):
                if verb: print("Generating bias simulation %d of %d"%(ii+1,self.N_it))
                self.g_bl_maps[ii] = load_sim_data(ii, Pk_input)

        else:
            self.preload = False
            if verb: print("No preloading; simulations will be loaded and accessed at runtime.")

            # Simply save iterator and continue (simulations will be processed in serial later) 
            self.load_sim_data = lambda ii: load_sim_data(ii, Pk_input)
    
    def apply_pointing(self, map_real, transpose=False):
        """Apply the pointing matrix to a map. This multiplies by the mask and optionally includes IC effects."""
        if self.add_RIC:
            if transpose:
                return utils.pointing_RIC_transpose(map_real, self.mask, self.mask_IC, self.radial_bins_RIC, self.base.modr_grid, self.base.nthreads)
            else:
                return utils.pointing_RIC(map_real, self.mask, self.mask_IC, self.radial_bins_RIC, self.base.modr_grid, self.base.nthreads)
        elif self.add_GIC:
            if transpose:
                return utils.pointing_GIC_transpose(map_real, self.mask, self.mask_IC, self.base.nthreads)
            else:
                return utils.pointing_GIC(map_real, self.mask, self.mask_IC, self.base.nthreads)
        else:
            out = self.base.map_utils.prod_real(map_real, self.mask)
            return out
    
    ### OPTIMAL ESTIMATOR
    def Bk_numerator(self, data, include_linear_term=True, verb=False):
        """
        Compute the numerator of the unwindowed bispectrum estimator, using the custom weighting function S^-1.

        We can optionally include a linear term to reduce the large-scale variance.
        """
        if not self.const_mask:
            assert data.shape == self.mask.shape, "Data must have same shape as the mask!"
        
        # Send to wrapper
        return self._compute_bk_numerator(data, include_linear_term=include_linear_term, filtering='Sinv', verb=verb)

    def compute_fisher_contribution(self, seed, verb=False):
        """
        This computes the contribution to the Fisher matrix from a single pair of GRF simulations, created internally.
        """
        return self._compute_fisher(seed, verb=verb)
    
    def compute_covariance_contribution(self, seed, Pk_cov, verb=False, mask_shot_2pt=None):
        """This computes the contribution to the Gaussian covariance matrix from a single pair of GRF simulations, created internally.
        
        This requires an input theory power spectrum, Pk_cov (without shot-noise). We can feed in a (two-point) shot-noise mask."""
        
        if (not self.const_mask) and (mask_shot_2pt is None):
            assert self.mask_shot is not None, "Must supply a shot-noise mask to compute the shot-noise contribution!"
            print("## Caution: assuming the same shot-noise mask for the power spectrum and bispectrum!")
            self.mask_shot_2pt = self.mask_shot
        else:
            self.mask_shot_2pt = mask_shot_2pt.astype(np.float64)
            
        assert len(Pk_cov)!=0, "Must provide fiducial power spectrum if computing covariance!"
        assert len(np.asarray(Pk_cov).shape)==2, "Pk should contain k and P_0 (and optionally P_2, P_4) columns"
        assert len(Pk_cov) in [2,3,4], "Pk should contain k and P_0 (and optionally P_2, P_4) columns"
        if (len(Pk_cov)-2)*2>self.lmax and self.base.sightline=='local':
            print("Regenerating spherical harmonics for lmax=%d"%((len(Pk_cov)-2)*2))
            self.Ylm_fourier = self.base.utils.compute_real_harmonics(np.asarray(self.base.k_grids), (len(Pk_cov)-2)*2, False, self.base.nthreads)
            self.Ylm_real = self.base.utils.compute_real_harmonics(np.asarray(self.base.r_grids), (len(Pk_cov)-2)*2, False, self.base.nthreads)
            # Copy to jax if necessary
            if self.base.backend=='jax':
                self.Ylm_fourier = {k: self.base.np.array(self.Ylm_fourier[k]) for k in self.Ylm_fourier.keys()}
                self.Ylm_real = {k: self.base.np.array(self.Ylm_real[k]) for k in self.Ylm_real.keys()}
            
        return self._compute_fisher(seed, verb=verb, compute_cov=True, Pk_cov=Pk_cov)
    
    def _compute_fisher(self, seed, compute_cov=False, Pk_cov=[], verb=False):
        """Internal function to compute the contribution to the Fisher matrix or covariance."""
 
        # Compute symmetry factor, if not already present
        if not hasattr(self, 'sym_factor'):
            self._compute_symmetry_factor()
        
        # Initialize output
        fish = np.zeros((self.N_bins,self.N_bins))
        
        # Precompute power spectrum fields
        if compute_cov:
            Pk_grid = {}
            if self.base.sightline=='global':
                Pk_grid[0] = interp1d(Pk_cov[0], Pk_cov[1], bounds_error=False, fill_value=0.)(self.base.modk_grid)
                if len(Pk_cov)>2:
                    Pk_grid[0] += legendre(2)(self.base.muk_grid)*interp1d(Pk_cov[0], Pk_cov[2], bounds_error=False, fill_value=0.)(self.base.modk_grid)
                if len(Pk_cov)>3:
                    Pk_grid[0] += legendre(4)(self.base.muk_grid)*interp1d(Pk_cov[0], Pk_cov[3], bounds_error=False, fill_value=0.)(self.base.modk_grid)   
            elif self.base.sightline=='local':
                Pk_grid[0] = interp1d(Pk_cov[0], Pk_cov[1], bounds_error=False, fill_value=0.)(self.base.modk_grid)
                if len(Pk_cov)>2:
                    Pk_grid[2] = interp1d(Pk_cov[0], Pk_cov[2], bounds_error=False, fill_value=0.)(self.base.modk_grid)
                if len(Pk_cov)>3:
                    Pk_grid[4] = interp1d(Pk_cov[0], Pk_cov[3], bounds_error=False, fill_value=0.)(self.base.modk_grid)    
        
        def apply_filter(input_map):
            """Apply S^-1 P or S^-1 C S^-dagger to a map. Note that the input is either in Fourier-space if const_mask=True else in real-space."""
            if not compute_cov:
                if self.const_mask:
                    return self.mask_mean*self.applySinv(input_map, input_type='fourier', output_type='fourier')
                else:
                    return self.applySinv(self.apply_pointing(input_map, transpose=False), input_type='real', output_type='fourier')
            else:
                # Apply S^-1 [P Cov P^dagger + N] S^-dagger in order to compute covariances
                if self.const_mask:
                    Sid_map = self.applySinv_transpose(input_map, input_type='fourier', output_type='fourier')
                    CSid_map = self.mask_mean**2*self.base.apply_xi(Sid_map, Ylm_real=self.Ylm_real, Ylm_fourier=self.Ylm_fourier, Pk_grid=Pk_grid, output_type='fourier') + self.mask_mean*Sid_map
                    return self.applySinv(CSid_map, input_type='fourier', output_type='fourier')
                else:
                    Sid_map = self.applySinv_transpose(input_map, input_type='real', output_type='real')
                    PdSid_map_real = self.apply_pointing(Sid_map, transpose=True)
                    PdSid_map = self.base.to_fourier(PdSid_map_real)
                    CPdSid_map = self.base.apply_xi(PdSid_map, PdSid_map_real, Ylm_real=self.Ylm_real, Ylm_fourier=self.Ylm_fourier, Pk_grid=Pk_grid, output_type='real')
                    CovSid_map = self.apply_pointing(CPdSid_map, transpose=False) + self.base.map_utils.prod_real(Sid_map, self.mask_shot_2pt)
                    return self.applySinv(CovSid_map, input_type='real', output_type='fourier')
                
        # Define Q map code
        def compute_Q(weighting, a_fourier1, a_real1, a_fourier2, a_real2):
            """
            Assemble and return the Q [ = partial_alpha zeta_ijk (weighting a)_j (weighting a)_k] maps in Fourier-space, given a weighting scheme.

            The outputs are Q_alpha arrays."""

            # Filter maps appropriately
            if weighting=='Sinv':
                # Compute S^-1 P a or S^-1 C S^-dagger a
                if self.const_mask:
                    weighted_map_fourier1 = apply_filter(a_fourier1)
                    weighted_map_fourier2 = apply_filter(a_fourier2)
                else:
                    weighted_map_fourier1 = apply_filter(a_real1)
                    weighted_map_fourier2 = apply_filter(a_real2)
            else:
                # Compute A^-1 a
                weighted_map_fourier1 = self.base.applyAinv(a_fourier1, input_type='fourier', output_type='fourier')
                weighted_map_fourier2 = self.base.applyAinv(a_fourier2, input_type='fourier', output_type='fourier')
            
            # Define real-space map where necessary
            if self.base.sightline=='local' and self.lmax>0:
                weighted_map_real1 = self.base.to_real(weighted_map_fourier1)
                weighted_map_real2 = self.base.to_real(weighted_map_fourier2)
            
            # Compute g_{b,0} maps
            if verb: print("Computing g_{b,0}(r) maps")
            g_b0_maps1 = self.base.real_zeros(self.Nk_squeeze, numpy=True)
            g_b0_maps2 = self.base.real_zeros(self.Nk_squeeze, numpy=True)
            for b in range(self.Nk_squeeze):
                g_b0_maps1[b] = self.base.to_real(self.base.map_utils.fourier_filter(weighted_map_fourier1, 0, self.k_bins_squeeze[b], self.k_bins_squeeze[b+1]))
                g_b0_maps2[b] = self.base.to_real(self.base.map_utils.fourier_filter(weighted_map_fourier2, 0, self.k_bins_squeeze[b], self.k_bins_squeeze[b+1]))
            
            # Define Legendre L_ell(k.n) weighting for all ell
            if self.base.sightline=='local' and self.lmax>0:
                leg_maps1 = self.base.complex_zeros(self.Nl-1, numpy=True)
                leg_maps2 = self.base.complex_zeros(self.Nl-1, numpy=True)
                for ell in range(2, self.lmax+1, 2):
                    leg_maps1[ell//2-1] = self.base.apply_fourier_harmonics(weighted_map_real1, self.Ylm_real[ell], self.Ylm_fourier[ell])
                    leg_maps2[ell//2-1] = self.base.apply_fourier_harmonics(weighted_map_real2, self.Ylm_real[ell], self.Ylm_fourier[ell])
            
            # Iterate over quadratic pairs of bins, starting with longer side
            for binB in range(self.Nk_squeeze):
                if verb: print("Computing matrix for k-bin %d of %d"%(binB+1,self.Nk_squeeze))

                # Compute all g_bB_l maps
                if self.lmax>0:
                    g_bBl_maps1 = self.base.real_zeros(self.Nl-1, numpy=True)
                    g_bBl_maps2 = self.base.real_zeros(self.Nl-1, numpy=True)
                    for ell in range(2,self.lmax+1,2):
                        if self.base.sightline=='global':
                            g_bBl_maps1[ell//2-1] = self.base.to_real(self.base.map_utils.fourier_filter(weighted_map_fourier1, ell, self.k_bins_squeeze[binB], self.k_bins_squeeze[binB+1]))
                            g_bBl_maps2[ell//2-1] = self.base.to_real(self.base.map_utils.fourier_filter(weighted_map_fourier2, ell, self.k_bins_squeeze[binB], self.k_bins_squeeze[binB+1]))
                        else:
                            g_bBl_maps1[ell//2-1] = self.base.to_real(self.base.map_utils.fourier_filter(leg_maps1[ell//2-1], 0, self.k_bins_squeeze[binB], self.k_bins_squeeze[binB+1]))
                            g_bBl_maps2[ell//2-1] = self.base.to_real(self.base.map_utils.fourier_filter(leg_maps2[ell//2-1], 0, self.k_bins_squeeze[binB], self.k_bins_squeeze[binB+1]))

                # Iterate over shorter side
                for binA in range(binB+1):

                    if weighting=='Sinv':
                    
                        # Find which elements of the Q3 matrix this pair is used for (with ordering)
                        these_ind = np.where((self.all_bins[:,2]==binB)&(self.all_bins[:,0]==binA))[0]
                        if len(these_ind)==0: 
                            continue
                        
                        for l in range(0, self.lmax+1, 2):
                            # Compute FT[g_{0, bA},g_{ell, bB}] 
                            if l==0:
                                ft_ABl = self.base.to_fourier(self.base.map_utils.prod_real_diff(g_b0_maps1[binA],g_b0_maps1[binB],g_b0_maps2[binA],g_b0_maps2[binB]))
                            else:
                                ft_ABl = self.base.to_fourier(self.base.map_utils.prod_real_diff(g_b0_maps1[binA],g_bBl_maps1[l//2-1],g_b0_maps2[binA],g_bBl_maps2[l//2-1]))
                             
                            # Iterate over these elements and add to the output arrays
                            for ii in these_ind[self.all_bins[these_ind,3]==l]:
                                fish[ii] = self.base.integrator.integrate_row(ft_ABl, Q_Ainv, self.k_bins_squeeze[self.all_bins[ii,1]], self.k_bins_squeeze[self.all_bins[ii,1]+1])
                                
                    elif weighting=='Ainv':

                        # Find which elements of the Q3 matrix this pair is used for (with ordering)
                        these_ind1 = np.where((self.all_bins[:,0]==binA)&(self.all_bins[:,1]==binB))[0]
                        these_ind2 = np.where((self.all_bins[:,1]==binA)&(self.all_bins[:,2]==binB))[0]
                        these_ind3 = np.where((self.all_bins[:,2]==binB)&(self.all_bins[:,0]==binA))[0]
                        if len(these_ind1)+len(these_ind2)+len(these_ind3)==0: 
                            continue
                    
                        # Compute FT[g_{0, bA},g_{ell, bB}] 
                        ft_ABl = np.empty((self.Nl, self.base.modk_grid.shape[0], self.base.modk_grid.shape[1], self.base.modk_grid.shape[2]), dtype=np.complex128)
                        ft_ABl[0] = self.base.to_fourier(self.base.map_utils.prod_real_diff(g_b0_maps1[binA],g_b0_maps1[binB],g_b0_maps2[binA],g_b0_maps2[binB]))
                        # Repeat for higher ell
                        for l in range(2, self.lmax+1, 2):
                            ft_ABl[l//2] = self.base.to_fourier(self.base.map_utils.prod_real_diff(g_b0_maps1[binA],g_bBl_maps1[l//2-1],g_b0_maps2[binA],g_bBl_maps2[l//2-1]))
                    
                        def add_Q_element(binC_index, these_ind, largest_l=True):
                            # Iterate over these elements and add to the output arrays
                            for ii in these_ind:
                                binC = self.all_bins[ii,binC_index]
                                l = self.all_bins[ii,3]

                                # Monopole
                                if l==0 or (l>0 and not largest_l):
                                    Q_Ainv[ii] += self.base.map_utils.fourier_filter(ft_ABl[l//2], 0, self.k_bins_squeeze[binC], self.k_bins_squeeze[binC+1])
                                else:
                                    # Apply legendre to external leg
                                    if self.base.sightline=='global':
                                        # Work in Fourier-space for global line-of-sight
                                        Q_Ainv[ii] += self.base.map_utils.fourier_filter(ft_ABl[0],  l, self.k_bins_squeeze[binC], self.k_bins_squeeze[binC+1])
                                    else:
                                        binC_ftAB = self.base.map_utils.fourier_filter(ft_ABl[0], 0, self.k_bins_squeeze[binC], self.k_bins_squeeze[binC+1])
                                        # Work in real-space for Yamamoto line-of-sight
                                        real_map = self.base.apply_real_harmonics(binC_ftAB, self.Ylm_real[l], self.Ylm_fourier[l])
                                        if self.const_mask:
                                            Q_Ainv[ii] += self.base.to_fourier(real_map)
                                        else:
                                            Q_Ainv[ii] += self.base.to_fourier(real_map)
                                            
                        add_Q_element(2, these_ind1, True) 
                        add_Q_element(0, these_ind2, False)
                        add_Q_element(1, these_ind3, False)
                        
        # Set up temporary array
        Q_Ainv = np.zeros((self.N_bins,self.base.modk_grid.shape[0],self.base.modk_grid.shape[1],self.base.modk_grid.shape[2]),dtype=np.complex128)
        if verb: print("Allocating %.2f GB of memory"%(Q_Ainv.nbytes/1024./1024./1024.))    

        # Compute a random realization with known power spectrum
        if verb: print("Generating GRFs")
        a_map_fourier1 = self.base.generate_data(seed=seed+int(1e7), output_type='fourier')
        a_map_fourier2 = self.base.generate_data(seed=seed+int(2e7), output_type='fourier')
        
        if not self.const_mask:
            a_map_real1 = self.base.to_real(a_map_fourier1)
            a_map_real2 = self.base.to_real(a_map_fourier2)
        else:
            a_map_real1 = None
            a_map_real2 = None
        
        if verb: print("\n# Computing Q[A^-1.a] maps")
        compute_Q('Ainv', a_map_fourier1, a_map_real1, a_map_fourier2, a_map_real2)
        
        # Add S^-1.P weighting to A^-1 maps
        for i in range(self.N_bins):
            if self.const_mask:
                Q_Ainv[i] = apply_filter(Q_Ainv[i])
            else:
                Q_Ainv[i] = apply_filter(self.base.to_real(Q_Ainv[i]))
        
        # Compute Q maps and form the Fisher matrix
        if verb and compute_cov: print("\n# Computing Q[S^-1.C.S^-dagger.a] maps")
        elif verb: print("\n# Computing Q[S^-1.P.a] maps")
        compute_Q('Sinv', a_map_fourier1, a_map_real1, a_map_fourier2, a_map_real2)
        
        # Add normalization and return output
        fish = 1./2.*fish*self.base.volume/self.base.gridsize.prod()**2/np.outer(self.sym_factor,self.sym_factor)

        return fish
    
    def compute_fisher(self, N_it, verb=False):
        """
        Compute the Fisher matrix using N_it realizations. Since the calculation is already parallelized, this is run in serial.
        """
        # Initialize output
        fish = np.zeros((self.N_bins,self.N_bins))
        
        # Iterate over realizations in serial
        for seed in range(N_it):
            if seed%5==0: print("Computing Fisher contribution %d of %d"%(seed+1,N_it))
            fish += self.compute_fisher_contribution(seed, verb=verb)/N_it
            
        self.fish = fish
        self.inv_fish = np.linalg.inv(fish)
        
        return fish
    
    def compute_shot_contribution(self, seed, verb=False, data=None, data2=None, cubic_only=False):
        """
        This computes the shot-noise from a single pair of GRF simulations, created internally. 
        
        We can optionally drop the quadratic term (~ P(k)/nbar).
        
        The quadratic piece involves both the singly- and doubly-weighted data (data and data2). If unspecified, we set data2 = data.
        """
        if not cubic_only:
            assert data is not None, "Must supply data to compute the quadratic shot-noise contribution!"

            if data2 is None:
                print("## Caution: assuming uniform weights!")
                data2 = data

            if not self.const_mask:
                assert data.shape == self.mask.shape, "Data must have same shape as the mask!"

            # Check input data type and convert to float64 if necessary
            if not self.base.backend=='jax':
                assert type(data[0,0,0]) in [np.float32, np.float64], "Data must be of type float32 or float64!"
                assert type(data2[0,0,0]) in [np.float32, np.float64], "Data must be of type float32 or float64!"
            if type(data[0,0,0])==np.float32: 
                data = np.asarray(data, order='C', dtype=np.float64)
            if type(data2[0,0,0])==np.float32: 
                data2 = np.asarray(data2, order='C', dtype=np.float64)
        
        else:
            print("## Caution: Dropping quadratic (~ P(k)/n) term!")
            
        assert hasattr(self, 'mask_shot'), "Must supply triply-weighted mask to compute the shot-noise contribution!" 
        
        # Compute symmetry factor, if not already present
        if not hasattr(self, 'sym_factor'):
            self._compute_symmetry_factor()
        
        if not cubic_only:
        
            # First remove the pixel window function, if present
            if self.base.pixel_window!='none':
                data_fourier = self.base.to_fourier(data)
                data_fourier /= self.base.pixel_window_grid

                # Apply S^-1 to data and transform to Fourier space
                Sinv_data_fourier = self.applySinv(data_fourier, input_type='fourier', output_type='fourier')
            else:
                # Apply S^-1 to data and transform to Fourier space
                Sinv_data_fourier = self.applySinv(data, input_type='real', output_type='fourier')
            
            # Store the real-space map if necessary
            if (self.base.sightline=='local' and self.lmax>0):
                Sinv_data_real = self.base.to_real(Sinv_data_fourier)

            # Compute g_{b,l} maps for data
            g_bl_maps = self.base.real_zeros((self.Nl, self.Nk_squeeze), numpy=True)
            if verb: print("")
            for l in range(0, self.lmax+1, 2):

                if verb: print("Computing g_{b,%d}(r) maps"%l)

                if l==0:
                    # Compute monopole
                    for b in range(self.Nk_squeeze):
                        g_bl_maps[0,b] = self.base.to_real(self.base.map_utils.fourier_filter(Sinv_data_fourier, 0, self.k_bins_squeeze[b], self.k_bins_squeeze[b+1]))

                elif self.base.sightline=='global':
                    # Compute higher multipoles, adding L_l(k.n) factor
                    for b in range(self.Nk_squeeze):
                        g_bl_maps[l//2,b] = self.base.to_real(self.base.map_utils.fourier_filter(Sinv_data_fourier, l, self.k_bins_squeeze[b], self.k_bins_squeeze[b+1]))
        
                else:
                    # Compute Legendre map using spherical harmonics
                    leg_map = self.base.apply_fourier_harmonics(Sinv_data_real, self.Ylm_real[l], self.Ylm_fourier[l])
                    for b in range(self.Nk_squeeze):
                        g_bl_maps[l//2,b] = self.base.to_real(self.base.map_utils.fourier_filter(leg_map, 0, self.k_bins_squeeze[b], self.k_bins_squeeze[b+1]))

        # Define an inverse power spectrum to draw GRFs from
        PkA = [self.base.Pfid[0],1./self.base.Pfid[1]]
        invPkA_grid = self.base.Pk0_grid
        
        # Compute a pair of random realizations with known power spectrum
        a_map_fourier1 = self.base.generate_data(seed=seed+int(1e7), Pk_input=PkA, output_type='fourier')
        a_map_fourier2 = self.base.generate_data(seed=seed+int(2e7), Pk_input=PkA, output_type='fourier')
        
        # Compute S^-1 N_ijk (A^-1 a1)_j (A^-1 a2)_k (and associated terms) in Fourier-space
        Ainv_a1 = self.base.applyAinv(a_map_fourier1, invPk0_grid=invPkA_grid, input_type='fourier', output_type='real')
        Ainv_a2 = self.base.applyAinv(a_map_fourier2, invPk0_grid=invPkA_grid, input_type='fourier', output_type='real')
        Sinv_N_Ainv_a1_Ainv_a2 = self.applySinv(self.base.map_utils.prod_real(self.base.map_utils.prod_real(Ainv_a1, Ainv_a2), self.mask_shot), input_type='real', output_type='fourier')
        if not cubic_only:
            Sinv_d2_Ainv_a1 = self.applySinv(self.base.map_utils.prod_real(Ainv_a1, data2), input_type='real', output_type='fourier')
            Sinv_d2_Ainv_a2 = self.applySinv(self.base.map_utils.prod_real(Ainv_a2, data2), input_type='real', output_type='fourier')
        del Ainv_a1, Ainv_a2
        
        # Compute S^-1 a1, S^-1 a2
        Sinv_a1 = self.applySinv(a_map_fourier1, input_type='fourier', output_type='fourier')
        Sinv_a2 = self.applySinv(a_map_fourier2, input_type='fourier', output_type='fourier')
        
        # Store the real-space maps if necessary
        if self.lmax>0 and self.base.sightline=='local':
            Sinv_a_real1 = self.base.to_real(Sinv_a1)
            Sinv_a_real2 = self.base.to_real(Sinv_a2)
            Sinv_N_Ainv_a1_Ainv_a2_real = self.base.to_real(Sinv_N_Ainv_a1_Ainv_a2)
            if not cubic_only:
                Sinv_d2_Ainv_a1_real = self.base.to_real(Sinv_d2_Ainv_a1)
                Sinv_d2_Ainv_a2_real = self.base.to_real(Sinv_d2_Ainv_a2)
        del a_map_fourier1, a_map_fourier2
        
        # Compute g_{b,l} maps
        g_bl_mapsA, g_bl_mapsB, g_bl_mapsC = [self.base.real_zeros((self.Nl, self.Nk_squeeze), numpy=True) for _ in range(3)]
        if not cubic_only:
            g_bl_mapsD, g_bl_mapsE = [self.base.real_zeros((self.Nl, self.Nk_squeeze), numpy=True) for _ in range(2)]
        if verb: print("")
        
        for l in range(0, self.lmax+1, 2):

            if verb: print("Computing shot-noise g_{b,%d}(r) maps"%l)

            if l==0:
                # Compute monopole
                for b in range(self.Nk_squeeze):
                    g_bl_mapsA[0,b] = self.base.to_real(self.base.map_utils.fourier_filter(Sinv_a1, 0, self.k_bins_squeeze[b], self.k_bins_squeeze[b+1]))
                    g_bl_mapsB[0,b] = self.base.to_real(self.base.map_utils.fourier_filter(Sinv_a2, 0, self.k_bins_squeeze[b], self.k_bins_squeeze[b+1]))
                    g_bl_mapsC[0,b] = self.base.to_real(self.base.map_utils.fourier_filter(Sinv_N_Ainv_a1_Ainv_a2, 0, self.k_bins_squeeze[b], self.k_bins_squeeze[b+1]))
                    if not cubic_only:
                        g_bl_mapsD[0,b] = self.base.to_real(self.base.map_utils.fourier_filter(Sinv_d2_Ainv_a1, 0, self.k_bins_squeeze[b], self.k_bins_squeeze[b+1]))
                        g_bl_mapsE[0,b] = self.base.to_real(self.base.map_utils.fourier_filter(Sinv_d2_Ainv_a2, 0, self.k_bins_squeeze[b], self.k_bins_squeeze[b+1]))

            elif self.base.sightline=='global':
                # Compute higher multipoles, adding L_l(k.n) factor
                for b in range(self.Nk_squeeze):
                    g_bl_mapsA[l//2,b] = self.base.to_real(self.base.map_utils.fourier_filter(Sinv_a1, l, self.k_bins_squeeze[b], self.k_bins_squeeze[b+1]))
                    g_bl_mapsB[l//2,b] = self.base.to_real(self.base.map_utils.fourier_filter(Sinv_a2, l, self.k_bins_squeeze[b], self.k_bins_squeeze[b+1]))
                    g_bl_mapsC[l//2,b] = self.base.to_real(self.base.map_utils.fourier_filter(Sinv_N_Ainv_a1_Ainv_a2, l, self.k_bins_squeeze[b], self.k_bins_squeeze[b+1]))
                    if not cubic_only:
                        g_bl_mapsD[l//2,b] = self.base.to_real(self.base.map_utils.fourier_filter(Sinv_d2_Ainv_a1, l, self.k_bins_squeeze[b], self.k_bins_squeeze[b+1]))
                        g_bl_mapsE[l//2,b] = self.base.to_real(self.base.map_utils.fourier_filter(Sinv_d2_Ainv_a2, l, self.k_bins_squeeze[b], self.k_bins_squeeze[b+1]))
            else:
                def apply_legendre(input_real_map, output_array):
                    # Compute Legendre map using spherical harmonics
                    leg_map = self.base.apply_fourier_harmonics(input_real_map, self.Ylm_real[l], self.Ylm_fourier[l])
                    for b in range(self.Nk_squeeze):
                        output_array[l//2,b] = self.base.to_real(self.base.map_utils.fourier_filter(leg_map, 0, self.k_bins_squeeze[b], self.k_bins_squeeze[b+1]))
                    
                apply_legendre(Sinv_a_real1, g_bl_mapsA)
                apply_legendre(Sinv_a_real2, g_bl_mapsB)
                apply_legendre(Sinv_N_Ainv_a1_Ainv_a2_real, g_bl_mapsC)
                if not cubic_only:
                    apply_legendre(Sinv_d2_Ainv_a1_real, g_bl_mapsD)
                    apply_legendre(Sinv_d2_Ainv_a2_real, g_bl_mapsE)
                
        # Compute cubic numerator
        if verb: print("\nAssembling cubic shot-noise term")
        shot_num_cubic = self.base.utils.assemble_bshot(g_bl_mapsA, g_bl_mapsB, g_bl_mapsC, self.all_bins, self.base.nthreads)
        shot_num_cubic *= 1./6.*self.base.volume/self.base.gridsize.prod()/self.sym_factor
        
        # Compute quadratic numerator
        if not cubic_only:
            if verb: print("Assembling quadratic shot-noise term\n")
            shot_num_quadratic = 0.5*self.base.utils.assemble_bshot(g_bl_maps, g_bl_mapsA, g_bl_mapsD, self.all_bins, self.base.nthreads)
            shot_num_quadratic += 0.5*self.base.utils.assemble_bshot(g_bl_maps, g_bl_mapsB, g_bl_mapsE, self.all_bins, self.base.nthreads)
            shot_num_quadratic *= 1./2.*self.base.volume/self.base.gridsize.prod()/self.sym_factor
        
        # Return output
        if cubic_only:
            return shot_num_cubic
        else:
            return shot_num_quadratic - 2*shot_num_cubic

    def Bk_unwindowed(self, data, fish=[], include_linear_term=True, verb=False):
        """
        Compute the unwindowed bispectrum estimator, using the custom weighting function S^-1.

        Note that the Fisher matrix must be computed before this is run, or it can be supplied separately.
        
        We can optionally drop the linear term (at the cost of slightly enhanced large-scale variance).
        """
        
        # Compute inverse Fisher
        if len(fish)!=0:
            self.fish = fish
            self.inv_fish = np.linalg.inv(fish)
        
        if not hasattr(self,'inv_fish'):
            raise Exception("Need to compute Fisher matrix first!")
        
        # Compute numerator
        Bk_num = self.Bk_numerator(data, include_linear_term=include_linear_term, verb=verb)

        # Apply normalization and restructure
        Bk_out = np.matmul(self.inv_fish,Bk_num)
        
        # Create output dictionary of spectra
        Bk_dict = {}
        index = 0
        for l in range(0,self.lmax+1,2):
            Bk_dict['b%s'%l] = Bk_out[index:index+self.N3]
            index += self.N3

        return Bk_dict
    
    def _compute_bk_numerator(self, data, include_linear_term=True, filtering='ideal', verb=False):
        """Internal function to compute the bispectrum numerator. This is used by the Bk_numerator and Bk_numerator_ideal functions."""
        assert filtering in ['ideal','Sinv'], "Unknown filtering option supplied!"
        
        if (not hasattr(self, 'preload')) and include_linear_term:
            raise Exception("Need to generate or specify bias simulations!")

        # Check input data type and convert to float64 if necessary
        if not self.base.backend=='jax':
            assert type(data[0,0,0]) in [np.float32, np.float64], "Data must be of type float32 or float64!"
        if type(data[0,0,0])==np.float32: 
            data = np.asarray(data, order='C', dtype=np.float64)

        # Compute symmetry factor if necessary
        if not hasattr(self, 'sym_factor'):
            self._compute_symmetry_factor()
        
        # Apply filtering and transform to Fourier-space, optionally removing the pixel window function
        if self.base.pixel_window!='none':
            
            # Transform to Fourier-space
            data_fourier = self.base.map_utils.div_fourier(self.base.to_fourier(data), self.base.pixel_window_grid)
                        
            if filtering=='ideal':
                # Filter by 1/P_fid
                Sinv_data_fourier = self.base.map_utils.prod_fourier(data_fourier, self.invPk0)/self.cube_mask_mean**(1./3.)
            elif filtering=='Sinv':
                Sinv_data_fourier = self.applySinv(data_fourier, input_type='fourier', output_type='fourier')
        
        else:
            if filtering=='ideal':
                # Filter by 1/P_fid
                Sinv_data_fourier = self.base.map_utils.prod_fourier(self.base.to_fourier(data), self.invPk0)/self.cube_mask_mean**(1./3.)
            elif filtering=='Sinv':
                Sinv_data_fourier = self.applySinv(data, input_type='real', output_type='fourier')
         
        # Store the real-space map if necessary
        if (self.base.sightline=='local' and self.lmax>0):
            Sinv_data_real = self.base.to_real(Sinv_data_fourier)
        
        # Compute g_{b,l} maps
        g_bl_maps = self.base.real_zeros((self.Nl,self.Nk_squeeze), numpy=True)
        if verb: print("")
        for l in range(0, self.lmax+1, 2):

            if verb: print("Computing g_{b,%d}(r) maps"%l)

            if l==0:
                # Compute monopole
                for b in range(self.Nk_squeeze):
                    g_bl_maps[0,b] = self.base.to_real(self.base.map_utils.fourier_filter(Sinv_data_fourier, 0, self.k_bins_squeeze[b], self.k_bins_squeeze[b+1]))

            elif self.base.sightline=='global':
                # Compute higher multipoles, adding L_l(k.n) factor
                for b in range(self.Nk_squeeze):
                    g_bl_maps[l//2,b] = self.base.to_real(self.base.map_utils.fourier_filter(Sinv_data_fourier, l, self.k_bins_squeeze[b], self.k_bins_squeeze[b+1]))
    
            else:
                # Compute Legendre map using spherical harmonics
                leg_map = self.base.apply_fourier_harmonics(Sinv_data_real, self.Ylm_real[l], self.Ylm_fourier[l])
                for b in range(self.Nk_squeeze):
                    g_bl_maps[l//2,b] = self.base.to_real(self.base.map_utils.fourier_filter(leg_map, 0, self.k_bins_squeeze[b], self.k_bins_squeeze[b+1]))
            
        # Compute numerator
        if verb: print("Computing cubic term")
        B3_num = self.base.utils.assemble_b3(g_bl_maps, self.all_bins, self.base.nthreads)
        
        # Compute b_1 part of cubic estimator, averaging over simulations
        B1_num = np.zeros(self.N_bins)
        if not include_linear_term:
            if verb: print("No linear correction applied!")
        else:
            if verb: print("Computing linear term")
            for ii in range(self.N_it):
                if (ii+1)%5==0 and verb: print("On simulation %d of %d"%(ii+1,self.N_it))

                # Load processed bias simulations 
                if self.preload:
                    this_g_bl_maps = self.base.np.array(self.g_bl_maps[ii])
                else:
                    this_g_bl_maps = self.load_sim_data(ii)

                # Compute numerator
                B1_num += -1./self.N_it*self.base.utils.assemble_b1(g_bl_maps, this_g_bl_maps, self.all_bins, self.base.nthreads)

        # Assemble output and add normalization
        norm_factor = self.base.volume/self.base.gridsize.prod()/self.sym_factor
        Bk_num = (B3_num+B1_num)*norm_factor
        return Bk_num

    ### IDEAL ESTIMATOR
    def Bk_numerator_ideal(self, data, verb=False):
        """Compute the numerator of the idealized bispectrum estimator, weighting by 1/P_fid(k) within each bin. 

        The estimator does *not* use the mask or S_inv weighting schemes, and does not remove the linear term (which vanishes under ideal circumstances). It also applies only for even ell.

        This can compute the ideal bispectrum of simulation volumes, or, for suitably normalized input, the FKP bispectrum.
        """
        if not self.const_mask:
            assert data.shape == self.mask.shape, "Data must have same shape as the mask!"
        
        # Send to wrapper
        return self._compute_bk_numerator(data, include_linear_term=False, filtering='ideal', verb=verb)

    def compute_fisher_ideal(self, discreteness_correction=True, verb=False):
        """This computes the idealized Fisher matrix for the power spectrum, weighting by 1/P_fid(k) within each bin. 

        We optionally include discreteness correction for ell>0. 
        
        If 'local' sightlines are being used, this assumes the distant-observer approximation.
        """
        # Compute symmetry factor if necessary
        if not hasattr(self, 'sym_factor'):
            self._compute_symmetry_factor()

        print("Computing ideal Fisher matrix")
        
        if discreteness_correction:

            # Define output array
            fish = np.zeros((self.N_bins,self.N_bins))
            
            # Define discrete binning functions
            if verb: print("Computing u_{b,l}(r) maps")
            bins_l = self.base.real_zeros((self.Nl, self.Nk_squeeze), numpy=True)
            for l in range(0, self.lmax+1, 2):
                for b in range(self.Nk_squeeze):
                    bins_l[l//2][b] = self.base.to_real(self.base.map_utils.fourier_filter(self.invPk0.astype(np.complex128), l, self.k_bins_squeeze[b], self.k_bins_squeeze[b+1]))

            # Iterate over Legendre multipoles
            for l in range(0, self.lmax+1, 2):
                for lp in range(l, self.lmax+1, 2):
                    if verb: print("Computing Fisher for (l, l') = (%d, %d)"%(l,lp))

                    # Compute double Legendre-weighted data
                    bins_llp = self.base.real_zeros(self.Nk_squeeze, numpy=True)
                    for b in range(self.Nk_squeeze):
                        bins_llp[b] = self.base.to_real(self.base.map_utils.fourier_filter_ll(self.invPk0.astype(np.complex128), l, lp, self.k_bins_squeeze[b], self.k_bins_squeeze[b+1]))
                    
                    # Assemble output, iterating over symmetries
                    for i in range(self.N3):
                        bin1, bin2, bin3 = self.all_bins[i,:3]
                        
                        if bin1!=bin2 and bin2!=bin3:
                            fish[l//2*self.N3+i,lp//2*self.N3+i] = self.base.map_utils.sum_triplet(bins_l[0][bin1],bins_l[0][bin2],bins_llp[bin3])
                        elif bin1==bin2 and bin2!=bin3:
                            fish[l//2*self.N3+i,lp//2*self.N3+i] = 2.*self.base.map_utils.sum_triplet(bins_l[0][bin1],bins_l[0][bin2],bins_llp[bin3])
                        elif bin1!=bin2 and bin2==bin3:
                            fish[l//2*self.N3+i,lp//2*self.N3+i] =  self.base.map_utils.sum_triplet(bins_l[0][bin1],bins_l[l//2][bin2],bins_l[lp//2][bin3])
                            fish[l//2*self.N3+i,lp//2*self.N3+i] += self.base.map_utils.sum_triplet(bins_l[0][bin1],bins_l[0][bin2],bins_llp[bin3])
                        elif bin1==bin2 and bin2==bin3:
                            fish[l//2*self.N3+i,lp//2*self.N3+i] = 4.*self.base.map_utils.sum_triplet(bins_l[0][bin1],bins_l[l//2][bin2],bins_l[lp//2][bin3])
                            fish[l//2*self.N3+i,lp//2*self.N3+i] += 2.*self.base.map_utils.sum_triplet(bins_l[0][bin1],bins_l[0][bin2],bins_llp[bin3])
                    
            # Add transpose symmetry
            fish = fish+fish.T-np.diag(np.diag(fish))

            # Normalize
            scaling = self.base.gridsize.prod()**2/self.base.volume
            fish = fish/np.outer(self.sym_factor, self.sym_factor)*scaling
            if verb: print("")

        else:
            if verb: print("Computing u_{b,0}(r) maps")
            bins = self.base.real_zeros(self.Nk_squeeze, numpy=True)
            for b in range(self.Nk_squeeze):
                bins[b] = self.base.to_real(self.base.map_utils.fourier_filter(self.invPk0.astype(np.complex128), 0, self.k_bins_squeeze[b], self.k_bins_squeeze[b+1]))

            # Assume Int L_ell(k.n) L_ell'(k.n) = 1/(2 ell + 1) Kronecker[ell, ell'] etc.
            fish_diag = np.zeros(self.N_bins)

            # Compute the number of triangles in the bin, N_0(b)
            N_triangles = np.zeros(self.N3)
            for i in range(self.N3):
                bin1, bin2, bin3 = self.all_bins[i,:3]
                N_triangles[i] = self.base.map_utils.sum_triplet(bins[bin1], bins[bin2], bins[bin3])

            # Assemble Fisher matrix
            for l in range(0, self.lmax+1, 2):
                fish_diag[l//2*self.N3:(l//2+1)*self.N3] = N_triangles/(2.*l+1.)

            # Normalize
            fish = np.diag(fish_diag/self.sym_factor)*self.base.gridsize.prod()**2/self.base.volume

        self.fish_ideal = fish
        self.inv_fish_ideal = np.linalg.inv(self.fish_ideal)
        self.discreteness_correction = discreteness_correction

        return fish
    
    def Bk_ideal(self, data, fish_ideal=[], discreteness_correction=True, verb=False):
        """Compute the (normalized) idealized bispectrum estimator, weighting by 1/P_fid(k) within each bin. 
        
        The estimator does *not* use the mask or S_inv weighting schemes (except for normalizing by < mask^3 >, and does not remove the linear term (which vanishes under ideal circumstances)
        
        We optionally include 
        discreteness corrections for ell>0.
    
        This can compute the ideal bispectrum of simulation volumes, or, for suitably normalized input, the FKP power spectrum.
        """
        
        if len(fish_ideal)!=0:
            self.fish_ideal = fish_ideal
            self.inv_fish_ideal = np.linalg.inv(fish_ideal)

        if not hasattr(self,'inv_fish_ideal'):
            self.compute_fisher_ideal(discreteness_correction=discreteness_correction, verb=verb)
        else:
            if self.discreteness_correction!=discreteness_correction:
                self.compute_fisher_ideal(discreteness_correction=discreteness_correction, verb=verb)
        
        # Compute numerator
        Bk_num_ideal = self.Bk_numerator_ideal(data, verb=verb)

        # Apply normalization and restructure
        Bk_out = np.matmul(self.inv_fish_ideal,Bk_num_ideal)
        
        # Create output dictionary of spectra
        Bk_dict = {}
        index = 0
        for l in range(0,self.lmax+1,2):
            Bk_dict['b%s'%l] = Bk_out[index:index+self.N3]
            index += self.N3

        return Bk_dict