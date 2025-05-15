### Code for ideal and unwindowed binned polyspectrum estimation in 3D. Author: Oliver Philcox (2023)
## This module contains the power spectrum estimation code

import numpy as np
import multiprocessing as mp
import tqdm, time
from scipy.special import legendre
from scipy.interpolate import interp1d

class PSpec():
    """Class containing the power spectrum estimators.
    
    Inputs:
    - base: PolyBin base class
    - k_bins: array of k bin edges (e.g., [0.01, 0.02, 0.03] would give two bins: [0.01,0.02] and [0.02,0.03]).
    - lmax: (optional) maximum Legendre multipole, default: 2.
    - mask: (optional) 3D mask, specifying the background density n(x,y,z).
    - applySinv: (optional) function which weights the data field. This is only used in unwindowed estimators.
    - odd_l: (optional) whether to include odd ell-modes in the estimator, default: False. Only relevant for local lines-of-sight.
    - add_GIC: (optional) whether to include the global integral constraint, default: False
    - add_RIC: (optional) whether to include the radial integral constraint, default: False
    - mask_IC: (optional) unweighted 3D mask used to model integral constraints.
    - radial_bins_RIC: (optional) radial bins used to define n(z) [used for the RIC correction].
    - mask_shot: (optional) doubly weighted 3D field used to define shot-noise.
    - applySinv_transpose: (optional) transpose of function which weights the data field. This is only used in unwindowed estimators.
    """
    def __init__(self, base, k_bins, lmax=2, mask=None, applySinv=None, odd_l=False, add_GIC=False, mask_IC=None, add_RIC=False, radial_bins_RIC=[], mask_shot=None, applySinv_transpose=None):
        # Read in attributes
        self.base = base
        self.applySinv = applySinv
        self.k_bins = np.asarray(k_bins)
        self.lmax = lmax
        self.Nk = len(k_bins)-1
        self.odd_l = odd_l
        if self.odd_l:
            self.Nl = self.lmax+1
        else:
            self.Nl = self.lmax//2+1
        self.Nl_even = self.lmax//2+1
        self.N_bins = self.Nk*self.Nl
        self.add_GIC = add_GIC
        self.add_RIC = add_RIC
        self.applySinv_transpose = applySinv_transpose
        
        print("")
        assert np.max(self.k_bins)<np.min(base.kNy), "k_max must be less than k_Nyquist!"
        assert np.min(self.k_bins)>=np.max(base.kF), "k_min must be at least the k_fundamental!"
        assert np.max(self.k_bins)<=base.Pfid[0][-1], "Fiducial power spectrum should extend up to k_max!"
        assert np.min(self.k_bins)>=base.Pfid[0][0], "Fiducial power spectrum should extend down to k_min!"
        print("Binning: %d bins in [%.3f, %.3f] h/Mpc"%(self.Nk,np.min(self.k_bins),np.max(self.k_bins)))
        print("l-max: %d"%(self.lmax))
        if self.odd_l:
            assert self.lmax>0, "lmax must be greater than 0 to use odd harmonics!"
            assert self.base.sightline=='local', "Odd spherical harmonics should only be used for local line-of-sight!"
            print("Including wide-angle effects")
        
        assert type(self.lmax)==int, "l-max must be an integer!"
        assert self.lmax<=4, "Only l-max<=4 is currently supported!"
        if self.odd_l: assert type(self.lmax%2==0), "Odd l-max requires odd_l mode to be switched on!"
        
        # Read-in mask
        if type(mask)==type(None):
            self.const_mask = True
            self.mask_mean = 1.
            self.sq_mask_mean = 1.
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
            # Compute the mean of the squared mask
            self.sq_mask_mean = np.mean(self.mask**2)
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
            print("Accounting for radial integral constraint across %d bins"%(len(radial_bins_RIC)-1))
            self.base.modr_grid = np.sqrt(self.base.r_grids[0]**2.+self.base.r_grids[1]**2.+self.base.r_grids[2]**2.)
            self.radial_bins_RIC = radial_bins_RIC
            
        # Define spherical harmonics [for computing power spectrum multipoles]
        if self.base.sightline=='local' and self.lmax>0:
            print("Generating spherical harmonics")
            self.Ylm_real = self.base.utils.compute_real_harmonics(np.asarray(self.base.r_grids), self.lmax, self.odd_l, self.base.nthreads)
            self.Ylm_fourier = self.base.utils.compute_real_harmonics(np.asarray(self.base.k_grids), self.lmax, self.odd_l, self.base.nthreads)
        else:
            self.Ylm_real = None
            self.Ylm_fourier = None
            
    def get_ks(self):
        """
        Return a list of the central k-values for each power spectrum bin.
        """
        # Iterate over bins
        ks = [0.5*(self.k_bins[b]+self.k_bins[b+1]) for b in range(self.Nk)]
        return np.asarray(ks)
        
    ### OPTIMAL ESTIMATOR
    def Pk_numerator(self, data):
        """
        Compute the numerator of the unwindowed power spectrum estimator, using the custom weighting function S^-1.
        """
        if not self.const_mask:
            assert data.shape == self.mask.shape, "Data must have same shape as the mask!"

        # Send to wrapper
        return self._compute_pk_numerator(data, filtering='Sinv')
    
    def compute_fisher_contribution(self, seed, verb=False):
        """This computes the contribution to the Fisher matrix from a single GRF simulation, created internally.
        
        An optional input is a function that applies (S^-1)^dagger to a map. The code will assume (S^-1)^dagger = S^-1 if this is not supplied."""
        
        # Run main algorithm
        return self._compute_fisher(seed, verb=verb, compute_cov=False)
    
    def compute_covariance_contribution(self, seed, Pk_cov, verb=False):
        """This computes the contribution to the Gaussian covariance matrix from a single GRF simulation, created internally.
        
        This requires an input theory power spectrum, Pk_cov (without shot-noise).
        
        An optional input is a function that applies (S^-1)^dagger to a map. The code will assume (S^-1)^dagger = S^-1 if this is not supplied."""
        if not self.const_mask:
            assert hasattr(self, 'mask_shot'), "Must supply mask_shot to compute shot-noise contribution"
        
        assert len(Pk_cov)!=0, "Must provide fiducial power spectrum if computing covariance!"
        assert len(np.asarray(Pk_cov).shape)==2, "Pk should contain k and P_0, (and optionally P_2, P_4) columns"
        assert len(Pk_cov) in [2,3,4], "Pk should contain k and P_0, (and optionally P_2, P_4) columns"
        assert (len(Pk_cov)-2)*2<=self.lmax, "Can't use higher multipoles than lmax"

        return self._compute_fisher(seed, verb=verb, compute_cov=True, Pk_cov=Pk_cov)
    
    def compute_theory_contribution(self, seed, k_bins_theory, lmax_theory=None, include_wideangle=False, verb=False):
        """This computes the a rectangular matrix used to window-convolve the theory, using a single GRF simulation, created internally.
        We also compute and output the contribution to the (square) Fisher matrix, used for normalization.
        
        We must specify the desired theory binning and lmax.
        
        Finally, we can optionally add wide-angle effects (for the theory matrix, not the Fisher matrix)."""

        # Check k-binning array
        k_bins_theory = np.asarray(k_bins_theory)
        assert np.max(k_bins_theory)<np.min(self.base.kNy), "k_max must be less than k_Nyquist!"
        assert np.min(k_bins_theory)>=np.max(self.base.kF), "k_min must be at least the k_fundamental!"
        assert np.max(k_bins_theory)<=self.base.Pfid[0][-1], "Fiducial power spectrum should extend up to k_max!"
        assert np.min(k_bins_theory)>=self.base.Pfid[0][0], "Fiducial power spectrum should extend down to k_min!"
        
        # Define lmax and check wide-angle effects
        if lmax_theory==None:
            lmax_theory = self.lmax
            assert lmax_theory>=self.lmax, "Can't use larger lmax for data than for theory!"
        if include_wideangle:
            assert not self.base.sightline=='global', "Wide-angle effects require local lines-of-sight!"
            assert not self.odd_l, "Cannot include wide-angle effects for odd l!"
            assert lmax_theory>1, "There are no wide-angle effects for the monopole!"
            assert lmax_theory<=4, "Higher-order effects not yet implemented!"
            if verb: print("Adding odd-ell wide-angle effects up to l = %d"%lmax_theory)
            if not hasattr(self.base,'modr_grid'):
                self.base.modr_grid = np.sqrt(self.base.r_grids[0]**2.+self.base.r_grids[1]**2.+self.base.r_grids[2]**2.)
        else:
            if not self.odd_l:
                assert lmax_theory%2==0, "Must use even lmax if not including wide-angle effects!"
        
        # Run main code
        return self._compute_fisher(seed, verb=verb, compute_theory=True, k_bins_theory=k_bins_theory, lmax_theory=lmax_theory, include_wideangle=include_wideangle)
    
    def apply_pointing(self, map_real, transpose=False):
        """Apply the pointing matrix to a map. This multiplies by the mask and optionally includes IC effects."""
        if self.add_RIC:
            if transpose:
                return self.base.utils.pointing_RIC_transpose(map_real, self.mask, self.mask_IC, self.radial_bins_RIC, self.base.modr_grid, self.base.nthreads)
            else:
                return self.base.utils.pointing_RIC(map_real, self.mask, self.mask_IC, self.radial_bins_RIC, self.base.modr_grid, self.base.nthreads)
        elif self.add_GIC:
            if transpose:
                return self.base.utils.pointing_GIC_transpose(map_real, self.mask, self.mask_IC, self.base.nthreads)
            else:
                return self.base.utils.pointing_GIC(map_real, self.mask, self.mask_IC, self.base.nthreads)
        else:
            out = self.base.map_utils.prod_real(map_real, self.mask)
            return out

    def _compute_fisher(self, seed, verb=False, compute_cov=False, Pk_cov=[], compute_theory=False, k_bins_theory=[], lmax_theory=None, include_wideangle=False):
        """Internal function to compute the contribution to the Fisher matrix, covariance, or theory matrix."""

        # Initialize output
        fisher_matrix = np.zeros((self.N_bins,self.N_bins))
        if compute_theory:
            # Define lmax and check wide-angle effects
            if include_wideangle:
                odd_l_theory = True
            else:
                odd_l_theory = self.odd_l
            if odd_l_theory and not include_wideangle:
                Nl_theory = lmax_theory+1
            else:
                Nl_theory = lmax_theory//2+1
            Nk_th = len(k_bins_theory)-1
            N_bins_th = Nl_theory*Nk_th
            binning_matrix = np.zeros((self.N_bins,N_bins_th))
        
        # Precompute power spectrum fields
        if compute_cov:
            assert not compute_theory
            assert not include_wideangle
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
        
        # Compute new spherical harmonics if necessary
        if compute_theory and self.base.sightline=='local':
            assert not compute_cov
            if (not hasattr(self,'Ylm_real_theory') and not hasattr(self,'Ylm_fourier_theory')):
                if (lmax_theory>self.lmax) or (odd_l_theory!=self.odd_l):
                    if verb: print("Generating spherical harmonics up to l = %d"%lmax_theory)
                    self.Ylm_real_theory = self.base.utils.compute_real_harmonics(np.asarray(self.base.r_grids), lmax_theory, odd_l_theory, self.base.nthreads)
                    self.Ylm_fourier_theory = self.base.utils.compute_real_harmonics(np.asarray(self.base.k_grids), lmax_theory, odd_l_theory, self.base.nthreads)
                else:
                    self.Ylm_real_theory = self.Ylm_real
                    self.Ylm_fourier_theory = self.Ylm_fourier

        # Compute a random realization with known power spectrum
        if verb: print("Generating GRF")
        a_map_fourier = self.base.generate_data(seed=seed+int(1e7), output_type='fourier')
        if not self.const_mask:
            a_map_real = self.base.to_real(a_map_fourier)
        
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
                    CovSid_map = self.apply_pointing(CPdSid_map, transpose=False) + self.base.map_utils.prod_real(Sid_map, self.mask_shot)
                    return self.applySinv(CovSid_map, input_type='real', output_type='fourier')
                
        def apply_filter_dagger(input_map, real=False):
            """Apply (S^-1 P)^dagger weighting to maps. Input is either a Fourier-space map or a real-space map."""
            if not compute_cov:
                if self.const_mask:
                    if real:
                        return self.mask_mean*self.applySinv_transpose(input_map,input_type='real',output_type='fourier')
                    else:
                        return self.mask_mean*self.applySinv_transpose(input_map,input_type='fourier',output_type='fourier')         
                else:
                    if real:
                        # Use real-space map directly
                        return self.base.to_fourier(self.apply_pointing(self.applySinv_transpose(input_map,input_type='real',output_type='real'), transpose=True))
                    else:
                        return self.base.to_fourier(self.apply_pointing(self.applySinv_transpose(input_map,input_type='fourier',output_type='real'), transpose=True))
            else:
                # Apply S^-1 P Cov P^dagger S^-dagger in order to compute covariances
                if self.const_mask:
                    if real:
                        Sid_map = self.applySinv_transpose(input_map, input_type='real', output_type='fourier')
                    else:
                        Sid_map = self.applySinv_transpose(input_map, input_type='fourier', output_type='fourier')
                    CSid_map = self.mask_mean**2*self.base.apply_xi(Sid_map, Ylm_real=self.Ylm_real, Ylm_fourier=self.Ylm_fourier, Pk_grid=Pk_grid, output_type='fourier') + self.mask_mean*Sid_map
                    return self.applySinv(CSid_map, input_type='fourier', output_type='fourier')
                else:
                    if real:
                        Sid_map = self.applySinv_transpose(input_map, input_type='real', output_type='real')
                    else:
                        Sid_map = self.applySinv_transpose(input_map, input_type='fourier', output_type='real')
                    PdSid_map_real = self.apply_pointing(Sid_map, transpose=True)
                    PdSid_map = self.base.to_fourier(PdSid_map_real)
                    CPdSid_map = self.base.apply_xi(PdSid_map, PdSid_map_real, Ylm_real=self.Ylm_real, Ylm_fourier=self.Ylm_fourier, Pk_grid=Pk_grid, output_type='real')
                    CovSid_map = self.apply_pointing(CPdSid_map, transpose=False) + self.base.map_utils.prod_real(Sid_map, self.mask_shot)
                    out = self.applySinv(CovSid_map, input_type='real', output_type='fourier')
                return out
                
        # Filter map by A^-1 and S^-1 P
        if verb: print("Computing A^-1.a and S^-1.P.a")
        Ainv_map_fourier = self.base.applyAinv(a_map_fourier, input_type='fourier', output_type='fourier')
        if self.const_mask:
            SinvP_map_fourier = apply_filter(a_map_fourier)
        else:
            SinvP_map_fourier = apply_filter(a_map_real)
               
        # Define real-space maps (where necessary)
        if self.base.sightline=='local' and self.lmax>0:
            SinvP_map_real = self.base.to_real(SinvP_map_fourier)
        if compute_theory:
            if self.base.sightline=='local' and lmax_theory>0:
                Ainv_map_real = self.base.to_real(Ainv_map_fourier)
                if include_wideangle:
                    inv_r_Ainv_map_real = self.base.map_utils.div_real(Ainv_map_real, self.base.modr_grid)
        elif (self.base.sightline=='local' and self.lmax>0):
            Ainv_map_real = self.base.to_real(Ainv_map_fourier)
            
        # Precompute all Legendre multipoles
        if verb: print("Computing Legendre multipoles for A^-1.a")
        leg_maps = {}
        if not compute_theory:
            if self.base.sightline=='local':
                for li in range(1,self.Nl):
                    # Compute e^{-ik.x}L_l(k.x)a(x)
                    leg_maps[li*(2-self.odd_l)] = self.base.apply_fourier_harmonics(Ainv_map_real, self.Ylm_real[(2-self.odd_l)*li], self.Ylm_fourier[(2-self.odd_l)*li])
        else:
            if self.base.sightline=='local' and not include_wideangle:
                for li in range(1,Nl_theory):
                    # Compute e^{-ik.x}L_l(k.x)a(x)
                    leg_maps[li*(2-odd_l_theory)] = self.base.apply_fourier_harmonics(Ainv_map_real, self.Ylm_real_theory[(2-odd_l_theory)*li], self.Ylm_fourier_theory[(2-odd_l_theory)*li])
            
            elif self.base.sightline=='local' and include_wideangle:
                for l in range(1,lmax_theory+1):
                    if l%2==0:
                        # Compute e^{-ik.x}L_l(k.x)a(x)
                        leg_maps[l] = self.base.apply_fourier_harmonics(Ainv_map_real, self.Ylm_real_theory[l], self.Ylm_fourier_theory[l])
                    else:
                        # Compute e^{-ik.x}L_l(k.x)a(x)/kx
                        leg_maps[l] = self.base.apply_fourier_harmonics(inv_r_Ainv_map_real, self.Ylm_real_theory[l], self.Ylm_fourier_theory[l])
                    
        def derivative_coefficient(ki):
            """Compute coefficients for numerical derivative of k * dP(k)/dk"""
            kfac = np.zeros(Nk_th)
            for ki in range(Nk_th):
                if ki-1>0 and ki-1<Nk_th-1: 
                    kfac[ki] += k_bins_theory[ki-1]/(k_bins_theory[ki]-k_bins_theory[ki-2])
                if ki+1>0 and ki+1<Nk_th-1:
                    kfac[ki] -= k_bins_theory[ki+1]/(k_bins_theory[ki+2]-k_bins_theory[ki])
                kfac[ki] += k_bins_theory[0]/(k_bins_theory[1]-k_bins_theory[0])*((ki==1)-(ki==0))
                kfac[ki] -= k_bins_theory[-1]/(k_bins_theory[-1]-k_bins_theory[-2])*((ki==Nk_th-1)-(ki==Nk_th-2))
            return kfac            

        def add_to_matrix(row, Q):
            """Compute a single row of the Fisher matrix (and optionally the binning matrix), given a Q map."""
                  
            def bin_integrate_all(input_map1, input_map2, k_bins=self.k_bins, li=0, coeff=None):
                """Integrate two maps over all k-bins"""
                if coeff is None:
                    integ = self.base.integrator.cross_integrate(input_map1,input_map2, k_bins, li*2)
                elif coeff is not None:
                    assert li==0
                    integ = self.base.integrator.cross_integrate_wcoeff(coeff, input_map1, input_map2, k_bins)
                scaling = 0.5*self.base.volume/self.base.gridsize.prod()**2
                return integ*scaling
                  
            # Assemble binning matrix for the monopole
            fisher_matrix[row, :self.Nk] += bin_integrate_all(Ainv_map_fourier,Q)
            if compute_theory:
                binning_matrix[row, :Nk_th] += bin_integrate_all(Ainv_map_fourier,Q,k_bins_theory)
                
            # Assemble matrix for higher-order multipoles
            if self.base.sightline=='global':
                for li in range(1,self.Nl):
                    fisher_matrix[row, li*self.Nk:(li+1)*self.Nk] += bin_integrate_all(Ainv_map_fourier, Q, li=li)
                if compute_theory: 
                    for li in range(1,Nl_theory):
                        binning_matrix[row, li*Nk_th:(li+1)*Nk_th] += bin_integrate_all(Ainv_map_fourier, Q, li=li, k_bins=k_bins_theory)

            elif self.base.sightline=='local':
                # Assemble Fisher matrix
                for li in range(1,self.Nl):
                    # Note that we add a phase for odd ell
                    if (self.odd_l and li%2==1):
                        fisher_matrix[row, li*self.Nk:(li+1)*self.Nk] += bin_integrate_all(leg_maps[li*(2-self.odd_l)], 1.0j*Q)
                    else:
                        fisher_matrix[row, li*self.Nk:(li+1)*self.Nk] += bin_integrate_all(leg_maps[li*(2-self.odd_l)], Q)

                if compute_theory:
                    # Assemble binning matrix
                    if not include_wideangle:
                        for li in range(1,Nl_theory):
                            # Note that we add a phase for odd ell
                            if (odd_l_theory and li%2==1):
                                binning_matrix[row, li*Nk_th:(li+1)*Nk_th] += bin_integrate_all(leg_maps[li*(2-odd_l_theory)], 1.0j*Q, k_bins=k_bins_theory)
                            else:
                                binning_matrix[row, li*Nk_th:(li+1)*Nk_th] += bin_integrate_all(leg_maps[li*(2-odd_l_theory)], Q, k_bins=k_bins_theory)                
                    else:
                        for l in range(1,lmax_theory+1):
                            # Compute even ell piece and add to output
                            if l%2==0:
                                binning_matrix[row, l//2*Nk_th:(l//2+1)*Nk_th] += bin_integrate_all(leg_maps[l], Q, k_bins=k_bins_theory)

                            # Compute wide-angle corrections from odd ell
                            else:
                                kfac = derivative_coefficient(ki)
                                # Add to ell=2
                            if l==1 and lmax_theory>=2:
                                binning_matrix[row, Nk_th:2*Nk_th] += bin_integrate_all(leg_maps[l], Q, coeff=3.0j/5.*(3+kfac), k_bins=k_bins_theory)
                            # Add to ell=2,4
                            if l==3 and lmax_theory>=2:
                                binning_matrix[row, Nk_th:2*Nk_th] += bin_integrate_all(leg_maps[l], Q, coeff=3.0j/5.*(2-kfac), k_bins=k_bins_theory)                                
                            if l==3 and lmax_theory>=4:
                                binning_matrix[row, 2*Nk_th:3*Nk_th] += bin_integrate_all(leg_maps[l], Q, coeff=10.0j/9.*(5+kfac), k_bins=k_bins_theory)                                
            
        # Compute Q derivative for the monopole and add to output
        if verb: print("Computing l = 0 output")
        for ki in range(self.Nk):
            kmap = self.base.map_utils.fourier_filter(SinvP_map_fourier, 0, self.k_bins[ki], self.k_bins[ki+1])
            add_to_matrix(ki, apply_filter_dagger(kmap, real=False))

        # Repeat for higher-order multipoles
        for li in range(1,self.Nl):
            
            if self.base.sightline=='global':
                if verb: print("Computing l = %d output"%(2*li))
            
                # Multiply by L_ell(mu) and add to bins
                for ki in range(self.Nk):
                    add_to_matrix(li*self.Nk+ki, apply_filter_dagger(self.base.map_utils.fourier_filter(SinvP_map_fourier, li*2, self.k_bins[ki], self.k_bins[ki+1]), real=False))

            else:
                if verb: print("Computing l = %d output"%((2-self.odd_l)*li))

                # First part: (-1)^l Theta_b(k) Sum_m Y_lm(k)* [S^-1.P.a]_lm(k)
                leg_map = self.base.apply_fourier_harmonics(SinvP_map_real, self.Ylm_real[(2-self.odd_l)*li], self.Ylm_fourier[(2-self.odd_l)*li])
                                
                # Add to bins
                for ki in range(self.Nk):
                    
                    # Add first part
                    kmap = self.base.map_utils.fourier_filter(leg_map, 0, self.k_bins[ki], self.k_bins[ki+1])
                    if (li%2==1 and self.odd_l):
                        kmap *= -1.0j
                    Q_map = 0.5*apply_filter_dagger(kmap, real=False)

                    # Second part: Sum_m Y_lm (x) Int_k e^ik.x Theta_b(k) Y_lm(k)*[P a](k)                        
                    real_map = self.base.real_zeros()
                    for lm_ind in range(len(self.Ylm_real[(2-self.odd_l)*li])):
                        # Apply Theta_b function
                        k_map = self.base.map_utils.prod_fourier_filter(SinvP_map_fourier, self.Ylm_fourier[(2-self.odd_l)*li][lm_ind], self.k_bins[ki], self.k_bins[ki+1])
                        
                        # Add to sum, being careful of real and imaginary parts
                        if (li%2==1 and self.odd_l):
                            k_map *= -1.0j
                        real_map += self.base.map_utils.prod_real(self.base.to_real(k_map), self.Ylm_real[(2-self.odd_l)*li][lm_ind])
                    
                    # Add second part, using the real-space map [which fills all Fourier modes, not just those in [k_min, k_max]]
                    Q_map += 0.5*apply_filter_dagger(real_map, real=True)

                    # Add to output
                    add_to_matrix(li*self.Nk+ki, Q_map)
                    
        if compute_theory:
            return fisher_matrix, binning_matrix
        else:
            return fisher_matrix
    
    def compute_shot_contribution(self, seed):
        """This computes the contribution to the shot-noise from a single GRF simulation, created internally."""
                
        if not self.const_mask:
            assert hasattr(self, 'mask_shot'), "Must supply mask_shot to compute shot-noise contribution"

        # Initialize output
        shot = np.zeros(self.N_bins)
        
        # Define an inverse power spectrum to draw GRFs from
        PkA = [self.base.Pfid[0],1./self.base.Pfid[1]]
        PkA_grid = interp1d(PkA[0], PkA[1], bounds_error=False, fill_value=(PkA[1][0],PkA[1][-1]))(self.base.modk_grid)
        invPkA_grid = np.zeros_like(PkA_grid)
        invPkA_grid[self.base.modk_grid!=0] = 1./PkA_grid[self.base.modk_grid!=0]

        # Compute a random realization with known power spectrum
        a_map_fourier = self.base.generate_data(seed=seed+int(1e7), Pk_input=PkA, output_type='fourier')
        
        # Compute S^-1 N A^-1 a in Fourier-space
        if self.const_mask:
            Ainv_a_fourier = self.base.applyAinv(a_map_fourier, invPk0_grid=invPkA_grid, input_type='fourier', output_type='fourier')
            Sinv_N_Ainv_a = self.mask_mean*self.applySinv(Ainv_a_fourier, input_type='fourier', output_type='fourier')
        else:
            Ainv_a = self.base.applyAinv(a_map_fourier, invPk0_grid=invPkA_grid, input_type='fourier', output_type='real')
            Sinv_N_Ainv_a = self.applySinv(self.base.map_utils.prod_real(Ainv_a, self.mask_shot), input_type='real', output_type='fourier')
        
        # Compute S^-1 a
        Sinv_a = self.applySinv(a_map_fourier, input_type='fourier', output_type='fourier')
        if self.lmax>0 and self.base.sightline=='local':
            Sinv_a_real = self.base.to_real(Sinv_a)
        
        # Assemble monopole
        shot[:self.Nk] = 0.5*self.base.integrator.cross_integrate(Sinv_a, Sinv_N_Ainv_a, self.k_bins, 0)
        
        # Repeat for higher-order multipoles
        for li in range(1,self.Nl):
            if self.base.sightline=='global':
                
                # Compute legendre-weighted integral
                shot[li*self.Nk:(li+1)*self.Nk] = 0.5*self.base.integrator.cross_integrate(Sinv_a, Sinv_N_Ainv_a, self.k_bins, 2*li)
            
            else:
                
                # Compute Sum_m Y_lm(k)[S^-1 a]^*_lm(k)
                lm_sum = self.base.apply_fourier_harmonics(Sinv_a_real, self.Ylm_real[(2-self.odd_l)*li], self.Ylm_fourier[(2-self.odd_l)*li])
                
                # Ensure output is real
                if (self.odd_l and li%2==1):
                    lm_sum *= -1.0j
                
                # Apply binning
                shot[li*self.Nk:(li+1)*self.Nk] = 0.5*self.base.integrator.cross_integrate(lm_sum, Sinv_N_Ainv_a, self.k_bins, 0)
                
        # Add FFT normalization
        shot *= self.base.volume/self.base.gridsize.prod()**2
        
        return shot

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
    
    def Pk_unwindowed(self, data, fish=[], shot_num=[], subtract_shotnoise=False):
        """
        Compute the unwindowed power spectrum estimator. 
        
        Note that the Fisher matrix and shot-noise terms must be computed before this is run, or they can be supplied separately.
        
        The Poisson shot-noise can be optionally subtracted. We return the imaginary part of any ell=odd spectra.
        """
        
        # Compute inverse Fisher
        if len(fish)!=0:
            self.fish = fish
            self.inv_fish = np.linalg.inv(fish)
            
        # Compute shot-noise
        if len(shot_num)!=0 or subtract_shotnoise:
            self.shot_num = shot_num
            self.shot = np.matmul(self.inv_fish,self.shot_num)
        
        if not hasattr(self,'inv_fish'):
            raise Exception("Need to compute Fisher matrix first!")
        
        if not hasattr(self,'shot') and subtract_shotnoise:
            raise Exception("Need to compute shot-noise first!")

        # Compute numerator
        Pk_num = self.Pk_numerator(data)

        # Apply normalization and restructure
        Pk_out = np.matmul(self.inv_fish,Pk_num)
        if subtract_shotnoise:
            Pk_out -= self.shot
        
        # Create output dictionary of spectra
        Pk_dict = {}
        index = 0
        for l in range(0,self.lmax+1,2-self.odd_l):
            Pk_dict['p%s'%l] = Pk_out[index:index+self.Nk]
            index += self.Nk

        return Pk_dict
       
    def _compute_pk_numerator(self, data, filtering='ideal'):
        """Internal function to compute the power spectrum numerator. This is used by the Pk_numerator and Pk_numerator_ideal functions."""
        assert filtering in ['ideal','Sinv'], "Unknown filtering option supplied!"

        # Check input data type and convert to float64 if necessary
        assert type(data[0,0,0]) in [np.float32, np.float64], "Data must be of type float32 or float64!"
        if type(data[0,0,0])==np.float32: 
            data = np.asarray(data, order='C', dtype=np.float64)

        # Apply filtering and transform to Fourier-space, optionally removing the pixel window function
        if self.base.pixel_window!='none':
            
            # Transform to Fourier-space
            data_fourier = self.base.map_utils.div_fourier(self.base.to_fourier(data), self.base.pixel_window_grid)
            
            if filtering=='ideal':
                # Filter by 1/P_fid
                Sinv_data_fourier = self.base.map_utils.prod_fourier(data_fourier, self.base.invPk0_grid)/np.sqrt(self.sq_mask_mean)
            elif filtering=='Sinv':
                Sinv_data_fourier = self.applySinv(data_fourier, input_type='fourier', output_type='fourier')
        
        else:
            if filtering=='ideal':
                # Filter by 1/P_fid
                Sinv_data_fourier = self.base.map_utils.prod_fourier(self.base.to_fourier(data),self.base.invPk0_grid)/np.sqrt(self.sq_mask_mean)
            elif filtering=='Sinv':
                Sinv_data_fourier = self.applySinv(data, input_type='real', output_type='fourier')
            
        # Compute real-space map where necessary
        if (self.base.sightline=='local' and self.lmax>0):
            Sinv_data_real = self.base.to_real(Sinv_data_fourier)
        
        # Define output array
        if filtering=='Sinv':
            Pk_out = np.zeros(self.N_bins, dtype=np.float64)
        else:
            Pk_out = np.zeros(self.Nl_even*self.Nk, dtype=np.float64)    
        
        # Compute monopole
        Pk_out[:self.Nk] = 0.5*self.base.integrator.integrate(Sinv_data_fourier, self.k_bins, 0)
        
        # Compute higher multipoles
        if self.lmax>0:
            
            # Check number of bins and ordering of spherical harmonic grids
            if filtering=='Sinv':
                Nl = self.Nl
                l_factor = (2-self.odd_l)
            else:
                Nl = self.Nl_even
                l_factor = 2
                
            # Iterate over multipoles
            for li in range(1,Nl):
                
                if self.base.sightline=='global':
                    # Compute L_ell(mu)*|S^-1 d(k)|^2, integrated over k
                    Pk_out[li*self.Nk:(li+1)*self.Nk] = 0.5*self.base.integrator.integrate(Sinv_data_fourier, self.k_bins, 2*li)                
                else:
                    # Compute Sum_m Y_lm(k)[S^-1 d](k)[S^-1 d]^*_lm(k)
                    lm_sum = self.base.apply_fourier_harmonics(Sinv_data_real, self.Ylm_real[l_factor*li], self.Ylm_fourier[l_factor*li])
                                            
                    # Ensure output is real
                    if (filtering=='Sinv' and self.odd_l and li%2==1):
                        lm_sum *= -1.0j
                    
                    # Apply binning
                    Pk_out[li*self.Nk:li*self.Nk+self.Nk] = 0.5*self.base.integrator.cross_integrate(Sinv_data_fourier, lm_sum, self.k_bins, 0)
        
        # Add normalization
        Pk_out *= self.base.volume/self.base.gridsize.prod()**2
        
        return Pk_out
              
    ### IDEAL ESTIMATOR
    def Pk_numerator_ideal(self, data):
        """Compute the numerator of the idealized power spectrum estimator, weighting by 1/P_fid(k) within each bin. 
        
        The estimator does *not* use the mask or S^-1 weighting schemes. It also applies only for even ell. We normalize by the mean of the squared mask.
        
        This can compute the ideal power spectrum of simulation volumes, or, for suitably normalized input, the FKP power spectrum.
        """
        if not self.const_mask:
            assert data.shape == self.mask.shape, "Data must have same shape as the mask!"
        
        # Send to internal function
        return self._compute_pk_numerator(data, filtering='ideal') 
                
    def compute_fisher_ideal(self, discreteness_correction=True, Pk_fid='default'):
        """This computes the idealized Fisher matrix for the power spectrum, weighting by 1/P_fid(k) within each bin. 
        
        We optionally include discreteness correction for ell>0. We can use a new fiducial power spectrum by specifying Pk_fid (which is None for unit spectrum).
    
        If 'local' sightlines are being used, this assumes the distant-observer approximation.
        """
        print("Computing ideal Fisher matrix")
        
        # Define output array
        fish = np.zeros((self.Nl_even*self.Nk,self.Nl_even*self.Nk))
        
        # Define fiducial power spectrum
        if Pk_fid!='default':
            if (np.asarray(Pk_fid)==None).all() or (Pk_fid=='none'):
                k_tmp = np.arange(np.min(self.base.kF)/10.,np.max(self.base.kNy)*2,np.min(self.base.kF)/10.)
                Pfid = np.asarray([k_tmp, 1.+0.*k_tmp])
            else:
                assert len(Pk_fid)==2, "Pk should contain k and P_0(k) columns"
                Pfid = np.asarray(Pk_fid)
            
            # Apply Pk to grid
            Pk0_grid = interp1d(Pfid[0], Pfid[1], bounds_error=False, fill_value=0.)(self.base.modk_grid)
            
            # Invert Pk
            invPk0_grid = np.zeros(self.base.modk_grid.shape)
            good_filter = (Pk0_grid!=0)&np.isfinite(Pk0_grid)
            invPk0_grid[good_filter] = 1./Pk0_grid[good_filter] 
            del Pk0_grid
            
        else:
            invPk0_grid = self.base.invPk0_grid.copy()
        
        if discreteness_correction:
            
            # Iterate over fields, assembling Fisher matrix diagonal
            for la in range(self.Nl_even):
                for lb in range(self.Nl_even):
                    fish[la*self.Nk:(la+1)*self.Nk,lb*self.Nk:(lb+1)*self.Nk] = np.diag(0.5*self.base.integrator.real_integrate_double(invPk0_grid, 2*la, 2*lb, self.k_bins))
                        
        else:
            # Replace Int L_ell(k.n) L_ell'(k.n) by 1/(2 ell + 1) Kronecker[ell, ell']
            fish_diag = 0.5*self.base.integrator.real_integrate_double(invPk0_grid, 0, 0, self.k_bins)
            for la in range(self.Nl_even):
                fish[la*self.Nk:(la+1)*self.Nk,la*self.Nk:(la+1)*self.Nk] = np.diag(fish_diag)/(4.*la+1.)
                    
        # Save attributes and return                     
        self.fish_ideal = fish
        self.inv_fish_ideal = np.linalg.inv(self.fish_ideal)
        
        return fish
    
    def Pk_ideal(self, data, fish_ideal=[], discreteness_correction=True):
        """Compute the (normalized) idealized power spectrum estimator, weighting by 1/P_fid(k) within each bin. 
        
        The estimator does *not* use the mask or S_inv weighting schemes (except for normalizing by < mask^2 >. It also applies only for even ell.
        
        We optionally include discreteness corrections in the Fisher matrix for ell>0.
    
        This can compute the ideal power spectrum of simulation volumes, or, for suitably normalized input, the FKP power spectrum.
        """

        if len(fish_ideal)!=0:
            self.fish_ideal = fish_ideal
            self.inv_fish_ideal = np.linalg.inv(fish_ideal)

        if not hasattr(self,'inv_fish_ideal'):
            self.compute_fisher_ideal(discreteness_correction=discreteness_correction)

        # Compute numerator
        Pk_num_ideal = self.Pk_numerator_ideal(data)

        # Apply normalization and restructure
        Pk_out = np.matmul(self.inv_fish_ideal,Pk_num_ideal)
        
        # Create output dictionary of spectra
        Pk_dict = {}
        index = 0
        for l in range(0,self.lmax+1,2):
            Pk_dict['p%s'%l] = Pk_out[index:index+self.Nk]
            index += self.Nk

        return Pk_dict
