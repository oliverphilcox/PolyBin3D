### Code for ideal and unwindowed binned polyspectrum estimation in 3D. Author: Oliver Philcox (2023)
## This module contains the power spectrum estimation code

import numpy as np
import multiprocessing as mp
import tqdm
from scipy.special import legendre

class PSpec():
    """Power spectrum estimation class. This takes a set of k-bins as input and a base class. 
    We also feed in a function that applies the S^-1 operator. 
    
    Inputs:
    - base: PolyBin class
    - k_bins: array of bin edges
    - lmax: (optional) maximum Legendre multipole, default: 4.
    - mask: (optional) 3D mask to deconvolve.
    - applySinv: (optional) function which returns S^-1 ( ~ [Mask*2PCF + 1]^-1 ) applied to a given input data field. This is only needed for unwindowed estimators.
    - odd_l: (optional) whether to include odd ell-modes in the estimator, deafult: False. Only relevant for local lines-of-sight.
    """
    def __init__(self, base, k_bins, lmax=4, mask=None, applySinv=None, odd_l=False):
        # Read in attributes
        self.base = base
        self.applySinv = applySinv
        self.k_bins = k_bins
        self.lmax = lmax
        self.Nk = len(k_bins)-1
        self.odd_l = odd_l
        if self.odd_l:
            self.Nl = self.lmax+1
        else:
            self.Nl = self.lmax//2+1
        self.Nl_even = self.lmax//2+1
        self.N_bins = self.Nk*self.Nl
        
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
            self.mask = np.ones(self.base.gridsize)
            self.const_mask = True
        else:
            self.mask = mask
            # Check if window is uniform
            if np.std(self.mask)<1e-12:
                self.const_mask = True
            else:
                self.const_mask = False
        if self.const_mask:
            print("Mask: constant")
        else:
            print("Mask: spatially varying")
        self.mask_mean = np.mean(self.mask)
        self.sq_mask_mean = np.mean(self.mask**2)
            
        # Create local copies of k arrays, filtered to desired k-range
        self.k_filt = (self.base.modk_grid>=self.k_bins[0])&(self.base.modk_grid<self.k_bins[-1])
        self.modk_grid = self.base.modk_grid[self.k_filt]
        self.invPk0_grid = self.base.invPk0_grid[self.k_filt]
        if self.lmax>=2:
            self.muk_grid = self.base.muk_grid[self.k_filt]
        
        # Define spherical harmonics [for computing power spectrum multipoles]
        if self.base.sightline=='local' and self.lmax>0:
            print("Generating spherical harmonics")
            self.Ylm_real = self.base._compute_real_harmonics(self.base.r_grids,self.lmax, odd_l=self.odd_l)
            self.Ylm_fourier = self.base._compute_real_harmonics(np.asarray(self.base.k_grids)[:,self.k_filt],self.lmax, odd_l=self.odd_l)
        
        # Define k filters
        self.k_filters = [(self.modk_grid>=self.k_bins[b])&(self.modk_grid<self.k_bins[b+1]) for b in range(self.Nk)]
        self.all_k_filters = np.stack(self.k_filters)
        
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
        
        if self.applySinv==None:
            raise Exception("Must supply S^-1 function to compute unwindowed estimators!")
        
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
        
        # Filter to modes of interest
        Sinv_data_fourier = Sinv_data_fourier[self.k_filt]
                
        # Compute numerator
        real_spec_squared = np.abs(Sinv_data_fourier)**2
        
        # Define output array
        Pk_out = np.zeros(self.N_bins)
        
        # Compute monopole
        for ki in range(self.Nk):
            filt = (self.modk_grid>=self.k_bins[ki])*(self.modk_grid<self.k_bins[ki+1]) # filter to k-bin
            Pk_out[ki] = 0.5*np.sum(real_spec_squared[filt]) # integrate over k
        
        # Compute higher multipoles
        for li in range(1,self.Nl):
            
            if self.base.sightline=='global':
                # Compute L_ell(mu)*|S^-1 d(k)|^2
                real_spec_squared_l = real_spec_squared*legendre(2*li)(self.muk_grid)

                # Integrate over k
                for ki in range(self.Nk):
                    filt = (self.modk_grid>=self.k_bins[ki])*(self.modk_grid<self.k_bins[ki+1])
                    Pk_out[li*self.Nk+ki] = 0.5*np.sum(real_spec_squared_l[filt])
            
            else:
                # Compute Sum_m Y_lm(k)[S^-1 d](k)[S^-1 d]^*_lm(k)
                lm_sum = np.sum([self.Ylm_fourier[(2-self.odd_l)*li][lm_ind]*self.base.to_fourier(Sinv_data_real*self.Ylm_real[(2-self.odd_l)*li][lm_ind])[self.k_filt].conj() for lm_ind in range(len(self.Ylm_real[(2-self.odd_l)*li]))],axis=0)
                if (self.odd_l and li%2==1):
                    spec_squared_l = np.real(1.0j*Sinv_data_fourier*lm_sum)
                else:
                    spec_squared_l = np.real(Sinv_data_fourier*lm_sum)
            
                # Integrate over k
                for ki in range(self.Nk):
                    filt = (self.modk_grid>=self.k_bins[ki])*(self.modk_grid<self.k_bins[ki+1])
                    Pk_out[li*self.Nk+ki] = 0.5*np.sum(spec_squared_l[filt])
                    
        # Add FFT normalization
        Pk_out *= self.base.volume/self.base.gridsize.prod()**2
        
        return Pk_out
        
    def compute_fisher_contribution(self, seed, verb=False):
        """This computes the contribution to the Fisher matrix from a single GRF simulation, created internally. This also computes the contribution to the shot-noise power spectrum."""
                
        if self.applySinv==None:
            raise Exception("Must supply S^-1 function to compute unwindowed estimators!")

        # Initialize output
        fish = np.zeros((self.N_bins,self.N_bins))
        
        # Compute a random realization with known power spectrum
        if verb: print("Generating GRF")
        a_map_fourier = self.base.generate_data(seed=seed+int(1e7), output_type='fourier')
        if not self.const_mask:
            a_map_real = self.base.to_real(a_map_fourier)
        
        # Define k-space filtering
        filt = lambda ki: (self.modk_grid>=self.k_bins[ki])*(self.modk_grid<self.k_bins[ki+1])
        
        # Define Q map code
        def compute_Q(weighting):
            """
            Assemble and return the Q [ = partial_l,b . xi . weighting . a] maps in Fourier-space, given a weighting scheme.

            The outputs are either Q(l,b) or S^-1.W.Q(l,b).
            """
            
            # Weight maps appropriately
            if weighting=='Sinv':
                # Compute S^-1 W a
                if self.const_mask:
                    weighted_map_fourier = self.applySinv(a_map_fourier, input_type='fourier', output_type='fourier')*self.mask_mean
                else:
                    weighted_map_fourier = self.applySinv(self.mask*a_map_real, input_type='real', output_type='fourier')
            else:
                # Compute A^-1 a
                weighted_map_fourier = self.base.applyAinv(a_map_fourier, input_type='fourier', output_type='fourier')
                
            # Define real-space map (where necessary), and drop Fourier-modes out of range
            if self.base.sightline=='local' and self.lmax>0:
                weighted_map_real = self.base.to_real(weighted_map_fourier)
            weighted_map_fourier = weighted_map_fourier[self.k_filt]
            
            # Define arrays
            Q_maps = np.zeros((self.N_bins,self.base.gridsize.prod()),dtype='complex')

            def _apply_weighting(input_map, real=False):
                """Apply S^-1.W weighting to maps if weighting=='Ainv', else return map. Input is either a Fourier-space map or a real-space map."""
                if weighting=='Ainv':
                    if self.const_mask:
                        if real:
                            # Use real-space map directly
                            return self.applySinv(input_map,input_type='real',output_type='fourier')*self.mask_mean
                        else:
                            # Cast to full Fourier-space
                            k_map = np.zeros(self.base.gridsize,dtype='complex')
                            k_map[self.k_filt] = input_map
                            return self.applySinv(k_map,input_type='fourier',output_type='fourier')*self.mask_mean                        
                    else:
                        if real:
                            # Use real-space map directly
                            return self.applySinv(self.mask*input_map,input_type='real',output_type='fourier')
                        else:
                            # Cast to full Fourier-space
                            k_map = np.zeros(self.base.gridsize,dtype='complex')
                            k_map[self.k_filt] = input_map
                            return self.applySinv(self.mask*self.base.to_real(k_map),input_type='real',output_type='fourier')
                else:
                    if real:
                        # Convert to Fourier space and filter
                        return self.base.to_fourier(input_map)
                    else:
                        # Cast to full Fourier-space
                        k_map = np.zeros(self.base.gridsize,dtype='complex')
                        k_map[self.k_filt] = input_map
                        return k_map
            
            # Compute Q derivative for the monopole, optionally adding S^-1.W weighting
            for ki in range(self.Nk):
                Q_maps[ki,:] = _apply_weighting(weighted_map_fourier*filt(ki), real=False).ravel()

            # Repeat for higher-order multipoles
            for li in range(1,self.Nl):

                if self.base.sightline=='global':
                    # Compute L_ell(mu)* W U^-1 a
                    leg_map = weighted_map_fourier*legendre(li*2)(self.muk_grid)
                    
                    # Add to bins
                    for ki in range(self.Nk):
                        Q_maps[li*self.Nk+ki,:] = _apply_weighting(leg_map*filt(ki), real=False).ravel()                           

                else:
                    # First part: (-1)^l Theta_b(k) Sum_m Y_lm(k)* [U^-1 a]_lm(k)
                    leg_map = np.sum([self.Ylm_fourier[(2-self.odd_l)*li][lm_ind]*self.base.to_fourier(weighted_map_real*self.Ylm_real[(2-self.odd_l)*li][lm_ind])[self.k_filt] for lm_ind in range(len(self.Ylm_real[(2-self.odd_l)*li]))],axis=0)
                    
                    # Add phase for odd ell
                    if (self.odd_l and li%2==1):
                        leg_map *= -1
                    
                    # Add to bins
                    for ki in range(self.Nk):
                        
                        # Add first part
                        Q_maps[li*self.Nk+ki,:] += 0.5*_apply_weighting(leg_map*filt(ki), real=False).ravel()

                        # Second part: Sum_m Y_lm (x) Int_k e^ik.x Theta_b(k) Y_lm(k)*[U^-1 a](k)                        
                        real_map = np.zeros(self.base.gridsize,dtype='complex')
                        for lm_ind in range(len(self.Ylm_real[(2-self.odd_l)*li])):
                            # Cast to full Fourier-space map
                            k_map = np.zeros(self.base.gridsize,dtype='complex')
                            k_map[self.k_filt] = weighted_map_fourier*self.Ylm_fourier[(2-self.odd_l)*li][lm_ind]*filt(ki)
                            real_map += self.base.to_real(k_map)*self.Ylm_real[(2-self.odd_l)*li][lm_ind]
                        
                        # Add second part, using the real-space map [which fills all Fourier modes, not just those in [k_min, k_max]]
                        Q_maps[li*self.Nk+ki,:] += 0.5*_apply_weighting(real_map, real=True).ravel()
                        
                        # Add 1.0j to imaginary parts to keep maps real
                        if (self.odd_l and li%2==1):
                            Q_maps[li*self.Nk+ki] *= 1.0j

            return Q_maps    
        
        if verb: print("Computing Q[S^-1.W.a] maps")
        Q_Sinv = compute_Q('Sinv')

        if verb: print("Computing S^-1.W.Q[A^-1.a] maps")
        Q_Ainv = compute_Q('Ainv')
        
        # Assemble Fisher matrix
        if verb: print("Assembling Fisher matrix")

        # Compute Fisher matrix as an outer product
        fish = 0.5*np.real(np.matmul(Q_Sinv.conj(),Q_Ainv.T))*self.base.volume/self.base.gridsize.prod()**2
        
        ## Compute shot-noise, as 1/2 Q2[S^-1.W.a] * S^-1.A^-1.a
        # Compute S^-1.A^-1.a
        if verb: print("Assembling shot-noise\n")
        del Q_Ainv
        Sinv_Ainv_a = self.applySinv(self.base.applyAinv(a_map_fourier, input_type='fourier', output_type='fourier'), input_type='fourier', output_type='fourier').ravel()
        
        # Assemble shot-noise
        shot_num = 0.5*np.real(np.inner(Q_Sinv.conj(),Sinv_Ainv_a))*self.base.volume/self.base.gridsize.prod()**2
        
        return fish, shot_num

    def compute_fisher(self, N_it, N_cpus=1, verb=False):
        """
        Compute the Fisher matrix and shot-noise term using N_it realizations. If N_cpus > 1, this parallelizes the operations.
        """
        
        if self.applySinv==None:
            raise Exception("Must supply S^-1 function to compute unwindowed estimators!")

        # Initialize output
        fish = np.zeros((self.N_bins,self.N_bins))
        shot_num = np.zeros((self.N_bins))

        global _iterable
        def _iterable(seed):
            return self.compute_fisher_contribution(seed, verb=verb)
        
        if N_cpus==1:
            for seed in range(N_it):
                if seed%5==0: print("Computing Fisher contribution %d of %d"%(seed+1,N_it))
                out = self.compute_fisher_contribution(seed, verb=verb)
                fish += out[0]/N_it
                shot_num += out[1]/N_it
        else:
            p = mp.Pool(N_cpus)
            print("Computing Fisher contribution from %d Monte Carlo simulations on %d threads"%(N_it, N_cpus))
            all_fish_shot = list(tqdm.tqdm(p.imap_unordered(_iterable,np.arange(N_it)),total=N_it))
            fish = np.sum([fs[0] for fs in all_fish_shot],axis=0)/N_it
            shot_num = np.sum([fs[1] for fs in all_fish_shot],axis=0)/N_it
        
        self.fish = fish
        self.inv_fish = np.linalg.inv(fish)
        self.shot_num = shot_num
        
        return fish, shot_num

    def Pk_unwindowed(self, data, fish=[], shot_num=[], subtract_shotnoise=False):
        """
        Compute the unwindowed power spectrum estimator. 
        
        Note that the Fisher matrix and shot-noise terms must be computed before this is run, or it can be supplied separately.
        
        The Poisson shot-noise can be optionally subtracted. We return the imaginary part of any ell=odd spectra.
        """
        
        if self.applySinv==None:
            raise Exception("Must supply S^-1 function to compute unwindowed estimators!")

        # Compute inverse Fisher
        if len(fish)!=0:
            self.fish = fish
            self.inv_fish = np.linalg.inv(fish)
            
        # Compute shot-noise
        if len(shot_num)!=0 or subtract_shotnoise:
            self.shot_num = shot_num
            self.shot = np.matmul(self.inv_fish,self.shot_num)
        
        if not hasattr(self,'inv_fish'):
            raise Exception("Need to compute Fisher matrix and shot-noise first!")
        
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
       
    ### IDEAL ESTIMATOR
    def Pk_numerator_ideal(self, data):
        """Compute the numerator of the idealized power spectrum estimator, weighting by 1/P_fid(k) within each bin. 
        
        The estimator does *not* use the mask or S_inv weighting schemes. It also applies only for even ell.
        
        This can compute the ideal power spectrum of simulation volumes, or, for suitably normalized input, the FKP power spectrum.
        """

        # First remove the pixel window function
        data_fourier = self.base.to_fourier(data)
        if self.base.pixel_window!='none':
            data_fourier /= self.base.pixel_window_grid
        
        # Apply filtering by inverse P(k) and mask mean
        data_fourier *= self.base.invPk0_grid/np.sqrt(self.sq_mask_mean)
        
        # Compute real-space map where necessary
        if (self.base.sightline=='local' and self.lmax>0):
            data_real = self.base.to_real(data_fourier)
        
        # Filter to modes of interest
        data_fourier = data_fourier[self.k_filt].conj()
        
        # Compute numerator, applying filtering by inverse P(k) and mask mean
        real_spec_squared = np.abs(data_fourier)**2
        
        # Define output array
        Pk_out = np.zeros(self.Nl_even*self.Nk)
        
        # Compute monopole
        for ki in range(self.Nk):
            filt = (self.modk_grid>=self.k_bins[ki])*(self.modk_grid<self.k_bins[ki+1]) # filter to k-bin
            Pk_out[ki] = 0.5*np.sum(real_spec_squared[filt]) # integrate over k
            
        # Compute higher multipoles
        for li in range(1,self.Nl_even):
            
            if self.base.sightline=='global':
                # Compute L_ell(mu)*|S^-1 d(k)|^2
                real_spec_squared_l = real_spec_squared*legendre(li*2)(self.muk_grid)

                # Integrate over k
                for ki in range(self.Nk):
                    filt = (self.modk_grid>=self.k_bins[ki])*(self.modk_grid<self.k_bins[ki+1])
                    Pk_out[li*self.Nk+ki] = 0.5*np.sum(real_spec_squared_l[filt])
            
            else:
                # Compute Sum_m Y_lm(k)[S^-1 d]^*(k)[S^-1 d]_lm(k)
                lm_sum = np.sum([self.Ylm_fourier[2*li][lm_ind]*self.base.to_fourier(data_real*self.Ylm_real[2*li][lm_ind])[self.k_filt] for lm_ind in range(len(self.Ylm_real[2*li]))],axis=0)
                spec_squared_l = np.real(data_fourier*lm_sum)
            
                # Integrate over k
                for ki in range(self.Nk):
                    filt = (self.modk_grid>=self.k_bins[ki])*(self.modk_grid<self.k_bins[ki+1])
                    Pk_out[li*self.Nk+ki] = 0.5*np.sum(spec_squared_l[filt])

        # Add normalization
        Pk_out *= self.base.volume/self.base.gridsize.prod()**2
        
        return Pk_out
                
    def compute_fisher_ideal(self):
        """This computes the idealized Fisher matrix for the power spectrum, weighting by 1/P_fid(k) within each bin. 
        
        If 'local' sightlines are being used, this assumes the distant-observer approximation.
        """
        print("Computing ideal Fisher matrix")
        
        # Define output array
        fish = np.zeros((self.Nl_even*self.Nk,self.Nl_even*self.Nk))
        
        # Define squared inverse-fiducial P(k), filtering to modes of interest
        sq_inv_Pk = self.base.invPk0_grid[self.k_filt]**2.
        
        # Iterate over fields
        for la in range(self.Nl_even):
            if la==0:
                lega = 1
            else:
                lega = legendre(2*la)(self.muk_grid)
            for lb in range(self.Nl_even):
                if lb==0:
                    legab = lega
                else:
                    legab = lega*legendre(2*lb)(self.muk_grid)
                # Assemble fisher matrix
                fish_diag = 0.5*np.sum(sq_inv_Pk*legab*self.all_k_filters,axis=1)
                
                # Add to output array
                fish[la*self.Nk:(la+1)*self.Nk,lb*self.Nk:(lb+1)*self.Nk] = np.diag(fish_diag)
                
        self.fish_ideal = fish
        self.inv_fish_ideal = np.linalg.inv(self.fish_ideal)
        
        return fish
    
    def Pk_ideal(self, data, fish_ideal=[]):
        """Compute the (normalized) idealized power spectrum estimator, weighting by 1/P_fid(k) within each bin. 
        
        The estimator does *not* use the mask or S_inv weighting schemes (except for normalizing by < mask^2 >. It also applies only for even ell.
        
        This can compute the ideal power spectrum of simulation volumes, or, for suitably normalized input, the FKP power spectrum.
        """

        if len(fish_ideal)!=0:
            self.fish_ideal = fish_ideal
            self.inv_fish_ideal = np.linalg.inv(fish_ideal)

        if not hasattr(self,'inv_fish_ideal'):
            self.compute_fisher_ideal()

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

    def compute_theory_matrix(self, seed, k_bins_theory, applySinv_transpose=None, verb=False):
        """This computes the a rectangular matrix used to window-convolve the theory, using a single GRF simulation, created internally.
        
        We must specify the desired theory binning. An optional input is a function that applies S^-T to a map. The code will assume S^-T = S^-1 if this is not supplied."""

        if self.applySinv==None:
            raise Exception("Must supply S^-1 function to compute unwindowed estimators!")

        if type(applySinv_transpose)!=type(lambda x: x):
            print(type(applySinv_transpose))
            applySinv_transpose = self.applySinv
            print("## Caution: assuming S^-1 is symmetric!")

        assert np.max(k_bins_theory)<np.min(self.base.kNy), "k_max must be less than k_Nyquist!"
        assert np.min(k_bins_theory)>=np.max(self.base.kF), "k_min must be at least the k_fundamental!"
        assert np.max(k_bins_theory)<=self.base.Pfid[0][-1], "Fiducial power spectrum should extend up to k_max!"
        assert np.min(k_bins_theory)>=self.base.Pfid[0][0], "Fiducial power spectrum should extend down to k_min!"
        
        # Initialize output
        Nk_th = len(k_bins_theory)-1
        N_bins_th = self.Nl*Nk_th
        fish = np.zeros((self.N_bins,N_bins_th))

        # Compute a random realization with known power spectrum
        if verb: print("Generating GRF")
        a_map_fourier = self.base.generate_data(seed=seed+int(1e7), output_type='fourier')
        if not self.const_mask:
            a_map_real = self.base.to_real(a_map_fourier)
            
        # Define spherical harmonics
        Ylm_fourier = self.base._compute_real_harmonics(self.base.k_grids, self.lmax, self.odd_l)
        
        # Define Q map code
        def compute_Q_Sinv():
            """
            Assemble and return the Q [ = partial_l,b . xi . S^-1 W . a] maps in Fourier-space.

            The outputs is W^TS^-T Q(l,b).
            """
            
            # Define k filter
            filt = lambda ki: (self.base.modk_grid>=self.k_bins[ki])*(self.base.modk_grid<self.k_bins[ki+1])
            
            # Compute S^-1 W a
            if self.const_mask:
                weighted_map_fourier = self.applySinv(a_map_fourier, input_type='fourier', output_type='fourier')*self.mask_mean
            else:
                weighted_map_fourier = self.applySinv(self.mask*a_map_real, input_type='real', output_type='fourier')
        
            # Define real-space map (where necessary)
            if self.base.sightline=='local' and self.lmax>0:
                weighted_map_real = self.base.to_real(weighted_map_fourier)
            
            # Define arrays
            Q_maps = np.zeros((self.N_bins,self.base.gridsize.prod()),dtype='complex')
            
            def _apply_weighting(input_map, real=False):
                """Apply W^T S^-T weighting to maps. Input is either a Fourier-space map or a real-space map."""
                if self.const_mask:
                    if real:
                        return self.mask_mean*applySinv_transpose(input_map,input_type='real',output_type='fourier')
                    else:
                        return self.mask_mean*applySinv_transpose(input_map,input_type='fourier',output_type='fourier')         
                else:
                    if real:
                        # Use real-space map directly
                        return self.base.to_fourier(self.mask*applySinv_transpose(input_map,input_type='real',output_type='real'))
                    else:
                        return self.base.to_fourier(self.mask*applySinv_transpose(input_map,input_type='fourier',output_type='real'))
            
            # Compute Q derivative for the monopole
            for ki in range(self.Nk):
                Q_maps[ki,:] = _apply_weighting(weighted_map_fourier*filt(ki), real=False).ravel()

            # Repeat for higher-order multipoles
            for li in range(1,self.Nl):

                if self.base.sightline=='global':
                    # Compute L_ell(mu)* S^-1 W a
                    leg_map = weighted_map_fourier*legendre(li*2)(self.muk_grid)

                    # Add to bins
                    for ki in range(self.Nk):
                        Q_maps[li*self.Nk+ki,:] = _apply_weighting(leg_map*filt(ki), real=False).ravel()                           

                else:
                    # First part: (-1)^l Theta_b(k) Sum_m Y_lm(k)* [U^-1 a]_lm(k)
                    leg_map = np.sum([Ylm_fourier[(2-self.odd_l)*li][lm_ind]*self.base.to_fourier(weighted_map_real*self.Ylm_real[(2-self.odd_l)*li][lm_ind]) for lm_ind in range(len(self.Ylm_real[(2-self.odd_l)*li]))],axis=0)

                    # Add phase for odd ell
                    if (self.odd_l and li%2==1):
                        leg_map *= -1

                    # Add to bins
                    for ki in range(self.Nk):
                        
                        # Add first part
                        Q_maps[li*self.Nk+ki,:] += 0.5*_apply_weighting(leg_map*filt(ki), real=False).ravel()

                        # Second part: Sum_m Y_lm (x) Int_k e^ik.x Theta_b(k) Y_lm(k)*[S^-1 W a](k)                        
                        real_map = np.zeros(self.base.gridsize,dtype='complex')
                        for lm_ind in range(len(self.Ylm_real[(2-self.odd_l)*li])):
                            # Cast to full Fourier-space map
                            k_map = weighted_map_fourier*Ylm_fourier[(2-self.odd_l)*li][lm_ind]*filt(ki)
                            real_map += self.base.to_real(k_map)*self.Ylm_real[(2-self.odd_l)*li][lm_ind]

                        # Add second part, using the real-space map [which fills all Fourier modes, not just those in [k_min, k_max]]
                        Q_maps[li*self.Nk+ki,:] += 0.5*_apply_weighting(real_map, real=True).ravel()

                        # Add 1.0j to imaginary parts to keep maps real
                        if (self.odd_l and li%2==1):
                            Q_maps[li*self.Nk+ki] *= 1.0j

            return Q_maps    

        def compute_Q_Ainv():
            """
            Assemble and return the Q [ = partial_l,b . xi . weighting . a] maps in Fourier-space.
            """
            
            # Define k filter
            filt = lambda ki: (self.base.modk_grid>=k_bins_theory[ki])*(self.base.modk_grid<k_bins_theory[ki+1])
            
            # Compute A^-1 a
            weighted_map_fourier = self.base.applyAinv(a_map_fourier, input_type='fourier', output_type='fourier')

            # Define real-space map (where necessary), and drop Fourier-modes out of range
            if self.base.sightline=='local' and self.lmax>0:
                weighted_map_real = self.base.to_real(weighted_map_fourier)
            
            # Define arrays
            Q_maps = np.zeros((N_bins_th,self.base.gridsize.prod()),dtype='complex')

            # Compute Q derivative for the monopole, optionally adding S^-1.W weighting
            for ki in range(Nk_th):
                Q_maps[ki,:] = (weighted_map_fourier*filt(ki)).ravel()
    
            # Repeat for higher-order multipoles
            for li in range(1,self.Nl):

                if self.base.sightline=='global':
                    # Compute L_ell(mu)* A-1 a
                    leg_map = weighted_map_fourier*legendre(li*2)(self.muk_grid)

                    # Add to bins
                    for ki in range(Nk_th):
                        Q_maps[li*Nk_th+ki,:] = (leg_map*filt(ki)).ravel()                           
        
                else:
                    
                    # First part: (-1)^l Theta_b(k) Sum_m Y_lm(k)* [U^-1 a]_lm(k)
                    leg_map = np.sum([Ylm_fourier[(2-self.odd_l)*li][lm_ind]*self.base.to_fourier(weighted_map_real*self.Ylm_real[(2-self.odd_l)*li][lm_ind]) for lm_ind in range(len(self.Ylm_real[(2-self.odd_l)*li]))],axis=0)

                    # Add phase for odd ell
                    if (self.odd_l and li%2==1):
                        leg_map *= -1

                    # Add to bins
                    for ki in range(Nk_th):
                        
                        # Add first part, absorbing second by symmetry
                        Q_maps[li*Nk_th+ki,:] += (leg_map*filt(ki)).ravel()
                        
                        # Add 1.0j to imaginary parts to keep maps real
                        if (self.odd_l and li%2==1):
                            Q_maps[li*Nk_th+ki] *= 1.0j

            
            return Q_maps    
                                                        
        if verb: print("Computing W^T.S^-T.Q[S^-1.W.a] maps")
        Q_Sinv = compute_Q_Sinv()

        if verb: print("Computing Q[A^-1.a] maps")
        Q_Ainv = compute_Q_Ainv()

        # Assemble Fisher matrix
        if verb: print("Assembling binning matrix\n")

        # Compute binning matrix as an outer product
        binning_matrix = 0.5*np.real(np.matmul(Q_Sinv.conj(),Q_Ainv.T))*self.base.volume/self.base.gridsize.prod()**2

        return binning_matrix
