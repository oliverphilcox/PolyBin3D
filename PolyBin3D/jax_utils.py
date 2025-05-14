import jax 
import jax.numpy as jnp
from functools import partial
dfloat = jnp.float64
dcomplex = jnp.complex128

class IntegrationUtils:

    def __init__(self, modk, real_fft, nthreads, degk=None, muk=None):
        """Class to perform integrals of pairs of maps across all k-bins. We can use either complex-to-complex or real-to-complex FFTs here.
        There are various functions integrating slightly different things.
        
        This is a JAX version of the class in PolyBin3D/cython/utils.pyx"""

        # Define local attributes
        self.modk = modk
        self.nthreads = nthreads
        self.real_fft = real_fft
        self.muk = muk

        # Load degeneracy factor
        if degk is not None:
            self.degk = degk
        else:
            self.degk = 1
            
    @jax.jit
    def integrate(self, spec, k_edges, l):
        """Compute Sum_k |d(k)|^2 Legendre(l, k) across all k in range."""
        leg = legendre(l, self.muk)
        return jnp.stack([jnp.sum(spec*spec.conjugate()*leg*self.degk*(self.modk>=k_edges[i])*(self.modk<k_edges[i+1])) for i in range(len(k_edges)-1)]).real
         
    @jax.jit
    def cross_integrate(self, spec1, spec2, k_edges, l):
        """Compute Sum_k d1(k)d2(-k) Legendre(l, k) across all k in range."""
        leg = legendre(l, self.muk)
        return jnp.stack([jnp.sum(spec1*spec2.conjugate()*leg*self.degk*(self.modk>=k_edges[i])*(self.modk<k_edges[i+1])) for i in range(len(k_edges)-1)]).real
    
    @jax.jit
    def cross_integrate_wcoeff(self, coeff, spec1, spec2, k_edges):
        """Compute Sum_k d1(k)d2(-k) across all k in range, weighted by complex coefficients."""
        return jnp.stack([coeff[i]*jnp.sum(spec1*spec2.conjugate()*self.degk*(self.modk>=k_edges[i])*(self.modk<k_edges[i+1])) for i in range(len(k_edges)-1)]).real
        
    @jax.jit
    def real_integrate_double(self, spec, l1, l2, k_edges):
        """Compute Sum_k d(k)^2 L_l1(k.n)L_l2(k.n) across all k in range for real d(k)."""
        
        # Define Legendre polynomial  
        leg = legendre(l1, self.muk)*legendre(l2, self.muk)

        # Sum over k
        return jnp.stack([jnp.sum(spec**2*leg*self.degk*(self.modk>=k_edges[i])*(self.modk<k_edges[i+1])) for i in range(len(k_edges)-1)])
         
    @jax.jit
    def integrate_row(self, input_map, input_array, kmin, kmax):
        """Compute Sum_k d(k)e_i(k)* across all k in [kmin, kmax)."""
        return jnp.sum(input_array*input_map.conjugate()*self.degk*(self.modk>=kmin)*(self.modk<kmax), axis=(1,2,3)).real

    def _tree_flatten(self):
        children = ()  # arrays / dynamic values
        aux_data = {'modk': self.modk, 'real_fft': self.real_fft, 'nthreads': self.nthreads, 'degk': self.degk, 'muk': self.muk}  # static values
        return (children, aux_data)
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
class MapUtils:

    def __init__(self, modk, gridsize, nthreads, muk=None):
        """Class to perform various low-level map related functions.
        
        This is a JAX version of the class in PolyBin3D/cython/utils.pyx. Most functions are trivial."""

        # Define local memviews
        self.gridsize = gridsize
        self.nthreads = nthreads
        self.modk = modk
        self.muk = muk
         
    @jax.jit
    def prod_real(self, map1, map2):
        return map1*map2
         
    @jax.jit
    def prod_fourier(self, map1, map2):
        return map1*map2
         
    @jax.jit
    def div_fourier(self, map1, map2):
        return jnp.where(map2!=0, map1/map2, 0)
         
    @jax.jit
    def div_real(self, map1, map2):
        return jnp.where(map2!=0, map1/map2, 0)
        
    @jax.jit
    def fourier_filter(self, kmap, l, kmin, kmax):
        """Filter f(k) to Fourier modes with k in [kmin, kmax), and multiply by a Legendre polynomial."""
        return kmap*legendre(l, self.muk)*(self.modk>=kmin)*(self.modk<kmax)

    @jax.jit
    def fourier_filter_ll(self, kmap, l1, l2, kmin, kmax):
        """Filter f(k) to Fourier modes with k in [kmin, kmax), and multiply by a Legendre polynomial."""
        return kmap*legendre(l1, self.muk)*legendre(l2, self.muk)*(self.modk>=kmin)*(self.modk<kmax)
         
    @jax.jit
    def prod_fourier_filter(self, map1, map2, kmin, kmax):
        """Compute map1(k)map2(k) in Fourier-space (where map2 is real), filtering to Fourier-modes with k in [kmin, kmax)."""
        return map1*map2*(self.modk>=kmin)*(self.modk<kmax)
         
    @jax.jit
    def sum_pair(self, map1, map2):
        """Sum over a pair of real-space maps."""
        return jnp.sum(map1*map2)
         
    @jax.jit
    def sum_triplet(self, map1, map2, map3):
        return jnp.sum(map1*map2*map3)
    
    @jax.jit
    def prod_real_diff(self, map1, map2, map3, map4):
        """Compute the difference of map1(x)map2(x) and map3(x)map4(x) (all real)."""
        return map1*map2-map3*map4
     
    def _tree_flatten(self):
        children = ()  # arrays / dynamic values
        aux_data = {'modk': self.modk, 'gridsize': self.gridsize, 'nthreads': self.nthreads, 'muk': self.muk}  # static values
        return (children, aux_data)
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

from jax import tree_util
tree_util.register_pytree_node(IntegrationUtils,IntegrationUtils._tree_flatten,IntegrationUtils._tree_unflatten)
tree_util.register_pytree_node(MapUtils,MapUtils._tree_flatten,MapUtils._tree_unflatten)
     
@jax.jit
def legendre(l, mu):
    """Compute Legendre polynomial L_l(mu) for l in [0, 6], using explicit formulas."""
    
    def L0(mu): return 1.+0.*mu
    def L1(mu): return mu
    def L2(mu): return 0.5 * (3 * mu**2 - 1)
    def L3(mu): return 0.5 * (-3 * mu + 5 * mu**3)
    def L4(mu): return (3 - 30 * mu**2 + 35 * mu**4) / 8.
    def L5(mu): return (15 * mu - 70 * mu**3 + 63 * mu**5) / 8.
    def L6(mu): return (-5 + 105 * mu**2 - 315 * mu**4 + 231 * mu**6) / 16.

    funcs = [L0, L1, L2, L3, L4, L5, L6]
    
    return jax.lax.switch(l, funcs, mu)           

@jax.jit
def pointing_GIC(map1, mask, mask_GIC, nthreads):
    """Compute pointing matrix with global integral constraint."""
    av_map = jnp.sum(map1*mask_GIC)/jnp.sum(mask_GIC)
    return mask*(map1-av_map)

@jax.jit
def pointing_GIC_transpose(map1, mask, mask_GIC, nthreads):
    """Compute pointing matrix transpose with global integral constraint."""
    av_map = jnp.sum(map1*mask)/jnp.sum(mask_GIC)
    return mask*map1 - mask_GIC*av_map
     
@jax.jit
def pointing_RIC(map1, mask, mask_GIC, radial_edges, radial_grid, nthreads):
    """Compute pointing matrix with radial integral constraint."""
    
    output = jnp.zeros_like(map1)
    for i in range(len(radial_edges)-1):
        radial_slice = (radial_grid>=radial_edges[i])*(radial_grid<radial_edges[i+1])
        # Compute average of data in each radial bin
        av_map = jnp.sum(map1*mask_GIC*radial_slice)/jnp.sum(mask_GIC*radial_slice)
        # Combine to form pointing matrix
        output += mask*(map1-av_map)*radial_slice
    return output

@jax.jit
def pointing_RIC_transpose(map1, mask, mask_GIC, radial_edges, radial_grid, nthreads):
    """Compute pointing matrix transpose with radial integral constraint."""
    
    output = jnp.zeros_like(map1)
    for i in range(len(radial_edges)-1):
        radial_slice = (radial_grid>=radial_edges[i])*(radial_grid<radial_edges[i+1])
        # Compute average of data in each radial bin
        av_map = jnp.sum(map1*mask*radial_slice)/jnp.sum(mask_GIC*radial_slice)
        # Add to output
        output += (mask*map1-mask_GIC*av_map)*radial_slice
    return output
    
def compute_real_harmonics(coordinates, lmax, odd_l, nthreads):
    """Compute the real spherical harmonics on the coordinate grid."""
    assert lmax >= 1
    Ylms = {}
    
    # Define coordinates (with unit norm)
    norm = jnp.sqrt(coordinates[0]**2 + coordinates[1]**2 + coordinates[2]**2)
    xh, yh, zh = jnp.where(norm==0, 0, coordinates/norm)

    # Compute spherical harmonics
    if odd_l and lmax >= 1:
        Ylms[1] = jnp.stack([yh, zh, xh], axis=0)

    if lmax >= 2:
        ylm2 = jnp.stack([
            6. * xh * yh * jnp.sqrt(1. / 12.),
            3. * yh * zh * jnp.sqrt(1. / 3.),
            zh**2 - 0.5 * (xh**2 + yh**2),
            3. * xh * zh * jnp.sqrt(1. / 3.),
            (3. * xh**2 - 3. * yh**2) * jnp.sqrt(1. / 12.)
        ], axis=0)
        Ylms[2] = ylm2

    if odd_l and lmax >= 3:
        ylm3 = jnp.stack([
            (45. * xh**2 * yh - 15. * yh**3) * jnp.sqrt(1. / 360.),
            30. * xh * yh * zh * jnp.sqrt(1. / 60.),
            (-1.5 * xh**2 * yh - 1.5 * yh**3 + 6. * yh * zh**2) * jnp.sqrt(1. / 6.),
            (-1.5 * xh**2 * zh - 1.5 * yh**2 * zh + zh**3),
            (-1.5 * xh**3 - 1.5 * xh * yh**2 + 6. * xh * zh**2) * jnp.sqrt(1. / 6.),
            (15. * xh**2 * zh - 15. * yh**2 * zh) * jnp.sqrt(1. / 60.),
            (15. * xh**3 - 45. * xh * yh**2) * jnp.sqrt(1. / 360.)
        ], axis=0)
        Ylms[3] = ylm3

    if lmax >= 4:
        ylm4 = jnp.stack([
            (420. * xh**3 * yh - 420. * xh * yh**3) * jnp.sqrt(1. / 20160.),
            (315. * xh**2 * yh * zh - 105. * yh**3 * zh) * jnp.sqrt(1. / 2520.),
            (-15. * xh**3 * yh - 15. * xh * yh**3 + 90. * xh * yh * zh**2) * jnp.sqrt(1. / 180.),
            (-7.5 * xh**2 * yh * zh - 7.5 * yh**3 * zh + 10. * yh * zh**3) * jnp.sqrt(1. / 10.),
            35. / 8. * zh**4 - 15. / 4. * zh**2 + 3. / 8.,
            (-15. / 2. * xh**3 * zh - 15. / 2. * xh * yh**2 * zh + 10. * xh * zh**3) * jnp.sqrt(1. / 10.),
            (-15. / 2. * xh**4 + 45. * xh**2 * zh**2 + 15. / 2. * yh**4 - 45. * yh**2 * zh**2) * jnp.sqrt(1. / 180.),
            (105. * xh**3 * zh - 315. * xh * yh**2 * zh) * jnp.sqrt(1. / 2520.),
            (105. * xh**4 - 630. * xh**2 * yh**2 + 105. * yh**4) * jnp.sqrt(1. / 20160.)
        ], axis=0)
        Ylms[4] = ylm4

    return Ylms
     
@jax.jit
def assemble_bshot(gA_maps, gB_maps, gC_maps, all_bins, nthreads):
    """Assemble the bispectrum numerator from a set of g_lb(x) maps."""
    n_bins = len(all_bins)
    out = jnp.zeros(n_bins, dtype=dfloat)

    for index in range(n_bins):
        # Define bins
        bin1,bin2,bin3,l = all_bins[index]
        
        # Sum over maps
        out = out.at[index].set(jnp.sum(gA_maps[0][bin1]*gB_maps[0][bin2]*gC_maps[l//2][bin3]+gA_maps[0][bin1]*gC_maps[0][bin2]*gB_maps[l//2][bin3]+gB_maps[0][bin1]*gA_maps[0][bin2]*gC_maps[l//2][bin3]+gB_maps[0][bin1]*gC_maps[0][bin2]*gA_maps[l//2][bin3]+gC_maps[0][bin1]*gA_maps[0][bin2]*gB_maps[l//2][bin3]+gC_maps[0][bin1]*gB_maps[0][bin2]*gA_maps[l//2][bin3]))
        
    return out
     
@jax.jit
def assemble_b1(g_maps, simg_maps, all_bins, nthreads):
    """Assemble the bispectrum numerator from a set of g_lb(x) maps."""
    n_bins = len(all_bins)
    out = jnp.zeros(n_bins, dtype=dfloat)

    for index in range(n_bins):
        # Define bins
        bin1,bin2,bin3,l = all_bins[index]
        
        # Sum over maps
        out = out.at[index].set(jnp.sum(g_maps[0][bin1]*simg_maps[0][bin2]*simg_maps[l//2][bin3])+jnp.sum(simg_maps[0][bin1]*g_maps[0][bin2]*simg_maps[l//2][bin3])+jnp.sum(simg_maps[0][bin1]*simg_maps[0][bin2]*g_maps[l//2][bin3]))
        
    return out

@jax.jit
def assemble_b3(g_maps, all_bins, nthreads):
    """Assemble the bispectrum numerator from a set of g_lb(x) maps."""
    n_bins = len(all_bins)
    out = jnp.zeros(n_bins, dtype=dfloat)

    for index in range(n_bins):
        # Define bins
        bin1,bin2,bin3,l = all_bins[index]
        
        # Sum over maps
        out = out.at[index].set(jnp.sum(g_maps[0][bin1]*g_maps[0][bin2]*g_maps[l//2][bin3]))
    return out

@jax.jit
def jax_fft(x):
    return jnp.fft.fftn(x,axes=(-3,-2,-1))

@jax.jit
def jax_rfft(x):
    return jnp.fft.rfftn(x,axes=(-3,-2,-1))

@jax.jit
def jax_ifft(x):
    return jnp.fft.ifftn(x,axes=(-3,-2,-1)).real

@jax.jit
def jax_irfft(x):
    return jnp.fft.irfftn(x,axes=(-3,-2,-1))

@partial(jax.jit, static_argnums=1)
def apply_fourier_harmonics(a_map_real, to_fourier, Ylm_real, Ylm_fourier):
    """Compute Sum_m Y_lm(k) a_lm^*(k) for some a(x)"""
    lm_sum = jnp.zeros_like(Ylm_fourier[0])
    for lm_ind in range(len(Ylm_real)):
        map_lm = a_map_real*Ylm_real[lm_ind]
        lm_sum += to_fourier(map_lm)*Ylm_fourier[lm_ind]
    return lm_sum

@partial(jax.jit, static_argnums=1)
def apply_real_harmonics(a_map_fourier, to_real, Ylm_real, Ylm_fourier):
    """Compute Sum_m Y_lm(x) a_lm^*(x) for some a(k)"""
    lm_sum = jnp.zeros_like(Ylm_real[0])
    for lm_ind in range(len(Ylm_fourier)):
        map_lm = a_map_fourier*Ylm_fourier[lm_ind]
        lm_sum += to_real(map_lm)*Ylm_real[lm_ind]
    return lm_sum