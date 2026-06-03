"""
Shared data loading, gridding, and PolyBin3D setup for DESI Pk + Bk analysis.
All functions take a cfg (Config object) as their first argument.
"""
import sys, os, time
import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import PolyBin3D as pb


def load_cosmology(cfg):
    """Initialize CLASS and return cosmology object + distance interpolators."""
    from classy import Class
    cosmo = Class()
    cosmo.set({
        'h': cfg.H_FID, 'omega_b': cfg.OMEGA_B, 'omega_cdm': cfg.OMEGA_CDM,
        'Omega_k': 0, 'N_ur': cfg.N_UR, 'N_ncdm': 1, 'm_ncdm': cfg.M_NU,
        'output': 'mPk', 'z_pk': 3., 'P_k_max_h/Mpc': 15.,
    })
    cosmo.compute()

    z_arr = np.linspace(0, 20, int(1e6))
    d_arr = np.array([
        np.sqrt(cosmo.luminosity_distance(z) * cosmo.angular_distance(z)) * cfg.H_FID
        for z in z_arr
    ])
    comoving_distance = interp1d(z_arr, d_arr)
    z_from_comoving = interp1d(d_arr, z_arr)

    return cosmo, comoving_distance, z_from_comoving


def load_catalogs(cfg, sample, hemisphere, zid, include_weights=True):
    """Load DESI FITS catalogs, filter to redshift bin, return (dat, ran)."""
    zmin, zmax, dz, P_fkp, fits_name = cfg.get_sample_params(sample, zid)

    datfile = os.path.join(cfg.DATA_DIR, f'{fits_name}_{hemisphere}_clustering.dat.fits')
    _dat = fits.open(datfile)[1].data
    _ran = np.concatenate([
        fits.open(os.path.join(cfg.DATA_DIR, f'{fits_name}_{hemisphere}_{i}_clustering.ran.fits'))[1].data
        for i in range(cfg.MAX_RAN)
    ])

    dat = _dat[(_dat['Z'] >= zmin) & (_dat['Z'] < zmax)]
    ran = _ran[(_ran['Z'] >= zmin) & (_ran['Z'] < zmax)]
    print(f"Using {len(dat)} data and {len(ran)} randoms")

    if not include_weights:
        dat['WEIGHT'] = dat['WEIGHT'] / dat['WEIGHT_SYS']
        ran['WEIGHT'] = ran['WEIGHT'] / ran['WEIGHT_SYS']

    return dat, ran


def sky_to_cartesian(ra, dec, z, comoving_distance):
    """Convert (RA, DEC, z) to Cartesian coordinates [Mpc/h]."""
    pos = np.array([
        np.cos(np.radians(dec)) * np.cos(np.radians(ra)),
        np.cos(np.radians(dec)) * np.sin(np.radians(ra)),
        np.sin(np.radians(dec)),
    ]).T
    return pos * comoving_distance(z)[:, None]


def compute_z_eff(ran, comoving_distance, zmin, zmax, dz):
    """Compute FKP-weighted effective redshift."""
    z_edges_fine = np.arange(zmin, zmax + 1e-6, dz / 10.)
    z_hist = np.histogram(ran['Z'], bins=z_edges_fine,
                          weights=ran['WEIGHT'] * ran['WEIGHT_FKP'])[0]
    z_av = 0.5 * (z_edges_fine[:-1] + z_edges_fine[1:])
    V_bins = 4. / 3. * np.pi * np.array([
        comoving_distance(z_edges_fine[i + 1]) ** 3 - comoving_distance(z_edges_fine[i]) ** 3
        for i in range(len(z_edges_fine) - 1)
    ])
    nz = z_hist / np.sum(z_hist) / V_bins
    return np.sum(V_bins * nz ** 2 * z_av) / np.sum(V_bins * nz ** 2)


def compute_fiducial_pk(cfg, cosmo, z_eff):
    """Compute fiducial linear P(k) from CLASS at z_eff."""
    k_arr = np.logspace(-4, 1, 100)
    pk_fid = np.array([cosmo.pk_lin(kk * cfg.H_FID, z_eff) * cfg.H_FID ** 3 for kk in k_arr])
    return k_arr, pk_fid


def create_grids(cfg, dat, ran, dat_pos, ran_pos, boxsize, gridsize,
                 out_str, proc_dir, include_weights=True, reload_grids=False):
    """Paint data/randoms to grid or load from cache. Return all grid products."""
    os.makedirs(proc_dir, exist_ok=True)

    weight_str = '' if include_weights else '_noweight'
    dat_grid_file = os.path.join(proc_dir, f'{out_str}dat_grid{weight_str}.npy')
    ran_grid_file = os.path.join(proc_dir, f'{out_str}ran_grid{weight_str}.npy')
    ran_shot_file = os.path.join(proc_dir, f'{out_str}ran_shot_grid{weight_str}.npy')
    ran_ic_file = os.path.join(proc_dir, f'{out_str}ran_ic_grid{weight_str}.npy')

    files_exist = all(os.path.exists(f) for f in [dat_grid_file, ran_grid_file, ran_shot_file, ran_ic_file])

    if reload_grids or not files_exist:
        from nbodykit.lab import ArrayCatalog
        print("Painting to grid...")

        boxcenter = (np.max(dat_pos, axis=0) + np.min(dat_pos, axis=0)) / 2.
        dat_pos_c = dat_pos - boxcenter
        ran_pos_c = ran_pos - boxcenter

        def _paint(positions, weights):
            cat = ArrayCatalog({'Position': positions, 'WEIGHT': weights})
            return cat.to_mesh(Nmesh=gridsize, BoxSize=boxsize, interlaced=True,
                               compensated=True, position='Position', weight='WEIGHT',
                               resampler='tsc').compute()

        np.save(dat_grid_file, _paint(dat_pos_c, dat['WEIGHT'] * dat['WEIGHT_FKP']))
        np.save(ran_grid_file, _paint(ran_pos_c, ran['WEIGHT'] * ran['WEIGHT_FKP']))
        np.save(ran_shot_file, _paint(ran_pos_c, ran['WEIGHT'] ** 2 * ran['WEIGHT_FKP'] ** 2))
        np.save(ran_ic_file, _paint(ran_pos_c, ran['WEIGHT']))

    print("Loading gridded data from file")
    V = boxsize.prod()
    dat_grid = np.load(dat_grid_file) * np.sum(dat['WEIGHT'] * dat['WEIGHT_FKP']) / V
    ran_grid = np.load(ran_grid_file) * np.sum(ran['WEIGHT'] * ran['WEIGHT_FKP']) / V
    ran_ic_grid = np.load(ran_ic_file) * np.sum(ran['WEIGHT']) / V
    ran_shot_raw = np.load(ran_shot_file) * np.sum(ran['WEIGHT'] ** 2 * ran['WEIGHT_FKP'] ** 2) / V

    alpha = np.sum(dat['WEIGHT'] * dat['WEIGHT_FKP']) / np.sum(ran['WEIGHT'] * ran['WEIGHT_FKP'])
    alpha_2 = np.sum(dat['WEIGHT'] ** 2 * dat['WEIGHT_FKP'] ** 2) / np.sum(ran['WEIGHT'] ** 2 * ran['WEIGHT_FKP'] ** 2)

    density_grid = (dat_grid - alpha * ran_grid).astype(np.float64)
    mask = (alpha * ran_grid).astype(np.float64)
    mask_ic = (alpha * ran_ic_grid).astype(np.float64)
    mask_shot = ((alpha_2 + alpha ** 2) * ran_shot_raw).astype(np.float64)

    return density_grid, mask, mask_ic, mask_shot, dat_grid, ran_grid, alpha


def setup_polybin(cfg, boxsize, gridsize, boxcenter, k_arr, pk_fid):
    """Create PolyBin3D base class."""
    return pb.PolyBin3D(
        boxsize, gridsize,
        Pk=[k_arr, pk_fid],
        boxcenter=boxcenter,
        pixel_window='none',
        backend=cfg.FFT_BACKEND,
        nthreads=int(os.environ.get('SLURM_CPUS_PER_TASK', os.environ.get('SLURM_CPUS_ON_NODE', os.cpu_count() or 32))),
        sightline='local',
    )


def _make_applySinv(base):
    """Create identity S^-1 weighting function."""
    def applySinv(input_data, input_type='real', output_type='real'):
        if input_type == 'fourier':
            return input_data if output_type == 'fourier' else base.to_real(input_data)
        else:
            return base.to_fourier(input_data) if output_type == 'fourier' else input_data
    return applySinv


def setup_pspec(cfg, base, mask, mask_ic, mask_shot, r_edges, k_edges=None):
    """Create PSpec estimator class."""
    if k_edges is None:
        k_edges = cfg.K_EDGES
    applySinv = _make_applySinv(base)
    return pb.PSpec(
        base, k_edges,
        applySinv=applySinv,
        mask=mask, lmax=cfg.LMAX, odd_l=False,
        add_GIC=cfg.ADD_GIC, add_RIC=cfg.ADD_RIC,
        mask_IC=mask_ic, radial_bins_RIC=r_edges,
        mask_shot=mask_shot, applySinv_transpose=applySinv,
    )


def full_setup(cfg, sample, hemisphere, zid, include_weights=True, reload_grids=False):
    """Complete Pk setup. Returns dict with all products."""
    zmin, zmax, dz, P_fkp, fits_name = cfg.get_sample_params(sample, zid)
    out_str = f'{sample}_{hemisphere}_z{zid}'

    os.makedirs(cfg.PK_OUT_DIR, exist_ok=True)
    outfile_base = os.path.join(cfg.PK_OUT_DIR, out_str + 'pk')
    if not include_weights:
        outfile_base += '_noweight'
    if cfg.ADD_GIC:
        outfile_base += '_GIC'
    elif cfg.ADD_RIC:
        outfile_base += '_RIC'
    else:
        outfile_base += '_simple'

    print(f"\n### {sample} {hemisphere} z{zid} [{zmin:.1f}, {zmax:.1f}]")

    print("Loading cosmology...")
    cosmo, comoving_distance, z_from_comoving = load_cosmology(cfg)

    print("Loading data...")
    dat, ran = load_catalogs(cfg, sample, hemisphere, zid, include_weights)

    z_eff = compute_z_eff(ran, comoving_distance, zmin, zmax, dz)
    print(f"z_eff = {z_eff:.3f}")
    k_arr, pk_fid = compute_fiducial_pk(cfg, cosmo, z_eff)

    z_edges = np.arange(zmin, zmax + 1e-6, dz)
    r_edges = np.array([comoving_distance(z) for z in z_edges])

    print("Converting coordinates...")
    dat_pos = sky_to_cartesian(dat['RA'], dat['DEC'], dat['Z'], comoving_distance)
    ran_pos = sky_to_cartesian(ran['RA'], ran['DEC'], ran['Z'], comoving_distance)

    boxsize = 1000 + (np.max(dat_pos, axis=0) - np.min(dat_pos, axis=0))
    boxcenter = (np.max(dat_pos, axis=0) + np.min(dat_pos, axis=0)) / 2.
    gridsize = np.asarray(np.ceil(1.01 * cfg.KMAX_GRID / (np.pi / boxsize)), dtype=int)
    gridsize = 2 * (gridsize // 2)

    density_grid, mask, mask_ic, mask_shot, dat_grid, ran_grid, alpha = \
        create_grids(cfg, dat, ran, dat_pos, ran_pos, boxsize, gridsize,
                     out_str, cfg.PK_PROC_DIR, include_weights, reload_grids)

    print("Loading PolyBin3D...")
    base = setup_polybin(cfg, boxsize, gridsize, boxcenter, k_arr, pk_fid)
    k_edges = cfg.get_k_edges(sample)
    pspec = setup_pspec(cfg, base, mask, mask_ic, mask_shot, r_edges, k_edges=k_edges)

    return {
        'dat': dat, 'ran': ran,
        'density_grid': density_grid, 'mask': mask, 'mask_ic': mask_ic,
        'mask_shot': mask_shot, 'dat_grid': dat_grid, 'ran_grid': ran_grid,
        'alpha': alpha,
        'base': base, 'pspec': pspec,
        'boxsize': boxsize, 'gridsize': gridsize, 'boxcenter': boxcenter,
        'z_eff': z_eff, 'r_edges': r_edges,
        'k_arr': k_arr, 'pk_fid': pk_fid,
        'comoving_distance': comoving_distance,
        'z_from_comoving': z_from_comoving,
        'out_str': out_str, 'outfile_base': outfile_base,
    }


# =========================================================================
#  Bk-specific functions
# =========================================================================

def rescale_fkp_for_bk(dat, ran, comoving_distance, zmin, zmax, dz):
    """Rescale FKP weights by n(z)^{-1/3} for optimal Bk weighting.

    Modifies dat/ran WEIGHT_FKP in-place. Returns (z_eff_pk, z_eff_bk).
    """
    z_edges_fine = np.arange(zmin, zmax + 1e-6, dz / 10.)
    z_av = 0.5 * (z_edges_fine[:-1] + z_edges_fine[1:])
    V_bins = 4. / 3. * np.pi * np.array([
        comoving_distance(z_edges_fine[i + 1]) ** 3 - comoving_distance(z_edges_fine[i]) ** 3
        for i in range(len(z_edges_fine) - 1)
    ])

    z_hist = np.histogram(ran['Z'], bins=z_edges_fine,
                          weights=ran['WEIGHT'] * ran['WEIGHT_FKP'])[0]
    nz = z_hist / np.sum(z_hist) / V_bins
    z_eff_pk = np.sum(V_bins * nz ** 2 * z_av) / np.sum(V_bins * nz ** 2)

    nz_func = interp1d(z_av, nz, kind='nearest', fill_value='extrapolate')
    rescale_norm = np.mean(nz_func(ran['Z']))
    ran['WEIGHT_FKP'] *= (nz_func(ran['Z']) / rescale_norm) ** (-1. / 3.)
    dat['WEIGHT_FKP'] *= (nz_func(dat['Z']) / rescale_norm) ** (-1. / 3.)

    z_hist = np.histogram(ran['Z'], bins=z_edges_fine,
                          weights=ran['WEIGHT'] * ran['WEIGHT_FKP'])[0]
    nz = z_hist / np.sum(z_hist) / V_bins
    z_eff_bk = np.sum(V_bins * nz ** 3 * z_av) / np.sum(V_bins * nz ** 3)

    return z_eff_pk, z_eff_bk


def create_grids_bk(cfg, dat, ran, dat_pos, ran_pos, boxsize, gridsize,
                    out_str, include_weights=True, reload_grids=False):
    """Paint Bk-specific grids (includes doubly and triply weighted fields)."""
    os.makedirs(cfg.BK_PROC_DIR, exist_ok=True)

    weight_str = '' if include_weights else '_noweight'
    dat_grid_file = os.path.join(cfg.BK_PROC_DIR, f'{out_str}dat_grid{weight_str}_bis.npy')
    ran_grid_file = os.path.join(cfg.BK_PROC_DIR, f'{out_str}ran_grid{weight_str}_bis.npy')
    ran_shot2_file = os.path.join(cfg.BK_PROC_DIR, f'{out_str}ran_shot_grid{weight_str}_bis2.npy')
    dat_shot2_file = os.path.join(cfg.BK_PROC_DIR, f'{out_str}dat_shot_grid{weight_str}_bis2.npy')
    ran_shot3_file = os.path.join(cfg.BK_PROC_DIR, f'{out_str}ran_shot_grid{weight_str}_bis3.npy')
    ran_ic_file = os.path.join(cfg.BK_PROC_DIR, f'{out_str}ran_ic_grid{weight_str}_bis.npy')

    all_files = [dat_grid_file, ran_grid_file, ran_shot2_file, dat_shot2_file,
                 ran_shot3_file, ran_ic_file]

    if reload_grids or not all(os.path.exists(f) for f in all_files):
        from nbodykit.lab import ArrayCatalog
        print("Painting Bk grids...")

        boxcenter = (np.max(dat_pos, axis=0) + np.min(dat_pos, axis=0)) / 2.
        dat_pos_c = dat_pos - boxcenter
        ran_pos_c = ran_pos - boxcenter

        def _paint(positions, weights):
            cat = ArrayCatalog({'Position': positions, 'WEIGHT': weights})
            return cat.to_mesh(Nmesh=gridsize, BoxSize=boxsize, interlaced=True,
                               compensated=True, position='Position', weight='WEIGHT',
                               resampler='tsc').compute()

        np.save(dat_grid_file, _paint(dat_pos_c, dat['WEIGHT'] * dat['WEIGHT_FKP']))
        np.save(ran_grid_file, _paint(ran_pos_c, ran['WEIGHT'] * ran['WEIGHT_FKP']))
        np.save(ran_shot2_file, _paint(ran_pos_c, ran['WEIGHT'] ** 2 * ran['WEIGHT_FKP'] ** 2))
        np.save(dat_shot2_file, _paint(dat_pos_c, dat['WEIGHT'] ** 2 * dat['WEIGHT_FKP'] ** 2))
        np.save(ran_shot3_file, _paint(ran_pos_c, ran['WEIGHT'] ** 3 * ran['WEIGHT_FKP'] ** 3))
        np.save(ran_ic_file, _paint(ran_pos_c, ran['WEIGHT']))

    print("Loading Bk gridded data from file")
    V = boxsize.prod()
    dat_grid = np.load(dat_grid_file) * np.sum(dat['WEIGHT'] * dat['WEIGHT_FKP']) / V
    ran_grid = np.load(ran_grid_file) * np.sum(ran['WEIGHT'] * ran['WEIGHT_FKP']) / V
    ran_ic_grid = np.load(ran_ic_file) * np.sum(ran['WEIGHT']) / V

    alpha = np.sum(dat['WEIGHT'] * dat['WEIGHT_FKP']) / np.sum(ran['WEIGHT'] * ran['WEIGHT_FKP'])
    alpha_ic = np.sum(dat['WEIGHT']) / np.sum(ran['WEIGHT'])
    alpha_2 = np.sum(dat['WEIGHT'] ** 2 * dat['WEIGHT_FKP'] ** 2) / np.sum(ran['WEIGHT'] ** 2 * ran['WEIGHT_FKP'] ** 2)
    alpha_3 = np.sum(dat['WEIGHT'] ** 3 * dat['WEIGHT_FKP'] ** 3) / np.sum(ran['WEIGHT'] ** 3 * ran['WEIGHT_FKP'] ** 3)

    density_grid = (dat_grid - alpha * ran_grid).astype(np.float64)
    mask = (alpha * ran_grid).astype(np.float64)
    mask_ic = (alpha_ic * ran_ic_grid).astype(np.float64)

    dat_grid2 = np.load(dat_shot2_file) * np.sum(dat['WEIGHT'] ** 2 * dat['WEIGHT_FKP'] ** 2) / V
    ran_grid2 = np.load(ran_shot2_file) * np.sum(ran['WEIGHT'] ** 2 * ran['WEIGHT_FKP'] ** 2) / V
    density_grid2 = (dat_grid2 - alpha_2 * ran_grid2).astype(np.float64)

    ran_grid3 = np.load(ran_shot3_file) * np.sum(ran['WEIGHT'] ** 3 * ran['WEIGHT_FKP'] ** 3) / V
    mask_shot_bk = ((alpha_3 + 1.5 * alpha * alpha_2 + 0.5 * alpha ** 3) * ran_grid3).astype(np.float64)
    mask_shot_2pt = ((alpha_2 + alpha ** 2) * ran_grid2.astype(np.float64))

    return {
        'density_grid': density_grid, 'mask': mask, 'mask_ic': mask_ic,
        'mask_shot_bk': mask_shot_bk, 'mask_shot_2pt': mask_shot_2pt,
        'density_grid2': density_grid2,
        'dat_grid': dat_grid, 'ran_grid': ran_grid,
        'alpha': alpha, 'alpha_ic': alpha_ic,
    }


def setup_bspec(cfg, base, mask, mask_ic, mask_shot_bk, r_edges):
    """Create BSpec estimator class."""
    applySinv = _make_applySinv(base)
    return pb.BSpec(
        base, cfg.BK_K_EDGES,
        applySinv=applySinv,
        mask=mask, lmax=cfg.BK_LMAX,
        k_bins_squeeze=cfg.BK_K_EDGES,
        include_partial_triangles=cfg.BK_INCLUDE_PARTIAL,
        add_GIC=cfg.ADD_GIC, add_RIC=cfg.ADD_RIC,
        mask_IC=mask_ic, radial_bins_RIC=r_edges,
        Pk_fid=None, mask_shot=mask_shot_bk,
        applySinv_transpose=applySinv,
    )


def full_setup_bk(cfg, sample, hemisphere, zid, include_weights=True, reload_grids=False):
    """Complete Bk setup. Returns dict with all Bk-specific products."""
    zmin, zmax, dz, P_fkp, fits_name = cfg.get_sample_params(sample, zid)
    out_str = f'{sample}_{hemisphere}_z{zid}'

    os.makedirs(cfg.BK_OUT_DIR, exist_ok=True)
    outfile_base = os.path.join(cfg.BK_OUT_DIR, out_str + 'bk')
    if not include_weights:
        outfile_base += '_noweight'
    if cfg.ADD_GIC:
        outfile_base += '_GIC'
    elif cfg.ADD_RIC:
        outfile_base += '_RIC'
    else:
        outfile_base += '_simple'

    print(f"\n### Bk: {sample} {hemisphere} z{zid} [{zmin:.1f}, {zmax:.1f}]")

    print("Loading cosmology...")
    cosmo, comoving_distance, z_from_comoving = load_cosmology(cfg)

    print("Loading data...")
    dat, ran = load_catalogs(cfg, sample, hemisphere, zid, include_weights)

    z_eff_pk, z_eff_bk = rescale_fkp_for_bk(dat, ran, comoving_distance, zmin, zmax, dz)
    print(f"z_eff (Pk) = {z_eff_pk:.3f}")
    print(f"z_eff (Bk) = {z_eff_bk:.3f}")

    k_arr, pk_fid = compute_fiducial_pk(cfg, cosmo, z_eff_bk)

    z_edges = np.arange(zmin, zmax + 1e-6, dz)
    r_edges = np.array([comoving_distance(z) for z in z_edges])

    print("Converting coordinates...")
    dat_pos = sky_to_cartesian(dat['RA'], dat['DEC'], dat['Z'], comoving_distance)
    ran_pos = sky_to_cartesian(ran['RA'], ran['DEC'], ran['Z'], comoving_distance)

    boxsize = 1000 + (np.max(dat_pos, axis=0) - np.min(dat_pos, axis=0))
    boxcenter = (np.max(dat_pos, axis=0) + np.min(dat_pos, axis=0)) / 2.
    gridsize = np.asarray(np.ceil(1.01 * cfg.BK_KMAX_GRID / (np.pi / boxsize)), dtype=int)
    gridsize = 2 * (gridsize // 2)

    grids = create_grids_bk(cfg, dat, ran, dat_pos, ran_pos, boxsize, gridsize,
                            out_str, include_weights, reload_grids)

    print("Loading PolyBin3D (Bk)...")
    base = setup_polybin(cfg, boxsize, gridsize, boxcenter, k_arr, pk_fid)
    bspec = setup_bspec(cfg, base, grids['mask'], grids['mask_ic'],
                        grids['mask_shot_bk'], r_edges)

    return {
        'dat': dat, 'ran': ran,
        **grids,
        'base': base, 'bspec': bspec,
        'boxsize': boxsize, 'gridsize': gridsize, 'boxcenter': boxcenter,
        'z_eff_pk': z_eff_pk, 'z_eff_bk': z_eff_bk, 'r_edges': r_edges,
        'k_arr': k_arr, 'pk_fid': pk_fid,
        'comoving_distance': comoving_distance,
        'z_from_comoving': z_from_comoving,
        'out_str': out_str, 'outfile_base': outfile_base,
    }
