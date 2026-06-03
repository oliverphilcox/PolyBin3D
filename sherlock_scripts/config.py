"""
Configuration loader for DESI Pk + Bk analysis.

Loads parameters from a YAML file and exposes them as module-level attributes.
All run scripts call config.load(path) at startup.

Usage:
    import config
    cfg = config.load('configs/v7_highk.yaml')
    print(cfg.K_EDGES, cfg.LMAX)
"""
import os
import numpy as np
import yaml


class Config:
    """Container for analysis configuration, loaded from YAML."""

    def __init__(self, yaml_path):
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)

        self._raw = raw
        self._yaml_path = os.path.abspath(yaml_path)

        # Version
        self.VERSION = raw['version']

        # ---- Pk ----
        pk = raw['pk']
        self._pk_raw = pk
        self.LMAX = pk['lmax']

        # k_edges: parametric (k_min, k_fine_max, dk, k_broad) or explicit list
        if 'k_edges' in pk:
            self.K_EDGES = np.array(pk['k_edges'])
        else:
            self.K_EDGES = self._make_k_edges(pk['k_min'], pk['k_fine_max'],
                                               pk['dk'], pk.get('k_broad', []))

        self.KMAX_GRID = pk['kmax_grid']
        self.DK_THEORY = pk['dk_theory']
        self.LMAX_THEORY = pk['lmax_theory']
        self.INCLUDE_WIDEANGLE = pk['include_wideangle']
        self.N_FISH = pk['n_fish']
        self.N_SHOT = pk['n_shot']
        self.N_COV = pk['n_cov']
        self.K_EDGES_THEORY = np.arange(0.0035, max(self.K_EDGES) + 0.05, self.DK_THEORY)

        # ---- Bk ----
        bk = raw['bk']
        self.BK_LMAX = bk['lmax']
        self.BK_K_EDGES = np.array(bk['k_edges'])
        self.BK_KMAX_GRID = bk['kmax_grid']
        self.BK_INCLUDE_PARTIAL = bk['include_partial_triangles']
        self.BK_N_FISH = bk['n_fish']
        self.BK_N_SHOT = bk['n_shot']
        self.BK_N_LIN = bk['n_lin']
        self.BK_N_COV = bk['n_cov']

        # ---- Shared ----
        ic = raw['integral_constraints']
        self.ADD_GIC = ic['add_gic']
        self.ADD_RIC = ic['add_ric']

        self.FFT_BACKEND = raw['fft_backend']
        self.MAX_RAN = raw['max_ran']

        fc = raw['fiber_collisions']
        self.FC_ANGLE_DEG = fc['angle_deg']
        self.FC_RMAX = fc['rmax']
        self.FC_DR = fc['dr']

        # ---- Cosmology ----
        cosmo = raw['cosmology']
        self.H_FID = cosmo['h']
        self.OMEGA_B = cosmo['omega_b']
        self.OMEGA_CDM = cosmo['omega_cdm']
        self.N_UR = cosmo['N_ur']
        self.M_NU = cosmo['m_nu']

        # ---- Paths (with template substitution) ----
        paths = raw['paths']
        oak = paths['oak']
        version = self.VERSION
        self.OAK = oak
        self.DATA_DIR = paths['data_dir'].format(oak=oak, version=version)
        self.FID_PK_DIR = paths['fid_pk_dir'].format(oak=oak, version=version)
        self.PK_OUT_DIR = paths['pk_out_dir'].format(oak=oak, version=version)
        self.PK_PROC_DIR = paths['pk_proc_dir'].format(oak=oak, version=version)
        self.BK_OUT_DIR = paths['bk_out_dir'].format(oak=oak, version=version)
        self.BK_PROC_DIR = paths['bk_proc_dir'].format(oak=oak, version=version)

        # ---- Samples ----
        self.SAMPLES = {}
        for name, info in raw['samples'].items():
            zbins = {int(k): tuple(v) for k, v in info['zbins'].items()}
            sample_dict = {
                'fits_name': info['fits_name'],
                'P_fkp': float(info['P_fkp']),
                'dz': float(info['dz']),
                'zbins': zbins,
            }
            if 'pk_dk' in info:
                sample_dict['pk_dk'] = float(info['pk_dk'])
            self.SAMPLES[name] = sample_dict
        self.HEMISPHERES = raw['hemispheres']

    @staticmethod
    def _make_k_edges(k_min, k_fine_max, dk, k_broad):
        """Generate k_edges from parametric specification."""
        n_fine = int(round((k_fine_max - k_min) / dk)) + 1
        fine = np.linspace(k_min, k_fine_max, n_fine)
        return np.concatenate([fine, np.array(k_broad)])

    def get_k_edges(self, sample=None):
        """Return k_edges for a sample (uses per-sample dk override if set)."""
        if sample is not None:
            info = self.SAMPLES.get(sample, {})
            if 'pk_dk' in info and 'k_min' in self._pk_raw:
                pk = self._pk_raw
                return self._make_k_edges(pk['k_min'], pk['k_fine_max'],
                                          info['pk_dk'], pk.get('k_broad', []))
        return self.K_EDGES

    def get_k_edges_theory(self, sample=None):
        """Return theory k_edges for a sample."""
        k_edges = self.get_k_edges(sample)
        return np.arange(0.0035, max(k_edges) + 0.05, self.DK_THEORY)

    def get_all_runs(self):
        """Return list of (sample, hemisphere, zid) for all valid combinations."""
        runs = []
        for sample, info in self.SAMPLES.items():
            for hemi in self.HEMISPHERES:
                for zid in info['zbins']:
                    runs.append((sample, hemi, zid))
        return runs

    def get_sample_params(self, sample, zid):
        """Return (ZMIN, ZMAX, dz, P_fkp, fits_name) for a given sample/zid."""
        info = self.SAMPLES[sample]
        zmin, zmax = info['zbins'][zid]
        return zmin, zmax, info['dz'], info['P_fkp'], info['fits_name']

    def summary(self):
        """Print configuration summary."""
        print(f"=== {self.VERSION} ({os.path.basename(self._yaml_path)}) ===")
        print(f"\n--- Pk ---")
        print(f"lmax = {self.LMAX}")
        print(f"k_edges: {len(self.K_EDGES)-1} bins in [{self.K_EDGES[0]:.4f}, {self.K_EDGES[-1]:.4f}]")
        print(f"kmax_grid = {self.KMAX_GRID:.3f}")
        print(f"k_edges_theory: {len(self.K_EDGES_THEORY)-1} bins up to {self.K_EDGES_THEORY[-1]:.4f}")
        print(f"N_fish={self.N_FISH}, N_shot={self.N_SHOT}, N_cov={self.N_COV}")
        print(f"\n--- Bk ---")
        print(f"lmax = {self.BK_LMAX}, kmax = {self.BK_K_EDGES[-1]:.3f}")
        print(f"k_edges: {len(self.BK_K_EDGES)-1} bins in [{self.BK_K_EDGES[0]:.4f}, {self.BK_K_EDGES[-1]:.4f}]")
        print(f"kmax_grid = {self.BK_KMAX_GRID:.3f}")
        print(f"N_fish={self.BK_N_FISH}, N_shot={self.BK_N_SHOT}, N_lin={self.BK_N_LIN}, N_cov={self.BK_N_COV}")
        # Show per-sample k_edges overrides
        for name, info in self.SAMPLES.items():
            if 'pk_dk' in info:
                ke = self.get_k_edges(name)
                print(f"\n  {name} override: dk={info['pk_dk']}, {len(ke)-1} bins in [{ke[0]:.4f}, {ke[-1]:.4f}]")

        print(f"\nAll runs ({len(self.get_all_runs())}):")
        for s, h, z in self.get_all_runs():
            zmin, zmax, dz, pfkp, _ = self.get_sample_params(s, z)
            print(f"  {s} {h} z{z}: [{zmin:.1f}, {zmax:.1f}]")


# Default config path (relative to this file)
_DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'v7_highk.yaml')

def load(yaml_path=None):
    """Load configuration from YAML. Uses default if no path given."""
    if yaml_path is None:
        yaml_path = _DEFAULT_CONFIG
    return Config(yaml_path)


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    cfg = load(path)
    cfg.summary()
