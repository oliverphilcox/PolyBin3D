"""
Compute DESI Pk: numerator, ideal Fisher, shot-noise, full Fisher + theory matrix.

Usage: python -u run_pk.py [--config CONFIG] SAMPLE HEMI ZID INCLUDE_WEIGHTS
  SAMPLE: BGS, LRG, ELG, QSO
  HEMI: NGC, SGC
  ZID: 1, 2, or 3
  INCLUDE_WEIGHTS: 1 (full weights) or 0 (remove WEIGHT_SYS)
"""
import argparse
import sys, os, time
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import config
from load_data import full_setup

parser = argparse.ArgumentParser()
parser.add_argument('--config', default=None, help='Path to YAML config file')
parser.add_argument('sample', type=str)
parser.add_argument('hemisphere', type=str)
parser.add_argument('zid', type=int)
parser.add_argument('include_weights', type=int)
args = parser.parse_args()

cfg = config.load(args.config)
sample = args.sample
hemisphere = args.hemisphere
zid = args.zid
include_weights = args.include_weights

# Per-sample k_edges (may differ from global, e.g. finer dk for QSO)
k_edges = cfg.get_k_edges(sample)
k_edges_theory = cfg.get_k_edges_theory(sample)
print(f"k_edges: {len(k_edges)-1} bins in [{k_edges[0]:.4f}, {k_edges[-1]:.4f}]")

# Full setup
# Check if already completed (reconstruct outfile path without full_setup)
_suffix = '_GIC' if cfg.ADD_GIC else ('_RIC' if cfg.ADD_RIC else '_simple')
_outfile_quick = os.path.join(cfg.PK_OUT_DIR,
    f'{sample}_{hemisphere}_z{zid}pk' + ('' if include_weights else '_noweight') +
    _suffix + '_all.npz')
if os.path.exists(_outfile_quick):
    print(f"Already completed: {_outfile_quick}")
    sys.exit(0)

ctx = full_setup(cfg, sample, hemisphere, zid, include_weights)
pspec = ctx['pspec']
outfile = ctx['outfile_base']

t_init = time.time()

# ---- Numerator ----
if not os.path.exists(outfile + '_num.npy'):
    print("\nComputing numerator")
    t1 = time.time()
    Pk_num = pspec.Pk_numerator(ctx['density_grid']).copy()
    print("Time: %.2f" % (time.time() - t1))
    np.save(outfile + '_num.npy', Pk_num)
else:
    print("Loading numerator from file")
    Pk_num = np.load(outfile + '_num.npy')

# ---- Normalization ----
norm = np.sum(ctx['dat_grid'] * ctx['ran_grid']) * ctx['alpha'] / ctx['gridsize'].prod()

# ---- Ideal Fisher ----
if not os.path.exists(outfile + '_ideal_fisher.npy'):
    print("\nComputing ideal Fisher")
    t1 = time.time()
    ideal_fisher = norm * pspec.compute_fisher_ideal(discreteness_correction=True, Pk_fid=None)
    print("Time: %.2f" % (time.time() - t1))
    np.save(outfile + '_ideal_fisher.npy', ideal_fisher)
else:
    print("Loading ideal Fisher from file")
    ideal_fisher = np.load(outfile + '_ideal_fisher.npy')

# ---- Shot-noise ----
P_shot = (1. / norm *
          (np.sum((ctx['dat']['WEIGHT'] * ctx['dat']['WEIGHT_FKP']) ** 2) +
           ctx['alpha'] ** 2 * np.sum((ctx['ran']['WEIGHT'] * ctx['ran']['WEIGHT_FKP']) ** 2))
          / ctx['boxsize'].prod())
np.save(outfile + '_Pshot.npy', P_shot)

# Free large arrays no longer needed (catalogs + intermediate grids)
# Masks stay alive via pspec references; catalogs/grids are truly freed.
import gc
for key in ['dat', 'ran', 'dat_grid', 'ran_grid', 'density_grid']:
    ctx.pop(key, None)
gc.collect()

# FKP power spectrum
Pk_fkp = np.linalg.inv(ideal_fisher) @ Pk_num

# ---- MC shot-noise ----
t1 = time.time()
shots = []
for i in range(cfg.N_SHOT):
    fname = outfile + '_shot%d.npy' % i
    if not os.path.exists(fname):
        print("Computing shot noise %d of %d" % (i + 1, cfg.N_SHOT))
        shot = pspec.compute_shot_contribution(int(1e6) + i)
        np.save(fname, shot)
    else:
        shot = np.load(fname)
    shots.append(shot)
print("Shot-noise time: %.2f" % (time.time() - t1))
full_shot = np.mean(shots, axis=0)

# ---- Full Fisher + theory matrix (if include_weights) ----
if include_weights:
    t1 = time.time()
    fishs, theories = [], []
    for i in range(cfg.N_FISH):
        fish_file = outfile + '_fish%d.npy' % i
        theory_file = outfile + '_theory%d.npy' % i
        if not (os.path.exists(fish_file) and os.path.exists(theory_file)):
            print("\nComputing Fisher matrix %d of %d" % (i + 1, cfg.N_FISH))
            t2 = time.time()
            fish, theory = pspec.compute_theory_contribution(
                int(1e5) + i, k_edges_theory,
                lmax_theory=cfg.LMAX_THEORY,
                include_wideangle=cfg.INCLUDE_WIDEANGLE,
                verb=(i == 0),
            )
            np.save(fish_file, fish)
            np.save(theory_file, theory)
            print("Time (single): %.2f" % (time.time() - t2))
        else:
            fish = np.load(fish_file)
            theory = np.load(theory_file)
        fishs.append(fish)
        theories.append(theory)
    full_fisher = np.mean(fishs, axis=0)
    full_theory = np.mean(theories, axis=0)
    print("Fisher time (all): %.2f" % (time.time() - t1))

    # Assemble optimal Pk
    inv_fisher = np.linalg.inv(full_fisher)
    Pk_opt = inv_fisher @ Pk_num

    np.savez(outfile + '_all.npz',
             k_edges=k_edges, Pk_num=Pk_num,
             ideal_fisher=ideal_fisher, full_fisher=full_fisher,
             P_shot=P_shot, Pk_fkp=Pk_fkp, Pk_opt=Pk_opt,
             all_fisher=fishs, z_eff=ctx['z_eff'],
             full_shot=full_shot, all_shot=shots,
             k_edges_theory=k_edges_theory,
             full_theory=full_theory, all_theory=theories)
else:
    np.savez(outfile + '_all.npz',
             k_edges=k_edges, Pk_num=Pk_num,
             ideal_fisher=ideal_fisher, P_shot=P_shot, Pk_fkp=Pk_fkp,
             z_eff=ctx['z_eff'],
             full_shot=full_shot, all_shot=shots,
             k_edges_theory=k_edges_theory)

print("\nSaved to %s" % (outfile + '_all.npz'))
print("Total time: %.2f" % (time.time() - t_init))
