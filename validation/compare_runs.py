"""Compare two OASIS runs made with run_pipeline.py and give a verdict.

Usage:
    python compare_runs.py <run_a> <run_b> [--mode exact|reference|tiled]

Haloes are matched by Halo_ID (row order may legitimately differ), then every
catalogue column and every halo's particle and substructure members are
compared. The verdict is PASS, PASS (with accepted differences) or FAIL; the
exit code is 0 for a pass and 1 for a fail, so the script can be used in shell
scripts.

With --count-only, no verdict is given: the script only reports how many
haloes differ (used to record the effect of a deliberate change, e.g. the
seed tie-break) and exits with 0.

Modes (which differences are accepted, each for a documented reason):
  exact      nothing: every value and every dtype must match (two runs of the
             same code, or code changes that should not change outputs).
  reference  run_a = the oasis-reference code, run_b = the current code:
             - Morb within 1e-6 relative: the reference summed N float32 copies
               of the particle mass; the current code uses N * m_p directly.
             - Halo_ID, PID, LIDX, RIDX, SLIDX, SRIDX stored as int64 (fixes 1-2:
               32-bit types would overflow for MDPL2-sized catalogues).
             - memb/Halo_ID stored as an integer type instead of an accidental
               float64.
             - LIDX/RIDX/SLIDX/SRIDX values not compared: the reference merged
               mini-box files in filesystem order, so member positions differ;
               the members of every halo are compared instead.
  tiled      run_a = a normal (mini-box) run, run_b = a tiled run of the same
             code and box:
             - members within a halo may be listed in a different order (the
               order follows how particles were loaded, which is not science);
             - member ID dtypes may differ (tiles keep the input ID type).
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('run_a')
p.add_argument('run_b')
p.add_argument('--mode', choices=('exact', 'reference', 'tiled'), default='exact')
p.add_argument('--count-only', action='store_true',
               help='only report how many haloes differ; always exits with 0')
args = p.parse_args()
A, B = args.run_a.rstrip('/') + '/', args.run_b.rstrip('/') + '/'

INT64_COLUMNS = {'Halo_ID', 'PID', 'LIDX', 'RIDX', 'SLIDX', 'SRIDX'}
MORB_RTOL = 1e-6
results = []          # (status, check, detail); status in OK / ACCEPTED / FAIL


def record(status, check, detail=''):
    results.append((status, check, detail))
    print(f"[{status:8s}] {check}" + (f"  ({detail})" if detail else ''))


def load(run):
    with h5py.File(run + 'run_reg/catalogue.hdf5', 'r') as f:
        cat = {k: f[k][()] for k in f.keys()}
    with h5py.File(run + 'run_reg/members.hdf5', 'r') as f:
        memb = {k: f[k][()] for k in f.keys()}
    return cat, memb


if args.count_only:
    ca, ma = load(A)
    cb, mb = load(B)
    ida, idb = set(ca['Halo_ID'].tolist()), set(cb['Halo_ID'].tolist())
    common = np.array(sorted(ida & idb), dtype=np.int64)
    ra = {h: i for i, h in enumerate(ca['Halo_ID'].tolist())}
    rb = {h: i for i, h in enumerate(cb['Halo_ID'].tolist())}
    xa = np.array([ra[h] for h in common.tolist()], dtype=np.int64)
    xb = np.array([rb[h] for h in common.tolist()], dtype=np.int64)
    changed = np.zeros(len(common), dtype=bool)
    for key in ('Norb', 'NSUBS', 'PID'):
        diff = ca[key][xa].astype(np.int64) != cb[key][xb].astype(np.int64)
        changed |= diff
        print(f"haloes in both runs with a different {key}: {int(diff.sum())}")
    n_changed = len(ida - idb) + len(idb - ida) + int(changed.sum())
    print(f"haloes only in run_a: {len(ida - idb)}, only in run_b: {len(idb - ida)}")
    print(f"haloes affected in total: {n_changed} of {len(ida)} "
          f"({100.0 * n_changed / max(len(ida), 1):.2f}%)")
    sys.exit(0)

# --- calibration data --------------------------------------------------------
with h5py.File(A + 'calibration_data.hdf5', 'r') as fa, \
        h5py.File(B + 'calibration_data.hdf5', 'r') as fb:
    for key in ('r', 'vr', 'lnv2'):
        same = np.array_equal(fa[key][()], fb[key][()])
        record('OK' if same else 'FAIL', f'calibration_data/{key} identical',
               f'n={fa[key].shape[0]}')

# --- mini-box files ----------------------------------------------------------
files_a = sorted(glob.glob(A + 'mini_boxes_nside_*/*.hdf5'))
same, missing = True, 0
for path_a in files_a:
    path_b = B + os.path.relpath(path_a, A)
    if not os.path.exists(path_b):
        missing += 1
        continue
    with h5py.File(path_a, 'r') as fa, h5py.File(path_b, 'r') as fb:
        for group in fa.keys():
            for name in fa[group].keys():
                x, y = fa[group][name], fb[group][name]
                if x.dtype != y.dtype or not np.array_equal(x[()], y[()]):
                    same = False
record('OK' if same and not missing and files_a else 'FAIL',
       'mini-box files identical (all datasets, dtypes, order)',
       f'{len(files_a)} files' + (f', {missing} missing in run_b' if missing else ''))

# --- catalogue -----------------------------------------------------------------
ca, ma = load(A)
cb, mb = load(B)
na, nb = len(ca['Halo_ID']), len(cb['Halo_ID'])
record('OK' if na == nb else 'FAIL', 'number of haloes', f'{na} vs {nb}')
for name, c in (('run_a', ca), ('run_b', cb)):
    unique = len(np.unique(c['Halo_ID'])) == len(c['Halo_ID'])
    record('OK' if unique else 'FAIL', f'Halo_ID unique in {name}')
ia = np.argsort(ca['Halo_ID'], kind='stable')
ib = np.argsort(cb['Halo_ID'], kind='stable')
same_ids = na == nb and np.array_equal(ca['Halo_ID'][ia], cb['Halo_ID'][ib])
record('OK' if same_ids else 'FAIL', 'same set of Halo_IDs')
if not same_ids:
    print('\nFAIL: the catalogues contain different haloes; stopping here.')
    sys.exit(1)

if set(ca) != set(cb):
    record('FAIL', 'same catalogue columns', f'{sorted(ca)} vs {sorted(cb)}')
for key in sorted(set(ca) & set(cb)):
    x, y = ca[key][ia], cb[key][ib]
    if key in ('LIDX', 'RIDX', 'SLIDX', 'SRIDX') and args.mode == 'reference':
        # Positions in members.hdf5. The reference merged mini-box files in
        # filesystem order, so the positions differ; the members they point to
        # are compared below, which is what matters.
        record('ACCEPTED', f'{key} not compared by value',
               'member positions depend on the reference merge order; '
               'members compared per halo below')
        continue
    if key == 'Morb':
        if np.array_equal(x, y):
            record('OK', 'Morb identical')
        else:
            rel = float(np.max(np.abs(x.astype(np.float64) - y) / np.abs(x)))
            if args.mode == 'reference' and rel <= MORB_RTOL:
                record('ACCEPTED', 'Morb', f'max relative difference {rel:.2e} '
                       '(reference summed float32 masses)')
            else:
                record('FAIL', 'Morb', f'max relative difference {rel:.2e}')
        continue
    same_values = np.array_equal(x, y)
    same_dtype = x.dtype == y.dtype
    if same_values and same_dtype:
        record('OK', f'{key} identical', f'{x.dtype}')
    elif same_values and args.mode == 'reference' and key in INT64_COLUMNS \
            and y.dtype == np.int64:
        record('ACCEPTED', f'{key} values identical', f'dtype {x.dtype} -> {y.dtype} (fixes 1-2)')
    else:
        record('FAIL', f'{key}', f'values identical={same_values}, '
               f'dtype {x.dtype} vs {y.dtype}')

# --- members -------------------------------------------------------------------
pid_order, pid_set = True, True
for i, j in zip(ia, ib):
    pa = ma['PID'][ca['LIDX'][i]:ca['RIDX'][i]]
    pb = mb['PID'][cb['LIDX'][j]:cb['RIDX'][j]]
    pid_order &= np.array_equal(pa, pb)
    pid_set &= np.array_equal(np.sort(pa), np.sort(pb))
if pid_order:
    record('OK', 'particle members of every halo identical (same order)')
elif pid_set and args.mode == 'tiled':
    record('ACCEPTED', 'particle members of every halo identical as sets',
           'order within a halo follows the load order (tiles vs mini-boxes)')
else:
    record('FAIL', 'particle members of every halo',
           f'same as sets={pid_set}, same order={pid_order}')
if ma['PID'].dtype != mb['PID'].dtype:
    record('ACCEPTED' if args.mode == 'tiled' else 'FAIL', 'members PID dtype',
           f"{ma['PID'].dtype} vs {mb['PID'].dtype}")
record('OK' if len(ma['PID']) == len(mb['PID']) else 'FAIL', 'total particle members',
       f"{len(ma['PID'])} vs {len(mb['PID'])}")

sub_same, sub_set, n_reordered = True, True, 0
for i, j in zip(ia, ib):
    sa = ma['Halo_ID'][ca['SLIDX'][i]:ca['SRIDX'][i]] if ca['SLIDX'][i] >= 0 else np.array([])
    sb = mb['Halo_ID'][cb['SLIDX'][j]:cb['SRIDX'][j]] if cb['SLIDX'][j] >= 0 else np.array([])
    sa, sb = sa.astype(np.int64), sb.astype(np.int64)
    if not np.array_equal(sa, sb):
        sub_same = False
        n_reordered += 1
        sub_set &= np.array_equal(np.sort(sa), np.sort(sb))
if sub_same:
    record('OK', 'substructure members of every halo identical (same order)')
else:
    record('FAIL', 'substructure members of every halo',
           f'same as sets={sub_set}, {n_reordered} host(s) differ')
if 'Halo_ID' in ma and 'Halo_ID' in mb and ma['Halo_ID'].dtype != mb['Halo_ID'].dtype:
    integer = np.issubdtype(mb['Halo_ID'].dtype, np.integer)
    accepted = (args.mode == 'reference' and integer) or args.mode == 'tiled'
    record('ACCEPTED' if accepted else 'FAIL', 'members Halo_ID dtype',
           f"{ma['Halo_ID'].dtype} -> {mb['Halo_ID'].dtype}")

# --- verdict -------------------------------------------------------------------
n_sub = int((ca['NSUBS'] > 0).sum())
print(f"\nhaloes: {na}, with substructure: {n_sub}, subhaloes (PID != -1): "
      f"{int((ca['PID'] != -1).sum())}, particle members: {len(ma['PID'])}")
statuses = {s for s, _, _ in results}
if 'FAIL' in statuses:
    print(f"FAIL ({sum(s == 'FAIL' for s, _, _ in results)} check(s) failed, mode={args.mode})")
    sys.exit(1)
if 'ACCEPTED' in statuses:
    print(f"PASS (with accepted differences, mode={args.mode})")
else:
    print(f"PASS (everything identical, mode={args.mode})")
