"""Padding (ribbon width) convergence test.

Runs OASIS on the same box with increasing padding and reports which haloes
still change. The padding is the distance around each region (mini-box or
tile core cell) from which particles and seeds are loaded; for tiles, the
ribbon only has to be at least the padding (a wider ribbon is cut away), so
this test determines the ribbon width as well. Because tiled and untiled runs
give identical catalogues for the same padding, the scan uses the existing
mini-box files.

Usage:
    python padding_convergence.py <data_dir> <work_dir> --boxsize 1000 \\
        --minisize 100 --paddings 5 7.5 10 15 20 [--n-orb-min 100] [--threads 50]

<data_dir> must contain calibration_pars.hdf5 and mini_boxes_nside_<n>/. It is
only read: <work_dir> gets symbolic links to both, and all runs are written
to <work_dir>/run_pad<p>/. Runs that already have a catalogue are reused.

Every padding is compared with the largest one (the reference) and with the
next larger one. For each comparison the script reports haloes found in only
one run, and haloes whose Norb, parent (PID), particle members or substructure
members differ, together with the size (2 R200b) and position (distance of the
center to the nearest mini-box face) of the changed haloes. A padding is
converged when nothing changes with respect to all larger paddings.
"""
import argparse
import json
import os
import sys
import time

import h5py
import numpy as np

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('data_dir')
p.add_argument('work_dir')
p.add_argument('--boxsize', type=float, required=True)
p.add_argument('--minisize', type=float, required=True)
p.add_argument('--paddings', type=float, nargs='+', required=True)
p.add_argument('--n-orb-min', type=int, default=100)
p.add_argument('--threads', type=int, default=None)
p.add_argument('--redshift', type=float, default=0.0)
p.add_argument('--particle-type', default='dm')
p.add_argument('--pkg', default=os.path.join(os.path.dirname(__file__), '..'),
               help='folder containing the oasis package (default: this repository)')
args = p.parse_args()

sys.path.insert(0, os.path.abspath(args.pkg))
from oasis.catalogue import run_orbiting_mass_assignment  # noqa: E402

data = os.path.abspath(args.data_dir)
work = os.path.abspath(args.work_dir) + '/'
nside = int(np.ceil(args.boxsize / args.minisize))
paddings = sorted(set(args.paddings))
os.makedirs(work, exist_ok=True)
for name in ('calibration_pars.hdf5', f'mini_boxes_nside_{nside}'):
    src, dst = os.path.join(data, name), work + name
    if not os.path.exists(src):
        sys.exit(f"{src} not found")
    if not os.path.lexists(dst):
        os.symlink(src, dst)


def tag(pad):
    return f'pad{pad:g}'.replace('.', 'p')


# --- runs ----------------------------------------------------------------------
timings = {}
for pad in paddings:
    run = work + f'run_{tag(pad)}/'
    if os.path.exists(run + 'catalogue.hdf5'):
        print(f'padding {pad:g}: reusing {run}')
        continue
    print(f'padding {pad:g}: running')
    t = time.perf_counter()
    run_orbiting_mass_assignment(
        load_path=work, run_name=tag(pad), min_num_part=args.n_orb_min,
        boxsize=args.boxsize, minisize=args.minisize, padding=pad,
        particle_type=args.particle_type, redshift=args.redshift,
        n_threads=args.threads, cleanup=True)
    timings[pad] = time.perf_counter() - t


# --- comparison ------------------------------------------------------------------
def load(pad):
    run = work + f'run_{tag(pad)}/'
    with h5py.File(run + 'catalogue.hdf5', 'r') as f:
        c = {k: f[k][()] for k in ('Halo_ID', 'Norb', 'PID', 'R200b', 'pos',
                                   'LIDX', 'RIDX', 'SLIDX', 'SRIDX')}
    with h5py.File(run + 'members.hdf5', 'r') as f:
        m = {k: f[k][()] for k in f}
    with open(run + 'timings.json') as f:
        n_loaded = json.load(f).get('n_particles_loaded')
    return c, m, n_loaded


def face_distance(pos):
    """Distance of each center to the nearest mini-box face."""
    u = np.mod(pos, args.minisize)
    return np.min(np.minimum(u, args.minisize - u), axis=1)


def compare(pa, pb):
    """Haloes that differ between padding pa and padding pb."""
    ca, ma, _ = cats[pa]
    cb, mb, _ = cats[pb]
    ra = {h: i for i, h in enumerate(ca['Halo_ID'].tolist())}
    rb = {h: i for i, h in enumerate(cb['Halo_ID'].tolist())}
    only_a, only_b = set(ra) - set(rb), set(rb) - set(ra)
    common = sorted(set(ra) & set(rb))
    changed = {'Norb': [], 'PID': [], 'members': [], 'subhaloes': []}
    for h in common:
        i, j = ra[h], rb[h]
        if ca['Norb'][i] != cb['Norb'][j]:
            changed['Norb'].append(h)
        if ca['PID'][i] != cb['PID'][j]:
            changed['PID'].append(h)
        pa_ = ma['PID'][ca['LIDX'][i]:ca['RIDX'][i]]
        pb_ = mb['PID'][cb['LIDX'][j]:cb['RIDX'][j]]
        if not np.array_equal(np.sort(pa_), np.sort(pb_)):
            changed['members'].append(h)
        sa = ma['Halo_ID'][ca['SLIDX'][i]:ca['SRIDX'][i]] if ca['SLIDX'][i] >= 0 else []
        sb = mb['Halo_ID'][cb['SLIDX'][j]:cb['SRIDX'][j]] if cb['SLIDX'][j] >= 0 else []
        if not np.array_equal(np.sort(sa), np.sort(sb)):
            changed['subhaloes'].append(h)
    any_changed = set().union(*changed.values()) | only_a | only_b
    # Size and position of every affected halo (taken from whichever run has it)
    rows = []
    for h in any_changed:
        c, i = (ca, ra[h]) if h in ra else (cb, rb[h])
        rows.append((2 * float(c['R200b'][i]),
                     float(face_distance(c['pos'][i:i + 1])[0])))
    rows = np.array(rows) if rows else np.zeros((0, 2))
    return dict(
        only_in_a=len(only_a), only_in_b=len(only_b),
        **{f'different_{k}': len(v) for k, v in changed.items()},
        affected=len(any_changed),
        affected_max_2R200b=float(rows[:, 0].max()) if len(rows) else None,
        affected_max_face_distance=float(rows[:, 1].max()) if len(rows) else None,
    )


cats = {pad: load(pad) for pad in paddings}
ref = paddings[-1]
n_ref = len(cats[ref][0]['Halo_ID'])
print(f"\nreference: padding {ref:g} ({n_ref} haloes)\n")
header = (f"{'padding':>8} {'haloes':>7} {'loaded':>12} | vs {'':<9} {'only':>5} {'only':>5}"
          f" {'Norb':>5} {'PID':>4} {'memb':>5} {'subs':>5} {'total':>6} {'max 2R200b':>10}"
          f" {'max face d':>10}")
print(header)
print(f"{'':>8} {'':>7} {'':>12} | {'':<12} {'here':>5} {'there':>5}")
summary = []
for k, pad in enumerate(paddings):
    for other, label in ((ref, f'ref {ref:g}'),
                         (paddings[k + 1] if k + 1 < len(paddings) else None, 'next')):
        if other is None or other == pad:
            continue
        d = compare(pad, other)
        summary.append(dict(padding=pad, compared_with=other, **d))
        mx = '-' if d['affected_max_2R200b'] is None else f"{d['affected_max_2R200b']:.2f}"
        mf = '-' if d['affected_max_face_distance'] is None else f"{d['affected_max_face_distance']:.2f}"
        print(f"{pad:>8g} {len(cats[pad][0]['Halo_ID']):>7} {cats[pad][2]:>12} | "
              f"{label + ' (' + format(other, 'g') + ')' if label == 'next' else label:<12} "
              f"{d['only_in_a']:>5} {d['only_in_b']:>5} {d['different_Norb']:>5} "
              f"{d['different_PID']:>4} {d['different_members']:>5} "
              f"{d['different_subhaloes']:>5} {d['affected']:>6} {mx:>10} {mf:>10}")

converged = [pad for pad in paddings
             if all(s['affected'] == 0 for s in summary
                    if s['padding'] >= pad and s['compared_with'] > s['padding'])]
print()
if converged and converged[0] < ref:
    print(f"converged from padding {converged[0]:g} (identical to every larger padding tested)")
else:
    print("not converged below the largest padding tested; extend --paddings")

with open(work + 'padding_convergence.json', 'w') as f:
    json.dump(dict(args=vars(args), reference=ref, comparisons=summary,
                   converged_from=converged[0] if converged else None,
                   run_seconds={f'{k:g}': v for k, v in timings.items()}), f, indent=1)
print(f"written {work}padding_convergence.json")
