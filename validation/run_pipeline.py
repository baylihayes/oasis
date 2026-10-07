"""Run one version of OASIS end to end on a regression box.

Usage:
    python run_pipeline.py <package_dir> <box.npz> <out_dir> [options]

<package_dir> is the folder that CONTAINS the `oasis` package to test, e.g. the
repository root for the current code, or an exported copy of the reference
(`git archive oasis-reference oasis | tar -x -C <dir>`). This lets two versions
of OASIS be run side by side without installing either.

Steps: split the box into mini-boxes, compute self-calibration data (compared
between runs), write cosmology-based calibration parameters, and build the halo
catalogue. With --tiles N, the catalogue is built with the tiled pipeline
(N tiles per side, ribbon = padding, inner cells = minisize) instead; this needs
a version of OASIS that has oasis.tiles.
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('pkg', help='folder containing the oasis package to run')
p.add_argument('data', help='box .npz from make_regression_box.py')
p.add_argument('out', help='output folder (created; must not already contain a run)')
p.add_argument('--threads', type=int, default=8)
p.add_argument('--minisize', type=float, default=25.0)
p.add_argument('--padding', type=float, default=5.0)
p.add_argument('--n-orb-min', type=int, default=20)
p.add_argument('--tiles', type=int, default=0,
               help='if > 0, build the catalogue with N tiles per side')
p.add_argument('--seed-tie-break', choices=('halo_id', 'load_order'), default=None,
               help="how ties in M200b (seed order) and Morb (percolation "
                    "order) are broken (current code only; omit to use the "
                    "package default)")
args = p.parse_args()
# Only pass the option when given, so the reference code (which does not have
# it) can still be run.
extra = {} if args.seed_tie_break is None else {'seed_tie_break': args.seed_tie_break}

sys.path.insert(0, os.path.abspath(args.pkg))
from oasis.calibration import calibrate, get_calibration_data  # noqa: E402
from oasis.catalogue import run_orbiting_mass_assignment  # noqa: E402
from oasis.minibox import process_simulation_data  # noqa: E402
import oasis.catalogue  # noqa: E402

print('using', oasis.catalogue.__file__)

BOX, MP, RHOM = 100.0, 1e10, 0.3 * 2.775e11
out = args.out.rstrip('/') + '/'
if os.path.exists(out + 'run_reg'):
    sys.exit(f"{out} already contains a run; use a new output folder")
os.makedirs(out, exist_ok=True)
d = np.load(args.data)
times = {}

t = time.perf_counter()
process_simulation_data(
    save_path=out, particle_type='seed', boxsize=BOX, minisize=args.minisize,
    positions=d['s_pos'].copy(), velocities=d['s_vel'].copy(), ids=d['hid'].copy(),
    mass=(d['s_m'].copy(), 'M200b'),
    data=([d['s_r'].copy(), d['s_rs'].copy()], ('R200b', 'Rs'), (np.float32, np.float32)),
    n_threads=args.threads)
process_simulation_data(
    save_path=out, particle_type='dm', boxsize=BOX, minisize=args.minisize,
    positions=d['pos'].copy(), velocities=d['vel'].copy(), ids=d['pid'].copy(),
    mass=(MP, 'mass'), n_threads=args.threads)
times['preprocess'] = time.perf_counter() - t

t = time.perf_counter()
get_calibration_data(
    n_seeds=150, seed_data=(d['s_pos'].copy(), d['s_vel'].copy(), d['s_m'].copy(), d['s_r'].copy()),
    r_max=3.0, boxsize=BOX, minisize=args.minisize, save_path=out, particle_type='dm',
    mass_density=RHOM, redshift=0.0, n_threads=args.threads, diagnostics=False,
    overwrite=True)
times['calibration_data'] = time.perf_counter() - t

calibrate(save_path=out, omega_m=0.3)

t = time.perf_counter()
if args.tiles > 0:
    from oasis.catalogue import run_tiled_orbiting_mass_assignment
    from oasis.tiles import TileSpec, build_tiles

    spec = TileSpec(boxsize=BOX, tiles_per_side=args.tiles,
                    buffer_width=args.padding, inner_cell_size=args.minisize)
    f32 = np.float32
    parts = dict(ID=d['pid'], pos=d['pos'], vel=d['vel'])
    seeds = dict(ID=d['hid'], pos=d['s_pos'], vel=d['s_vel'],
                 M200b=d['s_m'].astype(f32), R200b=d['s_r'].astype(f32),
                 Rs=d['s_rs'].astype(f32))

    def chunks(arrays, size):
        """Feed the box in pieces, like a reader of a large snapshot would."""
        def make():
            n = len(arrays['pos'])
            for a in range(0, n, size):
                yield {k: v[a:a + size] for k, v in arrays.items()}
        return make

    build_tiles(chunks(parts, 1_000_000), chunks(seeds, 10_000), out + 'tiles', spec,
                particle_mass=MP)
    times['build_tiles'] = time.perf_counter() - t
    t = time.perf_counter()
    run_tiled_orbiting_mass_assignment(
        tile_paths=[out + f'tiles/tile_{i}.hdf5' for i in range(spec.n_tiles)],
        load_path=out, run_name='reg', min_num_part=args.n_orb_min, boxsize=BOX,
        padding=args.padding, particle_type='dm', redshift=0.0, n_threads=args.threads,
        **extra)
else:
    run_orbiting_mass_assignment(
        load_path=out, run_name='reg', min_num_part=args.n_orb_min, boxsize=BOX,
        minisize=args.minisize, padding=args.padding, particle_type='dm',
        redshift=0.0, n_threads=args.threads, cleanup=False, **extra)
times['catalogue_total'] = time.perf_counter() - t

print(json.dumps(times, indent=1))
with open(out + 'times.json', 'w') as f:
    json.dump(dict(times, args=vars(args)), f, indent=1)
