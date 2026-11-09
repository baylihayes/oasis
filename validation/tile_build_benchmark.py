"""Benchmark and check tile building on a real snapshot.

Builds core + ribbon tiles from a Gadget-style HDF5 snapshot (e.g. Quijote:
snap_<nnn>.<i>.hdf5 with PartType1) and a Rockstar hlist HDF5 file, reading
the snapshot in bounded chunks, and records the following:
wall time, peak memory, rows and bytes per tile, and the duplication factor.
It also checks that every particle and seed in the built tiles' cores is
there exactly once.

Usage:
    python tile_build_benchmark.py <snapdir> <hlist.hdf5> <out_dir> \\
        --tiles-per-side 5 --ribbon 7.5 --cell 25 [--tile-ids 0 1 2 | --group-size 25]

--tile-ids builds only those tiles (one call). --group-size builds every
tile, N tiles per call, as one would for MDPL2 to bound the disk used by
bucket files. Without either, all tiles are built in one call. <out_dir>
must not contain tiles of this run already.
"""
import argparse
import glob
import json
import os
import resource
import time

try:
    import hdf5plugin  # noqa: F401  (Quijote snapshots may need the filters)
except ImportError:
    pass
import h5py
import numpy as np

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from oasis.tiles import TileSpec, build_tiles  # noqa: E402

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('snapdir', help='folder with snap_<nnn>.<i>.hdf5 files')
p.add_argument('hlist', help='Rockstar hlist HDF5 file (ID, M200b, Rs, X..VZ)')
p.add_argument('out_dir')
p.add_argument('--tiles-per-side', type=int, default=5)
p.add_argument('--ribbon', type=float, default=7.5)
p.add_argument('--cell', type=float, default=25.0, help='inner cell size')
p.add_argument('--tile-ids', type=int, nargs='+', default=None)
p.add_argument('--group-size', type=int, default=None)
p.add_argument('--chunk-rows', type=int, default=20_000_000,
               help='particles read per chunk (bounds memory)')
p.add_argument('--length-unit', type=float, default=1e-3,
               help='factor from snapshot positions and hlist Rs to Mpc/h (kpc/h: '
                    '1e-3); hlist positions X, Y, Z are taken as Mpc/h')
args = p.parse_args()

files = sorted(glob.glob(os.path.join(args.snapdir, 'snap_*.hdf5')),
               key=lambda s: int(s.rsplit('.', 2)[-2]))
with h5py.File(files[0], 'r') as f:
    head = dict(f['Header'].attrs)
boxsize = float(head['BoxSize']) * args.length_unit
particle_mass = float(head['MassTable'][1]) * 1e10
n_total = int(head['NumPart_Total'][1]) + (int(head['NumPart_Total_HighWord'][1]) << 32 \
    if 'NumPart_Total_HighWord' in head else 0)
rhom = float(head['Omega0']) * 2.77536627e11          # h^2 Msun / Mpc^3
spec = TileSpec(boxsize=boxsize, tiles_per_side=args.tiles_per_side,
                buffer_width=args.ribbon, inner_cell_size=args.cell)


def particle_chunks():
    """Each snapshot file in slices of --chunk-rows particles."""
    for path in files:
        with h5py.File(path, 'r') as f:
            n = f['PartType1/ParticleIDs'].shape[0]
            for a in range(0, n, args.chunk_rows):
                b = min(a + args.chunk_rows, n)
                yield dict(ID=f['PartType1/ParticleIDs'][a:b],
                           pos=f['PartType1/Coordinates'][a:b] * args.length_unit,
                           vel=f['PartType1/Velocities'][a:b])


with h5py.File(args.hlist, 'r') as f:
    m200b = f['M200b'][()]
    rs = f['Rs'][()] * args.length_unit
    r200b = np.power(3. * m200b / 4. / np.pi / 200. / rhom, 1. / 3.)
    keep = (r200b > 0) & (rs > 0)
    seeds = dict(ID=f['ID'][()][keep],
                 pos=np.vstack([f['X'][()], f['Y'][()], f['Z'][()]]).T[keep],
                 vel=np.vstack([f['VX'][()], f['VY'][()], f['VZ'][()]]).T[keep],
                 M200b=m200b[keep].astype(np.float32),
                 R200b=r200b[keep].astype(np.float32),
                 Rs=rs[keep].astype(np.float32))


def seed_chunks():
    yield seeds


if args.tile_ids is not None:
    groups = [sorted(set(args.tile_ids))]
elif args.group_size is not None:
    ids = list(range(spec.n_tiles))
    groups = [ids[a:a + args.group_size] for a in range(0, len(ids), args.group_size)]
else:
    groups = [list(range(spec.n_tiles))]

print(f"{len(files)} files, {n_total} particles, {len(seeds['ID'])} seeds, "
      f"box {boxsize:g}, m_p {particle_mass:.4g}")
print(f"{spec.n_tiles} tiles (core {spec.core_size:g}, ribbon {args.ribbon:g}, "
      f"cell {args.cell:g}, {spec.cells_per_side}^3 cells per tile); "
      f"building {sum(map(len, groups))} tiles in {len(groups)} call(s)")


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


# One plain read of the input, to separate I/O from the builder's own work.
t = time.perf_counter()
n_read = sum(len(c['ID']) for c in particle_chunks())
read_seconds = time.perf_counter() - t
print(f"one read of the snapshot: {read_seconds:.1f} s ({n_read} particles)")

calls = []
for group in groups:
    t = time.perf_counter()
    build_tiles(particle_chunks, seed_chunks, args.out_dir, spec,
                particle_mass=particle_mass, tile_ids=group)
    dt = time.perf_counter() - t
    calls.append(dict(tile_ids=group, seconds=dt, peak_rss_gb_so_far=peak_rss_gb()))
    print(f"  tiles {group[0]}..{group[-1]} ({len(group)}): {dt:.1f} s, "
          f"peak RSS so far {peak_rss_gb():.1f} GB")

# --- checks and per-tile numbers --------------------------------------------------
built = [t for g in groups for t in g]
tiles, core_ids, core_seeds = [], [], []
for t in built:
    path = os.path.join(args.out_dir, f'tile_{t}.hdf5')
    with h5py.File(path, 'r') as f:
        pc, sc = f['particles/in_core'][()], f['seeds/in_core'][()]
        core_ids.append(f['particles/ID'][()][pc])
        core_seeds.append(f['seeds/ID'][()][sc])
        tiles.append(dict(tile=t, rows=int(len(pc)), core_rows=int(pc.sum()),
                          seed_rows=int(len(sc)), core_seeds=int(sc.sum()),
                          bytes=os.path.getsize(path)))
core_ids = np.concatenate(core_ids)
core_seeds = np.concatenate(core_seeds)
rows = sum(x['rows'] for x in tiles)
core_rows = sum(x['core_rows'] for x in tiles)
dup = rows / max(core_rows, 1)
expected_dup = ((spec.core_size + 2 * args.ribbon) / spec.core_size) ** 3
unique_core = len(np.unique(core_ids)) == len(core_ids)
unique_seeds = len(np.unique(core_seeds)) == len(core_seeds)
complete = (len(built) == spec.n_tiles)
checks = dict(core_particle_ids_unique=bool(unique_core),
              core_seed_ids_unique=bool(unique_seeds))
if complete:
    checks['all_particles_in_a_core'] = bool(core_rows == n_total)
    checks['all_seeds_in_a_core'] = bool(len(core_seeds) == len(seeds['ID']))

total_bytes = sum(x['bytes'] for x in tiles)
print(f"\nrows {rows} ({core_rows} in cores), duplication {dup:.4f} "
      f"(uniform-density expectation {expected_dup:.4f})")
print(f"disk: {total_bytes / 1e9:.2f} GB for {len(tiles)} tiles "
      f"({total_bytes / rows:.1f} bytes per row incl. metadata)")
print(f"total build time {sum(c['seconds'] for c in calls):.1f} s, "
      f"peak RSS {peak_rss_gb():.1f} GB")
for k, v in checks.items():
    print(f"[{'OK' if v else 'FAIL'}] {k}")

summary = dict(args=vars(args), n_particles=n_total, n_seeds=len(seeds['ID']),
               boxsize=boxsize, particle_mass=particle_mass,
               read_seconds=read_seconds, calls=calls, tiles=tiles,
               rows=rows, core_rows=core_rows, duplication=dup,
               expected_duplication=expected_dup, bytes=total_bytes,
               bytes_per_row=total_bytes / rows, peak_rss_gb=peak_rss_gb(),
               checks=checks)
with open(os.path.join(args.out_dir, 'benchmark.json'), 'w') as f:
    json.dump(summary, f, indent=1)
print(f"written {os.path.join(args.out_dir, 'benchmark.json')}")
raise SystemExit(0 if all(checks.values()) else 1)
