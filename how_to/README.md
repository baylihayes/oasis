# OASIS How-To

This is documentation on how to run <span style="font-variant:small-caps;">Oasis</span>, from reading in halo seed catalogues and particle snapshots to finishing the final dynamical halo catalogue. It includes examples on how to run a small simulation (e.g. Quijote) that can fit into memory and a large simulation (e.g. MDPL2) that must be read and processed in chunks.

## Requirements

This section details the data requirements for <span style="font-variant:small-caps;">Oasis</span>. 

### Halo Catalog

The most common halo "seed" catalogue will be in the form of a ROCKSTAR halo catalogue. Any file format works, since you write the reader; HDF5 is used in the examples. 

The required data from the seed catalogue are the following:

- ID (hid)
- $\vec{x}$
- $\vec{v}$
- $M_{\rm 200b}$
- $R_{\rm 200b}$ (or computed from $M_{\rm 200b}$ with comoving mean density, as in the example below)
- $R_{s}$

$\vec{x}$, $R_{\rm 200b}$, and $R_{s}$ are assumed to be in comoving units, while $\vec{v}$ is assumed to be in physical units (usually in $km/s$). All positions must be in the same units. $R_{s}$ is usually presented in $h^{-1}{\rm kpc}$, so be mindful if the other positions are in different units.

### Snapshot Files

The snapshot files are where the particle data resides. These are usually presented as compressed HDF5 files or binary files in GADGET format. `build_tiles` reads one file at a time to cut back on memory usage. The examples will only include the HDF5 version, but feel free to use your preferred binary reader to extract the particle data. 

The required data from the particle snapshot files are the following:

- ID (pid)
- $\vec{x}$
- $\vec{v}$

Like the halo seed catalogue, $\vec{x}$ is expected to be in comoving units of the same kind as the halo seed catalogue. $\vec{v}$, however, is expected to be in GADGET convention, $\vec{v}_{\rm sim}$ = $\vec{v}_{\rm phys} / \sqrt{a}$. <span style="font-variant:small-caps;">Oasis</span> handles this internally, so no need to convert to physical units prior. 

## Reading in Data

The `build_tiles` function expects functions that read in the seed and particle data. These functions must be top-level functions so that they can be sent to worker processes. They should read one file at a time. Because the input data requirements are different for seed and particles, it is encouraged that there are two different functions: one to read in a halo catalogue file, and one to read in a particle snapshot file. This step should be the same for both smaller and larger simulations. An example of data-reading functions and how they are used is as follows: 

```python
import numpy as np
import h5py
import hdf5plugin
import glob
import os

from oasis.tiles import ChunkTasks, TileSpec, build_tiles
from oasis.calibration import calibrate
from oasis.catalogue import run_tiled_orbiting_mass_assignment, process_tile, merge_tiled_run
from oasis.common import G_GRAVITY

#==============================================================================
#
#                               Simulation parameters
#
#   These are EXAMPLE values. Change them to your simulation parameters BEFORE
#   running OASIS or you will get wrong results, especially for simulations with
#   redshifts greater than 0. Units only have to be consistent: here distances
#   in Mpc/h, masses in Msun/h, velocities in km/s.
# ==============================================================================
boxsize: float = 1000.                              # h^-1 Mpc
particle_mass: float = 77546570000.0                # h^-1 M_sun
redshift: float = 0.

# Cosmological parameters.
omega_m0: float = 0.3                               # Omega_m today (z = 0)
h: float = 0.67
omega_m: float = (omega_m0 * (1 + redshift)**3      # Omega_m at `redshift`
                  / (omega_m0 * (1 + redshift)**3 + 1 - omega_m0)) # (flat LambdaCDM)

# Mean matter density. OASIS works in comoving Mpc/h, so this is the z = 0 value
# at any redshift: Omega_m0 * rho_crit,0. With H0 = 100 h km/s/Mpc, rho_crit,0
# is in M_sun / Mpc^3; dividing by h^2 gives h^2 M_sun / Mpc^3.
H0_sq: float = (100. * h)**2
critical_density: float = 3. * H0_sq / 8. / np.pi / G_GRAVITY  # M_sun / Mpc^3
mass_density: float = omega_m0 * critical_density / h**2        # h^2 M_sun / Mpc^3

# ==============================================================================
#
#                               OASIS configuration
#
# ==============================================================================
# OASIS will save all results (tiles, calibration, catalogues) to this path.
save_path: str = 'path/save_path/'

n_threads: int = 32

# Calibration (self-calibration only).
n_seeds: int = 500
r_max: float = 5.0                 # h^-1 Mpc
calib_n_points: int = 20
calib_vrp_percent: float = 0.995
calib_vrn_width: float = 0.050
calib_gradient_radial_limits: tuple[float] = (0.2, 0.5)
diagnostic_plots = True

# Catalogue.
n_orb_min: int = 200
fast_mass: bool = False
ball_radius_solver: str = 'brentq'  # 'fsolve' reproduces the original OASIS exactly
run_name = '<cool name>'            # OASIS appends this name to identify the run.
                                    # This is useful to generate catalogues for the
                                    # same simulation box with different parameter
                                    # combinations.

particle_files: list[str] = sorted(glob.glob('/path/snapdir_000/snap_000.*'))
seed_files: list[str] = ['/path/rockstar/halos_0.hdf5']

def read_particle_file(path):
    with h5py.File(path, "r") as hdf:
        pid = hdf["PartType1/ParticleIDs"][()]
        pos = hdf["PartType1/Coordinates"][()] / 1000. # Could be in different units (e.g. Quijote), may need to convert from kpc/h to Mpc/h as shown here
        vel = hdf["PartType1/Velocities"][()]   # Should be in the GADGET convention
    return dict(ID=pid, pos=pos, vel=np.asarray(vel, np.float32))

def read_seed_file(path):
    with h5py.File(path, "r") as hdf:
        hid = hdf['ID'][()]
        m200b = hdf['M200b'][()]
        r200b = (3 * m200b / (4 * np.pi * 200 * mass_density))**(1/3)    # comoving Mpc/h
        rs = hdf['Rs'][()] / 1000.  # if it is a ROCKSTAR catalogue and the rest of the position variables are in Mpc/h, then it needs to be converted from kpc/h
        pos = np.vstack([hdf['X'][()], hdf['Y'][()], hdf['Z'][()],]).T
        vel = np.vstack([hdf['VX'][()], hdf['VY'][()], hdf['VZ'][()],]).T   # Should be in physical units, usually in km/s as ROCKSTAR gives them
    f32 = np.float32
    return dict(ID=hid, pos=pos, vel=np.asarray(vel, f32), M200b=np.asarray(m200b, f32),
                R200b=np.asarray(r200b, f32), Rs=np.asarray(rs, f32))
```
For the examples shown below, they use the same readers for HDF5 files. For GADGET binary files, you write an equivalent reader that returns the same dictionary. 

## Running `build_tiles` and `run_tiled_orbiting_mass_assignment`

Each reader and its list of files are wrapped in a `ChunkTasks`, which `build_tiles` calls one file at a time. 

After building the tiles, you must calibrate the kinetic energy cut lines. You may choose to self-calibrate with particle data or to use the cosmology calibration. Cosmology calibration is simplest and recommended for standard $\Lambda CDM$ boxes, whereas self-calibration is recommended for nonstandard cosmologies. If using the cosmology calibration, it is important to remember the `omega_m` passed to `calibrate` is the $\Omega_{m}$ at the snapshot's redshift, while `mass_density` stays at the comoving $z=0$ value.

Finally, you may run `run_tiled_orbiting_mass_assignment`. This is the final step of using <span style="font-variant:small-caps;">Oasis</span>, and will produce the following files to the `save_path`:

```bash
└── save_path
    ├── tiles
    ├── run_<run_name>
    │   ├── mini_box_catalogues
    │   ├── tiles_done
    │   ├── catalogue.hdf5
    │   ├── members.hdf5
    │   ├── manifest.json
    │   └── timings.json
    ├── calibration_data.hdf5
    └── calibration_pars.hdf5
```

Details of each file can be found in the [README](../README.md#data-products) file. If storage is a concern, the most important files are `catalogue.hdf5` and `members.hdf5`. If using the cosmology-calibrated option, `calibration_data.hdf5` will not appear. 

Below are examples of how to run `build_tiles`, `calibrate`, and `run_tiled_orbiting_mass_assignment` on a small simulation that can fit into memory (one tile) and a large simulation that must be processed with multiple tiles. 

### Example 1: a box that fits in memory (one tile)

With `tiles_per_side = 1` the whole periodic box is a single tile. There is no ribbon (`buffer_width=0`): a cell's padding wraps around the box. This gives exactly the same catalogue as the mini-box workflow of <span style="font-variant:small-caps;">Oasis</span> v1. 

```python
boxsize = 1000.
tiles_per_side = 1
buffer_width = 0.
inner_cell_size = 100.
padding = 5.
spec = TileSpec(boxsize=boxsize, tiles_per_side=tiles_per_side, buffer_width=buffer_width, inner_cell_size=inner_cell_size)
tile_paths = [save_path + 'tiles/tile_0.hdf5']

if __name__ == '__main__':
    # 1. Build the tile: each snapshot file is read once per pass, 8 at a time.
    # refuses to run if the tile already exists
    if not os.path.exists(save_path + 'tiles/tile_0.hdf5'):
        build_tiles(ChunkTasks(read_particle_file, particle_files),
                    ChunkTasks(read_seed_file, seed_files),
                    save_path + 'tiles', spec, particle_mass=particle_mass, n_workers=8)

    # 2. Calibrate: self-calibration on the simulation's own haloes (see template.py, option 2) ...
    seeds = [read_seed_file(f) for f in seed_files]
    pos, vel, m200b, r200b = (np.concatenate([s[k] for s in seeds])
                              for k in ('pos', 'vel', 'M200b', 'R200b'))
    calibrate(
        save_path=save_path,
        omega_m=None,                       # None selects self-calibration
        n_seeds=n_seeds,
        seed_data=(pos, vel, m200b, r200b),
        r_max=r_max,
        boxsize=boxsize,
        minisize=inner_cell_size,           # seeds are grouped by cell
        particle_type='dm',
        mass_density=mass_density,
        redshift=redshift,
        n_radial_bins=calib_n_points,
        percent=calib_vrp_percent,
        width=calib_vrn_width,
        gradient_radial_lims=calib_gradient_radial_limits,
        n_threads=n_threads,
        diagnostics=diagnostic_plots,
        overwrite=False,
        tile_paths=tile_paths,
    )
    #    ... or cosmology-calibration using `omega_m = omega_m`.

    # 3. Find haloes and merge into run_<run_name>/catalogue.hdf5 and members.hdf5.
    run_tiled_orbiting_mass_assignment(
        tile_paths=tile_paths, load_path=save_path, run_name=run_name,
        min_num_part=n_orb_min, boxsize=boxsize, padding=padding, particle_type='dm',
        redshift=redshift, fast_mass=fast_mass, n_threads=n_threads, ball_radius_solver=ball_radius_solver)
```
A Quijote box ($1024^{3}$ particles) builds in about 4 minutes and processes in about 3-6 minutes, depending on threads. The tile is about the size of the snapshot; memory per worker grows with `inner_cell_size`.

### Example 2: a box too large for one tile

The box is split into `tiles_per_side`$^{3}$ tiles. Each tile stores its core plus a ribbon of width `buffer_width` (copies of the neighboring tiles' particles), so it can be processed on its own. `buffer_width` must be at least `padding`. Each `build_tiles` call reads all the snapshot files twice, so the 5 groups shown in the following example mean 10 reads of the snapshot files (every group reads every file, even files with no particles in its tiles). 

```python
boxsize = 1000.
tiles_per_side = 5
inner_cell_size = 12.5
padding = 7.5
buffer_width = padding
spec = TileSpec(boxsize=boxsize, tiles_per_side=tiles_per_side, buffer_width=buffer_width, inner_cell_size=inner_cell_size)
tile_paths = [save_path + f'tiles/tile_{t}.hdf5' for t in range(spec.n_tiles)]   # 125 tiles

if __name__ == '__main__':
    # 1. Build the tiles in groups (here 5 slabs of 25): temporary bucket files exist
    #    only for the group being built, which bounds the disc used.
    for slab in range(5):
        ids = list(range(25 * slab, 25 * slab + 25))
        todo = [t for t in ids if not os.path.exists(save_path + f'tiles/tile_{t}.hdf5')]
        if todo:
            # refuses to run if the tile already exists
            build_tiles(ChunkTasks(read_particle_file, particle_files),
                        ChunkTasks(read_seed_file, seed_files),
                        save_path + 'tiles', spec, particle_mass=particle_mass,
                        tile_ids=todo, n_workers=8)

    # 2. Calibrate once (cosmology-based is the simplest for large boxes).
    calibrate(save_path=save_path, omega_m=omega_m)

    # 3. Process every tile, then merge and verify. Restartable: rerunning the same
    #    call skips finished tiles.
    run_tiled_orbiting_mass_assignment(
        tile_paths=tile_paths, load_path=save_path, run_name=run_name,
        min_num_part=n_orb_min, boxsize=boxsize, padding=padding, particle_type='dm',
        redshift=redshift, fast_mass=fast_mass, n_threads=n_threads, ball_radius_solver=ball_radius_solver)
```

### Running a large box as separate jobs

To run large simulations (e.g. MDPL2, VSMDPL), you may want to run one job per tile or per group of tiles. To do this, you may want to run `process_tile` for each tile, then `merge_tiled_run` once, after every tile of the box is finished. If you are testing a subset of the tiles, you may want to merge only the finished tiles; pass `partial=True` in `merge_tiled_run` to merge all finished tiles, regardless of whether all tiles of the box are finished. This would be done in place of `run_tiled_orbiting_mass_assignment`, which runs all tiles as one job and merges when all tiles are done processing. This allows one group's tile building to run while the previous group is being processed, which can cut down on runtime. While job `k` (`slab` from the code block above) processes group `k`, a separate job can build group `k + 1`. Never process the same tile in two jobs at once: there is no lock to stop one job from overwriting another job.

```python
# Job k processes group k (k = 0-4, e.g. k = int(sys.argv[1])).
# It can start as soon as the tiles of group k are built.
for t in range(25 * k, 25 * k + 25):
    process_tile(save_path + f'tiles/tile_{t}.hdf5', load_path=save_path, run_name=run_name,
                 min_num_part=n_orb_min, boxsize=boxsize, padding=padding, particle_type='dm',
                 redshift=redshift, fast_mass=fast_mass, n_threads=n_threads, ball_radius_solver=ball_radius_solver)

# once, after every job has finished:
merge_tiled_run(save_path, run_name)
```

### Choosing the parameters

Below is a table of the parameters that you may change depending on the geometry of the simulation. It includes examples of values you may choose. 

| Parameter | Rule | Example |
|---|---|---|
| `tiles_per_side` | 1 if the box fits into memory; otherwise enough that one tile fits comfortably in memory | 1 (Quijote), 5 (MDPL2) |
| `padding` | at least 2 x the largest seed $R_{\rm 200b}$ | 5 (Quijote), 7.5 (MDPL2) |
| `buffer_width` | 0 for one tile; otherwise at least `padding` | 0, 7.5 |
| `inner_cell_size` | must divide `boxsize / tiles_per_side` | 100 (Quijote), 12.5 (MDPL2) |
| `n_workers` | number of snapshot files read at the same time during `build_tiles`; memory grows with `n_workers` x one file | 8 (Quijote and MDPL2) |
| `n_threads` | number of cells processed in parallel (one process each); default is half the CPU cores. Memory grows with `n_threads` × one padded cell, so lower it if you run out of memory | 32 (Quijote), 48 (MDPL2) |

Per tile, the speed-up stops at the number of cells (`boxsize / tiles_per_side / inner_cell_size`)$^{3}$ (1000 cells for Quijote, 4096 per MDPL2 tile).

### Checking the result

The merge step already checks the catalogue (`verify_catalogue`): halo IDs are unique and every halo's member range matches its `Norb`. If it finished without an error, the catalogue is complete. Each halo's members are a contiguous slice of `members.hdf5`, from `LIDX` up to (not including) `RIDX`:

```python
import h5py
import numpy as np

run = save_path + f'run_{run_name}/'
with h5py.File(run + 'catalogue.hdf5', 'r') as cat:
    halo_id = cat['Halo_ID'][()]
    morb = cat['Morb'][()]             # orbiting mass, h^-1 M_sun
    m200b = cat['M200b'][()]
    pid = cat['PID'][()]               # -1 for host halos
    lidx, ridx = cat['LIDX'][()], cat['RIDX'][()]

print(f'{len(halo_id)} halos, {(pid == -1).sum()} hosts')
print('median Morb / M200b:', np.median(morb / m200b))   # should be close to 1

# Particle IDs of the orbiting particles of halo i (read only that slice:
# members.hdf5 can hold billions of rows).
i = 0
with h5py.File(run + 'members.hdf5', 'r') as mem:
    member_pids = mem['PID'][lidx[i]:ridx[i]]
```