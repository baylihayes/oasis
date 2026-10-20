![](res/oasis.jpg "title")
# Orbiting Mass Assignment Scheme

When using this code please cite Salazar et. al. (2025) ([arXiv:XXXX.XXXX]()).

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
    - [User parameters](#user-parameters)
    - [Build tiles](#step-1-build-tiles)
    - [Calibration](#step-2-calibration)
    - [Running Oasis](#step-3-run-orbiting-mass-assignment)
    - [Existing mini-box data](#existing-mini-box-data)
- [Data products](#data-products)
    - [Halo catalogue](#dynamical-halo-catalogue)
    - [Members catalogue](#halo-members-catalogue)

## Requirements
Create a conda environment. An `environment.yml` file is supplied as an example but feel free to use your own. It needs Python 3.10 or later with `numpy`, `scipy`, `pandas`, `h5py`, `tqdm`, and `matplotlib`; `pip` installs them with <span style="font-variant:small-caps;">Oasis</span> if they are missing.
```sh
$ conda env create -f environment.yml
$ conda activate oasis
```
Alternatively
```sh
$ conda create --name <env name> python=3.10 numpy scipy pandas h5py tqdm matplotlib
```
Reading compressed HDF5 snapshots (e.g. Quijote) may also need [`hdf5plugin`](https://github.com/silx-kit/hdf5plugin), imported in your own reading code.

## Installation
Build from source
```sh
$ git clone https://github.com/edgarmsalazar/oasis.git
$ cd oasis
$ python -m pip install .
```
To run the tests (about 3 minutes):
```sh
$ python -m pip install ".[test]"
$ python -m pytest test
```

## Usage
A [template](template.py) file is provided as an example of how to use the code. It consists of three basic steps: building tiles from the simulation data, calibrating, and finding haloes.

### User parameters
The choice of units is not strict to the ones shown below. However, it must be consistent across all quantities with the same physical dimensions. For example, if the units for the `boxsize` are in $h^{-1}{\rm Mpc}$, then all distances, positions and radii must also be provided in $h^{-1}{\rm Mpc}$.

- `boxsize`: side length of the simulation box in units of $h^{-1}{\rm Mpc}$.
- `tiles_per_side`: the box is stored as `tiles_per_side`$^3$ tile files. Use `1` for a box that fits in memory; large boxes (e.g. MDPL2) use several tiles per side, each built and processed on its own.
- `inner_cell_size`: side length of the cells each tile is split into, in units of $h^{-1}{\rm Mpc}$. Cells are processed in parallel; it must divide `boxsize / tiles_per_side`.
- `padding`: distance around each cell from which particles and seeds are also loaded, in units of $h^{-1}{\rm Mpc}$. It must be at least the largest $2R_{\rm 200b}$ of the seeds.
- `buffer_width`: width of the ribbon each tile stores around its core, so that its edge cells can be padded without opening another tile. `0` for one tile per side; otherwise at least `padding`.
- `mass_density`: mass density of the Universe in units of $h^{2} M_{\odot} / {\rm Mpc}^{3}$.
- `particle_mass`: dark matter particle mass in units of $h^{-1} M_{\odot}$.
- `save_path`: path to the directory where <span style="font-variant:small-caps;">Oasis</span> will save all data products.

With `tiles_per_side=1` and `buffer_width=0`, the catalogue is identical to the mini-box workflow of earlier <span style="font-variant:small-caps;">Oasis</span> versions with `minisize = inner_cell_size` and the same `padding`. The catalogue does not depend on `inner_cell_size`: larger cells are faster (fewer seeds are processed twice in neighboring cells' padding) but use more memory per worker.

### Step 1: Build tiles
<span style="font-variant:small-caps;">Oasis</span> does not assume any type or form of data storage format like Gadget, HDF, or any other, and leaves the task of loading data to the user. This is to avoid adding dependencies specific to any simulation and keeping the API simple and lightweight. The input data must include:

- Particles: ID (pid), $\vec{x}$, $\vec{v}$.
- Seeds: ID (hid), $\vec{x}$, $\vec{v}$, $M_{\rm 200b}$, $R_{\rm 200b}$, $R_{s}$.

where $R_{s}$ is the NFW scale radius, and $M_{\rm 200b}$ is [<span style="font-variant:small-caps;">Rockstar</span>](https://ui.adsabs.harvard.edu/abs/2013ApJ...762..109B/abstract) mass definition.

> Velocities are assumed to be physical velocities scaled by the scale factor $a$, like $v_{\rm phys}/\sqrt{a}$. The kinetic energy $T\sim a^{-2}$ so that $\ln\left(a^2v^2/v_{\rm 200m}\right) - r/R_{\rm 200m}$ space is within a fixed range. Please make sure the input velocities are in this convention.

The data are passed to `build_tiles` one file at a time, so the whole box never has to be in memory. You write one function that reads a single particle file and one that reads a single seed file, each returning a dictionary of arrays, and `ChunkTasks` applies them to your lists of files:
```python
import numpy as np
from oasis.tiles import ChunkTasks, TileSpec, build_tiles

def read_particle_file(path):
    pid, pos, vel = ...         # load one file
    return dict(ID=pid, pos=pos, vel=np.asarray(vel, np.float32))

def read_seed_file(path):
    hid, pos, vel, m200b, r200b, rs = ...         # load one file
    f32 = np.float32
    return dict(ID=hid, pos=pos, vel=np.asarray(vel, f32), M200b=np.asarray(m200b, f32),
                R200b=np.asarray(r200b, f32), Rs=np.asarray(rs, f32))

spec = TileSpec(boxsize=boxsize, tiles_per_side=1, buffer_width=0.,
                inner_cell_size=inner_cell_size)
build_tiles(ChunkTasks(read_particle_file, particle_files),
            ChunkTasks(read_seed_file, seed_files),
            save_path + 'tiles', spec, particle_mass=particle_mass, n_workers=8)
```
With `n_workers > 1`, that many files are read in parallel. The readers must be defined at the top level of a module, not inside another function, so that the worker processes can call them. Data already in memory can be passed with `oasis.tiles.array_chunks(dict_of_arrays)` instead of `ChunkTasks`. Large boxes can be built in groups of tiles with `tile_ids=[...]`. Existing mini-box files of earlier versions are converted with `oasis.minibox.minibox_chunks`.

The tiles need about the same disk space as the input data (plus the ribbons), and temporary files of the same size while they are built.

### Step 2: Calibration
Before finding haloes, <span style="font-variant:small-caps;">Oasis</span> needs to be calibrated. That is, it needs to find the cut line in $\ln\ a^2v^2-r$ space that classifies particles into orbiting and infalling. This is only done once per simulation box, so subsequent runs on the same box use the same calibration.

There are two options for calibrating the algorithm.

1. Assume cosmological dependence. The slope of both cut lines depends on $\Omega_m(z)$, the matter density parameter at the snapshot's redshift, as $$m = -1.903 + 0.4348\,\ln\left(\Omega_m(z)/0.5\right),$$ a fit to the self-calibrated slopes of the [Quijote](https://quijote-simulations.readthedocs.io/en/latest/) latin hypercube sample. The values of the lines at $r/R_{\rm 200m}=0.5$ are fixed to the values found in Salazar et. al. (2025) ([arXiv:XXXX.XXXX]()).

    > **RESULTS MAY VARY**. While this choice reduces noise in mass function measurements across different cosmologies, it is recommended to do a sanity check against calibration directly on the data.

    ```python
    from oasis.calibration import calibrate

    calibrate(save_path=save_path, omega_m=omega_m)
    ```

2. Use a sample of isolated haloes from the simulation data. This is part of the original pipeline but can introduce noise at the mass function level as it depends on the halo sample. The particles around the haloes are read from the tiles (`tile_paths`).
    - `n_seeds`: number of the most massive isolated seeds to use (depends on mass resolution).
    - `r_max`: search radius in units of $h^{-1}{\rm Mpc}$. All particles within this radius will be collected for calibration.
    - `minisize`: the tiles' `inner_cell_size` (seeds are grouped by cell).
    - `redshift`: cosmological redshift of the simulation box (snapshot).
    - `percent`: calibration parameter for $v_r>0$. Sets the target fraction of particles below the cut line. Defaults to $0.995$.
    - `width`: calibration parameter for $v_r<0$. Sets the width of the band around the cut line. Defaults to $0.050$.
    - `n_radial_bins`: number of gradient points to use when finding the slope of the cut line. Defaults to $20$.
    - `gradient_radial_lims`: radial interval where the gradient points are taken from. Defaults to $(0.2, 0.5)$ in units of $r/R_{\rm 200m}$.
    - `n_threads`: number of worker processes. Defaults to `None` (half the cores).

    > Input data order matters: $\vec{x}$, $\vec{v}$, $M_{\rm 200b}$, $R_{\rm 200b}$.

    ```python
    from oasis.calibration import calibrate

    calibrate(
        save_path=save_path,
        omega_m=None,                    # None selects self-calibration
        n_seeds=n_seeds,
        seed_data=(pos, vel, m200b, r200b),
        r_max=r_max,
        boxsize=boxsize,
        minisize=inner_cell_size,
        particle_type='dm',
        mass_density=mass_density,
        redshift=redshift,
        tile_paths=[save_path + f'tiles/tile_{t}.hdf5' for t in range(spec.n_tiles)],
        n_threads=n_threads,
    )
    ```

    Depending on the number of haloes and mass resolution of the simulation it may take from a couple of seconds to a couple minutes ($<10$ min) per box.

As a recommendation, always check that the calibration was done properly and makes sense before running the mass assignment. Here is the output of the previous function call.

<img src="res/calibration.png" alt="calibration" class="center" height="300"/>

### Step 3: Run orbiting mass assignment
Once calibrated, `run_tiled_orbiting_mass_assignment` generates a dynamical halo catalogue and the corresponding members catalogue. This is the main algorithm and can be run many times with different settings, which are saved into different directories.

The additional parameters to set are:

- `min_num_part`: only dynamical haloes with at least this number of orbiting particles will be considered.
- `fast_mass`: when `True`, <span style="font-variant:small-caps;">Oasis</span> will perform a simple percolation. It speeds things up but results are only comparable to the full percolation at the mass function level, i.e. members will differ from _true_ percolation and should only be set to `True` when there are hardware limitations. Defaults to `False`.
- `ball_radius_solver`: how the radius of the 6D ball used to identify substructure is found. `'brentq'` always finds the intended radius; `'fsolve'` (the default) is the original solver, which can fail for small substructures of massive hosts (which matters for high-resolution boxes).
- `run_name`: <span style="font-variant:small-caps;">Oasis</span> appends this name to identify the results and distinguish them from subsequent runs in the same box.

```python
from oasis.catalogue import run_tiled_orbiting_mass_assignment

run_tiled_orbiting_mass_assignment(
    tile_paths=[save_path + f'tiles/tile_{t}.hdf5' for t in range(spec.n_tiles)],
    load_path=save_path,
    run_name=run_name,
    min_num_part=n_orb_min,
    boxsize=boxsize,
    padding=padding,
    particle_type='dm',
    redshift=redshift,
    fast_mass=fast_mass,
    n_threads=n_threads,
    ball_radius_solver='brentq',
)
```

For large boxes, `process_tile` processes one tile per call (finished tiles are skipped when a run is restarted) and `merge_tiled_run` merges the finished tiles into one catalogue, checks it and writes a `manifest.json` with the settings and code version.

### Existing mini-box data
The mini-box workflow of earlier versions (`oasis.minibox.process_simulation_data` and `oasis.catalogue.run_orbiting_mass_assignment`) is still available, for existing mini-box data and for reproducing the original <span style="font-variant:small-caps;">Oasis</span> exactly. It gives the same catalogue as one tile per side, but holds the whole box in memory and writes one file per mini-box.

## Data products
Once <span style="font-variant:small-caps;">Oasis</span> is finished, all generated data products will be found at `save_path`.
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

- `tiles`: the simulation data arranged for <span style="font-variant:small-caps;">Oasis</span>, one file per tile (`tile_<t>.hdf5`), with the rows grouped by cell. It is recommended not to remove this directory until all subsequent runs on the same box are finished.
- `run_<run_name>/mini_box_catalogues`: the dynamical halo catalogue and members of every cell. These files are merged into `catalogue.hdf5` and `members.hdf5`, so the directory (and `tiles_done`, which records the finished tiles) can be removed once the catalogue is final.
- `run_<run_name>/manifest.json`: the run settings, the calibration file's checksum, the <span style="font-variant:small-caps;">Oasis</span> version (git commit) and per-tile timings; `timings.json` holds the time and memory of every stage.
- `calibration_data.hdf5` (self-calibration only): $r/R_{\rm 200m}$, $v_r/v_{\rm 200m}$ and $\ln(v^2/v^2_{\rm 200m})$ for all particles used for calibration.
- `calibration_pars.hdf5`: the slopes and values at the pivot of both cut lines at $v_r>0$ (`pos`) and $v_r<0$ (`neg/line`), and the low-radius correction (`neg/quad`).

If disc space is a concern, the most relevant files to keep are `catalogue.hdf5` and `members.hdf5` since the calibration is for internal use.

### Dynamical Halo Catalogue
The `catalogue.hdf5` file contains the following columns:
- `Halo_ID`: halo ID corresponding to the user input data `hid` for consistency.
- `M200b`, `R200b`, `pos`, `vel`: user input data.
- `Norb`: number of orbiting particles.
- `Morb`: orbiting mass, $M_{\rm orb}=m_p N_{\rm orb}$, where $m_p$ is the particle mass.
- `NSUBS`: number of orbiting seeds.
- `PID`: parent dynamical halo ID. If the seed is orbiting another halo, then `PID` is the `Halo_ID` of the parent halo, `PID=-1` otherwise.
- `LIDX`, `RIDX`: row index in the `members.hdf5` file corresponding to the first ('left') and last ('right') member particle of the halo.
- `SLIDX`, `SRIDX`: row index in the `members.hdf5` file corresponding to the first ('left') and last ('right') member seed of the halo. If the halo has no orbiting seeds then both `SLIDX=-1` and `SRIDX=-1`.

### Halo Members Catalogue
The `members.hdf5` file contains only two datasets:
- `Halo_ID`: halo ID corresponding to the user input data `hid`.
- `PID`: particle ID corresponding to the user input data `pid`.