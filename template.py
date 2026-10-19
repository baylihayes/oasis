"""Template file for running OASIS: Orbiting mAss asSIngment Scheme

All loading of simulation data and catalogues is left to the user (functions
`read_particle_file` and `read_seed_file` below, each reading one file), to
avoid dependencies specific to any simulation. OASIS then runs in three steps:

1. Build tiles: the box is stored as tiles_per_side**3 tile files, each split
   into inner cells of size inner_cell_size. For a box that fits in memory use
   tiles_per_side = 1 and buffer_width = 0: one tile for the whole box, which
   gives exactly the same catalogue as the mini-box workflow of earlier OASIS
   versions with minisize = inner_cell_size and the same padding. Large boxes
   use several tiles per side (each tile is then built and processed on its
   own, with a ribbon of width buffer_width >= padding around its core).
2. Calibrate: cosmology-based (needs only Omega_m) or self-calibration on the
   simulation's own isolated haloes.
3. Find haloes: classify the particles of every cell and merge the cells into
   one catalogue (catalogue.hdf5) and its members (members.hdf5).
"""
import numpy as np

from oasis.calibration import calibrate
from oasis.catalogue import run_tiled_orbiting_mass_assignment
from oasis.common import G_GRAVITY, ensure_dir_exists, timer
from oasis.tiles import ChunkTasks, TileSpec, build_tiles

# ==============================================================================
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
save_path: str = '/'

# Tiling. `padding` is the distance around each cell from which particles and
# seeds are also loaded: it must be at least the largest 2 * R200b of the
# seeds, so every seed sees all its particles. Larger inner cells reprocess
# fewer seeds (faster) but need more memory per worker; the catalogue does not
# depend on the cell size.
padding: float = 5.                                 # h^-1 Mpc
tiles_per_side: int = 1                             # 1 tile = the whole box
inner_cell_size: float = 100.                       # h^-1 Mpc; divides boxsize / tiles_per_side
buffer_width: float = 0. if tiles_per_side == 1 else padding
n_threads: int = 32

# Calibration (self-calibration only).
n_seeds: int = 500
r_max: float = 5.0                                  # h^-1 Mpc
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

spec = TileSpec(boxsize=boxsize, tiles_per_side=tiles_per_side,
                buffer_width=buffer_width, inner_cell_size=inner_cell_size)
tile_paths = [save_path + f'tiles/tile_{t}.hdf5' for t in range(spec.n_tiles)]

# ==============================================================================
#
#                               OASIS sample run
#
# ==============================================================================
# Your input files. Simulations usually come in many files (e.g. one GADGET
# snapshot split into hundreds of files); OASIS reads them one at a time, so
# the box never has to fit in memory. A single file per list also works.
particle_files: list[str] = []      # e.g. sorted(glob.glob('/path/snapdir_000/snap_000.*'))
seed_files: list[str] = []          # e.g. ['/path/rockstar/halos_0.hdf5']


# The two readers below must stay at the top level of this file (not inside
# another function), so that the worker processes of build_tiles can call them.
def read_particle_file(path: str) -> dict:
    # LOAD YOUR DATA HERE: IDs, positions and velocities of the particles in
    # the file `path`. Velocities must be v_phys / sqrt(a) (GADGET convention).
    pid, pos, vel = ()
    return dict(ID=pid, pos=pos, vel=np.asarray(vel, np.float32))


def read_seed_file(path: str) -> dict:
    # LOAD YOUR DATA HERE: IDs, positions, velocities, M200b, R200b and NFW
    # scale radii of the halo seeds in the file `path` (e.g. a Rockstar
    # catalogue).
    hid, pos, vel, m200b, r200b, rs = ()

    # Stored as float32, like the mini-box files of earlier versions.
    f32 = np.float32
    return dict(ID=hid, pos=pos, vel=np.asarray(vel, f32), M200b=np.asarray(m200b, f32),
                R200b=np.asarray(r200b, f32), Rs=np.asarray(rs, f32))


@timer(fancy=False)
def make_tiles() -> None:
    # ChunkTasks(reader, files) gives build_tiles one file at a time, read as
    # reader(file). With n_workers > 1, that many files are read in parallel,
    # so memory grows with n_workers times the size of one file; the tiles
    # are the same for any n_workers. Use tile_ids=[...] to build a large box
    # in groups of tiles. (Arrays already in memory can be passed with
    # oasis.tiles.array_chunks(dict_of_arrays) instead of ChunkTasks.)
    build_tiles(ChunkTasks(read_particle_file, particle_files),
                ChunkTasks(read_seed_file, seed_files),
                save_path + 'tiles', spec, particle_mass=particle_mass, n_workers=8)
    return None


@timer(fancy=False)
def calibration() -> None:

    # OPTION 1 ======================================================
    # Assume cosmological dependence on calibration parameters
    # (omega_m is Omega_m at the snapshot's redshift).
    calibrate(
        save_path=save_path,
        omega_m=omega_m,                      # omega_m = Omega_m at `redshift`
    )

    # OPTION 2 ======================================================
    # Calibrate on the simulation's own isolated haloes, reading the particles
    # around them from the tiles. Use one option or the other.

    # # All seeds at once (seeds are small next to the particles).
    # seeds = [read_seed_file(f) for f in seed_files]
    # pos, vel, m200b, r200b = (np.concatenate([s[k] for s in seeds])
    #                           for k in ('pos', 'vel', 'M200b', 'R200b'))
    # calibrate(
    #     save_path=save_path,
    #     omega_m=None,                       # None selects self-calibration
    #     n_seeds=n_seeds,
    #     seed_data=(pos, vel, m200b, r200b),
    #     r_max=r_max,
    #     boxsize=boxsize,
    #     minisize=inner_cell_size,           # seeds are grouped by cell
    #     particle_type='dm',
    #     mass_density=mass_density,
    #     redshift=redshift,
    #     n_radial_bins=calib_n_points,
    #     percent=calib_vrp_percent,
    #     width=calib_vrn_width,
    #     gradient_radial_lims=calib_gradient_radial_limits,
    #     n_threads=n_threads,
    #     diagnostics=diagnostic_plots,
    #     overwrite=False,
    #     tile_paths=tile_paths,
    # )

    return None


@timer(fancy=False)
def run_oasis():

    # Processes every tile and merges them into run_<run_name>/catalogue.hdf5
    # and members.hdf5. For large boxes, oasis.catalogue.process_tile (one
    # tile per call, restartable) and merge_tiled_run do the same in steps.
    run_tiled_orbiting_mass_assignment(
        tile_paths=tile_paths,
        load_path=save_path,
        run_name=run_name,
        min_num_part=n_orb_min,
        boxsize=boxsize,
        padding=padding,
        particle_type='dm',
        redshift=redshift,
        fast_mass=fast_mass,
        n_threads=n_threads,
        ball_radius_solver=ball_radius_solver,
    )

    return


@timer(fancy=False)
def main():

    make_tiles()
    calibration()
    run_oasis()

    return


if __name__ == '__main__':
    ensure_dir_exists(save_path)
    main()
