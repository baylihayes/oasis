import os
from multiprocessing import Pool
from typing import Dict, Optional, Tuple, Union

import h5py
import numpy
from matplotlib import pyplot
from matplotlib.colorbar import ColorbarBase
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit, minimize
from scipy.spatial import cKDTree
from scipy.stats import iqr as interquartile_range
from tqdm import tqdm

from oasis.common import (G_GRAVITY, _validate_boxsize_minisize,
                          _validate_coordinate_array, _validate_existing_path,
                          _validate_mini_box_id, _validate_positive_number,
                          _validate_positive_number_non_zero,
                          _validate_seed_data)
from oasis.coordinates import (_periodic_displacement, relative_coordinates,
                               velocity_components)
from oasis.minibox import get_mini_box_id, load_particles

__all__ = [
    'get_calibration_data',
    'calibrate',
]

# Label sizes
SIZE_TICKS: int = 12
SIZE_LEGEND: int = 14
SIZE_LABELS: int = 16

COLOR_BLUE = '#6591b5'
COLOR_GRAY = '#808080'
COLOR_RED = '#db715b'


def _compute_r200m_and_v200m(
    radial_distances: numpy.ndarray,
    particle_mass: Union[float, numpy.ndarray],
    mass_density: float
) -> Tuple[float, float]:
    """
    Compute overdensity radius R200m and circular velocity V200m for a halo.

    This function calculates the overdensity radius R200m and corresponding 
    circular velocity V200m of a halo by analyzing the radial density profile. 
    R200m is defined as the radius at which the mean enclosed density equals 200 
    times the mean matter density of the universe.

    Parameters
    ----------
    radial_distances : numpy.ndarray
        Array of radial distances from the halo center with shape (N,).
        Values must be non-negative.
    particle_mass : Union[float, numpy.ndarray]
        Mass of each simulation particle. Must be positive.
    mass_density : float
        Mean matter density of the universe. Must be positive.

    Returns
    -------
    r200m : float
        Overdensity radius R200m. Returns 0.0 if computation fails.
    v200_squared : float
        Square of circular velocity V200m. Returns 0.0 if computation fails.

    Raises
    ------
    TypeError
        If radial_distances cannot be converted to numpy array.
    ValueError
        If particle_mass or mass_density are not positive scalars, or if 
        radial_distances contains negative values.
    RuntimeError
        If density profile computation encounters numerical issues.

    See Also
    --------
    G_GRAVITY : Gravitational constant used in V200m computation

    Notes
    -----
    - The algorithm sorts particles by radius and computes cumulative mass profile
    - Density is computed as rho(r) = M(<r) / (4π/3 x r^3)  
    - R200m corresponds to the largest radius where rho(r) ≥ 200*rho_m
    - If no radius satisfies the criterion, uses the outermost particle
    - V200m^2 = G*M200m/R200m where G is the gravitational constant

    Examples
    --------
    >>> import numpy
    >>> distances = numpy.array([0.1, 0.2, 0.3, 0.5, 0.8, 1.2])
    >>> r200m, v200_sq = _compute_r200m_and_v200(distances, 1e10, 2.78e11)
    >>> print(f"R200m = {r200m:.3f} Mpc/h, V200m = {numpy.sqrt(v200_sq):.1f} km/s")

    >>> # Handle edge case with no particles
    >>> r200m, v200_sq = _compute_r200m_and_v200(numpy.array([]), 1e10, 2.78e11)
    >>> print(f"Empty input: R200m = {r200m}, V200m^2 = {v200_sq}")
    Empty input: R200m = 0.0, V200m^2 = 0.0
    """
    # Input validation
    if radial_distances.size == 0:
        raise ValueError("radial_distances is empty")
    _validate_positive_number_non_zero(mass_density, 'mass_density')

    if isinstance(particle_mass, numpy.ndarray) and \
        (radial_distances.size != particle_mass.size):
        raise ValueError("particle_mass must be of the same size as radial_distances")

    # Sort distances for cumulative mass profile
    argsort = numpy.argsort(radial_distances)
    sorted_distances = radial_distances[argsort]

    # Avoid division by zero for particles at the center. Replace zero distances 
    # with a small value
    min_distance = numpy.finfo(float).eps
    sorted_distances = numpy.maximum(sorted_distances, min_distance)

    # Compute cumulative mass profile
    if isinstance(particle_mass, numpy.ndarray):
        mass_profile = numpy.cumsum(particle_mass[argsort])
    else:
        n_particles = len(sorted_distances)
        mass_profile = particle_mass * numpy.arange(1, n_particles + 1)

    # Compute volume-averaged densities
    volumes = (4.0 / 3.0) * numpy.pi * sorted_distances**3
    densities = mass_profile / volumes

    # Find where density drops below 200 * mass_density
    target_density = 200.0 * mass_density
    mask_above_target = densities >= target_density

    if not numpy.any(mask_above_target):
        # If no particles satisfy criterion, use outermost particle
        r200m = sorted_distances[-1]
        m200m = mass_profile[-1]
    else:
        # Use last particle that satisfies the density criterion
        last_valid_idx = numpy.where(mask_above_target)[0][-1]
        r200m = sorted_distances[last_valid_idx]
        m200m = mass_profile[last_valid_idx]

    # Compute V200m^2 with safety check
    if r200m > 0 and m200m > 0:
        v200m_squared = G_GRAVITY * m200m / r200m
        if not numpy.isfinite(v200m_squared) or v200m_squared <= 0:
            v200m_squared = 0.0
    else:
        v200m_squared = 0.0

    return float(r200m), float(v200m_squared)


def _get_candidate_seed_particle_data(
    mini_box_id: int,
    position_seeds: numpy.ndarray,
    velocity_seeds: numpy.ndarray,
    r_max: float,
    boxsize: float,
    minisize: float,
    load_path: str,
    particle_type: str,
    mass_density: float,
    redshift: float,
) -> numpy.ndarray:
    """
    Extract and process particle data for all seeds within a single mini-box.

    This function loads particles from a specified mini-box and its neighbors, then
    processes each seed to compute scaled radial distances, radial velocities, and
    velocity magnitudes in units of the overdensity properties (R200m, V200m).

    The function performs the following steps for each seed:
    1. Loads particles within r_max of the seed position
    2. Computes relative positions and velocities 
    3. Calculates R200m and V200m from the density profile
    4. Scales all quantities by the overdensity properties

    Parameters
    ----------
    mini_box_id : int
        ID of the mini-box containing the seeds. Must be a valid ID within the 
        grid (0 ≤ mini_box_id < total_mini_boxes).
    position_seeds : numpy.ndarray
        Seed positions with shape (n_seeds, 3). Each row contains the 
        (x, y, z) coordinates of a seed.
    velocity_seeds : numpy.ndarray
        Seed velocities with shape (n_seeds, 3). Each row contains the
        (vx, vy, vz) velocity components of a seed.
    r_max : float
        Maximum distance from seed centers to consider particles. Must be 
        positive.
    boxsize : float
        Size of the cubic simulation box. Must be a positive.
    minisize : float  
        Size of each cubic mini-box. Must be positive and ≤ boxsize.
    load_path : str or Path
        Path to directory containing the mini-box HDF5 files. The directory
        should contain a subdirectory named 'mini_boxes_nside_<xx>/' where <xx>
        is the number of cells per side.
    particle_type : str
        Particle type identifier within each HDF5 file. Valid names are 'dm', 
        'gas' and 'star'. Default is 'dm' for same-mass objects.
    particle_mass : Union[float, numpy.ndarray]
        Mass of each simulation particle. Must be positive.
    mass_density : float
        Mean matter density of the universe. Must be positive.

    Returns
    -------
    numpy.ndarray
        Particle data array with shape (n_particles, 3) where each row contains:
        - Column 0: r/R200m - Radial distance scaled by R200m
        - Column 1: vr/V200m - Radial velocity scaled by V200m  
        - Column 2: ln(v^2/V200m^2) - Natural log of velocity squared scaled by V200m^2

        If no valid particles are found, returns empty array with shape (0, 3).

    Raises
    ------
    TypeError
        If mini_box_id is not an integer, or if array inputs cannot be
        converted to numpy arrays.
    ValueError
        If mini_box_id is negative, array shapes are incompatible, or
        if any scalar parameters are non-positive.
    FileNotFoundError
        If save_path doesn't exist or required mini-box files are missing.
    OSError
        If mini-box HDF5 files cannot be read or are corrupted.
    RuntimeError
        If particle loading fails or density profile computation encounters
        numerical issues.

    See Also
    --------
    _compute_r200m_and_v200m : Computes overdensity radius and velocity
    load_particles : Loads particles from mini-box files
    velocity_components : Decomposes velocities into radial/tangential components

    Notes
    -----
    - Uses periodic boundary conditions when computing relative coordinates
    - R200m is defined as the radius where mean enclosed density equals 200*rhom
    - V200m = \sqrt(G*M200m/R200m) where M200m is the mass within R200m
    - Memory usage scales with the number of particles within r_max of all seeds

    Examples
    --------
    >>> # Process seeds in mini-box 42
    >>> pos_seeds = numpy.array([[10.0, 15.0, 20.0], [25.0, 30.0, 35.0]])
    >>> vel_seeds = numpy.array([[100.0, 50.0, -75.0], [-80.0, 120.0, 90.0]])
    >>> data = _get_candidate_seed_particle_data(
    ...     mini_box_id=42,
    ...     position_seeds=pos_seeds, 
    ...     velocity_seeds=vel_seeds,
    ...     r_max=5.0,
    ...     boxsize=100.0,
    ...     minisize=10.0,
    ...     load_path="/data/miniboxes/",
    ...     particle_mass=1e10,
    ...     mass_density=2.78e11
    ... )
    >>> print(f"Processed {data.shape[1]} particles from {len(pos_seeds)} seeds")
    """
    # Validate inputs
    _validate_boxsize_minisize(boxsize, minisize)
    _validate_mini_box_id(mini_box_id, int(numpy.ceil(boxsize/minisize)))
    _validate_coordinate_array(position_seeds, 'seed positions')
    _validate_coordinate_array(velocity_seeds, 'seed velocities')
    _validate_existing_path(load_path)
    _validate_positive_number_non_zero(r_max, 'r_max')
    _validate_positive_number_non_zero(mass_density, 'rhom')

    # Load particles in mini-box.
    position_particles, velocity_particles, _, mass_particles = \
        load_particles(mini_box_id, boxsize, minisize, load_path, particle_type)

    # Periodic KD-tree over the particles, built once per mini-box. Candidates
    # are searched with a slightly larger radius, then the same cubic mask as
    # before is applied to them, so the selected particles (and their order)
    # are identical to masking all particles.
    position_tree = cKDTree(
        numpy.mod(position_particles.astype(numpy.float64), boxsize),
        boxsize=boxsize,
    )
    search_margin = 1e-5 * boxsize

    # Iterate over seeds in current mini-box.
    radius, radial_velocity, log_velocity_squared = ([] for _ in range(3))
    for position_seed_i, velocity_seed_i in zip(position_seeds, velocity_seeds):
        # Find particles within r_max of the seed
        candidates = numpy.sort(numpy.asarray(
            position_tree.query_ball_point(position_seed_i, r_max + search_margin,
                                           p=numpy.inf),
            dtype=numpy.intp,
        ))
        relative_position = _periodic_displacement(
            position_particles[candidates], position_seed_i, boxsize)
        mask_close = numpy.all(numpy.abs(relative_position) <= r_max, axis=1)
        idx_close = candidates[mask_close]

        # Apply mask. Particle velocity is stored as v_phys / sqrt(a), so the conversion is made before
        # finding the difference between the particle velocity and the seed velocity. 
        relative_position = relative_position[mask_close]
        relative_velocity = velocity_particles[idx_close] / numpy.sqrt(1 + redshift) - velocity_seed_i
        # Use a per-seed name so the full particle mass array is not
        # overwritten by the masked one for the next seed.
        mass_close = mass_particles[idx_close] if \
            isinstance(mass_particles, numpy.ndarray) else mass_particles

        # Compute radial distance (L2 norm). No need to further filter by r_max
        # since we already applied a cubic mask.
        rps = numpy.linalg.norm(relative_position, axis=1)

        # Compute velocity components
        vrp, _, v2p = velocity_components(relative_position, relative_velocity)

        # Compute R200m and M200m
        r200m, v200m_sq = _compute_r200m_and_v200m(rps, mass_close, mass_density)

        # Append rescaled quantities to containers
        radius.append(rps / r200m)
        radial_velocity.append(vrp / numpy.sqrt(v200m_sq))

        # Prevent log(0) or log(negative) by setting minimum value
        v_sq_ratio = v2p / v200m_sq
        v_sq_ratio = numpy.maximum(v_sq_ratio, 1e-10)

        log_velocity_squared.append(numpy.log(v_sq_ratio))

    # Concatenate arrays
    radius = numpy.concatenate(radius)
    radial_velocity = numpy.concatenate(radial_velocity)
    log_velocity_squared = numpy.concatenate(log_velocity_squared)

    return numpy.vstack([radius, radial_velocity, log_velocity_squared])


def _get_candidate_seed_particle_data_star(args: tuple) -> numpy.ndarray:
    """Unpack arguments for `_get_candidate_seed_particle_data` (for Pool.imap)."""
    return _get_candidate_seed_particle_data(*args)


def _find_isolated_seeds(
    position: numpy.ndarray,
    mass: Union[float, numpy.ndarray],
    radius: Union[float, numpy.ndarray],
    max_seeds: int,
    boxsize: float,
    isolation_factor: float = 0.2,
    isolation_radius_factor: float = 2.0,
) -> numpy.ndarray:
    """
    Find seeds that are isolated from other massive neighbors.

    This function identifies seeds that dominate their local environment by
    checking that all neighboring seeds within a specified isolation radius
    have masses below a threshold fraction of the candidate seed's mass.

    The isolation criterion requires that all neighbors within an isolation
    radius (isolation_radius_factor * R200m) must have masses less than
    isolation_factor*M200m of the candidate seed.

    Parameters
    ----------
    position : numpy.ndarray
        Seed positions with shape (n_seeds, 3) in cartesian coordinates. Each 
        row contains the (x, y, z) coordinates of a seed.
    mass : Union[float, numpy.ndarray]
        Mass of each seed. Can be a scalar if all seeds have the same mass, or 
        array with shape (n_seeds,). Must be positive.
    radius : Union[float, numpy.ndarray]
        Overdensity radius R200m, of each seed in simulation units. Can be
        a scalar if all seeds have the same radius, or array with shape 
        (n_seeds,). Must be positive.
    max_seeds : int
        Maximum number of isolated seeds to search for. Must be positive.
        Function stops searching once this many isolated seeds are found. The 
        number of returned seeds can be lower than this number if there are no
        isolated seeds or
    boxsize : float
        Size of the cubic simulation box. Must be a positive.
    isolation_factor : float, optional
        Maximum allowed mass ratio for neighbors. Neighbors must have
        mass < isolation_factor * seed_mass to satisfy isolation criterion.
        Default is 0.2 (20%).
    isolation_radius_factor : float, optional
        Factor multiplying R200m to define isolation radius. Neighbors within
        isolation_radius_factor * R200m are checked. Default is 2.0.

    Returns
    -------
    numpy.ndarray
        Array of indices of isolated seeds with shape (n_isolated,).
        Indices correspond to rows in the input position array.
        Returned in order of discovery, up to max_seeds entries.

    Raises
    ------
    IndexError
        If no isolated seeds where found.
    TypeError
        If position cannot be converted to numpy array, or if max_seeds
        is not an integer.
    ValueError
        If array shapes are incompatible, if any scalar parameters are
        non-positive, or if position array has wrong dimensions.

    See Also
    --------
    relative_coordinates : Computes relative positions with periodic boundaries
    _select_candidate_seeds : Select isolated seeds and particles around them
    _get_candidate_seed_particle_data : Processes particles around seeds in one mini-box

    Notes
    -----
    - Uses periodic boundary conditions for distance calculations
    - Searches seeds in input order, stopping when max_seeds are found
    - A seed with no neighbors within isolation radius is automatically isolated
    - Isolation radius scales with each seed's individual R200m value
    - Memory usage is O(n_seeds^2) in worst case for distance calculations

    Examples
    --------
    >>> # Find up to 100 isolated seeds
    >>> positions = numpy.random.uniform(0, 100, (1000, 3))
    >>> masses = numpy.random.uniform(1e12, 1e15, 1000)
    >>> radii = (masses / 1e12) ** (1/3) * 0.5
    >>> isolated = _find_isolated_seeds(
    ...     position=positions,
    ...     mass=masses, 
    ...     radius=radii,
    ...     max_seeds=100,
    ...     boxsize=100.0,
    ...     isolation_factor=0.3,
    ...     isolation_radius_factor=2.5
    ... )
    >>> print(f"Found {len(isolated)} isolated seeds")

    >>> # Handle uniform mass case
    >>> isolated = _find_isolated_seeds(
    ...     position=positions,
    ...     mass=1e13,  # All seeds have same mass
    ...     radius=1.0, # All seeds have same radius
    ...     max_seeds=50,
    ...     boxsize=100.0
    ... )
    """
    # Input validation
    _validate_coordinate_array(position, 'position')
    _validate_positive_number_non_zero(max_seeds, 'max_seeds')
    _validate_positive_number_non_zero(boxsize, 'boxsize')
    _validate_positive_number_non_zero(isolation_factor, 'isolation_factor')
    _validate_positive_number_non_zero(
        isolation_radius_factor, 'isolation_radius_factor')

    # This should never be needed as all seeds are assumed to have different
    # masses and radii. But just in case.
    if isinstance(mass, (int, float)):
        mass = numpy.full(position.shape[0], mass)

    if isinstance(radius, (int, float)):
        radius = numpy.full(position.shape[0], radius)

    n_seeds = position.shape[0]
    isolated_indices = []

    # Pre-compute isolation radii for efficiency
    isolation_radii = isolation_radius_factor * radius
    max_neighbor_masses = isolation_factor * mass

    # Compute KDTree with seed positions. Got a 40% improvement over previous 
    # method.
    position_tree = cKDTree(position, boxsize=boxsize)

    for i in range(n_seeds):
        # If all requested isolated seeds have been found, exit loop
        if len(isolated_indices) >= max_seeds:
            break

        current_position = position[i]
        isolation_radius = isolation_radii[i]
        max_neighbor_mass = max_neighbor_masses[i]

        # Use a KDTree to find neighbouring seeds.
        idx_neighbor = position_tree.query_ball_point(current_position, 
                                                      isolation_radius,
                                                      p=numpy.inf,
                                                      return_sorted=True)
        idx_neighbor = [item for item in idx_neighbor if item != i]

        # No neighbors found - seed is isolated
        if len(idx_neighbor) == 0:
            isolated_indices.append(i)
            continue

        neighbor_masses = mass[idx_neighbor]

        # Check isolation criterion: all neighbors must have mass < threshold
        if numpy.all(neighbor_masses < max_neighbor_mass):
            isolated_indices.append(i)
    
    if len(isolated_indices) == 0:
        raise IndexError('Found no isolated seeds. Please check input list.')

    return numpy.array(isolated_indices)


def _group_seeds_by_minibox(
    position: numpy.ndarray,
    velocity: numpy.ndarray,
    boxsize: float,
    minisize: float,
) -> Dict[int, Tuple[numpy.ndarray, numpy.ndarray]]:
    """
    Group seeds by their containing mini-box for efficient parallel processing.

    This function assigns each seed to its containing mini-box based on spatial
    coordinates and returns a dictionary mapping mini-box IDs to the positions
    and velocities of seeds within that mini-box.

    Parameters
    ----------
    position : numpy.ndarray
        Seed positions with shape (n_seeds, 3) in cartesian coordinates. Each 
        row contains the (x, y, z) coordinates of a seed.
    velocity : numpy.ndarray
        Seed velocities with shape (n_seeds, 3). Each row contains the
        (vx, vy, vz) velocity components of a seed in simulation units.
    boxsize : float
        Size of the cubic simulation box. Must be a positive.
    minisize : float  
        Size of each cubic mini-box. Must be positive and ≤ boxsize.

    Returns
    -------
    Dict[int, Tuple[numpy.ndarray, numpy.ndarray]]
        Dictionary mapping mini-box IDs to tuples of (positions, velocities).
        Each mini-box ID maps to:
        - positions: numpy.ndarray with shape (n_seeds_in_box, 3)
        - velocities: numpy.ndarray with shape (n_seeds_in_box, 3)
        Only mini-boxes containing at least one seed are included.

    Raises
    ------
    TypeError
        If position or velocity arrays cannot be converted to numpy arrays.
    ValueError
        If array shapes are incompatible, if boxsize or minisize are
        non-positive, or if minisize > boxsize.

    See Also
    --------
    get_mini_box_id : Computes mini-box ID from spatial coordinates

    Notes
    -----
    - Minibox IDs are computed using spatial hashing of seed coordinates
    - Empty mini-boxes (containing no seeds) are not included in the result
    - Seeds exactly on mini-box boundaries are assigned consistently
    - Memory usage scales linearly with the number of seeds

    Examples
    --------
    >>> # Group 1000 seeds into mini-boxes
    >>> positions = numpy.random.uniform(0, 100, (1000, 3))
    >>> velocities = numpy.random.normal(0, 200, (1000, 3))
    >>> groups = _group_seeds_by_minibox(
    ...     position=positions,
    ...     velocity=velocities, 
    ...     boxsize=100.0,
    ...     minisize=10.0
    ... )
    >>> print(f"Seeds distributed across {len(groups)} mini-boxes")
    >>> for box_id, (pos, vel) in groups.items():
    ...     print(f"Minibox {box_id}: {len(pos)} seeds")
    """
    # Validate inputs
    _validate_coordinate_array(position, 'position')
    _validate_coordinate_array(velocity, 'velocity')
    _validate_boxsize_minisize(boxsize, minisize)

    # Get mini-box IDs for all seeds
    mini_box_ids = get_mini_box_id(position, boxsize, minisize)

    # Group by mini-box ID using dictionary comprehension
    unique_ids = numpy.unique(mini_box_ids)
    minibox_groups = {}
    for minibox_id in unique_ids:
        mask = mini_box_ids == minibox_id
        minibox_groups[int(minibox_id)] = (position[mask], velocity[mask])

    return minibox_groups


def _select_candidate_seeds(
    n_seeds: int,
    seed_data: Tuple[numpy.ndarray],
    r_max: float,
    boxsize: float,
    minisize: float,
    load_path: str,
    particle_type: str,
    mass_density: float,
    redshift: float,
    isolation_factor: float = 0.2,
    isolation_radius_factor: float = 2.0,
    n_threads: Optional[int] = None,
) -> tuple[numpy.ndarray]:
    """
    Select isolated massive seeds and extract scaled particle data around them.

    This function identifies the most massive isolated seeds from the input
    catalog and loads particles within r_max of each seed to compute r/R200m and 
    kinetic energy. Only seeds that dominate their local environment are 
    considered (isolation criterion: all neighbors within 2*R200m must have
    mass < 0.2*M200 of the seed).

    The function performs the following workflow:
    1. Sorts seeds by mass in descending order
    2. Finds up to n_seeds isolated seeds using the isolation criterion
    3. Groups selected seeds by mini-box for efficient parallel processing
    4. Extracts and processes particles around each seed
    5. Returns scaled data by the seed radius and kinetic energy for all 
        processed particles

    Parameters
    ----------
    n_seeds : int
        Maximum number of seeds to process. Must be positive.
        Function selects up to this many of the most massive isolated seeds.
    seed_data : Tuple[numpy.ndarray]
        Tuple containing (position, velocity, mass, radius) arrays:
        - position: shape (n_total_seeds, 3) - seed coordinates
        - velocity: shape (n_total_seeds, 3) - seed velocities  
        - mass: shape (n_total_seeds,) - seed masses (M200m)
        - radius: shape (n_total_seeds,) - seed overdensity radii (R200m)
    r_max : float
        Maximum distance from seed centers to consider particles. Must be 
        positive.
    boxsize : float
        Size of the cubic simulation box. Must be a positive.
    minisize : float  
        Size of each cubic mini-box. Must be positive and ≤ boxsize.
    load_path : str
        Path to directory containing the mini-box HDF5 files. The directory
        should contain a subdirectory named 'mini_boxes_nside_<xx>/' where <xx>
        is the number of cells per side.
    particle_type : str
        Particle type identifier within each HDF5 file. Valid names are 'dm', 
        'gas' and 'star'. Default is 'dm' for same-mass objects.
    particle_mass : float
        Mass of each simulation particle. Must be positive.
    mass_density : float
        Mean matter density of the universe. Must be positive.
    isolation_factor : float, optional
        Maximum allowed mass ratio for isolation criterion. Neighbors must have
        mass < isolation_factor * seed_mass to satisfy isolation criterion.
        Default is 0.2 (20%).
    isolation_radius_factor : float, optional
        Factor multiplying R200m to define isolation radius. Neighbors within
        isolation_radius_factor * R200m are checked. Default is 2.0.
    n_threads : int, optional
        Number of threads for parallel processing. If None, uses half of
        available CPU cores, capped by the number of mini-boxes to process.

    Returns
    -------
    numpy.ndarray
        Particle data array with shape (n_particles, 3) where each row contains:
        - Column 0: r/R200m - Radial distance scaled by R200m
        - Column 1: vr/V200m - Radial velocity scaled by V200m  
        - Column 2: ln(v^2/V200m^2) - Natural log of velocity squared scaled by V200m^2

        All quantities are dimensionless and scaled by the overdensity properties
        computed individually for each seed.

    Raises
    ------
    TypeError
        If n_seeds is not an integer, or if array inputs in seed_data cannot
        be converted to numpy arrays.
    ValueError
        If n_seeds is non-positive, if array shapes in seed_data are incompatible,
        or if any scalar parameters are non-positive.
    FileNotFoundError
        If save_path doesn't exist or required mini-box files are missing.
    RuntimeError
        If parallel processing fails and fallback to sequential mode is needed,
        or if particle loading encounters errors.

    See Also
    --------
    _find_isolated_seeds : Identifies seeds meeting isolation criterion
    _group_seeds_by_minibox : Groups seeds for efficient parallel processing
    _get_candidate_seed_particle_data : Processes particles around seeds in one mini-box

    Notes
    -----
    - Seeds are ranked by mass, with the most massive considered first
    - Isolation criterion ensures selected seeds dominate their local environment
    - Uses periodic boundary conditions for all distance calculations
    - Automatically falls back to sequential processing if parallelization fails
    - Memory usage scales with the number of particles within r_max of all seeds
    - Progress bars show processing status for mini-boxes

    Examples
    --------
    >>> # Process 100 most massive isolated seeds
    >>> positions = numpy.random.uniform(0, 100, (10000, 3))
    >>> velocities = numpy.random.normal(0, 200, (10000, 3)) 
    >>> masses = numpy.random.lognormal(30, 1, 10000)
    >>> radii = (masses / 1e12) ** (1/3) * 0.5
    >>> seed_data = (positions, velocities, masses, radii)
    >>> 
    >>> results = _select_candidate_seeds(
    ...     n_seeds=100,
    ...     seed_data=seed_data,
    ...     r_max=5.0,
    ...     boxsize=100.0,
    ...     minisize=10.0, 
    ...     save_path="/data/mini-boxes/",
    ...     particle_mass=1e10,
    ...     mass_density=2.78e11,
    ...     n_threads=8
    ... )
    >>> print(f"Processed {len(results)} particles from isolated seeds")
    """
    # Validate inputs
    _validate_positive_number_non_zero(n_seeds, 'n_seeds')
    _validate_seed_data(seed_data)
    _validate_positive_number_non_zero(r_max, 'r_max')
    _validate_existing_path(load_path)
    _validate_boxsize_minisize(boxsize, minisize)
    _validate_positive_number_non_zero(mass_density, 'mass_density')
    if n_threads is not None:
        _validate_positive_number_non_zero(n_threads, 'n_threads')

    # Unpack seed data
    position, velocity, mass, radius = seed_data

    # Trim values outside boxsize due to floating point precision
    position = numpy.mod(position, boxsize)

    # Rank order by mass.
    order = numpy.argsort(-mass)
    position = position[order]
    velocity = velocity[order]
    radius = radius[order]
    mass = mass[order]

    # Search for eligible (isolated) seeds.
    isolated_indices = _find_isolated_seeds(position, mass, radius, n_seeds,
                                            boxsize, isolation_factor,
                                            isolation_radius_factor)

    # Select only the candidate seeds.
    position = position[isolated_indices]
    velocity = velocity[isolated_indices]

    # Group seeds by mini-box for efficient processing
    minibox_groups = _group_seeds_by_minibox(
        position, velocity, boxsize, minisize)

    # Cap the number of threads to the total number of mini-boxes to process
    n_miniboxes = len(minibox_groups)
    if n_threads is None:
        n_threads = min(max(1, os.cpu_count()//2), n_miniboxes)
    else:
        n_threads = min(n_threads, n_miniboxes)
    print(f"Processing {n_miniboxes} mini-boxes using {n_threads} threads...")

    # Set up multiprocessing
    processing_args = [
        (minibox_id, position_group, velocity_group, r_max, boxsize, minisize,
         load_path, particle_type, mass_density, redshift)
        for minibox_id, (position_group, velocity_group) in minibox_groups.items()
    ]

    results = []
    # Safely handle multiprocessing falure with a fall back to a single thread.
    if n_threads > 1 and n_miniboxes > 1:
        try:
            with Pool(n_threads) as pool, tqdm(total=n_miniboxes, colour="blue",
                                               desc='Processing candidates',
                                               ncols=100) as pbar:
                # imap keeps the input order (same output as starmap) but
                # yields results as they finish, so the progress bar updates.
                for result in pool.imap(_get_candidate_seed_particle_data_star,
                                        processing_args):
                    results.append(result)
                    pbar.update()
        except Exception as e:
            print(
                f"Warning: Parallel processing failed ({e}), falling back to sequential")
            # Fall back to sequential processing
            n_threads = 1

    if n_threads == 1:
        for args in tqdm(processing_args, desc='Processing mini-boxes',
                         colour='blue', ncols=100):
            result = _get_candidate_seed_particle_data(*args)
            results.append(result)

    results = numpy.concatenate(results, axis=1).T

    return results


def _diagnostic_calibration_data_plot(
    save_path: str,
    radius: numpy.ndarray,
    radial_velocity: numpy.ndarray,
    log_velocity_squared: numpy.ndarray,
) -> None:
    """Generate diagnostic plots for calibration data visualization.

    This function creates a two-panel diagnostic plot showing the distribution
    of particle data in the (r/R200m, ln(v^2/V200m^2)) space, separated by the
    sign of the radial velocity. The plots help visualize the phase space
    distribution of particles around halos for calibration purposes.

    Parameters
    ----------
    save_path : str
        Directory path where the diagnostic plot will be saved. Must be a valid
        directory path with write permissions. The plot is saved as 
        'calibration_data.png'.
    radius : numpy.ndarray
        Array of radial distances scaled by R200m with shape (n_particles,).
        Values should typically be positive and in the range [0, 3].
    radial_velocity : numpy.ndarray
        Array of radial velocities scaled by V200m with shape (n_particles,).
        Values can be positive (outflow) or negative (inflow).
    log_velocity_squared : numpy.ndarray
        Array of natural logarithm of velocity squared scaled by V200m^2 with
        shape (n_particles,). Values typically range from -3 to +3.

    Returns
    -------
    None
        Function saves the plot to disk but returns nothing.

    Raises
    ------
    ValueError
        If array shapes are incompatible or if save_path is not a valid string.
    OSError
        If save_path directory doesn't exist or lacks write permissions.

    See Also
    --------
    matplotlib.pyplot.hist2d : Creates 2D histogram plots
    matplotlib.colorbar.ColorbarBase : Creates standalone colorbars
    _hist2d_mesh : Computes 2D histogram and returns x, y, z as meshgrids
    _smooth_2d_hist : Applies Gaussian smoothing

    Notes
    -----
    - Creates two side-by-side 2D histograms for vr > 0 and vr < 0
    - Uses 200x200 bins for high-resolution visualization
    - Plot ranges are fixed: r/R200m ∈ [0,2], ln(v^2/V200m^2) ∈ [-2,2.5]
    - Uses 'terrain' colormap for particle density visualization
    - Includes LaTeX formatting for axis labels and titles
    - Colorbar shows relative particle counts but without tick labels
    - Figure is saved at 300 DPI for publication quality

    Examples
    --------
    >>> # Generate diagnostic plot for calibration data
    >>> radii = numpy.random.uniform(0.1, 2.0, 50000) 
    >>> vr = numpy.random.normal(0, 1, 50000)
    >>> log_v2 = numpy.random.normal(0, 1.5, 50000)
    >>> _diagnostic_calibration_data_plot(
    ...     save_path="/output/plots/",
    ...     radius=radii,
    ...     radial_velocity=vr, 
    ...     log_velocity_squared=log_v2
    ... )
    >>> # Plot saved to /output/plots/calibration_data.png
    """
    pyplot.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "figure.dpi": 150,
    })

    cmap = 'terrain'
    limits = ((0, 2), (-2, 2.5))
    levels = 80

    mask_negative_vr = (radial_velocity < 0)
    mask_positive_vr = ~mask_negative_vr

    # Compute counts histograms for top two panels
    x_mesh_positive, y_mesh_positive, z_positive = \
        _hist2d_mesh(radius[mask_positive_vr],
                     log_velocity_squared[mask_positive_vr],
                     limits=limits,
                     n_bins=200,
                     gradient=False,
                     )
    z_positive = _smooth_2d_hist(z_positive)

    x_mesh_negative, y_mesh_negative, z_negative = \
        _hist2d_mesh(radius[mask_negative_vr],
                     log_velocity_squared[mask_negative_vr],
                     limits=limits,
                     n_bins=200,
                     gradient=False,
                     )
    z_negative = _smooth_2d_hist(z_negative)

    fig, axes = pyplot.subplots(1, 2, figsize=(8, 4.2))
    fig.suptitle('Calibration data', fontsize=SIZE_LABELS)
    axes = axes.flatten()

    for ax in axes:
        ax.set_xlabel(r'$r/R_{\rm 200m}$', fontsize=SIZE_LABELS)
        ax.set_ylabel(r'$\ln(a^2v^2/v_{\rm 200m}^2)$', fontsize=SIZE_LABELS)
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
        ax.tick_params(axis="both", which="major", labelsize=SIZE_TICKS)

    # Create a color bar to display mass ranges.
    cax = fig.add_axes([1.01, 0.2, 0.01, 0.7])
    cbar = ColorbarBase(cax, cmap=cmap, orientation="vertical", extend='max')
    cbar.set_label(r'Counts (a.u.)', fontsize=SIZE_LEGEND)
    cbar.set_ticklabels([], fontsize=SIZE_TICKS)
    cbar.ax.tick_params(size=0, labelleft=False, labelright=False,
                        labeltop=False, labelbottom=False)

    pyplot.sca(axes[0])
    pyplot.title(r'$v_r > 0$', fontsize=SIZE_LABELS)
    pyplot.contourf(x_mesh_positive, y_mesh_positive, z_positive.T,
                    levels=levels, cmap=cmap)

    pyplot.sca(axes[1])
    pyplot.title(r'$v_r < 0$', fontsize=SIZE_LABELS)
    pyplot.contourf(x_mesh_negative, y_mesh_negative, z_negative.T,
                    levels=levels, cmap=cmap)

    pyplot.tight_layout()
    pyplot.savefig(save_path + 'calibration_data.png', dpi=300,
                   bbox_inches='tight')

    return None


def get_calibration_data(
    n_seeds: int,
    seed_data: Tuple[numpy.ndarray],
    r_max: float,
    boxsize: float,
    minisize: float,
    save_path: str,
    particle_type: str,
    mass_density: float,
    redshift: float,
    isolation_factor: float = 0.2,
    isolation_radius_factor: float = 2.0,
    n_threads: Optional[int] = None,
    diagnostics: bool = True,
    overwrite: bool = False,
) -> Tuple[numpy.ndarray]:
    """
    Generate or load calibration data from isolated massive seed halos.

    This function either loads pre-computed calibration data from an HDF5 file
    or generates new calibration data by processing particles around isolated
    massive seeds. The calibration data consists of the scaled quantities: 
    radial distance, radial velocity, velocity magnitude.

    The function implements a caching mechanism: if 'calibration_data.hdf5' exists
    in save_path, the data is loaded from disk. Otherwise, new data is computed
    and saved for future use.

    Parameters
    ----------
    n_seeds : int
        Maximum number of seeds to process for calibration. Must be positive.
        Function selects up to this many of the most massive isolated seeds.
    seed_data : Tuple[numpy.ndarray]
        Tuple containing (position, velocity, mass, radius) arrays:
        - position: shape (n_total_seeds, 3) - seed coordinates
        - velocity: shape (n_total_seeds, 3) - seed velocities  
        - mass: shape (n_total_seeds,) - seed masses (M200m)
        - radius: shape (n_total_seeds,) - seed overdensity radii (R200m)
    r_max : float
        Maximum distance from seed centers to consider particles. Must be 
        positive, typically 2*R200m.
    boxsize : float
        Size of the cubic simulation box. Must be a positive.
    minisize : float  
        Size of each cubic mini-box. Must be positive and ≤ boxsize.
    save_path : str
        Directory path for reading/writing calibration data and diagnostic plots.
        Must be a valid directory path with read/write permissions.
    particle_type : str
        Particle type identifier within each HDF5 file. Valid names are 'seed', 
        'dm', 'gas' and 'star'.
    particle_mass : float
        Mass of each simulation particle. Must be positive.
    mass_density : float
        Mean matter density of the universe. Must be positive.
    redshift : float
        Cosmological redshift of the universe in simulation. Must be zero or 
        positive. Scales kinetic energies as ln(a^2v^2/V200m^2).
    isolation_factor : float, optional
        Maximum allowed mass ratio for isolation criterion. Neighbors must have
        mass < isolation_factor * seed_mass to satisfy isolation criterion.
        Default is 0.2 (20%).
    isolation_radius_factor : float, optional
        Factor multiplying R200m to define isolation radius. Neighbors within
        isolation_radius_factor * R200m are checked. Default is 2.0.
    n_threads : int, optional
        Number of threads for parallel processing. If None, uses half of
        available CPU cores. Only used when generating new data, capped by the 
        number of mini-boxes to process.
    diagnostics : bool, optional
        Whether to generate diagnostic plots of the calibration data.
        Default is True.
    overwrite : bool, optional
        Whether to overwrite the existing calibration data. Default is False.

    Returns
    -------
    Tuple[numpy.ndarray]
        Tuple containing three arrays with calibration data:
        - radius: numpy.ndarray with shape (n_particles,)
            Radial distances scaled by R200m (dimensionless)
        - radial_velocity: numpy.ndarray with shape (n_particles,) 
            Radial velocities scaled by V200m (dimensionless)
        - log_velocity_squared: numpy.ndarray with shape (n_particles,)
            Natural log of velocity squared scaled by V200m^2 (dimensionless)

    Raises
    ------
    TypeError
        If n_seeds is not an integer, or if array inputs in seed_data cannot
        be converted to numpy arrays.
    ValueError
        If n_seeds is non-positive, if array shapes in seed_data are incompatible,
        or if any scalar parameters are non-positive.
    FileNotFoundError
        If save_path doesn't exist, or if required mini-box files are missing
        when generating new data.
    OSError
        If HDF5 file cannot be read/written, or if directory permissions
        are insufficient.
        
    See Also
    --------
    _select_candidate_seeds : Core function for generating calibration data
    _diagnostic_calibration_data_plot : Creates diagnostic visualizations
    h5py.File : HDF5 file interface for data persistence

    Notes
    -----
    - Implements caching via HDF5 file to avoid recomputation
    - Automatically generates diagnostic plots when diagnostics=True
    - Uses isolation criterion to ensure seeds dominate their environment
    - All returned quantities are dimensionless and scaled by overdensity properties
    - HDF5 file structure: 'r', 'vr', 'lnv2' datasets
    - Diagnostic plot saved as 'calibration_data.png' in save_path

    Examples
    --------
    >>> # Generate calibration data from 500 massive isolated seeds
    >>> positions = numpy.random.uniform(0, 100, (50000, 3))
    >>> velocities = numpy.random.normal(0, 200, (50000, 3))
    >>> masses = numpy.random.lognormal(30, 1, 50000) 
    >>> radii = (masses / 1e12) ** (1/3) * 0.5
    >>> seed_data = (positions, velocities, masses, radii)
    >>> 
    >>> r, vr, lnv2 = get_calibration_data(
    ...     n_seeds=500,
    ...     seed_data=seed_data,
    ...     r_max=4.0,
    ...     boxsize=100.0,
    ...     minisize=10.0,
    ...     save_path="/data/calibration/",
    ...     particle_mass=1e10,
    ...     mass_density=2.78e11,
    ...     redshift=0,
    ...     n_threads=16,
    ...     diagnostics=True
    ... )
    >>> print(f"Calibration data: {len(r)} particles")
    >>> print(f"Radius range: [{r.min():.2f}, {r.max():.2f}]")

    >>> # Load existing calibration data (fast)
    >>> r, vr, lnv2 = get_calibration_data(
    ...     n_seeds=500,  # Parameters don't matter for loading
    ...     seed_data=seed_data,
    ...     r_max=4.0,
    ...     boxsize=100.0,
    ...     minisize=10.0,
    ...     save_path="/data/calibration/",  # Must contain calibration_data.hdf5
    ...     particle_mass=1e10,
    ...     mass_density=2.78e11,
    ...     redshift=0,
    ...     diagnostics=False  # Skip plot generation
    ... )
    """
    file_name = save_path + 'calibration_data.hdf5'
    _validate_positive_number(redshift, 'redshift')
    
    # Flag to trigger except clause given the overwrite flag.
    execute_fallback = overwrite

    if not execute_fallback:
        try:
            with h5py.File(file_name, 'r') as hdf:
                radius = hdf['r'][()]
                radial_velocity = hdf['vr'][()]
                log_velocity_squared = hdf['lnv2'][()]
        except:
            execute_fallback = True
    
    if execute_fallback:
        results = _select_candidate_seeds(
            n_seeds=n_seeds,
            seed_data=seed_data,
            r_max=r_max,
            boxsize=boxsize,
            minisize=minisize,
            load_path=save_path,
            particle_type=particle_type,
            mass_density=mass_density,
            redshift = redshift,
            isolation_factor=isolation_factor,
            isolation_radius_factor=isolation_radius_factor,
            n_threads=n_threads,
        )

        radius = results[:, 0]
        radial_velocity = results[:, 1]
        log_velocity_squared = results[:, 2]

        # Introducing a factor of a from converting v200^2 (more specifically R200) from comoving to physical
        if redshift > 0.:
            a = 1 / (1 + redshift)
            log_velocity_squared += numpy.log(a)

        with h5py.File(file_name, 'w') as hdf:
            hdf.create_dataset('r', data=radius)
            hdf.create_dataset('vr', data=radial_velocity)
            hdf.create_dataset('lnv2', data=log_velocity_squared)

    if diagnostics:
        _diagnostic_calibration_data_plot(
            save_path=save_path,
            radius=radius,
            radial_velocity=radial_velocity,
            log_velocity_squared=log_velocity_squared
        )

    return radius, radial_velocity, log_velocity_squared


def _cost_percentile(abscissa: float, *data) -> float:
    """
    Compute cost for fitting line abscissa to achieve target percentile.

    This cost function optimizes the y-intercept (at a pivot radius) of a line
    with fixed slope such that a specified percentile of data points falls below
    the line. The cost is minimized when the fraction of points below the line
    matches the target percentile. 

    Parameters
    ----------
    abscissa : float
        Y-intercept of the line at the pivot radius (radius_pivot), i.e. the
        value of ln(v^2/v_200m^2) at the pivot radius. Can be positive or negative.
    *data : tuple
        Variable-length argument tuple containing exactly 5 elements:
        - radius : numpy.ndarray with shape (n_particles,)
            Scaled radial distances (r/R200m) for all particles.
        - log_velocity_squared : numpy.ndarray with shape (n_particles,)
            Natural log of velocity squared scaled by V200m^2 (ln(v^2/V200m^2)).
        - slope : float
            Fixed slope of the line in (ln(v^2), r/R200m) space. Can be
            positive, negative, or zero.
        - target : float
            Target percentile as a fraction in range [0, 1]. For example,
            0.9 means 90% of points should be below the line.
        - radius_pivot : float
            Pivot radius where the line has value abscissa. Typically chosen 
            at 0.5*R200m for numerical stability.

    Returns
    -------
    cost : float
        Logarithmic cost function value. Returns ln((target - actual_fraction)^2).
        Minimum occurs when actual fraction equals target. Always finite for
        target in (0, 1).

    Raises
    ------
    ValueError
        If data tuple doesn't contain exactly 5 elements, or if arrays have
        incompatible shapes.
    TypeError
        If inputs cannot be converted to appropriate numeric types.

    See Also
    --------
    scipy.optimize.minimize : Optimization routine for this cost function
    _cost_perpendicular_distance_abscissa : Alternative cost based on distances

    Notes
    -----
    - The line equation is: y = slope * (radius - radius_pivot) + abscissa
    - Cost function uses logarithm to handle small differences robustly
    - The function is designed for use with scipy.optimize.minimize
    - When target = 0.5, finds the median line through the data

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.optimize import minimize
    >>> 
    >>> # Generate sample data
    >>> radius = np.random.uniform(0.1, 2.0, 1000)
    >>> log_v2 = 0.5 * radius + np.random.normal(0, 0.3, 1000)
    >>> 
    >>> # Find intercept for 90th percentile line
    >>> data = (radius, log_v2, 0.5, 0.9, 1.0)
    >>> result = minimize(_cost_percentile, args=data, method='brent')
    >>> optimal_b = result.x
    >>> print(f"Optimal intercept: {optimal_b:.3f}")

    >>> # Verify that ~90% of points are below the line
    >>> line = 0.5 * (radius - 1.0) + optimal_b
    >>> fraction_below = (log_v2 < line).sum() / len(radius)
    >>> print(f"Actual fraction below: {fraction_below:.2%}")
    """
    radius, log_velocity_squared, slope, target, radius_pivot = data
    line = slope * (radius - radius_pivot) + abscissa

    # Total number of elements below the line
    below_line = (log_velocity_squared < line).sum()

    # Fraction of elements below the line
    fraction = target - below_line / radius.shape[0]

    # Cost to minimize
    cost = numpy.log(fraction ** 2)
    return cost


def _cost_perpendicular_distance_abscissa(abscissa: float, *data) -> float:
    """
    Compute cost for fitting line abscissa to maximize perpendicular distances.

    This cost function optimizes the y-intercept (at a pivot radius) of a line
    with fixed slope to maximize the mean squared perpendicular distance of
    points within a specified band around the line.

    Parameters
    ----------
    abscissa : float
        Y-intercept of the line at the pivot radius (radius_pivot), i.e. the
        value of ln(v^2/v_200m^2) at the pivot radius.
    *data : tuple
        Variable-length argument tuple containing exactly 5 elements:
        - radius : numpy.ndarray with shape (n_particles,)
            Scaled radial distances (r/R200m) for all particles.
        - log_velocity_squared : numpy.ndarray with shape (n_particles,)
            Natural log of velocity squared scaled by V200m^2 (ln(v^2/V200m^2)).
        - slope : float
            Fixed slope of the line in (ln(v^2), r/R200m) space. Can be
            positive, negative, or zero.
        - width : float
            Half-width of band around the line in same units as 
            log_velocity_squared. Only points within this perpendicular
            distance contribute to the cost. Must be positive.
        - radius_pivot : float
            Pivot radius where the line has value abscissa. Typically chosen 
            at 0.5*R200m for numerical stability.

    Returns
    -------
    cost : float
        Negative logarithm of mean squared perpendicular distance for points
        within the band. Returns -ln(mean(distance^2)). More negative values
        indicate better fits (larger mean distances to maximize).

    Raises
    ------
    ValueError
        If data tuple doesn't contain exactly 5 elements, if arrays have
        incompatible shapes, or if no points fall within the band.
    TypeError
        If inputs cannot be converted to appropriate numeric types.

    See Also
    --------
    scipy.optimize.minimize : Optimization routine for this cost function
    _cost_perpendicular_distance_slope : Optimize slope with fixed intercept
    _cost_percentile : Alternative cost based on percentiles

    Notes
    -----
    - Line equation: y = slope * (radius - radius_pivot) + abscissa
    - Perpendicular distance: d = |y - line| / \sqrt(1 + slope^2)
    - Only points with d < width contribute to cost
    - Cost is negated because we maximize distance but minimize cost

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.optimize import minimize
    >>> 
    >>> # Generate data with a clear ridge
    >>> radius = np.random.uniform(0.1, 2.0, 2000)
    >>> log_v2 = 0.3 * radius + 0.5 + np.random.normal(0, 0.2, 2000)
    >>> 
    >>> # Find intercept that maximizes distances within band
    >>> data = (radius, log_v2, 0.3, 0.5, 1.0)
    >>> result = minimize(_cost_perpendicular_distance_abscissa, 
    ...                          args=data, method='brent')
    >>> optimal_b = result.x
    >>> print(f"Optimal intercept: {optimal_b:.3f}")

    >>> # Visualize the fitted line
    >>> import matplotlib.pyplot as plt
    >>> line = 0.3 * (radius - 1.0) + optimal_b
    >>> plt.scatter(radius, log_v2, alpha=0.3, s=1)
    >>> plt.plot(radius, line, 'r-', linewidth=2)
    >>> plt.xlabel('r/R200m')
    >>> plt.ylabel('ln(v^2/V200m^2)')
    >>> plt.show()
    """
    radius, log_velocity_squared, slope, width, radius_pivot = data
    line = slope * (radius - radius_pivot) + abscissa

    # Perpendicular distance to the line
    distance = numpy.abs(log_velocity_squared - line) / \
        numpy.sqrt(1 + slope**2)

    # Select only elements within the width of the band
    distance_within_band = distance[(distance < width)]

    # Cost to maximize (thus negative)
    cost = -numpy.log(numpy.mean(distance_within_band ** 2))
    return cost
    

def _diagnostic_gradient_minima_plot(
    save_path:  str,
    log_velocity_squared_bins: numpy.ndarray,
    counts: numpy.ndarray,
    counts_gradient: numpy.ndarray,
    counts_gradient_smooth: numpy.ndarray,
    radial_bins: numpy.ndarray,
    counts_gradient_minima: numpy.ndarray,
    diagnostics_title: str,
) -> None:
    """
    Generate diagnostic plots for ln(v^2) gradient minimum in radial bins.

    This function creates an (n, m)-panel plot showing the distribution of 
    ln(v^2/V200m^2), its gradient and a smoothed gradient in radial bins 
    (radial_bins). The total number of panels is n_panels = n*m and are aranged
    in a rectangular plot. The plots help visualize the gradient minimum finding
    algorithm.

    Parameters
    ----------
    save_path : str
        Directory path where the diagnostic plot will be saved. Must be a valid
        directory path with write permissions. The plot is saved as 
        'calibration_gradient_minima_{diagnostics_title}.png'.
    log_velocity_squared_bins : numpy.ndarray
        Array of natural logarithm of velocity squared scaled by V200m^2 with
        shape (n_panels, n_bins), where n_bins is the number of bins in the 
        histogram.
    counts : numpy.ndarray
        Array of ln(v^2/V200m^2) histogram counts with shape (n_panels, n_bins).
        Counts are normalized to the absolute maximum value.
    counts_gradient : numpy.ndarray
        Array of the gradient of ln(v^2/V200m^2) histogram counts with shape 
        (n_panels, n_bins). Counts are normalized to the absolute maximum value.
    counts_gradient_smooth : numpy.ndarray
        Array of the smooth gradient of ln(v^2/V200m^2) histogram counts with 
        shape (n_panels, n_bins). Counts are normalized to the absolute maximum 
        value.
    radial_bins : numpy.ndarray
        Array of radial bins scaled by R200m with shape (n_panels).
    counts_gradient_minima : numpy.ndarray
        Minimum of the smooth gradient at each radial bin with shape (n_panels).
    diagnostics_title : str
        Plot title. First word is appended as description in the name of the 
        saved file.

    Returns
    -------
    None
        Function saves the plot to disk but returns nothing.

    Raises
    ------
    ValueError
        If array shapes are incompatible or if save_path is not a valid string.
    OSError
        If save_path directory doesn't exist or lacks write permissions.

    See Also
    --------
    matplotlib.pyplot.hist2d : Creates 2D histogram plots
    matplotlib.colorbar.ColorbarBase : Creates standalone colorbars
    _hist2d_mesh : Computes 2D histogram and returns x, y, z as meshgrids
    _smooth_2d_hist : Applies Gaussian smoothing

    Notes
    -----
    - Creates a (n, m) grid of ln(v^2/V200m^2) histograms in radial bins from
        previously computed data
    - Plot ranges are fixed: ln(v^2/V200m^2) ∈ [-2,2.5]
    - All counts are normalized to the absolute maximum value so they all peak
        at 1 or -1
    - The number of rows is calculated as: floor(\sqrt(n_panels))
    - The number of columns is calculated as: ceil(\sqrt(n_panels))
    - Includes LaTeX formatting for axis labels and titles
    - Figure is saved at 300 DPI for publication quality
    """


    pyplot.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "figure.dpi": 150,
    })

    limits = ((-2, 2.5), (-1.1, 1.1))

    # Figure out the number of panels
    n_panels = log_velocity_squared_bins.shape[0]
    n_rows = int(numpy.floor(numpy.sqrt(n_panels)))
    n_cols = int(numpy.ceil(numpy.sqrt(n_panels)))

    fig, axes = pyplot.subplots(n_rows, n_cols, sharex=True, sharey=True,
                                figsize=(2.5*n_cols, 2.0*n_rows),
                                gridspec_kw={'hspace': 0, 'wspace': 0})
    fig.suptitle(diagnostics_title, fontsize=SIZE_LABELS)

    for i, ax in enumerate(axes.flatten()):
        ax.text(-1.8, 0.8, r'$r/R_{\rm 200m} = %.2f$' % radial_bins[i])
        if i % n_cols == 0:
            ax.set_ylabel('Counts (a.u.)', fontsize=SIZE_LABELS)
        if i >= n_panels - n_cols:
            ax.set_xlabel(r'$\ln(a^2v^2/v_{\rm 200m}^2)$', fontsize=SIZE_LABELS)
        ax.tick_params(axis="both", which="major", labelsize=SIZE_TICKS)
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])

        ax.plot(log_velocity_squared_bins[i, :], counts[i, :],
                color=COLOR_GRAY, lw=1.0, label='Counts')
        ax.plot(log_velocity_squared_bins[i, :], counts_gradient[i,
                :], color=COLOR_BLUE, lw=1.0, label='Gradient')
        ax.plot(log_velocity_squared_bins[i, :], counts_gradient_smooth[i,
                :], color=COLOR_RED, lw=1.0, label='Smoothed gradient')
        ax.axvline(
            counts_gradient_minima[i], color='k', lw=1.5, ls='--', label='Minimum')

        if i == 0:
            ax.legend()

    pyplot.tight_layout()
    plot_suffix = diagnostics_title.split()[0]
    pyplot.savefig(save_path + f'calibration_gradient_minima_{plot_suffix.lower()}.png',
                   dpi=300, bbox_inches='tight')

    return None


def _gradient_minima(
    radius: numpy.ndarray,
    log_velocity_squared: numpy.ndarray,
    n_radial_bins: int,
    r_min: float,
    r_max: float,
    save_path: str,
    sigma_smooth: float = 1.5,
    lnvsq_lims: tuple = (-2.0, 2.5),
    diagnostics: bool = True,
    diagnostics_title: Optional[str] = None,
) -> Tuple[numpy.ndarray]:
    """
    Find gradient minima in KE-radius phase space to identify boundaries.

    This function find the minimum of the ln(v^2/V200m^2) distribution as a 
    function of radius by computing histograms in radial bins, then finding the 
    minimum of the gradient of ln(v^2/V200m^2). These minima correspond to edges
    in the phase space distribution, useful for identifying distinct populations
    (e.g., splashback radius, infall region boundaries).

    The algorithm proceeds as follows for each radial bin:
    1. Histogram the log_velocity_squared values using adaptive bin width
    2. Compute the gradient of the histogram counts
    3. Apply Gaussian smoothing to the gradient
    4. Find the minimum of the smoothed gradient

    The number of histogram bins is automatically determined using Silverman's
    rule of thumb for optimal histogram bin width, based on the total number of 
    data points within the specifiec radial range [r_min, r_max].

    Parameters
    ----------
    radius : numpy.ndarray
        Scaled radial distances (r/R200m) with shape (n_particles,).
        Values should typically be in range [0, 3]. Must be non-negative.
    log_velocity_squared : numpy.ndarray
        Natural log of velocity squared scaled by V200m^2 with shape (n_particles,).
        Values ln(v^2/V200m^2) typically range from -3 to +3.
    n_radial_bins : int
        Number of radial bins to compute gradient minima in. Must be positive
        Typical values are 10-50. More points give finer radial resolution but 
        require more particles for stable statistics.
    r_min : float
        Minimum radius for analysis in units of R200m. Must be non-negative
        and less than r_max. Typical value is from 0.0 to 0.3.
    r_max : float
        Maximum radius for analysis in units of R200m. Must be positive and
        greater than r_min. Typical value is from 0.4 to 0.6.
    save_path : str
        Directory path where diagnostic plots will be saved. Must be a valid
        directory path with write permissions. Plot saved as
        'gradient_minima_{diagnostics_title}.png'.
    sigma_smooth : float, optional
        Standard deviation for Gaussian smoothing kernel applied to gradient.
        Must be positive and non-zero. Default is 1.5. Larger values produce
        smoother gradients but may obscure fine structure. Typical range is
        [1.0, 3.0].
    lnvsq_lims : tuple, optional
        Lower and upper bounds (min, max) for log_velocity_squared histogram
        range, as a tuple of two floats. Default is (-2.0, 2.5). This range
        should encompass all relevant velocity data. Used with adaptive bin
        width calculation to determine number of bins.
    diagnostics : bool, optional
        Whether to generate diagnostic plots showing histograms, gradients,
        and identified minima. Default is True.
    diagnostics_title : str, optional
        Additional identifier for diagnostic plot filename. If None, uses
        default filename 'gradient_minima.png'. If provided, saves as
        'gradient_minima_{diagnostics_title}.png'. Default is None.

    Returns
    -------
    radial_bins : numpy.ndarray
        Center positions of radial bins with shape (n_radial_bins,). Values are
        in units of R200m and span [r_min, r_max].
    counts_gradient_minima : numpy.ndarray
        Kinetic energy values at gradient minima for each radial bin with shape
        (n_radial_bins,). Values are ln(v^2/V200m^2) and identify boundaries in
        velocity space as a function of radius.

    Raises
    ------
    ValueError
        If array shapes are incompatible, if r_min >= r_max, if n_radial_bins is
        non-positive, if sigma_smooth is non-positive, or if
        radius or log_velocity_squared contain invalid values.
    TypeError
        If inputs cannot be converted to appropriate numeric types, or if
        lnvsq_lims is not a tuple or sequence.
    OSError
        If save_path doesn't exist or lacks write permissions when diagnostics=True.
    RuntimeWarning
        If radial bins contain very few particles, statistics may be unreliable.
        
    See Also
    --------
    _diagnostic_gradient_minima_plot : Creates visualization of results
    _smooth_2d_hist : Applies Gaussian smoothing
    self_calibration : Uses this function for phase space calibration
    matplotlib.pyplot.hist2d : Creates 2D histogram plots
    numpy.gradient : Gradient computation
    scipy.ndimage.gaussian_filter : Gaussian smoothing implementation
    scipy.stats.iqr : Interquartile range computation
    
    Notes
    -----
    **Automatic Bin Width Selection**:
    
    The number of velocity bins is automatically computed using Silverman's
    rule of thumb:
    
    h = 0.9 * min(\sigma, IQR/1.34) * N^(-1/5)
    
    where:
    - h is the optimal bin width
    - \sigma is the standard deviation of log_velocity_squared in [r_min, r_max]
    - IQR is the interquartile range (scaled to normal distribution)
    - N is the number of particles in [r_min, r_max]
    
    Number of bins = ceil((lnvsq_lims[1] - lnvsq_lims[0]) / h)
    
    This adaptive approach ensures appropriate resolution for different
    data distributions and sample sizes.
    
    **Algorithm Details**:
    
    - The gradient is normalized by its maximum absolute value in each bin
    - Only positive log_velocity_squared values are considered when finding minima
    - The smoothing kernel helps avoid spurious minima from statistical noise
    - Empty radial bins will have undefined gradient minima (may be NaN or zero)
    - The method is most reliable with >100 particles per radial bin
    - Diagnostic plots show: counts, gradient, smoothed gradient, and minima
    
    **Velocity Range Masking**:
    
    Before finding the minimum, the algorithm masks to only consider positive
    velocity values (log_velocity_squared > 0). This ensures the identified
    boundary corresponds to kinematically significant features rather than
    low-velocity artifacts.

    Examples
    --------
    >>> import numpy as np
    >>> 
    >>> # Generate sample phase space data
    >>> n_particles = 10000
    >>> radius = np.random.uniform(0.1, 2.5, n_particles)
    >>> # Create bimodal velocity distribution
    >>> log_v2 = np.where(radius < 1.0,
    ...                   np.random.normal(0.5, 0.3, n_particles),
    ...                   np.random.normal(-0.2, 0.4, n_particles))
    >>> 
    >>> # Find gradient minima with default parameters
    >>> r_bins, v_minima = _gradient_minima(
    ...     radius=radius,
    ...     log_velocity_squared=log_v2,
    ...     n_radial_bins=20,
    ...     r_min=0.2,
    ...     r_max=0.5,
    ...     save_path="/output/",
    ...     diagnostics=True
    ... )
    >>> 
    >>> # Plot the boundary
    >>> import matplotlib.pyplot as plt
    >>> plt.scatter(radius, log_v2, alpha=0.1, s=1)
    >>> plt.plot(r_bins, v_minima, 'r-', linewidth=2, label='Boundary')
    >>> plt.xlabel('r/R200m')
    >>> plt.ylabel('ln(v^2/V200m^2)')
    >>> plt.legend()
    >>> plt.show()

    >>> # Calibration for positive radial velocity
    >>> r_bins_pos, v_minima_pos = _gradient_minima(
    ...     radius=radius[radial_velocity > 0],
    ...     log_velocity_squared=log_v2[radial_velocity > 0],
    ...     n_radial_bins=20,
    ...     r_min=0.2,
    ...     r_max=0.5,
    ...     save_path="/output/calibration/",
    ...     sigma_smooth=1.5,
    ...     lnvsq_lims=(-2.0, 2.5),
    ...     diagnostics=True,
    ...     diagnostics_title='Positive vr slope calibration'
    ... )

    >>> # Calibration for negative radial velocity with custom parameters
    >>> r_bins_neg, v_minima_neg = _gradient_minima(
    ...     radius=radius[radial_velocity < 0],
    ...     log_velocity_squared=log_v2[radial_velocity < 0],
    ...     n_radial_bins=25,
    ...     r_min=0.2,
    ...     r_max=0.5,
    ...     save_path="/output/calibration/",
    ...     sigma_smooth=2.0,  # More smoothing
    ...     lnvsq_lims=(-3.0, 3.0),  # Wider velocity range
    ...     diagnostics=True,
    ...     diagnostics_title='Negative vr slope calibration'
    ... )

    >>> # Compare different smoothing scales
    >>> fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    >>> for ax, sigma in zip(axes, [1.0, 1.5, 2.5]):
    ...     r_b, v_m = _gradient_minima(
    ...         radius=radius,
    ...         log_velocity_squared=log_v2,
    ...         n_radial_bins=20,
    ...         r_min=0.2,
    ...         r_max=0.5,
    ...         save_path="/output/",
    ...         sigma_smooth=sigma,
    ...         diagnostics=False
    ...     )
    ...     ax.scatter(radius, log_v2, alpha=0.1, s=1)
    ...     ax.plot(r_b, v_m, 'r-', linewidth=2)
    ...     ax.set_title(f'sigma = {sigma}')
    ...     ax.set_xlabel('r/R200m')
    ...     ax.set_ylabel('ln(v^2/V200m^2)')
    >>> plt.tight_layout()
    >>> plt.show()
    """
    # Validate inputs
    _validate_positive_number_non_zero(n_radial_bins, 'n_radial_bins')
    _validate_positive_number(r_min, 'r_min')
    _validate_positive_number_non_zero(sigma_smooth, 'sigma_smooth')

    if r_min >= r_max:
        raise ValueError('r_max must be larger than r_min')

    mask = (radius >= r_min) & (radius <= r_max)
    iqr = interquartile_range(log_velocity_squared[mask], scale='normal')
    # Silvermann's rule of thumb.
    h = 0.9 * numpy.min([numpy.std(log_velocity_squared[mask]), iqr]) * \
        mask.sum()**(-0.2)
    n_bins = int(numpy.ceil((lnvsq_lims[1] - lnvsq_lims[0]) / h))

    # Compute radial bins and initialize empty container for gradient minima
    radius_edges = numpy.linspace(r_min, r_max, n_radial_bins + 1)
    counts_gradient_minima = numpy.zeros(n_radial_bins)

    # Initialize empty containers for diagnostic plot even if diagnostics = False
    log_velocity_squared_bins_out = numpy.zeros((n_radial_bins, n_bins))
    counts_out = numpy.zeros((n_radial_bins, n_bins))
    counts_gradient_out = numpy.zeros((n_radial_bins, n_bins))
    counts_gradient_smooth_out = numpy.zeros((n_radial_bins, n_bins))

    # Every radial bin lies inside [r_min, r_max], so only that subset needs to
    # be scanned for each bin (same selected values, in the same order).
    radius_in = radius[mask]
    log_velocity_squared_in = log_velocity_squared[mask]

    for i in range(n_radial_bins):
        # Create mask for current r bin
        radius_mask = (radius_in > radius_edges[i]) * \
            (radius_in < radius_edges[i + 1])

        # Compute histogram of lnv2 values within the r bin and the vr mask
        counts, log_velocity_squared_edges = numpy.histogram(
            log_velocity_squared_in[radius_mask], bins=numpy.linspace(*lnvsq_lims, n_bins+1))

        # Compute the gradient of the histogram
        counts_gradient = numpy.gradient(
            counts, numpy.mean(numpy.diff(log_velocity_squared_edges)))
        counts_gradient /= numpy.max(numpy.abs(counts_gradient))

        # Smooth the gradient
        counts_gradient_smooth = gaussian_filter(
            counts_gradient, sigma_smooth)

        # Find the lnv2 value corresponding to the minimum of the smoothed gradient
        log_velocity_squared_bins = 0.5 * (log_velocity_squared_edges[:-1] +
                                           log_velocity_squared_edges[1:])
        
        mask_log_velocity_squared = (0. < log_velocity_squared_bins)
        counts_gradient_minima[i] = log_velocity_squared_bins[mask_log_velocity_squared][numpy.argmin(
            counts_gradient_smooth[mask_log_velocity_squared])]

        # Store diagnostics for plotting only
        log_velocity_squared_bins_out[i, :] = log_velocity_squared_bins
        counts_out[i, :] = counts / numpy.max(counts)
        counts_gradient_out[i, :] = counts_gradient
        counts_gradient_smooth_out[i, :] = counts_gradient_smooth

    # Compute r bin centres
    radial_bins = 0.5 * (radius_edges[:-1] + radius_edges[1:])

    # Return diagnostics if requested
    if diagnostics:
        _diagnostic_gradient_minima_plot(
            save_path,
            log_velocity_squared_bins_out,
            counts_out,
            counts_gradient_out,
            counts_gradient_smooth_out,
            radial_bins,
            counts_gradient_minima,
            diagnostics_title,
        )
    return radial_bins, counts_gradient_minima


def _hist2d_mesh(
    x: numpy.ndarray,
    y: numpy.ndarray,
    limits: Tuple[tuple],
    n_bins: int,
    gradient: bool = False,
) -> Tuple[numpy.ndarray]:
    """
    Compute 2D histogram with optional gradient for phase space analysis.

    This function creates a 2D histogram of (x, y) data and optionally computes
    the gradient of the histogram density with respect to y in each x bin.
    Returns meshgrid arrays suitable for contour plots and gradient analysis.

    Parameters
    ----------
    x : numpy.ndarray
        X-coordinates of data points with shape (n_particles,). Typically
        represents radial distance (r/R200m)
    y : numpy.ndarray
        Y-coordinates of data points with shape (n_particles,). Typically
        represents kinetic energy quantities (ln(v^2/V200m^2)).
    limits : Tuple[tuple]
        2D histogram limits as ((x_min, x_max), (y_min, y_max)). Must be
        a tuple of two tuples, each containing two floats. Defines the
        rectangular region for histogram computation.
    n_bins : int
        Number of bins in each dimension. Must be positive. The total number
        of bins in the 2D histogram is n_bins^2. Typical values are 50-200.
    gradient : bool, optional
        Whether to compute gradient of density with respect to y in each x bin.
        If True, returns gradient array instead of density. Default is False.

    Returns
    -------
    x_mesh : numpy.ndarray
        X-coordinates of bin centers with shape (n_bins, n_bins). Created by
        numpy.meshgrid for use in contour/surface plots.
    y_mesh : numpy.ndarray
        Y-coordinates of bin centers with shape (n_bins, n_bins). Created by
        numpy.meshgrid for use in contour/surface plots.
    z : numpy.ndarray
        If gradient=False: 2D histogram density with shape (n_bins, n_bins).
            Values are normalized probability densities.
        If gradient=True: Gradient of density with respect to y with shape
            (n_bins, n_bins). Values are \partial\rho / \partial y for each x bin.

    Raises
    ------
    ValueError
        If x and y have different shapes, if limits tuple has wrong structure,
        or if n_bins is non-positive.
    TypeError
        If inputs cannot be converted to appropriate numeric types.

    See Also
    --------
    numpy.histogram2d : Underlying 2D histogram computation
    numpy.gradient : Gradient computation method
    numpy.meshgrid : Meshgrid creation for plotting
    _gradient_minima : Alternative method for finding boundaries

    Notes
    -----
    - Histogram is computed using numpy.histogram2d with density=True
    - Bin centers are computed as midpoints between edges
    - Empty bins will have zero density/gradient
    - Meshgrids follow numpy convention: first index is x, second is y
    - Gradient is computed using numpy.gradient along y-axis for each x bin
    - The gradient computation uses mean bin width for numerical derivative

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> 
    >>> # Generate sample phase space data
    >>> n_radial_bins = 5000
    >>> x = np.random.uniform(0, 2, n_radial_bins)
    >>> y = 0.5 * x + np.random.normal(0, 0.3, n_radial_bins)
    >>> 
    >>> # Compute 2D histogram
    >>> limits = ((0, 2), (-1, 2))
    >>> x_mesh, y_mesh, density = _hist2d_mesh(x, y, limits, n_bins=100)
    >>> 
    >>> # Plot density
    >>> plt.contourf(x_mesh, y_mesh, density.T, levels=20)
    >>> plt.colorbar(label='Density')
    >>> plt.xlabel('r/R200m')
    >>> plt.ylabel('ln(v^2/V200m^2)')
    >>> plt.show()

    >>> # Compute and plot gradient
    >>> x_mesh, y_mesh, grad = _hist2d_mesh(x, y, limits, n_bins=100, 
    ...                                     gradient=True)
    >>> plt.contourf(x_mesh, y_mesh, grad.T, levels=20, cmap='RdBu_r')
    >>> plt.colorbar(label='∂ρ/∂y')
    >>> plt.xlabel('r/R200m')
    >>> plt.ylabel('ln(v^2/V200m^2)')
    >>> plt.show()

    >>> # Find ridges in phase space (where gradient ≈ 0)
    >>> ridge_mask = np.abs(grad) < 0.1
    >>> plt.scatter(x_mesh[ridge_mask], y_mesh[ridge_mask], c='red', s=1)
    """
    _validate_positive_number_non_zero(n_bins, "n_bins")
    
    hist_z, hist_x, hist_y = numpy.histogram2d(x, y, bins=n_bins, range=limits,
                                               density=True)

    # Bin centres
    x = 0.5 * (hist_x[:-1] + hist_x[1:])
    y = 0.5 * (hist_y[:-1] + hist_y[1:])

    # Bin widths
    dy = numpy.mean(numpy.diff(y))

    # Meshgrids
    x_mesh, y_mesh = numpy.meshgrid(x, y)

    # Compute gradient of z as a function of y for each x bin
    if gradient:
        z_grad = numpy.zeros_like(hist_z)
        for i in range(n_bins):
            z_grad[i, :] = numpy.gradient(hist_z[i, :], dy)
        return x_mesh, y_mesh, z_grad
    else:
        return x_mesh, y_mesh, hist_z


def _smooth_2d_hist(arr: numpy.ndarray, sigma: float = 2.0) -> numpy.ndarray:
    """
    Apply Gaussian smoothing to 2D histogram for noise reduction.

    This function smooths a 2D array (typically a histogram or density map) using
    a Gaussian filter. Smoothing reduces statistical noise and makes underlying
    structures more visible, which is useful for feature detection and boundary
    identification in phase space distributions.

    Parameters
    ----------
    arr : numpy.ndarray
        2D array to smooth, typically with shape (n_bins, n_bins). Usually
        represents a histogram density or gradient field. Can contain any
        numeric values including zeros and NaNs (which will be smoothed).
    sigma : float, optional
        Standard deviation of the Gaussian kernel in bin units. Must be
        non-negative. Default is 2.0. Larger values produce stronger smoothing:
        - sigma < 1: Minimal smoothing, preserves fine structure
        - sigma = 1-3: Moderate smoothing, good for most applications
        - sigma > 5: Strong smoothing, may obscure real features

    Returns
    -------
    arr_smooth : numpy.ndarray
        Smoothed 2D array with same shape as input. Values are weighted averages
        of neighboring bins using Gaussian kernel. Edge effects are handled by
        scipy's boundary conditions (typically reflection).

    Raises
    ------
    ValueError
        If arr is not 2-dimensional or if sigma is negative.
    TypeError
        If arr cannot be converted to numpy array or sigma is not numeric.

    See Also
    --------
    scipy.ndimage.gaussian_filter : Underlying smoothing implementation
    _hist2d_mesh : Creates 2D histograms for smoothing
    scipy.ndimage.uniform_filter : Alternative box-car smoothing
    scipy.signal.wiener : Adaptive Wiener filtering for noise reduction

    Notes
    -----
    - Uses scipy.ndimage.gaussian_filter for efficient computation
    - The Gaussian kernel has standard deviation sigma in array index units
    - Effective smoothing radius is approximately 3*sigma bins
    - Smoothing is applied equally in both dimensions
    - Edge effects are handled by reflecting values across boundaries
    - NaN values are propagated through smoothing (consider masking first)

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> 
    >>> # Create noisy 2D histogram
    >>> np.random.seed(42)
    >>> x = np.random.uniform(0, 2, 5000)
    >>> y = 0.3 * x + np.random.normal(0, 0.2, 5000)
    >>> hist, x_edges, y_edges = np.histogram2d(x, y, bins=50, 
    ...                                         range=((0, 2), (-1, 1)))
    >>> 
    >>> # Add noise to simulate low statistics
    >>> hist_noisy = hist + np.random.poisson(2, hist.shape)
    >>> 
    >>> # Apply smoothing
    >>> hist_smooth = _smooth_2d_hist(hist_noisy, sigma=2.0)
    >>> 
    >>> # Compare original and smoothed
    >>> fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    >>> im1 = axes[0].imshow(hist_noisy.T, origin='lower', aspect='auto')
    >>> axes[0].set_title('Noisy histogram')
    >>> plt.colorbar(im1, ax=axes[0])
    >>> 
    >>> im2 = axes[1].imshow(hist_smooth.T, origin='lower', aspect='auto')
    >>> axes[1].set_title('Smoothed histogram')
    >>> plt.colorbar(im2, ax=axes[1])
    >>> plt.show()

    >>> # Effect of different sigma values
    >>> sigmas = [0.5, 1.0, 2.0, 5.0]
    >>> fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    >>> for ax, sigma in zip(axes, sigmas):
    ...     smoothed = _smooth_2d_hist(hist_noisy, sigma=sigma)
    ...     ax.imshow(smoothed.T, origin='lower', aspect='auto')
    ...     ax.set_title(f'sigma = {sigma}')
    >>> plt.tight_layout()
    >>> plt.show()

    >>> # Smoothing gradient fields
    >>> # Compute gradient of density
    >>> grad_y = np.gradient(hist, axis=1)
    >>> grad_y_smooth = _smooth_2d_hist(grad_y, sigma=1.5)
    >>> 
    >>> # Smoothed gradients are better for finding features
    >>> plt.imshow(grad_y_smooth.T, origin='lower', aspect='auto', cmap='RdBu_r')
    >>> plt.colorbar(label='Smoothed ∂ρ/∂y')
    >>> plt.title('Smoothed density gradient')
    >>> plt.show()
    """
    arr_smooth = gaussian_filter(arr, sigma=sigma)
    return arr_smooth


def _diagnostic_self_calibration_plot(
    save_path: str,
    radius: numpy.ndarray,
    radial_velocity: numpy.ndarray,
    log_velocity_squared: numpy.ndarray,
    parameters: Tuple[float],
    gradient_values: Tuple[numpy.ndarray],
) -> None:
    """
    Generate diagnostic plots showing calibrated classification boundaries.

    This function creates a 2x2 panel diagnostic plot visualizing the
    self-calibrated phase-space classification scheme. The top panels show
    particle distributions with fitted classification boundaries, while the
    bottom panels display gradient fields used to determine optimal cut lines.
    Separate visualizations are provided for positive and negative radial
    velocity regimes.

    Parameters
    ----------
    save_path : str
        Directory path where the diagnostic plot will be saved. Must be a valid
        directory path with write permissions. The plot is saved as
        'calibration_classification_lines.png'.
    radius : numpy.ndarray
        Array of radial distances scaled by R200m with shape (n_particles,).
        Values should typically be positive and in the range [0, 2].
    radial_velocity : numpy.ndarray
        Array of radial velocities scaled by V200m with shape (n_particles,).
        Values can be positive (outflow) or negative (inflow). Used to
        separate particles into two regimes.
    log_velocity_squared : numpy.ndarray
        Array of natural logarithm of velocity squared scaled by V200m^2 with
        shape (n_particles,). Values typically range from -2 to +2.5.
    parameters : Tuple[float]
        Calibrated classification parameters in the following order:
        (radius_pivot, slope_positive_vr, abscissa_positive_vr,
         slope_negative_vr, abscissa_negative_vr, alpha, beta, gamma).
        - radius_pivot: Transition radius between linear and quadratic regimes
        - slope_positive_vr, abscissa_positive_vr: Linear cut for vr > 0
        - slope_negative_vr, abscissa_negative_vr: Linear cut for vr < 0 at r > pivot
        - alpha, beta, gamma: Quadratic coefficients for vr < 0 at r < pivot
    gradient_values : Tuple[numpy.ndarray]
        Gradient analysis results in the following order:
        (radial_bins_positive, gradient_minimum_positive,
         radial_bins_negative, gradient_minimum_negative).
        Each array contains the radial bins and corresponding gradient minima
        used to determine classification boundaries.

    Returns
    -------
    None
        Function saves the plot to disk but returns nothing.

    Raises
    ------
    ValueError
        If array shapes are incompatible or if parameters tuple has incorrect length.
    OSError
        If save_path directory doesn't exist or lacks write permissions.

    See Also
    --------
    _diagnostic_calibration_data_plot : Visualizes raw calibration data without cuts.
    _hist2d_mesh : Computes 2D histogram and returns x, y, z as meshgrids.
    _smooth_2d_hist : Applies Gaussian smoothing to 2D histograms.

    Notes
    -----
    Figure structure (2x2 panels):
    - Top-left: Particle counts for vr > 0 with linear classification boundary
    - Top-right: Particle counts for vr < 0 with hybrid linear/quadratic boundary
    - Bottom-left: Gradient field for vr > 0 showing optimal cut determination
    - Bottom-right: Gradient field for vr < 0 showing optimal cut determination

    The classification boundary for vr < 0 transitions from quadratic to linear
    at the pivot radius to better capture the orbital/infalling transition at
    small radii.

    Plot specifications:
    - Uses 200x200 bins for 2D histograms
    - Plot ranges: r/R200m ∈ [0,2], ln(v^2/V200m^2) ∈ [-2,2.5]
    - Uses 'terrain' colormap with 80 contour levels
    - Includes LaTeX-formatted equations for classification boundaries
    - Saved at 300 DPI for publication quality
    - Colorbar shows relative counts without specific tick values

    The gradient panels show where the phase-space density gradient is steepest,
    which corresponds to the optimal location for classification boundaries
    separating orbiting from infalling populations.

    Examples
    --------
    >>> # Visualize calibrated classification scheme
    >>> radii = numpy.random.uniform(0.1, 2.0, 50000)
    >>> vr = numpy.random.normal(0, 1, 50000)
    >>> log_v2 = numpy.random.normal(0, 1.5, 50000)
    >>> params = (0.5, 1.5, 0.2, 1.0, 0.1, 2.0, 1.0, 0.5)
    >>> grad_vals = (
    ...     numpy.linspace(0, 2, 100),
    ...     numpy.random.uniform(-1, 1, 100),
    ...     numpy.linspace(0, 2, 100),
    ...     numpy.random.uniform(-1, 1, 100)
    ... )
    >>> _diagnostic_self_calibration_plot(
    ...     save_path="/output/plots/",
    ...     radius=radii,
    ...     radial_velocity=vr,
    ...     log_velocity_squared=log_v2,
    ...     parameters=params,
    ...     gradient_values=grad_vals
    ... )
    >>> # Plot saved to /output/plots/calibration_classification_lines.png
    """
    pyplot.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "figure.dpi": 150,
    })

    radius_pivot, slope_positive_vr, abscissa_positive_vr, slope_negative_vr, \
        abscissa_negative_vr, alpha, beta, gamma = parameters

    # Compute cut lines
    x = numpy.linspace(0, 2, 1000)
    line_positive_vr = slope_positive_vr * \
        (x - radius_pivot) + abscissa_positive_vr

    x_low = numpy.linspace(0, radius_pivot, 1000)
    x_high = numpy.linspace(radius_pivot, 2, 1000)
    line_negative_vr = slope_negative_vr * \
        (x - radius_pivot) + abscissa_negative_vr
    curve_low = alpha*x_low**2 + beta*x_low + gamma
    curve_high = slope_negative_vr * \
        (x_high - radius_pivot) + abscissa_negative_vr

    label_positive_vr = fr"${slope_positive_vr:.3f}(x-{radius_pivot:.2f}){abscissa_positive_vr:+.3f}$"
    label_negative_vr = fr"${slope_negative_vr:.3f}(x-{radius_pivot:.2f}){abscissa_negative_vr:+.3f}$"

    cmap = 'terrain'
    limits = ((0, 2), (-2, 2.5))
    levels = 80
    mask_negative_vr = (radial_velocity < 0)
    mask_positive_vr = ~mask_negative_vr

    radial_bins_positive, gradient_minumum_positive, radial_bins_negative, \
        gradient_minumum_negative = gradient_values

    fig, axes = pyplot.subplots(2, 2, figsize=(8, 8))
    fig.suptitle('Calibration data', fontsize=SIZE_LABELS)
    axes = axes.flatten()

    for ax in axes:
        ax.set_xlabel(r'$r/R_{\rm 200m}$', fontsize=SIZE_LABELS)
        ax.set_ylabel(r'$\ln(a^2v^2/v_{\rm 200m}^2)$', fontsize=SIZE_LABELS)
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
        ax.tick_params(axis="both", which="major", labelsize=SIZE_TICKS)

    # Create a color bar to display mass ranges.
    cax = fig.add_axes([1.01, 0.2, 0.01, 0.7])
    cbar = ColorbarBase(cax, cmap=cmap, orientation="vertical", extend='max')
    cbar.set_label(r'Counts (a.u.)', fontsize=SIZE_LEGEND)
    cbar.set_ticklabels([], fontsize=SIZE_TICKS)
    cbar.ax.tick_params(size=0, labelleft=False, labelright=False,
                        labeltop=False, labelbottom=False)

    # Compute counts histograms for top two panels
    x_mesh_positive, y_mesh_positive, z_positive = \
        _hist2d_mesh(radius[mask_positive_vr],
                     log_velocity_squared[mask_positive_vr],
                     limits=limits,
                     n_bins=200,
                     gradient=False,
                     )
    z_positive = _smooth_2d_hist(z_positive)

    x_mesh_negative, y_mesh_negative, z_negative = \
        _hist2d_mesh(radius[mask_negative_vr],
                     log_velocity_squared[mask_negative_vr],
                     limits=limits,
                     n_bins=200,
                     gradient=False,
                     )
    z_negative = _smooth_2d_hist(z_negative)

    # Top-left
    pyplot.sca(axes[0])
    pyplot.title(r'$v_r > 0$', fontsize=SIZE_LABELS)
    pyplot.contourf(x_mesh_positive, y_mesh_positive, z_positive.T,
                    levels=levels, cmap=cmap)
    pyplot.plot(x, line_positive_vr, lw=2.0,
                color="r", label=label_positive_vr)
    pyplot.legend(loc='lower left', fontsize=SIZE_LEGEND)

    # Top-right
    pyplot.sca(axes[1])
    pyplot.title(r'$v_r < 0$', fontsize=SIZE_LABELS)
    pyplot.contourf(x_mesh_negative, y_mesh_negative, z_negative.T,
                    levels=levels, cmap=cmap)
    pyplot.plot(x, line_negative_vr, lw=2.0,
                color="k", label=label_negative_vr)
    pyplot.plot(x_low, curve_low, lw=2.0, color='r', ls='--')
    pyplot.plot(x_high, curve_high, lw=2.0, color='r',
                ls='--', label='Low radius correction')
    pyplot.legend(loc='lower left', fontsize=SIZE_LEGEND)

    # Compute counts gradient for lower two panels
    x_mesh_positive, y_mesh_positive, z_grad_positive = \
        _hist2d_mesh(radius[mask_positive_vr],
                     log_velocity_squared[mask_positive_vr],
                     limits=limits,
                     n_bins=200,
                     gradient=True,
                     )
    z_grad_positive = _smooth_2d_hist(z_grad_positive)

    x_mesh_negative, y_mesh_negative, z_grad_negative = \
        _hist2d_mesh(radius[mask_negative_vr],
                     log_velocity_squared[mask_negative_vr],
                     limits=limits,
                     n_bins=200,
                     gradient=True,
                     )
    z_grad_negative = _smooth_2d_hist(z_grad_negative)

    # Bottom-left
    pyplot.sca(axes[2])
    pyplot.contourf(x_mesh_positive, y_mesh_positive, z_grad_positive.T,
                    levels=levels, cmap=cmap)
    pyplot.plot(radial_bins_positive,
                gradient_minumum_positive, lw=1.0, color="r")
    pyplot.plot(x, line_positive_vr, lw=2.0, color="r", label='Cut line')

    # Bottom-right
    pyplot.sca(axes[3])
    pyplot.contourf(x_mesh_negative, y_mesh_negative, z_grad_negative.T,
                    levels=levels, cmap=cmap)
    pyplot.plot(radial_bins_negative,
                gradient_minumum_negative, lw=1.0, color="r")
    pyplot.plot(x, line_negative_vr, lw=2.0, color="k", label='Cut line')
    pyplot.plot(x_low, curve_low, lw=2.0, color='r', ls='--')
    pyplot.plot(x_high, curve_high, lw=2.0, color='r',
                ls='--', label='Low radius correction')

    pyplot.tight_layout()
    pyplot.savefig(save_path + 'calibration_classification_lines.png', dpi=300,
                   bbox_inches='tight')

    return None


def self_calibration(
    n_seeds: int,
    seed_data: Tuple[numpy.ndarray],
    r_max: float,
    boxsize: float,
    minisize: float,
    save_path: str,
    particle_type: str,
    mass_density: float,
    redshift: float,
    n_radial_bins: int = 20,
    gradient_radial_lims: Tuple[float] = (0.2, 0.5),
    percent: float = 0.995,
    width: float = 0.05,
    radius_pivot: float = 0.5,
    isolation_factor: float = 0.2,
    isolation_radius_factor: float = 2.0,
    n_threads: Optional[int] = None,
    diagnostics: bool = True,
    overwrite: bool = False,
) -> None:
    """
    Runs calibration from isolated halo samples.

    This function calibrates boundary lines in kinetic energy-radius phase space
    by analyzing particle distributions around isolated massive halos. Separately
    calibrates boundaries for particles with positive (outflowing) and negative
    (inflowing) radial velocities using different optimization strategies:
    
    - **Positive radial velocity**: Finds a percentile-based upper boundary line
      that contains a specified fraction of particles below it.
    - **Negative radial velocity**: Finds a boundary line that maximizes
      perpendicular distances within a specified band width, then applies
      a low-radius quadratic correction.
    
    The calibration parameters are saved to 'calibration_pars.hdf5' for use
    in halo finding algorithms.

    Parameters
    ----------
    n_seeds : int
        Number of isolated massive seeds to use for calibration. Must be positive.
        Function selects up to this many of the most massive isolated seeds.
        Typical values are 100-500. More seeds improve statistics but increase
        computation time. Only the most massive isolated halos are selected.
    seed_data : Tuple[numpy.ndarray]
        Tuple containing (position, velocity, mass, radius) arrays:
        - position: shape (n_total_seeds, 3) - seed coordinates
        - velocity: shape (n_total_seeds, 3) - seed velocities  
        - mass: shape (n_total_seeds,) - seed masses (M200m)
        - radius: shape (n_total_seeds,) - seed overdensity radii (R200m)
    r_max : float
        Maximum distance from seed centers to consider particles. Must be 
        positive, typically 2*R200m.
    boxsize : float
        Size of the cubic simulation box. Must be a positive.
    minisize : float  
        Size of each cubic mini-box. Must be positive and ≤ boxsize.
    save_path : str
        Directory path for reading mini-box data and writing calibration results.
        Must be a valid directory path with read/write permissions.
        Calibration parameters saved as 'calibration_pars.hdf5'.
    particle_mass : float
        Mass of each simulation particle. Must be positive.
    mass_density : float
        Mean matter density of the universe. Must be positive.
    redshift : float
        Cosmological redshift of the universe in simulation. Must be zero or 
        positive. Scales kinetic energies as ln(a^2v^2/V200m^2).
    n_radial_bins : int, optional
        Number of radial bins for gradient minima computation. Must be positive.
        Default is 20. More points give finer radial resolution but require more
        particles for stable statistics.
    gradient_radial_lims : Tuple[float], optional
        Radial interval (r_min, r_max) in units of R200m for computing gradient
        minima. Default is (0.2, 0.5). This range should capture the linear
        regime of the phase space boundaries.
    percent : float, optional
        Target percentile for positive radial velocity boundary as a fraction
        in range (0, 1). Default is 0.995 (99.5%). This fraction of positive-vr
        particles will fall below the calibrated boundary line.
    width : float, optional
        Half-width of band around line for negative radial velocity calibration 
        in units of ln(v^2/V200m^2). Must be positive. Default is 0.05. This is 
        scaled by (1 + z/3) to handle increased numerical noise. Smaller values 
        fit tighter to the data but may be noisier.
    radius_pivot : floar, optional
        Pivot radius that marks the transition to the low radius correction. 
        Default is 0.5
    isolation_factor : float, optional
        Maximum allowed mass ratio for isolation criterion. Neighbors must have
        mass < isolation_factor * seed_mass to satisfy isolation criterion.
        Default is 0.2 (20%).
    isolation_radius_factor : float, optional
        Factor multiplying R200m to define isolation radius. Neighbors within
        isolation_radius_factor * R200m are checked. Default is 2.0.
    n_threads : int, optional
        Number of threads for parallel processing. If None, uses half of
        available CPU cores. Only used when generating new data, capped by the 
        number of mini-boxes to process.
    diagnostics : bool, optional
        Whether to generate diagnostic plots of calibration process and results.
        Default is True. Plots saved to save_path.
    overwrite : bool, optional
        Whether to overwrite the existing calibration data. Default is False.
 
    Returns
    -------
    None
        Function saves calibration parameters to HDF5 file but returns nothing.

    Raises
    ------
    TypeError
        If n_seeds is not an integer, or if array inputs in seed_data cannot
        be converted to numpy arrays.
    ValueError
        If n_seeds is non-positive, if array shapes in seed_data are incompatible,
        if any scalar parameters are non-positive, if percent is not in (0, 1),
        or if gradient_radial_lims is invalid.
    FileNotFoundError
        If save_path doesn't exist or if required mini-box files are missing.
    OSError
        If HDF5 files cannot be read/written, or if directory permissions
        are insufficient.

    See Also
    --------
    calibrate : High-level interface with cosmology-based calibration option
    get_calibration_data : Loads particle data for calibration
    _gradient_minima : Finds boundaries from gradient analysis
    _cost_percentile : Cost function for percentile-based fitting
    _cost_perpendicular_distance_abscissa : Cost function for distance-based fitting

    Notes
    -----
    **Algorithm Overview**:
    
    1. Load calibration data from isolated massive halos
    2. Separate particles by radial velocity sign
    3. For positive radial velocity:
       - Compute gradient minima to find initial boundary slope
       - Fit linear model to minima (with outlier rejection)
       - Optimize intercept to achieve target percentile
    4. For negative radial velocity:
       - Compute gradient minima similarly
       - Optimize intercept to maximize perpendicular distances
       - Compute low-radius quadratic correction
    5. Save all parameters to HDF5 file
    
    **Calibration Parameters Saved**:
    
    - 'pos': [slope, intercept] for positive vr boundary (linear)
    - 'neg/line': [slope, intercept] for negative vr boundary (linear at large r)
    - 'neg/quad': [alpha, beta, gamma] for negative vr correction (quadratic at small r)
    
    The negative vr boundary uses a piecewise model:
    - Linear: y = slope * (r - radius_pivot) + intercept for r > radius_pivot
    - Quadratic: y = alpha * r² + beta * r + gamma for r ≤ radius_pivot
    
    **Outlier Rejection**:
    
    Uses median absolute deviation (MAD) with scale factor 1.4826 to identify
    and reject outliers in gradient minima before fitting. Points beyond 3*MAD
    from the median are excluded.
    
    **Redshift Scaling**:
    
    The band width for negative vr calibration is scaled as width * (1 + z/3)
    to handle increased numerical noise at higher redshifts. Calibrated up to z~3.

    Examples
    --------
    >>> import numpy as np
    >>> 
    >>> # Prepare seed catalog from halo finder
    >>> positions = np.random.uniform(0, 100, (1000, 3))
    >>> velocities = np.random.normal(0, 200, (1000, 3))
    >>> masses = np.random.lognormal(30, 1, 1000)
    >>> radii = (masses / 1e12) ** (1/3) * 0.5
    >>> seed_data = (positions, velocities, masses, radii)
    >>> 
    >>> # Run self-calibration
    >>> self_calibration(
    ...     n_seeds=200,
    ...     seed_data=seed_data,
    ...     r_max=4.0,
    ...     boxsize=100.0,
    ...     minisize=10.0,
    ...     save_path="/data/calibration/",
    ...     particle_mass=1e10,
    ...     mass_density=2.78e11,
    ...     redshift=0.0,
    ...     percent=0.995,
    ...     diagnostics=True
    ... )
    >>> 
    >>> # Load calibrated parameters
    >>> import h5py
    >>> with h5py.File("/data/calibration/calibration_pars.hdf5", 'r') as f:
    ...     slope_pos, b_pos = f['pos'][()]
    ...     slope_neg, b_neg = f['neg/line'][()]
    ...     alpha, beta, gamma = f['neg/quad'][()]
    >>> print(f"Positive vr: slope={slope_pos:.3f}, intercept={b_pos:.3f}")
    >>> print(f"Negative vr: slope={slope_neg:.3f}, intercept={b_neg:.3f}")

    >>> # Calibration at higher redshift
    >>> self_calibration(
    ...     n_seeds=300,
    ...     seed_data=seed_data_z2,
    ...     r_max=4.0,
    ...     boxsize=200.0,
    ...     minisize=10.0,
    ...     save_path="/data/calibration_z2/",
    ...     particle_mass=1e10,
    ...     mass_density=2.78e11,
    ...     redshift=2.0,  # Higher redshift
    ...     width=0.05,     # Will be scaled to 0.05 * (1 + 2/3) ≈ 0.083
    ...     n_threads=16,
    ...     diagnostics=True
    ... )
    """
    radius, radial_velocity, log_velocity_squared = get_calibration_data(
        n_seeds=n_seeds,
        seed_data=seed_data,
        r_max=r_max,
        boxsize=boxsize,
        minisize=minisize,
        save_path=save_path,
        particle_type=particle_type,
        mass_density=mass_density,
        redshift=redshift,
        n_threads=n_threads,
        isolation_factor=isolation_factor,
        isolation_radius_factor=isolation_radius_factor,
        diagnostics=diagnostics,
        overwrite=overwrite,
    )

    mask_negative_vr = (radial_velocity < 0)
    mask_positive_vr = ~mask_negative_vr
    mask_radius = radius <= 2.0
    mask_low_radius = radius <= 1.0
    def line_model(x, slope, abscissa): return slope * \
        (x - radius_pivot) + abscissa

    # ==========================================================================
    # 
    #                           Positive radial velocity
    # 
    # ==========================================================================
    radial_bins_positive, gradient_minumum_positive = _gradient_minima(
        radius=radius[mask_positive_vr],
        log_velocity_squared=log_velocity_squared[mask_positive_vr],
        n_radial_bins=n_radial_bins,
        r_min=gradient_radial_lims[0],
        r_max=gradient_radial_lims[1],
        save_path=save_path,
        diagnostics=diagnostics,
        diagnostics_title=r'Positive $v_r$ slope calibration',
    )

    # Mask outliers
    abs_dev = numpy.abs(gradient_minumum_positive - numpy.median(gradient_minumum_positive))
    median_abs_dev = 1.4286 * numpy.median(abs_dev)
    mask_outliers = abs_dev <= 3. * median_abs_dev
    
    # Find slope by fitting to the minima.
    (slope_positive_vr, abscissa_p), _ = curve_fit(line_model, 
                                                   radial_bins_positive[mask_outliers],
                                                   gradient_minumum_positive[mask_outliers],
                                                   p0=[-1, 2], bounds=((-5, 0), (0, 5)))

    # Find intercept by finding the value that contains 'perc' percent of
    # particles below the line at fixed slope 'm_pos'.
    result = minimize(
        fun=_cost_percentile,
        x0=1.1 * abscissa_p,
        bounds=((abscissa_p, 5.0),),
        args=(radius[mask_positive_vr & mask_radius],
              log_velocity_squared[mask_positive_vr & mask_radius],
              slope_positive_vr, percent, radius_pivot),
        method='Nelder-Mead',
    )
    abscissa_positive_vr = result.x[0]

    # ==========================================================================
    # 
    #                           Negative radial velocity
    # 
    # ==========================================================================
    radial_bins_negative, gradient_minumum_negative = _gradient_minima(
        radius=radius[mask_negative_vr],
        log_velocity_squared=log_velocity_squared[mask_negative_vr],
        n_radial_bins=n_radial_bins,
        r_min=gradient_radial_lims[0],
        r_max=gradient_radial_lims[1],
        save_path=save_path,
        diagnostics=diagnostics,
        diagnostics_title=r'Negative $v_r$ slope calibration',
    )

    # Mask outliers
    abs_dev = numpy.abs(gradient_minumum_negative - numpy.median(gradient_minumum_negative))
    median_abs_dev = 1.4286 * numpy.median(abs_dev)
    mask_outliers = abs_dev <= 3. * median_abs_dev

    # Find abscissa initial guess by fitting to the minima.
    (_, abscissa_n), _ = curve_fit(line_model, 
                                   radial_bins_negative[mask_outliers],
                                   gradient_minumum_negative[mask_outliers],
                                   p0=[-1, 2], bounds=((-5, 0), (0, 3)))
    # Fix slope to the positive radial velocity value
    slope_negative_vr = slope_positive_vr

    # The user input width is scaled with redshift such that it is double at 
    # z = 3. This was needed due to the numerical noise in the cost function 
    # when using small band widths. Has not been tested beyond this redshift.
    width *= (1. + redshift/3.)

    # Find intercept by finding the value that maximizes the perpendicular
    # distance to the line at fixed slope of all points within a perpendicular
    # 'width' distance from the line (ignoring all others).
    result = minimize(
        fun=_cost_perpendicular_distance_abscissa,
        x0=0.5 * abscissa_n,
        bounds=((0., abscissa_n),),
        args=(radius[mask_negative_vr & mask_low_radius], 
              log_velocity_squared[mask_negative_vr & mask_low_radius],
              slope_negative_vr, width, radius_pivot),
        method='Nelder-Mead',
    )
    abscissa_negative_vr = result.x[0]

    # Compute low radius correction parameters
    b_neg = abscissa_negative_vr - slope_negative_vr * radius_pivot
    gamma = 2.
    alpha = (gamma - b_neg) / radius_pivot**2
    beta = slope_negative_vr - 2 * alpha * radius_pivot

    # Save to file
    with h5py.File(save_path + 'calibration_pars.hdf5', 'w') as hdf:
        hdf.create_dataset(
            'pos', data=[slope_positive_vr, abscissa_positive_vr])
        hdf.create_dataset(
            'neg/line', data=[slope_negative_vr, abscissa_negative_vr])
        hdf.create_dataset('neg/quad', data=[alpha, beta, gamma])

    if diagnostics:
        parameters = (
            radius_pivot,
            slope_positive_vr,
            abscissa_positive_vr,
            slope_negative_vr,
            abscissa_negative_vr,
            alpha,
            beta,
            gamma,
        )

        gradient_values = [
            radial_bins_positive,
            gradient_minumum_positive,
            radial_bins_negative,
            gradient_minumum_negative,
        ]

        _diagnostic_self_calibration_plot(
            save_path=save_path,
            radius=radius,
            radial_velocity=radial_velocity,
            log_velocity_squared=log_velocity_squared,
            parameters=parameters,
            gradient_values=gradient_values,
        )

    return


def calibrate(
    save_path: str,
    omega_m: Optional[float] = None,
    **kwargs,
) -> None:
    """
    Calibrate halo finder parameters using cosmology-based or simulation-based 
    approach.

    This is a high-level interface function that provides two calibration modes:
    
    1. **Cosmology-based calibration** (omega_m provided): Uses pre-calibrated
       empirical relations between calibration parameters and matter density
       parameter \Omega_m. This is fast and doesn't require simulation data, but
       assumes standard Lambda-CDM cosmology. This relation was calibrated with
       a sample of the latin-hypercube from the Quijote suite of simulations. 
       More information at https://quijote-simulations.readthedocs.io/en/latest/
       
    2. **Self-calibration** (omega_m=None): Performs full self-calibration from
       simulation data by analyzing particle distributions around isolated halos.
       This is more accurate for non-standard cosmologies or specific simulations
       but requires more computation.

    Both modes save calibration parameters to 'calibration_pars.hdf5' in the
    specified directory for use in halo finding.

    Parameters
    ----------
    save_path : str
        Directory path where calibration parameters will be saved. Must be a
        valid directory path with write permissions. For self-calibration mode,
        this directory must also contain the mini-box data files.
        Calibration parameters saved as 'calibration_pars.hdf5'.
    omega_m : float, optional
        Matter density parameter at redshift z, \Omega_m(z) for cosmology-based 
        calibration. If provided, uses empirical scaling relations. Must be 
        positive. Standard Lambda-CDM values:
        - Planck 2018: \Omega_m ≈ 0.315
        - WMAP: \Omega_m ≈ 0.27
        If None (default), performs full self-calibration from simulation data.
    **kwargs : dict
        Additional keyword arguments passed to self_calibration when omega_m=None.
        Ignored when using cosmology-based calibration. Required parameters for
        self-calibration include:
        - n_seeds : int - Number of halos for calibration
        - seed_data : tuple - Halo catalog data
        - r_max : float - Maximum radius for particle selection
        - boxsize : float - Simulation box size
        - minisize : float - Minibox size
        - particle_mass : float - Particle mass
        - mass_density : float - Mean matter density
        - redshift : float - Simulation redshift
        See self_calibration docstring for complete parameter descriptions.

    Returns
    -------
    None
        Function saves calibration parameters to HDF5 file but returns nothing.

    Raises
    ------
    ValueError
        If omega_m is provided but is non-positive or unreasonably large (>1).
    OSError
        If save_path doesn't exist or lacks write permissions, or if
        required mini-box files are missing (self-calibration mode only).
    TypeError
        If required kwargs are missing or have wrong types (self-calibration mode).

    See Also
    --------
    self_calibration : Full self-calibration from simulation data
    get_calibration_data : Loads particle data for calibration

    Notes
    -----
    **Cosmology-based Calibration**:
    
    Uses empirical fitting formulas calibrated from a suite of simulations:
    
    - Positive radial velocity boundary:
      * slope = -2.1203 + 0.7614 * (\Omega_m(z) - 0.3)
      * abscissa = 1.8087 (fixed)
      
    - Negative radial velocity boundary (at r_pivot = 0.5):
      * slope = -1.5812 + 0.8603 * (\Omega_m(z) - 0.3)
      * abscissa = 0.6204 (fixed)
      * Quadratic correction computed to match boundary at r=0.5
      
    These relations are valid for Lambda-CDM cosmologies and were calibrated at 
    z ∈ [0, 3] on the latin-hypercube sample of the Quijote suite of simulations.
    For other redshifts or non-standard cosmologies, self-calibration is recommended.
    
    **Self-calibration**:
    
    Performs full calibration by analyzing particle phase space around isolated
    massive halos. See self_calibration docstring for algorithm details.
    Recommended when:
    - Using non-standard cosmology
    - Simulations with modified gravity or dark energy
    - Requiring maximum accuracy for specific simulation
    - Calibrating at high redshift (z > 3)
    
    **HDF5 File Structure**:
    
    The output file 'calibration_pars.hdf5' contains:
    - 'pos': [slope, intercept] for positive vr boundary
    - 'neg/line': [slope, intercept] for negative vr boundary (linear part)
    - 'neg/quad': [alpha, beta, gamma] for negative vr boundary (quadratic correction)

    Examples
    --------
    >>> # Example 1: Fast cosmology-based calibration
    >>> calibrate(
    ...     save_path="/data/calibration/",
    ...     omega_m=0.3  # Use \Omega_m = 0.3 (close to Planck 2018)
    ... )
    >>> 
    >>> # Verify saved parameters
    >>> import h5py
    >>> with h5py.File("/data/calibration/calibration_pars.hdf5", 'r') as f:
    ...     slope_pos, b_pos = f['pos'][()]
    ...     print(f"Positive vr: slope={slope_pos:.3f}, intercept={b_pos:.3f}")
    Positive vr: slope=-1.915, intercept=1.664

    >>> # Example 2: Self-calibration from simulation data
    >>> import numpy as np
    >>> 
    >>> # Prepare halo catalog
    >>> positions = np.random.uniform(0, 100, (1000, 3))
    >>> velocities = np.random.normal(0, 200, (1000, 3))
    >>> masses = np.random.lognormal(30, 1, 1000)
    >>> radii = (masses / 1e12) ** (1/3) * 0.5
    >>> seed_data = (positions, velocities, masses, radii)
    >>> 
    >>> calibrate(
    ...     save_path="/data/calibration/",
    ...     omega_m=None,  # Trigger self-calibration mode
    ...     n_seeds=200,
    ...     seed_data=seed_data,
    ...     r_max=4.0,
    ...     boxsize=100.0,
    ...     minisize=10.0,
    ...     particle_mass=1e10,
    ...     mass_density=2.78e11,
    ...     redshift=0.0,
    ...     diagnostics=True
    ... )

    >>> # Example 3: Different cosmologies
    >>> for omega in [0.25, 0.30, 0.35]:
    ...     output_dir = f"/data/calibration_Om{omega:.2f}/"
    ...     calibrate(save_path=output_dir, omega_m=omega)
    ...     print(f"Calibrated for \Omega_m = {omega}")

    >>> # Example 4: High-redshift self-calibration
    >>> calibrate(
    ...     save_path="/data/calibration_z2/",
    ...     omega_m=None,
    ...     n_seeds=300,
    ...     seed_data=seed_data_z2,
    ...     r_max=4.0,
    ...     boxsize=200.0,
    ...     minisize=10.0,
    ...     particle_mass=1e10,
    ...     mass_density=2.78e11,
    ...     redshift=2.0,  # High redshift
    ...     n_threads=16,
    ...     diagnostics=True
    ... )
    """
    if omega_m:
        if omega_m <=0 or omega_m > 1:
            raise ValueError("Omega_m out of bounds.")
        
        # Positive radial velocity
        # slope_pos = -2.17603504 + 1.03114681 * (omega_m - 0.3)
        # b_pivot_pos = 1.83532578

        # slope_pos = -2.174666 + 1.435353 * (omega_m - 0.3)
        slope_pos = -1.9029634573662748 + 0.4347721236533314 * numpy.log(omega_m / 0.5)
        b_pivot_pos = 1.808536

        # Negative radial velocity
        slope_neg = slope_pos
        b_pivot_neg = 0.665695
        
        # Low radius correction
        x0 = 0.5
        gamma = 2.
        b_neg = b_pivot_neg - slope_neg * x0
        alpha = (gamma - b_neg) / x0**2
        beta = slope_neg - 2 * alpha * x0

        with h5py.File(save_path + 'calibration_pars.hdf5', 'w') as hdf:
            hdf.create_dataset('pos', data=[slope_pos, b_pivot_pos])
            hdf.create_dataset('neg/line', data=[slope_neg, b_pivot_neg])
            hdf.create_dataset('neg/quad', data=[alpha, beta, gamma])
    else:
        self_calibration(save_path=save_path, **kwargs)

    return
