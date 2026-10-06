import os
from multiprocessing.dummy import Pool as ThreadPool
from pathlib import Path
from typing import List, Optional, Tuple, Union

import h5py
import numpy
from tqdm import tqdm

from oasis.common import (TimerContext, _validate_boxsize_minisize,
                          _validate_coordinate_array, _validate_inputs_load,
                          _validate_mini_box_id, _validate_process_objects,
                          ensure_dir_exists, get_min_uint_dtype)
from oasis.coordinates import _periodic_displacement

__all__ = [
    'get_mini_box_id',
    'get_adjacent_mini_box_ids',
    'process_simulation_data',
    'load_particles',
    'load_seeds',
]


def get_mini_box_id(
    position: numpy.ndarray,
    boxsize: float,
    minisize: float,
) -> Union[int, numpy.ndarray]:
    """
    Returns the mini-box ID(s) to which the given coordinates fall into.

    This function divides a simulation box into smaller cubic mini-boxes and 
    determines which mini-box each position belongs to. Mini-boxes are indexed
    using a 3D grid system converted to unique 1D IDs.

    Parameters
    ----------
    position : numpy.ndarray
        Position(s) in cartesian coordinates. Can be:
        - 1D array of shape (3,) for a single position
        - 2D array of shape (N, 3) for N positions
        Each position should have [x, y, z] coordinates within [0, boxsize).
    boxsize : float
        Size of the cubic simulation box. Must be a positive.
    minisize : float  
        Size of each cubic mini-box. Must be positive and ≤ boxsize.

    Returns
    -------
    int or numpy.ndarray
        Mini-box ID(s). Returns:
        - int: if input was a single position (1D array)
        - numpy.ndarray: if input was multiple positions (2D array)

    Raises
    ------
    TypeError
        If position is not a numpy array or boxsize or minisize are not numeric.
    ValueError
        If minisize > boxsize, or if any coordinate is outside [0, boxsize),
        or if position array has wrong dimensions, or if boxsize or minisize ≤ 0.

    Notes
    -----
    - Positions exactly at the upper boundary (boxsize) are adjusted inward
      by a small tolerance to ensure they fall within valid mini-boxes.
    - The function modifies the input array in-place for memory efficiency.
    - The function uses a 3D-to-1D mapping: ID = i + j*nx + k*nx*ny
      where (i,j,k) are grid indices and nx, ny, nz are grid dimensions.

    Examples
    --------
    >>> import numpy as numpy
    >>> pos = numpy.array([1.5, 2.5, 3.5])
    >>> get_mini_box_id(pos, boxsize=10.0, minisize=2.0)
    32

    >>> positions = numpy.array([[1.5, 2.5, 3.5], [0.1, 0.1, 0.1]])
    >>> get_mini_box_id(positions, boxsize=10.0, minisize=2.0)
    array([32,  0])
    """
    EDGE_TOL = 1e-8

    # Input validation
    _validate_coordinate_array(position, name="position")
    _validate_boxsize_minisize(boxsize, minisize)

    # Validate coordinate bounds
    if numpy.any(position < 0) or numpy.any(position > boxsize):
        raise ValueError(f"All coordinates must be within [0, {boxsize}]")

    # Pre-compute grid parameters
    n_cells_per_side = int(numpy.ceil(boxsize / minisize))
    shift = numpy.array([1, n_cells_per_side, n_cells_per_side**2])

    # Handle boundary conditions
    # Points at upper boundary (within numerical precision)
    upper_mask = numpy.abs(position - boxsize) < EDGE_TOL
    position[upper_mask] -= EDGE_TOL

    # Points at lower boundary (within numerical precision)
    lower_mask = numpy.abs(position) < EDGE_TOL
    position[lower_mask] += EDGE_TOL

    # Compute grid indices
    grid_indices = numpy.floor(position / minisize).astype(int)

    # Additional safety check after grid computation
    max_index = n_cells_per_side - 1
    if numpy.any(grid_indices < 0) or numpy.any(grid_indices > max_index):
        raise RuntimeError(
            "Grid index computation resulted in out-of-bounds indices")

    # Compute unique IDs using dot product for efficiency
    if grid_indices.ndim == 1:
        ids = int(numpy.dot(shift, grid_indices))
    else:
        ids = numpy.sum(shift * grid_indices, axis=1)

    # Return appropriate type based on input
    return ids


def get_adjacent_mini_box_ids(
    mini_box_id: Union[int, numpy.integer],
    boxsize: float,
    minisize: float,
) -> numpy.ndarray:
    """
    Returns all mini-box IDs adjacent to a specified mini-box, including itself.

    This function finds all 27 mini-boxes in the 3x3x3 neighborhood surrounding
    the given mini-box ID in a 3D grid. The neighborhood includes the center box
    itself and all 26 surrounding boxes. Periodic boundary conditions are applied,
    so boxes at the grid edges wrap around to the opposite side.

    Parameters
    ----------
    mini_box_id : int or numpy.integer
        ID of the central mini-box. Must be a valid ID within the grid
        (0 ≤ mini_box_id < total_mini_boxes).
    boxsize : float
        Size of the cubic simulation box. Must be positive.
    minisize : float
        Size of each cubic mini-box. Must be positive and ≤ boxsize.

    Returns
    -------
    numpy.ndarray
        Array of shape (27,) containing all adjacent mini-box IDs, including
        the input ID. The first element is always the input mini_box_id, 
        followed by the 26 neighboring IDs in no particular order.
        Array dtype is numpy.int32.

    Raises
    ------
    TypeError
        If mini_box_id is not an integer type, or if boxsize/minisize are not numeric.
    ValueError
        If minisize > boxsize, or if boxsize/minisize ≤ 0, or if mini_box_id
        is outside the valid range [0, total_mini_boxes-1].

    See Also
    --------
    get_mini_box_id : Convert coordinates to mini-box ID

    Notes
    -----
    - Uses periodic boundary conditions: mini-boxes at grid boundaries are
      considered adjacent to mini-boxes on the opposite boundary.
    - The 3D grid uses the mapping: ID = k + jxnx + ixnx², where (i,j,k) are
      grid coordinates and nx is the number of cells per side.
    - For a grid with n cells per side, total mini-boxes = n³.
    - The function always returns exactly 27 IDs regardless of grid size.

    Examples
    --------
    >>> import numpy as np
    >>> # 2x2x2 grid (8 total mini-boxes)
    >>> get_adjacent_mini_box_ids(0, boxsize=2.0, minisize=1.0)
    array([0, 1, 2, 3, 4, 5, 6, 7, ...], dtype=int32)  # All boxes due to wrapping

    >>> # Larger grid - interior box
    >>> ids = get_adjacent_mini_box_ids(111, boxsize=10.0, minisize=1.0)  # 10x10x10 grid
    >>> len(ids)
    27
    >>> ids[0]  # First element is always the input ID
    111
    """
    # Input validation
    _validate_boxsize_minisize(boxsize, minisize)

    cells_per_side = int(numpy.ceil(boxsize / minisize))
    _validate_mini_box_id(mini_box_id, cells_per_side)

    # Grid parameters
    total_mini_boxes = cells_per_side**3
    max_id = total_mini_boxes - 1

    # Convert 1D ID to 3D grid coordinates (i, j, k)
    # Using the mapping: ID = k + j*cells_per_side + i*cells_per_side²
    i = mini_box_id // (cells_per_side**2)
    remainder = mini_box_id % (cells_per_side**2)
    j = remainder // cells_per_side
    k = remainder % cells_per_side

    # Verify the coordinate conversion (safety check)
    reconstructed_id = k + j * cells_per_side + i * (cells_per_side**2)
    if reconstructed_id != mini_box_id:
        raise RuntimeError(
            f"Grid coordinate conversion failed: ID {mini_box_id} -> "
            f"coords ({i},{j},{k}) -> ID {reconstructed_id}"
        )

    # Pre-allocate array for all 27 adjacent IDs
    adjacent_ids = numpy.empty(27, dtype=numpy.int32)

    # First element is always the input ID
    adjacent_ids[0] = mini_box_id
    idx = 1

    # Generate 3x3x3 neighborhood with periodic boundary conditions
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            for dk in [-1, 0, 1]:
                # Skip the center cell (already added)
                if di == 0 and dj == 0 and dk == 0:
                    continue

                # Apply periodic boundary conditions using modulo
                ni = (i + di) % cells_per_side
                nj = (j + dj) % cells_per_side
                nk = (k + dk) % cells_per_side

                # Convert 3D coordinates back to 1D ID
                neighbor_id = nk + nj * cells_per_side + \
                    ni * (cells_per_side**2)
                adjacent_ids[idx] = neighbor_id
                idx += 1

    return adjacent_ids


def split_simulation_into_mini_boxes(
    positions: numpy.ndarray,
    velocities: numpy.ndarray,
    uid: numpy.ndarray,
    save_dir: str,
    boxsize: float,
    minisize: float,
    particle_type: str,
    mass: numpy.ndarray,
    mass_label: str,
    props: Optional[Tuple[List[numpy.ndarray], List[str], List[numpy.dtype]]] = None,
    n_threads: Optional[int] = None,
) -> None:
    """
    Split simulation data into mini-boxes and save to HDF5 files.

    This function partitions simulation data (positions, velocities, IDs) into 
    smaller cubic boxes for efficient processing. Each mini-box is saved as a 
    separate HDF5 file containing all objects within that region.

    Parameters
    ----------
    positions : numpy.ndarray
        Positions with shape (n, 3) in cartesian coordinates, where n is
        the number of objects.
    velocities : numpy.ndarray
        Velocities with shape (n, 3) in cartesian velocity components.
    uid : numpy.ndarray
        Unique identifiers for each object with shape (n,). Examples include 
        particle IDs (PID) or halo IDs (HID).
    save_path : str
        Target directory path where mini-box files will be saved. A subdirectory
        will be created automatically at /save_path/mini_boxes_nside_<xx>/
    boxsize : float
        Size of the cubic simulation box. Must be a positive.
    minisize : float
        Target size of each mini-box. Must be positive and ≤ boxsize.
    name : str, optional
        Additional identifier to create a group within each HDF5 file.
        If None, datasets are stored at the root level. Default is None.
    props : tuple of (list, list, list), optional
        Additional particle properties to include. Must contain exactly 3 lists:
        - List of numpy arrays with additional object data. For example M200, 
            Mvir, Rvir, etc.
        - List of string labels for each array that will be used as dataset name
            in the HDF5 file.
        - List of numpy dtypes for each array
        All lists must have the same number of elements and all arrays must have
        shape (n, ...). Default is None.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If input arrays have incompatible shapes, if boxsize or minisize are
        invalid, or if props tuple has incorrect structure.
    OSError
        If save_path cannot be created or accessed.
    RuntimeError
        If chunking parameters result in processing errors.

    Notes
    -----
    - The function creates a subdirectory named 'mini_boxes_nside_<xx>/' where
      <xx> is the number of cells per side
    - Each mini-box is saved as '{mini_box_id}.hdf5'
    - Memory usage is optimized by processing particles in chunks
    - Progress bars show completion status for ID computation and file saving
    - Positions are wrapped into [0, boxsize); positions that would round up
      to boxsize when stored as float32 are stored as 0 (periodically the same)

    Examples
    --------
    >>> positions = numpy.random.rand(1000, 3) * 100.0
    >>> velocities = numpy.random.rand(1000, 3) * 10.0
    >>> particle_ids = numpy.arange(1000)
    >>> split_simulation_into_mini_boxes(
    ...     positions, velocities, particle_ids,
    ...     save_path="/data/output/",
    ...     boxsize=100.0, minisize=10.0
    ... )
    """
    n_items = positions.shape[0]
    if n_items == 0:
        raise ValueError("No objects provided (empty arrays)")

    # Determine number of partitions per side
    n_cells = int(numpy.ceil(boxsize / minisize))**3

    # Trim values outside boxsize due to floating point precision
    positions = numpy.mod(positions, boxsize)

    # Positions are stored as float32. Values just below 'boxsize' can round up
    # to exactly 'boxsize' (or above) in float32, which the periodic KD-trees
    # reject. Such a position is periodically equivalent to 0, so store it as 0. 
    on_edge = positions.astype(numpy.float32).astype(numpy.float64) >= boxsize
    if numpy.any(on_edge):
        positions[on_edge] = 0.0

    # Compute mini-box IDs for all objects in chunks with size of the average 
    # number of items per mini-box to improve computation speed and reduce 
    # memory usage. This is important for large simulations with billions of 
    # particles.
    chunksize = int(max(1, numpy.ceil(n_items / n_cells)))  # Ensure chunksize >= 1
    uint_dtype = get_min_uint_dtype(n_cells)
    mini_box_ids = numpy.zeros(n_items, dtype=uint_dtype)

    def get_chunk_mini_box_ids(i):
        low = i * chunksize
        # Ensure we don't exceed array bounds
        upp = min((i + 1) * chunksize, n_items)
        mini_box_ids[low:upp] = get_mini_box_id(positions[low:upp], boxsize, minisize)

    n_chunks = (n_items + chunksize - 1) // chunksize  # Ceiling division
    
    # Cap the number of threads to the total number of mini-boxes to process
    if n_threads is None:
        n_threads = min(max(1, os.cpu_count()//2), n_chunks)
    else:
        n_threads = min(n_threads, n_chunks)

    # Safely handle multiprocessing falure with a fall back to a single thread.
    if n_threads > 1:
        try:
            with ThreadPool(n_threads) as pool:
                list(tqdm(pool.imap(get_chunk_mini_box_ids, range(n_chunks)),
                          total=n_chunks, desc='Computing IDs', ncols=100, 
                          colour='blue'))
        except Exception as e:
            print(f"Warning: Parallel processing failed ({e}), "
                  "falling back to sequential")
            # Fall back to sequential processing
            n_threads = 1
            
    if n_threads == 1:
        for chunk in tqdm(range(n_chunks), desc='Computing IDs', ncols=100, 
                          colour='blue'):
            low = chunk * chunksize
            # Ensure we don't exceed array bounds
            upp = min((chunk + 1) * chunksize, n_items)
            mini_box_ids[low:upp] = get_mini_box_id(positions[low:upp], boxsize, minisize)
    
    # Sort the mini-box IDs once. The arrays themselves are not reordered
    # (that would make a full sorted copy of every array); each mini-box
    # gathers its rows through its slice of `sort_indices` instead, which
    # writes exactly the same rows in the same order.
    with TimerContext("Sorting by mini-box ID", fancy=False):
        sort_indices = numpy.argsort(mini_box_ids)

        if props:
            data_arrays, data_labels, data_dtypes = props
    
    # Find boundaries of each mini-box in the sorted order from the number of
    # items per mini-box (avoids a second sort inside numpy.unique).
    counts = numpy.bincount(mini_box_ids, minlength=n_cells)
    del mini_box_ids
    unique_ids = numpy.flatnonzero(counts)
    end_all = numpy.cumsum(counts)
    start_indices = (end_all - counts)[unique_ids]
    end_indices = end_all[unique_ids]
    
    n_mini_box_ids = len(unique_ids)

    # Get smallest data type to represent IDs
    uint_dtype_pid = get_min_uint_dtype(numpy.max(uid))

    # For each mini-box, save all items in that box to a separate HDF5 file.
    def write_mini_box(idx):
        box_id = unique_ids[idx]
        start = start_indices[idx]
        end = end_indices[idx]
        
        # Rows of this mini-box, in the same order as a full sort would give
        rows = sort_indices[start:end]
        pos = positions[rows]
        vel = velocities[rows]
        ids = uid[rows]
        
        # Use 'a' mode to append data if file already exists.
        with h5py.File(save_dir + f'{box_id}.hdf5', 'a') as hdf:
            hdf.create_dataset(name=f'{particle_type}/ID', data=ids, dtype=uint_dtype_pid)
            hdf.create_dataset(name=f'{particle_type}/pos', data=pos, dtype=numpy.float32)
            hdf.create_dataset(name=f'{particle_type}/vel', data=vel, dtype=numpy.float32)
            
            if isinstance(mass, float):
                hdf.create_dataset(name=f'{particle_type}/{mass_label}', 
                                   data=mass, dtype=numpy.float32)
            else:
                hdf.create_dataset(name=f'{particle_type}/{mass_label}', 
                                   data=mass[rows], dtype=numpy.float32)

            if props:
                for arr_i, label_i, dtype_i in zip(data_arrays, data_labels, data_dtypes):
                    hdf.create_dataset(name=f'{particle_type}/{label_i}', 
                                    data=arr_i[rows], dtype=dtype_i)

    if n_threads > 1:
        try:
            with ThreadPool(n_threads) as pool:
                list(tqdm(pool.imap_unordered(write_mini_box, range(n_mini_box_ids)),
                  total=n_mini_box_ids, desc='Saving mini-boxes', ncols=100, colour='green'))
        except Exception as e:
            print(
                f"Warning: Parallel processing failed ({e}), falling back to sequential")
            # Fall back to sequential processing
            n_threads = 1

    if n_threads == 1:
        for i in tqdm(range(n_mini_box_ids), desc='Saving mini-boxes', ncols=100, 
                      colour='blue'):
            write_mini_box(i)

    return None


def process_simulation_data(
    save_path: str,
    particle_type: str,
    boxsize: float,
    minisize: float,
    positions: numpy.ndarray,
    velocities: numpy.ndarray,
    ids: numpy.ndarray,
    mass: Tuple[Union[float, numpy.ndarray], str],
    data: Optional[Tuple[Tuple[numpy.ndarray], Tuple[str], Tuple[numpy.dtype]]] = None,
    n_threads: Optional[int] = None,
) -> None:
    """
    Validate and process simulation particle data before saving.

    Parameters
    ----------
    save_path : str
        Path to save processed data.
    particle_type : str
        Particle category (e.g., "dm", "gas", "stars").
    boxsize : float
        Simulation box size.
    minisize : float
        Size of smaller sub-boxes.
    positions : numpy.ndarray
        Particle position array of shape (N, 3).
    velocities : numpy.ndarray
        Particle velocity array of shape (N, 3).
    ids : numpy.ndarray
        Unique particle IDs of shape (N,).
    mass : tuple of (float or numpy.ndarray, str)
        Mass information and corresponding label.
    data : tuple, optional
        Additional simulation arrays with (arrays, labels, dtypes).
    n_threads : int, optional
        Number of threads to use for parallel operations.

    Raises
    ------
    ValueError
        If arrays have mismatched lengths or invalid types.
    """
    # Validate inputs
    _validate_boxsize_minisize(boxsize, minisize)
    _validate_coordinate_array(positions, "positions")
    _validate_coordinate_array(velocities, "velocities")

    if ids.ndim != 1:
        raise ValueError("ids must be a 1D array")

    if not isinstance(mass, tuple):
        raise ValueError("Input 'mass' must be either tuple(float, str) or tuple(array, str)")
    mass, mass_label = mass

    if isinstance(mass, float):
        shapes = (velocities.shape[0], ids.shape[0])
    else:
        shapes = (velocities.shape[0], ids.shape[0], mass.shape[0])

    n_particles = positions.shape[0]
    if any(shape != n_particles for shape in shapes):
        raise ValueError("All positions, velocities, ids and masses must have "
                         "the same number of elements")

    # If particle type is 'dm' then data must be a float.
    if particle_type in ('dm', 'part') and not isinstance(mass, (float, int)):
        raise ValueError("Expected 'data' to be scalar for selected "
                         f"particle_type but found type {type(mass)}")
    
    # Validate seed data shape and type
    if data:
        _validate_process_objects(data, positions.shape[0])
    
    # Determine number of partitions per side
    cells_per_side = int(numpy.ceil(boxsize / minisize))
    # Create target directory
    save_dir = save_path + f'mini_boxes_nside_{cells_per_side}/'
    ensure_dir_exists(save_dir)

    split_simulation_into_mini_boxes(
        positions=positions,
        velocities=velocities,
        uid=ids,
        save_dir=save_dir,
        boxsize=boxsize,
        minisize=minisize,
        particle_type=particle_type,
        mass=mass,
        mass_label=mass_label,
        props=data,
        n_threads=n_threads,
    )

    return None


def _unique_in_order(values: numpy.ndarray) -> list:
    """Return the unique values in order of first appearance."""
    return list(dict.fromkeys(numpy.asarray(values).tolist()))


def _mini_box_center(
    mini_box_id: Union[int, numpy.integer],
    cells_per_side: int,
    minisize: float,
) -> numpy.ndarray:
    """Center coordinates of a mini-box. Grid cells start at (0,0,0) with size
    minisize, using the mapping ID = k + j*cells_per_side + i*cells_per_side²."""
    i = mini_box_id // (cells_per_side**2)
    remainder = mini_box_id % (cells_per_side**2)
    j = remainder // cells_per_side
    k = remainder % cells_per_side
    return numpy.array([
        (k + 0.5) * minisize,
        (j + 0.5) * minisize,
        (i + 0.5) * minisize
    ])


def load_particles(
    mini_box_id: Union[int, numpy.integer],
    boxsize: float,
    minisize: float,
    load_path: Union[str, Path],
    particle_type: str,
    padding: float = 5.0,
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, Union[float, numpy.ndarray]]:
    """
    Load particles from a mini-box and its adjacent neighbors within padding distance.

    This function loads particle data (positions, velocities, IDs) from a specified 
    mini-box and all 26 neighboring mini-boxes, then filters particles to only 
    include those within a specified padding distance from the target mini-box edges.
    This is useful for analysis that requires particles near box boundaries to avoid
    edge effects.

    Parameters
    ----------
    mini_box_id : int or numpy.integer
        ID of the target mini-box from which to load particles. Must be a valid 
        ID within the grid (0 ≤ mini_box_id < total_mini_boxes).
    boxsize : float
        Size of the cubic simulation box. Must be a positive.
    minisize : float  
        Size of each cubic mini-box. Must be positive and ≤ boxsize.
    load_path : str or Path
        Path to directory containing the mini-box HDF5 files. The directory
        should contain a subdirectory named 'mini_boxes_nside_<xx>/' where <xx>
        is the number of cells per side.
    particle_type : str
        Name of the particle group in the mini-box files (e.g. 'dm').
    padding : float, optional
        Maximum distance from mini-box edges to include particles. Particles 
        further than this distance from any edge of the target mini-box will be 
        excluded. Must be positive. Default is 5.0.

    Returns
    -------
    positions : numpy.ndarray
        Particle positions with shape (n_particles, 3) in cartesian coordinates 
        of particles within the padding distance.
    velocities : numpy.ndarray  
        Particle velocities with shape (n_particles, 3) in cartesian velocity 
        components corresponding to the returned positions.
    particle_ids : numpy.ndarray
        Particle IDs with shape (n_particles,) containing unique identifiers
        corresponding to the returned positions and velocities.
    masses : float or numpy.ndarray
        Particle mass: a single value for constant-mass particles, otherwise
        one value per returned particle.

    Raises
    ------
    TypeError
        If mini_box_id is not an integer, or if boxsize, minisize or padding are
        not numeric, or load_path is not a string or Path.
    ValueError
        If mini_box_id is negative or invalid for the grid, boxsize or minisize 
        are non-positive, minisize > boxsize, or padding is not positive.
    FileNotFoundError
        If load_path or its 'mini_boxes_nside_<xx>/' folder does not exist.
    NotADirectoryError
        If load_path is not a directory.
    OSError
        If HDF5 files cannot be read or are corrupted.
    ValueError
        If no file of the mini-box or its neighbours contains particle_type,
        or if they contain no rows at all. (If rows exist but none lie within
        the padding, empty arrays are returned.)

    See Also
    --------
    get_adjacent_mini_box_ids : Get IDs of neighboring mini-boxes
    split_simulation_into_mini_boxes : Create mini-box files

    Notes
    -----
    - Uses periodic boundary conditions when calculating relative coordinates
    - Reads the target mini-box and its 26 neighbours, each file once (fewer
      files when the grid has fewer than 3 cells per side)
    - Mini-boxes without a file (nothing in them) and files without
      particle_type (e.g. seeds only) are skipped
    - The padding mask is applied to each file as it is read, so memory scales
      with the selected particles, not with all 27 mini-boxes

    Examples
    --------
    >>> positions, velocities, ids, masses = load_particles(
    ...     mini_box_id=42,
    ...     boxsize=100.0,
    ...     minisize=10.0,
    ...     load_path="/data/simulation/",
    ...     particle_type="dm",
    ...     padding=2.0
    ... )
    >>> print(f"Loaded {len(positions)} particles")
    """
    # Input validation
    _validate_inputs_load(mini_box_id, boxsize, minisize, load_path, padding)

    # Get the adjacent mini-box IDs. With fewer than 3 cells per side the same
    # neighbour appears several times; load each mini-box only once.
    mini_box_ids = _unique_in_order(get_adjacent_mini_box_ids(
        mini_box_id=mini_box_id,
        boxsize=boxsize,
        minisize=minisize
    ))
    
    # Determine number of partitions per side
    cells_per_side = int(numpy.ceil(boxsize / minisize))

    # Select particles within a padding distance of the edge of the box in each
    # direction. The mask is applied to each file as it is read, so only the
    # selected particles are kept in memory and concatenated.
    center = _mini_box_center(mini_box_id, cells_per_side, minisize)
    padded_distance = 0.5 * minisize + padding

    # Create empty lists (containers) to save the data from file for each ID
    positions, velocities, ids, masses = ([] for _ in range(4))
    n_loaded = 0
    n_with_type = 0

    # The minibox folder must exist (a missing folder means a wrong
    # load_path or minisize, not an empty region).
    box_dir = Path(load_path) / f'mini_boxes_nside_{cells_per_side}'
    if not box_dir.is_dir():
        raise FileNotFoundError(f"Minibox folder not found: {box_dir}")

    # Load all adjacent boxes. Miniboxes with nothing in them have no file,
    # and a file may hold seeds but no particles of this type; both are 
    # skipped. Any other read error is raised.
    for mini_box in mini_box_ids:
        file_name = box_dir / f'{mini_box}.hdf5'
        if not file_name.exists():
            continue
        with h5py.File(file_name, 'r') as hdf:
            if particle_type not in hdf:
                continue
            n_with_type += 1
            pos_i = hdf[f'{particle_type}/pos'][()]
            n_loaded += len(pos_i)
            mask = numpy.all(numpy.abs(_periodic_displacement(
                pos_i, center, boxsize)) <= padded_distance, axis=1)

            positions.append(pos_i[mask])
            velocities.append(hdf[f'{particle_type}/vel'][()][mask])
            ids.append(hdf[f'{particle_type}/ID'][()][mask])
                
            if hdf[f'{particle_type}/mass'].shape == ():
                masses = hdf[f'{particle_type}/mass'][()]
            else:
                masses.append(hdf[f'{particle_type}/mass'][()][mask])
    if n_with_type == 0:
        raise ValueError(
            f"No '{particle_type}' particles found in minibox {mini_box_id} or "
            f"its neighbors in {box_dir}; check particle_type and load_path."
        )

    # No rows in any file (same error as the original code, which raised it
    # from relative_coordinates on the concatenated array).
    if n_loaded == 0:
        raise ValueError("Input positions must contain at least one particle")

    # Concatenate all loaded data into single arrays
    positions = numpy.concatenate(positions)
    velocities = numpy.concatenate(velocities)
    ids = numpy.concatenate(ids)
    if isinstance(masses, list):
        masses = numpy.concatenate(masses)

    return positions, velocities, ids, masses


def load_seeds(
    mini_box_id: int,
    boxsize: float,
    minisize: float,
    load_path: str,
    seed_prop_names: Tuple[str] = ('M200b', 'R200b', 'Rs'),
    padding: float = 5.0,
) -> Tuple[numpy.ndarray]:
    """
    Load seeds from a mini-box and its adjacent neighbors within padding distance.

    This function loads seed data (positions, velocities, IDs, R200, M200, Rs, mask) 
    from a specified mini-box and all 26 neighboring mini-boxes, then filters seeds 
    to only include those within a specified padding distance from the target 
    mini-box edges. This is useful for analysis that requires particles near box 
    boundaries to avoid edge effects.

    An additional boolean mask is returned indicating which seeds are located in
    the target mini-box versus the neighboring boxes.

    The returned arrays are sorted in descending order by M200.

    Parameters
    ----------
    mini_box_id : int or numpy.integer
        ID of the target mini-box from which to load seeds. Must be a valid ID
        within the grid (0 ≤ mini_box_id < total_mini_boxes).
    boxsize : float
        Size of the cubic simulation box. Must be a positive.
    minisize : float  
        Size of each cubic mini-box. Must be positive and ≤ boxsize.
    load_path : str or Path
        Path to directory containing the mini-box HDF5 files. The directory
        should contain a subdirectory named 'mini_boxes_nside_<xx>/' where <xx>
        is the number of cells per side.
    seed_prop_names : Tuple[str], optional
        Tuple with three strings: mass, radius, and scale radius label names in 
        the mini-box HDF5 files, in case other names (e.g. Mvir, Rvir) were used. 
        Default is ('M200b', 'R200b', 'Rs').
    padding : float, optional
        Maximum distance from mini-box edges to include seeds. Seeds further 
        than this distance from any edge of the target mini-box will be 
        excluded. Must be non-negative. Default is 5.0.

    Returns
    -------
    positions : numpy.ndarray
        Seed positions with shape (n_seeds, 3) in cartesian coordinates of 
        particles within the padding distance.
    velocities : numpy.ndarray  
        Seed velocities with shape (n_seeds, 3) in cartesian velocity components
        corresponding to the returned positions.
    seed_ids : numpy.ndarray
        Seed IDs with shape (n_seeds,) containing unique identifiers 
        corresponding to the returned positions and velocities.
    r200 : numpy.ndarray
        R200 values with shape (n_seeds,) corresponding to the standard halo
        boundary defined by 200 times the critical density.
    m200 : numpy.ndarray
        M200 values with shape (n_seeds,) corresponding to the standard halo
        boundary defined by 200 times the critical density.
    rs : numpy.ndarray
        Scale radius values with shape (n_seeds,) corresponding to the NFW
        profile scale radius fit by Rockstar.
    mini_box_mask : numpy.ndarray
        Boolean mask with shape (n_seeds,) indicating which seeds are located
        in the target mini-box (True) versus neighboring boxes (False).

    Raises
    ------
    TypeError
        If mini_box_id is not an integer, or if boxsize, minisize or padding are
        not numeric, or load_path is not a string or Path.
    ValueError
        If mini_box_id is negative or invalid for the grid, boxsize or minisize 
        are non-positive, minisize > boxsize, or padding is negative.
    FileNotFoundError
        If load_path doesn't exist or required mini-box files are missing.
    NotADirectoryError
        If load_path is not a directory.
    OSError
        If HDF5 files cannot be read or are corrupted.
    ValueError
        If the mini-box and its neighbours contain no rows at all. (If rows
        exist but none lie within the padding, empty arrays are returned.)

    See Also
    --------
    get_adjacent_mini_box_ids : Get IDs of neighboring mini-boxes
    split_simulation_into_mini_boxes : Create mini-box files
    load_particles : Load particles from a given mini-box.

    Notes
    -----
    - Uses periodic boundary conditions when calculating relative coordinates
    - Loads from all 27 mini-boxes (target + 26 neighbors) to ensure complete
      coverage within padding distance
    - Memory usage scales with the number of particles in the 27 mini-boxes

    Examples
    --------
    >>> positions, velocities, ids, r200, m200, rs, mask = load_seeds(
    ...     mini_box_id=42,
    ...     boxsize=100.0,
    ...     minisize=10.0,
    ...     load_path="/data/simulation/",
    ...     seed_prop_names=('mvir', 'rvir', 'rs')
    ...     padding=2.0
    ... )
    >>> print(f"Loaded {mask.sum()} seeds from mini-box")
    """
    # Input validation
    _validate_inputs_load(mini_box_id, boxsize, minisize, load_path, padding)
    label_m200, label_r200, label_rs = seed_prop_names

    # Get the adjacent mini-box IDs. With fewer than 3 cells per side the same
    # neighbour appears several times; load each mini-box only once.
    mini_box_ids = _unique_in_order(get_adjacent_mini_box_ids(
        mini_box_id=mini_box_id,
        boxsize=boxsize,
        minisize=minisize
    ))
    
    # Determine number of partitions per side
    cells_per_side = int(numpy.ceil(boxsize / minisize))

    # Select seeds within a padding distance of the edge of the box in each
    # direction. The mask is applied to each file as it is read.
    center = _mini_box_center(mini_box_id, cells_per_side, minisize)
    padded_distance = 0.5 * minisize + padding

    # Create empty lists (containers) to save the data from file for each ID
    positions, velocities, ids, r200, m200, rs, mini_box_mask = (
        [] for _ in range(7))
    n_read = 0
    
    # Load all adjacent boxes
    for mini_box in mini_box_ids:
        file_name = load_path + f'mini_boxes_nside_{cells_per_side}/{mini_box}.hdf5'
        with h5py.File(file_name, 'r') as hdf:
            if not 'seed' in hdf.keys():
                continue
            pos_i = hdf['seed/pos'][()]
            n_read += len(pos_i)
            mask = numpy.all(numpy.abs(_periodic_displacement(
                pos_i, center, boxsize)) <= padded_distance, axis=1)
            positions.append(pos_i[mask])
            velocities.append(hdf['seed/vel'][()][mask])
            ids.append(hdf['seed/ID'][()][mask])
            r200.append(hdf[f'seed/{label_r200}'][()][mask])
            m200.append(hdf[f'seed/{label_m200}'][()][mask])
            rs.append(hdf[f'seed/{label_rs}'][()][mask])
            mini_box_mask.append(
                numpy.full(mask.sum(), mini_box == mini_box_id, dtype=bool))

    # Concatenate all loaded data into single arrays
    positions = numpy.concatenate(positions)
    velocities = numpy.concatenate(velocities)
    ids = numpy.concatenate(ids)
    r200 = numpy.concatenate(r200)
    m200 = numpy.concatenate(m200)
    rs = numpy.concatenate(rs)
    mini_box_mask = numpy.concatenate(mini_box_mask)

    # Same error as before when the seed datasets are empty.
    if n_read == 0:
        raise ValueError("Input positions must contain at least one particle")

    # Final validation
    n_loaded = len(positions)
    if n_loaded == 0:
        return (() for _ in range(7))

    # Sort seeds by M200 (largest first)
    argorder = numpy.argsort(-m200)
    m200 = m200[argorder]
    r200 = r200[argorder]
    positions = positions[argorder]
    velocities = velocities[argorder]
    ids = ids[argorder]
    rs = rs[argorder]
    mini_box_mask = mini_box_mask[argorder]

    return positions, velocities, ids, r200, m200, rs, mini_box_mask
