import numpy

from oasis.common import (_validate_coordinate_array,
                          _validate_positive_number_non_zero)


def relative_coordinates(
    positions: numpy.ndarray,
    reference: numpy.ndarray,
    boxsize: float,
    periodic: bool = True
) -> numpy.ndarray:
    """
    Compute positions relative to a reference point, with optional
    periodic boundary conditions.

    This function calculates the displacement vectors from a reference point
    to a set of positions, optionally applying the minimum image convention
    for periodic boundary conditions.

    Parameters
    ----------
    positions : numpy.ndarray
        Array of positions with shape (N, 3) or (3,) for a single position.
        If 1D array with shape (3,) is provided, it will be automatically
        reshaped to (1, 3). Each row represents the (x, y, z) coordinates
        of a particle.
    reference : numpy.ndarray
        Reference position with shape (3,) representing the (x, y, z)
        coordinates of the reference point.
    boxsize : float
        Size of the cubic simulation box. Must be a positive scalar.
    periodic : bool, default=True
        Whether to apply periodic boundary conditions using the minimum
        image convention. If True, particles are mapped to their nearest
        periodic images relative to the reference point.

    Returns
    -------
    numpy.ndarray
        Relative positions with shape (N, 3). Each row contains the
        displacement vector from the reference point to the corresponding
        particle, accounting for periodic boundary conditions if enabled.

    Raises
    ------
    ValueError
        If `positions` does not have shape (N, 3) or (3,), `reference` does not
        have shape (3,), or `boxsize` is not a positive scalar.
    TypeError
        If inputs cannot be converted to appropriate numpy arrays.

    Notes
    -----
    The minimum image convention ensures that the displacement vectors
    have the smallest possible magnitude in a periodic system. For a
    cubic box of size L, displacements are wrapped to the interval
    [-L/2, L/2) in each dimension.

    Examples
    --------
    Multiple positions:
    >>> positions = numpy.array([[0.1, 0.2, 0.3], [1.1, 0.8, 0.7]])
    >>> reference = numpy.array([0.5, 0.5, 0.5])
    >>> rel_pos = relative_coordinates(positions, reference, 1.0)
    >>> print(rel_pos)
    [[-0.4 -0.3 -0.2]
     [ 0.6  0.3  0.2]]

    Single position (1D input):
    >>> single_pos = numpy.array([0.1, 0.2, 0.3])
    >>> reference = numpy.array([0.5, 0.5, 0.5])
    >>> rel_pos = relative_coordinates(single_pos, reference, 1.0)
    >>> print(rel_pos)
    [[-0.4 -0.3 -0.2]]

    Without periodic boundary conditions:
    >>> rel_pos_no_pbc = relative_coordinates(positions, reference, 1.0, periodic=False)
    >>> print(rel_pos_no_pbc)
    [[-0.4 -0.3 -0.2]
     [ 0.4  0.3  0.2]]
    """
    # Input validation and type conversion
    _validate_coordinate_array(positions, "positions")
    _validate_coordinate_array(reference, "reference")
    _validate_positive_number_non_zero(boxsize, "boxsize")

    # Ensure positions is 2D array with shape (N, 3)
    if positions.ndim == 1:
        positions = positions.reshape(1, 3)

    # Compute displacement vectors
    displacement = positions - reference

    # Apply periodic boundary conditions using minimum image convention
    if periodic:
        half_box = 0.5 * boxsize
        displacement = (displacement + half_box) % boxsize - half_box

    return displacement


def _periodic_displacement(
    positions: numpy.ndarray,
    reference: numpy.ndarray,
    boxsize: float,
) -> numpy.ndarray:
    """Unvalidated version of `relative_coordinates` (periodic=True) for hot
    loops. Performs exactly the same arithmetic, so results are identical.
    `positions` must already have shape (N, 3)."""
    half_box = 0.5 * boxsize
    return (positions - reference + half_box) % boxsize - half_box


def velocity_components(
    positions: numpy.ndarray,
    velocities: numpy.ndarray
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Compute radial and tangential velocity components from Cartesian coordinates.

    This function decomposes velocity vectors into radial (along the position
    vector) and tangential (perpendicular to position vector) components.

    Parameters
    ----------
    positions : numpy.ndarray
        Array of Cartesian coordinates with shape (N, 3) or (3,) for a single
        position. If 1D array with shape (3,) is provided, it will be automatically
        reshaped to (1, 3). Each row represents the (x, y, z) coordinates of a 
        single particle.
    velocities : numpy.ndarray
        Array of Cartesian velocities with shape (N, 3) or (3,) for a single
        velocity. If 1D array with shape (3,) is provided, it will be automatically
        reshaped to (1, 3). Each row represents the (vx, vy, vz) velocity components
        of a single particle.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        A tuple containing:
        - radial_velocity : numpy.ndarray with shape (N,)
            Component of velocity along the radial direction (positive = outward).
        - tangential_velocity : numpy.ndarray with shape (N,)
            Magnitude of velocity component perpendicular to the radial direction.
        - velocity_squared : numpy.ndarray with shape (N,)
            Total velocity magnitude squared (vx^2 + vy^2 + vz^2).

    Raises
    ------
    ValueError
        If `positions` or `velocities` do not have shape (N, 3) or (3,),
        or if arrays have incompatible shapes.
    TypeError
        If inputs cannot be converted to appropriate numpy arrays.

    Notes
    -----
    The radial velocity is computed as \vec{v} \cdot \hat{r}, where \hat{r} is 
    the unit vector pointing from the origin to the particle position. The 
    tangential velocity magnitude is computed as \sqrt(v^2 - v_r^2), ensuring 
    numerical stability by clamping negative values to zero.

    For particles exactly at the origin (r = 0), the radial velocity is set
    to zero and the tangential velocity equals the total velocity magnitude.

    Examples
    --------
    Multiple particles:
    >>> positions = numpy.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    >>> velocities = numpy.array([[0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]])
    >>> vr, vt, v2 = velocity_components(positions, velocities)
    >>> print(f"Radial velocities: {vr}")
    >>> print(f"Tangential velocities: {vt}")
    Radial velocities: [0.5 -0.5]
    Tangential velocities: [0.5  0.5]

    Single particle (1D input):
    >>> single_pos = numpy.array([1.0, 1.0, 0.0])
    >>> single_vel = numpy.array([0.0, 1.0, 0.0])
    >>> vr, vt, v2 = velocity_components(single_pos, single_vel)
    >>> print(f"Radial velocity: {vr[0]:.3f}")
    >>> print(f"Tangential velocity: {vt[0]:.3f}")
    Radial velocity: 0.707
    Tangential velocity: 0.707

    With radial motion from origin:
    >>> positions = numpy.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    >>> velocities = numpy.array([[0.5, 0.0, 0.0], [0.0, -1.0, 0.0]])
    >>> vr, vt, v2 = velocity_components(positions, velocities)
    >>> print(f"Radial velocities: {vr}")
    >>> print(f"Tangential velocities: {vt}")
    Radial velocities: [ 0.5 -1.0]
    Tangential velocities: [0.0 0.0]
    """
    # Input validation and type conversion
    _validate_coordinate_array(positions, "positions")
    _validate_coordinate_array(velocities, "velocities")

    # Ensure positions and velocities are 2D arrays with shape (N, 3)
    if positions.ndim == 1:
        positions = positions.reshape(1, 3)
    if velocities.ndim == 1:
        velocities = velocities.reshape(1, 3)
    
    # Check that positions and velocities have the same shape
    if positions.shape != velocities.shape:
        raise ValueError(f"`positions` and `velocities` must have the same shape, "
                         f"got {positions.shape} and {velocities.shape}.")

    # Compute radial distances from origin
    radial_distances_squared = numpy.sum(positions**2, axis=1)
    radial_distances = numpy.sqrt(radial_distances_squared)

    # Initialize output arrays
    num_particles = positions.shape[0]
    radial_velocity = numpy.zeros(num_particles)
    tangential_velocity = numpy.zeros(num_particles)
    velocity_squared = numpy.sum(velocities**2, axis=1)

    # For particles at the origin (r = 0), radial velocity is 0
    # and tangential velocity equals total velocity magnitude
    zero_mask = radial_distances == 0
    if numpy.any(zero_mask):
        tangential_velocity[zero_mask] = numpy.sqrt(
            velocity_squared[zero_mask])
    
    # Handle particles not at the center (r > 0)
    nonzero_mask = ~zero_mask
    
    # Compute unit radial vectors for non-zero positions
    radial_unit_vectors = positions[nonzero_mask] / \
        radial_distances[nonzero_mask, numpy.newaxis]

    # Compute radial velocity as \vec{v} \cdot \hat{r}
    radial_velocity[nonzero_mask] = numpy.sum(
        velocities[nonzero_mask] * radial_unit_vectors, axis=1
    )

    # Compute tangential velocity magnitude as \sqrt(v^2 - v_r^2)
    radial_velocity_squared = radial_velocity[nonzero_mask]**2
    # Clamp to avoid numerical issues with floating point precision
    tangential_velocity_squared = numpy.maximum(
        0, velocity_squared[nonzero_mask] - radial_velocity_squared
    )
    tangential_velocity[nonzero_mask] = numpy.sqrt(
        tangential_velocity_squared)

    return radial_velocity, tangential_velocity, velocity_squared
