import os
from functools import partial
from multiprocessing import Pool
from warnings import filterwarnings

import h5py
import numpy
import pandas
from scipy.optimize import fsolve
from scipy.spatial import cKDTree
from tqdm import tqdm

from oasis.common import G_GRAVITY, ensure_dir_exists
from oasis.coordinates import _periodic_displacement
from oasis.datasource import LegacyMiniBoxDataSource, SpatialDataSource

filterwarnings('ignore')


class MiniBoxClassifier:
    """
    Parallel-friendly wrapper for classify_single_mini_box().
    Each instance handles exactly == one mini box ==.
    """

    def __init__(
        self,
        mini_box_id: int,
        min_num_part: int,
        boxsize: float,
        minisize: float,
        load_path: str,
        run_name: str,
        particle_type: str,
        seed_prop_names: tuple[str],
        redshift: float,
        padding: float = 5.0,
        fast_mass: bool = False,
        disable_tqdm: bool = True,
        data_source: SpatialDataSource | None = None,
    ):
        # Input parameters
        self.mini_box_id = mini_box_id
        self.min_num_part = min_num_part
        self.boxsize = boxsize
        self.minisize = minisize
        self.load_path = load_path
        self.run_name = run_name
        self.particle_type = particle_type
        self.padding = padding
        self.seed_prop_names = seed_prop_names
        self.redshift = redshift
        self.fast_mass = fast_mass
        self.disable_tqdm = disable_tqdm

        # Where seeds and particles come from. By default, the current minibox
        # files, so existing scripts behave as before.
        if data_source is None:
            data_source = LegacyMiniBoxDataSource(
                load_path=load_path, boxsize=boxsize, minisize=minisize,
                particle_type=particle_type, seed_prop_names=seed_prop_names,
                padding=padding,
            )
        if data_source.boxsize != boxsize:
            raise ValueError(
                f"data_source.boxsize ({data_source.boxsize}) must equal the"
                f" global boxsize ({boxsize})"
            )
        self.data_source = data_source

        # Internal parameters
        self.save_path = None
        # Seed properties
        self.pos_seed = self.vel_seed = self.hid = None
        self.r200b = self.m200b = self.rs = self.mask_mb = None
        # Particle properties
        self.pos_part = self.vel_part = self.pid_part = self.mass_part = None
        # Classification parameters
        self.pars = None
        self.deltac = None
        self.n_seeds = 0

    # ==========================================================================
    # Phase 1: Setup and loading
    # ==========================================================================
    def _make_output_dir(self):
        self.save_path = self.load_path + f"run_{self.run_name}/mini_box_catalogues/"
        ensure_dir_exists(self.save_path)

    def _load_seeds_and_filter(self):
        seeds = self.data_source.load_seeds(self.mini_box_id)

        # Nothing else to do if there are no seeds around this mini-box.
        if len(seeds) == 0:
            self.n_seeds = 0
            return

        # Process seeds in descending M200b order. The stable sort keeps the
        # order of equal masses exactly as the data source gave them, so it 
        # changes nothing for the legacy source (already sorted) and is 
        # deterministic for any other source. 
        order = numpy.argsort(-seeds.m200b, kind = 'stable')
        self.pos_seed = seeds.pos[order]
        self.vel_seed = seeds.vel[order]
        self.hid = seeds.hid[order]
        self.r200b = seeds.r200b[order]
        self.m200b = seeds.m200b[order]
        self.rs = seeds.rs[order]
        self.mask_mb = seeds.in_core[order]

        if self.fast_mass:
            m200b_mask = self.m200b > (5.0 * numpy.min(self.m200b))

            self.pos_seed = self.pos_seed[m200b_mask]
            self.vel_seed = self.vel_seed[m200b_mask]
            self.hid = self.hid[m200b_mask]
            self.r200b = self.r200b[m200b_mask]
            self.m200b = self.m200b[m200b_mask]
            self.rs = self.rs[m200b_mask]
            self.mask_mb = self.mask_mb[m200b_mask]

        self.n_seeds = len(self.hid)

    def _early_exit_if_no_seeds(self):
        # Test the number of seeds.
        return self.n_seeds == 0

    def _compute_deltac(self):
        """Compute the characteristic density of an NFW profile.

        This method calculates the dimensionless characteristic density (delta_c)
        for Navarro-Frenk-White (NFW) density profiles based on the concentration
        parameter c200 = r200/rs. The characteristic density relates to the density
        normalization of the NFW profile.

        Parameters
        ----------
        None
            Uses instance attributes `self.r200b` and `self.rs`.

        Returns
        -------
        None
            Sets `self.deltac` (numpy.ndarray) with the computed characteristic
            densities for each halo.

        Raises
        ------
        ZeroDivisionError
            If any element in `self.r200b` or `self.rs` is zero.

        Notes
        -----
        The characteristic density is computed as:
        
            delta_c = (200/3) * c^3 / (ln(1+c) - c/(1+c))
        
        where c = r200/rs is the concentration parameter.

        See Also
        --------
        _rho_nfw_roots : Computes NFW density profile intersections.
        """
        if numpy.any(self.rs == 0.) or numpy.any(self.r200b == 0.):
            raise ZeroDivisionError('Neither r200 nor rs can be zero.')

        c200 = self.r200b / self.rs
        self.deltac = (200./3.) * (c200 ** 3 / (numpy.log(1 + c200) - (c200 / (1 + c200))))

    def _build_seed_tree(self):
        """Build a periodic KD-tree over the seed positions once per mini-box.

        Used by `_apply_6d_ball` to find candidate neighbouring seeds instead
        of computing distances to every seed for every halo. Candidate searches
        use a radius enlarged by `self._search_margin`; the original float32
        distance cut is then applied to the candidates, so the selected seeds
        (and particles) are exactly the same as with a brute-force search.
        """
        self._search_margin = 1e-5 * self.boxsize
        self.seed_tree = cKDTree(
            numpy.mod(self.pos_seed.astype(numpy.float64), self.boxsize),
            boxsize=self.boxsize,
        )

    @staticmethod
    def _rho_nfw_roots(
        x: float,
        delta1: float,
        rs1: float,
        delta2: float,
        rs2: float,
        r12: float,
    ):
        """Find the radius where two NFW density profiles are equal.

        This function computes the difference between two NFW density profiles
        at a given radius, used to find their intersection point via root-finding.
        The NFW density profile is given by:
        
            rho(r)/rho_c = delta_c / [(r/R_s)(1 + r/R_s)^2]

        Parameters
        ----------
        x : float
            Radial coordinate at which to evaluate the density difference, 
            measured from the center of the second object.
        delta1 : float
            Characteristic density of the first (central) object.
        rs1 : float
            Scale radius of the first (central) object.
        delta2 : float
            Characteristic density of the second (substructure) object.
        rs2 : float
            Scale radius of the second (substructure) object.
        r12 : float
            Radial separation between the centers of the two objects.

        Returns
        -------
        float
            Difference between the first and second NFW density profiles at
            radius x. Returns zero at the intersection point.

        Notes
        -----
        The function evaluates rho_1(R-x)/rho_c - rho_2(x)/rho_c, where R is
        the separation r12. This is designed to be used with scipy.optimize.fsolve
        to find the radius where the two density profiles intersect.

        See Also
        --------
        _apply_6d_ball : Uses this function to define search radii.
        scipy.optimize.fsolve : Root-finding algorithm typically used with this function.
        """
        x1 = (r12 - x) / rs1
        x2 = x / rs2
        frac1 = delta1 / (x1 * (1 + x1)**2)
        frac2 = delta2 / (x2 * (1 + x2)**2)
        return frac1 - frac2

    def _load_particles(self):
        particles = self.data_source.load_particles(self.mini_box_id)
        self.pos_part = particles.pos
        self.vel_part = particles.vel
        self.pid_part = particles.pid
        self.mass_part = particles.mass

        # Converting particle velocity from v_sim to v_phys
        a = 1.0 / (1.0 + self.redshift)
        self.vel_part *= numpy.sqrt(a)

        # Sliding-midpoint trees build much faster than median-balanced trees
        # and return the same query results.
        self.position_tree = cKDTree(self.pos_part, boxsize=self.boxsize,
                                     balanced_tree=False, compact_nodes=False)

    def _load_calibration_parameters(self):
        with h5py.File(self.load_path + "calibration_pars.hdf5", "r") as hdf:
            self.pars = (
                *hdf["pos"][()],
                *hdf["neg/line"][()],
                *hdf["neg/quad"][()],
            )

    def _init_catalogue_dataframe(self):
        col_names = (
            "Halo_ID",
            "pos",
            "vel",
            "R200b",
            "M200b",
            "Morb",
            "Norb",
            "LIDX",
            "RIDX",
            "INMB",
            "NSUBS",
            "PID",
            "SLIDX",
            "SRIDX",
        )
        
        # Empty dataframe with column names
        self.haloes = pandas.DataFrame(columns=col_names)
        self.haloes_perc = pandas.DataFrame(columns=col_names)

        # Members are stored as local indices into the loaded particle/seed
        # arrays (not IDs) so percolation can use Boolean "seen" arrays.
        self.orb_sidx = []   # seed indices of orbiting substructures
        self.orb_pidx = []   # particle indices of orbiting particles
        self.orb_mass = []   # only filled for variable-mass particles
        self.n_tot_p = 0
        self.n_tot_s = 0

        # If not -1 then seed is an orbiting structure
        self.parent_id_seed = numpy.full(self.n_seeds, -1, dtype=int)

    # ==========================================================================
    @staticmethod
    def _classify(
        rel_pos: numpy.ndarray,
        rel_vel: numpy.ndarray,
        redshift: float,
        r200: float,
        m200: float,
        class_pars: list | tuple | numpy.ndarray,
        max_radius: float = 2.0,
        pivot_radius: float = 0.5
    ) -> numpy.ndarray:
        """Classify particles as orbiting or infalling based on kinematics.

        This method uses a phase-space classification scheme that separates
        orbiting particles from infalling particles based on their positions
        and velocities relative to a halo center. The classification boundary
        transitions from linear to quadratic at the pivot radius.

        Parameters
        ----------
        rel_pos : numpy.ndarray
            Relative positions of particles with respect to the halo center,
            shape (N, 3). Each row contains (x, y, z) displacement components.
        rel_vel : numpy.ndarray
            Relative velocities of particles with respect to the halo center,
            shape (N, 3). Each row contains (vx, vy, vz) velocity components.
        redshift : float
            Redshift of the simulation. 
        r200 : float
            Overdensity radius of the halo (R200), defining the boundary where 
            the mean enclosed density is 200 times the critical density.
        m200 : float
            Overdensity mass of the halo (M200), the mass enclosed within R200.
        class_pars : list, tuple, or numpy.ndarray
            Classification parameters [m_pos, b_pos, m_neg, b_neg, alpha, beta, gamma]
            defining the orbiting/infalling boundary in phase space. These parameters
            are typically obtained from calibration procedures.
        max_radius : float, default=2.0
            Maximum radius (in units of r200) where orbiting particles can be found.
            All particles beyond this radius are classified as infalling.
        pivot_radius : float, default=0.5
            Transition radius (in units of r200) where the classification boundary
            changes from linear to quadratic form.

        Returns
        -------
        numpy.ndarray
            Boolean array of shape (N,) where True indicates an orbiting particle
            and False indicates an infalling particle.

        Notes
        -----
        The classification uses normalized coordinates where radii are expressed
        in units of r200 and velocities squared are in units of v200^2, where
        v200 = sqrt(G*m200/r200).

        For particles with positive radial velocity (moving outward):
            Classification uses a linear boundary in (r/r200, ln(v^2/v200^2)) space.

        For particles with negative radial velocity (moving inward):
            - Below pivot_radius: quadratic boundary
            - Above pivot_radius: linear boundary

        Examples
        --------
        >>> rel_pos = np.array([[0.1, 0.2, 0.3], [0.5, 0.6, 0.7]])
        >>> rel_vel = np.array([[-1.0, 0.5, 0.2], [0.8, -0.3, 0.1]])
        >>> redshift = 0.0
        >>> r200 = 1.0
        >>> m200 = 1e12
        >>> pars = [1.5, 0.2, 1.0, 0.1, 2.0, 1.0, 0.5]
        >>> mask = MiniBoxClassifier._classify(rel_pos, rel_vel, r200, m200, pars)
        >>> print(mask)
        [True False]
        """
        m_pos, b_pos, m_neg, b_neg, alpha, beta, gamma = class_pars

        #Compute comoving v200^2 if r200 is comoving
        v200_sq = G_GRAVITY * m200 / r200

        a = 1 / (1 + redshift)

        # switching to physical units, r200_phys = a * r200
        # rel_vel is in physical units already
        v200phys_sq = v200_sq / a

        # Compute the radius to seed_i in r200 units, and ln(v^2) in v200 units
        part_ln_vel = numpy.log(numpy.sum(numpy.square(rel_vel), axis=1) / v200phys_sq)
        part_radius = numpy.sqrt(numpy.sum(numpy.square(rel_pos), axis=1)) / r200

        # Create a mask for particles with positive radial velocity
        mask_vr_positive = numpy.sum(rel_vel * rel_pos, axis=1) > 0

        # Orbiting classification for vr > 0
        line = m_pos * (part_radius - pivot_radius) + b_pos
        mask_cut_pos = part_ln_vel < line

        # Orbiting classification for vr < 0
        mask_small_radius = part_radius <= pivot_radius
        curve = alpha * part_radius ** 2 + beta * part_radius + gamma
        line = m_neg * (part_radius - pivot_radius) + b_neg

        mask_cut_neg = ((part_ln_vel < curve) & mask_small_radius) ^ \
            ((part_ln_vel < line) & ~mask_small_radius)

        # Particle is infalling if it is below both lines and 2*R00
        mask_orb = (part_radius <= max_radius) & (
            (mask_cut_pos & mask_vr_positive) ^
            (mask_cut_neg & ~mask_vr_positive)
        )

        return mask_orb


    def _classify_particles(self, i: int):
        """Classify particles around a single seed halo candidate.

        This method identifies particles within 2*R200b of a seed and classifies
        them as either orbiting or infalling using phase-space criteria. Seeds
        with insufficient orbiting particles are rejected.

        Parameters
        ----------
        i : int
            Index of the seed halo candidate in the internal seed arrays
            (self.pos_seed, self.vel_seed, etc.).

        Returns
        -------
        tuple or None
            If the seed qualifies as a halo (has at least `self.min_num_part`
            orbiting particles), returns a tuple containing:
            
            - within_r200b : numpy.ndarray
                Indices of particles within 2*R200b of the seed.
            - orb_mask : numpy.ndarray
                Boolean mask indicating which particles in `within_r200b` are
                classified as orbiting.
            
            Returns None if the seed does not meet the minimum particle threshold.

        Notes
        -----
        The method performs the following steps:
        1. Queries the spatial tree for particles within 2*R200b
        2. Computes relative positions and velocities
        3. Applies the phase-space classification criterion
        4. Checks if the number of orbiting particles meets the minimum threshold

        The factor of 2*R200b ensures sufficient volume for detecting orbiting
        particles while maintaining computational efficiency.

        See Also
        --------
        _classify : Core classification algorithm for phase-space cuts.
        _apply_6d_ball : Subsequent processing step to handle substructures.
        """
        # Select all particles around the seed
        # Convert the index list to an array once; indexing with a Python list
        # converts it again every time it is used.
        within_r200b = numpy.asarray(
            self.position_tree.query_ball_point(self.pos_seed[i],
                                                2.*self.r200b[i],
                                                p=numpy.inf,
                                                return_sorted=True),
            dtype=numpy.intp,
        )
        # Skip if not enough particles within R200b
        if len(within_r200b) < self.min_num_part:
            return None

        # Relative coordinates of particles w.r.t seed position
        rel_pos = _periodic_displacement(self.pos_part[within_r200b],
                                         self.pos_seed[i], self.boxsize)

        # Relative velocity of particles w.r.t seed peculiar velocity 
        # in physical units.
        rel_vel = self.vel_part[within_r200b] - self.vel_seed[i]

        # Classify particles around the seed.
        orb_mask = self._classify(rel_pos, rel_vel, self.redshift,self.r200b[i], self.m200b[i],
        self.pars)
        
        # Ignore seed if it does not have the minimum mass to be considered a
        # halo. Early exit to avoid further computation for a non-halo seed.
        if orb_mask.sum() < self.min_num_part:
            return None

        return (within_r200b, orb_mask)

    def _apply_6d_ball(
        self, 
        i: int, 
        within_r200b: numpy.ndarray, 
        orb_mask: numpy.ndarray,
    ):
        """Apply 6D phase-space ball criterion to identify orbiting substructures.

        This method identifies nearby less-massive seed halos that are orbiting
        within the current halo's potential. It uses a 6D ball (3D position + 3D
        velocity) centered on each nearby seed to determine if it and its particles
        should be classified as a orbiting.

        Parameters
        ----------
        i : int
            Index of the current seed halo candidate.
        within_r200b : numpy.ndarray
            Indices of particles within 2*R200b of the seed (from _classify_particles).
        orb_mask : numpy.ndarray
            Boolean mask indicating which particles in `within_r200b` are initially
            classified as orbiting.

        Returns
        -------
        tuple, int, or None
            - If the seed remains a halo with substructures, returns a tuple:
            (orb_mask_new, orb_seeds) where orb_mask_new is the updated boolean
            mask and orb_seeds is an array of seed indices (into the local seed
            arrays) classified as substructures.
            - If no nearby seeds are found, returns 0.
            - If the seed fails to meet minimum particle requirements after
            processing, returns None.

        Notes
        -----
        The 6D ball search radius is defined by:
        - Spatial radius: minimum of (r_ball, R200b_nearby), where r_ball is the
        radius where NFW density profiles of the two halos intersect
        - Velocity radius: v_ball = 2 * sqrt(G*M200b_nearby/R200b_nearby)

        A nearby seed is classified as orbiting if the fraction of orbiting particles
        within its 6D ball exceeds a threshold:
        
            f_threshold = max(0.5, 1 - exp(-(r_ij/R200b_nearby)^2))

        This adaptive threshold increases for more distant seeds to reduce
        false positives.

        The method updates `self.parent_id_seed` to track which seeds have been
        identified as substructures.

        See Also
        --------
        _rho_nfw_roots : Computes NFW profile intersection radius.
        _classify_particles : Initial particle classification step.
        """
        # Only work with less massive seeds within a 2*R200b sphere. Candidate
        # seeds come from the seed KD-tree (slightly enlarged radius), then the
        # exact same float32 distance cut as before is applied to them.
        r_max = 2.0 * self.r200b[i]
        candidates = numpy.sort(numpy.asarray(
            self.seed_tree.query_ball_point(self.pos_seed[i],
                                            r_max + self._search_margin),
            dtype=numpy.intp,
        ))
        candidates = candidates[candidates > i]
        candidates = candidates[self.parent_id_seed[candidates] == -1]
        if candidates.size == 0:
            return 0

        rel_pos = _periodic_displacement(self.pos_seed[candidates],
                                         self.pos_seed[i], self.boxsize)
        mask_seed = numpy.sum(numpy.square(rel_pos), axis=1) <= r_max**2

        # Skip if not enough seeds within 2*R200b
        n_seeds_near = mask_seed.sum()
        if n_seeds_near <= 0:
            return 0

        # Select seeds in the vicinity (ascending index = descending mass)
        near_idx = candidates[mask_seed]
        rel_pos = rel_pos[mask_seed]
        pos_near = self.pos_seed[near_idx]
        vel_near = self.vel_seed[near_idx]
        deltac_near = self.deltac[near_idx]
        r200b_near = self.r200b[near_idx]
        rs_near = self.rs[near_idx]

        # Defines the search velocity of the 6D ball, converted to physical
        # units. Same element-wise arithmetic as the per-seed scalar version.
        a = 1 / (1 + self.redshift)
        v_ball_sq_phys_near = 2**2 * G_GRAVITY * self.m200b[near_idx] / r200b_near / a

        # Loop over other seeds
        j = 0
        is_halo = True
        orb_seeds = []
        orb_mask_new = numpy.copy(orb_mask)
        n_orb = int(numpy.count_nonzero(orb_mask_new))

        while is_halo and (j < n_seeds_near):
            # Distance from the current seed to the substructure.
            r_ij = numpy.linalg.norm(rel_pos[j])
            
            # Defines the search radius of the 6D ball. Distance from the 
            # substructure where the NFW density of both objects is equal.
            r_ball = fsolve(
                func=self._rho_nfw_roots,
                # Start at half the distance bewteen seeds.
                x0=0.5*r_ij,
                args=(
                    self.deltac[i], 
                    self.rs[i], 
                    deltac_near[j], 
                    rs_near[j], 
                    r_ij
                ))
            # Kept as a float64 numpy scalar so the comparison against float32
            # squared distances is done in float64, as before.
            r_ball = numpy.float64(min(float(r_ball[0]), float(r200b_near[j])))

            # Particles (positions into within_r200b) inside the 6D ball.
            ball6d = self._particles_in_6d_ball(
                within_r200b, pos_near[j], vel_near[j], r_ball,
                v_ball_sq_phys_near[j],
            )

            # Compare to the original orbiting population.
            n_ball = ball6d.size
            n_in = int(numpy.count_nonzero(orb_mask_new[ball6d]))
            frac_inside = n_in / n_ball if n_ball > 0 else numpy.nan

            # If more than half the particles in the vicinity of the seed are 
            # orbiting, the seed is tagged as orbiting. The seed is infalling 
            # otherwise and all the particles within the box are tagged as 
            # infalling too.
            upper_threshold = 1. - numpy.exp(-(r_ij / r200b_near[j])**2)
            f_threshold = max(0.5, float(upper_threshold))

            if frac_inside >= f_threshold:
                orb_seeds.append(near_idx[j])
                orb_mask_new[ball6d] = True
                n_orb += n_ball - n_in
            else:
                orb_mask_new[ball6d] = False
                n_orb -= n_in

            # Check wether seed is still a halo.
            is_halo = n_orb >= self.min_num_part

            # Next item.
            j += 1

        if is_halo:
            return (orb_mask_new, numpy.array(orb_seeds, dtype=numpy.intp))
        else:
            return None

    def _particles_in_6d_ball(
        self,
        within_r200b: numpy.ndarray,
        center_pos: numpy.ndarray,
        center_vel: numpy.ndarray,
        r_ball: numpy.float64,
        v_ball_sq: numpy.floating,
    ) -> numpy.ndarray:
        """Return positions in `within_r200b` of the particles inside a 6D ball.

        Equivalent to evaluating `(r^2 <= r_ball^2) & (v^2 <= v_ball_sq)` for
        every particle in `within_r200b` (which must be sorted), but only the
        particles returned by a spatial query of the particle KD-tree are
        evaluated. The query radius is enlarged by `self._search_margin` and
        the original cut is applied afterwards, so the result is identical.
        """
        r_ball_sq = r_ball**2
        if numpy.isnan(r_ball_sq):
            return numpy.empty(0, dtype=numpy.intp)

        if numpy.isinf(r_ball_sq):
            loc = numpy.arange(within_r200b.size)
        else:
            candidates = numpy.asarray(
                self.position_tree.query_ball_point(
                    center_pos, abs(r_ball) + self._search_margin),
                dtype=numpy.intp,
            )
            # Keep only candidates that are also within 2*R200b of the host.
            loc = numpy.searchsorted(within_r200b, candidates)
            loc[loc == within_r200b.size] = 0
            loc = loc[within_r200b[loc] == candidates]

        idx = within_r200b[loc]
        rel_pos_part = _periodic_displacement(self.pos_part[idx], center_pos,
                                              self.boxsize)
        rel_vel_part = self.vel_part[idx] - center_vel
        rp_sq = numpy.sum(numpy.square(rel_pos_part), axis=1)
        vp_sq = numpy.sum(numpy.square(rel_vel_part), axis=1)
        return loc[(rp_sq <= r_ball_sq) & (vp_sq <= v_ball_sq)]

    def _classify_single_seed(self, i: int):
        """Execute complete classification scheme for a single seed halo.

        This method orchestrates the full halo identification pipeline for one
        seed, including particle classification, substructure detection, and
        catalog entry creation.

        Parameters
        ----------
        i : int
            Index of the seed halo candidate in the internal seed arrays.

        Returns
        -------
        dict or None
            If the seed qualifies as a halo, returns a dictionary containing:
            
            - Halo_ID : int
                Unique halo identifier.
            - pos : numpy.ndarray
                3D position of the halo center.
            - vel : numpy.ndarray
                3D velocity of the halo center.
            - R200b : float
                Overdensity radius.
            - M200b : float
                Overdensity mass.
            - Morb : float
                Total orbiting mass.
            - Norb : int
                Number of orbiting particles.
            - LIDX : int
                Left index for particles in packed arrays.
            - RIDX : int
                Right index for particles in packed arrays.
            - INMB : bool
                Whether the halo center is within the current mini-box.
            - NSUBS : int
                Number of identified substructures.
            - PID : int
                Parent halo ID (-1 if no parent).
            - SLIDX : int
                Left index for substructures in packed arrays.
            - SRIDX : int
                Right index for substructures in packed arrays.
            
            Returns None if the seed fails any classification stage.

        Notes
        -----
        The method executes three stages:
        1. Particle classification (_classify_particles)
        2. Substructure detection (_apply_6d_ball)
        3. Catalog entry creation with proper indexing

        The method updates several instance attributes:
        - self.parent_id_seed: Tracks hierarchical relationships
        - self.orb_sidx, self.orb_pidx, self.orb_mass: Accumulate member data
          (seed indices, particle indices, and masses for variable-mass only)
        - self.n_tot_p, self.n_tot_s: Update global counters for indexing

        See Also
        --------
        _classify_particles : First stage of classification.
        _apply_6d_ball : Second stage identifying substructures.
        _process_all_seeds : Calls this method for all seeds.
        """
        # Classify particles around the seed
        result_1 = self._classify_particles(i=i)
        if result_1 is None:
            return None
        within_r200b, orb_mask = result_1

        # Classify seeds in the vicinity as orbiting structures
        result_2 = self._apply_6d_ball(i, within_r200b, orb_mask)
        if result_2 == 0:
            orb_mask_final = orb_mask
            n_subs = 0
            sub_idx = numpy.empty(0, dtype=numpy.intp)
        elif result_2 is None:
            return None
        else:
            # Un-pack results. sub_idx are the seed indices of the orbiting
            # substructures (seed IDs are assumed unique).
            orb_mask_final, sub_idx = result_2
            
            # Set parent halo ID for seeds (these are no longer free).
            self.parent_id_seed[sub_idx] = self.hid[i]
            n_subs = sub_idx.size

        # Compute orbiting mass, and append orbiting objects to global lists
        n_orb = orb_mask_final.sum()
        orb_idx = within_r200b[orb_mask_final]
        if isinstance(self.mass_part, numpy.ndarray):
            orb_mass = self.mass_part[orb_idx]
            morb = numpy.sum(orb_mass)
            self.orb_mass.append(orb_mass)
        else:
            # Constant particle mass: no per-particle mass array is needed.
            morb = n_orb * self.mass_part
        
        self.orb_sidx.append(sub_idx)
        self.orb_pidx.append(orb_idx)
        
        # Build the row
        row = dict(
            Halo_ID=int(self.hid[i]),
            pos=self.pos_seed[i],
            vel=self.vel_seed[i],
            R200b=float(self.r200b[i]),
            M200b=float(self.m200b[i]),
            Morb=float(morb),
            Norb=int(n_orb),
            LIDX=int(self.n_tot_p),
            RIDX=int(self.n_tot_p + n_orb),
            INMB=bool(self.mask_mb[i]),
            NSUBS=int(n_subs),
            PID=self.parent_id_seed[i],
            SLIDX=int(self.n_tot_s),
            SRIDX=int(self.n_tot_s + n_subs),
        )

        # Update running counters for packed outputs
        self.n_tot_p += n_orb
        self.n_tot_s += n_subs

        return row

    def _process_all_seeds(self):
        """Execute classification pipeline for all seeds in the mini-box.

        This method iterates through all seed halo candidates, applies the
        complete classification scheme to each, and assembles the results into
        a catalog sorted by orbiting mass.

        Parameters
        ----------
        None
            Operates on instance attributes initialized during setup.

        Returns
        -------
        None
            Updates instance attributes:
            
            - self.haloes : pandas.DataFrame
                Catalog of identified halos sorted by orbiting mass (descending).
            - self.orb_pidx : numpy.ndarray
                Concatenated local indices of all orbiting particles.
            - self.orb_sidx : numpy.ndarray
                Concatenated local indices of all identified substructures.
            - self.orb_mass : numpy.ndarray
                Concatenated particle masses of all orbiting particles (only
                for variable-mass particles).

        Notes
        -----
        The method uses a progress bar (tqdm) unless disabled via the
        `disable_tqdm` parameter during initialization.

        Failed seed classifications (returning None from _classify_single_seed)
        are silently skipped and do not appear in the final catalog.

        The resulting catalog is sorted by orbiting mass to prioritize more
        massive halos during the percolation stage, ensuring particles are
        assigned to their most massive host first.

        See Also
        --------
        _classify_single_seed : Classifies individual seeds.
        _percolation : Subsequent processing to handle particle uniqueness.
        """
        results = []
        for i in tqdm(range(self.n_seeds), ncols=100, desc='Finding haloes',
                      colour='green', disable=self.disable_tqdm):
            row = self._classify_single_seed(i)
            if row is not None:
                results.append(row)
                

        if results:
            df = pandas.DataFrame(results)
            self.haloes = pandas.concat([self.haloes, df], ignore_index=True)
        
        # Sort haloes by their orbiting mass
        self.haloes.sort_values(by='Morb', ascending=False, inplace=True, 
                                ignore_index=True)
        self.orb_sidx = numpy.concatenate(self.orb_sidx) if self.orb_sidx \
            else numpy.empty(0, dtype=numpy.intp)
        self.orb_pidx = numpy.concatenate(self.orb_pidx) if self.orb_pidx \
            else numpy.empty(0, dtype=numpy.intp)
        if len(self.orb_mass) > 0:
            self.orb_mass = numpy.concatenate(self.orb_mass)

    # ==========================================================================
    def _percolation(self):
        """Ensure unique particle/substructure assignment to most massive hosts.

        This method implements a percolation algorithm that ensures each particle
        and substructure is only counted as orbiting the single most massive halo
        it belongs to. It processes halos in descending mass order, recomputing
        orbiting masses after particle reassignment.

        Parameters
        ----------
        None
            Operates on instance attributes from _process_all_seeds.

        Returns
        -------
        None
            Creates new instance attributes:
            
            - self.haloes_perc : pandas.DataFrame or None
                Final percolated halo catalog containing only halos within the
                current mini-box. Returns None if no halos remain after.
            - self.orb_pid_perc : numpy.ndarray
                Percolated array of particle IDs with unique assignments.
            - self.orb_hid_perc : numpy.ndarray
                Percolated array of substructure halo IDs with unique assignments.

        Notes
        -----
        The percolation algorithm:
        1. Iterates through halos sorted by orbiting mass (descending)
        2. For each halo, removes particles/substructures already claimed by
        more massive hosts
        3. Recomputes orbiting mass with remaining particles
        4. Discards halos falling below the minimum particle threshold
        5. Retains only halos with centers inside the mini-box (INMB=True)

        Seen particles and substructures are tracked with Boolean arrays over
        the local particle and seed arrays, so each halo costs O(members).

        This step is critical for avoiding double-counting in overlapping halo
        regions and ensuring mass conservation across the catalog.

        See Also
        --------
        _process_all_seeds : Generates the initial halo catalog.
        _save_catalogues : Saves the percolated catalog to disk.
        """
        # Pull columns out of the DataFrame once; pandas scalar access inside
        # the loop is slow.
        haloes = self.haloes
        n_haloes = len(haloes.index)
        lidx = haloes["LIDX"].to_numpy(dtype=numpy.int64)
        ridx = haloes["RIDX"].to_numpy(dtype=numpy.int64)
        slidx = haloes["SLIDX"].to_numpy(dtype=numpy.int64)
        sridx = haloes["SRIDX"].to_numpy(dtype=numpy.int64)
        inmb = haloes["INMB"].to_numpy(dtype=bool)
        variable_mass = isinstance(self.mass_part, numpy.ndarray)

        # Boolean "seen" flags over the local particle and seed arrays. These
        # replace the Python sets of particle and seed IDs.
        particle_seen = numpy.zeros(len(self.pid_part), dtype=bool)
        seed_seen = numpy.zeros(self.n_seeds, dtype=bool)

        rows, morb_perc, norb_perc, nsubs_perc = [], [], [], []
        orb_pidx_perc, orb_sidx_perc = [], []

        for i in tqdm(range(n_haloes), ncols=100, 
                      desc='Percolating particles', colour='green', 
                      disable=self.disable_tqdm):

            # Select particles not orbiting anything more massive. They are
            # claimed even if this halo is discarded below.
            pidx_range = self.orb_pidx[lidx[i]:ridx[i]]
            keep = ~particle_seen[pidx_range]
            new_pidx = pidx_range[keep]
            particle_seen[new_pidx] = True
            n_orb = new_pidx.size
            
            # Skip to next seed if no longer a halo
            if n_orb < self.min_num_part:
                continue

            # Compute final orbiting mass
            if variable_mass:
                morb = numpy.sum(self.orb_mass[lidx[i]:ridx[i]][keep])
            else:
                morb = n_orb * self.mass_part

            # Select seeds not orbiting anything more massive.
            sidx_range = self.orb_sidx[slidx[i]:sridx[i]]
            new_sidx = sidx_range[~seed_seen[sidx_range]]
            seed_seen[new_sidx] = True

            # Ignore seed if it is not within the current mini-box
            if not inmb[i]:
                continue

            rows.append(i)
            morb_perc.append(morb)
            norb_perc.append(n_orb)
            nsubs_perc.append(new_sidx.size)
            orb_pidx_perc.append(new_pidx)
            orb_sidx_perc.append(new_sidx)

        # Build the percolated catalogue once.
        rows = numpy.array(rows, dtype=numpy.int64)
        norb_perc = numpy.array(norb_perc, dtype=numpy.int64)
        nsubs_perc = numpy.array(nsubs_perc, dtype=numpy.int64)
        lidx_perc = numpy.concatenate(([0], numpy.cumsum(norb_perc)[:-1])) \
            if rows.size else norb_perc
        slidx_perc = numpy.concatenate(([0], numpy.cumsum(nsubs_perc)[:-1])) \
            if rows.size else nsubs_perc

        self.haloes_perc = pandas.DataFrame({
            "Halo_ID": haloes["Halo_ID"].to_numpy(dtype=numpy.int64)[rows],
            "pos": haloes["pos"].to_numpy()[rows],
            "vel": haloes["vel"].to_numpy()[rows],
            "R200b": haloes["R200b"].to_numpy(dtype=numpy.float64)[rows],
            "M200b": haloes["M200b"].to_numpy(dtype=numpy.float64)[rows],
            "Morb": numpy.array(morb_perc, dtype=numpy.float64),
            "Norb": norb_perc,
            "LIDX": lidx_perc,
            "RIDX": lidx_perc + norb_perc,
            "INMB": numpy.ones(rows.size, dtype=bool),
            "NSUBS": nsubs_perc,
            "PID": haloes["PID"].to_numpy(dtype=numpy.int64)[rows],
            "SLIDX": slidx_perc,
            "SRIDX": slidx_perc + nsubs_perc,
        }, columns=self.haloes_perc.columns)

        # Concatenate lists into arrays of particle and seed IDs
        self.orb_pid_perc = self.pid_part[numpy.concatenate(orb_pidx_perc)] \
            if orb_pidx_perc else numpy.array([])
        self.orb_hid_perc = self.hid[numpy.concatenate(orb_sidx_perc)] \
            if orb_sidx_perc else numpy.array([])

    # ==========================================================================
    def _save_catalogues(self):
        """Write percolated halo and member catalogs to HDF5 file.

        This method saves the final halo catalog and associated particle/substructure
        membership data to a single HDF5 file in the mini-box output directory.

        Parameters
        ----------
        None
            Saves data from instance attributes created during percolation.

        Returns
        -------
        None
            Writes HDF5 file to disk at:
            {self.save_path}/{self.mini_box_id}.hdf5

        Notes
        -----
        The HDF5 file structure contains two main groups:

        /halo/ group:
            - Halo_ID, pos, vel, R200b, M200b, Morb, Norb, LIDX, RIDX, NSUBS,
            PID, SLIDX, SRIDX
            - Data types are optimized (e.g., uint32 for indices, float32 for
            physical quantities)
            - The INMB column is excluded as all halos satisfy this criterion
            after percolation

        /memb/ group:
            - PID: Particle IDs of all orbiting particles
            - Halo_ID: Halo IDs of all orbiting substructures

        Array indices (LIDX, RIDX, SLIDX, SRIDX) in the halo catalog provide
        efficient slicing into the packed member arrays without storing redundant
        halo associations for each particle/substructure.

        Raises
        ------
        IOError
            If the output directory is not writable or disk space is insufficient.

        See Also
        --------
        _percolation : Creates the data structures saved by this method.
        merge_catalogues : Combines individual mini-box catalogs.
        """

        full_path = self.save_path + f"{self.mini_box_id}.hdf5"
        dtypes = (
            numpy.uint32,   # Halo_ID
            numpy.float32,  # pos
            numpy.float32,  # vel
            numpy.float32,  # R200b
            numpy.float32,  # M200b
            numpy.float32,  # Morb
            numpy.uint32,   # Norb
            numpy.uint32,   # LIDX
            numpy.uint32,   # RIDX
            None,           # INMB is ignored
            numpy.uint32,   # NSUBS
            numpy.int32,    # PID
            numpy.uint32,   # SLIDX
            numpy.uint32    # SRIDX
        )

        if len(self.haloes_perc) > 0:
            with h5py.File(full_path, "w") as hdf:
                # Halo catalogue
                for i, key in enumerate(self.haloes_perc.columns):
                    if key == 'INMB':
                        continue
                    if key in ['pos', 'vel']:
                        data = numpy.stack(self.haloes_perc[key].values)
                    else:
                        data = self.haloes_perc[key].values
                    hdf.create_dataset(f'halo/{key}', data=data, dtype=dtypes[i])

                # Particles
                hdf.create_dataset('memb/PID', data=self.orb_pid_perc,
                                    dtype=self.orb_pid_perc[0].dtype)

                # Seeds
                if len(self.orb_hid_perc) > 0:
                    hdf.create_dataset('memb/Halo_ID', data=self.orb_hid_perc,
                                    dtype=self.orb_hid_perc[0].dtype)

    def run(self):
        """Execute the complete halo finding pipeline for this mini-box.

        This is the main entry point that orchestrates all stages of halo
        identification, from data loading through classification to catalog output.

        Parameters
        ----------
        None
            Uses parameters provided during class initialization.

        Returns
        -------
        None
            Produces side effects:
            - Creates output directory structure
            - Writes HDF5 catalog file to disk
            - Modifies instance attributes throughout execution

        Notes
        -----
        The pipeline executes the following stages in order:
        
        1. Setup: Create output directories
        2. Load seeds and apply mass filtering
        3. Early exit if no seeds present
        4. Compute NFW characteristic densities
        5. Load particle data and build spatial tree
        6. Load calibration parameters
        7. Initialize catalog data structures
        8. Process all seeds (particle classification + substructure detection)
        9. Percolation (unique particle assignment)
        10. Save catalogs to disk

        The method returns immediately (without error) if no valid seeds are
        present in the mini-box, producing no output files.

        Examples
        --------
        >>> classifier = MiniBoxClassifier(
        ...     mini_box_id=0,
        ...     min_num_part=20,
        ...     boxsize=100.0,
        ...     minisize=25.0,
        ...     load_path="/data/sim/",
        ...     run_name="test_run",
        ...     particle_type="dm"
        ... )
        >>> classifier.run()

        See Also
        --------
        process_minibox : Wrapper function for parallel processing.
        process_all_miniboxes : Orchestrates processing across all mini-boxes.
        """
        self._make_output_dir()
        self._load_seeds_and_filter()

        if self._early_exit_if_no_seeds():
            return None
        
        self._compute_deltac()
        self._build_seed_tree()
        self._load_particles()
        self._load_calibration_parameters()
        self._init_catalogue_dataframe()
        self._process_all_seeds()
        self._percolation()
        self._save_catalogues()
        
        return


def process_minibox(i, **kwargs):
    """Wrapper function for parallel halo finding in a single mini-box.

    This function creates a MiniBoxClassifier instance and executes the
    complete halo finding pipeline. It is designed as a lightweight wrapper
    for use with multiprocessing.Pool.

    Parameters
    ----------
    i : int
        Mini-box identifier (0-based index).
    **kwargs : dict
        Keyword arguments forwarded to MiniBoxClassifier constructor.
        Required keys: min_num_part, boxsize, minisize, load_path, run_name,
        particle_type. Optional keys: padding, fast_mass, disable_tqdm.

    Returns
    -------
    None
        Produces side effects via MiniBoxClassifier.run(), writing catalog
        files to disk.

    Notes
    -----
    This function is intentionally minimal to reduce pickling overhead when
    used with multiprocessing. All configuration is passed through kwargs
    rather than using closures or partial application.

    Examples
    --------
    >>> from multiprocessing import Pool
    >>> from functools import partial
    >>> 
    >>> params = {
    ...     'min_num_part': 20,
    ...     'boxsize': 100.0,
    ...     'minisize': 25.0,
    ...     'load_path': '/data/sim/',
    ...     'run_name': 'test_run',
    ...     'particle_type': 'dm'
    ... }
    >>> with Pool(4) as pool:
    ...     pool.map(partial(process_minibox, **params), range(64))

    See Also
    --------
    MiniBoxClassifier : The class that performs the actual halo finding.
    process_all_miniboxes : Higher-level function that manages parallelization.
    """
    classifier = MiniBoxClassifier(mini_box_id=i, **kwargs)
    classifier.run()
    return


def process_all_miniboxes(
    load_path: str,
    run_name: str,
    min_num_part: int,
    boxsize: float,
    minisize: float,
    padding: float,
    particle_type: str,
    redshift: float,
    seed_prop_names: tuple[str],
    fast_mass: bool = False,
    n_threads: int = None,
    data_source: SpatialDataSource | None = None,
) -> None:
    """Process all mini-boxes in parallel to generate individual halo catalogs.

    This function divides the simulation volume into mini-boxes and processes
    each in parallel using multiprocessing. It automatically handles thread
    management and falls back to sequential processing if parallel execution
    fails.

    Parameters
    ----------
    load_path : str
        Base directory containing simulation data. Must end with '/'.
    run_name : str
        Identifier for this run. Creates output directory 'run_{run_name}/'.
    min_num_part : int
        Minimum number of orbiting particles required to classify a seed
        as a halo.
    boxsize : float
        Total size of the cubic simulation volume.
    minisize : float
        Size of each cubic mini-box subdivision. Should evenly divide boxsize
        for optimal performance.
    padding : float
        Buffer distance beyond mini-box edges for particle loading. Ensures
        halos near boundaries are properly captured.
    particle_type : str
        Type of particles to load (e.g., 'dm' for dark matter, 'gas').
    redshift : float
        Redshift of simulation.
    seed_prop_names : Tuple[str], optional
        Tuple with three strings: mass, radius, and scale radius label names in 
        the mini-box HDF5 files, in case other names (e.g. Mvir, Rvir) were used. 
        Default is ('M200b', 'R200b', 'Rs').
    fast_mass : bool, default=False
        If True, apply additional mass-based filtering to seeds before
        classification to improve performance.
    n_threads : int, optional
        Number of parallel workers. If None, uses half of available CPU cores,
        capped at the number of mini-boxes.

    Returns
    -------
    None
        Produces side effects:
        - Creates output directory structure
        - Writes individual HDF5 catalog files for each mini-box
        - Displays progress bars during execution

    Notes
    -----
    The function automatically computes the number of mini-boxes as
    ceil(boxsize/minisize)^3.

    Thread management:
    - Default: min(cpu_count/2, n_mini_boxes)
    - Capped at total number of mini-boxes to avoid idle workers
    - Falls back to sequential processing if multiprocessing fails

    Each mini-box produces an independent catalog file that must be merged
    using merge_catalogues() to create the final unified catalog.

    Examples
    --------
    >>> process_all_miniboxes(
    ...     load_path='/data/simulation/',
    ...     run_name='production_v1',
    ...     min_num_part=20,
    ...     boxsize=100.0,
    ...     minisize=25.0,
    ...     padding=5.0,
    ...     particle_type='dm',
    ...     redshift=0.0,
    ...     seed_prop_names=('mvir', 'rvir', 'rs')
    ...     n_threads=8
    ... )

    See Also
    --------
    process_minibox : Processes individual mini-boxes.
    merge_catalogues : Combines individual catalogs into final output.
    MiniBoxClassifier : Core classification algorithm.
    """
    # Create directory if it does not exist
    save_path = load_path + f'run_{run_name}/mini_box_catalogues/'
    ensure_dir_exists(save_path)

    if data_source is None:
        data_source = LegacyMiniBoxDataSource(
            load_path = load_path, boxsize = boxsize, minisize = minisize,
            particle_type = particle_type, seed_prop_names = seed_prop_names,
            padding = padding,
        )
    region_ids = data_source.region_ids()
    n_regions = len(region_ids)
    
    # Cap the number of threads to the total number of mini-boxes/regions to process
    if n_threads is None:
        n_threads = min(max(1, os.cpu_count()//2), n_regions)
    else:
        n_threads = min(n_threads, n_regions)

    # Parallel processing of miniboxes.
    func = partial(
        process_minibox, 
        min_num_part=min_num_part,
        boxsize=boxsize, 
        minisize=minisize, 
        load_path=load_path,
        run_name=run_name, 
        particle_type=particle_type,
        redshift=redshift, 
        seed_prop_names=seed_prop_names,
        padding=padding, 
        fast_mass=fast_mass, 
        disable_tqdm=True,
        data_source=data_source,
    )
    
    # Safely handle multiprocessing falure with a fall back to a single thread.
    if n_threads > 1:
        # Submit the most expensive (most populated) mini-boxes first so a few
        # dense boxes do not end up running alone at the end of the run. Each
        # mini-box is independent, so the order does not change the output.
        box_order = data_source.region_ids_by_workload()
        try:
            with Pool(n_threads) as pool, \
                tqdm(total=n_regions, colour="green", ncols=100,
                    desc='Generating halo catalogue') as pbar:
                for _ in pool.imap_unordered(func, box_order):
                    pbar.update()
        except RuntimeError as e:
            # Fall back to sequential processing
            n_threads = 1
            raise RuntimeError(e)
    
    if n_threads == 1:
        for box_i in tqdm(region_ids, colour="green", ncols=100,
                        desc='Generating halo catalogue'):
            func(box_i)

    return None


def append_dataset(group: h5py.Group, name: str, data: numpy.ndarray) -> None:
    """Appends data to dataset or creates it if name does not exist.

    Parameters
    ----------
    group : h5py.Group
        HDF group
    name : str
        Dataset name
    data : numpy.ndarray
        Data to save under name.
    """
    if name in group.keys():
        ds = group[name]
        old_shape = ds.shape[0]
        new_shape = old_shape + data.shape[0]
        ds.resize((new_shape), axis=0)
        ds[old_shape:] = data
    else:
        group.create_dataset(name=name, data=data, chunks=True, maxshape=(None,))
    return


def merge_catalogues(
    load_path: str,
    run_name: str,
) -> None:
    """Merge individual mini-box catalogs into unified halo and member files.

    This function combines the HDF5 catalog files produced by individual
    mini-box processing into two final output files: a complete halo catalog
    and a member (particle/substructure) catalog with properly offset indices.

    Parameters
    ----------
    load_path : str
        Base directory containing simulation data and mini-box catalogs.
        Must match the path used in process_all_miniboxes.
    run_name : str
        Run identifier used to locate mini-box catalog directory and
        name output files.

    Returns
    -------
    None
        Produces two HDF5 files:
        - {load_path}/run_{run_name}/catalogue.hdf5: Combined halo properties
        - {load_path}/run_{run_name}/members.hdf5: Combined particle/substructure IDs

    Raises
    ------
    FileNotFoundError
        If no mini-box catalog files are found in the expected directory.

    Notes
    -----
    Output file structure:

    catalogue.hdf5:
        - Halo_ID, pos, vel, R200b, M200b, Morb, Norb, LIDX, RIDX, NSUBS,
          PID, SLIDX, SRIDX
        - Indices (LIDX, RIDX, SLIDX, SRIDX) are offset to account for
          concatenation across mini-boxes
        - Halos without substructures have SLIDX=SRIDX=-1

    members.hdf5:
        - PID: Concatenated particle IDs for all orbiting particles
        - Halo_ID: Concatenated halo IDs for all orbiting substructures

    The function makes two passes over the mini-box files: the first reads
    only dataset shapes, the second copies each file into its slice of
    preallocated output datasets. Peak memory is one mini-box catalogue.
    Files are merged in mini-box ID order, so the output order is deterministic.

    Missing mini-box files (from boxes with no halos) are automatically
    skipped without error.

    Examples
    --------
    >>> n_boxes = int(np.ceil(100.0 / 25.0))**3  # 64 mini-boxes
    >>> merge_catalogues(
    ...     load_path='/data/simulation/',
    ...     run_name='production_v1',
    ...     n_mini_boxes=n_boxes
    ... )

    See Also
    --------
    process_all_miniboxes : Generates the mini-box catalogs to be merged.
    run_orbiting_mass_assignment : High-level function that calls both.
    """
    save_path = load_path + f'run_{run_name}/mini_box_catalogues/'

    # Merge in mini-box ID order so the output order is deterministic.
    def _file_key(name):
        stem = os.path.splitext(name)[0]
        return (0, int(stem), name) if stem.isdigit() else (1, 0, name)
    files = sorted(os.listdir(save_path), key=_file_key)

    if len(files) == 0:
        raise FileNotFoundError("No mini-box catalogue files found.")

    # First pass: only read dataset shapes to find the size of each output
    # dataset and each file's offsets, so the outputs can be preallocated and
    # written with one slice per file. Peak memory is one mini-box catalogue.
    entries = []
    halo_specs = None
    pid_dtype, hid_dtype = None, None
    n_halo_tot, n_part, n_seed = 0, 0, 0
    for file in files:
        file_path = os.path.join(save_path, file)
        with h5py.File(file_path, 'r') as hdf_load:
            if 'halo' not in hdf_load.keys():
                continue
            if halo_specs is None:
                halo_specs = {key: (ds.dtype, ds.shape[1:])
                              for key, ds in hdf_load['halo'].items()}
            n_halo_this = hdf_load['halo/Halo_ID'].shape[0]
            n_part_this = hdf_load['memb/PID'].shape[0]
            if pid_dtype is None:
                pid_dtype = hdf_load['memb/PID'].dtype
            n_seed_this = 0
            if 'Halo_ID' in hdf_load['memb'].keys():
                n_seed_this = hdf_load['memb/Halo_ID'].shape[0]
                if hid_dtype is None:
                    hid_dtype = hdf_load['memb/Halo_ID'].dtype
        entries.append((file_path, n_halo_tot, n_halo_this, n_part, n_part_this,
                        n_seed, n_seed_this))
        n_halo_tot += n_halo_this
        n_part += n_part_this
        n_seed += n_seed_this

    if halo_specs is None:
        raise FileNotFoundError("No mini-box catalogue files found.")

    # Second pass: copy each file into its slice of the preallocated outputs.
    with h5py.File(load_path + f'run_{run_name}/members.hdf5', 'w') as hdf_memb, \
            h5py.File(load_path + f'run_{run_name}/catalogue.hdf5', 'w') as hdf:
        memb_pid = hdf_memb.create_dataset('PID', shape=(n_part,), dtype=pid_dtype)
        memb_hid = None
        if n_seed > 0:
            memb_hid = hdf_memb.create_dataset('Halo_ID', shape=(n_seed,),
                                               dtype=hid_dtype)

        halo_out = {}
        for key, (dtype, tail) in halo_specs.items():
            if key in {'SLIDX', 'SRIDX'}:
                dtype = numpy.int32
            halo_out[key] = hdf.create_dataset(key, shape=(n_halo_tot,) + tail,
                                               dtype=dtype)

        for (file_path, h0, nh, p0, n_part_this, s0, n_seed_this) in tqdm(
                entries, ncols=100, desc='Merging catalogues', colour='green'):
            with h5py.File(file_path, 'r') as hdf_load:
                # Member data ==================================================
                memb_pid[p0:p0 + n_part_this] = hdf_load['memb/PID'][()]
                if n_seed_this > 0:
                    memb_hid[s0:s0 + n_seed_this] = hdf_load['memb/Halo_ID'][()]

                # Halo data ====================================================
                for key in halo_specs:
                    data = hdf_load[f"halo/{key}"][()]
                    # Offset indices by number of objects in previous files
                    if key in {"LIDX", "RIDX"}:
                        data += p0
                    elif key in {"SLIDX", "SRIDX"}:
                        data += s0
                    halo_out[key][h0:h0 + nh] = data

        # Set the seed index to -1 for all those haloes without subhaloes.
        if 'SLIDX' in halo_out and 'SRIDX' in halo_out:
            slidx = halo_out['SLIDX'][()]
            sridx = halo_out['SRIDX'][()]
            mask = (sridx - slidx == 0)
            slidx[mask] = -1
            sridx[mask] = -1
            halo_out['SLIDX'][()] = slidx
            halo_out['SRIDX'][()] = sridx
    
    return None


def run_orbiting_mass_assignment(
    load_path: str,
    run_name: str,
    min_num_part: int,
    boxsize: float,
    minisize: float,
    padding: float,
    particle_type: str,
    redshift: float,
    seed_prop_names: tuple[str] = ('M200b', 'R200b', 'Rs'),
    fast_mass: bool = False,
    n_threads: int = None,
    cleanup: bool | str = False,
    data_source: SpatialDataSource | None = None,
) -> None:
    """Generate complete halo catalog using kinetic energy classification.

    This is the main entry point for the halo finding pipeline. It orchestrates
    mini-box processing, catalog merging, and optional cleanup of intermediate
    files to produce a final unified halo catalog with particle membership.

    Parameters
    ----------
    load_path : str
        Base directory containing simulation data. Must contain:
        - Seed files (halo candidates)
        - Particle data files
        - calibration_pars.hdf5 (classification parameters)
    run_name : str
        Unique identifier for this run. Creates directory structure:
        run_{run_name}/mini_box_catalogues/ (temporary, optional)
        run_{run_name}/catalogue.hdf5 (final output)
        run_{run_name}/members.hdf5 (final output)
    min_num_part : int
        Minimum number of orbiting particles required to classify a structure
        as a halo. Typical values: 20-100 depending on resolution.
    boxsize : float
        Size of the cubic simulation volume in simulation units (e.g., Mpc/h).
    minisize : float
        Size of cubic mini-box subdivisions. Smaller values increase
        parallelism but add overhead. Recommended: boxsize/4 to boxsize/8.
    padding : float
        Buffer distance for particle loading beyond mini-box boundaries.
        Should be at least ~5 Mpc/h to capture halo outskirts properly.
    particle_type : str
        Type of particles to process. Common values: 'dm' (dark matter),
        'gas', 'stars'.
    redshift : float,
        Redshift of simulation.
    seed_prop_names : Tuple[str], optional
        Tuple with three strings: mass, radius, and scale radius label names in 
        the mini-box HDF5 files, in case other names (e.g. Mvir, Rvir) were used. 
        Default is ('M200b', 'R200b', 'Rs').
    fast_mass : bool, default=False
        Enable aggressive mass-based filtering of seed candidates to improve
        performance. May miss low-mass halos near the resolution limit.
    n_threads : int, optional
        Number of parallel workers for mini-box processing. If None,
        automatically determined as min(cpu_count/2, n_mini_boxes).
    cleanup : bool | str, default=False
        If True, delete individual mini-box catalog files after merging to
        save disk space. Final catalogs are retained. Also removes the mini-box
        directory if set to 'all'.

    Returns
    -------
    None
        Produces side effects:
        - Creates run_{run_name}/ directory structure
        - Writes catalogue.hdf5 and members.hdf5 in run_{run_name}/
        - Optionally removes temporary mini-box catalogs

    Notes
    -----
    The kinetic mass criterion classifies particles as orbiting or infalling
    based on their phase-space coordinates (position and velocity) relative
    to halo centers. This approach provides more accurate mass estimates than
    spherical overdensity methods in dynamically active environments.

    Processing pipeline:
    1. Divide volume into mini-boxes (automatically determined)
    2. Process each mini-box in parallel:
       - Load seeds and particles
       - Classify particles using phase-space cuts
       - Identify substructures with 6D ball criterion
       - Percolate to ensure unique particle assignment
    3. Merge mini-box catalogs into unified outputs
    4. Optionally clean up intermediate files

    The final catalogue.hdf5 contains halo properties while members.hdf5
    contains particle/substructure IDs. Use the index columns (LIDX, RIDX,
    SLIDX, SRIDX) to slice the member arrays for each halo.

    Examples
    --------
    Basic usage:
    >>> run_orbiting_mass_assignment(
    ...     load_path='/data/simulations/cosmo_box/',
    ...     run_name='z0_halos',
    ...     min_num_part=20,
    ...     boxsize=100.0,
    ...     minisize=25.0,
    ...     padding=5.0,
    ...     particle_type='dm'
    ... )

    High-performance run with cleanup:
    >>> run_orbiting_mass_assignment(
    ...     load_path='/data/simulations/high_res/',
    ...     run_name='production_run',
    ...     min_num_part=50,
    ...     boxsize=200.0,
    ...     minisize=25.0,
    ...     padding=5.0,
    ...     particle_type='dm',
    ...     redshift=0.0,
    ...     fast_mass=True,
    ...     seed_prop_names=('mvir', 'rvir', 'rs')
    ...     n_threads=16,
    ...     cleanup=True
    ... )

    See Also
    --------
    process_all_miniboxes : Handles parallel mini-box processing.
    merge_catalogues : Combines individual catalogs.
    MiniBoxClassifier : Core classification algorithm and data structure.
    """
    process_all_miniboxes(
        load_path=load_path,
        run_name=run_name,
        min_num_part=min_num_part,
        boxsize=boxsize,
        minisize=minisize,
        padding=padding,
        particle_type=particle_type,
        redshift = redshift,
        seed_prop_names=seed_prop_names,
        fast_mass=fast_mass,
        n_threads=n_threads,
        data_source = data_source,
    )

    merge_catalogues(
        load_path=load_path,
        run_name=run_name,
    )

    if cleanup:
        catalogues_path = load_path + f'run_{run_name}/mini_box_catalogues/'
        for item in os.listdir(catalogues_path):
            os.remove(catalogues_path + item)
        os.removedirs(catalogues_path)
        print("Catalogues Cleaned Up")
    if cleanup == 'all':
        n_cells = int(numpy.ceil(boxsize / minisize))
        mini_boxes_path = load_path + f'mini_boxes_nside_{n_cells}/'
        for item in os.listdir(mini_boxes_path):
            os.remove(mini_boxes_path + item)
        os.removedirs(mini_boxes_path)

    return None
