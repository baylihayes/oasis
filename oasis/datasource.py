"""
Data sources that supply seeds and particles to the OASIS classifier.

The classifier only needs arrays for one region (a minibox plus padding).
A data source hides where those arrays come from (minibox files or prebuilt
core + ribbon tiles), so the science stays the same.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

import h5py
import numpy as np

from oasis.minibox import load_particles, load_seeds
from oasis.coordinates import _periodic_displacement


@dataclass
class SeedSet:
    """
    Seeds around one region in global coordinates.
    
    """
    pos: np.ndarray     #(n,3)
    vel: np.ndarray     #(n,3)
    hid: np.ndarray     #(n,)
    r200b: np.ndarray   #(n,)
    m200b: np.ndarray   #(n,)
    rs: np.ndarray      #(n,)
    in_core: np.ndarray #(n,) bool: True if this region owns the seed

    def __len__(self) -> int:
        # return the length of the hid array
        return len(self.hid)


@dataclass
class ParticleSet:
    """
    Particles around one region, in global coordinates. Velocities are as 
    stored in the simulation (v_sim = v_part / sqrt(a)). The classifier converts
    them to the appropriate units.

    """
    pos: np.ndarray     #(n,3)
    vel: np.ndarray     #(n,3)
    pid: np.ndarray     #(n,)
    mass: Union[float, np.ndarray]


class SpatialDataSource(ABC):
    """
    Supplies seeds and particles for each region of the simulation.

    'boxsize' must always be the size of the whole periodic simulation, not
    the size of a region or tile.
    
    """
    boxsize: float

    @abstractmethod
    def region_ids(self) -> list[int]:
        """
        All region IDs to process.

        """
    def region_ids_by_workload(self) -> list[int]:
        """
        Region IDs in the order they should be submitted to workers.

        """
        return self.region_ids()

    @abstractmethod
    def load_seeds(self, region_id: int) -> SeedSet:
        """
        Seeds in the region plus padding, with an ownership mask.

        """

    @abstractmethod
    def load_particles(self, region_id: int) -> ParticleSet:
        """
        Particles in the region plus padding.
        
        """

class LegacyMiniBoxDataSource(SpatialDataSource):
    """
    Reads today's minibox HDF5 files (mini_boxes_nside_<n>/<id>.hdf5).
    
    """
    def __init__(self, load_path, boxsize, minisize, particle_type,
                 seed_prop_names = ('M200b', 'R200b', 'Rs'), padding=5.0):
        self.load_path = load_path
        self.boxsize = boxsize
        self.minisize = minisize
        self.particle_type = particle_type
        self.seed_prop_names = seed_prop_names
        self.padding = padding
        self.cells_per_side = int(np.ceil(boxsize / minisize))

    def region_ids(self) -> list[int]:
        return list(range(self.cells_per_side**3))

    def region_ids_by_workload(self) -> list[int]:
        # Most populated miniboxes first; unreadable files last.
        n_regions = self.cells_per_side**3
        counts = np.zeros(n_regions, dtype = np.int64)
        for box_id in range(n_regions):
            file_name = (self.load_path + 
                        f'mini_boxes_nside_{self.cells_per_side}/{box_id}.hdf5')
            try:
                with h5py.File(file_name, 'r') as hdf:
                    counts[box_id] = hdf[f'{self.particle_type}/ID'].shape[0]
            except (OSError, KeyError):
                counts[box_id] = -1
        return np.argsort(-counts, kind = 'stable').tolist()

    def load_seeds(self, region_id: int) -> SeedSet:
        pos, vel, hid, r200b, m200b, rs, mask = load_seeds(
            region_id, self.boxsize, self.minisize, self.load_path,
            self.seed_prop_names, self.padding)
        return SeedSet(pos, vel, hid, r200b, m200b, rs, in_core=mask)

    def load_particles(self, region_id: int) -> ParticleSet:
        pos, vel, pid, mass = load_particles(
            region_id, self.boxsize, self.minisize, self.load_path,
            self.particle_type, self.padding)
        return ParticleSet(pos, vel, pid, mass)

class BufferedTileDataSource(SpatialDataSource):
    """
    Serves the inner cells of one prebuilt core + ribbon tile (see oasis.tiles)
    
    Regions are the tile's CORE cells, identified by their GLOBAL cell ID (the
    same number a mini-box of size inner_cell_size would have), so region IDs
    never clash between tiles. Each region is a cell plus 'padding', read from the cell and its neighbors with 
    direct slices (rows are grouped by cell, see 'cell_offset'). Ribbon cells are
    only ever read as neighbors, never processed as regions.

    Parameters
    ----------
    tile_path : str
        Tile file written by oasis.tiles.build_tiles().
    padding : float, default=5.0
        Distance around each core cell to load. Must not exceed the tile's
        ribbon width (buffer_width).
    seed_prop_names : tuple of str, default=('M200b', 'R200b', 'Rs')
        Mass, radius and scale radius dataset names in the 'seeds' group.

    The particle mass is read from the tile metadata (as float32, like the
    minibox files); if absent, a per-particle 'mass' dataset is used.

    Raises
    ------
    ValueError
        If padding is larger than the tile's buffer_width.
    """

    def __init__(self, tile_path, padding = 5.0,
                 seed_prop_names = ('M200b', 'R200b', 'Rs')):
        self.tile_path = tile_path
        self.padding = padding
        self.seed_prop_names = seed_prop_names
        with h5py.File(tile_path, 'r') as f:
            meta = f['tile_metadata'].attrs
            self.boxsize = float(meta['global_boxsize'])
            self.cell_size = float(meta['inner_cell_size'])
            self.buffer_width = float(meta['buffer_width'])
            self.ring = int(meta['ring'])
            self.m = int(meta['cells_per_side'])
            core_min = np.asarray(meta['core_bounds'])[:,0]
            # stored as float32 like the minibox files, so Morb = N*m is
            # computed with the same precision as before
            self.particle_mass = (np.float32(meta['particle_mass'])
                                    if 'particle_mass' in meta else None)
            self._n_part_per_cell = np.diff(f['particles/cell_offset'][()])
        if padding > self.buffer_width:
            raise ValueError(
                f"padding ({padding}) is larger than tile's ribbon"
                f" ({self.buffer_width}): rows near the core edge would be missing.")
        # Neighbor layers needed to cover 'padding' (at most 'ring').
        self.layers = int(np.ceil(padding / self.cell_size))
        # Global index of the tile's first core cell along each axis
        self._g0 = np.rint(core_min / self.cell_size).astype(np.int64)
        # Global cell grid: the same numbering as miniboxes of size cell_size
        self.n_global = int(round(self.boxsize / self.cell_size))
        self._n_core = self.m - 2 * self.ring

    # -- geometry --------------------------------------------------------
    def _ijk(self, cell):
        m = self.m
        return np.array([cell % m , (cell // m) % m, cell // m**2])

    def region_center(self, region_id):
        """Center of a core cell in global coordinates, computed with the same 
        formula as the minibox loader: (global cell index + 0.5) * size."""
        n = self.n_global
        g = np.array([region_id % n, (region_id // n) % n, region_id // n**2])
        return (g + 0.5) * self.cell_size

    def region_ids(self):
        """Global cell IDs of the tile's core cells, numbered like miniboxes of
        size inner_cell_size, so region ID never clash between tiles."""
        n, core = self.n_global, range(self._n_core)
        gx, gy, gz = self._g0
        return [(gx + i) + (gy + j) * n + (gz + k) * n**2 for k in core for j in core for i in core]

    def _local_cell(self, region_id):
        """Local cell index inside this tile of a global core-cell ID."""
        n, m = self.n_global, self.m
        g = np.array([region_id % n, (region_id // n) % n, region_id // n**2])
        ijk = g - self._g0 + self.ring
        if np.any(ijk < self.ring) or np.any(ijk >= m - self.ring):
            raise ValueError(f"region {region_id} is not a core cell of {self.tile_path}")
        return int(ijk[0] + ijk[1] * m + ijk[2] * m**2)        

    def region_ids_by_workload(self):
        ids = np.array(self.region_ids())
        counts = self._n_part_per_cell[[self._local_cell(r) for r in ids]]
        return ids[np.argsort(-counts, kind='stable')].tolist()

    # -- reading ----------------------------------------------------------
    def _read(self, group, cell):
        """Rows of 'cell' and its neighbors (self.layers deep), and a mask of 
        the rows that belong to 'cell' itself."""
        i, j, k = self._ijk(cell)
        L, m = self.layers, self.m
        offsets = group['cell_offset'][()]
        names = [n for n in group.keys() if n != 'cell_offset']
        pieces = {n: [] for n in names}
        own = []
        for dz in range(-L, L + 1):
            for dy in range(-L, L + 1):
                # Cells next to each other in x are next to each other on disk,
                # so each (dy, dz) row of 2L + 1 cells is one contiguous slice.
                first = (i - L) + (j + dy) * m + (k + dz) * m**2
                a, b = offsets[first], offsets[first + 2 * L + 1]
                for n in names:
                    pieces[n].append(group[n][a:b])
                mask = np.zeros(b - a, dtype = bool)
                if dy == 0 and dz == 0:
                    mask[offsets[cell] - a:offsets[cell + 1] - a] = True
                own.append(mask)
        return ({n: np.concatenate(v) for n, v in pieces.items()},
                np.concatenate(own))

    def _within_padding(self, pos, region_id):
        """Same cut as the minibox loader: |x - center| <= size / 2 + padding"""
        rel = _periodic_displacement(pos, self.region_center(region_id), self.boxsize)
        return np.all(np.abs(rel) <= 0.5 * self.cell_size + self.padding, axis = 1)

    def load_seeds(self, region_id):
        with h5py.File(self.tile_path, 'r') as f:
            data, own = self._read(f['seeds'], self._local_cell(region_id))
        keep = self._within_padding(data['pos'], region_id)
        m200, r200, rs = self.seed_prop_names
        return SeedSet(
            pos = data['pos'][keep], vel = data['vel'][keep], hid = data['ID'][keep],
            r200b = data[r200][keep], m200b = data[m200][keep], rs = data[rs][keep],
            # Owned only if in this cell and in the tile's core.
            in_core = (own & data['in_core'])[keep],
        )

    def load_particles(self, region_id):
        with h5py.File(self.tile_path, 'r') as f:
            data, _ = self._read(f['particles'], self._local_cell(region_id))
        keep = self._within_padding(data['pos'], region_id)
        mass = (self.particle_mass if self.particle_mass is not None 
                else data['mass'][keep])
        return ParticleSet(data['pos'][keep], data['vel'][keep],
                           data['ID'][keep], mass)