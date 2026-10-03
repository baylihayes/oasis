"""
Data sources that supply seeds and particles to the OASIS classifier.

The classifier only needs arrays for one region (a minibox plus padding).
A data source hides where those arrays come from (minibox files currently, 
pre-buffered large tiles later), so the science stays the same. 
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

import h5py
import numpy as np

from oasis.minibox import load_particles, load_seeds



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