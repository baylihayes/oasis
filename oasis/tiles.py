"""Build core+ribbon tiles for very large simulations (MDPL2)

The box is split into tiles_per_side**3 cubic cores. Each tile stores the rows
of its core plus a ribbon of width 'buffer_width' around it, always in GLOBAL
coordinates (nothing is shifted or wrapped per tile). Inside a tile, rows are
grouped by inner cell, with an offset table so a cell is one contiguous slice.

"""
import os
from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterator

import h5py
import numpy as np



@dataclass(frozen=True)
class TileSpec:
    """Geometry of the core + ribbon tiling.

    The box is split into tiles_per_side**3 cubic cores of size
    boxsize / tiles_per_side. Each tile also stores a ribbon of width
    buffer_width around its core. Inside a tile, rows are grouped into cubic
    inner cells of size inner_cell_size; `ring` extra cells per side hold the
    ribbon.

    Raises
    ------
    ValueError
        If buffer_width < 0, if the ribbon is so wide that a tile would overlap
        itself across the periodic boundary, or if inner_cell_size does not
        divide the core size.
    """

    boxsize: float          #global periodic box size, e.g. 1000.0
    tiles_per_side: int     # e.g. 5 -> 125 tiles
    buffer_width: float     # ribbon width w
    inner_cell_size: float # e.g. 25.0

    def __post_init__(self):
        if self.buffer_width < 0:
            raise ValueError("buffer_width must be >= 0.")
        if self.core_size + 2 * self.buffer_width > self.boxsize:
            raise ValueError("ribbon too wide: a tile would overlap itself "
                             "across the periodic boundary.")
        n_inner = self.core_size / self.inner_cell_size
        if abs(n_inner - round(n_inner)) > 1e-9:
            raise ValueError("inner_cell_size must divide the core size.")

    @property
    def core_size(self) -> float:
        return self.boxsize / self.tiles_per_side

    @property
    def n_tiles(self) -> int:
        return self.tiles_per_side**3

    @property
    def ring(self) -> int:
        """
        Inner cells added on each side of the core to hold the ribbon.
    
        """
        return int(np.ceil(self.buffer_width / self.inner_cell_size))

    @property
    def cells_per_side(self) -> int:
        return int(round(self.core_size / self.inner_cell_size)) + 2 * self.ring

    def tile_index(self, tile_id: int) -> np.ndarray:
        """
        (ix, iy, iz) of a tile; x varies the fastest, as for the minibox IDs.
    
        """
        n = self.tiles_per_side
        return np.array([tile_id % n, (tile_id // n) % n, tile_id // n**2])

    def core_bounds(self, tile_id: int) -> np.ndarray:
        """
        [[xmin,xmax], [ymin,ymax],[zmin, zmax]] of the tile's core.
    
        """
        lo = self.tile_index(tile_id) * self.core_size
        return np.stack([lo, lo+self.core_size], axis=1)


#Below are the functions to assign rows to tiles.

def wrap_positions(pos: np.ndarray, boxsize: float) -> np.ndarray:
    """
    Wrap into [0, boxsize) and return float32 as stored on disk. Values that
    round up to the boxsize in float32 become 0 (same rules as the minibox writer).
    
    """
    pos = np.mod(np.asarray(pos, dtype=np.float64), boxsize)
    pos[pos.astype(np.float32).astype(np.float64) >= boxsize] = 0.0
    return pos.astype(np.float32)

def assign_to_tiles(pos: np.ndarray, spec: TileSpec):
    """
    Every (row, tile) pair where the position lies in that tile's core or
    ribbon. Returns (rows, tiles, in_core); each row has exactly one in_core
    pair, plus one pair for each ribbon it falls in (up to 7 near corners).
    
    """
    n, core, w = spec.tiles_per_side, spec.core_size, spec.buffer_width
    x = np.asarray(pos, dtype=np.float64)

    if x.shape[0] == 0:
        # Nothing to assign (e.g. an empty input chunk).
        return (np.empty(0, dtype = np.int64), np.empty(0, dtype = np.int64),
                np.empty(0, dtype = bool))

    i0 = np.minimum((x // core).astype(np.int64), n - 1)    # core tile per axis
    t = x - i0 * core                                       # position inside core
    lo = t <= w              # also inside the ribbon of the tile below
    hi = t >= core - w      # also inside the ribbon of the tile above

    rows_out, tiles_out, core_out = [],[],[]
    for off in product((-1,0,1), repeat = 3):
        mask = np.ones(len(x), dtype = bool)
        for axis, o in enumerate(off):
            if o == -1:
                mask &= lo[:, axis]
            elif o == 1:
                mask &= hi[:, axis]
        rows = np.flatnonzero(mask)
        if rows.size == 0:
            continue
        ijk = (i0[rows] + np.array(off)) % n                # periodic wrap
        rows_out.append(rows)
        tiles_out.append(ijk[:,0] + ijk[:,1] * n + ijk[:,2] * n**2)
        core_out.append(np.full(rows.size, off == (0,0,0)))
    return (np.concatenate(rows_out), np.concatenate(tiles_out), 
            np.concatenate(core_out))

def local_cell_ids(pos: np.ndarray, tile_id: int, spec: TileSpec) -> np.ndarray:
    """
    Inner-cell index within one tile for positions in its core or ribbon.
    
    """
    L, w, m = spec.boxsize, spec.buffer_width, spec.cells_per_side
    core_min = spec.core_bounds(tile_id)[:,0]
    u = np.mod(np.asarray(pos, dtype = np.float64) - core_min, L)
    u = np.where(u >= L - w, u - L, u)      # lower ribbon: [-w, 0)
    ijk = np.floor(u / spec.inner_cell_size).astype(np.int64) + spec.ring
    # Rows at the very edge of the ribbon can round one cell out; keep them in
    # the outermost cell rather than failing.
    ijk = np.clip(ijk, 0, m - 1)
    return ijk[:,0] + ijk[:,1] * m + ijk[:,2] * m **2




# Below is where the tiles are written
# Stage 1
ChunkFactory = Callable[[], Iterator[dict]]     # returns a new iterator each call

def bucket_into_tiles(make_chunks: ChunkFactory, out_dir: str, spec: TileSpec,
                      kind: str, tile_ids = None) -> np.ndarray:
    """
    Stream input chunks into one unsorted bucket file per tile.
    
    Each chunk is a dict of equal-length arrays and must contain 'pos'. 
    make_chunks() is called twice: first to count rows per tile, then to write
    them into preallocated datasets. Returns the number of rows per tile.
    Only the tiles in tile_ids (default: all) get a bucket file; the counts returned
    are for all tiles. 
    
    Raises
    ------
    ValueError
        If make_chunks() yields no chunks.
    RuntimeError
        If the input changes between the two passes.

    """
    tile_ids = (np.arange(spec.n_tiles) if tile_ids is None
                else np.unique(np.asarray(tile_ids, dtype = np.int64)))
    wanted = np.zeros(spec.n_tiles, dtype = bool)
    wanted[tile_ids] = True

    counts = np.zeros(spec.n_tiles, dtype = np.int64)
    fields = None
    for chunk in make_chunks():         # pass 1: count
        _, tiles, _ = assign_to_tiles(wrap_positions(chunk['pos'], spec.boxsize), spec)
        counts += np.bincount(tiles, minlength=spec.n_tiles)
        if fields is None:
            fields = {k: (np.asarray(v).dtype, np.shape(v)[1:])
                      for k, v in chunk.items()}
            fields['pos'] = (np.dtype(np.float32), (3,))
            if 'vel' in fields:
                #   Velocities are stored as float32, like the minibox files.
                fields['vel'] = (np.dtype(np.float32), (3,))
    if fields is None:
        raise ValueError(f"No input chunks for '{kind}': make_chunks() yielded nothing. "
                         "Check input path.")

    os.makedirs(out_dir, exist_ok = True)
    files = {t: h5py.File(os.path.join(out_dir, f'bucket_{kind}_{t}.hdf5'), 'w')
             for t in tile_ids.tolist()}
    
    try: 
        dsets = {}
        for t, f in files.items():
            d = {k: f.create_dataset(k, shape=(counts[t],) + tail, dtype = dt)
                 for k, (dt, tail) in fields.items()}
            d['in_core'] = f.create_dataset('in_core', shape=(counts[t],), dtype = bool)
            dsets[t] = d

        cursor = np.zeros(spec.n_tiles, dtype = np.int64)
        for chunk in make_chunks():                 # pass 2: write
            chunk = dict(chunk, pos = wrap_positions(chunk['pos'], spec.boxsize))
            if 'vel' in chunk:
                chunk['vel'] = np.asarray(chunk['vel'], dtype = np.float32)
            rows, tiles, in_core = assign_to_tiles(chunk['pos'], spec)
            keep = wanted[tiles]
            rows, tiles, in_core = rows[keep], tiles[keep], in_core[keep]
            order = np.argsort(tiles, kind = 'stable')
            rows, tiles, in_core = rows[order], tiles[order], in_core[order]
            bounds = np.searchsorted(tiles, np.arange(spec.n_tiles + 1))
            for t in np.flatnonzero(np.diff(bounds)):
                sel = slice(bounds[t], bounds[t + 1])
                r, c0 = rows[sel], cursor[t]
                for k in fields:
                    dsets[t][k][c0:c0 + r.size] = np.asarray(chunk[k])[r]
                dsets[t]['in_core'][c0:c0 + r.size] = in_core[sel]
                cursor[t] += r.size
        if not np.array_equal(cursor[tile_ids], counts[tile_ids]):
            raise RuntimeError("input changed between the two passes.")
    finally:
        for f in files.values():
            f.close()
    return counts

# Stage 2: sort one tile's bucket by inner cell and write the tile file
def build_tile(bucket_path: str, tile_path: str, tile_id: int, spec: TileSpec,
               kind: str, rows_per_block: int = 50_000_000,
               attrs: dict | None = None) -> None:
    """
    Group one tile's bucket rows by inner cell into group 'kind' of the tile
    file, with cell_offset[c]:cell_offset[c+1] giving the rows of cell c.

    rows_per_block limits how many bucket rows are held in memory at once.

    Raises
    ------
    FileExistsError
        If the tile file already contains a group named 'kind'.
    """
    m3 = spec.cells_per_side**3 
    with h5py.File(bucket_path, 'r') as src, h5py.File(tile_path, 'a') as dst:
        if kind in dst:
            raise FileExistsError(f"{tile_path} already contains a '{kind}' group.")

        n_rows = src['pos'].shape[0]
        blocks = [slice(a, min(a + rows_per_block, n_rows))
                  for a in range(0, n_rows, rows_per_block)]

        counts = np.zeros(m3, dtype = np.int64)         # pass 1: rows per cell
        for b in blocks:
            counts += np.bincount(local_cell_ids(src['pos'][b], tile_id, spec),
                                minlength = m3)

        offsets = np.concatenate(([0], np.cumsum(counts)))

        meta = dst.require_group('tile_metadata')
        meta.attrs.update(dict(
            global_boxsize=spec.boxsize, tiles_per_side = spec.tiles_per_side,
            tile_id = tile_id, tile_index = spec.tile_index(tile_id),
            core_bounds = spec.core_bounds(tile_id), buffer_width = spec.buffer_width,
            inner_cell_size = spec.inner_cell_size, cells_per_side = spec.cells_per_side, 
            ring = spec.ring, **(attrs or {})
        ))

        grp = dst.create_group(kind)
        out = {k: grp.create_dataset(k, shape = src[k].shape, dtype = src[k].dtype) 
               for k in src.keys()}
        grp.create_dataset('cell_offset', data = offsets)

        cursor = offsets[:-1].copy()
        for b in blocks:                                  # pass 2: fill cell slices
            cell = local_cell_ids(src['pos'][b], tile_id, spec)
            order = np.argsort(cell, kind = 'stable')
            cell = cell[order]
            data = {k: src[k][b][order] for k in src.keys()}
            starts = np.searchsorted(cell, np.arange(m3 + 1))
            for c in np.flatnonzero(np.diff(starts)):
                lo, hi = starts[c], starts[c+1]
                for k in out:
                    out[k][cursor[c]:cursor[c] + hi - lo] = data[k][lo:hi]
                cursor[c] += hi - lo

def build_tiles(make_particle_chunks: ChunkFactory, make_seed_chunks: ChunkFactory, 
                out_dir: str, spec: TileSpec, particle_mass: float | None = None,
                keep_buckets: bool = False, overwrite: bool = False,
                tile_ids = None) -> None:
    """
    Builds the tiles in tile_ids (default: all), so a large box can be built in groups
    of tiles: each call reads the whole input twice, but bucket files (as large as the 
    tiles themselves) exist only for the tiles being built. Refuses to run if any
    of this call's tile or bucket files already exist, unless overwrite = True, in 
    which case only those files are deleted first. Tiles of other calls in the same
    out_dir are left alone.

    Parameters
    ----------
    make_particle_chunks, make_seed_chunks : callable
        Called with no arguments; each call must return a NEW iterator over
        chunks (each input is read twice). A chunk is a dict of equal-length
        arrays with at least 'ID', 'pos' (n, 3) and 'vel' (n, 3); seed chunks
        also need the seed properties (e.g. 'M200b', 'R200b', 'Rs'). Positions
        are wrapped into the box and stored as float32, velocities as float32;
        other columns keep their dtype.
    out_dir : str
        Folder for the tile files (created if needed).
    spec : TileSpec
        Tiling geometry.
    particle_mass : float, optional
        Constant particle mass, stored in the tile metadata. If None, particle
        chunks must contain a per-particle 'mass' column.
    keep_buckets : bool, default=False
        Keep the intermediate bucket files (for debugging).
    overwrite : bool, default=False
        Delete this call's existing tile/bucket files in out_dir first.
    tile_ids : iterable of int, optional
        Tiles to build (0 <= id < spec.n_tiles). Default: all tiles. 

    Raises
    ------
    FileExistsError
        If any of this call's tile or bucket files already exist and
        overwrite is False.
    ValueError
        If a chunk factory yields no chunks.
    ValueError
        If a tile ID is outside 0 <= id < spec.n_tiles.
    """
    tile_ids = (list(range(spec.n_tiles)) if tile_ids is None
                else sorted({int(t) for t in tile_ids}))
    if any(t < 0 or t >= spec.n_tiles for t in tile_ids):
        raise ValueError(f"tile_ids must be between 0 and {spec.n_tiles - 1}.")

    mine = [os.path.join(out_dir, name) for t in tile_ids
            for name in (f'tile_{t}.hdf5', f'bucket_particles_{t}.hdf5',
                         f'bucket_seeds_{t}.hdf5')]

    existing = [p for p in mine if os.path.exists(p)]
    if existing:
        if not overwrite: 
            raise FileExistsError(
                f"{len(existing)} tile/bucket files already exist in {out_dir}."
                "Use a new folder or pass overwrite = True."
            )
        for path in existing:
            os.remove(path)


    bucket_into_tiles(make_particle_chunks, out_dir, spec, 'particles', tile_ids)
    bucket_into_tiles(make_seed_chunks, out_dir, spec, 'seeds', tile_ids)
    attrs = {} if particle_mass is None else {'particle_mass': particle_mass}
    for t in tile_ids:
        tile_path = os.path.join(out_dir, f'tile_{t}.hdf5')
        for kind in ('particles', 'seeds'):
            bucket = os.path.join(out_dir, f'bucket_{kind}_{t}.hdf5')
            build_tile(bucket, tile_path, t, spec, kind, attrs= attrs)
            if not keep_buckets:
                os.remove(bucket)