"""Build core+ribbon tiles for very large simulations (MDPL2)

The box is split into tiles_per_side**3 cubic cores. Each tile stores the rows
of its core plus a ribbon of width 'buffer_width' around it, always in global
coordinates (nothing is shifted or wrapped per tile). Inside a tile, rows are
grouped by inner cell, with an offset table so a cell is one contiguous slice.

"""
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
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

    With tiles_per_side = 1 the single tile is the whole periodic box: no
    ribbon is stored (buffer_width must be 0) and a cell's padding is read
    from the opposite side of the same tile, wrapping around the box. The
    cells then load exactly the particles and seeds of mini-boxes of size
    inner_cell_size with the same padding, so the catalogue equals the
    mini-box pipeline's.

    Raises
    ------
    ValueError
        If tiles_per_side is < 1, if inner_cell_size <=0, if buffer_width < 0 
        or greater than the core size, if the ribbon is so wide that a tile would 
        overlap itself across the periodic boundary, if tiles_per_side is 1 and
        buffer_width is not 0, or if inner_cell_size does not divide the core
        size.
    """

    boxsize: float          #global periodic box size, e.g. 1000.0 Mpc / h
    tiles_per_side: int     # e.g. 5 -> 125 tiles
    buffer_width: float     # ribbon width w
    inner_cell_size: float # e.g. 25.0 Mpc / h

    def __post_init__(self):
        if self.tiles_per_side < 1 or self.inner_cell_size <= 0:
            raise ValueError("tiles_per_side must be >= 1 and "
            "inner_cell_size > 0.")
        if self.buffer_width < 0:
            raise ValueError("buffer_width must be >= 0.")
        if self.tiles_per_side == 1:
            # The one tile is the whole box: its neighbours are itself, so a
            # ribbon would only duplicate its own rows.
            if self.buffer_width != 0:
                raise ValueError("with tiles_per_side = 1 the tile is the whole periodic "
                                 "box and stores no ribbon: set buffer_width = 0 (the "
                                 "padding wraps around the box).")
        else:
            if self.buffer_width > self.core_size:
                raise ValueError("buffer_width must not exceed the core size "
                                 "(boxsize / tiles_per_side).")
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
    pos = np.asarray(pos)
    # Fast path: float32 input already inside [0, boxsize) is returned as is;
    # the general path would not change such values.
    if pos.dtype == np.float32 and pos.size and pos.min() >= 0 and pos.max() < boxsize:
        return pos
    pos = np.mod(np.asarray(pos, dtype=np.float64), boxsize)
    pos[pos.astype(np.float32).astype(np.float64) >= boxsize] = 0.0
    return pos.astype(np.float32)

def assign_to_tiles(pos: np.ndarray, spec: TileSpec):
    """
    Every (row, tile) pair where the position lies in that tile's core or
    ribbon. Returns (rows, tiles, in_core); each row has exactly one in_core
    pair, plus one pair for each ribbon it falls in (up to 7 near corners).
    With tiles_per_side = 1 every row is in the core of tile 0 and in no ribbon.
    
    """
    n, core, w = spec.tiles_per_side, spec.core_size, spec.buffer_width
    x = np.asarray(pos, dtype=np.float64)

    if x.shape[0] == 0:
        # Nothing to assign (e.g. an empty input chunk).
        return (np.empty(0, dtype = np.int64), np.empty(0, dtype = np.int64),
                np.empty(0, dtype = bool))
    if n == 1:
        # Single periodic tile: no ribbon (a row at a core face would otherwise
        # be copied into the tile's own "neighbour", i.e. itself).
        return (np.arange(len(x), dtype=np.int64), np.zeros(len(x), dtype=np.int64),
                np.ones(len(x), dtype=bool))

    i0 = np.minimum((x // core).astype(np.int64), n - 1)    # core tile per axis
    t = x - i0 * core                                       # position inside core
    lo = t <= w              # also inside the ribbon of the tile below
    hi = t >= core - w      # also inside the ribbon of the tile above

    # Every row is in its own core; only rows near a core face can also be in
    # a ribbon, so the 26 ribbon offsets are tested on those rows only. Pairs
    # come out in the same order as testing every offset on every row.
    near = np.flatnonzero(np.any(lo | hi, axis=1))
    lo_n, hi_n, i0_n = lo[near], hi[near], i0[near]
    core_tile = i0[:, 0] + i0[:, 1] * n + i0[:, 2] * n**2
    rows_out, tiles_out, core_out = [], [], []
    for off in product((-1, 0, 1), repeat=3):
        if off == (0, 0, 0):
            rows_out.append(np.arange(len(x), dtype=np.int64))
            tiles_out.append(core_tile)
            core_out.append(np.ones(len(x), dtype=bool))
            continue
        mask = np.ones(len(near), dtype=bool)
        for axis, o in enumerate(off):
            if o == -1:
                mask &= lo_n[:, axis]
            elif o == 1:
                mask &= hi_n[:, axis]
        if not mask.any():
            continue
        rows = near[mask]
        ijk = (i0_n[mask] + np.array(off)) % n                # periodic wrap
        rows_out.append(rows)
        tiles_out.append(ijk[:, 0] + ijk[:, 1] * n + ijk[:, 2] * n**2)
        core_out.append(np.zeros(rows.size, dtype=bool))
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


class ChunkTasks:
    """
    A chunk factory whose chunks can be read in parallel: chunk k is
    ``load(tasks[k])``. Calling it returns the chunks in order, so it can be
    used anywhere a plain chunk factory is accepted; build_tiles(n_workers > 1)
    reads and assigns several chunks at once. ``load`` must be a module-level
    function (it is sent to worker processes), e.g. one snapshot file per task.
    """
    def __init__(self, load: Callable[[object], dict], tasks):
        self.load = load
        self.tasks = list(tasks)

    def __call__(self) -> Iterator[dict]:
        return (self.load(t) for t in self.tasks)

def _candidate_rows(pos: np.ndarray, spec: TileSpec, wanted: np.ndarray):
    """
    Rows that may lie in the core or ribbon of a wanted tile, or None if every
    row may (all tiles wanted along every axis).

    A cheap, conservative pre-filter (it may keep extra rows, never drops a needed
    one): per axis, a row is kept if its coordinate is within buffer_width, plus one
    lookup bin of margin, of the core of some wanted tile along that axis. 
    assign_to_tiles then decides exactly.
    """
    n, core, w, L = (spec.tiles_per_side, spec.core_size, spec.buffer_width,
                     spec.boxsize)
    nb = 64 * n                         # lookup bins per axis
    g = L / nb                          # bin width; nb * g == L, so bins wrap exactly
    idx = spec.tile_index(np.flatnonzero(wanted))
    keep = None
    for axis in range(3):
        allowed = np.zeros(nb, dtype = bool)
        for k in np.unique(idx[axis]):
            first = int(np.floor((k * core - w) / g)) - 1
            last = int(np.floor(((k + 1) * core + w) / g)) + 1
            allowed[np.arange(first, last + 1) % nb] = True
        if allowed.all():
            continue
        b = np.minimum((pos[:, axis] * np.float32(1.0 / g)).astype(np.int64), nb - 1)
        keep = allowed[b] if keep is None else keep & allowed[b]
    return None if keep is None else np.flatnonzero(keep)

def _interior_tile(pos: np.ndarray, spec: TileSpec):
    """
    Tile whose core holds every row of `pos` in a chunk with no row in any ribbon, or None.

    Then assign_to_tiles would return exactly (arange(n), that tile, all True):
    t = x - i0 * core is monotonic in x for a fixed i0, so the rows' extreme
    values decide the ribbon tests (t <= w, t >= core - w) for all rows.
    """
    if pos.shape[0] == 0:
        return None
    n, core, w = spec.tiles_per_side, spec.core_size, spec.buffer_width
    lo = np.asarray(pos.min(axis=0), dtype=np.float64)
    hi = np.asarray(pos.max(axis=0), dtype=np.float64)
    i_lo = np.minimum((lo // core).astype(np.int64), n - 1)
    i_hi = np.minimum((hi // core).astype(np.int64), n - 1)
    if np.any(i_lo != i_hi):
        return None
    if np.any(lo - i_lo * core <= w) or np.any(hi - i_lo * core >= core - w):
        return None
    return int(i_lo[0] + i_lo[1] * n + i_lo[2] * n**2)


def array_chunks(arrays: dict, chunk_rows: int = 10_000_000) -> ChunkFactory:
    """
    Chunk factory over arrays that are already in memory.

    Parameters
    ----------
    arrays : dict
        Column name -> array, all of the same length, with at least 'pos'
        (n, 3); e.g. dict(ID=..., pos=..., vel=...) for particles, plus the
        seed properties ('M200b', 'R200b', 'Rs') for seeds.
    chunk_rows : int, default=10_000_000
        Rows per chunk (bounds the memory used while building tiles).

    Returns
    -------
    callable
        Returns a new iterator over chunks (slices of the arrays) each call,
        as build_tiles requires.

    Raises
    ------
    ValueError
        If 'pos' is missing or the arrays have different lengths.
    """
    if 'pos' not in arrays:
        raise ValueError("arrays must contain 'pos'")
    n = len(arrays['pos'])
    if any(len(v) != n for v in arrays.values()):
        raise ValueError("all arrays must have the same length")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be >= 1")

    def make():
        for a in range(0, n, chunk_rows):
            yield {k: v[a:a + chunk_rows] for k, v in arrays.items()}
    return make


def _rows_for_tiles(pos: np.ndarray, spec: TileSpec, wanted: np.ndarray):
    """(rows, tiles, in_core) of one chunk's wrapped positions, as from
    assign_to_tiles on the pre-filtered rows (rows index the chunk)."""
    t = _interior_tile(pos, spec)
    if t is not None and wanted[t]:
        # Every row is in the core of one wanted tile: the pre-filter keeps all
        # rows and assign_to_tiles returns exactly this.
        n = pos.shape[0]
        return (np.arange(n, dtype=np.int64), np.full(n, t, dtype=np.int64),
                np.ones(n, dtype=bool))
    cand = _candidate_rows(pos, spec, wanted)
    if cand is None:
        return assign_to_tiles(pos, spec)
    rows, tiles, in_core = assign_to_tiles(pos[cand], spec)
    return cand[rows], tiles, in_core


def _chunk_fields(chunk: dict) -> dict:
    """Column name -> (dtype, trailing shape) as stored in the buckets."""
    fields = {k: (np.asarray(v).dtype, np.shape(v)[1:]) for k, v in chunk.items()}
    fields['pos'] = (np.dtype(np.float32), (3,))
    if 'vel' in fields:
        # Velocities are stored as float32, like the minibox files.
        fields['vel'] = (np.dtype(np.float32), (3,))
    return fields


def _count_chunk(chunk: dict, spec: TileSpec, wanted: np.ndarray):
    """Pass 1 for one chunk: rows per tile, and the chunk's columns."""
    pos = wrap_positions(chunk['pos'], spec.boxsize)
    _, tiles, _ = _rows_for_tiles(pos, spec, wanted)
    return np.bincount(tiles, minlength=spec.n_tiles), _chunk_fields(chunk)


def _pieces_chunk(chunk: dict, spec: TileSpec, wanted: np.ndarray, names):
    """Pass 2 for one chunk: [(tile, {column: rows}, in_core)] for the wanted
    tiles, in increasing tile order and in chunk order within the tile."""
    chunk = dict(chunk, pos=wrap_positions(chunk['pos'], spec.boxsize))
    if 'vel' in chunk:
        chunk['vel'] = np.asarray(chunk['vel'], dtype=np.float32)
    rows, tiles, in_core = _rows_for_tiles(chunk['pos'], spec, wanted)
    keep = wanted[tiles]
    rows, tiles, in_core = rows[keep], tiles[keep], in_core[keep]
    order = np.argsort(tiles, kind='stable')
    rows, tiles, in_core = rows[order], tiles[order], in_core[order]
    bounds = np.searchsorted(tiles, np.arange(spec.n_tiles + 1))
    pieces = []
    for t in np.flatnonzero(np.diff(bounds)):
        sel = slice(bounds[t], bounds[t + 1])
        r = rows[sel]
        pieces.append((int(t), {k: np.asarray(chunk[k])[r] for k in names}, in_core[sel]))
    return pieces


def _count_task(args):
    """
    Wrapper for _count_chunk() to run inside a worker process.

    See also
    --------
    _count_chunk(chunk: dict, spec: TileSpec, wanted: np.ndarray)
    """
    load, task, spec, wanted = args
    return _count_chunk(load(task), spec, wanted)


def _pieces_task(args):
    """
    Wrapper for _pieces_chunk() to run inside a worker process.

    See also
    --------
    _pieces_chunk(chunk: dict, spec: TileSpec, wanted: np.ndarray, names)
    """
    load, task, spec, wanted, names = args
    return _pieces_chunk(load(task), spec, wanted, names)


def _in_order(fn, jobs, n_workers):
    """fn(job) for each job, yielded in job order. With n_workers > 1 the jobs
    run in worker processes, at most 2 * n_workers ahead of the consumer (so
    finished results waiting to be written stay bounded in memory)."""
    if n_workers <= 1:
        for job in jobs:
            yield fn(job)
        return
    with ProcessPoolExecutor(n_workers) as pool:
        pending = deque()
        for job in jobs:
            pending.append(pool.submit(fn, job))
            if len(pending) >= 2 * n_workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def bucket_into_tiles(make_chunks: ChunkFactory, out_dir: str, spec: TileSpec,
                      kind: str, tile_ids = None, n_workers: int = 1) -> np.ndarray:
    """
    Stream input chunks into one unsorted bucket file per tile.
    
    Each chunk is a dict of equal-length arrays and must contain 'pos'. 
    make_chunks() is called twice: first to count rows per tile, then to write
    them into preallocated datasets. Returns the number of rows per tile.
    Only the tiles in tile_ids (default: all) get a bucket file, and rows that
    cannot reach them are skipped early; the counts returned are complete only
    for those tiles.

    With n_workers > 1 and a ChunkTasks factory, chunks are read and assigned
    to tiles in parallel worker processes; they are still written in chunk
    order, so the bucket files are identical to a serial run. Other factories
    are always processed serially.
    
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
    parallel = n_workers > 1 and isinstance(make_chunks, ChunkTasks)

    # Pass 1: count rows per tile
    if parallel:
        results = _in_order(_count_task, [(make_chunks.load, task, spec, wanted)
                                          for task in make_chunks.tasks], n_workers)
    else:
        results = (_count_chunk(chunk, spec, wanted) for chunk in make_chunks())
    counts = np.zeros(spec.n_tiles, dtype = np.int64)
    fields = None
    for chunk_counts, chunk_fields in results:
        counts += chunk_counts
        if fields is None:
            fields = chunk_fields
    if fields is None:
        raise ValueError(f"No input chunks for '{kind}': make_chunks() yielded nothing. "
                         "Check input path.")
    names = list(fields)

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

        # Pass 2: write each chunk's rows at the end of its tiles' buckets
        if parallel:
            results = _in_order(_pieces_task, [(make_chunks.load, task, spec, wanted, names)
                                               for task in make_chunks.tasks], n_workers)
        else:
            results = (_pieces_chunk(chunk, spec, wanted, names) for chunk in make_chunks())
        cursor = np.zeros(spec.n_tiles, dtype = np.int64)
        for pieces in results:
            for t, columns, in_core in pieces:
                c0, n = cursor[t], in_core.size
                for k in names:
                    dsets[t][k][c0:c0 + n] = columns[k]
                dsets[t]['in_core'][c0:c0 + n] = in_core
                cursor[t] += n
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

        # Pass 1: rows per cell. Each block's cell IDs are kept for pass 2, in
        # the smallest integer type that holds them (see _cell_dtype).
        cell_dtype = _cell_dtype(spec)
        counts = np.zeros(m3, dtype = np.int64)
        cells = []
        for b in blocks:
            cells.append(local_cell_ids(src['pos'][b], tile_id, spec).astype(cell_dtype))
            counts += np.bincount(cells[-1], minlength = m3)

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
        for b, cell in zip(blocks, cells):                # pass 2: fill cell slices
            order = np.argsort(cell, kind = 'stable')
            cell = cell[order]
            # np.take reorders rows faster than fancy indexing (same result).
            data = {k: np.take(src[k][b], order, axis = 0) for k in src.keys()}
            starts = np.searchsorted(cell, np.arange(m3 + 1))
            for c in np.flatnonzero(np.diff(starts)):
                lo, hi = starts[c], starts[c+1]
                for k in out:
                    out[k][cursor[c]:cursor[c] + hi - lo] = data[k][lo:hi]
                cursor[c] += hi - lo

def build_tiles(make_particle_chunks: ChunkFactory, make_seed_chunks: ChunkFactory, 
                out_dir: str, spec: TileSpec, particle_mass: float | None = None,
                keep_buckets: bool = False, overwrite: bool = False,
                tile_ids = None, n_workers: int = 1) -> None:
    """
    Builds the tiles in tile_ids (default: all), so a large box can be built in groups
    of tiles: each call reads the whole input twice, but bucket files (as large as the 
    tiles themselves) exist only for the tiles being built. Refuses to run if any
    of this call's tile or bucket files already exist, unless overwrite = True, in 
    which case only those files are deleted first. Tiles of other calls in the same
    out_dir are left alone. With one tile (tiles_per_side = 1), the rows are
    written straight into the tile, without bucket files; the tile is the same.

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
        Keep the intermediate bucket files (for debugging). One tile has none.
    overwrite : bool, default=False
        Delete this call's existing tile/bucket files in out_dir first.
    tile_ids : iterable of int, optional
        Tiles to build (0 <= id < spec.n_tiles). Default: all tiles. 
    n_workers : int, default=1
        Worker processes. Chunks of ChunkTasks factories are read and assigned
        to tiles in parallel (written in order), and tiles are sorted into
        cells in parallel (with one tile, each chunk is sorted into cells by
        its worker). The tile files are identical for any n_workers.

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


    attrs = {} if particle_mass is None else {'particle_mass': particle_mass}
    if spec.n_tiles == 1:
        # One tile: every row belongs to it, so the rows are written straight
        # into the tile, without bucket files (same tile file).
        tile_path = os.path.join(out_dir, 'tile_0.hdf5')
        _write_single_tile(make_particle_chunks, tile_path, spec, 'particles',
                           attrs, n_workers)
        _write_single_tile(make_seed_chunks, tile_path, spec, 'seeds', attrs,
                           n_workers)
        return

    bucket_into_tiles(make_particle_chunks, out_dir, spec, 'particles', tile_ids,
                      n_workers=n_workers)
    bucket_into_tiles(make_seed_chunks, out_dir, spec, 'seeds', tile_ids,
                      n_workers=n_workers)
    # Each tile is sorted into its own file, so tiles can be sorted in parallel.
    jobs = [(out_dir, t, spec, attrs, keep_buckets) for t in tile_ids]
    for _ in _in_order(_sort_tile_task, jobs, min(n_workers, len(jobs))):
        pass


def _cell_dtype(spec: TileSpec):
    """Smallest integer type holding the cell IDs of a tile (a stable sort of
    int16 is a radix sort, much faster than for int64; the order is the same)."""
    return np.int16 if spec.cells_per_side**3 <= np.iinfo(np.int16).max else np.int32


def _single_tile_cells(chunk: dict, spec: TileSpec):
    """One chunk of a one-tile box: its columns as stored in the tile and the
    cell ID of each row. With one tile, every row is in its core, in chunk
    order (as assign_to_tiles returns for tiles_per_side = 1)."""
    columns = {k: np.asarray(v) for k, v in chunk.items()}
    columns['pos'] = wrap_positions(chunk['pos'], spec.boxsize)
    if 'vel' in columns:
        columns['vel'] = np.asarray(columns['vel'], dtype=np.float32)
    columns['in_core'] = np.ones(len(columns['pos']), dtype=bool)
    cell = local_cell_ids(columns['pos'], 0, spec).astype(_cell_dtype(spec))
    return columns, cell


def _single_tile_count_task(args):
    """Pass 1 for one chunk of a one-tile box: rows per cell and the chunk's columns."""
    load, task, spec = args
    chunk = load(task) if load is not None else task
    pos = wrap_positions(chunk['pos'], spec.boxsize)
    cell = local_cell_ids(pos, 0, spec).astype(_cell_dtype(spec))
    return np.bincount(cell, minlength=spec.cells_per_side**3), _chunk_fields(chunk)


def _single_tile_rows_task(args):
    """Pass 2 for one chunk of a one-tile box: its rows sorted by cell (stable,
    so rows keep their input order within a cell) and rows per cell."""
    load, task, spec = args
    columns, cell = _single_tile_cells(load(task) if load is not None else task, spec)
    order = np.argsort(cell, kind='stable')
    # np.take reorders rows faster than fancy indexing (same result).
    columns = {k: np.take(v, order, axis=0) for k, v in columns.items()}
    return columns, np.bincount(cell, minlength=spec.cells_per_side**3)


def _write_single_tile(make_chunks: ChunkFactory, tile_path: str, spec: TileSpec,
                       kind: str, attrs: dict, n_workers: int = 1) -> None:
    """
    Writes group 'kind' of the tile file of a one-tile box straight from the
    input chunks, without bucket files. Pass 1 counts the rows per cell; pass 2
    writes each chunk's rows, sorted by cell, after the rows of earlier chunks
    in the same cell. The file is identical to bucket_into_tiles + build_tile.
    """
    m3 = spec.cells_per_side**3
    parallel = n_workers > 1 and isinstance(make_chunks, ChunkTasks)
    def run(task_fn):
        if parallel:
            return _in_order(task_fn, [(make_chunks.load, t, spec) for t in make_chunks.tasks],
                             n_workers)
        return (task_fn((None, chunk, spec)) for chunk in make_chunks())

    counts = np.zeros(m3, dtype=np.int64)               # pass 1: rows per cell
    fields = None
    for chunk_counts, chunk_fields in run(_single_tile_count_task):
        counts += chunk_counts
        if fields is None:
            fields = chunk_fields
    if fields is None:
        raise ValueError(f"No input chunks for '{kind}': make_chunks() yielded nothing. "
                         "Check input path.")
    fields['in_core'] = (np.dtype(bool), ())
    offsets = np.concatenate(([0], np.cumsum(counts)))

    os.makedirs(os.path.dirname(tile_path) or '.', exist_ok=True)
    with h5py.File(tile_path, 'a') as dst:
        if kind in dst:
            raise FileExistsError(f"{tile_path} already contains a '{kind}' group.")
        # Same metadata, datasets and order of writes as build_tile, so the
        # file is identical (datasets in name order, like a bucket file's keys).
        meta = dst.require_group('tile_metadata')
        meta.attrs.update(dict(
            global_boxsize=spec.boxsize, tiles_per_side = spec.tiles_per_side,
            tile_id = 0, tile_index = spec.tile_index(0),
            core_bounds = spec.core_bounds(0), buffer_width = spec.buffer_width,
            inner_cell_size = spec.inner_cell_size, cells_per_side = spec.cells_per_side,
            ring = spec.ring, **(attrs or {})
        ))
        grp = dst.create_group(kind)
        out = {k: grp.create_dataset(k, shape=(counts.sum(),) + tail, dtype=dt)
               for k, (dt, tail) in sorted(fields.items())}
        grp.create_dataset('cell_offset', data=offsets)

        cursor = offsets[:-1].copy()
        for columns, chunk_counts in run(_single_tile_rows_task):  # pass 2
            starts = np.concatenate(([0], np.cumsum(chunk_counts)))
            for c in np.flatnonzero(chunk_counts):
                lo, hi = starts[c], starts[c + 1]
                for k in out:
                    out[k][cursor[c]:cursor[c] + hi - lo] = columns[k][lo:hi]
                cursor[c] += hi - lo
        if not np.array_equal(cursor, offsets[1:]):
            raise RuntimeError("input changed between the two passes.")


def _sort_tile_task(args):
    """Stage 2 for one tile: both buckets into the tile file, then delete them."""
    out_dir, t, spec, attrs, keep_buckets = args
    tile_path = os.path.join(out_dir, f'tile_{t}.hdf5')
    for kind in ('particles', 'seeds'):
        bucket = os.path.join(out_dir, f'bucket_{kind}_{t}.hdf5')
        build_tile(bucket, tile_path, t, spec, kind, attrs=attrs)
        if not keep_buckets:
            os.remove(bucket)