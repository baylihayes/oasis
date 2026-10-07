"""Tests for the core+ribbon tile builder."""
import h5py
import numpy as np
import pytest

from oasis.tiles import (TileSpec, assign_to_tiles, build_tiles, 
                         local_cell_ids, wrap_positions)

L = 100.0

def _in_stored_region(pos, spec, tile_id):
    """
    Brute force check: is each position inside the tile's core or ribbon?
    
    """
    core_min = spec.core_bounds(tile_id)[:,0]
    u = np.mod(pos.astype(np.float64) - (core_min - spec.buffer_width), L)
    return np.all(u <= spec.core_size + 2 * spec.buffer_width, axis = 1)

def _chunks(arrays, size):
    def make():
        n = len(arrays['pos'])
        for a in range(0, n, size):
            yield {k: v[a:a + size] for k, v in arrays.items()}
    return make


@pytest.mark.parametrize("n_side", [2,4])
def test_assignment_matches_brute_force(n_side):
    spec = TileSpec(boxsize = L, tiles_per_side = n_side, buffer_width = 5.0,
                    inner_cell_size = L / n_side / 5)
    rng = np.random.default_rng(0)
    pos = wrap_positions(rng.uniform(0, L, (20000, 3)), L)
    pos[:4] = [[0.0, 0.0, 0.0], [L / n_side, 3.0, 99.0],
               [99.999, 50.0, 0.001], [L / n_side - 1e-4, 0.0, 0.0]]      # edges and corners
    rows, tiles, in_core = assign_to_tiles(pos, spec)
    # every row in exactly one core
    assert np.array_equal(np.bincount(rows[in_core], minlength = len(pos)),
                          np.ones(len(pos)))
    # (row, tile) pairs are exactly the brute-force membership
    got = set(zip(rows.tolist(), tiles.tolist()))
    want = {(r, t) for t in range(spec.n_tiles)
            for r in np.flatnonzero(_in_stored_region(pos, spec, t)).tolist()}
    assert got == want

def test_duplication_factor_matches_volume():
    spec = TileSpec(boxsize = L, tiles_per_side = 5, buffer_width = 2.0, inner_cell_size = 5.0)
    pos = wrap_positions(np.random.default_rng(1).uniform(0, L, (400_000, 3)), L)
    rows, _, _ = assign_to_tiles(pos, spec)
    expected = ((spec.core_size + 2 * spec.buffer_width) / spec.core_size) **3
    assert abs(len(rows) / len(pos) / expected - 1) < 0.01

def test_too_wide_ribbon_rejected():
    with pytest.raises(ValueError):
        TileSpec(boxsize = L, tiles_per_side = 2, buffer_width = 30.0, inner_cell_size = 10.0)

def test_built_tiles_are_complete_and_grouped(tmp_path):
    spec = TileSpec(boxsize = L, tiles_per_side = 2, buffer_width = 5.0, inner_cell_size = 10.0)
    rng = np.random.default_rng(2)
    n = 30000
    parts = dict(ID = np.arange(n), pos = rng.uniform(0, L, (n,3)),
                 vel = rng.normal(size = (n,3)).astype(np.float32))
    seeds = dict(ID = np.arange(500), pos = rng.uniform(0, L, (500,3)),
                 vel = np.zeros((500, 3), np.float32), M200b = np.full(500, 1e13),
                 R200b = np.full(500, 0.5, np.float32), Rs = np.full(500, 0.1, np.float32))
    build_tiles(_chunks(parts, 7000), _chunks(seeds, 200), str(tmp_path), spec, 
                particle_mass = 1e10)

    stored_pos = wrap_positions(parts['pos'], L)
    core_ids, seed_owner = [], np.zeros(500, int)
    for t in range(spec.n_tiles):
        with h5py.File(tmp_path / f'tile_{t}.hdf5') as f:
            assert f['tile_metadata'].attrs['particle_mass'] == 1e10
            p = f['particles']
            ids, pos, off = p['ID'][()], p['pos'][()], p['cell_offset'][()]
            # global coordinates preserved exactly
            np.testing.assert_array_equal(pos, stored_pos[ids])
            # exactly the rows that belong to this tile
            assert set(ids.tolist()) == set(
                np.flatnonzero(_in_stored_region(stored_pos, spec, t)).tolist()
            )
            # each cell slice holds only rows of that cell
            cells = local_cell_ids(pos, t, spec)
            for c in range(len(off) - 1):
                assert np.all(cells[off[c]:off[c + 1]] == c)
            core_ids.append(ids[p['in_core'][()]])
            s = f['seeds']
            np.add.at(seed_owner, s['ID'][()][s['in_core'][()]], 1)
    # every particle in exactly one core; every seed owned exactly once
    assert np.array_equal(np.sort(np.concatenate(core_ids)), np.arange(n))
    assert np.all(seed_owner == 1)
    assert not list(tmp_path.glob('bucket_*'))          # buckets cleaned up

def _make_inputs(n=5000, n_seeds=100, seed=4):
    rng = np.random.default_rng(seed)
    parts = dict(ID=np.arange(n), pos=rng.uniform(0, L, (n, 3)),
                 vel=rng.normal(size=(n, 3)).astype(np.float32))
    seeds = dict(ID=np.arange(n_seeds), pos=rng.uniform(0, L, (n_seeds, 3)),
                 vel=np.zeros((n_seeds, 3), np.float32), M200b=np.full(n_seeds, 1e13),
                 R200b=np.full(n_seeds, 0.5, np.float32), Rs=np.full(n_seeds, 0.1, np.float32))
    return parts, seeds

def _read_tiles(out_dir, spec):
    """All datasets of all tiles, keyed by (tile, group, name)."""
    out = {}
    for t in range(spec.n_tiles):
        with h5py.File(out_dir / f'tile_{t}.hdf5') as f:
            for kind in ('particles', 'seeds'):
                for name, ds in f[kind].items():
                    out[(t, kind, name)] = ds[()]
    return out

def test_assign_to_tiles_handles_empty_input():
    spec = TileSpec(boxsize=L, tiles_per_side=2, buffer_width=5.0, inner_cell_size=10.0)
    rows, tiles, in_core = assign_to_tiles(np.empty((0, 3), np.float32), spec)
    assert rows.size == tiles.size == in_core.size == 0

def test_empty_chunks_are_skipped(tmp_path):
    spec = TileSpec(boxsize=L, tiles_per_side=2, buffer_width=5.0, inner_cell_size=10.0)
    parts, seeds = _make_inputs()
    plain = _chunks(parts, 1000)

    def with_empty_chunks():
        yield {k: v[:0] for k, v in parts.items()}     # empty chunk first...
        yield from plain()
        yield {k: v[:0] for k, v in parts.items()}     # ...and last

    build_tiles(plain, _chunks(seeds, 50), str(tmp_path / 'a'), spec)
    build_tiles(with_empty_chunks, _chunks(seeds, 50), str(tmp_path / 'b'), spec)
    a, b = _read_tiles(tmp_path / 'a', spec), _read_tiles(tmp_path / 'b', spec)
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])

def test_no_input_chunks_raises(tmp_path):
    spec = TileSpec(boxsize=L, tiles_per_side=2, buffer_width=5.0, inner_cell_size=10.0)
    with pytest.raises(ValueError, match="No input chunks"):
        build_tiles(lambda: iter([]), lambda: iter([]), str(tmp_path), spec)

def test_existing_tiles_refused_unless_overwrite(tmp_path):
    spec = TileSpec(boxsize=L, tiles_per_side=2, buffer_width=5.0, inner_cell_size=10.0)
    parts, seeds = _make_inputs()
    args = (_chunks(parts, 1000), _chunks(seeds, 50), str(tmp_path), spec)
    build_tiles(*args)
    first = _read_tiles(tmp_path, spec)

    with pytest.raises(FileExistsError):
        build_tiles(*args)                      # refused, nothing touched
    assert _read_tiles(tmp_path, spec).keys() == first.keys()

    build_tiles(*args, overwrite=True)          # rebuilt from scratch
    second = _read_tiles(tmp_path, spec)
    assert second.keys() == first.keys()
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])

def test_velocities_stored_as_float32(tmp_path):
    """float64 input velocities are stored as float32, like the mini-box files."""
    spec = TileSpec(boxsize=L, tiles_per_side=2, buffer_width=5.0, inner_cell_size=10.0)
    parts, seeds = _make_inputs()
    rng = np.random.default_rng(5)
    parts['vel'] = rng.normal(scale=300.0, size=(len(parts['ID']), 3))   # float64
    seeds['vel'] = rng.normal(scale=300.0, size=(len(seeds['ID']), 3))   # float64
    build_tiles(_chunks(parts, 1000), _chunks(seeds, 50), str(tmp_path), spec)
    for t in range(spec.n_tiles):
        with h5py.File(tmp_path / f'tile_{t}.hdf5') as f:
            for kind, src in (('particles', parts), ('seeds', seeds)):
                group = f[kind]
                assert group['vel'].dtype == np.float32
                np.testing.assert_array_equal(
                    group['vel'][()], src['vel'][group['ID'][()]].astype(np.float32))

def test_rows_exactly_on_ribbon_edge_are_stored():
    """A row exactly buffer_width outside a core belongs to that tile's ribbon,
    on both sides: the region loader keeps |dx| <= size/2 + padding (inclusive),
    so the tiles must store those rows too."""
    spec = TileSpec(boxsize = L, tiles_per_side = 2, buffer_width = 5.0,
                    inner_cell_size = 10.0)
    # Tile 0 has core [0, 50)^3.
    pos = wrap_positions(np.array([
        [55.0, 20.0, 20.0],     # 5 above the core's upper x edge
        [95.0, 20.0, 20.0],     # 5 below the core's lower x edge (= -5, periodic)
        [20.0, 55.0, 20.0],     # same in y
        [20.0, 20.0, 95.0],     # same in z
    ]), L)
    rows, tiles, in_core = assign_to_tiles(pos, spec)
    in_tile0 = set(rows[(tiles == 0) & ~in_core].tolist())
    assert in_tile0 == {0, 1, 2, 3}


def test_tile_source_loads_rows_exactly_at_padding_edge(tmp_path):
    """BufferedTileDataSource must return the same particles as the region cut
    |x - center| <= size/2 + padding, including rows exactly on that boundary,
    for core cells next to a tile edge (and across the periodic wrap)."""
    from oasis.datasource import BufferedTileDataSource

    spec = TileSpec(boxsize = L, tiles_per_side = 2, buffer_width = 5.0,
                    inner_cell_size = 10.0)
    edge = np.array([
        [55.0, 5.0, 5.0],       # exactly padding past global cell (4,0,0) -> region 4
        [95.0, 5.0, 5.0],       # exactly padding past global cell (0,0,0) -> region 0
    ])
    rng = np.random.default_rng(1)
    pos = np.vstack([edge, rng.uniform(0, L, (5000, 3))])
    n = len(pos)
    parts = dict(ID = np.arange(n, dtype = np.int64), pos = pos,
                 vel = np.zeros((n, 3)))
    seeds = dict(ID = np.array([0], dtype = np.int64), pos = np.array([[25.0, 25.0, 25.0]]),
                 vel = np.zeros((1, 3)), M200b = np.array([1e12], dtype = np.float32),
                 R200b = np.array([0.5], dtype = np.float32),
                 Rs = np.array([0.1], dtype = np.float32))
    build_tiles(_chunks(parts, 2000), _chunks(seeds, 1), str(tmp_path), spec,
                particle_mass = 1.0)

    source = BufferedTileDataSource(str(tmp_path / 'tile_0.hdf5'), padding = 5.0)
    stored = wrap_positions(pos, L).astype(np.float64)
    for region_id, edge_id in ((4, 0), (0, 1)):
        loaded = set(source.load_particles(region_id).pid.tolist())
        # Expected: the same inclusive cut as the mini-box loader
        rel = np.mod(stored - source.region_center(region_id) + L / 2, L) - L / 2
        expected = np.flatnonzero(np.all(np.abs(rel) <= 5.0 + 5.0, axis = 1))
        assert edge_id in loaded
        assert loaded == set(expected.tolist())

 

def _read_tile_files(out_dir, tile_ids):
    """All datasets and metadata of the given tiles, keyed by (tile, group, name)."""
    out = {}
    for t in tile_ids:
        with h5py.File(out_dir / f'tile_{t}.hdf5') as f:
            for kind in ('particles', 'seeds'):
                for name, ds in f[kind].items():
                    out[(t, kind, name)] = ds[()]
            for name, value in f['tile_metadata'].attrs.items():
                out[(t, 'tile_metadata', name)] = np.asarray(value)
    return out


def _assert_same(a, b):
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_array_equal(a[key], b[key], err_msg=str(key))
        assert a[key].dtype == b[key].dtype, key


def test_single_tile_equals_tile_from_full_build(tmp_path):
    """A tile built on its own is identical to the same tile from a full
    build: same rows in the same order, same cell offsets and metadata."""
    spec = TileSpec(boxsize=L, tiles_per_side=2, buffer_width=5.0, inner_cell_size=10.0)
    parts, seeds = _make_inputs()
    build_tiles(_chunks(parts, 1000), _chunks(seeds, 50), str(tmp_path / 'all'), spec,
                particle_mass=1e10)
    build_tiles(_chunks(parts, 1000), _chunks(seeds, 50), str(tmp_path / 'one'), spec,
                particle_mass=1e10, tile_ids=[5])

    assert sorted(p.name for p in (tmp_path / 'one').iterdir()) == ['tile_5.hdf5']
    _assert_same(_read_tile_files(tmp_path / 'one', [5]),
                 _read_tile_files(tmp_path / 'all', [5]))


def test_building_in_groups_equals_one_call(tmp_path):
    """Building the tiles in several calls into one folder gives the same
    files as one call, and a call does not touch the tiles of other calls."""
    spec = TileSpec(boxsize=L, tiles_per_side=2, buffer_width=5.0, inner_cell_size=10.0)
    parts, seeds = _make_inputs()
    build_tiles(_chunks(parts, 1000), _chunks(seeds, 50), str(tmp_path / 'all'), spec)
    for group in ([0, 1, 2], [7, 3], [4, 6, 5]):     # any order, any grouping
        build_tiles(_chunks(parts, 1000), _chunks(seeds, 50), str(tmp_path / 'groups'),
                    spec, tile_ids=group)

    tiles = range(spec.n_tiles)
    _assert_same(_read_tile_files(tmp_path / 'groups', tiles),
                 _read_tile_files(tmp_path / 'all', tiles))
    assert not list((tmp_path / 'groups').glob('bucket_*'))


def test_existing_tiles_of_other_calls_do_not_block(tmp_path):
    """Only this call's own tile files make it refuse (without overwrite)."""
    spec = TileSpec(boxsize=L, tiles_per_side=2, buffer_width=5.0, inner_cell_size=10.0)
    parts, seeds = _make_inputs()
    args = (_chunks(parts, 1000), _chunks(seeds, 50), str(tmp_path), spec)
    build_tiles(*args, tile_ids=[0])
    before = _read_tile_files(tmp_path, [0])

    build_tiles(*args, tile_ids=[1])                 # different tile: allowed
    with pytest.raises(FileExistsError):
        build_tiles(*args, tile_ids=[1, 2])          # tile 1 exists: refused
    assert not (tmp_path / 'tile_2.hdf5').exists()   # nothing written
    _assert_same(_read_tile_files(tmp_path, [0]), before)


@pytest.mark.parametrize("bad", [[-1], [8], [0, 8]])
def test_invalid_tile_ids_rejected(tmp_path, bad):
    spec = TileSpec(boxsize=L, tiles_per_side=2, buffer_width=5.0, inner_cell_size=10.0)
    parts, seeds = _make_inputs()
    with pytest.raises(ValueError, match="tile_ids"):
        build_tiles(_chunks(parts, 1000), _chunks(seeds, 50), str(tmp_path), spec,
                    tile_ids=bad)
    assert not list(tmp_path.iterdir())              # nothing written


def _assign_to_tiles_reference(pos, spec):
    """The original assign_to_tiles: every offset tested on every row. Kept
    here to pin the exact output (pairs, order and dtypes) of the faster
    version, which tests the 26 ribbon offsets on rows near a core face only."""
    from itertools import product
    n, core, w = spec.tiles_per_side, spec.core_size, spec.buffer_width
    x = np.asarray(pos, dtype=np.float64)
    i0 = np.minimum((x // core).astype(np.int64), n - 1)
    t = x - i0 * core
    lo, hi = t <= w, t >= core - w
    rows_out, tiles_out, core_out = [], [], []
    for off in product((-1, 0, 1), repeat=3):
        mask = np.ones(len(x), dtype=bool)
        for axis, o in enumerate(off):
            if o == -1:
                mask &= lo[:, axis]
            elif o == 1:
                mask &= hi[:, axis]
        rows = np.flatnonzero(mask)
        if rows.size == 0:
            continue
        ijk = (i0[rows] + np.array(off)) % n
        rows_out.append(rows)
        tiles_out.append(ijk[:, 0] + ijk[:, 1] * n + ijk[:, 2] * n**2)
        core_out.append(np.full(rows.size, off == (0, 0, 0)))
    return (np.concatenate(rows_out), np.concatenate(tiles_out),
            np.concatenate(core_out))


_SPECS = [dict(boxsize=1000.0, tiles_per_side=5, buffer_width=7.5, inner_cell_size=25.0),
          dict(boxsize=L, tiles_per_side=2, buffer_width=5.0, inner_cell_size=10.0),
          dict(boxsize=L, tiles_per_side=4, buffer_width=5.0, inner_cell_size=5.0),
          dict(boxsize=L, tiles_per_side=3, buffer_width=10.0, inner_cell_size=L / 6)]


def _test_positions(spec, rng, n=50000):
    """Random rows plus rows exactly on every core edge and ribbon edge."""
    c, w, box = spec.core_size, spec.buffer_width, spec.boxsize
    edges = np.mod([v for k in range(spec.tiles_per_side + 1)
                    for v in (k * c - w, k * c + w, k * c, (k + 1) * c - w)], box)
    grid = np.array(np.meshgrid(edges, edges, edges)).reshape(3, -1).T
    return wrap_positions(np.vstack([rng.uniform(0, box, (n, 3)), grid]), box)


@pytest.mark.parametrize("kw", _SPECS)
def test_assign_to_tiles_matches_reference_exactly(kw):
    """Same (row, tile, in_core) triples in the same order and dtypes as the
    original implementation, so tile files stay byte-for-byte identical."""
    spec = TileSpec(**kw)
    pos = _test_positions(spec, np.random.default_rng(5))
    for got, ref in zip(assign_to_tiles(pos, spec), _assign_to_tiles_reference(pos, spec)):
        assert got.dtype == ref.dtype
        np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("kw", _SPECS)
def test_candidate_rows_never_drop_a_needed_row(kw):
    """The pre-filter may keep extra rows but must keep every row that lies in
    the core or ribbon of a wanted tile, for any set of wanted tiles."""
    from oasis.tiles import _candidate_rows
    spec = TileSpec(**kw)
    rng = np.random.default_rng(6)
    pos = _test_positions(spec, rng)
    rows, tiles, _ = assign_to_tiles(pos, spec)
    for _ in range(25):
        wanted = np.zeros(spec.n_tiles, dtype=bool)
        wanted[rng.choice(spec.n_tiles, rng.integers(1, spec.n_tiles + 1),
                          replace=False)] = True
        cand = _candidate_rows(pos, spec, wanted)
        if cand is None:                     # nothing filtered: trivially safe
            continue
        needed = np.unique(rows[wanted[tiles]])
        assert np.isin(needed, cand).all()
    # A single tile keeps only a small fraction of a large box (the speed-up).
    if spec.tiles_per_side == 5:
        one = np.zeros(spec.n_tiles, dtype=bool)
        one[0] = True
        assert len(_candidate_rows(pos, spec, one)) < 0.05 * len(pos)
