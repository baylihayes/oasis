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
    return np.all(u < spec.core_size + 2 * spec.buffer_width, axis = 1)

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
 