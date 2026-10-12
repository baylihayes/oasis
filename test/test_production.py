"""Tests for restartable tiled production runs (process_tile, merge_tiled_run)."""
import json
import os

import h5py
import numpy as np
import pytest

from oasis.calibration import calibrate
from oasis.catalogue import merge_tiled_run, process_tile, run_tiled_orbiting_mass_assignment

from test.test_datasource import (BOX, PAD, TILE_SPEC, _build_tiles_from_box,
                                  _make_box, _write_box)

COMMON = dict(min_num_part=10, boxsize=BOX, padding=PAD, particle_type='dm',
              redshift=0.0, n_threads=1)


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    """A box with tiles and calibration, plus a reference uninterrupted run."""
    path = str(tmp_path_factory.mktemp('prod')) + '/'
    d = _make_box()
    _write_box(path, d)
    _build_tiles_from_box(d, path + 'tiles')
    calibrate(save_path=path, omega_m=0.3)
    tiles = [path + f'tiles/tile_{t}.hdf5' for t in range(TILE_SPEC.n_tiles)]
    run_tiled_orbiting_mass_assignment(tile_paths=tiles, load_path=path,
                                       run_name='ref', **COMMON)
    return path, tiles


def _read(path, run):
    out = {}
    for name in ('catalogue', 'members'):
        with h5py.File(path + f'run_{run}/{name}.hdf5') as f:
            for k in f:
                out[(name, k)] = f[k][()]
    return out


def _assert_same(a, b):
    assert a.keys() == b.keys()
    for k in a:
        np.testing.assert_array_equal(a[k], b[k], err_msg=str(k))


def test_restart_processes_only_unfinished_tiles(prepared):
    """Tiles done before an interruption are skipped on restart, and the
    result equals an uninterrupted run."""
    path, tiles = prepared
    for t in tiles[:3]:                                  # "job 1" dies after 3 tiles
        process_tile(t, load_path=path, run_name='restart', **COMMON)
    done_dir = path + 'run_restart/tiles_done/'
    before = {n: os.path.getmtime(done_dir + n) for n in os.listdir(done_dir)}
    assert len(before) == 3

    with pytest.raises(RuntimeError, match="not finished"):
        merge_tiled_run(path, 'restart')                 # refuses a partial run

    run_tiled_orbiting_mass_assignment(tile_paths=tiles, load_path=path,
                                       run_name='restart', **COMMON)
    for n, t in before.items():                          # finished tiles untouched
        assert os.path.getmtime(done_dir + n) == t
    _assert_same(_read(path, 'restart'), _read(path, 'ref'))

    with open(path + 'run_restart/manifest.json') as f:
        manifest = json.load(f)
    assert manifest['verify'] == 'passed'
    assert manifest['n_tiles'] == TILE_SPEC.n_tiles == len(manifest['tiles'])
    assert manifest['settings']['padding'] == PAD
    assert len(manifest['settings']['calibration_sha256']) == 64
    assert manifest['oasis_uncommitted_changes'] in (True, False, None)
    with h5py.File(path + 'run_restart/catalogue.hdf5') as f:
        assert manifest['n_haloes'] == len(f['Halo_ID'])


def test_rerun_removes_leftover_cell_files(prepared):
    """A cell file left by an interrupted attempt is deleted before the tile is
    processed again, so it cannot reach the merge."""
    path, tiles = prepared
    first = process_tile(tiles[0], load_path=path, run_name='leftover', **COMMON)
    cell_dir = path + 'run_leftover/mini_box_catalogues/'
    stale = cell_dir + f"{first['region_ids'][0]}.hdf5"
    with h5py.File(stale, 'w') as f:                     # fake output of a crash
        f.create_dataset('halo/Halo_ID', data=np.array([999999]))
    process_tile(tiles[0], load_path=path, run_name='leftover', redo=True, **COMMON)
    if os.path.exists(stale):                            # rewritten, not the fake
        with h5py.File(stale) as f:
            assert 999999 not in f['halo/Halo_ID'][()]


def test_finished_tile_with_other_settings_is_refused(prepared):
    path, tiles = prepared
    process_tile(tiles[1], load_path=path, run_name='settings', **COMMON)
    other = dict(COMMON, padding=PAD / 2)
    with pytest.raises(ValueError, match="different settings"):
        process_tile(tiles[1], load_path=path, run_name='settings', **other)


def test_merge_refuses_mixed_settings(prepared):
    path, tiles = prepared
    for t in tiles[:-1]:
        process_tile(t, load_path=path, run_name='mixed', **COMMON)
    process_tile(tiles[-1], load_path=path, run_name='mixed',
                 **dict(COMMON, min_num_part=20))
    with pytest.raises(RuntimeError, match="different settings"):
        merge_tiled_run(path, 'mixed')


def test_merge_refuses_stray_cell_files(prepared):
    path, tiles = prepared
    for t in tiles:
        process_tile(t, load_path=path, run_name='stray', **COMMON)
    with h5py.File(path + 'run_stray/mini_box_catalogues/123456.hdf5', 'w') as f:
        f.create_dataset('halo/Halo_ID', data=np.array([1]))
    with pytest.raises(RuntimeError, match="do not belong"):
        merge_tiled_run(path, 'stray')


def test_partial_merge_holds_exactly_the_finished_tiles_haloes(prepared):
    """partial=True merges only the finished tiles: the catalogue holds exactly
    the haloes of those tiles' cores, identical to the full run, and the
    manifest says the run is partial."""
    path, tiles = prepared
    done = [0, 3, 5]
    for t in done:
        process_tile(tiles[t], load_path=path, run_name='partial', **COMMON)
    with pytest.raises(RuntimeError, match="partial=True"):
        merge_tiled_run(path, 'partial')
    manifest = merge_tiled_run(path, 'partial', partial=True)

    assert manifest['partial'] is True
    assert manifest['tiles_missing'] == sorted(set(range(TILE_SPEC.n_tiles)) - set(done))
    assert [t['tile_id'] for t in manifest['tiles']] == done

    # Haloes of the full run whose centre lies in one of the finished cores
    with h5py.File(path + 'run_ref/catalogue.hdf5') as f:
        ref_id, ref_norb, ref_pos = f['Halo_ID'][()], f['Norb'][()], f['pos'][()]
    core = TILE_SPEC.core_size
    ijk = np.minimum((np.mod(ref_pos, BOX) // core).astype(int),
                     TILE_SPEC.tiles_per_side - 1)
    tile_of = ijk[:, 0] + ijk[:, 1] * 2 + ijk[:, 2] * 4
    keep = np.isin(tile_of, done)
    with h5py.File(path + 'run_partial/catalogue.hdf5') as f:
        got = dict(zip(f['Halo_ID'][()].tolist(), f['Norb'][()].tolist()))
    assert len(got) == keep.sum() > 0
    assert got == dict(zip(ref_id[keep].tolist(), ref_norb[keep].tolist()))


def test_ball_radius_solver_recorded_and_not_mixed(prepared):
    """The solver is part of the run settings: it is written to the manifest,
    and a tile cannot be reprocessed into the same run with the other solver."""
    path, tiles = prepared
    for t in tiles:
        process_tile(t, load_path=path, run_name='brentq', ball_radius_solver='brentq',
                     **COMMON)
    manifest = merge_tiled_run(path, 'brentq')
    assert manifest['settings']['ball_radius_solver'] == 'brentq'
    assert manifest['verify'] == 'passed'
    with open(path + 'run_ref/manifest.json') as f:
        assert json.load(f)['settings']['ball_radius_solver'] == 'fsolve'
    with pytest.raises(ValueError, match="different settings"):
        process_tile(tiles[0], load_path=path, run_name='brentq', **COMMON)
