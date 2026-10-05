"""
Tests for the data-source layer.

"""
import h5py
import numpy as np
import pytest

from oasis.calibration import calibrate
from oasis.catalogue import MiniBoxClassifier, run_orbiting_mass_assignment
from oasis.coordinates import relative_coordinates
from oasis.datasource import (LegacyMiniBoxDataSource, ParticleSet, SeedSet,
                              SpatialDataSource, BufferedTileDataSource)
from oasis.minibox import get_mini_box_id, process_simulation_data
from oasis.tiles import TileSpec, build_tiles

BOX, MINI, PAD, MP = 100.0, 25.0, 5.0, 1e10

def _grid(x):
    """
    Snap to (k+0.5)/1024: exact in float32, never exactly on a minibox edge,
    and shifts by whole miniboxes stay exact.

    """
    return (np.floor(np.mod(x,BOX) * 1024) + 0.5) / 1024

def _make_box(seed = 0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, BOX, (20,3))
    centers[:5, 0] = [0.3, 24.8, 25.2, 49.7, 99.7]   # near mini-box / periodic edges
    pos = [c + rng.normal(scale=0.2, size=(400, 3)) for c in centers]
    vel = [rng.normal(scale=80.0, size=(400, 3)) for _ in centers]
    pos.append(rng.uniform(0, BOX, (20000, 3)))
    vel.append(rng.normal(scale=300.0, size=(20000, 3)))
    pos = _grid(np.concatenate(pos))
    m200b = 4e12 * (1.0 + 0.01 * np.arange(len(centers)))   # distinct: no ties
    r200b = (3 * m200b / (4 * np.pi * 200 * 0.3 * 2.775e11)) ** (1 / 3)
    return dict(pos=pos, vel=np.concatenate(vel), pid=np.arange(len(pos)) + 1,
                s_pos=_grid(centers), s_vel=np.zeros((len(centers), 3)),
                hid=np.arange(len(centers)), m200b=m200b, r200b=r200b,
                rs=r200b / 5)

def _write_box(path, d):
    process_simulation_data(
        save_path=path, particle_type='seed', boxsize=BOX, minisize=MINI,
        positions=d['s_pos'].copy(), velocities=d['s_vel'].copy(), ids=d['hid'],
        mass=(d['m200b'], 'M200b'),
        data=([d['r200b'], d['rs']], ('R200b', 'Rs'), (np.float32, np.float32)),
        n_threads=1)
    process_simulation_data(
        save_path=path, particle_type='dm', boxsize=BOX, minisize=MINI,
        positions=d['pos'].copy(), velocities=d['vel'].copy(), ids=d['pid'],
        mass=(MP, 'mass'), n_threads=1)

class InMemoryDataSource(SpatialDataSource):
    """
    Serves regions straight from arrays without files. Test only.

    """

    def __init__(self, d, boxsize, minisize, padding):
        self.boxsize, self.minisize, self.padding = boxsize, minisize, padding
        self.n = int(np.ceil(boxsize / minisize))
        f32 = np.float32
        self.p = (d['pos'].astype(f32), d['vel'].astype(f32), d['pid'])
        self.s = (d['s_pos'].astype(f32), d['s_vel'].astype(f32), d['hid'],
                  d['r200b'].astype(f32), d['m200b'].astype(f32),
                  d['rs'].astype(f32))
        self.owner = get_mini_box_id(d['s_pos'].copy(), boxsize, minisize)

    def region_ids(self):
        return list(range(self.n**3))

    def _in_region(self, pos, region):
        ix, iy, iz = region % self.n, (region // self.n) % self.n, region // self.n**2
        center = (np.array([ix, iy, iz]) + 0.5) * self.minisize
        rel = relative_coordinates(pos, center, self.boxsize)
        return np.all(np.abs(rel) <= 0.5 * self.minisize + self.padding, axis=1)

    def load_seeds(self, region):
        m = self._in_region(self.s[0], region)
        return SeedSet(*(a[m] for a in self.s), in_core=(self.owner[m] == region))

    def load_particles(self, region):
        m = self._in_region(self.p[0], region)
        return ParticleSet(*(a[m] for a in self.p), mass=np.float32(MP))  

def _classify_region(clf):
    """
    Run one region up to percolation, without writing files.
    
    """
    clf._load_seeds_and_filter()
    if clf._early_exit_if_no_seeds():
        return None
    clf._compute_deltac(); clf._build_seed_tree(); clf._load_particles()
    clf._load_calibration_parameters(); clf._init_catalogue_dataframe()
    clf._process_all_seeds(); clf._percolation()
    return clf

def test_each_seed_owned_by_exactly_one_region(tmp_path):
    path = str(tmp_path) + '/'
    d = _make_box()
    d['s_pos'][0] = [25.0, 50.0, 75.0]    # exactly on mini-box edges
    d['s_pos'][1] = [0.0, 0.0, 0.0]       # exactly on the periodic edge
    _write_box(path, d)
    src = LegacyMiniBoxDataSource(path, BOX, MINI, 'dm', padding=PAD)
    owners = np.zeros(len(d['hid']), dtype=int)
    for region in src.region_ids():
        seeds = src.load_seeds(region)
        if len(seeds):
            np.add.at(owners, seeds.hid[seeds.in_core], 1)
    assert np.all(owners == 1)

def test_neighbour_region_keeps_global_coordinates(tmp_path):
    path = str(tmp_path) + '/'
    d = _make_box()                       # seed 0 sits at x ~ 0.3
    _write_box(path, d)
    src = LegacyMiniBoxDataSource(path, BOX, MINI, 'dm', padding=PAD)
    owner = get_mini_box_id(d['s_pos'][0].copy(), BOX, MINI)
    neighbour = owner - owner % 4 + 3     # same y, z cell; x cell at 75-100
    seeds = src.load_seeds(neighbour)
    k = np.flatnonzero(seeds.hid == 0)
    assert k.size == 1 and not seeds.in_core[k[0]]
    np.testing.assert_array_equal(seeds.pos[k[0]], d['s_pos'][0].astype(np.float32))

def test_in_memory_source_matches_legacy_files(tmp_path):
    path = str(tmp_path) + '/'
    d = _make_box()
    _write_box(path, d)
    calibrate(save_path=path, omega_m=0.3)
    kw = dict(min_num_part=10, boxsize=BOX, minisize=MINI, load_path=path,
              particle_type='dm', seed_prop_names=('M200b', 'R200b', 'Rs'),
              redshift=0.0, padding=PAD)
    mem = InMemoryDataSource(d, BOX, MINI, PAD)
    n_haloes = 0
    for region in mem.region_ids():
        a = _classify_region(MiniBoxClassifier(mini_box_id=region, run_name='a', **kw))
        b = _classify_region(MiniBoxClassifier(mini_box_id=region, run_name='b',
                                               data_source=mem, **kw))
        if a is None or b is None:
            assert a is None and b is None
            continue
        for col in ('Halo_ID', 'Morb', 'Norb', 'NSUBS', 'PID', 'M200b', 'R200b'):
            np.testing.assert_array_equal(a.haloes_perc[col].to_numpy(),
                                          b.haloes_perc[col].to_numpy())
        for lo, hi in zip(a.haloes_perc['LIDX'], a.haloes_perc['RIDX']):
            np.testing.assert_array_equal(np.sort(a.orb_pid_perc[lo:hi]),
                                          np.sort(b.orb_pid_perc[lo:hi]))
        n_haloes += len(a.haloes_perc)
    assert n_haloes > 0

def test_catalogue_is_invariant_under_periodic_shift(tmp_path):
    d = _make_box()
    shifted = dict(d, pos=np.mod(d['pos'] + 50.0, BOX),
                   s_pos=np.mod(d['s_pos'] + 50.0, BOX))
    cats = []
    for name, data in (('a', d), ('b', shifted)):
        (tmp_path / name).mkdir()
        path = str(tmp_path / name) + '/'
        _write_box(path, data)
        calibrate(save_path=path, omega_m=0.3)
        run_orbiting_mass_assignment(
            load_path=path, run_name='t', min_num_part=10, boxsize=BOX,
            minisize=MINI, padding=PAD, particle_type='dm', redshift=0.0,
            n_threads=1)
        with h5py.File(path + 'run_t/catalogue.hdf5') as f, \
                h5py.File(path + 'run_t/members.hdf5') as m:
            cats.append(({k: f[k][()] for k in f}, m['PID'][()]))
    (ca, pa), (cb, pb) = cats
    ia, ib = np.argsort(ca['Halo_ID']), np.argsort(cb['Halo_ID'])
    assert len(ia) > 0
    for key in ('Halo_ID', 'Norb', 'Morb', 'NSUBS', 'PID', 'M200b', 'R200b'):
        np.testing.assert_array_equal(ca[key][ia], cb[key][ib])
    np.testing.assert_array_equal(
        np.mod(ca['pos'][ia] + np.float32(50.0), np.float32(BOX)), cb['pos'][ib])
    for i, j in zip(ia, ib):
        np.testing.assert_array_equal(np.sort(pa[ca['LIDX'][i]:ca['RIDX'][i]]),
                                      np.sort(pb[cb['LIDX'][j]:cb['RIDX'][j]]))

###################### TILE TESTS #############################################

# 2 tiles per side (core 50), ribbon = padding = 5, cells = mini-boxes = 25.
TILE_SPEC = TileSpec(boxsize=BOX, tiles_per_side=2, buffer_width=PAD,
                     inner_cell_size=MINI)

def _build_tiles_from_box(d, out_dir, spec=TILE_SPEC):
    """Build tiles from the same synthetic box, with the same float32 types
    the mini-box writer uses."""
    f32 = np.float32
    parts = dict(ID=d['pid'], pos=d['pos'], vel=d['vel'].astype(f32))
    seeds = dict(ID=d['hid'], pos=d['s_pos'], vel=d['s_vel'].astype(f32),
                 M200b=d['m200b'].astype(f32), R200b=d['r200b'].astype(f32),
                 Rs=d['rs'].astype(f32))
    one_chunk = lambda arrays: (lambda: iter([arrays]))
    build_tiles(one_chunk(parts), one_chunk(seeds), out_dir, spec, particle_mass=MP)

def _tile_sources(tile_dir, spec=TILE_SPEC):
    return [BufferedTileDataSource(f'{tile_dir}/tile_{t}.hdf5', padding=PAD)
            for t in range(spec.n_tiles)]

def _matching_mini_box(tile, region):
    return int(get_mini_box_id(tile.region_center(region).copy(), BOX, MINI))

def test_tile_regions_are_core_cells(tmp_path):
    _build_tiles_from_box(_make_box(), str(tmp_path))
    tile = _tile_sources(tmp_path)[0]
    assert len(tile.region_ids()) == (tile.m - 2 * tile.ring) ** 3   # 2x2x2 core cells
    assert sorted(tile.region_ids_by_workload()) == sorted(tile.region_ids())

def test_tile_source_rejects_padding_wider_than_ribbon(tmp_path):
    _build_tiles_from_box(_make_box(), str(tmp_path))
    with pytest.raises(ValueError):
        BufferedTileDataSource(str(tmp_path / 'tile_0.hdf5'), padding=PAD + 1.0)

def test_tile_source_serves_same_data_as_mini_boxes(tmp_path):
    path = str(tmp_path) + '/'
    d = _make_box()
    _write_box(path, d)                                   # mini-box files
    _build_tiles_from_box(d, path + 'tiles')              # tiles of the same data
    legacy = LegacyMiniBoxDataSource(path, BOX, MINI, 'dm', padding=PAD)
    covered = []
    for tile in _tile_sources(path + 'tiles'):
        for region in tile.region_ids():
            box_id = _matching_mini_box(tile, region)
            covered.append(box_id)
            a, b = legacy.load_seeds(box_id), tile.load_seeds(region)
            assert len(a) == len(b)
            if len(a):
                ia, ib = np.argsort(a.hid), np.argsort(b.hid)
                for name in ('hid', 'pos', 'vel', 'm200b', 'r200b', 'rs', 'in_core'):
                    np.testing.assert_array_equal(np.asarray(getattr(a, name))[ia],
                                                  np.asarray(getattr(b, name))[ib])
            pa, pb = legacy.load_particles(box_id), tile.load_particles(region)
            ia, ib = np.argsort(pa.pid), np.argsort(pb.pid)
            for name in ('pid', 'pos', 'vel'):
                np.testing.assert_array_equal(getattr(pa, name)[ia], getattr(pb, name)[ib])
            assert pa.mass == pb.mass
    # every mini-box is matched by exactly one tile core cell
    assert sorted(covered) == list(range(int(BOX / MINI) ** 3))

def test_classifier_on_tiles_matches_mini_boxes(tmp_path):
    path = str(tmp_path) + '/'
    d = _make_box()
    _write_box(path, d)
    _build_tiles_from_box(d, path + 'tiles')
    calibrate(save_path=path, omega_m=0.3)
    kw = dict(min_num_part=10, boxsize=BOX, minisize=MINI, load_path=path,
              particle_type='dm', seed_prop_names=('M200b', 'R200b', 'Rs'),
              redshift=0.0, padding=PAD)
    n_haloes = 0
    for tile in _tile_sources(path + 'tiles'):
        for region in tile.region_ids():
            box_id = _matching_mini_box(tile, region)
            a = _classify_region(MiniBoxClassifier(mini_box_id=box_id, run_name='a', **kw))
            b = _classify_region(MiniBoxClassifier(mini_box_id=region, run_name='b',
                                                   data_source=tile, **kw))
            if a is None or b is None:
                assert a is None and b is None
                continue
            for col in ('Halo_ID', 'Morb', 'Norb', 'NSUBS', 'PID', 'M200b', 'R200b'):
                np.testing.assert_array_equal(a.haloes_perc[col].to_numpy(),
                                              b.haloes_perc[col].to_numpy())
            for lo, hi in zip(a.haloes_perc['LIDX'], a.haloes_perc['RIDX']):
                np.testing.assert_array_equal(np.sort(a.orb_pid_perc[lo:hi]),
                                              np.sort(b.orb_pid_perc[lo:hi]))
            n_haloes += len(a.haloes_perc)
    assert n_haloes > 0