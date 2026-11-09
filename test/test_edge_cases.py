"""Regression tests for edge cases fixed during the performance refactor."""
import numpy as np
import h5py
from scipy.spatial import cKDTree
import json
import pytest

from oasis import calibration, minibox
from oasis.catalogue import MiniBoxClassifier, run_orbiting_mass_assignment, _offset_indices, merge_catalogues
from oasis.common import StageTimer
from oasis.datasource import ParticleSet, SeedSet, SpatialDataSource


def _write_seeds(path, boxsize, minisize, pos, ids):
    n = len(ids)
    minibox.process_simulation_data(
        save_path=path, particle_type='seed', boxsize=boxsize, minisize=minisize,
        positions=pos, velocities=np.zeros((n, 3)), ids=ids,
        mass=(np.full(n, 1e13), 'M200b'),
        data=([np.full(n, 0.5), np.full(n, 0.1)], ('R200b', 'Rs'),
              (np.float32, np.float32)),
        n_threads=1)


def test_seed_with_halo_id_zero_is_not_skipped(tmp_path):
    """A mini-box whose only seed has ID 0 must not be treated as empty."""
    path = str(tmp_path) + '/'
    _write_seeds(path, 10.0, 10.0, np.array([[5.0, 5.0, 5.0]]), np.array([0]))
    clf = MiniBoxClassifier(
        mini_box_id=0, min_num_part=1, boxsize=10.0, minisize=10.0,
        load_path=path, run_name='t', particle_type='dm',
        seed_prop_names=('M200b', 'R200b', 'Rs'), redshift=0.0, padding=1.0)
    clf._load_seeds_and_filter()
    assert clf.n_seeds == 1
    assert not clf._early_exit_if_no_seeds()


def test_calibration_variable_mass_not_overwritten(tmp_path):
    """With variable particle masses, each seed's result must not depend on
    the seeds processed before it."""
    path = str(tmp_path) + '/'
    rng = np.random.default_rng(0)
    n = 2000
    pos = rng.uniform(0, 10, (n, 3))
    minibox.process_simulation_data(
        save_path=path, particle_type='gas', boxsize=10.0, minisize=10.0,
        positions=pos, velocities=rng.normal(size=(n, 3)),
        ids=np.arange(n), mass=(rng.uniform(1e9, 2e9, n), 'mass'), n_threads=1)
    seeds = np.array([[3.0, 3.0, 3.0], [7.0, 7.0, 7.0]])
    kwargs = dict(mini_box_id=0, r_max=2.0, boxsize=10.0, minisize=10.0,
                  load_path=path, particle_type='gas', mass_density=1e10,
                  redshift=0.0)
    both = calibration._get_candidate_seed_particle_data(
        position_seeds=seeds, velocity_seeds=np.zeros((2, 3)), **kwargs)
    second = calibration._get_candidate_seed_particle_data(
        position_seeds=seeds[1:], velocity_seeds=np.zeros((1, 3)), **kwargs)
    np.testing.assert_array_equal(both[:, -second.shape[1]:], second)


def test_no_duplicate_loading_with_two_cells_per_side(tmp_path):
    """With fewer than 3 mini-boxes per side, neighbouring mini-boxes repeat;
    each particle and seed must still be loaded only once."""
    path = str(tmp_path) + '/'
    rng = np.random.default_rng(1)
    n = 500
    minibox.process_simulation_data(
        save_path=path, particle_type='dm', boxsize=10.0, minisize=5.0,
        positions=rng.uniform(0, 10, (n, 3)), velocities=np.zeros((n, 3)),
        ids=np.arange(n), mass=(1e10, 'mass'), n_threads=1)
    _write_seeds(path, 10.0, 5.0, rng.uniform(0, 10, (50, 3)), np.arange(50))

    _, _, pid, _ = minibox.load_particles(0, 10.0, 5.0, path, 'dm', padding=1.0)
    assert len(pid) == len(np.unique(pid))

    _, _, hid, *_ = minibox.load_seeds(0, 10.0, 5.0, path, padding=1.0)
    assert len(hid) == len(np.unique(hid))

def test_large_halo_ids_survive_catalogue(tmp_path):
    """Halo and parent IDs above the 32-bit range must be written unchanged."""
    path = str(tmp_path) + '/'
    big = np.array([5_000_000_000, 5_000_000_001], dtype=np.int64)   # > 2**32
    # Host at the centre, a slower-moving companion inside it so it becomes a subhalo.
    _write_seeds(path, 10.0, 10.0, np.array([[5.0, 5.0, 5.0], [5.2, 5.0, 5.0]]), big)
    rng = np.random.default_rng(2)
    n = 3000
    pos = np.vstack([5.0 + rng.normal(scale=0.15, size=(n, 3)),
                     np.array([5.2, 5.0, 5.0]) + rng.normal(scale=0.03, size=(300, 3))])
    minibox.process_simulation_data(
        save_path=path, particle_type='dm', boxsize=10.0, minisize=10.0,
        positions=np.mod(pos, 10.0), velocities=rng.normal(scale=50.0, size=(len(pos), 3)),
        ids=np.arange(len(pos)), mass=(1e10, 'mass'), n_threads=1)
    calibration.calibrate(save_path=path, omega_m=0.3)
    run_orbiting_mass_assignment(
        load_path=path, run_name='t', min_num_part=20, boxsize=10.0, minisize=10.0,
        padding=1.0, particle_type='dm', redshift=0.0, n_threads=1)
    with h5py.File(path + 'run_t/catalogue.hdf5') as f:
        hid, pid = f['Halo_ID'][()], f['PID'][()]
    assert hid.dtype == np.int64 and pid.dtype == np.int64
    assert set(hid.tolist()) <= set(big.tolist())   # every ID unchanged
    assert big[0] in hid

def test_offset_indices_do_not_overflow():
    """Offsetting 32-bit indices past the uint32 limit must stay exact."""
    data = np.array([0, 3_000_000_000, 4_294_967_295], dtype=np.uint32)
    out = _offset_indices(data, 5_000_000_000)
    assert out.dtype == np.int64
    np.testing.assert_array_equal(
        out, [5_000_000_000, 8_000_000_000, 9_294_967_295])

def _write_minibox_catalogue(path, n_memb, lidx, ridx, slidx, sridx, n_sub):
    """Write a tiny hand-made mini-box catalogue file as _save_catalogues does."""
    n = len(lidx)
    with h5py.File(path, 'w') as hdf:
        for key, val, dt in (
            ('Halo_ID', np.arange(n), np.int64), ('Norb', np.ones(n), np.uint32),
            ('LIDX', lidx, np.uint32), ('RIDX', ridx, np.uint32),
            ('NSUBS', np.asarray(sridx) - np.asarray(slidx), np.uint32),
            ('PID', np.full(n, -1), np.int64),
            ('SLIDX', slidx, np.uint32), ('SRIDX', sridx, np.uint32)):
            hdf.create_dataset(f'halo/{key}', data=val, dtype=dt)
        hdf.create_dataset('memb/PID', data=np.arange(n_memb), dtype=np.uint32)
        if n_sub:
            hdf.create_dataset('memb/Halo_ID', data=np.arange(n_sub), dtype=np.int64)

def test_merge_offsets_member_indices(tmp_path):
    """Merged LIDX/RIDX/SLIDX/SRIDX are int64, offset by previous files,
    and haloes without substructure get -1."""
    load_path = str(tmp_path) + '/'
    cat_dir = tmp_path / 'run_t' / 'mini_box_catalogues'
    cat_dir.mkdir(parents=True)
    # File 0: two haloes with 3 and 2 members; first halo has 1 subhalo.
    _write_minibox_catalogue(cat_dir / '0.hdf5', 5, [0, 3], [3, 5], [0, 1], [1, 1], 1)
    # File 1: one halo with 4 members and 2 subhaloes.
    _write_minibox_catalogue(cat_dir / '1.hdf5', 4, [0], [4], [0], [2], 2)
    merge_catalogues(load_path=load_path, run_name='t')
    with h5py.File(load_path + 'run_t/catalogue.hdf5') as f:
        for key in ('LIDX', 'RIDX', 'SLIDX', 'SRIDX'):
            assert f[key].dtype == np.int64
        np.testing.assert_array_equal(f['LIDX'][()], [0, 3, 5])
        np.testing.assert_array_equal(f['RIDX'][()], [3, 5, 9])
        np.testing.assert_array_equal(f['SLIDX'][()], [0, -1, 1])
        np.testing.assert_array_equal(f['SRIDX'][()], [1, -1, 3])


def test_positions_rounding_up_to_boxsize_are_wrapped(tmp_path):
    """A position just below boxsize must not be stored as exactly boxsize
    (float32 rounding), which would crash the periodic KD-tree."""
    path = str(tmp_path) + '/'
    box = 100.0
    pos = np.array([[box - 1e-9, 50.0, 50.0],    # rounds to 100.0 in float32
                    [50.0, box - 1e-9, 50.0],
                    [10.0, 20.0, 30.0]])          # ordinary particle
    minibox.process_simulation_data(
        save_path=path, particle_type='dm', boxsize=box, minisize=50.0,
        positions=pos, velocities=np.zeros((3, 3)), ids=np.arange(3),
        mass=(1e10, 'mass'), n_threads=1)

    stored = {}
    for fname in (tmp_path / 'mini_boxes_nside_2').glob('*.hdf5'):
        with h5py.File(fname) as f:
            for i, p in zip(f['dm/ID'][()], f['dm/pos'][()]):
                stored[int(i)] = (int(fname.stem), p)
    allpos = np.array([p for _, p in stored.values()])

    assert np.all(allpos >= 0) and np.all(allpos < box)
    np.testing.assert_allclose(stored[0][1], np.float32([0.0, 50.0, 50.0]), atol=1e-6)
    np.testing.assert_allclose(stored[1][1], np.float32([50.0, 0.0, 50.0]), atol=1e-6)
    np.testing.assert_array_equal(stored[2][1], np.float32([10.0, 20.0, 30.0]))
    cKDTree(allpos, boxsize=box)    # raised ValueError before the fix

class _PoolThatFailsAfterOneResult:
    """Stands in for multiprocessing.Pool: returns one result, then fails."""
    def __init__(self, n_threads):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def imap(self, func, args):
        args = list(args)
        yield func(args[0])
        raise RuntimeError("simulated worker crash")


def test_calibration_parallel_failure_does_not_duplicate(tmp_path, monkeypatch):
    """If the worker pool fails partway, the sequential fallback must give the
    same calibration data as a purely sequential run (no duplicated mini-boxes)."""
    path = str(tmp_path) + '/'
    rng = np.random.default_rng(3)
    seeds = np.array([[2.5, 2.5, 2.5], [7.5, 2.5, 2.5],
                      [2.5, 7.5, 2.5], [7.5, 7.5, 7.5]])     # 4 different mini-boxes
    pos = np.vstack([s + rng.normal(scale=0.3, size=(500, 3)) for s in seeds])
    minibox.process_simulation_data(
        save_path=path, particle_type='dm', boxsize=10.0, minisize=5.0,
        positions=np.mod(pos, 10.0), velocities=rng.normal(scale=50.0, size=(len(pos), 3)),
        ids=np.arange(len(pos)), mass=(1e10, 'mass'), n_threads=1)
    kwargs = dict(n_seeds=4,
                  seed_data=(seeds, np.zeros((4, 3)), np.full(4, 1e13), np.full(4, 0.5)),
                  r_max=1.0, boxsize=10.0, minisize=5.0, load_path=path,
                  particle_type='dm', mass_density=1e10, redshift=0.0)

    expected = calibration._select_candidate_seeds(**kwargs, n_threads=1)
    monkeypatch.setattr(calibration, 'Pool', _PoolThatFailsAfterOneResult)
    got = calibration._select_candidate_seeds(**kwargs, n_threads=4)
    np.testing.assert_array_equal(got, expected)

def test_stage_timer_accumulates():
    timer = StageTimer()
    for _ in range(2):
        with timer('work'):
            sum(range(10_000))
    assert set(timer.stages) == {'work'}
    assert timer.stages['work']['seconds'] > 0
    assert timer.stages['work']['peak_rss_mb'] > 0


def test_run_writes_timings(tmp_path):
    """A normal run writes run_<name>/timings.json with stages and counts."""
    path = str(tmp_path) + '/'
    rng = np.random.default_rng(6)
    pos = np.vstack([5.0 + rng.normal(scale = 0.15, size = (2000, 3)),
                     rng.uniform(0, 10, (2000, 3))])
    _write_seeds(path, 10.0, 10.0, np.array([[5.0, 5.0, 5.0]]), np.array([7]))
    minibox.process_simulation_data(
        save_path=path, particle_type='dm', boxsize=10.0, minisize=10.0,
        positions=np.mod(pos, 10.0), velocities=rng.normal(scale=50.0, size=(len(pos), 3)),
        ids=np.arange(len(pos)), mass=(1e10, 'mass'), n_threads=1)
    calibration.calibrate(save_path=path, omega_m=0.3)
    run_orbiting_mass_assignment(
        load_path=path, run_name='t', min_num_part=20, boxsize=10.0, minisize=10.0,
        padding=1.0, particle_type='dm', redshift=0.0, n_threads=1)
    with open(path + 'run_t/timings.json') as f:
        t = json.load(f)
    assert {'process_regions', 'merge'} <= set(t['run_stages'])
    assert {'load_seeds', 'load_particles', 'classification', 'percolation'} \
        <= set(t['region_stage_totals'])
    assert t['n_regions'] == 1 and t['n_seeds'] == 1 and t['n_particles_loaded'] == 4000    


class _TiedSeedsSource(SpatialDataSource):
    """Five seeds, three with identical M200b, served in a given order."""
    boxsize = 10.0

    def __init__(self, order):
        hid = np.array([40, 10, 30, 20, 50])
        m200b = np.array([1e13, 2e13, 1e13, 1e13, 5e12], dtype=np.float32)
        self.seeds = SeedSet(pos=np.full((5, 3), 5.0, np.float32), vel=np.zeros((5, 3), np.float32),
                             hid=hid[order], r200b=np.full(5, 0.5, np.float32),
                             m200b=m200b[order], rs=np.full(5, 0.1, np.float32),
                             in_core=np.ones(5, bool))

    def region_ids(self):
        return [0]

    def load_seeds(self, region_id):
        return self.seeds

    def load_particles(self, region_id):
        raise NotImplementedError


def _seed_order(order, tie_break):
    clf = MiniBoxClassifier(mini_box_id=0, min_num_part=1, boxsize=10.0, minisize=10.0,
                            load_path='', run_name='t', particle_type='dm',
                            seed_prop_names=('M200b', 'R200b', 'Rs'), redshift=0.0,
                            data_source=_TiedSeedsSource(order), seed_tie_break=tie_break)
    clf._load_seeds_and_filter()
    return clf.hid.tolist()


def test_halo_id_tie_break_is_independent_of_load_order():
    """With the default, seeds with equal M200b are processed by increasing
    Halo_ID, whatever order the data source returns them in."""
    for order in ([0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [2, 4, 0, 3, 1]):
        assert _seed_order(order, 'halo_id') == [10, 20, 30, 40, 50]


def test_load_order_tie_break_keeps_source_order():
    """'load_order' keeps equal masses in the order the source gave them."""
    assert _seed_order([0, 1, 2, 3, 4], 'load_order') == [10, 40, 30, 20, 50]
    assert _seed_order([4, 3, 2, 1, 0], 'load_order') == [10, 20, 30, 40, 50]


def test_invalid_tie_break_rejected():
    with pytest.raises(ValueError):
        _seed_order([0, 1, 2, 3, 4], 'alphabetical')


def _percolation_order(rows, tie_break):
    """Halo_IDs in the order _process_all_seeds hands them to percolation.

    rows are the per-seed classification results in seed processing order;
    the classification itself is replaced by returning those rows."""
    clf = MiniBoxClassifier.__new__(MiniBoxClassifier)
    clf.seed_tie_break = tie_break
    clf.disable_tqdm = True
    clf.n_seeds = len(rows)
    clf._init_catalogue_dataframe()
    clf._classify_single_seed = lambda i: rows[i]
    clf._process_all_seeds()
    return clf.haloes['Halo_ID'].tolist()


def test_equal_morb_keep_seed_order_whatever_else_is_loaded():
    """Two haloes with equal Morb (common: Morb = Norb * m_p) must enter
    percolation in seed order (here halo 7 before halo 3), however many other
    haloes the region holds. An unstable sort reorders such ties depending on
    the size of the table, so the result depended on the padding."""
    rng = np.random.default_rng(0)
    tied = [dict(Halo_ID=7, M200b=2e13, Morb=1.68e13),
            dict(Halo_ID=3, M200b=1e13, Morb=1.68e13)]
    for n_other in range(0, 400, 7):
        others = [dict(Halo_ID=100 + k, M200b=5e12, Morb=float(m))
                  for k, m in enumerate(rng.uniform(1e12, 1e14, n_other))]
        # Seed order: the tied pair somewhere among the other haloes
        at = n_other // 2
        rows = others[:at] + tied + others[at:]
        order = _percolation_order(rows, 'halo_id')
        assert order.index(7) < order.index(3), f"tie reordered with {n_other} others"
