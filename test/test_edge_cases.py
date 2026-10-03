"""Regression tests for edge cases fixed during the performance refactor."""
import numpy as np
import h5py

from oasis import calibration, minibox
from oasis.catalogue import MiniBoxClassifier, run_orbiting_mass_assignment



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

