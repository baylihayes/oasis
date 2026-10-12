import numpy
import pytest

from oasis.catalogue import MiniBoxClassifier


def _characteristic_density(r200, rs):
    """delta_c from MiniBoxClassifier._compute_deltac, without loading any data.

    The method only uses self.r200b and self.rs, so an instance is created
    without running __init__ and only those two attributes are set.
    """
    classifier = MiniBoxClassifier.__new__(MiniBoxClassifier)
    classifier.r200b = numpy.atleast_1d(numpy.asarray(r200, dtype=float))
    classifier.rs = numpy.atleast_1d(numpy.asarray(rs, dtype=float))
    classifier._compute_deltac()
    return classifier.deltac


def test_characteristic_density():
    r200 = 1.
    rs = 1.

    # Test zero division
    with pytest.raises(ZeroDivisionError):
        _characteristic_density(r200=r200, rs=0.)

    with pytest.raises(ZeroDivisionError):
        _characteristic_density(r200=0., rs=rs)

    with pytest.raises(ZeroDivisionError):
        _characteristic_density(r200=numpy.arange(3), rs=numpy.arange(3))

    delta = _characteristic_density(r200=r200, rs=rs)
    assert delta[0] == pytest.approx(200. / 3. / (numpy.log(2) - 0.5))


def test_rho_nfw_roots():
    r200 = 1.
    rs = 1.
    r12 = 2.5
    delta = _characteristic_density(r200=r200, rs=rs)[0]
    # Avoid x = 0 and x = r12, where one of the profiles diverges.
    x = numpy.linspace(0.01, r12 - 0.01, 1_000)
    froot = MiniBoxClassifier._rho_nfw_roots(x=x, delta1=delta, rs1=rs,
                                             delta2=delta, rs2=rs, r12=r12)

    # Two identical profiles: equal density exactly halfway between the centres,
    # the second (x measured from it) dominates close to x = 0, the first close
    # to x = r12.
    midpoint = MiniBoxClassifier._rho_nfw_roots(x=r12 / 2, delta1=delta, rs1=rs,
                                                delta2=delta, rs2=rs, r12=r12)
    assert midpoint == pytest.approx(0., abs=1e-12)
    assert numpy.all(froot[x < r12 / 2] < 0.)
    assert numpy.all(froot[x > r12 / 2] > 0.)


def _ball_radius(solver, *args):
    classifier = MiniBoxClassifier.__new__(MiniBoxClassifier)
    classifier.ball_radius_solver = solver
    return classifier._ball_radius(*args)


def test_ball_radius_brentq_where_fsolve_diverges():
    """A small, concentrated substructure 2 Mpc/h from a host: fsolve, started
    at half the distance, diverges to a huge negative radius (which made the
    6D ball cover the whole host); brentq finds the root between the seeds."""
    args = (1e4, 0.4, 2e4, 0.01, 2.0)       # delta1, rs1, delta2, rs2, r12
    assert _ball_radius('fsolve', *args) < -1e10
    r = _ball_radius('brentq', *args)
    assert 0. < r < args[-1]
    f = MiniBoxClassifier._rho_nfw_roots
    assert f(r * (1 - 1e-6), *args) < 0. < f(r * (1 + 1e-6), *args)


@pytest.mark.parametrize("args", [(1e4, 0.3, 1e4, 0.02, 1.5), (5e3, 0.5, 1e4, 0.03, 3.0),
                                  (2e4, 0.25, 1e4, 0.05, 1.0), (1e4, 1.0, 1e4, 1.0, 2.5)])
def test_ball_radius_solvers_agree_where_fsolve_converges(args):
    assert _ball_radius('brentq', *args) == pytest.approx(_ball_radius('fsolve', *args),
                                                           rel=1e-6)


def test_ball_radius_coincident_seeds_fall_back_to_fsolve():
    args = (1e4, 0.3, 1e4, 0.02, 0.0)
    numpy.testing.assert_equal(_ball_radius('brentq', *args), _ball_radius('fsolve', *args))


def test_unknown_ball_radius_solver_rejected():
    from test.test_edge_cases import _TiedSeedsSource
    with pytest.raises(ValueError, match="ball_radius_solver"):
        MiniBoxClassifier(mini_box_id=0, min_num_part=1, boxsize=10.0, minisize=10.0,
                          load_path='', run_name='t', particle_type='dm',
                          seed_prop_names=('M200b', 'R200b', 'Rs'), redshift=0.0,
                          data_source=_TiedSeedsSource([0, 1, 2, 3, 4]),
                          ball_radius_solver='newton')
