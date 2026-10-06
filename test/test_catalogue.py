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
