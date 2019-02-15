"""
Tests the measure module
"""

import pytest
from vmd import measure, molecule

def test_measure(file_3nob):
    m1 = molecule.load("mae", file_3nob)
    m2 = molecule.load("mae", file_3nob)

    with pytest.warns(SyntaxWarning):
        assert measure.bond(0, 0, m1, m2, frame=0, first=0) == [pytest.approx(0.)]

    assert measure.bond(atom1=0, atom2=10, molid=m1)[0] == pytest.approx(4.97422456)
    assert measure.bond(atom1=0, atom2=10)[0] == pytest.approx(4.97422456)

    # One atom listed multiple times
    with pytest.raises(RuntimeError):
        measure.angle(0, 0, 0)

    with pytest.warns(SyntaxWarning):
        assert measure.angle(0, 1, 2, frame=0,
                             last=1) == [pytest.approx(119.7387237548)]

    assert measure.angle(atom1=0, atom2=1, atom3=0, molid=m1, molid2=m2,
                         molid3=m1)[0] == pytest.approx(0.0)

    with pytest.raises(RuntimeError):
        measure.dihedral(0,0,0,0)

    molecule.delete(m1)
    molecule.delete(m2)


def test_multiple_frames(file_3frames):
    m = molecule.load("pdb", file_3frames)

    assert measure.bond(0, 1) == [pytest.approx(1.0093883275)]

    with pytest.warns(SyntaxWarning):
        assert measure.dihedral(0, 1, 2, 3, molid=m, frame=0,
                                first=1, last=2)[0] == pytest.approx(-70.578407)

    x = measure.dihedral(0, 1, 2, 3, first=0, last=-1)
    assert len(x) == 3
    assert x == [pytest.approx(-70.578407)] * 3

    molecule.delete(m)

