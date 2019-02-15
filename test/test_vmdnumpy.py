"""
Tests the vmdnumpy module, if available
"""

import pytest
import numpy
from vmd import molecule
try:
    from vmd import vmdnumpy
except:
    pytestmark = pytest.mark.skipif(True,
                                    reason="VMD not compiled with numpy support")

def arr_equals(a1, a2):
    return numpy.all(numpy.equal(a1, a2))

def test_vmdnumpy(file_3frames):
    m = molecule.load("pdb", file_3frames)

    # Timestep and positions are equivalent
    assert arr_equals(vmdnumpy.timestep(m, 0), vmdnumpy.positions(m, 0))

    with pytest.raises(ValueError):
        vmdnumpy.timestep(molid=m+1)

    x = vmdnumpy.atomselect(selection="index 0")
    assert x.shape == (16,)
    assert x[0] == 1
    x[0] = 0
    assert arr_equals(x, numpy.zeros(x.shape))

    with pytest.raises(ValueError):
        vmdnumpy.atomselect("garbage atomselection", molid=m, frame=0)

    assert vmdnumpy.velocities(molid=m, frame=0) is None

    # Invalid frame
    with pytest.raises(ValueError):
        assert vmdnumpy.velocities(molid=m, frame=10) is None

    # Test on deleted molecule
    molecule.delete(m)
    with pytest.raises(ValueError):
        vmdnumpy.atomselect("protein")

