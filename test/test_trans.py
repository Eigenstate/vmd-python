"""
Tests the trans module
"""

import pytest
from vmd import molecule, trans

def test_show_fix(file_3nob):
    m = molecule.load("mae", file_3nob)

    trans.show(molid=m, shown=True)
    assert trans.is_shown(m)

    with pytest.raises(ValueError):
        trans.show(m+1, True)

    with pytest.raises(ValueError):
        trans.is_shown(m+1)

    trans.show(m, False)
    assert not trans.is_shown(molid=m)

    trans.fix(m, fixed=False)
    assert not trans.is_fixed(m)

    trans.fix(m, True)
    assert trans.is_fixed(m)

    with pytest.raises(ValueError):
        trans.fix(m+1, True)

    with pytest.raises(ValueError):
        trans.is_fixed(m+1)

    molecule.delete(m)


def test_scene_trans():

    with pytest.warns(DeprecationWarning):
        trans.rotate(axis='x', angle=0.)

    with pytest.warns(DeprecationWarning):
        trans.scale(1.0)

    with pytest.warns(DeprecationWarning):
        trans.translate(x=0., y=0., z=0.)

    with pytest.raises(ValueError):
        trans.rotate_scene(axis="n", angle=120.5)

    with pytest.raises(ValueError):
        trans.rotate_scene(axis="xaxis", angle=200.0)

    trans.rotate_scene('z', 10.0)
    trans.translate_scene(10., -10., 10.)
    trans.scale_scene(2.0)


def test_mol_center(file_3nob):

    m = molecule.load("mae", file_3nob)

    init_center = trans.get_center(m)
    trans.set_center(m, center=(1., 1., 1.))
    assert trans.get_center(m) == pytest.approx((1., 1., 1.))
    trans.set_center(m, center=[0., 0., 0.])
    assert trans.get_center(m) == pytest.approx((0., 0., 0.))

    with pytest.raises(ValueError):
        trans.get_center(molid=m+1)

    with pytest.raises(ValueError):
        trans.set_center(m+1, (3., 2., 1.))

    with pytest.raises(TypeError):
        trans.set_center(molid=m, center=(1.))

    trans.resetview(m)
    assert trans.get_center(m) == pytest.approx(init_center)

    with pytest.raises(ValueError):
        trans.resetview(molid=m+1)

    molecule.delete(m)


def test_mol_scale(file_3nob):

    m = molecule.load("mae", file_3nob)

    def_scale = trans.get_scale(m)
    trans.set_scale(molid=m, scale=1.0)
    assert trans.get_scale(molid=m) == pytest.approx(1.0)
    trans.set_scale(m, 2.0)
    assert trans.get_scale(m) == pytest.approx(2.0)

    with pytest.raises(ValueError):
        trans.set_scale(m+1, 10.0)

    with pytest.raises(ValueError):
        trans.get_scale(molid=m+1)

    trans.resetview(m)
    assert trans.get_scale(m) == pytest.approx(def_scale)
    molecule.delete(m)


def test_mol_trans(file_3nob):

    m = molecule.load("mae", file_3nob)

    t = trans.get_translation(m)
    assert t == pytest.approx((0., 0., 0.))

    with pytest.raises(ValueError):
        trans.get_translation(molid=m+1)

    with pytest.raises(ValueError):
        trans.set_translation(m+1, (1., 2., 3.))

    with pytest.warns(DeprecationWarning):
        trans.set_trans(trans=(10., 20., -10.), molid=m)
    assert trans.get_translation(molid=m) == pytest.approx((10., 20., -10.))

    trans.resetview(m)
    with pytest.warns(DeprecationWarning):
        assert trans.get_trans(m) == t

    molecule.delete(m)


def test_mol_rotate(file_3nob):

    m = molecule.load("mae", file_3nob)

    r = trans.get_rotation(molid=m)
    assert r == pytest.approx((1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0,
                               0.0, 0.0, 1.0, 0.0,
                               0.0, 0.0, 0.0, 1.0))

    newrot = [x+2. for x in r]
    trans.set_rotation(m, newrot)
    assert trans.get_rotation(molid=m) == pytest.approx(newrot)

    with pytest.raises(ValueError):
        trans.get_rotation(m+1)

    with pytest.raises(ValueError):
        trans.set_rotation(m+1, [0.]*16)

    with pytest.raises(TypeError):
        trans.set_rotation(molid=m, matrix=12)

    with pytest.raises(TypeError):
        trans.set_rotation(m, [True]*16)

    trans.resetview(m)
    assert trans.get_rotation(m) == r
    molecule.delete(m)
