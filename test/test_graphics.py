"""
Tests the graphics module.
Also tests the fake way of getting molecules to draw graphics on
"""

import pytest
from vmd import graphics, molecule

def test_shapes():

    mid2 = molecule.load("graphics", "test")

    assert graphics.listall(mid2) == []

    with pytest.raises(ValueError):
        graphics.cone(molid=mid2+1, v1=(0,0,0), v2=(3,5,9))

    assert graphics.cone(molid=mid2, v1=(0,0,0), v2=(3,5,9)) == 0
    assert graphics.triangle(mid2, (1,0,0), (0,1,0), (0,0,1)) == 1
    assert graphics.listall(mid2) == [0, 1]

    graphics.trinorm(mid2, (1,0,0), (0,1,0), (0,0,1),
                     n1=(0,1,0), n2=(1,0,0), n3=(0,0,1))

    graphics.cylinder(mid2, v1=(10,10,0), v2=(0,10,10), radius=3.,
                      resolution=4, filled=True)

    graphics.point(mid2, (20, 3, 1))

    graphics.line(mid2, (-1,1,-3), (3,4,5))
    with pytest.raises(ValueError):
        graphics.line(mid2, (-1,1,-3), (3,4,5), style="invalid")

    assert len(graphics.listall(mid2)) == 6
    graphics.delete(mid2, "all")
    assert len(graphics.listall(mid2)) == 0
    with pytest.raises(ValueError):
        graphics.delete(mid2, "invalid")
    with pytest.raises(TypeError):
        graphics.delete(mid2, 39.0)

    molecule.delete(mid2)


def test_replace():

    mid2 = molecule.load("graphics", "heya")
    rc = graphics.point(mid2, (0,0,0))
    assert graphics.info(mid2, rc) == "point {0.000000 0.000000 0.000000}"
    rc2 = graphics.point(mid2, (1,1,1))
    assert graphics.listall(mid2) == [rc, rc2]

    graphics.replace(molid=mid2, graphic=rc)
    assert graphics.listall(mid2) == [rc2]

    rc3 = graphics.point(mid2, (2,2,2))
    assert graphics.listall(mid2) == [rc3, rc2]
    assert graphics.info(mid2, rc2) == "point {1.000000 1.000000 1.000000}"

    molecule.delete(mid2)


def test_materials():
    mid = molecule.new(name="Test")
    graphics.materials(molid=mid, on=True)
    graphics.material(molid=mid, name="AOShiny")

    rc = graphics.sphere(molid=mid, center=(1,1,1), radius=5.1, resolution=8)
    assert graphics.info(mid, rc) == "sphere {1.000000 1.000000 1.000000} " +\
        "radius 5.100000 resolution 8"

    with pytest.raises(ValueError):
        graphics.sphere(molid=mid, center=(1,1))

    rc = graphics.text(mid, (0,0,0), "hello world", size=4.0, width=2.0)
    with pytest.raises(ValueError):
        graphics.text(mid, 0, "oops")

    assert graphics.info(mid, rc) == "text {0.000000 0.000000 0.000000} " + \
        "{hello world} size 4.000000 thickness 2.000000"

    with pytest.raises(ValueError):
        graphics.info(mid+1, 3)
    with pytest.raises(IndexError):
        graphics.info(mid, 3000)

    graphics.delete(mid, rc)
    with pytest.raises(IndexError):
        graphics.info(mid, rc)

    molecule.delete(mid)


def test_colors():
    mid = molecule.new(name="colortest")

    graphics.color(mid, 1)
    graphics.color(mid, "blue2")

    with pytest.raises(ValueError):
        graphics.color(mid, -1)

    with pytest.raises(ValueError):
        graphics.color(mid, "not a color")

    with pytest.raises(TypeError):
        graphics.color(mid, 32.0)

    molecule.delete(mid)

