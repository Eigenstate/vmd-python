"""
Tests the material module
"""

import pytest
from vmd import material

def test_rename():
    name = material.listall()[0]

    material.rename(name, newname="renamed")
    assert name not in material.listall()

    with pytest.raises(ValueError):
        material.rename(name=name, newname="incorrect")

    with pytest.raises(ValueError):
        material.rename("renamed", material.listall()[1])


def test_change():
    name = material.listall()[0]
    props = material.settings(name=name)

    material.default(name)
    assert material.settings(name) == props

    with pytest.raises(ValueError):
        material.default("nonexistent")

    with pytest.raises(ValueError):
        material.change(name="nonexistent")

    material.change(name, ambient=3.0, diffuse=2.0, transmode=True)
    newprops = material.settings(name=name)
    for prop in ["specular", "shininess", "mirror", "opacity",
                 "outline", "outlinewidth"]:
        assert props[prop] == newprops[prop]

    assert newprops["ambient"] == pytest.approx(3.0)
    assert newprops["diffuse"] == pytest.approx(2.0)
    assert newprops["transmode"] == True

    material.default(name=name)
    assert material.settings(name) == props


def test_add():

    # Add with no information
    num = len(material.listall())
    material.add()
    newname = "Material%d" % num
    assert newname in material.listall()

    with pytest.raises(ValueError):
        material.add(name=newname)

    # Add a copy of another material
    material.add(copy=newname, name="copyMaterial")
    assert "copyMaterial" in material.listall()
    assert material.settings(newname) == material.settings(name="copyMaterial")

    with pytest.raises(ValueError):
        material.add(copy="invalid")

    # Delete the copy
    material.delete("copyMaterial")
    assert "copyMaterial" not in material.listall()

    with pytest.raises(ValueError):
        material.delete("copyMaterial")
