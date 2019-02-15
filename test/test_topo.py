"""
Test cases for topo module
"""
import pytest
from vmd import molecule, topology

def test_bonds(file_3nob):
    with pytest.raises(ValueError):
        topology.bonds(molid=0)

    m = molecule.load("mae", file_3nob)
    molecule.set_top(m)

    with pytest.raises(ValueError):
        topology.bonds(molid=m, type=3, order=True)

    with pytest.warns(DeprecationWarning):
        x = topology.bonds(molid=m, type=3)
        assert len(x) == 2493
        assert x[0] == (0, 1, None, 1.0)

    with pytest.warns(DeprecationWarning):
        x = topology.bonds(molid=m, type=2)
        assert len(x) == 2493
        assert x[0] == (0, 1, 1.0)

    with pytest.warns(DeprecationWarning):
        x = topology.bonds(molid=m, type=1)
        assert len(x) == 2493
        assert x[0] == (0, 1, None)

    x = topology.bonds(molid=m, type=True)
    assert len(x) == 2493
    assert x[0] == (0, 1, None)

    x = topology.bonds(molid=m, type=False, order=True)
    assert len(x) == 2493
    assert x[0] == (0, 1, 1.0)

    x = topology.bonds(molid=m, type=True, order=True)
    assert len(x) == 2493
    assert x[0] == (0, 1, None, 1.0)

    x = topology.bonds(molid=m)
    assert len(x) == 2493
    assert len(x[0]) == 2
    assert (0, 1) in x

    # Test adding a bond
    assert (0, 2) not in x
    topology.addbond(i=0, j=2, molid=-1)
    assert (0, 2, None, 1.0) in topology.bonds(molid=m, order=True, type=True)

    assert (2, 3) not in x
    assert topology.bondtypes(m) == []
    topology.addbond(i=2, j=3, molid=m, order=2.0, type="test")
    assert (2, 3, "test", 2.0) in topology.bonds(molid=m, order=True, type=True)
    assert topology.bondtypes(m) == ["test"]

    # Test deleting a bond
    assert topology.delbond(i=2, j=0, molid=m)
    assert (0, 2) not in topology.bonds(molid=m)

    assert not topology.delbond(0, 0, molid=m)
    assert topology.delbond(i=2, j=3)
    assert topology.bondtypes(m) == ["test"] # bond type not deleted

    with pytest.raises(ValueError):
        topology.delbond(i=-1, j=5, molid=m)

    with pytest.raises(ValueError):
        topology.delbond(i=0, j=1, molid=-5)

    with pytest.raises(ValueError):
        topology.delallbonds(molid=2389182)

    assert topology.delallbonds(m) == 2493
    assert len(topology.bonds(m)) == 0
    assert topology.delallbonds(m) == 0

    molecule.delete(m)

def test_angles(file_3nob):
    with pytest.raises(ValueError):
        topology.angles(molid=0)

    m = molecule.load("mae", file_3nob)
    molecule.set_top(m)

    # No angles defined in a mae file
    x = topology.angles(molid=m, type=True)
    print(len(x))
    assert len(x) == 0
    assert topology.angletypes() == []

    # pass invalid molid
    with pytest.raises(ValueError):
        topology.addangle(0, 0, 0, molid=-1999)

    assert topology.addangle(i=0, j=1, k=2, molid=m) == 0
    assert topology.angles(m, True) == [(0, 1, 2, None)]

    x = topology.addangle(0, 0, 0, molid=m, type="angle2")
    assert topology.angles(m, True)[x] == (0, 0, 0, "angle2")

    assert topology.delangle(i=0, j=0, k=0, molid=m)
    assert topology.angles(m, True) == [(0, 1, 2, None)]

    assert not topology.delangle(1, 2, 3, molid=m)

    assert topology.delallangles(m) == 1
    assert topology.angletypes() == ["angle2"]

    molecule.delete(m)

def test_dihedrals(file_3nob):
    with pytest.raises(ValueError):
        topology.dihedrals(molid=0)

    m = molecule.load("mae", file_3nob)
    molecule.set_top(m)

    # No dihedrals defined in a mae file
    x = topology.dihedrals(molid=m, type=True)
    print(len(x))
    assert len(x) == 0
    assert topology.angletypes() == []

    # pass invalid molid
    with pytest.raises(ValueError):
        topology.adddihedral(0, 0, 0, 0, molid=1999)

    assert topology.adddihedral(i=0, j=1, k=2, l=3, molid=m) == 0
    assert topology.dihedrals(m, True) == [(0, 1, 2, 3, None)]

    x = topology.adddihedral(0, 0, 0, 0, molid=m, type="angle4")
    assert topology.dihedrals(m, True)[x] == (0, 0, 0, 0, "angle4")
    assert topology.dihedraltypes() == ["angle4"]

    assert topology.deldihedral(i=0, j=0, k=0, l=0, molid=m)
    assert topology.dihedrals(m, True) == [(0, 1, 2, 3, None)]

    assert not topology.deldihedral(1, 2, 3, 4, molid=m)

    assert topology.delalldihedrals(molid=m) == 1
    assert topology.dihedraltypes() == ["angle4"]

    molecule.delete(m)


def test_impropers(file_3nob):
    with pytest.raises(ValueError):
        topology.impropers(molid=0)

    m = molecule.load("mae", file_3nob)
    molecule.set_top(m)

    # No impropers defined in a mae file
    x = topology.impropers(molid=m, type=True)
    print(len(x))
    assert len(x) == 0
    assert topology.angletypes() == []

    # pass invalid molid
    with pytest.raises(ValueError):
        topology.addimproper(0, 0, 0, 0, molid=-1999)

    assert topology.addimproper(i=0, j=1, k=2, l=3, molid=m) == 0
    assert topology.impropers(m, True) == [(0, 1, 2, 3, None)]

    x = topology.addimproper(0, 0, 0, 0, molid=m, type="angle5")
    assert topology.impropers(m, True)[x] == (0, 0, 0, 0, "angle5")
    assert topology.impropertypes() == ["angle5"]

    assert topology.delimproper(i=0, j=0, k=0, l=0, molid=m)
    assert topology.impropers(m, True) == [(0, 1, 2, 3, None)]

    assert not topology.delimproper(1, 2, 3, 4, molid=m)

    assert topology.delallimpropers(molid=m) == 1
    assert topology.impropertypes() == ["angle5"]

    molecule.delete(m)

