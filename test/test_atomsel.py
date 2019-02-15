"""
Tests the atomsel type
"""

import pytest
import os
import math
import numpy as np
from pytest import approx
from vmd import atomsel, molecule, selection

def test_basic_getset(file_3frames):

    m = molecule.load("pdb", file_3frames)
    sel = atomsel(selection="protein", molid=m, frame=1)

    assert sel.frame == 1
    assert sel.molid == m
    assert len(sel.bonds) == 10
    assert str(sel) == "protein"
    assert repr(sel) == "atomsel('protein', molid=%d, frame=%d)" % (m, 1)

    # Default selections
    molecule.set_frame(m, 2)
    molecule.set_top(m)
    sel = atomsel()
    assert repr(sel) == "atomsel('all', molid=%d, frame=-1)" % m

    # Set something without a setter
    with pytest.raises(AttributeError):
        sel.molid = m+1

    # Invalid selection text
    with pytest.raises(ValueError):
        atomsel(selection="cannot be parsed")

    molecule.delete(m)

    # Selection on deleted molecule
    with pytest.raises(ValueError):
        assert sel.frame == 1
    with pytest.raises(ValueError):
        assert sel.molid == m
    with pytest.raises(ValueError):
        assert sel.bonds == []


def test_atomsel_tpattro(file_3nob):

    m = molecule.load("mae", file_3nob)
    sel = atomsel("resname ACE")

    # Invalid attribute
    with pytest.raises(AttributeError):
        sel.invalid_attribute
    with pytest.raises(AttributeError):
        sel.invalid_thingie = 3

    # Non keyword/singleword attribute
    with pytest.raises(AttributeError):
        sel.log
    with pytest.raises(AttributeError):
        sel.tan = 5

    # Invalid data type for attribute
    with pytest.raises(ValueError):
        sel.resid = ["hello"] * len(sel)

    # Deprecated getset
    assert set(sel.element) == set(["C", "O", "H"])
    with pytest.warns(DeprecationWarning):
        sel.set(attribute="element", value="P")
    with pytest.warns(DeprecationWarning):
        assert set(sel.get("element")) == set(["P"])

    # List attributes
    assert len(sel.list_attributes()) == 139
    assert "backbone" not in sel.list_attributes(changeable=True)
    assert "log" not in sel.list_attributes()

    molecule.delete(m)

    # Selection on deleted molecule
    with pytest.raises(ValueError):
        sel.element = "X"
    with pytest.raises(ValueError):
        sel.element


def test_atomsel_update(file_3frames):

    m = molecule.load("pdb", file_3frames)
    atomsel("resid 1", m, frame=2).user = 1.0
    sel = atomsel("user 1.0", molid=m, frame=-1)

    assert sel.frame == -1
    assert sel.index == list(range(10))

    altersel = atomsel("index 9", m, frame=2)
    altersel.user = 0.0
    assert atomsel("index 8 9", m, frame=2).user == approx([1.0, 0.0])
    assert atomsel("index 8 9", m, frame=0).user == approx([0.0, 0.0])

    # Selection should only change if the frame is set to update
    molecule.set_frame(m, 2)
    assert sel.index == list(range(10))
    sel.update()
    assert sel.index == list(range(9))

    # Now put it back to another frame
    sel.frame = 0
    assert sel.frame == 0
    assert sel.index == list(range(9))
    sel.update()
    assert sel.index == []

    molecule.delete(m)


def test_atomsel_write(tmpdir, file_3frames):

    m = molecule.load("pdb", file_3frames)
    tmpdir = str(tmpdir)

    # Frame out of bounds error
    sel = atomsel("residue 0", frame=5)
    with pytest.raises(ValueError):
        sel.write("mae", "wrong_frame.mae")

    # Write to mae and check it
    sel.frame = 1
    sel.write("mae", os.path.join(tmpdir, "written.mae"))

    w = molecule.load("mae", os.path.join(tmpdir, "written.mae"))
    assert atomsel("all", w).residue == [0] * 10
    molecule.delete(w)

    # Selection on deleted molecule
    molecule.delete(m)
    with pytest.raises(ValueError):
        sel.write("mae", "deleted.mae")


def test_atomsel_bonds(file_3frames):

    m = molecule.load("pdb", file_3frames)
    sel = atomsel(selection="protein", molid=m, frame=1)

    bonds = sel.bonds
    assert bonds[0] == [1,2]

    # Non sequence argument
    with pytest.raises(TypeError):
        sel.bonds = True

    # Wrong length list of bonds vs atoms
    with pytest.raises(ValueError):
        sel.bonds = ((2), (2), (2), (2))

    # Non sequence inside list of bonds
    with pytest.raises(TypeError):
        sel.bonds = [1] * 10

    # Non integers inside list of bonds
    with pytest.raises(TypeError):
        sel.bonds = [(2.8, 3.2)] * 10

    # Set using a list
    bonds[0] = [2,4]
    sel.bonds = bonds
    assert sel.bonds[0] == [2,4]

    # Set using a tuple
    bonds[1] = (2,3,8)
    sel.bonds = bonds
    assert sel.bonds[1] == [2,3,8]

    # Deleted molecule error
    molecule.delete(m)
    with pytest.raises(ValueError):
        sel.bonds =  [0,0] * 10


def test_atomsel_minmax_center(file_3frames):

    m = molecule.load("pdb", file_3frames)
    sel = atomsel(selection="protein", molid=m, frame=1)
    alasel = atomsel("resname ALA", molid=m)

    # Minmax without radii
    assert sel.minmax() == (approx((2.497, 0.724, -0.890)),
                            approx((6.009, 4.623, 2.131)))


    # Minmax with radii
    sel.radius  = 10.0
    assert sel.minmax(radii=True) == (approx((2.497-10., 0.724-10., -0.890-10.)),
                                      approx((6.009+10., 4.623+10., 2.131+10.)))

    assert sel.center(weight=None) == approx((4.040, 2.801, 0.492), rel=1e-3)
    assert sel.center(weight=[0.]*9+ [1.]) == atomsel("index 9", m).center()

    # Wrong number of weights
    with pytest.raises(TypeError):
        sel.center(weight=3.0)
    with pytest.raises(ValueError):
        sel.center([3.,2.])


    sel = atomsel("all")
    assert sel.centerperresidue() == [approx((4.040, 2.801, 0.492), rel=1e-3),
                                      approx((7.260, 3.987, 0.000), rel=1e-3)]
    assert sel.centerperresidue(weight=None)[0] == alasel.centerperresidue()[0]


    # Centerperresidue with weights
    with pytest.raises(TypeError):
        sel.centerperresidue(weight=[True]*len(sel))
    with pytest.raises(TypeError):
        sel.centerperresidue(3)
    with pytest.raises(ValueError):
        sel.centerperresidue([1.0,2.0])

    weights = [1.0]*10 + [0.0]*6
    assert sel.centerperresidue(weight=weights)[0] == approx(alasel.center(),
                                                             rel=1e-3)
    assert all([math.isnan(x) for x in sel.centerperresidue(weights)[1]])
    molecule.delete(m)

    # Operations on deleted molecule
    with pytest.raises(ValueError):
        sel.minmax(radii=False)
    with pytest.raises(ValueError):
        sel.centerrperres()
    with pytest.raises(ValueError):
        sel.center()


def test_atomsel_rmsf(file_3frames):

    m = molecule.load("pdb", file_3frames)
    sel = atomsel("resname NMA", m)
    asel = atomsel("all", m)

    # RMSF
    assert sel.rmsf() == [approx(0.0, abs=1e-5)] * len(sel)
    assert asel.rmsfperresidue() == approx([0.0, 0.0], abs=1e-5)
    assert np.mean(sel.rmsf()) == approx(asel.rmsfperresidue()[1], abs=1e-5)

    # Move one atom so values aren't all zero
    atomsel("index 11", m, frame=0).x = 1.0
    atomsel("index 11", m, frame=0).y = 1.0
    atomsel("index 11", m, frame=0).z = 1.0
    assert atomsel("all", m, frame=0).x[11] == 1.0
    assert atomsel("all", m, frame=0).y[11] == 1.0
    assert atomsel("all", m, frame=0).z[11] == 1.0

    # Ignore frame 0, or have only frame 0, should still be 0
    assert sel.rmsf(first=1) == approx([0.0] * len(sel), abs=1e-5)
    assert sel.rmsf(first=0, last=0)[0] == approx(0.0, abs=1e-5)
    assert sel.rmsf() == approx([0.0, 2.8730211, 0.0, 0.0, 0.0, 0.0], abs=1e-5)

    # Out of bounds
    with pytest.raises(RuntimeError):
        sel.rmsf(first=0, last=10000)
        asel.rmsfperresidue(first=2000)

    # RMSF per-residiue
    assert asel.rmsfperresidue() == approx([0.0, 0.4788369], abs=1e-5)
    atomsel("index 1", m, frame=2).x = 10.0
    assert asel.rmsfperresidue(first=1) == approx([0.30455, 0.0], abs=1e-5)
    assert asel.rmsfperresidue(last=1) == approx([0.0, 0.50788], abs=1e-5)

    molecule.delete(m)

    # Operations on deleted molecule
    with pytest.raises(ValueError):
        sel.rmsf()
    with pytest.raises(ValueError):
        sel.rmsfperresidue()


def test_atomsel_rmsd(file_3frames):

    m1 = molecule.load("pdb", file_3frames)
    m2 = molecule.load("pdb", file_3frames)
    atomsel("resname ALA", m1).moveby((4.0, 0.0, 0.0))
    atomsel("resname NMA", m1).moveby((0.0, 0.0, -4.0))

    sel1 = atomsel("resname ALA", m1)
    sel2 = atomsel("resname ALA", m2)
    assert sel1.rmsd(sel2) == approx(4.0)

    # Weights cannot be all zero
    with pytest.raises(ValueError):
        sel1.rmsd(sel2, [0.0]*len(sel1))

    # RMSDq - rotation invariant
    sel2.move(np.sin(60) * np.reshape(np.identity(4), (16,)))
    assert sel1.rmsdQCP(sel2) == approx(0.0, abs=1e-5)

    weights = [2.0]*(len(sel1)-1) + [1.0]
    assert sel1.rmsd(selection=sel2, weight=weights) == approx(4.0)
    assert sel1.rmsdQCP(sel2, weights) == approx(0.0, abs=1e-5)

    # RMSD per residue
    perres = atomsel("all", m1).rmsdperresidue(atomsel(molid=m2))
    assert len(perres) == 2
    assert perres[0] == approx(sel1.rmsd(sel2))
    assert perres[1] == approx(4.0)

    wpr = atomsel(molid=m1).rmsdperresidue(atomsel(molid=m2),
                                           weight=[1.0]*10 + [0.0]*6)
    assert wpr[0] == approx(sel1.rmsd(sel2))
    assert np.isnan(wpr[1])

    # Wrong number atoms or weights
    # Same code should catch this for rmsdQCP too
    with pytest.raises(ValueError):
        sel1.rmsd(atomsel(molid=m2))
    with pytest.raises(TypeError):
        sel1.rmsd(selection=sel2, weight=0.0)
    with pytest.raises(ValueError):
        sel1.rmsd(selection=sel2, weight="hello")
    with pytest.raises(ValueError):
        sel1.rmsd(selection=sel2, weight=[1.0, 2.0])
    with pytest.raises(ValueError):
        sel1.rmsd(selection=sel2, weight=[True, True])


    # Operations on deleted molecule
    # Same code should catch this for rmsdQCP etc
    molecule.delete(m1)
    with pytest.raises(ValueError):
        sel2.rmsd(sel1)

    molecule.delete(m2)
    with pytest.raises(ValueError):
        sel1.rmsd(sel2)


def test_atomsel_rgyr(file_3frames):

    m = molecule.load("pdb", file_3frames)
    atomsel(molid=m, frame=0).moveby((4.0, 0.0, 0.0))
    atomsel("all", m, 1).moveby((0.0, -4.0, 0.0))

    sel = atomsel("resname ALA", m)
    assert sel.rgyr() == approx(1.71880459)

    # Empty selection
    with pytest.raises(ValueError):
        atomsel("resname NOPE", m).rgyr()

    molecule.delete(m)
    with pytest.raises(ValueError):
        sel.rgyr()


def test_atomsel_fit_move(file_3nob):

    m1 = molecule.load("mae", file_3nob)
    m2 = molecule.load("mae", file_3nob)

    sel1 = atomsel("protein", m1)
    sel2 = atomsel("protein", m2)

    # Move one selection over
    assert sel1.x[0] == approx(sel2.x[0])
    sel1.moveby((1.0, 0.0, 0.0))
    assert sel1.x[0] == approx(sel2.x[0] + 1.0)
    sel1.moveby([1.0, 0.0, 0.0])
    assert sel1.x[0] == approx(sel2.x[0] + 2.0)
    assert sel1.y[0] == approx(sel2.y[0])
    assert sel1.z[0] == approx(sel2.z[0])

    # Fit, no weights
    fit1 = sel1.fit(sel2)
    assert fit1[0] == approx(1.0, abs=1e-5)

    # Fit, with weights
    fit0 = sel1.fit(selection=sel2, weight=[0.0] + [1.0]*(len(sel2)-1))
    assert fit0 == approx((1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                           1.0, 0.0, -2.0, 0.0, 0.0, 1.0))

    # Test selection invertible
    fit2 = np.reshape(sel2.fit(sel1), (4,4))
    assert np.linalg.inv(fit2) == approx(np.reshape(fit1, (4,4)))

    # Move
    sel1.move(fit1)
    assert sel1.x[0] == sel2.x[0]

    # Move with a numpy array - all zeros nans the array?
    sel1.move(np.zeros((16,)))
    assert sel1.y == approx(sel1.x, nan_ok=True)
    assert sel1.y != approx(sel1.x, nan_ok=False)

    # Operations on empty selection
    isel = atomsel("resname NOPE", m1)
    with pytest.raises(ValueError):
        isel.moveby((1., 2., 3.))

    with pytest.raises(ValueError):
        isel.move(fit1)

    with pytest.raises(ValueError):
        isel.fit(isel)

    molecule.delete(m1)
    with pytest.raises(ValueError):
        sel1.fit(sel2)

    molecule.delete(m2)
    with pytest.raises(ValueError):
        sel1.moveby((1,2,3))
    with pytest.raises(ValueError):
        sel2.move(fit1)


def test_atomsel_contacts(file_3nob):

    m1 = molecule.load("mae", file_3nob)
    m2 = molecule.load("mae", file_3nob)

    s1 = atomsel("resname PRO", m1)
    s2 = atomsel("resname PRO", m2)
    s1.moveby((5., 0., 0.))

    # Test empty selection
    print(len(atomsel("resname NOPE", m1)))
    with pytest.raises(ValueError):
        atomsel("resname NOPE", m1).contacts(s2, 100.0)
    with pytest.raises(ValueError):
        s1.contacts(atomsel("resname NOPE", m1), 100.0)

    # Empty contacts beyond cutoff
    x = s1.contacts(selection=s2, cutoff=1.0)
    assert x == [ [], [] ]

    # Indices should match for within cutoff
    x = s1.contacts(s2, 6.0)
    assert set(x[0]) == set(x[1])

    # Should go both ways, since indices are the same
    assert set(x[0]) == set(s2.contacts(s1, 6.0)[0])

    # Atoms directly bonded should not come up
    assert atomsel("index 601", m1).contacts(atomsel("index 602", m1), 10.0) \
        == [[], []]

    # Test invalid cutoff
    with pytest.raises(ValueError):
        s1.contacts(s2, cutoff=-10.0)

    # Test selection on deleted molecule
    molecule.delete(m1)
    with pytest.raises(ValueError):
        s1.contacts(s2, 1.0)

    molecule.delete(m2)
    with pytest.raises(ValueError):
        s2.contacts(s2, 1.0)


def test_atomsel_hbonds(file_3frames, file_3nob):

    m1 = molecule.load("pdb", file_3frames)
    m2 = molecule.load("mae", file_3nob)

    s1 = atomsel("resid 1", molid=m1, frame=0)
    s2 = atomsel("protein", molid=m2, frame=0)

    # Test donor and acceptor
    assert s1.hbonds(5.0, 180.0, atomsel("resid 2 and noh", m1)) \
        ==  [ [0, 2, 4, 4, 4, 0, 2, 4, 4, 4],
              [10, 10, 10, 10, 10, 12, 12, 12, 12, 12],
              [1, 3, 5, 6, 7, 1, 3, 5, 6, 7] ]

    # Test no acceptor
    assert len(s2.hbonds(cutoff=4.0, maxangle=40.0)[0]) == 421

    # Test invalid distances and angles
    with pytest.raises(ValueError):
        s1.hbonds(-1.0, 180.0)

    with pytest.raises(ValueError):
        s2.hbonds(cutoff=5.0, maxangle=-10.0)

    # Test invalid acceptor
    with pytest.raises(TypeError):
        s1.hbonds(1.0, 180.0, acceptor=[3,2,3])

    # Test empty coordinates
    with pytest.raises(ValueError):
        print(atomsel("resname NOPE").x)
        atomsel("resname NOPE", m1).hbonds(5.0, 180.0)

    # Test donor/acceptor in different molecules
    with pytest.raises(ValueError):
        s1.hbonds(5.0, 180.0, s2)

    # Test on deleted molecule
    molecule.delete(m1)
    with pytest.raises(ValueError):
        s1.hbonds(cutoff=5.0, maxangle=180.)

    molecule.delete(m2)
    with pytest.raises(ValueError):
        s2.hbonds(cutoff=5.0, maxangle=180.)

def test_atomsel_sasa(file_3frames):

    m1 = molecule.load("pdb", file_3frames)
    m2 = molecule.load("pdb", file_3frames)
    s1 = atomsel("resname ALA", m1)
    s2 = atomsel("all", m2)

    # Invalid radius
    with pytest.raises(ValueError):
        s1.sasa(srad=-2.0)

    assert s1.sasa(srad=0.0) == approx(93.86288452)

    # Invalid samples
    with pytest.raises(ValueError):
        s1.sasa(srad=1.0, samples=0)
    with pytest.raises(ValueError):
        s1.sasa(srad=1.0, samples=-10)

    # Points object invalid
    with pytest.raises(TypeError):
        s1.sasa(srad=1.0, points=[])

    with pytest.raises(TypeError):
        s1.sasa(srad=1.0, points=(3,3))

    # Test correct calculation
    assert s1.sasa(srad=1.0) == approx(176.46739)
    assert s2.sasa(1.0, restrict=s1) == approx(142.44455)
    assert s1.sasa(0.5) == approx(s1.sasa(0.5, restrict=s1))

    # Using points object
    _, points = s1.sasa(0.5, points=True, samples=1)
    assert len(points) == 3 # Samples is an upper bound
    assert points[0] == approx((3.99603, 3.19342, 3.62426), rel=1e-3)

    # Test on deleted molecule
    molecule.delete(m1)
    with pytest.raises(ValueError):
        s1.sasa(srad=5.0)

    with pytest.raises(ValueError):
        s2.sasa(1.0, restrict=s1)

    molecule.delete(m2)

@pytest.mark.skipif(not getattr(atomsel, "mdff", False),
                    reason="VMD not compiled with CUDA support")
def test_atomsel_mdff(file_3frames):

    m1 = molecule.load("pdb", file_3frames)
    s1 = atomsel("resname ALA", m1)

    # Invalid resolution and spacing
    with pytest.raises(ValueError):
        s1.mdffsim(spacing=-1.0)

    with pytest.raises(ValueError):
        s1.mdffsim(spacing=5.2, resolution=-1.5)

    r = s1.mdffsim(resolution=10.0, spacing=1.0)
    assert len(r) == 4
    assert r[2] == approx(s1.center())

    # Test deleted
    molecule.delete(m1)
    with pytest.raies(ValueError):
        s1.mdffsim()

# Test defining a macro with selection
def test_atomsel_selection(file_3nob):

    m = molecule.load("mae", file_3nob)

    with pytest.raises(ValueError):
        atomsel("resname NMA or isala")

    selection.add_macro(name="isala", selection="resname ALA")
    s1 = atomsel("resname NMA or isala")
    assert len(s1) == 46

    # TODO: Calling add_macro twice makes 2 entries???

    selection.add_macro(name="isala", selection="resname NMA")
    s1.update()
    assert len(s1) == 6

    selection.del_macro(name="isala")
    molecule.delete(m)
