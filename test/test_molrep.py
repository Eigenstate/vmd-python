"""
Tests the molrep module
"""

import pytest
from vmd import molrep, molecule

def test_add_representation(file_3nob):
    m = molecule.load("mae", file_3nob)

    assert molrep.num(m) == 1
    with pytest.raises(ValueError):
        molrep.num(m+1)

    with pytest.raises(ValueError):
        molrep.addrep(molid=m, style="Invalid")

    with pytest.raises(ValueError):
        molrep.addrep(m, style="NewCartoon", color="Invalid")

    with pytest.raises(ValueError):
        molrep.addrep(m, "NewCartoon", "Type", selection="unparseable")

    with pytest.raises(ValueError):
        molrep.addrep(m, "NewCartoon", "Type", "protein", material="Invalid")

    r = molrep.addrep(style="NewCartoon", color="ResID", selection="protein",
                      material="AOShiny", molid=m)
    assert r == 1

    with pytest.raises(ValueError):
        molrep.delrep(m+1, 0)

    with pytest.raises(ValueError):
        molrep.delrep(m, rep=r+1)

    molrep.delrep(rep=r, molid=m)
    assert molrep.num(m) == 1

    molecule.delete(m)


def test_rep_attributes(file_3nob):
    m = molecule.load("mae", file_3nob)

    r = molrep.addrep(style="NewCartoon", color="ResID", selection="protein",
                      material="AOShiny", molid=m)

    # Query name
    assert molrep.get_repname(molid=m, rep=r) == "rep1"
    with pytest.raises(ValueError):
        molrep.get_repname(m+1, 0)
    with pytest.raises(ValueError):
        molrep.get_repname(m, r+1)

    # Query ID
    assert molrep.repindex(m, "rep1") == r
    assert molrep.repindex(m, "nonexistent") is None
    with pytest.raises(ValueError):
        molrep.repindex(m+1, "wrong")

    # Query style
    assert molrep.get_style(m, r) == "NewCartoon"
    with pytest.raises(ValueError):
        molrep.get_style(m+1, 0)
    with pytest.raises(ValueError):
        molrep.get_style(m, r+1)

    # Query selection
    assert molrep.get_selection(m, r) == "protein"
    with pytest.raises(ValueError):
        molrep.get_selection(m+1, 0)
    with pytest.raises(ValueError):
        molrep.get_selection(m, r+1)

    # Query color
    assert molrep.get_color(m, r) == "ResID"
    with pytest.raises(ValueError):
        molrep.get_color(m+1, 0)
    with pytest.raises(ValueError):
        molrep.get_color(m, r+1)

    # Query material
    assert molrep.get_material(m, r) == "AOShiny"
    with pytest.raises(ValueError):
        molrep.get_material(m+1, 0)
    with pytest.raises(ValueError):
        molrep.get_material(m, r+1)

    molecule.delete(m)


def test_modrep(file_3nob):
    m = molecule.load("mae", file_3nob)
    r = molrep.addrep(style="Licorice", color="ResID", selection="noh",
                      material="AOEdgy", molid=m)

    with pytest.raises(ValueError):
        molrep.modrep(m+1, 0)

    with pytest.raises(ValueError):
        molrep.modrep(m, r+1)

    assert molrep.get_style(m, r) == "Licorice"
    assert molrep.modrep(m, r, style="Lines")
    assert molrep.get_style(m, r) == "Lines"
    assert not molrep.modrep(m, r, style="Invalid")
    assert molrep.get_style(m, r) == "Lines"

    assert molrep.modrep(m, r, color="ColorID 0", selection="resname TIP3",
                         material="Transparent")
    assert molrep.get_selection(m, r) == "resname TIP3"
    assert molrep.get_material(m, r) == "Transparent"
    assert molrep.get_color(m, r) == "ColorID 0"

    molecule.delete(m)


def test_autoupdates(file_3nob):
    m = molecule.load("mae", file_3nob)
    r = molrep.addrep(color="User", selection="all", molid=m)

    # Color update
    assert not molrep.get_colorupdate(m, r)
    with pytest.raises(ValueError):
        molrep.get_colorupdate(m+1, 0)
    with pytest.raises(ValueError):
        molrep.get_colorupdate(m, r+1)

    molrep.set_colorupdate(m, rep=r, autoupdate=True)
    assert molrep.get_colorupdate(rep=r, molid=m)
    with pytest.raises(ValueError):
        molrep.set_colorupdate(m+1, 0, False)
    with pytest.raises(ValueError):
        molrep.set_colorupdate(m, r+1, False)

    # Selection update
    assert not molrep.get_autoupdate(molid=m, rep=r)
    with pytest.raises(ValueError):
        molrep.get_autoupdate(m+1, 0)
    with pytest.raises(ValueError):
        molrep.get_autoupdate(m, r+1)

    molrep.set_autoupdate(m, rep=r, autoupdate=True)
    assert molrep.get_autoupdate(rep=r, molid=m)
    with pytest.raises(ValueError):
        molrep.set_autoupdate(m+1, 0, False)
    with pytest.raises(ValueError):
        molrep.set_autoupdate(m, r+1, False)

    molecule.delete(m)


def test_colorscale(file_3nob):
    m = molecule.load("mae", file_3nob)
    r = molrep.addrep(color="User2", selection="lipid", molid=m)

    assert molrep.get_scaleminmax(m, r) == pytest.approx((0., 0.))
    with pytest.raises(ValueError):
        molrep.get_scaleminmax(m+1, 0)
    with pytest.raises(ValueError):
        molrep.get_scaleminmax(m, r+1)

    molrep.set_scaleminmax(molid=m, rep=r, scale_min=-10., scale_max=200.)
    assert molrep.get_scaleminmax(m, r) == pytest.approx((-10., 200.))
    with pytest.raises(ValueError):
        molrep.set_scaleminmax(m+1, 0, 0, 12)
    with pytest.raises(ValueError):
        molrep.set_scaleminmax(m, r+1, 12, 13)
    with pytest.raises(RuntimeError):
        molrep.set_scaleminmax(m, r, scale_min=100, scale_max=0)

    # Test reset
    molrep.reset_scaleminmax(molid=m, rep=r)
    assert molrep.get_scaleminmax(m, r) == pytest.approx((-10., 200.))
    with pytest.raises(ValueError):
        molrep.reset_scaleminmax(m+1, 0)
    with pytest.raises(ValueError):
        molrep.reset_scaleminmax(m, r+1)

    # Test changing with modrep
    assert molrep.modrep(m, r, scaleminmax=(2.0, 3.0))
    assert molrep.get_scaleminmax(m, r) == pytest.approx((2.0, 3.0))
    assert molrep.modrep(m, r, scaleminmax=[-10., -5.])
    assert molrep.get_scaleminmax(m, r) == pytest.approx((-10., -5.))

    molecule.delete(m)


def test_visible_and_smoothing(file_3nob):
    m = molecule.load("mae", file_3nob)
    r = molrep.addrep(material="AOChalky", selection="name C", molid=m)

    # Get visible
    assert molrep.get_visible(m, r)
    with pytest.raises(ValueError):
        molrep.get_visible(m+1, 0)
    with pytest.raises(ValueError):
        molrep.get_visible(m, r+1)

    # Set visible
    molrep.set_visible(m, r, True)
    assert molrep.get_visible(m, r)
    molrep.set_visible(m, r, visible=False)
    assert not molrep.get_visible(m, r)
    with pytest.raises(ValueError):
        molrep.set_visible(m+1, 0, False)
    with pytest.raises(ValueError):
        molrep.set_visible(m, r+1, True)

    # Get smoothing
    assert molrep.get_smoothing(molid=m, rep=r) == 0
    with pytest.raises(ValueError):
        molrep.get_smoothing(m+1, 0)
    with pytest.raises(ValueError):
        molrep.get_smoothing(m, r+1)

    # Set smoothing
    molrep.set_smoothing(m, r, 10)
    assert molrep.get_smoothing(rep=r, molid=m) == 10
    molrep.set_smoothing(rep=r, molid=m, smoothing=1)
    assert molrep.get_smoothing(rep=r, molid=m) == 1
    with pytest.raises(ValueError):
        molrep.set_smoothing(m, r, -1)
    with pytest.raises(ValueError):
        molrep.set_smoothing(m+1, 0, 12)
    with pytest.raises(ValueError):
        molrep.set_smoothing(m, r+1, 2)

    molecule.delete(m)

