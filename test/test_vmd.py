# Tests covalently bonded ligand
import inspect
import pytest
import os

dir = os.path.dirname(__file__)

#==============================================================================

def test_import():
    import vmd
    assert os.environ.get("VMDDIR")

#==============================================================================

def test_caps_import():
    from vmd import Molecule
    assert inspect.isclass(Molecule.Molecule)
    assert inspect.isclass(Molecule.MoleculeRep)
    assert callable(Molecule.Molecule.addRep)

#==============================================================================

def test_load_mae(file_rho):
    from vmd import molecule, atomsel
    assert callable(atomsel)

    molecule.load('mae', file_rho)
    chrg = set(atomsel().get('charge'))
    assert chrg == set([0.0, 1.0, -1.0])

    ins = set(atomsel().get('insertion'))
    assert set(_.strip() for _ in ins) == set(['', 'A'])

#==============================================================================

def test_evaltcl(file_rho):
    from vmd import evaltcl, atomsel, molecule

    molid = int(evaltcl("mol new"))
    assert molecule.get_top() == molid
    assert evaltcl("mol addfile %s type mae waitfor all" % file_rho)
    assert "molecule%d" % molid in evaltcl("mol list")

    with pytest.raises(ValueError):
        evaltcl("atomsel all")

    assert evaltcl("set all_atoms [atomselect %s \" all \" frame %s]"
                   % (molid, 0))  == "atomselect0"
    assert set(evaltcl("$all_atoms get chain").split()) == \
           set(atomsel("all").chain)
    assert set(evaltcl("$all_atoms get name").split()) == \
           set(atomsel("all").name)

    with pytest.raises(ValueError):
        evaltcl("$all_atoms get invalid")

#==============================================================================
