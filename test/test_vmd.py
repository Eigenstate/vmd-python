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

def test_load_mae():
    from vmd import molecule, atomsel
    assert callable(atomsel)

    molecule.load('mae', os.path.join(dir, "rho_test.mae"))
    chrg = set(atomsel().get('charge'))
    assert chrg == set([0.0, 1.0, -1.0])

    ins = set(atomsel().get('insertion'))
    assert ins == set([' ', 'A'])

#==============================================================================
