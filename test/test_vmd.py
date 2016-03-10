# Tests covalently bonded ligand
import pytest
import subprocess, os

dir = os.path.dirname(__file__)

#==============================================================================

def test_import():
    import vmd

#==============================================================================

def test_load_mae():
    import vmd, molecule
    from atomsel import atomsel

    molecule.load('mae', os.path.join(dir, "rho_test.mae"))
    chrg = set(atomsel().get('charge'))
    assert chrg == set([0.0, 1.0, -1.0])

#==============================================================================

