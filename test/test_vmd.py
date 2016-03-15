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

def test_list_egg():
    import os
    os.listdir("/home/travis/miniconda/envs/test-environment/lib/python2.7/site-packages/vmd-1.9.2a1-py2.7.egg")
    os.listdir("/home/travis/miniconda/envs/test-environment/lib/python2.7/site-packages/")
    raise ValueError

#==============================================================================

