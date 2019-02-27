import pytest
import os

def pytest_runtest_teardown():
    from vmd import molecule
    for _ in molecule.listall():
        molecule.delete(_)

def _get_file(request, filename):
    thedir = os.path.split(str(request.fspath))[0]
    return os.path.join(thedir, filename)

@pytest.fixture(scope="module")
def file_3nob(request):
    """ Gets the path to 3nob.mae """
    return _get_file(request, "3nob.mae")

@pytest.fixture(scope="module")
def file_3frames(request):
    """ Gets the path to ala_nma_3frames.pdb"""
    return _get_file(request, "ala_nma_3frames.pdb")

@pytest.fixture(scope="module")
def file_psf(request):
    """ Gets the path to ala_nmapsf"""
    return _get_file(request, "ala_nma.psf")

@pytest.fixture(scope="module")
def file_dx(request):
    """ Gets the path to 0.dx """
    return _get_file(request, "0.dx")

@pytest.fixture(scope="module")
def file_rho(request):
    return _get_file(request, "rho_test.mae")
