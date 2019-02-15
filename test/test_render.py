"""
Tests the render module
"""
import os
import pytest
from vmd import render, molecule

def teardown_module(module):
    for _ in molecule.listall():
        molecule.delete(_)

def test_render(tmpdir):

    tmpdir = str(tmpdir)
    x = render.listall()
    assert len(x) == 16

    with pytest.raises(RuntimeError):
        render.render(method="bad", filename="test")

    render.render(x[0], os.path.join(tmpdir, "test.out"))

    assert os.path.isfile(os.path.join(tmpdir, "test.out"))

def test_snapshot(tmpdir):

    from vmd import molecule, molrep, display
    m =  molecule.load("mae", "3nob.mae")
    molrep.addrep(m, "NewCartoon")
    display.set(size=(512, 512))

    tmpdir = str(tmpdir)
    render.render("snapshot", os.path.join(tmpdir, "test.tga"))

    # If it's an empty header, the file size will be way too small here
    assert os.path.getsize(os.path.join(tmpdir, "test.tga")) == 786450

