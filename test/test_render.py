"""
Tests the render module
"""
import os
import pytest
from vmd import render

def test_render(tmpdir):

    tmpdir = str(tmpdir)
    x = render.listall()
    assert len(x) == 16

    with pytest.raises(RuntimeError):
        render.render(method="bad", filename="test")

    render.render(x[0], os.path.join(tmpdir, "test.out"))

    assert os.path.isfile(os.path.join(tmpdir, "test.out"))

@pytest.mark.skip(reason="PBuffer support not implemented")
def test_snapshot(tmpdir, file_3nob):

    from vmd import molecule, molrep, display
    m =  molecule.load("mae", file_3nob)
    molrep.addrep(m, "NewCartoon")
    display.set(size=(512, 512))

    tmpdir = str(tmpdir)
    render.render("snapshot", os.path.join(tmpdir, "test.tga"))

    # If it's an empty header, the file size will be way too small here
    assert os.path.getsize(os.path.join(tmpdir, "test.tga")) == 786450

