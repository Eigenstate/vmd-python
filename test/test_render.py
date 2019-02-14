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
