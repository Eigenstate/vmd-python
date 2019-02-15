"""
Tests the mouse module
"""

import pytest
from vmd import mouse

def test_mouse():

    mouse.mode(mode=mouse.ROTATE)
    mouse.mode(mode=mouse.QUERY)
    mouse.mode(mouse.LIGHT, 0)

    with pytest.raises(ValueError):
        mouse.mode(mode=mouse.ROTATE, lightnum=mouse.LABELATOM)

    with pytest.raises(ValueError):
        mouse.mode(mode=mouse.LIGHT)

    with pytest.raises(ValueError):
        mouse.mode(mouse.LIGHT, lightnum=3599)

