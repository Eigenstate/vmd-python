"""
Tests the axes module
"""

import pytest
from vmd import axes

def test_locations():

    axes.set_location(axes.OFF)
    assert axes.get_location() == "Off"

    with pytest.raises(ValueError):
        axes.set_location("invalid")

    axes.set_location(axes.ORIGIN)
    assert axes.get_location() == axes.ORIGIN

    axes.set_location("LowerLeft")
    assert axes.get_location() == axes.LOWERLEFT



