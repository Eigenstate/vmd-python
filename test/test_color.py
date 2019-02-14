"""
Test cases for color module
"""
import pytest
from vmd import color

def test_colormap_categories():
    assert len(color.categories()) == 16
    assert "Display" in color.categories()

    dispd = color.get_colormap("Display")
    assert len(dispd) == 5

    with pytest.raises(ValueError):
        color.get_colormap("doesn't exist")

    with pytest.raises(ValueError):
        color.set_colormap(name="doesn't exist", pairs={"nonexistent": "wrong"})

    with pytest.raises(ValueError):
        color.set_colormap(name="Display", pairs={"nonexistent": "wrong"})

    with pytest.raises(ValueError):
        color.set_colormap(name="Display", pairs={"Background": "wrong"})

    color.set_colormap(name="Display", pairs={"Background": "red",
                                              "FPS": "black"})
    dispd2 = color.get_colormap("Display")
    assert dispd2["Background"] == "red"
    assert dispd2["FPS"] == "black"
    assert dispd2["BackgroundTop"] == dispd["BackgroundTop"]

def test_colors():
    names = color.get_colors()
    vals = color.get_colorlist()

    assert set(names.values()) == set(vals)
    assert "blue" in names.keys()
    assert "wrong" not in names.keys()
    assert names["black"] == (0.0, 0.0, 0.0)
    assert names["white"] == (1.0, 1.0, 1.0)

    color.set_colors({"black": (1.0, 1.0, 1.0), "white": (0.0, 0.0, 0.0)})
    assert color.get_colors()["black"] == (1.0, 1.0, 1.0)
    assert color.get_colors()["white"] == (0.0, 0.0, 0.0)

    with pytest.raises(ValueError):
        color.set_colors({"wrong": (1.0, 1.0, 1.0)})

    with pytest.raises(ValueError):
        color.set_colors({"black": 3})

    with pytest.raises(ValueError):
        color.set_colors({"black": ("red", "green", "blue")})

    assert names["blue"] == (0.0, 0.0, 1.0)
    color.set_colorid(id=0, rgb=(1.0, 0.0, 0.0))
    assert color.get_colors()["blue"] == (1.0, 0.0, 0.0)
    color.set_colorid(0, (0.0, 1.0, 0.0))
    assert color.get_colors()["blue"] == (0.0, 1.0, 0.0)

    with pytest.raises(ValueError):
        color.set_colorid(id=3000, rgb=(0.0, 0.0, 0.0))


def test_scales():
    meths = color.scale_methods()
    assert len(meths) == 12
    assert color.scale_method() in meths
    assert "WBlk" in meths

    color.set_scale(method="WBlk")
    assert color.scale_method() == "WBlk"
    color.set_scale(midpoint=3.0)
    assert color.scale_midpoint() == 3.0
    color.set_scale(min=1.0, max=10.0)
    assert color.scale_min() == 1.0
    assert color.scale_max() == 10.0

    with pytest.raises(ValueError):
        color.set_scale(method="wrong")
