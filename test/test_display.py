"""
Test cases for display module
"""
import pytest
from vmd import display

def test_update():

    display.update_off()
    display.update()
    display.update_on()

def test_invalidkeywords():

    with pytest.raises(TypeError):
        display.set(nonsense=5.0)

    with pytest.raises(ValueError):
        display.get("nonsense")

def test_rendergetset():
    """
    tests:
    antialias, ambientocclusion, aoambient, aodirect, shadows,
    """
    display.set(antialias=False, ambientocclusion=True)

    with pytest.raises(TypeError):
        display.set(antialias=5)

    assert not display.get("antialias")
    assert display.get("ambientocclusion")

    display.set(ambientocclusion=False, aoambient=1.0, aodirect=3.0)
    assert not display.get("ambientocclusion")
    assert display.get("aoambient") == 1.0
    assert display.get("aodirect") == 3.0

    display.set(shadows=True)
    assert display.get("shadows")

def test_size():
    """
    size
    """
    currsize = display.get("size")
    assert len(currsize) == 2
    display.set(size=currsize[::1])
    assert display.get("size") == currsize[::1]

    display.set(size=(100,200))
    assert display.get("size") == [100, 200]

def test_dof():
    """
    dof, dof_fnumber, dof_focaldist
    """
    display.set(dof=True, dof_fnumber=3.2, dof_focaldist=12.0)
    assert display.get("dof")
    assert display.get("dof_fnumber") == pytest.approx(3.2)
    assert display.get("dof_focaldist") == 12.0

@pytest.mark.skip(reason="Unimplemented OpenGL PBuffer")
def test_3dgetset():
    """
    tests: eyesep, focallength, height, distance, nearclip, farclip,
    depthcueue, culling, stereo, projection
    """

    modes = display.stereomodes()
    assert len(modes) == 11

    display.set(eyesep=2.0, focallength=10.0, height=200,
                distance=20, stereo=modes[0])

    assert display.get("eyesep") == 2.0
    assert display.get("focallength") == 10.0
    assert display.get("height") == 200
    assert display.get("distance") == 20
    assert display.get("stereo") == modes[0]

    # Default values from DisplayDevice.C
    assert display.get("nearclip") == 0.5
    assert display.get("farclip") == 10.0
    display.set(nearclip=10.0, farclip=30.0)
    assert display.get("nearclip") == 10.0
    assert display.get("farclip") == 30.0

    with pytest.raises(ValueError):
        display.set(nearclip=10.0, farclip=5.0)

    display.set(culling=False)
    assert not display.get("culling")
    with pytest.raises(TypeError):
        display.set(culling=5.0)

    assert "QuadBuffered" in modes
    display.set(stereo="QuadBuffered")
    assert display.get("stereo") == "QuadBuffered"

    display.set(projection=display.PROJ_PERSP)
    assert display.get("projection") == display.PROJ_PERSP

    with pytest.raises(ValueError):
        display.set(projection="invalid")

    display.set(depthcue=True)
    assert display.get("depthcue")
    with pytest.raises(TypeError):
        display.set(depthcueue="hello")


