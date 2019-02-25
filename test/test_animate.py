"""
Tests the animate module
"""

import pytest
from vmd import molecule, animate, display

def test_animate(file_3frames):
    m = molecule.load("pdb", file_3frames)
    assert molecule.numframes(m) == 3
    molecule.set_frame(m, 0)

    with pytest.raises(ValueError):
        animate.activate(-3000, True)

    animate.activate(molid=m, active=False)
    assert not animate.is_active(m)

    animate.activate(m, True)
    assert animate.is_active(m)

    animate.goto(0)
    display.update()
    assert molecule.get_frame(m) == 0

    animate.goto(1)
    display.update()
    assert molecule.get_frame(m) == 1

    animate.forward()
    animate.reverse()
    animate.next()
    animate.prev()
    animate.pause()

    molecule.delete(m)


def test_methods():

    animate.once()
    assert animate.style() == "Once"

    animate.rock()
    assert animate.style() == "Rock"

    animate.loop()
    assert animate.style() == "Loop"


def test_settables():

    animate.speed(value=0.5)
    assert animate.speed() == 0.5

    with pytest.raises(ValueError):
        animate.speed(2)

    assert animate.skip(value=3) == 3

    with pytest.raises(ValueError):
        animate.skip(value=-1)

