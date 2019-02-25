"""
Tests the label module
"""

import pytest
from vmd import label, molecule

def test_label_atom(file_3nob):
    m1 = molecule.load("mae", file_3nob)

    with pytest.raises(ValueError):
        label.add("invalid", (m1,), (0,))

    with pytest.raises(ValueError):
        label.add(label.ATOM, molids=(), atomids=())

    with pytest.raises(ValueError):
        label.add("Atoms", molids=(m1+1,), atomids=(0,))

    with pytest.raises(ValueError):
        label.add("Atoms", molids=(m1,), atomids=(-13,))

    md = label.add(category=label.ATOM, molids=(m1,), atomids=(0,))

    with pytest.raises(ValueError):
        label.set_visible("invalid", {}, False)

    with pytest.raises(ValueError):
        label.set_visible("Atoms", {}, True)

    label.set_visible("Atoms", md, True)
    assert label.listall(label.ATOM) == [{"molid": (m1,),
                                          "atomid": (0,),
                                          "value": 0.0,
                                          "on": True}]

    label.set_visible("Atoms", md, False)
    assert label.listall(label.ATOM)[0]["on"] == False

    with pytest.raises(ValueError):
        label.get_values("none", md)

    with pytest.raises(ValueError):
        label.get_values("Atoms", {})

    assert label.get_values(category="Atoms", label=md) == None

    molecule.delete(m1)


def test_label_bond(file_3nob):
    m1 = molecule.load("mae", file_3nob)
    m2 = molecule.load("mae", file_3nob)

    with pytest.raises(ValueError):
        label.add("Bonds", molids=(m1, m2), atomids=(0,))

    with pytest.raises(ValueError):
        label.add("Bonds", molids=[m1,], atomids=(0,))

    md = label.add("Bonds", molids=(m1, m2), atomids=[0, 100])
    assert label.listall(label.BOND) == [{"molid": (m1, m2),
                                          "atomid": (0, 100),
                                          "value": pytest.approx(19.011491775),
                                          "on": True}]

    assert label.get_values("Bonds", label=md) == [pytest.approx(19.011491775)]

    molecule.delete(m1)
    molecule.delete(m2)


def test_label_text():

    with pytest.raises(ValueError):
        label.text_size(-3.0)

    with pytest.raises(TypeError):
        label.text_size("hey")

    assert label.text_size(5.0) == pytest.approx(5.0)
    assert label.text_size() == pytest.approx(5.0)

    with pytest.raises(TypeError):
        label.text_thickness(3)

    with pytest.raises(ValueError):
        label.text_thickness(0.0)

    assert label.text_thickness() == pytest.approx(1.0)
    label.text_thickness(thickness=8.2)
    assert label.text_thickness() == pytest.approx(8.2)

