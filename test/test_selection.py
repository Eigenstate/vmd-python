"""
Tests the new selection module
"""

import pytest
from vmd import selection

def test_builtins():

    assert selection.stringfunctions() == ["sequence", "rasmol"]
    assert selection.functions() ==['sqr', 'sqrt', 'abs', 'floor', 'ceil',
                                    'sin', 'cos', 'tan', 'atan', 'asin',
                                    'acos', 'sinh', 'cosh', 'tanh', 'exp',
                                    'log', 'log10']

    assert len(selection.booleans()) == 65
    assert len(selection.keywords()) == 74

def test_macros():

    x = selection.all_macros()
    assert "noh" in x
    assert selection.get_macro(name="noh") == "not hydrogen"

    with pytest.raises(ValueError):
        selection.get_macro("nonexistent")

    # Test adding a macro
    selection.add_macro(name="test", selection="resname TEST")
    assert "test" not in x
    assert "test" in selection.all_macros()
    assert selection.get_macro("test") == "resname TEST"

    # No unparseable selections
    with pytest.raises(ValueError):
        selection.add_macro("invalid", "not parseable atom selection")

    # No redefinition of built in keywords
    with pytest.raises(ValueError):
        selection.add_macro("sin", "resname BUILTIN")

    # No recursive selection
    with pytest.raises(ValueError):
        selection.add_macro("test2", "noh and test2")

    selection.del_macro("test")
    assert "test" not in selection.all_macros()

    with pytest.raises(ValueError):
        selection.del_macro("test")

