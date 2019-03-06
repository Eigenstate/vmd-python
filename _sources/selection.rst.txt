.. _selection:
.. currentmodule:: vmd.selection
.. highlight:: python

Selection
=========

Methods for creating, modifying, or deleting macros for atom selections.
Often you'll want to create a short word that expands to mean more in an
atom selection context: for example, the built-in macro `noh` expands to
`not hydrogen`.

.. autosummary::
    :toctree: api/generated/

    add_macro
    all_macros
    booleans
    del_macro
    functions
    get_macro
    keywords
    stringfunctions
