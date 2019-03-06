.. _trans:
.. currentmodule:: vmd.trans
.. highlight:: python

Trans
=====

Methods for manipulating the transformations applied to a molecule in VMD
coordinate space, including position, rotation, center, and scale.


Per-molecule transformations
----------------------------
.. autosummary::
    :toctree: api/generated/

    fix
    get_center
    get_rotation
    get_scale
    get_translation
    is_fixed
    is_shown
    set_center
    set_rotation
    set_translation
    show

Global transformations
----------------------
.. autosummary::
    :toctree: api/generated/

    resetview
    rotate_scene
    scale_scene
    translate_scene

