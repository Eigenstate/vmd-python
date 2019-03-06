.. _color:
.. currentmodule:: vmd.color

Color
=====

Methods for changing how VMD colors things. Can be useful for rendering
from the Python API.

Basic colors
------------

VMD defines many colors by name. You can actually assign any RGB value to
these names. However, you currently cannot add additional colors. For use
of many colors, see below for how to use color scales.

.. autosummary::
    :toctree: api/generated/

    get_colors
    get_colorlist
    set_colors
    set_colorid
    categories

Color mappings
--------------

VMD likes to classify colors into categories. For example, the "Name" category
is used when coloring atoms by name. The "Type" category is used to color atoms
by type, etc. Some categories are used for internal display purposes, such as
the "Display" and "Axes" categories.

Changing the color mappings can be useful for things like coloring the carbons
in a ligand differently than those in the protein, while preserving the standard
coloring scheme for other elements.

    >>> from vmd import color
    >>> color.set_colormap("Type", {"C": "green3"})

.. autosummary::
    :toctree: api/generated/

    get_colormap
    set_colormap

Color scale
-----------

The color scale is used for coloring continuous values, such as the "User" or
"Charge" coloring schemes. It's a gradient of color with a minimum, middle,
and maximum value.

    >>> from vmd import color
    >>> color.scale_methods()
    ['RWB', 'BWR', 'RGryB', 'BGryR', 'RGB', 'BGR', 'RWG', 'GWR', 'GWB', 'BWG', 'BlkW', 'WBlk']

.. autosummary::
    :toctree: api/generated

    scale_method
    scale_methods
    scale_midpoint
    scale_min
    scale_max
    set_scale



