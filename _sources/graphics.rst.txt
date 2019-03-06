.. _graphcis:
.. currentmodule:: vmd.graphics
.. highlight:: python

Graphics
========

Draw graphics primitives to the render window. This module is pretty
useless in vmd-python without EGL support.

Graphics primitives
-------------------
Here are all of the simple shapes you can draw:

.. autosummary::
    :toctree: api/generated/

    cone
    cylinder
    line
    point
    sphere
    text
    triangle
    trinorm


Manipulating graphics primitives
--------------------------------

.. autosummary::
    :toctree: api/generated/

    listall
    replace
    delete
    info


Changing the style of graphics
------------------------------

VMD draws graphics in a "stack" of commands that are applied in order. Instead
of setting the material or color for individual graphics primitives, the
color or material is defined as a command put on the stack, then all subsequent
primitives will be drawn with that style until a new color or material is set.

.. autosummary::
    :toctree: api/generated/

    color
    material
    materials

The following code will draw a red cylinder, then a blue sphere:

.. code-block:: python

    from vmd import molecule, graphics

    drawmol = molecule.new(name="Drawings")
    graphics.color(drawmol, 1)
    cylinder_id = graphics.cylinder(drawmol, (0,0,0), (0,1,0), filled=True)
    graphics.color(drawmol, 0)
    sphere_id = graphics.sphere(drawmol, (0,0,0), radius=0.3)

