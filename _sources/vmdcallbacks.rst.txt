.. highlight:: python

The `vmdcallbacks` and `vmdmenu` modules are only useful in the context of
GUI VMD using a compiled-in Python interpreter.

.. _callbacks:

VMD Callbacks
=============
.. currentmodule:: vmd.vmdcallbacks

Callbacks are methods that are invoked whenever some GUI trigger happens,
for example, clicking an item in a menu. Adding callbacks is useful to
create interactive graphs or plugins that alter VMD or molecule state in
response to user input.

.. autosummary::
    :toctree: api/generated/

    add_callback
    del_callback

.. _menu:

VMD Menu
========
.. currentmodule:: vmd.vmdmenu

New menus can be added to the GUI using the `vmdmenu` module:

.. autosummary::
    :toctree: api/generated/

    add
    location
    register
    show

