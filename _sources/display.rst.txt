.. _display:
.. currentmodule:: vmd.display
.. highlight:: python

Display
=======

The display module controls how the VMD render window and GUI look and
are refreshed. These methods can be useful for automated rendering from
Python.

Methods
-------

.. autosummary::
    :toctree: api/generated/

    update
    update_ui
    update_on
    update_off
    set
    get
    stereomodes

Attributes
----------

====================    ==============  ========================================================
Keyword                 Type            Description
--------------------    --------------  --------------------------------------------------------
``eyesep``              ``float``       Distance between eyes for stereo display
``focallength``         ``float``       Focal length
``height``              ``float``       Screen height
``distance``            ``float``       Screen distance
``nearclip``            ``float``       Near clip plane
``farclip``             ``float``       Far clip plane
``antialias``           ``bool``        If anti-aliasing is used
``depthcueue``          ``bool``        If depth cueing should be done
``culling``             ``bool``        If surface culling should be done
``ambientocclusion``    ``bool``        If ambient occlusion should used
``stereo``              ``string``      Stereo mode, in output from ``stereomodes()``
``projection``          ``string``      Projection mode, either Persepective' or 'Orthographic'
``size``                ``(int, int)``  Render window size
``aoambient``           ``float``       Ambient lighting for ambient occlusion
``aodirect``            ``float``       Direct lighting for ambient occlusion
``shadows``             ``bool``        If shadows should be rendered
``dof``                 ``bool``        If rendering should be done with depth of field
``dof_fnumber``         ``float``       F-number for depth of field
``dof_focaldist``       ``float``       Focal distance for depth of field
====================    ==============  ========================================================

