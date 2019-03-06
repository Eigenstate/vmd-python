.. _topology:
.. currentmodule:: vmd.topology
.. highlight:: python

Topology
========

Methods for querying or modifying the topology of a molecule, which consists
of all defined atoms, bonds, angles, dihedrals, and impropers, for applicable
topology formats (usually NAMD).

Not all molecular file formats will explicitly define these terms. Carefully
check if these methods are applicable to your file.


Editing individual terms
------------------------
.. autosummary::
    :toctree: api/generated/

    addangle
    addbond
    adddihedral
    addimproper
    delangle
    delbond
    deldihedral
    delimproper

Per-molecule functions
----------------------
.. autosummary::
    :toctree: api/generated/

    angles
    bonds
    dihedrals
    impropers
    delallangles
    delallbonds
    delalldihedrals
    delallimpropers


Forcefield parameters
---------------------
.. autosummary::
    :toctree: api/generated/

    angletypes
    bondtypes
    dihedraltypes
    impropertypes
