.. _atomsel:
.. currentmodule:: vmd.atomsel
.. highlight:: python

Atomsel
=======

This is the low-level (non object-oriented) interface to atom selection
objects.

Basic actions
-------------

To create a new atomsel:
.. currentmodule:: vmd.atomsel
.. automethod:: __init__

Atomsel objects have attributes. New in vmd-python > 3.0.0, you can access
these attributes directly from the atomsel object to both get and set!
For example:

.. code-block:: python

    protein = atomsel("protein")
    protein.chain = "A"

Here are basic methods working with an atom selection object:

.. autosummary::
    :toctree: api/generated

    update
    write


Fitting selections to each other
--------------------------------

These methods can be used to move selections around or calculate the differences
between them.

.. autosummary::
    :toctree: api/generated/

    fit
    moveby
    move


Numerical calculations
----------------------

Many calculations can be performed on atomsel objects. For speed, the
following are implemented internally in C:

.. autosummary::
    :toctree: api/generated/

    center
    centerperresidue
    contacts
    hbonds
    minmax
    rgyr
    rmsd
    rmsdQCP
    rmsdperresidue
    rmsf
    rmsfperresidue
    sasa


Attributes
----------

There are many attributes, some of which are changeable by the user. To
get them, use:

.. autosummary::
    :toctree: api/generated

    list_attributes

Here are the atomselection attributes that may be get or set using this
module. If the attribute refers to some inherent feature of the atom (such
as "backbone"), using the get function returns a list of booleans.

If changing attributes, make sure to call `update()` on the
atom selection to ensure that changes are applied to your atom selection.

Molecule attributes
^^^^^^^^^^^^^^^^^^^
These attributes describe the atom selection itself, and may be assigned to.

.. autosummary::
    :toctree: api/generated

    bonds
    frame
    molid

Per-atom attributes
^^^^^^^^^^^^^^^^^^^
These attributes are features characteristic of an atom:

- ``name``: Atom name
- ``type``: Atom type
- ``index``: Atom index, counting from 1
- ``serial``: Atom index, counting from 0
- ``mass``: Atomic mass
- ``atomicnumber``: Atomic number
- ``element``: Atomic element
- ``altloc``: Alternate atom location, if present
- ``insertion``: Atom insertion code
- ``numbonds``: Number of bonds atom participates in. Does not take into account bond order
- ``beta``: Beta factor for the atom, if present
- ``occupancy``: Occupancy of the atom, if present
- ``charge``: Charge on the atom, if present
- ``radius``: The radius of the atom. Used in SASA calculations and visualization

Per-atom spatial attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^^
These attributes deal mostly with the coordinates of the atoms

- ``x``, ``y`` and ``z``: X, Y, or Z coordinate of the atom
- ``vx``, ``vy``, and ``vz``: X, Y, or Z velocity of the atom, if present
- ``ufx``, ``ufy``, and ``ufz``: X, Y, or Z force on the atom, if present

Context attributes
^^^^^^^^^^^^^^^^^^
These attributes give information about the atom in the context of its
environment.

- ``residue``: Internal residue number to which atom belongs, counting from 0.
- ``resid``: Residue ID to which atom belongs, using canonical numbering if present in input molecule
- ``resname``: Residue name to which atom belongs
- ``chain``: Chain to which atom belongs, usually a single letter
- ``segname``: Segment name to which atom belongs, if present
- ``segid``: Segment index to which atom belongs
- ``fragment``: Molecular fragment to which atom belongs, counting from 0
- ``pfrag``: Protein fragment to which atom belongs, counting from 0. If atom is not part of a protein fragment, is set to -1
- ``nfrag``: Nucleic acid fragment to which atom belongs, counting from 0. If atom is not part of a nucleic acid, is set to -1
- ``phi``: Phi backbone angle for the residue this atom is in
- ``psi``: Psi backbone angel for the residue this atom is in
- ``backbone``: If the atom is part of a protein backbone
- ``sidechain``: If the atom is in a protein sidechain
- ``protein``: If the atom is part of a protein
- ``nucleic``: If the atom is part of a nucleic acid
- ``water`` or ``waters``: If the atom is water
- ``vmd_fast_hydrogen``: True if the atom is hydrogen, faster than using element.

Secondary structure attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If there is no secondary structure information parsed from the input molecule
file, the accuracy of these attributes depends on the quality of predicted
secondary structure by VMD or STRIDE.

The DSSP classification corresponding to each keyword is also listed.

- ``helix``: *G,H,I* If the atom is part of a helix
- ``alpha_helix``: *H* If the atom is part of an alpha helix
- ``helix_3_10``: *G* If the atom is part of a 3 :sub:`10` `helix <https://en.wikipedia.org/wiki/310_helix>`_
- ``pi_helix``: *I* If the atom is part of a `pi helix <https://en.wikipedia.org/wiki/Pi_helix>`_
- ``sheet``, ``betasheet``, or ``beta_sheet``: *E,B* If the atom is part of a beta sheet
- ``extended_beta``: *E* If the atom is part of an extended sheet
- ``bridge_beta``: *B* If the atom is part of an isolated beta bridge
- ``turn``: *T* If the atom is part of a hydrogen-bonded turn
- ``coil``: *C* If the atom is part of any other secondary structure conformation
- ``structure``: The DSSP classification of the secondary structure
- ``pucker``: The amount of pucker on a ring, designed for carbohydrates

Volumetric attributes
^^^^^^^^^^^^^^^^^^^^^
These fields are populated when there is a volumetric dataset loaded into the
molecule. Otherwise, they're all NaN.

- ``volindex0`` through ``volindex7``: Indices into volumetric dataset for atom
- ``vol0`` through ``vol7``: Volume value at this atom
- ``interpvol0`` through ``interpvol7``: Interpolated volume value at this atom

Custom attributes
^^^^^^^^^^^^^^^^^
These fields can be used to assign per-atom data of your choice:
- ``user``, ``user2``, ``user3``, ``user4``: Four floating-point fields for your data
- ``flag1`` through ``flag7``: Seven boolean fields for your data
