.. _molecule:
.. currentmodule:: vmd.molecule
.. highlight:: python

Molecule
=========

Methods to interact with molecule, including loading data, querying molecule
attributes, working with volumetric data, or setting its visibility in the
render window.

Basic functionality
-------------------
.. autosummary::
    :toctree: api/generated/

    cancel
    delete
    delframe
    dupframe
    exists
    listall
    load
    new
    num
    read
    rename
    write
    ssrecalc

Typically for molecular dynamics simulation trajectories, a topology file
will be loaded, then a trajectory file will be read in to it in a way that
blocks program execution until all frames are done loading. This code snippit
does this, then saves the last frame as a PDB file.

.. code-block:: python

    from vmd import molecule

    molid = molecule.load("parm7", "system.prmtop")
    molecule.read(molid, "netcdf", "simulation.nc", skip=5, waitfor=-1)
    last_frame = molecule.numframes - 1
    molecule.write(molid, "pdb", "last_frame.pdb", first=last_frame)


VMD has the concept of the "top" molecule, which is the default molecule
for all operations unless otherwise specified.

.. autosummary::
    :toctree: api/generated/

    get_top
    set_top

Querying loaded molecules
-------------------------
Depending on the information available in the input molecule files, these
fields may or may not be populated.

.. autosummary::
    :toctree: api/generated/

    get_accessions
    get_databases
    get_periodic
    get_physical_time
    get_remarks
    set_periodic
    set_physical_time

These attributes are always available:

.. autosummary::
    :toctree: api/generated/

    get_filenames
    get_filetypes
    get_frame
    get_visible
    name
    numatoms
    numframes
    set_frame
    set_visible

Working with volumetric data
----------------------------

.. autosummary::
    :toctree: api/generated/

    add_volumetric
    delete_volumetric
    get_volumetric
    num_volumetric
