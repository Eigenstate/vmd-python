.. vmd-python documentation master file, created by
   sphinx-quickstart on Tue Oct 10 12:13:39 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. highlight:: bash

VMD-Python
===========

VMD is an excellent visualization program, with powerful command-line
scripting functionality. Recently, VMD's C functions can be invoked using
a set of easy to use python modules. Vmd-python is installable VMD as a
Python module.

I've pulled some bonus features out of the source code and made it work
with Python 2 or Python 3, as well as being (relatively) easy to build.

Although there is some `online documentation <http://www.ks.uiuc.edu/Research/vmd/current/ug/node160.html>`_, it is not quite complete. Here I've collected a full reference
of all available functionality in the vmd-python module. Note that this
does not include GUI features you may find helpful, such as Python callbacks
or adding Tkinter menus.

Features
--------

- All features of VMD from the 1.9.4 alpha release
- DMS plugin for DESRES molecular file format
- HOOMD plugin (with libexpat)
- Doesn't depend on the exact numpy version it was built agains
- Can be built with any Python
- Support for Python 2 or 3

Installation
------------

I recommend using the pre-built binary packages with the
`Anaconda Python Distribution <https://www.anaconda.com/download>`_.
Currently, there are packages available for Linux and OSX::

    conda install -c conda-forge vmd-python

For other architectures, you can download the
`source <https://github.com/Eigenstate/vmd-python>`_::

    python setup.py build
    python setup.py install

VMD-Python has the following dependencies:

- libnetcdf >= 4.3
- numpy
- libexpat

Modules
-------
.. toctree::
   :maxdepth: 1

   animate
   atomsel
   axes
   color
   display
   evaltcl
   graphics
   imd
   label
   material
   measure
   molecule
   molrep
   mouse
   render
   selection
   topology
   trans
   vmdcallbacks
   vmdnumpy

