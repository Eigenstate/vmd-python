# vmd-python
Installable VMD as a python module

*NEW*: Version 3.0 has the following new features

* Atomselection attributes can be accessed or set more easily:
  `atomsel.get("x")` can be written as `atomsel.x`!
* Removed extra info dumped to stdout
* Docstrings for all methods and modules
* Much more functionality as Python methods! `atomsel.hbonds`, etc
* The `selection` module lets you define custom atomselection macros
* More rigorous reference counting / fewer memory leaks
* Much prettier code

*NEW*: Support for Python 3!!!

![CI status](https://img.shields.io/travis/Eigenstate/vmd-python.svg)
![Downloads](https://anaconda.org/rbetz/vmd-python/badges/downloads.svg)
![Conda](https://anaconda.org/rbetz/vmd-python/badges/installer/conda.svg)
![PyPi](https://anaconda.org/rbetz/vmd-python/badges/installer/pypi.svg)

## Features
All features of VMD from the 1.9.4 tree, plus some
optional plugins not included in binary distributions:

* Read and write formal charges to MAE files
* DMS plugin for DESRES molecular file format
* HOOMD plugin
* It doesn't crash when you import it
* Doesn't care which numpy you compile against
* Support for Python 2 or Python 3

### [Read the documentation!](https://vmd.robinbetz.com)

### Included modules
The following sub-modules are part of VMD. The import system
makes more sense now, so standard importing like `from vmd import atomsel`
will work correctly.

Some of these modules don't make much sense without the graphical display
window. I don't currently distribute a VMD binary with Python built in,
but you can look at my build scripts and compile your own, or use the
available ones from the official developers.

The following python modules are part of vmd-python. Some of these
are currently not documented on the website (indicated with \*).

* animate
* axes
* atomsel (atom selection language)
* color
* display
* evaltcl (run all your old tcl scripts)
* graphics
* imd
* label
* material
* measure\*
* molecule (read and write all kinds of molecules)
* molrep
* mouse\* (of limited utility in the command line...)
* render
* trans
* topology\*
* vmdmenu\*
* vmdnumpy (very very useful!)

There are also some object oriented classes that provide a higher-level
interface to the sub-modules above. Note these are case sensitive!!

* Label
* Material
* Molecule

## Usage
Here's a very simple example of using the VMD-python API to calculate the root-mean-square-
fluctuation of all tyrosine residues in a loaded trajectory `molid` relative to some
average structure `molid_avg`:

    from vmd import molecule, vmdnumpy
    import numpy as np
    mask = vmdnumpy.atomselect(molid_avg, 0, "resname TYR")
    ref = np.compress(mask, vmdnumpy.timestep(molid_avg, 0), axis=0)

    rmsf = np.zeros(len(ref))
    for frame in range(molecule.numframes(molid)):
        frame = np.compress(mask, vmdnumpy.timestep(molid, f), axis=0)
        rmsf += np.sqrt(np.sum((frame-ref)**2, axis=1))
    rmsf /= float(molecule.numframes(molid)-minframe)
    rmsf = np.sqrt(rmsf)


Please refer to the [documentation](http://www.ks.uiuc.edu/Research/vmd/current/ug/node160.html)
for description of all functions. Also note that the built in help() command in Python
may sometimes give you more up-to-date information than the website.

## Warnings
Don't confuse the Molecule and the molecule modules. Molecule is part of the high-level
Python interface, and can be pickled and have different representations. It's a class. However
molecule is a module that allows loading and saving of molecules while tracking them by an
integer index.

## Installation
The easiest way to install the module is by using the [Conda](https://conda.io/en/latest/)
Python package manager. Vmd-python is in the [conda forge](https://conda-forge.org/)
channel:

Simple binary installation with conda (currently linux-x86\_64 only)

    conda install -c conda-forge vmd-python

For other architectures, you can build from source by cloning this repo:

    python setup.py build
    python setup.py install

Installation can take a while since it compiles VMD from source.

### Installing with EGL support (experimental)

NOTE: The build process for this is currently broken.

VMD-Python can render to an offscreen buffer if you build it yourself with
the --egl flag. You'll need the following files on your system to build and
link successfully:

    * GL/gl.h
    * EGL/egl.h, EGL/eglext.h
    * libOpenGL.so

In Debian, these are provided in `libegl1-mesa-dev`, `mesa-common-dev`,
and `libopengl0` packages. To build:

    python setup.py --egl build
    python setup.py install

Please file issues if you have problems with this!!

### OSX support

Users have reported that on OSX with Tcl/Tk 8.6 the module will
segfault on import. Unfortunately I'm the only developer of this software and
don't have access to a machine running OSX with which to debug the issue.

## Dependencies
vmd-python has the following dependencies:

    * python 2.7 or 3.6
    * numpy
    * libnetcdf >= 4.3
    * expat
    * sqlite
    * Tcl/Tk = 8.5 (8.6 will crash on OSX)

To build on Debian, that's the following packages:

    * libsqlite3-dev
    * libexpat1-dev
    * libnetcdf-dev
    * tcl8.5-dev
    * libegl1-dev
    * libopengl0

## Licensing

This software is classified as a "derivative work" of the original
Visual Molecular Dynamics software. It uses the main VMD source code
and Python compilation options as well as code I have written for simple
compilation and installation, as well as several patches I have applied.

As such, this is a MODIFIED VERSION of VMD and is not the original
software distributed by Illinois.

"This software includes code developed by the Theoretical and Computational
Biophysics Group at the Beckman Institute for Advanced Science and
Technology at the University of Illinois at Urbana-Champaign."

Official VMD web page: http://www.ks.uiuc.edu/Research/vmd

