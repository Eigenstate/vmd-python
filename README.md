# vmd-python
Installable VMD as a python module
*NEW* Support for Python 3!!!

![CI status](https://anaconda.org/rbetz/vmd-python/badges/build.svg)
![Downloads](https://anaconda.org/rbetz/vmd-python/badges/downloads.svg)

## Features
All features of VMD from the current CVS tree, plus some
optional plugins not included in binary distributions:

* Read and write formal charges to MAE files
* DMS plugin for DESRES molecular file format
* HOOMD plugin
* It doesn't crash when you import it
* Doesn't care which numpy you compile against
* Support for Python 2 or Python 3

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
* evaltcl (run all your old scripts)
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
Please refer to the [documentation](http://www.ks.uiuc.edu/Research/vmd/current/ug/node160.html)
for description of all functions. Also note that the built in help() command in Python
may sometimes give you more up-to-date information than the website.

## Warnings
Don't confuse the Molecule and the molecule modules. Molecule is part of the high-level
Python interface, and can be pickled and have different representations. It's a class. However
molecule is a module that allows loading and saving of molecules while tracking them by an
integer index.

## Installation
Wow it is INSTALLABLE NOW! This has been really hard to get working
so please be happy for me.

Simple binary installation with conda (currently linux-x86\_64 only)

    conda install -c https://conda.anaconda.org/rbetz vmd-python

For other architectures, use pip:
Easy installation with pip:

    pip install -i https://pypi.anaconda.org/rbetz/simple vmd-python

Or, if you download the source:

    python setup.py build
    python setup.py install

Installation can take a while since it compiles VMD from source.

## Dependencies
vmd-python has the following dependencies:

    * libnetcdf
    * numpy
    * python

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

