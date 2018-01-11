# vmd-python

![CI status](https://img.shields.io/travis/Eigenstate/vmd-python.svg)
![Downloads](https://anaconda.org/rbetz/vmd-python/badges/downloads.svg)
![Conda](https://anaconda.org/rbetz/vmd-python/badges/installer/conda.svg)
![PyPi](https://anaconda.org/rbetz/vmd-python/badges/installer/pypi.svg)

Installable VMD as a python module

## Features
All features of VMD from the 1.9.3 release, plus some
optional plugins not included in binary distributions:

* Read and write insertion codes in MAE files
* DMS plugin for DESRES molecular file format
* HOOMD plugin
* Doesn't care which numpy you compile against
* Support for Python 2 or Python 3

## Installation

Simple binary installation with conda (currently linux-x86\_64 only)

    conda install -c https://conda.anaconda.org/rbetz vmd-python

## Building from source

If there is no binary available for your system, you'll have to build
vmd-python either with pip or setuptools. Please report any issues you have
with this process, as it's still under heavy development. I am especially
interested in bug reports for OSX!

First, make sure you have the following dependencies installed somehow and
visible to the compiler and linker:

    * netcdf >= 4.3 (on OSX: `brew install netcdf`)
    * numpy
    * python 2.7 or 3.6

Easy installation with pip:

    pip install -i https://pypi.anaconda.org/rbetz/simple vmd-python

For the latest version:

    git clone https://github.com/Eigenstate/vmd-python.git
    cd vmd-python
    python setup.py build
    python setup.py install

Installation can take a while since it compiles VMD from source.


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

## Dependencies
vmd-python has the following dependencies:

    * libnetcdf >= 4.3
    * numpy
    * python 2.7 or 3.6

On Mac systems, an easy way to satisfy these dependencies is to install 
[homebrew](https://brew.sh/) and run

    brew install netcdf
    pip install numpy
    

    
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

