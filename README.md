# vmd-python
Installable VMD as a python module

## Features
All features of VMD from the current CVS tree, plus some 
optional plugins not included in binary distributions:

    * Read and write formal charges to MAE files
    * DMS plugin for DESRES molecular file format
    * HOOMD plugin
    * It doesn't crash when you import it

The following python modules:

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
    * molecule (read and write all kinds of molecules)
    * molrep
    * render
    * trans
    * vmdnumpy (very very useful!)
    * vmd
    * VMD meta-module (different from vmd. yeah...)

## Usage
To use: `import vmd`.
Then you can import the modules you want.

Please refer to the [documentation](http://www.ks.uiuc.edu/Research/vmd/current/ug/node160.html)
for description of all functions. Also note that the built in help() command in Python
may sometimes give you more up-to-date information than the website.

## Warnings
Don't confuse the Molecule and the molecule modules. Molecule is part of the high-level
Python interface, and can be pickled and have different representations. It's a class. However
molecule is a module that allows loading and saving of molecules while tracking them by an
integer index.

## Installation
Yeah, good luck right now.
Soon an installable version will be on conda.
For now, `python setup.py install`

Soon, `conda install --channel https://conda.binstar.org/rbetz vmd-python`

## Dependencies
This package was designed for use in the [conda](http://conda.pydata.org/)
Python environment. It has the following dependencies:

    * expat >= 2.1.0 (`conda install --channel https://conda.binstar.org/flynn expat`)
    * libnetcdf >= 4.3.2
    * numpy = 1.9.2
    * python 2.7.\*

VMD really cares about the numpy it was compiled against. Make sure
the numpy you are currently using in your python environment matches the one used
when build was invoked. I've required a specific version in hopes of avoiding this
problem.

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

