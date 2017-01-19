Plumed-GUI
==========

Toni Giorgino  
Consiglio Nazionale delle Ricerche (IN-CNR)



A Plumed collective variable analysis tool for VMD
------------

The PLUMED-GUI collective variable analysis tool is a plugin for the Visual Molecular Dynamics (VMD) software that provides access to the extensive set of collective variables (CV) defined in the PLUMED. It allows you to:

- analyze the currently loaded trajectory by evaluating and plotting arbitrary CVs
- use VMD's atom selection keywords to define atom groups and ready-made templates for common CVs
- export the CV definition file for use in MD simulations
- prepare reference files for RMSD, path-variable, native contacts, etc.
 
The code is hosted on GitHub at
[tonigi/vmd_plumed](https://github.com/tonigi/vmd_plumed).




Installation
------------

First, you need PLUMED 1.3's *driver* and/or PLUMED 2's *plumed*
executables: see [INSTALL-PLUMED-FIRST.md](doc/INSTALL-PLUMED-FIRST.md).

Second, you may want to update PLUMED-GUI to its latest version,
rather than using the one distributed with VMD. See instructions
in [INSTALL.md](doc/INSTALL.md).




Documentation
-------------

Please find

- A short manual and quickstart in the [doc/README.md](doc/README.md) file
- An extensive description in (please read and cite) 
  * Toni Giorgino, _Plumed-GUI: an environment for the interactive development of molecular dynamics analysis and biasing scripts_ (2014) Computer Physics Communications, Volume 185, Issue 3, March 2014, Pages 1109-1114, [doi:10.1016/j.cpc.2013.11.019](http://dx.doi.org/10.1016/j.cpc.2013.11.019), or [arXiv:1312.3190](http://arxiv.org/abs/1312.3190)  
- Information on the PLUMED engine at http://www.plumed.org 



Citation
--------

You are kindly requested to cite the following paper in any
publication resulting from the use of Plumed-GUI (in addition to other
possibly relevant Plumed citations):

- Toni Giorgino, _Plumed-GUI: an environment for the interactive
  development of molecular dynamics analysis and biasing scripts_
  (2014) Computer Physics Communications,
  [doi:10.1016/j.cpc.2013.11.019](http://dx.doi.org/10.1016/j.cpc.2013.11.019)
  or http://arxiv.org/abs/1312.3190 .





