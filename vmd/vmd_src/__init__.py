"""
The VMD Python module, based off VMD 1.9.4

This software includes code developed by the Theoretical and Computational
Biophysics Group in the Beckman Institute for Advanced Science and
Technology at the University of Illinois at Urbana-Champaign.

Official VMD web page: http://www.ks.uiuc.edu/Research/vmd/

"""

__version__ = '3.0.6'
__author__ = 'Robin Betz'

import imp
import os
from pkg_resources import resource_filename

# Need to set VMDDIR environment variable for vmd.so to import
vmdlib = resource_filename(__name__, "libvmd.so")
os.environ['VMDDIR'] = os.path.split(vmdlib)[0]
os.environ['VMDDISPLAYDEVICE'] = "OPENGLPBUFFER" # For off screen rendering

# Load the library
vmd = imp.load_dynamic(__name__, vmdlib)

# These modules define classes for convenient manipulation of VMD state.
# Import them strangely so they're in the same python namespace
vmd.Molecule = imp.load_source("Molecule", os.path.join(os.environ["VMDDIR"],
                                                        "Molecule.py"))
vmd.Label = imp.load_source("Label", os.path.join(os.environ["VMDDIR"],
                                                  "Label.py"))
vmd.Material= imp.load_source("Material", os.path.join(os.environ["VMDDIR"],
                                                       "Material.py"))

# Set version and author
vmd.__version__ = __version__
vmd.__author__ = __author__

#==============================================================================

