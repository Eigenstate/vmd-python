"""
The VMD Python module.

This software includes code developed by the Theoretical and Computational 
Biophysics Group in the Beckman Institute for Advanced Science and 
Technology at the University of Illinois at Urbana-Champaign.

Official VMD web page: http://www.ks.uiuc.edu/Research/vmd/

"""

__version__ = '1.9.2'
__author__ = 'Robin Betz'

import imp
import os
import sys
from pkg_resources import resource_filename

# Need to set VMDDIR environment variable for vmd.so to import
os.environ['VMDDIR'] = resource_filename(__name__, "vmd.so").replace("/vmd.so","")

# Need VMD python scripts accessible (for import molecule, etc)
sys.path.append(resource_filename(__name__, "scripts/python/VMD.py").replace("VMD.py", ""))

# Load the library
imp.load_dynamic(__name__, resource_filename(__name__, "vmd.so"))

