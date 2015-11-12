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

# Need to put python installation libs in LD_LIBRARY_PATH since that is what
# vmd.so was dynamically linked to at compile time
# This is only during the duration of this executable
libdir = os.path.abspath(sys.executable.replace("/bin/python","/lib"))
os.environ['LD_LIBRARY_PATH'] = "%s:%s" % (libdir, os.environ.get('LD_LIBRARY_PATH', default=""))

# Load the library
imp.load_dynamic(__name__, resource_filename(__name__, "vmd.so"))

