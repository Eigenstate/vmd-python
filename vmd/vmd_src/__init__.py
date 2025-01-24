"""
The VMD Python module, based off VMD 1.9.4

This software includes code developed by the Theoretical and Computational
Biophysics Group in the Beckman Institute for Advanced Science and
Technology at the University of Illinois at Urbana-Champaign.

Official VMD web page: http://www.ks.uiuc.edu/Research/vmd/

"""

__version__ = "3.1.3"
__author__ = "Robin Betz"

import sys
import os

import importlib.resources
import importlib.util
import importlib.machinery

# Need to set VMDDIR environment variable for vmd.so to import
vmdlib = importlib.resources.files(__name__) / "libvmd.so"
os.environ["VMDDIR"] = os.path.split(vmdlib)[0]
os.environ["VMDDISPLAYDEVICE"] = "OPENGLPBUFFER"  # For off screen rendering

# Load the library

loader = importlib.machinery.ExtensionFileLoader(__name__, str(vmdlib))
spec = importlib.util.spec_from_loader(__name__, loader)
vmdlib = importlib.util.module_from_spec(spec)
loader.exec_module(vmdlib)

# These modules define classes for convenient manipulation of VMD state.
# Import them strangely so they're in the same python namespace
vmdlib.Molecule = importlib.machinery.SourceFileLoader(
    "Molecule", os.path.join(os.environ["VMDDIR"], "Molecule.py")
).load_module()
vmdlib.Label = importlib.machinery.SourceFileLoader("Label", os.path.join(os.environ["VMDDIR"], "Label.py")).load_module()
vmdlib.Material = importlib.machinery.SourceFileLoader(
    "Material", os.path.join(os.environ["VMDDIR"], "Material.py")
).load_module()

# Set version and author
vmdlib.__version__ = __version__
vmdlib.__author__ = __author__

sys.modules["vmd"] = vmdlib
