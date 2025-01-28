"""
The VMD Python module, based off VMD 1.9.4

This software includes code developed by the Theoretical and Computational
Biophysics Group in the Beckman Institute for Advanced Science and
Technology at the University of Illinois at Urbana-Champaign.

Official VMD web page: http://www.ks.uiuc.edu/Research/vmd/

"""

__version__ = "3.1.5"
__author__ = "Robin Betz"

import sys
import os

import importlib.resources
import importlib.util
import importlib.machinery

# Need to set VMDDIR environment variable for vmd.so to import
vmd = importlib.resources.files(__name__) / "libvmd.so"
os.environ["VMDDIR"] = os.path.split(vmd)[0]
os.environ["VMDDISPLAYDEVICE"] = "OPENGLPBUFFER"  # For off screen rendering

# Load the library

loader = importlib.machinery.ExtensionFileLoader(__name__, str(vmd))
spec = importlib.util.spec_from_loader(__name__, loader)
vmd = importlib.util.module_from_spec(spec)
loader.exec_module(vmd)

# These modules define classes for convenient manipulation of VMD state.
# Import them strangely so they're in the same python namespace
vmd.Molecule = importlib.machinery.SourceFileLoader(
    "Molecule", os.path.join(os.environ["VMDDIR"], "Molecule.py")
).load_module()
vmd.Label = importlib.machinery.SourceFileLoader("Label", os.path.join(os.environ["VMDDIR"], "Label.py")).load_module()
vmd.Material = importlib.machinery.SourceFileLoader(
    "Material", os.path.join(os.environ["VMDDIR"], "Material.py")
).load_module()

# Set version and author
vmd.__version__ = __version__
vmd.__author__ = __author__

sys.modules["vmd"] = vmd
