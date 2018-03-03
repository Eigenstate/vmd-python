"""
The VMD Python module, based off VMD 1.9.3.

This software includes code developed by the Theoretical and Computational
Biophysics Group in the Beckman Institute for Advanced Science and
Technology at the University of Illinois at Urbana-Champaign.

Official VMD web page: http://www.ks.uiuc.edu/Research/vmd/

"""

__version__ = '2.0.8'
__author__ = 'Robin Betz'

import imp
import os
from pkg_resources import resource_filename

# Need to set VMDDIR environment variable for vmd.so to import
os.environ['VMDDIR'] = resource_filename(__name__, "libvmd.so").replace("/libvmd.so","")

# Load the library
vmd = imp.load_dynamic(__name__, resource_filename(__name__, "libvmd.so"))

# These modules define classes for convenient manipulation of VMD state.
# Import them strangely so they're in the same python namespace
vmd.Molecule = imp.load_source("Molecule", os.path.join(os.environ["VMDDIR"],
                                                        "Molecule.py"))
vmd.Label = imp.load_source("Label", os.path.join(os.environ["VMDDIR"],
                                                  "Label.py"))
vmd.Material= imp.load_source("Material", os.path.join(os.environ["VMDDIR"],
                                                       "Material.py"))

#==============================================================================

# This evaluates tcl commands
def evaltcl(args):
  """
  Use this method to execute Tcl script code from python. Since many new
  features in VMD are first implemented for the Tcl text interpreter,
  this will help to bridge the time until those are also available through
  the python script interface. It can also be used to execute whole Tcl
  scripts using the VMD 'play' command, so it is the equivalent to
  'gopython' in the Tcl interpreter. If the tcl is not available, it
  will have no effect.

  Usage examples:

  from VMD import evaltcl

  versnum=evaltcl('vmdinfo version')
  evaltcl('play somescript.tcl')
  """
  try:
    from vmd import VMDevaltcl
  except:
    return
  return VMDevaltcl(args)

#==============================================================================

vmd.evaltcl = evaltcl

