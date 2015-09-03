############################################################################
#cr
#cr            (C) Copyright 1995-2007 The Board of Trustees of the
#cr                        University of Illinois
#cr                         All Rights Reserved
#cr
############################################################################

############################################################################
# RCS INFORMATION:
#
#       $RCSfile: Molecule.py,v $
#       $Author: akohlmey $        $Locker:  $             $State: Exp $
#       $Revision: 1.30 $        $Date: 2009/07/21 17:21:14 $
#
############################################################################

"""
/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2007 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/
"""

import VMD
import molecule, molrep

class Molecule:
  """
  The Molecule class is a proxy for molecules loaded into VMD.  Most 
  operations raise ValueError if the proxy no longer refers to a valid
  molecule (i.e. if the molecule has been deleted).
  """
  
  def __init__(self, name=None, id=None, atoms=0):
    """
    Creating a new Molecule instance with no arguments will create a new 
    empty molecule in VMD.  Passing a valid molecule id will make the 
    Molecule instance mirror the state of the corresponding molecule in VMD. 
    If id is None, and a name is given, the new molecule will have that name.
    """
    if id is None:
      if name is None:
        name = "molecule"
      self.id = molecule.new(name, atoms)
    else:
      self.id = id
      self.atoms = molecule.numatoms(self.id)

    if not molecule.exists(self.id):
      raise ValueError, "Molecule with id %d does not exist." % self.id
  
  def __int__(self):
    """ Casting a Molecule to an int returns the molecule ID."""
    return self.id

  def __getstate__(self):
    """ We define __getstate__ and _setstate__ methods so that Molecule
    instances can be pickled and restored with their original state. """
    return [self.name(), self.files(), self.types(), self.reps()]

  def __setstate__(self, state):
    name, files, types, reps = state
    self.id = molecule.new(name, 0)
    for file, type in zip(files, types):
      self.load(file, type)
    self.clearReps()
    for rep in reps:
      self.addRep(rep)

  def rename(self, newname):
    """ Change the name of the molecule.  """
    molecule.rename(self.id, newname)
    return self

  def name(self):
    """ Returns the name of the molecule. """
    return molecule.name(self.id)

  def delete(self):
    """ Deletes the molecule corresonding to this Molecule instance.  The
    object can no longer be used."""
    molecule.delete(self.id)
    
  def _guessFiletype(self, filename):
    """ Utility routine for guessing filetype from filename."""
    filetype='pdb'
    ind=filename.rfind('.')
    if ind >= 0:
      filetype = filename[ind+1:]
    else:
      if len(filename) is 4:
        filetype = 'webpdb'
    return filetype

  def load(self, filename, filetype=None, first=0, last=-1, step=1, waitfor=-1, volsets=[0]):
    """
    Load molecule data from the given file.  The filetype will be guessed from
    the filename extension; this can be overridden by setting the filetype
    option.  first, last, and step control which coordinates frames to load,
    if any.
    """
    if filetype is None:
      filetype = self._guessFiletype(filename)
    if molecule.read(self.id, filetype, filename, first, last, step, waitfor, volsets) < 0:
      raise IOError, "Unable to load file: %s" % filename
    return self

  def save(self, filename, filetype=None, first=0, last=-1, step=1, waitfor=-1, sel=None):
    """
    Save timesteps to the given file.  The filetype will be guessed from
    the filename extension; this can be overridden by setting the filetype
    option.  first, last, and step control which timesteps to save.  Returns
    the number of frames written before the command completes.
    """
    if filetype is None:
      filetype = self._guessFiletype(filename)
####################################################
# XXX AtomSel is no longer imported automatically,
#     so the following piece of code will fail when 
#     using the save() method with a selection.
#     if the selection is an atomsel builtin type, it
#     can be passed on directly. otherwise we get a
#     reasonable error message. so this chunk is no
#     longer needed.  AK. 2009/07/21
####################################################
#    if sel:
#      if isinstance(sel, VMD.AtomSel.AtomSel):
#        sel=sel.list()
#      else:
#        try:
#          sel=tuple(sel)
#        except:
#          sel=None
####################################################
    nframes = molecule.write(self.id, filetype, filename, first, last, step, waitfor, selection=sel)
    if nframes < 0:
      raise IOError, "Unable to save file: %s" % filename
    return nframes

  def files(self):
    """ Returns list of files that have been loaded for this molecule. """
    return molecule.get_filenames(self.id)

  def types(self):
    """ Returns list of filetypes corresponding to files(). """
    return molecule.get_filetypes(self.id)

  def numAtoms(self):
    """ Returns the number of atoms in the molecule."""
    return molecule.numatoms(self.id)

  def numFrames(self):
    """ Returns the number of coordinate frames in the molecule. """
    return molecule.numframes(self.id)

  def setFrame(self, frame):
    """ Set the coordinate frame to the given value.  Must be in the range
    [0, numFrames())"""
    molecule.set_frame(self.id, frame)
    return self

  def curFrame(self):
    """ Returns the current coordinate frame for the molecule. """
    return molecule.get_frame(self.id)
  
  def delFrame(self, first=0, last=-1, step=-1):
    """ Delete the given range of frames."""
    molecule.delframe(self.id, beg=first, end=last, skip=step)
    return self

  def dupFrame(self, frame = None):
    """ Duplicate the given frame, appending it to the end.  If no frame is
    given then the current frame is used."""
    if frame is None: 
      frame = self.curFrame()
    molecule.dupframe(self.id, frame)
    return self

     
  def numReps(self):
    """ Returns the number of molecular representations (reps) in the given 
    molecule. """
    return molrep.num(self.id)
    
  def reps(self):
    """ Creates MoleculeRep objects, one for each rep in the molecule. """
    result = []
    for i in range(molrep.num(self.id)):
      style = molrep.get_style(self.id, i)
      color = molrep.get_color(self.id, i)
      sel   = molrep.get_selection(self.id, i)
      mat   = molrep.get_material(self.id, i)
      name  = molrep.get_repname(self.id, i)
      rep=MoleculeRep(style=style, color=color, selection=sel, material=mat)
      rep.assign_molecule(self.id, name)
      result.append(rep)
    return result

  def addRep(self, rep):
    """ Add the given rep to the molecule.  Modifications to the rep will
    affect all molecules to which the rep has been added."""
    if rep.molecules.has_key(self.id):
      raise ValueError, "This molecule already has this rep"
    molrep.addrep(self.id, style=rep.style, color=rep.color, selection=rep.selection, material=rep.material)
    repid = molrep.num(self.id)-1
    repname = molrep.get_repname(self.id, repid)
    rep.assign_molecule(self.id, repname)
    
  def delRep(self, rep):
    """ Remove the given rep from the molecule."""
    repname = rep.remove_molecule(self.id)
    if repname:
      repid = molrep.repindex(self.id, repname)
      if repid >= 0:
        molrep.delrep(self.id, repid)

  def clearReps(self):
    """ Removes all reps from this molecule."""
    for i in range(molrep.num(self.id)):
      molrep.delrep(self.id, 0)

  def autoUpdate(self, rep, onoff = None):
    """ Get/set whether this rep should be recalculated whenever the coordinate
    frame changes for this molecule. """

    # Get the repid for this rep
    try:
      name = rep.molecules[self.id]
    except KeyError:
      raise ValueError, "This molecule does not contain this rep."
    repid = molrep.repindex(self.id, name)
    if onoff is None:
      return molrep.get_autoupdate(self.id, repid)
    molrep.set_autoupdate(self.id, repid, onoff)
    return self

  def ssRecalc(self):
    """ Recalculate the secondary structure for this molecule."""
    return molecule.ssrecalc(self.id)
  
def moleculeList():
  """ Returns a Molecule instance for all current molecules. """
  return [Molecule(id=id) for id in molecule.listall()]

class MoleculeRep:
  """
  The MoleculeRep class defines a representation for a molecule.  Adding a
  rep to a molecule causes VMD to draw that rep using the atoms of the
  molecule.  Changing the rep properties changes the view of that rep for
  all molecules to which the rep has been assigned.  Deleting the rep does
  nothing; remove the rep from the molecule in order to remove the view.
  """
  defStyle = "Lines"
  defColor = "Name"
  defSelection = "all"
  defMaterial = "Opaque"

  def __init__(self, style=defStyle, color=defColor, selection=defSelection, material=defMaterial):
    """ Constructor - initialize with style, color, selection or material if 
    you like."""
    
    self.style=style
    self.color=color
    self.selection=selection
    self.material=material

    # molecules hashes Molecule objects to rep names
    self.molecules = {} 

  def __str__(self):
    """ Returns human-readable summary of the rep properties."""
    return "Molecule.MoleculeRep instance\n-------------\n  Style:     %s\n  Color:     %s\n  Selection: %s\n  Material:  %s" % (self.style, self.color, self.selection, self.material)

  def __getstate__(self):
    d=self.__dict__.copy()
    del d['molecules']
    return d

  def __setstate__(self, state):
    self.__dict__.update(state)
    self.molecules = {}

  def assign_molecule(self, id, name):
    """ Internal method. """
    self.molecules[id] = name

  def remove_molecule(self, id):
    """ Internal method. """
    try:
      name = self.molecules[id]
      del self.molecules[id]
      return name
    except:
      return None

  def changeStyle(self, style):
    """ Change the draw style of the rep to 'style'."""
    for id,name in self.molecules.items():
      repid = molrep.repindex(id, name)
      if not molrep.modrep(id, repid, style=style):
        raise ValueError, "Invalid style '%s'" % style
    self.style = str(style)
  
  def changeColor(self, color):
    """ Change the coloring method the rep to 'color'."""
    for id,name in self.molecules.items():
      repid = molrep.repindex(id, name)
      if not molrep.modrep(id, repid, color=color):
        raise ValueError, "Invalid color '%s'" % color 
    self.color = str(color)
  
  def changeSelection(self, selection):
    """ Change the atom selection of the rep to 'selection'."""
    for id,name in self.molecules.items():
      repid = molrep.repindex(id, name)
      if not molrep.modrep(id, repid, sel=selection):
        raise ValueError, "Invalid selection '%s'" % selection 
    self.selection = str(selection)
  
  def changeMaterial(self, material):
    """ Change the material for the rep to 'material'."""
    for id,name in self.molecules.items():
      repid = molrep.repindex(id, name)
      if not molrep.modrep(id, repid, material=material):
        raise ValueError, "Invalid material'%s'" % material 
    self.material = str(material)

# The following functions return strings that can be passed to 
# MoleculeRep.changeStyle to set a particular drawing style.
# Use keyword arguments to set the various style parameters.

defCartoonRes    = 12
defBondRes       = 6
defSphereRes     = 8
defLineThickness = 1

def linesStyle(thickness=defLineThickness):
  return "Lines %d" % thickness

def bondsStyle(rad=0.3, res=defBondRes):
  return "Bonds %f %d" % (rad, res)

def dynamicBondsStyle(cutoff=3.0, rad=0.3, res=defBondRes):
  return "DynamicBonds %f %f %d" % (cutoff, rad, res)

def hbondsStyle(cutoff=3.0, angle=20.0, thickness=1):
  return "HBonds %f %f %d" % (cutoff, angle, thickness)

def pointsStyle():
  return "Points"

def vdwStyle(rad=1.0, res=defSphereRes):
  return "VDW %f %d" % (rad, res)

def cpkStyle(sphereRad=1.0, bondRad=0.3, sphereRes=defSphereRes, bondRes=defBondRes):
  return "CPK %f %f %d %d" % (sphereRad, bondRad, sphereRes, bondRes)

def licoriceStyle(bondRad=0.3, sphereRes=defSphereRes, bondRes=defBondRes):
  return "Licorice %f %s %s" % (bondRad, sphereRes, bondRes)

def traceStyle(rad=0.3, res=defBondRes):
  return "Trace %f %d" % (rad, res)

def tubeStyle(rad=0.3, res=defBondRes):
  return "Tube %f %d" % (rad, res)

def ribbonsStyle(rad=0.3, res=defBondRes, thickness=2.0):
  return "Ribbons %f %d %f" % (rad, res, thickness)

def newRibbonsStyle(rad=0.3, res=defBondRes, thickness=3.0):
  return "NewRibbons %f %d %f" % (rad, res, thickness)

def cartoonStyle(rad=2.1, res=defCartoonRes, thickness=5.0):
  return "Cartoon %f %d %f" % (rad, res, thickness)

def msmsStyle(probe=1.5, density=1.5, allAtoms=0, wireframe=0):
  return "MSMS %f %f %d %d" % (probe, density, allAtoms, wireframe)

def surfStyle(probe=1.4, wireframe=0):
  return "Surf %f %d" % (probe, wireframe)

def volumeSliceStyle(slice=0.5, volID=0, axis=0, quality=1):
  return "VolumeSlice %f %d %d %d" % (slice, volID, axis, quality)

def isosurfaceStyle(isovalue=0.5, volID=0, boxOnly=1, wireframe=1):
  return "Isosurface %f %d %d %d" % (isovalue, volID, boxOnly, wireframe)

def dottedStyle(rad=1.0, res=defSphereRes):
  return "Dotted %f %d" % (rad, res)

def solventStyle(probe=0, detail=7, method=1):
  return "Solvent %f %d %d" % (probe, detail, method)

def offStyle():
  return "Off"


