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
#       $RCSfile: AtomSel.py,v $
#       $Author: johns $        $Locker:  $             $State: Exp $
#       $Revision: 1.22 $        $Date: 2007/03/03 19:54:47 $
#
############################################################################

_deprecation_warning = "The AtomSel module is deprecated; please use atomsel"
try:
  import warnings
except:
  print _deprecation_warning
else:
  warnings.warn(_deprecation_warning, DeprecationWarning)

import atomselection, molecule

def macro(*args, **keywds):
  return apply(atomselection.macro, args, keywds)

class AtomSel:
  def __init__(self, selection='all', molid=0, frame=0):
    self.__molid = molid
    self.__text = selection
    self.__frame = frame
    self.__list = atomselection.create(molid, frame, selection)
  def select(self, selection):
    self.__list = atomselection.create(self.__molid, self.__frame,\
      selection)
    self.__text = selection
  def molid(self):
    return self.__molid
  def frame(self, value=-1):
    if value >= 0:
      self.__frame = value
      return self
    else:
      return self.__frame
  def list(self):
    """Return a copy of the selected atom indices."""
    return self.__list[:]
  def write(self, filename, filetype=None):
    """Write the atoms in the selection to filename.  Filetype is guessed
    from the filename, or can be specified with filetype."""
    if filetype is None:
      filetype = 'pdb'
      ind=filename.rfind('.')
      if ind >= 0:
        filetype = filename[ind+1:]
    if molecule.write(self.__molid, filetype, filename, self.__frame, \
        self.__frame, 1, -1, selection=self.__list) != 1:
      raise IOError, "Unable to write selection to file '%s'." % filename
    return self
  def get(self, *attrlist):
    result = [] 
    for attr in attrlist:
      result.append(atomselection.get( \
        self.__molid, self.__frame, self.__list, attr))
    if len(result) == 1:
      result = result[0]
    return result
  def set(self, attr, val):
    valtype = type(val)
    if valtype == type('') or valtype == type(1) or \
      valtype == type(1.0): 
      atomselection.set( self.__molid, self.__frame, \
        self.__list, attr, (val,))
    elif valtype == type([]) or valtype == type(()):
      atomselection.set(self.__molid, self.__frame, \
        self.__list, attr, tuple(val))
    else:
      raise AttributeError
  def getbonds(self):
    return atomselection.getbonds(self.__molid, self.__list)
  def setbonds(self, bonds):
    return atomselection.setbonds(self.__molid, self.__list, bonds)

  def minmax(self):
    return atomselection.minmax(self.__molid, self.__frame, self.__list);
  def center(self, weight=None):
    """Return the center of the selected atoms, possibly weighted by 
    weight."""
    return atomselection.center(self.__molid, self.__frame, self.__list, weight);
  def rmsd(self,sel=None, frame=None, weight=None):
    if sel is None:
      sel = self
    if frame is None:
      frame = sel.__frame
    return atomselection.rmsd(self.__molid, self.__frame, self.__list, sel.__molid, frame, sel.__list, weight)
  def align(self, ref=None, move=None, frame=None, weight=None):
    """
    align(ref=None, move=None, frame=None, weight=None): RMSD alignment
    Finds transformation to align atoms in selection with ref,
    using given weight, and applies the transformation to atoms
    in move.  If ref is not given, coordinates in frame 0 of the 
    selection are used.  If move is not given, all atoms in the
    molecule are moved; otherwise, only atoms in move are moved.
    frame can be used to apply the transformation to a different
    timestep than the current one.  Note that frame overrides the frame
    in both sel and move.  If you want to use different frames for 
    sel and move, use sel.frame(i).align(move.frame(j))
    """

    if frame is None:
      haveFrame = 0
      frame = self.__frame
    else:
      haveFrame = 1

    if ref is None:
      rmol=self.__molid
      rframe=0
      ratms = self.__list
    else:
      rmol=ref.__molid
      rframe=ref.__frame
      ratms=ref.__list

    if move is None:
      # signal that all atoms are to be used by passing
      # molid=-1.
      mmol, mframe, matms = -1, frame, () 
    else:
      if haveFrame:
        mframe = frame
      else:
        mframe = move.__frame
      mmol, matms = move.__molid, move.__list
    atomselection.align(self.__molid, frame, self.__list, \
      rmol, rframe, ratms, \
      mmol, mframe, matms, weight)
    return self

  def contacts(self, cutoff, sel = None):
    if sel is None:
      sel = self
    return atomselection.contacts(self.__molid, self.__frame, self.__list, \
                                  sel.__molid, sel.__frame, sel.__list, \
                                  float(cutoff))
  def sasa(self, srad, **kwds):
    """
    sasa(srad, samples=-1, points=None, restrict=None)
    return solvent-accessible surface area of selection using srad for
    solvent radius.  Optional arguments: samples = number of samples around
    each atom (determines accuracy); pass a list as points argument to get
    the coordinates of the solvent accessible points; pass a selection
    as restrict to include only points in restrict in the result.
    """
    res=kwds.get("restrict",None)
    if res is not None:
      try:
        kwds["restrict"] = res.__list
      except:
        raise ValueError, "'restrict' argument must be an AtomSel instance."
    return atomselection.sasa(float(srad), self.__molid, self.__frame, \
      self.__list, **kwds)

  def __and__(self, other):
    newseltext = '(' + self.__text + ') and (' + other.__text + ')'
    return AtomSel(newseltext, self.__molid, self.__frame)
  def __or__(self, other):
    newseltext = '(' + self.__text + ') or (' + other.__text + ')'
    return AtomSel(newseltext, self.__molid, self.__frame)
  def __neg__(self):
    newseltext = 'not (' + self.__text + ')'
    return AtomSel(newseltext, self.__molid, self.__frame)

  def __str__(self):
    return self.__text 
  def __len__(self):
    return len(self.__list)
  def __getitem__(self, key):
    return self.__list[key]
  def __getslice__(self, low, high):
    return self.__list[low:high]
  def __array__(self, c):
    return self.__list

