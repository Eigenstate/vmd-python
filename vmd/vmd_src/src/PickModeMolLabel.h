/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2019 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: PickModeMolLabel.h,v $
 *      $Author: johns $        $Locker:  $                $State: Exp $
 *      $Revision: 1.28 $      $Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * The PickMode object which allows a pointer to be used to create new 
 * geometrical monitoring labels.  This particular version is used only for
 * adding molecular labels, i.e. Atoms, Bonds, Angles, and Dihedrals.  As
 * more atoms are selected, they are remembered until enough have been picked
 * to create the relevant label (i.e. 3 atoms to create an Angle label).
 *
 ***************************************************************************/
#ifndef PICKMODEMOLLABEL_H
#define PICKMODEMOLLABEL_H

#include "PickMode.h"
class Molecule;
class DisplayDevice;
class VMDApp;

/// PickMode subclass to add labels to atoms
class PickModeMolLabel : public PickMode {
private:
  float pPos[3];   ///< pointer coords when this started
  int needItems;   ///< items we need to have
  int haveItems;   ///< items we've collected so far
  int needName;    ///< whether we need to print the info out
  int *molids;     ///< molecule IDs for picked atoms so far
  int *atmids;     ///< atom IDs for picked atoms so far
  int *cells;      ///< unit cell IDs for picked atoms so far
  int atom;        ///< last atom picked at the start
  int lastCell[3]; ///< last unit cell picked
  char *modename;  ///< name of this pick mode

protected:
  VMDApp *app;

  /// constructor: name, how many items this object needs, and a VMDApp
  PickModeMolLabel(const char *, int, VMDApp *);

public:
  virtual ~PickModeMolLabel(void);
  
  virtual void pick_molecule_start(DrawMolecule *, DisplayDevice *,
  				int, int, const int *cell, int, const float *);
  virtual void pick_molecule_move (DrawMolecule *, DisplayDevice *,
  				int, int, const float *);
  virtual void pick_molecule_end  (DrawMolecule *, DisplayDevice *);

  virtual void pick_graphics(int molid, int tag, int btn, DisplayDevice *d);
};

/// PickMode subclass for labeling atoms
class PickModeAtoms : public PickModeMolLabel {
public:
  PickModeAtoms(VMDApp *vmdapp) : PickModeMolLabel("Atoms",1,vmdapp) {}
};

/// PickMode subclass for bond/distance labels
class PickModeBonds : public PickModeMolLabel {
public:
  PickModeBonds(VMDApp *vmdapp) : PickModeMolLabel("Bonds",2,vmdapp) {}
};

/// PickMode subclass for angle labels
class PickModeAngles : public PickModeMolLabel {
public:
  PickModeAngles(VMDApp *vmdapp) : PickModeMolLabel("Angles",3,vmdapp) {}
};

/// PickMode subclass for dihedral labels
class PickModeDihedrals : public PickModeMolLabel {
public:
  PickModeDihedrals(VMDApp *vmdapp) : PickModeMolLabel("Dihedrals",4,vmdapp) {}
};

#endif

