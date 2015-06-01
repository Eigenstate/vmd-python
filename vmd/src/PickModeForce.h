/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2011 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: PickModeForce.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.14 $       $Date: 2010/12/16 04:08:35 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Pick mode for applying forces to atoms.
 ***************************************************************************/

#ifndef PICK_MODE_FORCE_H__
#define PICK_MODE_FORCE_H__

#include "PickMode.h"
struct MolAtom;

/// PickMode subclass for applying forces to atoms
class PickModeForce : public PickMode {
protected:
  int mytag;

  PickModeForce() {}
  void get_force(const float *atomPos, const float *pos,
                              DrawMolecule *m, int dim, DisplayDevice *d,
                              float *newforce);
  void set_force(MolAtom *, const float *);
};

// The cool way to do this would be to define an iterator in each subclass,
// but alas, I'm not cool enough at the moment.

/// PickMode subclass to apply a force to a single atom
class PickModeForceAtom : public PickModeForce {
public:
  PickModeForceAtom() {} 
  virtual void pick_molecule_start(DrawMolecule *, DisplayDevice *,
                             int btn, int tag, const int *cell,
                             int dim, const float *pos);
  virtual void pick_molecule_move (DrawMolecule *, DisplayDevice *,
                             int tag, int dim, const float *pos);
};

/// PickMode subclass to apply a force to a single residue
class PickModeForceResidue : public PickModeForce {
public:
  PickModeForceResidue() {}
  virtual void pick_molecule_start(DrawMolecule *, DisplayDevice *,
                             int btn, int tag, const int *cell,
                             int dim, const float *pos);
  virtual void pick_molecule_move (DrawMolecule *, DisplayDevice *,
                             int tag, int dim, const float *pos);
};

/// PickMode subclass to apply a force to a single fragment
class PickModeForceFragment : public PickModeForce {
public:
  PickModeForceFragment() {}
  virtual void pick_molecule_start(DrawMolecule *, DisplayDevice *,
                             int btn, int tag, const int *cell,
                             int dim, const float *pos);
  virtual void pick_molecule_move (DrawMolecule *, DisplayDevice *,
                             int tag, int dim, const float *pos);
};

#endif

