/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2019 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

#ifndef PICKMODEUSER_H
#define PICKMODEUSER_H

#include "PickMode.h"
class Molecule;
class DisplayDevice;
class VMDApp;

/// PickMode subclass to add bonds between two atoms
class PickModeUser: public PickMode {
private:
  float pPos[3]; ///< pointer coords when this started

  /// items we need to have, and have collected so far, and whether we
  /// need to actually print the info out 
  int haveItems, needName;

  int molid; ///< selected molecule IDs
  int atmid; ///< selected atom IDs
  int atom;  ///< last atom picked at the start

protected:
  VMDApp *app;

public:
  PickModeUser(VMDApp *);
  virtual void pick_molecule_start(DrawMolecule *, DisplayDevice *,
  				int, int, const int *cell, int, const float *);
  virtual void pick_molecule_end  (DrawMolecule *, DisplayDevice *);
};

#endif

