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
 *	$RCSfile: PickModeCenter.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.21 $	$Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Pick on an atom to change its centering/global translation
 *
 ***************************************************************************/

#ifndef PICKMODECENTER_H
#define PICKMODECENTER_H

#include "PickMode.h"

class VMDApp;

/// PickMode subclass to define the molecule's centering/global translation
class PickModeCenter : public PickMode {
private:
  float pPos[3]; ///< pointer coords when this started
  int pCell[3];  ///< Which unit cell the pick came from
  int pAtom;     ///< atom number we start with
  int needName;  ///< need printout of name
  VMDApp *app;   ///< VMDApp pointer

public:
  PickModeCenter(VMDApp *);
  
  virtual void pick_molecule_start(DrawMolecule *, DisplayDevice *,
  				int, int, const int *, int, const float *);
  virtual void pick_molecule_move(DrawMolecule *, DisplayDevice *,
  				int, int, const float *);
  virtual void pick_molecule_end(DrawMolecule *, DisplayDevice *);
};
#endif

