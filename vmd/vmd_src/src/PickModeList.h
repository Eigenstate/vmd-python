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
 *      $RCSfile: PickModeList.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.13 $       $Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  List of all available pick modes
 ***************************************************************************/

#ifndef PICK_MODE_LIST_H
#define PICK_MODE_LIST_H

#include "NameList.h"

class PickMode;
class VMDApp;

/// list of all available PickMode subclasses
class PickModeList {
public:
  // these enum constants must match the order in which the modes are loaded
  // in PickModeList::PickModeList(VMDApp *)
  enum {QUERY=0, CENTER, LABELATOM, LABELBOND, LABELANGLE, LABELDIHEDRAL, MOVEATOM, MOVERES, MOVEFRAG, MOVEMOL, FORCEATOM, FORCERES, FORCEFRAG, MOVEREP, ADDBOND, PICK};
        
private:
  NameList<PickMode *>pickmodelist;
  PickMode *curpickmode;

public:
  PickModeList(VMDApp *);
  ~PickModeList();

  PickMode *current_pick_mode() { return curpickmode; }
  int set_pick_mode(int mode) {
    if (mode < 0 || mode >= pickmodelist.num()) return FALSE;
    curpickmode = pickmodelist.data(mode);
    return TRUE;
  }
};
  
#endif
