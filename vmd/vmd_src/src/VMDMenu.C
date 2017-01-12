/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2016 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: VMDMenu.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.10 $       $Date: 2016/11/28 03:05:05 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  VMD menu base class.
 ***************************************************************************/

#include "VMDMenu.h"
#include <utilities.h>

VMDMenu::VMDMenu(const char *menuname, VMDApp *vmdapp)
: UIObject(vmdapp) {
  
  name = stringdup(menuname);
}

VMDMenu::~VMDMenu() {
  delete [] name;
}

