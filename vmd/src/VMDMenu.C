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
 *      $RCSfile: VMDMenu.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.9 $       $Date: 2010/12/16 04:08:46 $
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

