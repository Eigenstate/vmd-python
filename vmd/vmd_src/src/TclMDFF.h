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
 *      $RCSfile: TclMDFF.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.2 $      $Date: 2019/01/17 21:38:55 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Tcl bindings for MDFF functions
 *
 ***************************************************************************/

#ifndef TCLMDFF_H
#define TCLMDFF_H

#include "VMDApp.h"
#include <tcl.h>

int mdff_sim(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp);
int mdff_cc(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp);

#endif
