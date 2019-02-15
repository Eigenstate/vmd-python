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
 *      $RCSfile: py_atomsel.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.27 $      $Date: 2018/03/20 05:04:30 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   New Python atom selection interface
 ***************************************************************************/

#include "py_commands.h"
#include "AtomSel.h"
#include "VMDApp.h"
#include "MoleculeList.h"
#include "SymbolTable.h"
#include "Measure.h"
#include "SpatialSearch.h"
