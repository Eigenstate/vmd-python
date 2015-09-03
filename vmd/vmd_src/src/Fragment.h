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
 *	$RCSfile: Fragment.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.18 $	$Date: 2010/12/16 04:08:16 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * A Fragment contains a list of residues which are connected
 * each other, and to no one else.  This is at the residue
 * level, and not the atom level.  The residue numbers are the
 * unique_resid as assigned in BaseMolecule
 *
 ***************************************************************************/
#ifndef FRAGMENT_H
#define FRAGMENT_H

#include "ResizeArray.h"

/// A Fragment contains a list of residues which are connected
/// each other, and to no one else.  This is at the residue
/// level, and not the atom level.  The residue numbers are the
/// unique_resid as assigned in BaseMolecule
class Fragment {
public:
  ResizeArray<int> residues;
  
  Fragment(void) : residues(1) {
  }
  
  int num(void) { return residues.num(); }
  int operator [](int i) { return residues[i]; }
  void append(int i) { residues.append(i); }
};

#endif

