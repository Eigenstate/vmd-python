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
 *	$RCSfile: SmallRing.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.9 $	$Date: 2010/12/16 04:08:39 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * A SmallRing contains an ordered list of atoms which are connected
 * to each other to form a loop.  The atom numbers are the
 * unique atom numbers as used in BaseMolecule. The ordering of
 * the atoms, in addition to specifying how the atoms in the ring are
 * connected, also gives the orientation (handedness) of the ring.
 *
 ***************************************************************************/
#ifndef SMALLRING_H
#define SMALLRING_H

#include "ResizeArray.h"
#include "Inform.h"

/// A SmallRing contains a list of atoms which are connected
/// to each other to form a loop.  The atom numbers are the
/// unique atom numbers as used in BaseMolecule. The ordering of
/// the atoms, in addition to specifying how the atoms in the ring are
/// connected, also gives the orientation (handedness) of the ring
/// if orientated is non-zero.
class SmallRing {
public:
  ResizeArray<int> atoms;
  short int orientated;
  
  SmallRing(void) : atoms(1), orientated(0) {}
  
  int num(void) { return atoms.num(); }
  int operator [](int i) { return atoms[i]; }
  void append(int i) { atoms.append(i); }

  int last_atom(void) { return atoms[atoms.num()-1]; }
  int first_atom(void) { return atoms[0]; }
  int closed(void) { return first_atom() == last_atom(); }
  void remove_last(void) { atoms.truncatelastn(1); }
 
  void reverse(void) {
    ResizeArray<int> atomscopy(atoms.num());
    int i, len;
    len = atoms.num();
    
    for (i=0;i<len;i++) atomscopy.append(atoms[i]);
    atoms.clear();
    for (i=len-1;i>=0;i--) atoms.append(atomscopy[i]);
  }

  void clear(void) {
    atoms.clear();
    orientated = 0;
  }
  
  SmallRing* copy(void) {
    SmallRing *ringcopy;
    int i, len;
    
    ringcopy = new SmallRing();
    len = num();
    for (i=0; i < len; i++) ringcopy->append(atoms[i]);
    ringcopy->orientated = orientated;
    
    return ringcopy;
  }
  
  friend Inform& operator << (Inform &os, SmallRing &sr) {
    int i, len;
    len = sr.num();
    
    for (i=0; i < len; i++) {
        os << sr[i];
        if (i == len-1) break;
        os << ", ";
    }
    
    return os;
  }
  
};

#endif
