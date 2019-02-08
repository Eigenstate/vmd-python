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
 *	$RCSfile: Residue.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.36 $	$Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  This is based on the uniq_resid assigned in BaseMolecule
 * A residue knows the other residues to which it is connected.  It also
 * has a list of all the atoms contained by this residue
 *
 ***************************************************************************/
#ifndef RESIDUE_H
#define RESIDUE_H

#include "Atom.h" // to get enums from Atom
#include "ResizeArray.h"

// these are secondary structure definitions from Stride
#define SS_HELIX_ALPHA 0
#define SS_HELIX_3_10  1
#define SS_HELIX_PI    2
#define SS_BETA        3
#define SS_BRIDGE      4
#define SS_TURN        5
#define SS_COIL        6

/// Based on the uniq_resid assigned in BaseMolecule, a residue knows 
/// what other residues it is connected to, and maintains a list of
/// the atoms it contains
class Residue {
  public:
    int resid;                    ///< non-unique resid from the original file
    signed char residueType;      ///< residue type code
    signed char sstruct;          ///< secondary structure for this residue

    int fragment;                 ///< a fragment is a set of connect residues
                                  ///<  which are not connected to any other
                                  ///<  residues
    ResizeArray<int> atoms;       ///< list of atoms in this residue

    Residue(int realid, int newtype) : atoms(3) {
      resid = realid; // non-unique resid from file...
      residueType = newtype;
      fragment = -1;
      sstruct = SS_COIL;
    }

    void add_atom(int atomindex) {
      atoms.append(atomindex);
    };
};

#endif

