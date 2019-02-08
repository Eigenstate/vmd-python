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
 *	$RCSfile: BondSearch.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.27 $	$Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Distance based bond search code 
 *
 ***************************************************************************/

#ifndef BONDSEARCH_H__
#define BONDSEARCH_H__

#include "ResizeArray.h"
#include "SpatialSearch.h"

class Timestep;
class BaseMolecule;

/// Grid search for the case of a single set of atoms. It ignore pairs 
/// between atoms with identical coords.  The maxpairs parameter is 
/// set to -1 for no-limit pairlist calculation, or a maximum value otherwise.
/// This is the same code as gridsearch1(), but simplified and hopefully
/// a bit faster.
GridSearchPairlist *vmd_gridsearch_bonds(const float *pos, const float *radii,
                                         int n, float dist, int maxpairs);

/// Compute bonds for the molecule using the given timestep (which must
/// not be NULL) and adds them to the given molecule.  Return success.
/// The code currently calls gridsearch1 with a pairlist limit of 
/// 27 * natoms, which should easily be sufficient for any real structure.
int vmd_bond_search(BaseMolecule *mol, const Timestep *ts, 
                    float cutoff, int dupcheck);

/// Multithreaded bond search worker routine handles spawning and joining all
/// of the worker threads, and merging their results into a single list.
int vmd_bondsearch_thr(const float *pos, const float *radii,
                       GridSearchPairlist * cur, int totb, 
                       int **boxatom, int *numinbox, int **nbrlist, 
                       int maxpairs, float pairdist);



#endif

