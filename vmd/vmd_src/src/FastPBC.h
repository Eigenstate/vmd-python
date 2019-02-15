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
 *      $RCSfile: FastPBC.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.2 $       $Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Code to measure atom distances, angles, dihedrals, etc.
 ***************************************************************************/

#include "Molecule.h"
#include "AtomSel.h"

void fpbc_exec_join(Molecule* mol, int first, int last, int fnum, 
                    int *compoundmap, int sellen, int* indexlist);

void fpbc_exec_join_cpu(Molecule* mol, int first, int last, int fnum, 
                        int *compoundmap, int sellen, int* indexlist);

void fpbc_exec_wrapcompound(Molecule* mol, int first, int last, int fnum, 
                            int *compoundmap, int sellen, 
                            int* indexlist, float* weights, 
                            AtomSel* csel, float* center, float * massarr);

void fpbc_exec_wrapcompound_cpu(Molecule* mol, int first, int last, int fnum, 
                                int *compoundmap, int sellen, 
                                int* indexlist, float* weights, 
                                AtomSel* csel, float* center, float * massarr);

void fpbc_exec_wrapatomic(Molecule* mol, int first, int last, 
                          int sellen, int* indexlist, float* weights, 
                          AtomSel* csel, float* center);

void fpbc_exec_wrapatomic_cpu(Molecule* mol, int first, int last, int sellen, 
                              int* indexlist, float* weights, 
                              AtomSel* csel, float* center);

void fpbc_exec_unwrap(Molecule* mol, int first, int last, 
                      int sellen, int* indexlist);

void fpbc_exec_unwrap_cpu(Molecule* mol, int first, int last, 
                          int sellen, int* indexlist);

void fpbc_exec_recenter_cpu(Molecule* mol, int first, int last, 
                            int csellen, int* cindexlist, int fnum, 
                            int *compoundmap, int sellen, 
                            int* indexlist, float* weights, 
                            AtomSel* csel, float* massarr);

void fpbc_exec_recenter(Molecule* mol, int first, int last, 
                        int csellen, int* cindexlist, 
                        int fnum, int *compoundmap, int sellen, 
                        int *indexlist, float* weights, 
                        AtomSel* csel, float* massarr);
