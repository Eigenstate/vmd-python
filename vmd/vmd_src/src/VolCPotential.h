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
 *      $RCSfile: VolCPotential.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.7 $      $Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************/
#ifndef VOLCPOTENTIAL_H
#define VOLCPOTENTIAL_H

int vol_cpotential(long int natoms, float* atoms, float* grideners, long int numplane, long int numcol, long int numpt, float gridspacing); 

#endif
