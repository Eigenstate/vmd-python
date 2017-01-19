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
 *      $RCSfile: VolCPotential.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.6 $      $Date: 2016/11/28 03:05:06 $
 *
 ***************************************************************************/
#ifndef VOLCPOTENTIAL_H
#define VOLCPOTENTIAL_H

int vol_cpotential(long int natoms, float* atoms, float* grideners, long int numplane, long int numcol, long int numpt, float gridspacing); 

#endif
