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
 *      $RCSfile: MDFF.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.2 $      $Date: 2019/01/17 21:38:55 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Multi-core CPU versions of MDFF functions
 *
 * "GPU-Accelerated Analysis and Visualization of Large Structures
 *  Solved by Molecular Dynamics Flexible Fitting"
 *  John E. Stone, Ryan McGreevy, Barry Isralewitz, and Klaus Schulten.
 *  Faraday Discussion 169, 2014. (In press)
 *  Online full text available at http://dx.doi.org/10.1039/C4FD00005F
 *
 ***************************************************************************/

int cc_threaded(VolumetricData *qsVol, 
                const VolumetricData *targetVol, 
                double *cc, double threshold);

