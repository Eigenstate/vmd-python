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
 *      $RCSfile: VMDMPI.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.7 $      $Date: 2016/11/28 03:05:05 $
 *
 ***************************************************************************/
#ifndef VMDMPI_INC
#define VMDMPI_INC 1

int vmd_mpi_init(int *argc, char ***argv);
int vmd_mpi_barrier();
int vmd_mpi_fini();
int vmd_mpi_nodescan(int *noderank, int *nodecount,
                     char *nodename, int maxnodenamelen,
                     int gpucount);

#endif
