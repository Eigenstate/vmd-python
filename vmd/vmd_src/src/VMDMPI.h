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
 *      $RCSfile: VMDMPI.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.6 $      $Date: 2010/12/16 04:08:46 $
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
