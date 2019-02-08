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
 *      $RCSfile: CUDASpatialSearch.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.2 $      $Date: 2019/01/17 21:38:55 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   CUDA accelerated functions for building spatial searching, sorting, 
 *   and hashing data structures, used by QuickSurf, MDFF, and other routines.
 *
 ***************************************************************************/

#define GRID_CELL_EMPTY 0xffffffff

int vmd_cuda_build_density_atom_grid(int natoms,
                                     const float4 * xyzr_d,
                                     const float4 * color_d,
                                     float4 * sorted_xyzr_d,
                                     float4 * sorted_color_d,
                                     unsigned int *atomIndex_d,
                                     unsigned int *atomHash_d,
                                     uint2 * cellStartEnd_d,
                                     int3 volsz,
                                     float invgridspacing);


int vmd_cuda_build_density_atom_grid(int natoms,
                                     const float4 * xyzr_d,
                                     float4 *& sorted_xyzr_d,
                                     uint2 *& cellStartEnd_d,
                                     int3 volsz,
                                     float invgridspacing);


