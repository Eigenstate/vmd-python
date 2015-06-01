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
 *	$RCSfile: CUDAOrbital.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.1 $	$Date: 2013/03/12 13:27:17 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   CUDA-accelerated molecular orbital representation
 ***************************************************************************/
#ifndef CUDAORBITAL_H
#define CUDAORBITAL_H

int vmd_cuda_evaluate_orbital_grid(wkf_threadpool_t *devpool,
                       int numatoms,
                       const float *wave_f, int num_wave_f,
                       const float *basis_array, int num_basis,
                       const float *atompos,
                       const int *atom_basis,
                       const int *num_shells_per_atom,
                       const int *num_prim_per_shell,
                       const int *shell_types,
                       int num_shells,
                       const int *numvoxels,
                       float voxelsize,
                       const float *origin,
                       int density,
                       float *orbitalgrid);

#endif
