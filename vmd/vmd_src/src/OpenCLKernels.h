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
 *      $RCSfile: OpenCLKernels.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.8 $        $Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Wrapper for OpenCL kernels
 ***************************************************************************/
#ifndef OPENCLKERNELS_H
#define OPENCLKERNELS_H

#if defined(__APPLE__)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "WKFThreads.h"

int vmd_opencl_vol_cpotential(long int natoms, float* atoms, float* grideners,
                            long int numplane, long int numcol, long int numpt,
                            float gridspacing);

typedef struct {
  cl_context ctx;        // cached
  cl_command_queue cmdq; // cached
  cl_device_id *devs;    // cached
  cl_program pgm;
  cl_kernel kconstmem;
  cl_kernel kconstmem_vec4;
  cl_kernel ktiledshared;
} vmd_opencl_orbital_handle;

//
// Molecular orbital calculation routines
//
vmd_opencl_orbital_handle * vmd_opencl_create_orbital_handle(cl_context ctx, cl_command_queue cmdq, cl_device_id *devs);

int vmd_opencl_destroy_orbital_handle(vmd_opencl_orbital_handle * orbh);

int vmd_opencl_evaluate_orbital_grid(wkf_threadpool_t *devpool,
                       vmd_opencl_orbital_handle *orbh,
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

