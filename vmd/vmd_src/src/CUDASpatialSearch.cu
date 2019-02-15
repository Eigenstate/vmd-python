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
 *      $RCSfile: CUDASpatialSearch.cu,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.8 $      $Date: 2019/01/17 21:38:55 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   CUDA accelerated functions for building spatial searching, sorting, 
 *   and hashing data structures, used by QuickSurf, MDFF, and other routines.
 *
 ***************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

#if CUDART_VERSION < 4000
#error The VMD MDFF feature requires CUDA 4.0 or later
#endif

#include "Inform.h"
#include "utilities.h"
#include "WKFThreads.h"
#include "WKFUtils.h"
#include "CUDAKernels.h" 
#include "CUDASpatialSearch.h"
#include "CUDASort.h"

#include "AtomSel.h"
#include "VMDApp.h"
#include "MoleculeList.h"
#include "VolumetricData.h"
#include "VolMapCreate.h" // volmap_write_dx_file()

#include <tcl.h>
#include "TclCommands.h"

#if 1
#define CUERR { cudaError_t err; \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
  printf("CUDA error: %s, %s line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
  printf("Thread aborting...\n"); \
  return NULL; }}
#else
#define CUERR
#endif


//
// Restrict macro to make it easy to do perf tuning tests
//
#if 0
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif

#if __CUDA_ARCH__ >= 300
#define MAXTHRSORTHASH 512
#define MINBLOCKSORTHASH 4
#elif __CUDA_ARCH__ >= 200
#define MAXTHRSORTHASH 512
#define MINBLOCKSORTHASH 1
#else
#define MAXTHRSORTHASH 512
#define MINBLOCKSORTHASH 1
#endif

//
// Linear-time density kernels that use spatial hashing of atoms 
// into a uniform grid of atom bins to reduce the number of 
// density computations by truncating the gaussian to a given radius
// and only considering bins of atoms that fall within that radius.
//

#define GRID_CELL_EMPTY 0xffffffff

// calculate cell address as the hash value for each atom
__global__ static void 
// __launch_bounds__ ( MAXTHRSORTHASH, MINBLOCKSORTHASH )
hashAtoms(unsigned int natoms,
          const float4 * RESTRICT xyzr,
          int3 numcells,
          float invgridspacing,
          unsigned int * RESTRICT atomIndex,
          unsigned int * RESTRICT atomHash) {
  unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= natoms)
    return;

  float4 atom = xyzr[index]; // read atom coordinate and radius

  // compute cell index, clamped to fall within grid bounds
  int3 cell;
  cell.x = min(int(atom.x * invgridspacing), numcells.x-1);
  cell.y = min(int(atom.y * invgridspacing), numcells.y-1);
  cell.z = min(int(atom.z * invgridspacing), numcells.z-1);

  unsigned int hash = (cell.z * numcells.y * numcells.x) + 
                      (cell.y * numcells.x) + cell.x;

  atomIndex[index] = index; // original atom index
  atomHash[index] = hash;   // atoms hashed to cell address
}


// build cell lists and reorder atoms and colors using sorted atom index list
__global__ static void
// __launch_bounds__ ( MAXTHRSORTHASH, MINBLOCKSORTHASH )
sortAtomsColorsGenCellLists(unsigned int natoms,
                            const float4 * RESTRICT xyzr_d,
                            const float4 * RESTRICT color_d,
                            const unsigned int *atomIndex_d,
                            const unsigned int *atomHash_d,
                            float4 * RESTRICT sorted_xyzr_d,
                            float4 * RESTRICT sorted_color_d,
                            uint2 * RESTRICT cellStartEnd_d) {
  extern __shared__ unsigned int hash_s[]; // blockSize + 1 elements
  unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned int hash;

  if (index < natoms) {
    hash = atomHash_d[index];
    hash_s[threadIdx.x+1] = hash; // use smem to avoid redundant loads
    if (index > 0 && threadIdx.x == 0) {
      // first thread in block must load neighbor particle hash
      hash_s[0] = atomHash_d[index-1];
    }
  }

  __syncthreads();

  if (index < natoms) {
    // Since atoms are sorted, if this atom has a different cell
    // than its predecessor, it is the first atom in its cell, and
    // it's index marks the end of the previous cell.
    if (index == 0 || hash != hash_s[threadIdx.x]) {
      cellStartEnd_d[hash].x = index; // set start
      if (index > 0)
        cellStartEnd_d[hash_s[threadIdx.x]].y = index; // set end
    }

    if (index == natoms - 1) {
      cellStartEnd_d[hash].y = index + 1; // set end
    }

    // Reorder atoms according to sorted indices
    unsigned int sortedIndex = atomIndex_d[index];
    float4 pos = xyzr_d[sortedIndex];
    sorted_xyzr_d[index] = pos;

    // Reorder colors according to sorted indices, if provided
    if (color_d != NULL) {
      float4 col = color_d[sortedIndex];
      sorted_color_d[index] = col;
    }
  }
}



// build cell lists and reorder atoms using sorted atom index list
__global__ static void
// __launch_bounds__ ( MAXTHRSORTHASH, MINBLOCKSORTHASH )
sortAtomsGenCellLists(unsigned int natoms,
                      const float4 * RESTRICT xyzr_d,
                      const unsigned int *atomIndex_d,
                      const unsigned int *atomHash_d,
                      float4 * RESTRICT sorted_xyzr_d,
                      uint2 * RESTRICT cellStartEnd_d) {
  extern __shared__ unsigned int hash_s[]; // blockSize + 1 elements
  unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned int hash;

  if (index < natoms) {
    hash = atomHash_d[index];
    hash_s[threadIdx.x+1] = hash; // use smem to avoid redundant loads
    if (index > 0 && threadIdx.x == 0) {
      // first thread in block must load neighbor particle hash
      hash_s[0] = atomHash_d[index-1];
    }
  }

  __syncthreads();

  if (index < natoms) {
    // Since atoms are sorted, if this atom has a different cell
    // than its predecessor, it is the first atom in its cell, and
    // it's index marks the end of the previous cell.
    if (index == 0 || hash != hash_s[threadIdx.x]) {
      cellStartEnd_d[hash].x = index; // set start
      if (index > 0)
        cellStartEnd_d[hash_s[threadIdx.x]].y = index; // set end
    }

    if (index == natoms - 1) {
      cellStartEnd_d[hash].y = index + 1; // set end
    }

    // Reorder atoms according to sorted indices
    unsigned int sortedIndex = atomIndex_d[index];
    float4 pos = xyzr_d[sortedIndex];
    sorted_xyzr_d[index] = pos;
  }
}



int vmd_cuda_build_density_atom_grid(int natoms,
                                     const float4 * xyzr_d,
                                     const float4 * color_d,
                                     float4 * sorted_xyzr_d,
                                     float4 * sorted_color_d,
                                     unsigned int *atomIndex_d,
                                     unsigned int *atomHash_d,
                                     uint2 * cellStartEnd_d,
                                     int3 volsz,
                                     float invgridspacing) {

  // Compute block and grid sizes to use for various kernels
  dim3 hBsz(256, 1, 1);

  // If we have a very large atom count, we must either use
  // larger thread blocks, or use multi-dimensional grids of thread blocks.
  // We can use up to 65535 blocks in a 1-D grid, so we can use
  // 256-thread blocks for less than 16776960 atoms, and use 512-thread
  // blocks for up to 33553920 atoms.  Beyond that, we have to use 2-D grids
  // and modified kernels.
  if (natoms > 16000000)
    hBsz.x = 512; // this will get us

  dim3 hGsz(((natoms+hBsz.x-1) / hBsz.x), 1, 1);

  // Compute grid cell address as atom hash
  // XXX need to use 2-D indexing for large atom counts or we exceed the
  //     per-dimension 65535 block grid size limitation
  hashAtoms<<<hGsz, hBsz>>>(natoms, xyzr_d, volsz, invgridspacing,
                            atomIndex_d, atomHash_d);

  // Sort atom indices by their grid cell address
  // XXX no pre-allocated workspace yet...
  if (dev_radix_sort_by_key(atomHash_d, atomIndex_d, natoms, 
                            (unsigned int *) NULL, (unsigned int *) NULL,
                            NULL, 0, 0U, 0U) != 0) {
    // It is common to encounter thrust memory allocation issues, so
    // we have to catch thrown exceptions here, otherwise we're guaranteed
    // to eventually have a crash.  If we get a failure, we have to bomb
    // out entirely and fall back to the CPU.
    printf("dev_radix_sort_by_key() failed: %s line %d\n", __FILE__, __LINE__);
    return -1;
  }

  // Initialize all cells to empty
  int ncells = volsz.x * volsz.y * volsz.z;
  cudaMemset(cellStartEnd_d, GRID_CELL_EMPTY, ncells*sizeof(uint2));

  // Reorder atoms into sorted order and find start and end of each cell
  // XXX need to use 2-D indexing for large atom counts or we exceed the
  //     per-dimension 65535 block grid size limitation
  unsigned int smemSize = sizeof(unsigned int)*(hBsz.x+1);
  sortAtomsColorsGenCellLists<<<hGsz, hBsz, smemSize>>>(
                       natoms, xyzr_d, color_d, atomIndex_d, atomHash_d,
                       sorted_xyzr_d, sorted_color_d, cellStartEnd_d);

  // To gain a bit more performance we can disable detailed error checking
  // and use an all-or-nothing approach where errors fall through all
  // CUDA API calls until the end, and we only do cleanup at the end.
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s, %s line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    return -1;
  }

  return 0;
}



int vmd_cuda_build_density_atom_grid(int natoms,
                                     const float4 * xyzr_d,
                                     float4 *& sorted_xyzr_d,
                                     uint2 *& cellStartEnd_d,
                                     int3 volsz,
                                     float invgridspacing) {
  // Allocate work buffers and output array for atom hashing and sorting
  unsigned int *atomIndex_d = NULL;
  unsigned int *atomHash_d = NULL;

  cudaMalloc((void**)&sorted_xyzr_d, natoms * sizeof(float4));
  cudaMalloc((void**)&atomIndex_d, natoms * sizeof(unsigned int));
  cudaMalloc((void**)&atomHash_d, natoms * sizeof(unsigned int));

  // Allocate arrays of cell atom list starting and ending addresses
  int ncells = volsz.x * volsz.y * volsz.z;
  cudaMalloc((void**)&cellStartEnd_d, ncells * sizeof(uint2));

  // Compute block and grid sizes to use for various kernels
  dim3 hBsz(256, 1, 1);
  dim3 hGsz(((natoms+hBsz.x-1) / hBsz.x), 1, 1);

  // Compute grid cell address as atom hash
  hashAtoms<<<hGsz, hBsz>>>(natoms, xyzr_d, volsz, invgridspacing,
                            atomIndex_d, atomHash_d);

  // Sort atom indices by their grid cell address
  // XXX no pre-allocated workspace yet...
  if (dev_radix_sort_by_key(atomHash_d, atomIndex_d, natoms,
                            (unsigned int *) NULL, (unsigned int *) NULL,
                            NULL, 0, 0U, 0U) != 0) {
    // It is common to encounter thrust memory allocation issues, so
    // we have to catch thrown exceptions here, otherwise we're guaranteed
    // to eventually have a crash.  If we get a failure, we have to bomb
    // out entirely and fall back to the CPU.
    printf("dev_radix_sort_by_key() failed: %s line %d\n", __FILE__, __LINE__);

    // free memory allocations 
    cudaFree(sorted_xyzr_d);
    cudaFree(atomIndex_d);
    cudaFree(atomHash_d);
    cudaFree(cellStartEnd_d);

    return -1;
  }

  // Initialize all cells to empty
  cudaMemset(cellStartEnd_d, GRID_CELL_EMPTY, ncells*sizeof(uint2));

  // Reorder atoms into sorted order and find start and end of each cell
  // XXX need to use 2-D indexing for large atom counts or we exceed the
  //     per-dimension 65535 block grid size limitation
  unsigned int smemSize = sizeof(unsigned int)*(hBsz.x+1);
  sortAtomsGenCellLists<<<hGsz, hBsz, smemSize>>>(
                       natoms, xyzr_d, atomIndex_d, atomHash_d, 
                       sorted_xyzr_d, cellStartEnd_d);

  // To gain a bit more performance we can disable detailed error checking
  // and use an all-or-nothing approach where errors fall through all
  // CUDA API calls until the end, and we only do cleanup at the end.
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s, %s line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    return -1;
  }

  // Free temporary working buffers, caller must free the rest
  cudaFree(atomIndex_d);
  cudaFree(atomHash_d);

  return 0;
}



