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
 *      $RCSfile: OpenCLOrbital.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.32 $      $Date: 2019/01/17 21:38:55 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   This source file contains the OpenCL kernels for computing molecular
 *  orbital amplitudes on a uniformly spaced grid, using one ore more GPUs.
 *
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__APPLE__)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "WKFThreads.h"
#include "OpenCLKernels.h"

#if 1
#define CLERR \
  if (clerr != CL_SUCCESS) {                     \
    printf("opencl error %d, %s line %d\n", clerr, __FILE__, __LINE__); \
    return NULL;                                   \
  }
#else
#define CLERR
#endif

typedef union flint_t {
  float f;
  int i;
} flint;

typedef struct {
  int numatoms;
  const float *wave_f; 
  int num_wave_f;
  const float *basis_array;
  int num_basis;
  const float *atompos;
  const int *atom_basis;
  const int *num_shells_per_atom;
  const int *num_prim_per_shell;
  const int *shell_symmetry;
  int num_shells;
  const int *numvoxels;
  float voxelsize;
  const float *origin;
  int density;
  float *orbitalgrid;
  vmd_opencl_orbital_handle *orbh;
} orbthrparms;

/* thread prototype */
static void * openclorbitalthread(void *);

// GPU block layout
#define UNROLLX       1
#define UNROLLY       1
#define BLOCKSIZEX    8 
#define BLOCKSIZEY    8
#define BLOCKSIZE     BLOCKSIZEX * BLOCKSIZEY

// required GPU array padding to match thread block size
#define TILESIZEX BLOCKSIZEX*UNROLLX
#define TILESIZEY BLOCKSIZEY*UNROLLY
#define GPU_X_ALIGNMASK (TILESIZEX - 1)
#define GPU_Y_ALIGNMASK (TILESIZEY - 1)

#define V4UNROLLX       4
#define V4UNROLLY       1
#define V4BLOCKSIZEX    2 
#define V4BLOCKSIZEY    8
#define V4BLOCKSIZE     BLOCKSIZEX * BLOCKSIZEY

// required GPU array padding to match thread block size
#define V4TILESIZEX V4BLOCKSIZEX*V4UNROLLX
#define V4TILESIZEY V4BLOCKSIZEY*V4UNROLLY
#define V4GPU_X_ALIGNMASK (V4TILESIZEX - 1)
#define V4GPU_Y_ALIGNMASK (V4TILESIZEY - 1)

#define MEMCOALESCE  384

// orbital shell types 
#define S_SHELL 0
#define P_SHELL 1
#define D_SHELL 2
#define F_SHELL 3
#define G_SHELL 4
#define H_SHELL 5

//
// Constant arrays to store 
//
#define MAX_ATOM_SZ 256
#define MAX_ATOMPOS_SZ (MAX_ATOM_SZ)
#define MAX_ATOM_BASIS_SZ (MAX_ATOM_SZ)
#define MAX_ATOMSHELL_SZ (MAX_ATOM_SZ)
#define MAX_BASIS_SZ 6144 
#define MAX_SHELL_SZ 1024
#define MAX_WAVEF_SZ 6144


// Shared memory tiling parameters:
//   We need a tile size of at least 16 elements for coalesced memory access,
// and it must be the next power of two larger than the largest number of 
// wavefunction coefficients that will be consumed at once for a given 
// shell type.  The current code is designed to handle up to "G" shells 
// (15 coefficients), since that's the largest shell type supported by
// GAMESS, which is the only format we can presently read.

// The maximum shell coefficient count specifies the largest number
// of coefficients that might have to be read for a single shell,
// e.g. for the highest supported shell type.
// This is a "G" shell coefficient count
#define MAXSHELLCOUNT 15   

// Mask out the lower N bits of the array index
// to compute which tile to start loading shared memory with.
// This must be large enough to gaurantee coalesced addressing, but
// be half or less of the minimum tile size
#define MEMCOAMASK  (~15)

// The shared memory basis set and wavefunction coefficient arrays 
// must contain a minimum of at least two tiles, but can be any 
// larger multiple thereof which is also a multiple of the thread block size.
// Larger arrays reduce duplicative data loads/fragmentation 
// at the beginning and end of the array
#define SHAREDSIZE 256

//
// OpenCL kernel source code as a giant string
//
const char *clorbitalsrc = 
  "// unit conversion                                                      \n"
  "#define ANGS_TO_BOHR 1.8897259877218677f                                \n"
  "                                                                        \n"
  "// orbital shell types                                                  \n"
  "#define S_SHELL 0                                                       \n"
  "#define P_SHELL 1                                                       \n"
  "#define D_SHELL 2                                                       \n"
  "#define F_SHELL 3                                                       \n"
  "#define G_SHELL 4                                                       \n"
  "#define H_SHELL 5                                                       \n"
  "                                                                        \n"
  "//                                                                      \n"
  "// OpenCL using const memory for almost all of the key arrays           \n"
  "//                                                                      \n"
  "__kernel __attribute__((reqd_work_group_size(BLOCKSIZEX, BLOCKSIZEY, 1))) \n"
  "void clorbitalconstmem(int numatoms,                                    \n"
#if defined(__APPLE__)
  // Workaround for Apple OpenCL compiler bug related to constant memory
  //   Although the OpenCL full profile requires a minimum of 8 constant args
  //   of minimum size 64-kB each, Apple's compiler has problems with more 
  //   than 3, giving wrong answers for 4 or more.
  "                       __global float *const_atompos,                 \n"
  "                       __global int *const_atom_basis,                \n"
  "                       __global int *const_num_shells_per_atom,       \n"
  "                       __global float *const_basis_array,             \n"
  "                       __global int *const_num_prim_per_shell,        \n"
  "                       __global int *const_shell_symmetry,            \n"
  "                       __global float *const_wave_f,                  \n"
#else
  "                       __constant float *const_atompos,                 \n"
  "                       __constant int *const_atom_basis,                \n"
  "                       __constant int *const_num_shells_per_atom,       \n"
  "                       __constant float *const_basis_array,             \n"
  "                       __constant int *const_num_prim_per_shell,        \n"
  "                       __constant int *const_shell_symmetry,            \n"
  "                       __constant float *const_wave_f,                  \n"
#endif
  "                       float voxelsize,                                 \n"
  "                       float originx,                                   \n"
  "                       float originy,                                   \n"
  "                       float grid_z,                                    \n"
  "                       int density,                                     \n"
  "                       __global float * orbitalgrid) {                  \n"
  "  unsigned int xindex  = get_global_id(0);                              \n"
  "  unsigned int yindex  = get_global_id(1);                              \n"
  "  unsigned int outaddr = get_global_size(0) * yindex + xindex;          \n"
  "  float grid_x = originx + voxelsize * xindex;                          \n"
  "  float grid_y = originy + voxelsize * yindex;                          \n"
  "                                                                        \n"
  "  // similar to C version                                               \n"
  "  int at;                                                               \n"
  "  int prim, shell;                                                      \n"
  "                                                                        \n"
  "  // initialize value of orbital at gridpoint                           \n"
  "  float value = 0.0f;                                                   \n"
  "                                                                        \n"
  "  // initialize the wavefunction and shell counters                     \n"
  "  int ifunc = 0;                                                        \n"
  "  int shell_counter = 0;                                                \n"
  "  // loop over all the QM atoms                                         \n"
  "  for (at = 0; at < numatoms; at++) {                                   \n"
  "    // calculate distance between grid point and center of atom         \n"
  "    int maxshell = const_num_shells_per_atom[at];                       \n"
  "    int prim_counter = const_atom_basis[at];                            \n"
  "                                                                        \n"
  "    float xdist = (grid_x - const_atompos[3*at  ])*ANGS_TO_BOHR;        \n"
  "    float ydist = (grid_y - const_atompos[3*at+1])*ANGS_TO_BOHR;        \n"
  "    float zdist = (grid_z - const_atompos[3*at+2])*ANGS_TO_BOHR;        \n"
  "                                                                        \n"
  "    float xdist2 = xdist*xdist;                                         \n"
  "    float ydist2 = ydist*ydist;                                         \n"
  "    float zdist2 = zdist*zdist;                                         \n"
  "                                                                        \n"
  "    float dist2 = xdist2 + ydist2 + zdist2;                             \n"
  "                                                                        \n"
  "    // loop over the shells belonging to this atom (or basis function)  \n"
  "    for (shell=0; shell < maxshell; shell++) {                          \n"
  "      float contracted_gto = 0.0f;                                      \n"
  "                                                                        \n"
  "      // Loop over the Gaussian primitives of this contracted           \n"
  "      // basis function to build the atomic orbital                     \n"
  "      int maxprim = const_num_prim_per_shell[shell_counter];            \n"
  "      int shell_type = const_shell_symmetry[shell_counter];             \n"
  "      for (prim=0; prim < maxprim;  prim++) {                           \n"
  "        float exponent       = const_basis_array[prim_counter    ];     \n"
  "        float contract_coeff = const_basis_array[prim_counter + 1];     \n"
  "                                                                        \n"
  "        // By premultiplying the stored exponent factors etc,           \n"
  "        // we can use exp2f() rather than exp(), giving us full         \n"
  "        // precision, but with the speed of __expf()                    \n"
  "        contracted_gto += contract_coeff * native_exp2(-exponent*dist2);\n"
  "        prim_counter += 2;                                              \n"
  "      }                                                                 \n"
  "                                                                        \n"
  "      /* multiply with the appropriate wavefunction coefficient */      \n"
  "      float tmpshell=0.0f;                                              \n"
  "      switch (shell_type) {                                             \n"
  "        case S_SHELL:                                                   \n"
  "          value += const_wave_f[ifunc++] * contracted_gto;              \n"
  "          break;                                                        \n"
  "                                                                        \n"
  "        case P_SHELL:                                                   \n"
  "          tmpshell += const_wave_f[ifunc++] * xdist;                    \n"
  "          tmpshell += const_wave_f[ifunc++] * ydist;                    \n"
  "          tmpshell += const_wave_f[ifunc++] * zdist;                    \n"
  "          value += tmpshell * contracted_gto;                           \n"
  "          break;                                                        \n"
  "                                                                        \n"
  "        case D_SHELL:                                                   \n"
  "          tmpshell += const_wave_f[ifunc++] * xdist2;                   \n"
  "          tmpshell += const_wave_f[ifunc++] * xdist * ydist;            \n"
  "          tmpshell += const_wave_f[ifunc++] * ydist2;                   \n"
  "          tmpshell += const_wave_f[ifunc++] * xdist * zdist;            \n"
  "          tmpshell += const_wave_f[ifunc++] * ydist * zdist;            \n"
  "          tmpshell += const_wave_f[ifunc++] * zdist2;                   \n"
  "          value += tmpshell * contracted_gto;                           \n"
  "          break;                                                        \n"
  "                                                                        \n"
  "        case F_SHELL:                                                   \n"
  "          tmpshell += const_wave_f[ifunc++] * xdist2 * xdist;           \n"
  "          tmpshell += const_wave_f[ifunc++] * xdist2 * ydist;           \n"
  "          tmpshell += const_wave_f[ifunc++] * ydist2 * xdist;           \n"
  "          tmpshell += const_wave_f[ifunc++] * ydist2 * ydist;           \n"
  "          tmpshell += const_wave_f[ifunc++] * xdist2 * zdist;           \n"
  "          tmpshell += const_wave_f[ifunc++] * xdist * ydist * zdist;    \n"
  "          tmpshell += const_wave_f[ifunc++] * ydist2 * zdist;           \n"
  "          tmpshell += const_wave_f[ifunc++] * zdist2 * xdist;           \n"
  "          tmpshell += const_wave_f[ifunc++] * zdist2 * ydist;           \n"
  "          tmpshell += const_wave_f[ifunc++] * zdist2 * zdist;           \n"
  "          value += tmpshell * contracted_gto;                           \n"
  "          break;                                                        \n"
  "                                                                        \n"
  "        case G_SHELL:                                                   \n"
  "          tmpshell += const_wave_f[ifunc++] * xdist2 * xdist2;          \n"
  "          tmpshell += const_wave_f[ifunc++] * xdist2 * xdist * ydist;   \n"
  "          tmpshell += const_wave_f[ifunc++] * xdist2 * ydist2;          \n"
  "          tmpshell += const_wave_f[ifunc++] * ydist2 * ydist * xdist;   \n"
  "          tmpshell += const_wave_f[ifunc++] * ydist2 * ydist2;          \n"
  "          tmpshell += const_wave_f[ifunc++] * xdist2 * xdist * zdist;   \n"
  "          tmpshell += const_wave_f[ifunc++] * xdist2 * ydist * zdist;   \n"
  "          tmpshell += const_wave_f[ifunc++] * ydist2 * xdist * zdist;   \n"
  "          tmpshell += const_wave_f[ifunc++] * ydist2 * ydist * zdist;   \n"
  "          tmpshell += const_wave_f[ifunc++] * xdist2 * zdist2;          \n"
  "          tmpshell += const_wave_f[ifunc++] * zdist2 * xdist * ydist;   \n"
  "          tmpshell += const_wave_f[ifunc++] * ydist2 * zdist2;          \n"
  "          tmpshell += const_wave_f[ifunc++] * zdist2 * zdist * xdist;   \n"
  "          tmpshell += const_wave_f[ifunc++] * zdist2 * zdist * ydist;   \n"
  "          tmpshell += const_wave_f[ifunc++] * zdist2 * zdist2;          \n"
  "          value += tmpshell * contracted_gto;                           \n"
  "          break;                                                        \n"
  "      } // end switch                                                   \n"
  "                                                                        \n"
  "      shell_counter++;                                                  \n"
  "    }                                                                   \n"
  "  }                                                                     \n"
  "                                                                        \n"
  "  // return either orbital density or orbital wavefunction amplitude    \n"
  "  if (density) {                                                        \n"
  "    orbitalgrid[outaddr] = copysign(value*value, value);                \n"
  "  } else {                                                              \n"
  "    orbitalgrid[outaddr] = value;                                       \n"
  "  }                                                                     \n"
  "}                                                                       \n"
  "                                                                        \n"
  "                                                                        \n"
  "//                                                                      \n"
  "// OpenCL using const memory for almost all of the key arrays           \n"
  "//                                                                      \n"
  "__kernel __attribute__((reqd_work_group_size(V4BLOCKSIZEX, V4BLOCKSIZEY, 1))) \n"
  "void clorbitalconstmem_vec4(int numatoms,                               \n"
#if defined(__APPLE__)
  // Workaround for Apple OpenCL compiler bug related to constant memory
  //   Although the OpenCL full profile requires a minimum of 8 constant args
  //   of minimum size 64-kB each, Apple's compiler has problems with more 
  //   than 3, giving wrong answers for 4 or more.
  "                       __global float *const_atompos,                 \n"
  "                       __global int *const_atom_basis,                \n"
  "                       __global int *const_num_shells_per_atom,       \n"
  "                       __global float *const_basis_array,             \n"
  "                       __global int *const_num_prim_per_shell,        \n"
  "                       __global int *const_shell_symmetry,            \n"
  "                       __global float *const_wave_f,                  \n"
#else
  "                       __constant float *const_atompos,                 \n"
  "                       __constant int *const_atom_basis,                \n"
  "                       __constant int *const_num_shells_per_atom,       \n"
  "                       __constant float *const_basis_array,             \n"
  "                       __constant int *const_num_prim_per_shell,        \n"
  "                       __constant int *const_shell_symmetry,            \n"
  "                       __constant float *const_wave_f,                  \n"
#endif
  "                       float voxelsize,                                 \n"
  "                       float originx,                                   \n"
  "                       float originy,                                   \n"
  "                       float grid_z,                                    \n"
  "                       int density,                                     \n"
  "                       __global float * orbitalgrid) {                  \n"
  "  unsigned int xindex  = (get_global_id(0) - get_local_id(0)) * V4UNROLLX + get_local_id(0); \n"
  "  unsigned int yindex  = get_global_id(1);                              \n"
  "  unsigned int outaddr = get_global_size(0) * V4UNROLLX * yindex + xindex;\n"
  "  float4 gridspacing_v4 = { 0.f, 1.f, 2.f, 3.f };                       \n"
  "  gridspacing_v4 *= (voxelsize * V4BLOCKSIZEX);                         \n"
  "  float4 grid_x = originx + voxelsize * xindex + gridspacing_v4;        \n"
  "  float grid_y = originy + voxelsize * yindex;                          \n"
  "                                                                        \n"
  "  // similar to C version                                               \n"
  "  int at;                                                               \n"
  "  int prim, shell;                                                      \n"
  "                                                                        \n"
  "  // initialize value of orbital at gridpoint                           \n"
  "  float4 value = 0.0f;                                                  \n"
  "                                                                        \n"
  "  // initialize the wavefunction and shell counters                     \n"
  "  int ifunc = 0;                                                        \n"
  "  int shell_counter = 0;                                                \n"
  "  // loop over all the QM atoms                                         \n"
  "  for (at = 0; at < numatoms; at++) {                                   \n"
  "    // calculate distance between grid point and center of atom         \n"
  "    int maxshell = const_num_shells_per_atom[at];                       \n"
  "    int prim_counter = const_atom_basis[at];                            \n"
  "                                                                        \n"
  "    float4 xdist = (grid_x - const_atompos[3*at  ])*ANGS_TO_BOHR;       \n"
  "    float ydist = (grid_y - const_atompos[3*at+1])*ANGS_TO_BOHR;        \n"
  "    float zdist = (grid_z - const_atompos[3*at+2])*ANGS_TO_BOHR;        \n"
  "                                                                        \n"
  "    float4 xdist2 = xdist*xdist;                                        \n"
  "    float ydist2 = ydist*ydist;                                         \n"
  "    float zdist2 = zdist*zdist;                                         \n"
  "                                                                        \n"
  "    float4 dist2 = xdist2 + ydist2 + zdist2;                            \n"
  "                                                                        \n"
  "    // loop over the shells belonging to this atom (or basis function)  \n"
  "    for (shell=0; shell < maxshell; shell++) {                          \n"
  "      float4 contracted_gto = 0.0f;                                     \n"
  "                                                                        \n"
  "      // Loop over the Gaussian primitives of this contracted           \n"
  "      // basis function to build the atomic orbital                     \n"
  "      int maxprim = const_num_prim_per_shell[shell_counter];            \n"
  "      int shell_type = const_shell_symmetry[shell_counter];             \n"
  "      for (prim=0; prim < maxprim;  prim++) {                           \n"
  "        float exponent       = const_basis_array[prim_counter    ];     \n"
  "        float contract_coeff = const_basis_array[prim_counter + 1];     \n"
  "                                                                        \n"
  "        // By premultiplying the stored exponent factors etc,           \n"
  "        // we can use exp2f() rather than exp(), giving us full         \n"
  "        // precision, but with the speed of __expf()                    \n"
  "        contracted_gto += contract_coeff * native_exp2(-exponent*dist2);\n"
  "        prim_counter += 2;                                              \n"
  "      }                                                                 \n"
  "                                                                        \n"
  "      /* multiply with the appropriate wavefunction coefficient */      \n"
  "      float4 tmpshell=0.0f;                                             \n"
  "      switch (shell_type) {                                             \n"
  "        case S_SHELL:                                                   \n"
  "          value += const_wave_f[ifunc++] * contracted_gto;              \n"
  "          break;                                                        \n"
  "                                                                        \n"
  "        case P_SHELL:                                                   \n"
  "          tmpshell += const_wave_f[ifunc++] * xdist;                    \n"
  "          tmpshell += const_wave_f[ifunc++] * ydist;                    \n"
  "          tmpshell += const_wave_f[ifunc++] * zdist;                    \n"
  "          value += tmpshell * contracted_gto;                           \n"
  "          break;                                                        \n"
  "                                                                        \n"
  "        case D_SHELL:                                                   \n"
  "          tmpshell += const_wave_f[ifunc++] * xdist2;                   \n"
  "          tmpshell += const_wave_f[ifunc++] * xdist * ydist;            \n"
  "          tmpshell += const_wave_f[ifunc++] * ydist2;                   \n"
  "          tmpshell += const_wave_f[ifunc++] * xdist * zdist;            \n"
  "          tmpshell += const_wave_f[ifunc++] * ydist * zdist;            \n"
  "          tmpshell += const_wave_f[ifunc++] * zdist2;                   \n"
  "          value += tmpshell * contracted_gto;                           \n"
  "          break;                                                        \n"
  "                                                                        \n"
  "        case F_SHELL:                                                   \n"
  "          tmpshell += const_wave_f[ifunc++] * xdist2 * xdist;           \n"
  "          tmpshell += const_wave_f[ifunc++] * xdist2 * ydist;           \n"
  "          tmpshell += const_wave_f[ifunc++] * ydist2 * xdist;           \n"
  "          tmpshell += const_wave_f[ifunc++] * ydist2 * ydist;           \n"
  "          tmpshell += const_wave_f[ifunc++] * xdist2 * zdist;           \n"
  "          tmpshell += const_wave_f[ifunc++] * xdist * ydist * zdist;    \n"
  "          tmpshell += const_wave_f[ifunc++] * ydist2 * zdist;           \n"
  "          tmpshell += const_wave_f[ifunc++] * zdist2 * xdist;           \n"
  "          tmpshell += const_wave_f[ifunc++] * zdist2 * ydist;           \n"
  "          tmpshell += const_wave_f[ifunc++] * zdist2 * zdist;           \n"
  "          value += tmpshell * contracted_gto;                           \n"
  "          break;                                                        \n"
  "                                                                        \n"
  "        case G_SHELL:                                                   \n"
  "          tmpshell += const_wave_f[ifunc++] * xdist2 * xdist2;          \n"
  "          tmpshell += const_wave_f[ifunc++] * xdist2 * xdist * ydist;   \n"
  "          tmpshell += const_wave_f[ifunc++] * xdist2 * ydist2;          \n"
  "          tmpshell += const_wave_f[ifunc++] * ydist2 * ydist * xdist;   \n"
  "          tmpshell += const_wave_f[ifunc++] * ydist2 * ydist2;          \n"
  "          tmpshell += const_wave_f[ifunc++] * xdist2 * xdist * zdist;   \n"
  "          tmpshell += const_wave_f[ifunc++] * xdist2 * ydist * zdist;   \n"
  "          tmpshell += const_wave_f[ifunc++] * ydist2 * xdist * zdist;   \n"
  "          tmpshell += const_wave_f[ifunc++] * ydist2 * ydist * zdist;   \n"
  "          tmpshell += const_wave_f[ifunc++] * xdist2 * zdist2;          \n"
  "          tmpshell += const_wave_f[ifunc++] * zdist2 * xdist * ydist;   \n"
  "          tmpshell += const_wave_f[ifunc++] * ydist2 * zdist2;          \n"
  "          tmpshell += const_wave_f[ifunc++] * zdist2 * zdist * xdist;   \n"
  "          tmpshell += const_wave_f[ifunc++] * zdist2 * zdist * ydist;   \n"
  "          tmpshell += const_wave_f[ifunc++] * zdist2 * zdist2;          \n"
  "          value += tmpshell * contracted_gto;                           \n"
  "          break;                                                        \n"
  "      } // end switch                                                   \n"
  "                                                                        \n"
  "      shell_counter++;                                                  \n"
  "    }                                                                   \n"
  "  }                                                                     \n"
  "                                                                        \n"
  "  // return either orbital density or orbital wavefunction amplitude    \n"
  "  if (density) {                                                        \n"
  "    orbitalgrid[outaddr               ] = copysign(value.x*value.x, value.x); \n"
  "    orbitalgrid[outaddr+1*V4BLOCKSIZEX] = copysign(value.y*value.y, value.y); \n"
  "    orbitalgrid[outaddr+2*V4BLOCKSIZEX] = copysign(value.z*value.z, value.z); \n"
  "    orbitalgrid[outaddr+3*V4BLOCKSIZEX] = copysign(value.w*value.w, value.w); \n"
  "  } else {                                                              \n"
  "    orbitalgrid[outaddr               ] = value.x;                      \n"
  "    orbitalgrid[outaddr+1*V4BLOCKSIZEX] = value.y;                      \n"
  "    orbitalgrid[outaddr+2*V4BLOCKSIZEX] = value.z;                      \n"
  "    orbitalgrid[outaddr+3*V4BLOCKSIZEX] = value.w;                      \n"
  "  }                                                                     \n"
  "}                                                                       \n"
  "                                                                        \n"
  "                                                                        \n"
  "//                                                                      \n"
  "// OpenCL loading from global memory into shared memory for all arrays  \n"
  "//                                                                      \n"
  "                                                                        \n"
  "typedef union flint_t {                                                 \n"
  "  float f;                                                              \n"
  "  int i;                                                                \n"
  "} flint;                                                                \n"
  "                                                                        \n"
  "__kernel __attribute__((reqd_work_group_size(BLOCKSIZEX, BLOCKSIZEY, 1))) \n"
  "void clorbitaltiledshared(int numatoms,                                 \n"
  "                          __global float *wave_f,                       \n"
  "                          __global float *basis_array,                  \n"
  "                          __global flint *atominfo,                     \n"
  "                          __global int *shellinfo,                      \n"
  "                          float voxelsize,                              \n"
  "                          float originx,                                \n"
  "                          float originy,                                \n"
  "                          float grid_z,                                 \n"
  "                          int density,                                  \n"
  "                          __global float *orbitalgrid) {                \n"
  "  unsigned int xindex  = get_global_id(0);                              \n"
  "  unsigned int yindex  = get_global_id(1);                              \n"
  "  unsigned int outaddr = get_global_size(0) * yindex + xindex;          \n"
  "  float grid_x = originx + voxelsize * xindex;                          \n"
  "  float grid_y = originy + voxelsize * yindex;                          \n"
  "                                                                        \n"
  "  // XXX NVIDIA-specific warp logic                                     \n"
  "  int sidx = get_local_id(1) * get_local_size(0) + get_local_id(0);     \n"
  "                                                                        \n"
  "  __local float s_wave_f[SHAREDSIZE];                                   \n"
  "  int sblock_wave_f = 0;                                                \n"
  "  s_wave_f[sidx      ] = wave_f[sidx      ];                            \n"
  "  s_wave_f[sidx +  64] = wave_f[sidx +  64];                            \n"
  "  s_wave_f[sidx + 128] = wave_f[sidx + 128];                            \n"
  "  s_wave_f[sidx + 192] = wave_f[sidx + 192];                            \n"
  "  barrier(CLK_LOCAL_MEM_FENCE);                                         \n"
  "                                                                        \n"
  "  // similar to C version                                               \n"
  "  int at;                                                               \n"
  "  int prim, shell;                                                      \n"
  "                                                                        \n"
  "  // initialize value of orbital at gridpoint                           \n"
  "  float value = 0.0f;                                                   \n"
  "                                                                        \n"
  "  // initialize the wavefunction and primitive counters                 \n"
  "  int ifunc = 0;                                                        \n"
  "  int shell_counter = 0;                                                \n"
  "  int sblock_prim_counter = -1; // sentinel indicates no data loaded    \n"
  "  // loop over all the QM atoms                                         \n"
  "  for (at = 0; at < numatoms; at++) {                                   \n"
  "    __local flint s_atominfo[5];                                        \n"
  "    __local float s_basis_array[SHAREDSIZE];                            \n"
  "    barrier(CLK_LOCAL_MEM_FENCE);                                       \n"
  "    if (sidx < 5)                                                       \n"
  "      s_atominfo[sidx].i = atominfo[(at<<4) + sidx].i;                  \n"
  "    barrier(CLK_LOCAL_MEM_FENCE);                                       \n"
  "    int prim_counter = s_atominfo[3].i;                                 \n"
  "    int maxshell     = s_atominfo[4].i;                                 \n"
  "    int new_sblock_prim_counter = prim_counter & MEMCOAMASK;            \n"
  "    if (sblock_prim_counter != new_sblock_prim_counter) {               \n"
  "      sblock_prim_counter = new_sblock_prim_counter;                    \n"
  "      s_basis_array[sidx      ] = basis_array[sblock_prim_counter + sidx      ]; \n"
  "      s_basis_array[sidx +  64] = basis_array[sblock_prim_counter + sidx +  64]; \n"
  "      s_basis_array[sidx + 128] = basis_array[sblock_prim_counter + sidx + 128]; \n"
  "      s_basis_array[sidx + 192] = basis_array[sblock_prim_counter + sidx + 192]; \n"
  "      prim_counter -= sblock_prim_counter;                              \n"
  "      barrier(CLK_LOCAL_MEM_FENCE);                                     \n"
  "    }                                                                   \n"
  "                                                                        \n"
  "    // calculate distance between grid point and center of atom         \n"
  "    float xdist = (grid_x - s_atominfo[0].f)*ANGS_TO_BOHR;              \n"
  "    float ydist = (grid_y - s_atominfo[1].f)*ANGS_TO_BOHR;              \n"
  "    float zdist = (grid_z - s_atominfo[2].f)*ANGS_TO_BOHR;              \n"
  "                                                                        \n"
  "    float xdist2 = xdist*xdist;                                         \n"
  "    float ydist2 = ydist*ydist;                                         \n"
  "    float zdist2 = zdist*zdist;                                         \n"
  "                                                                        \n"
  "    float dist2 = xdist2 + ydist2 + zdist2;                             \n"
  "                                                                        \n"
  "    // loop over the shells belonging to this atom (or basis function)  \n"
  "    for (shell=0; shell < maxshell; shell++) {                          \n"
  "      float contracted_gto = 0.0f;                                      \n"
  "                                                                        \n"
  "      // Loop over the Gaussian primitives of this contracted           \n"
  "      // basis function to build the atomic orbital                     \n"
  "      __local int s_shellinfo[2];                                       \n"
  "                                                                        \n"
  "      barrier(CLK_LOCAL_MEM_FENCE);                                     \n"
  "      if (sidx < 2)                                                     \n"
  "        s_shellinfo[sidx] = shellinfo[(shell_counter<<4) + sidx];       \n"
  "      barrier(CLK_LOCAL_MEM_FENCE);                                     \n"
  "      int maxprim = s_shellinfo[0];                                     \n"
  "      int shell_type = s_shellinfo[1];                                  \n"
  "                                                                        \n"
  "      if ((prim_counter + (maxprim<<1)) >= SHAREDSIZE) {                \n"
  "        prim_counter += sblock_prim_counter;                            \n"
  "        sblock_prim_counter = prim_counter & MEMCOAMASK;                \n"
  "        s_basis_array[sidx      ] = basis_array[sblock_prim_counter + sidx      ]; \n"
  "        s_basis_array[sidx +  64] = basis_array[sblock_prim_counter + sidx +  64]; \n"
  "        s_basis_array[sidx + 128] = basis_array[sblock_prim_counter + sidx + 128]; \n"
  "        s_basis_array[sidx + 192] = basis_array[sblock_prim_counter + sidx + 192]; \n"
  "        prim_counter -= sblock_prim_counter;                            \n"
  "        barrier(CLK_LOCAL_MEM_FENCE);                                   \n"
  "      }                                                                 \n"
  "      for (prim=0; prim < maxprim;  prim++) {                           \n"
  "        float exponent       = s_basis_array[prim_counter    ];         \n"
  "        float contract_coeff = s_basis_array[prim_counter + 1];         \n"
  "                                                                        \n"
  "        // By premultiplying the stored exponent factors etc,           \n"
  "        // we can use exp2f() rather than exp(), giving us full         \n"
  "        // precision, but with the speed of __expf()                    \n"
  "        contracted_gto += contract_coeff * native_exp2(-exponent*dist2);\n"
  "                                                                        \n"
  "        prim_counter += 2;                                              \n"
  "      }                                                                 \n"
  "                                                                        \n"
  "      // XXX should use a constant memory lookup table to store         \n"
  "      // shared mem refill constants, and dynamically lookup the        \n"
  "      // number of elements referenced in the next iteration.           \n"
  "      if ((ifunc + MAXSHELLCOUNT) >= SHAREDSIZE) {                      \n"
  "        ifunc += sblock_wave_f;                                         \n"
  "        sblock_wave_f = ifunc & MEMCOAMASK;                             \n"
  "        barrier(CLK_LOCAL_MEM_FENCE);                                   \n"
  "        s_wave_f[sidx      ] = wave_f[sblock_wave_f + sidx      ];      \n"
  "        s_wave_f[sidx +  64] = wave_f[sblock_wave_f + sidx +  64];      \n"
  "        s_wave_f[sidx + 128] = wave_f[sblock_wave_f + sidx + 128];      \n"
  "        s_wave_f[sidx + 192] = wave_f[sblock_wave_f + sidx + 192];      \n"
  "        barrier(CLK_LOCAL_MEM_FENCE);                                   \n"
  "        ifunc -= sblock_wave_f;                                         \n"
  "      }                                                                 \n"
  "                                                                        \n"
  "      /* multiply with the appropriate wavefunction coefficient */      \n"
  "      float tmpshell=0.0f;                                              \n"
  "      switch (shell_type) {                                             \n"
  "        case S_SHELL:                                                   \n"
  "          value += s_wave_f[ifunc++] * contracted_gto;                  \n"
  "          break;                                                        \n"
  "                                                                        \n"
  "        case P_SHELL:                                                   \n"
  "          tmpshell += s_wave_f[ifunc++] * xdist;                        \n"
  "          tmpshell += s_wave_f[ifunc++] * ydist;                        \n"
  "          tmpshell += s_wave_f[ifunc++] * zdist;                        \n"
  "          value += tmpshell * contracted_gto;                           \n"
  "          break;                                                        \n"
  "                                                                        \n"
  "        case D_SHELL:                                                   \n"
  "          tmpshell += s_wave_f[ifunc++] * xdist2;                       \n"
  "          tmpshell += s_wave_f[ifunc++] * xdist * ydist;                \n"
  "          tmpshell += s_wave_f[ifunc++] * ydist2;                       \n"
  "          tmpshell += s_wave_f[ifunc++] * xdist * zdist;                \n"
  "          tmpshell += s_wave_f[ifunc++] * ydist * zdist;                \n"
  "          tmpshell += s_wave_f[ifunc++] * zdist2;                       \n"
  "          value += tmpshell * contracted_gto;                           \n"
  "          break;                                                        \n"
  "                                                                        \n"
  "        case F_SHELL:                                                   \n"
  "          tmpshell += s_wave_f[ifunc++] * xdist2 * xdist;               \n"
  "          tmpshell += s_wave_f[ifunc++] * xdist2 * ydist;               \n"
  "          tmpshell += s_wave_f[ifunc++] * ydist2 * xdist;               \n"
  "          tmpshell += s_wave_f[ifunc++] * ydist2 * ydist;               \n"
  "          tmpshell += s_wave_f[ifunc++] * xdist2 * zdist;               \n"
  "          tmpshell += s_wave_f[ifunc++] * xdist * ydist * zdist;        \n"
  "          tmpshell += s_wave_f[ifunc++] * ydist2 * zdist;               \n"
  "          tmpshell += s_wave_f[ifunc++] * zdist2 * xdist;               \n"
  "          tmpshell += s_wave_f[ifunc++] * zdist2 * ydist;               \n"
  "          tmpshell += s_wave_f[ifunc++] * zdist2 * zdist;               \n"
  "          value += tmpshell * contracted_gto;                           \n"
  "          break;                                                        \n"
  "                                                                        \n"
  "        case G_SHELL:                                                   \n"
  "          tmpshell += s_wave_f[ifunc++] * xdist2 * xdist2;              \n"
  "          tmpshell += s_wave_f[ifunc++] * xdist2 * xdist * ydist;       \n"
  "          tmpshell += s_wave_f[ifunc++] * xdist2 * ydist2;              \n"
  "          tmpshell += s_wave_f[ifunc++] * ydist2 * ydist * xdist;       \n"
  "          tmpshell += s_wave_f[ifunc++] * ydist2 * ydist2;              \n"
  "          tmpshell += s_wave_f[ifunc++] * xdist2 * xdist * zdist;       \n"
  "          tmpshell += s_wave_f[ifunc++] * xdist2 * ydist * zdist;       \n"
  "          tmpshell += s_wave_f[ifunc++] * ydist2 * xdist * zdist;       \n"
  "          tmpshell += s_wave_f[ifunc++] * ydist2 * ydist * zdist;       \n"
  "          tmpshell += s_wave_f[ifunc++] * xdist2 * zdist2;              \n"
  "          tmpshell += s_wave_f[ifunc++] * zdist2 * xdist * ydist;       \n"
  "          tmpshell += s_wave_f[ifunc++] * ydist2 * zdist2;              \n"
  "          tmpshell += s_wave_f[ifunc++] * zdist2 * zdist * xdist;       \n"
  "          tmpshell += s_wave_f[ifunc++] * zdist2 * zdist * ydist;       \n"
  "          tmpshell += s_wave_f[ifunc++] * zdist2 * zdist2;              \n"
  "          value += tmpshell * contracted_gto;                           \n"
  "          break;                                                        \n"
  "      } // end switch                                                   \n"
  "                                                                        \n"
  "      shell_counter++;                                                  \n"
  "    }                                                                   \n"
  "  }                                                                     \n"
  "                                                                        \n"
  "  // return either orbital density or orbital wavefunction amplitude    \n"
  "  if (density) {                                                        \n"
  "    orbitalgrid[outaddr] = copysign(value*value, value);                \n"
  "  } else {                                                              \n"
  "    orbitalgrid[outaddr] = value;                                       \n"
  "  }                                                                     \n"
  "}                                                                       \n"
  "                                                                        \n"
  "                                                                        \n";


cl_program vmd_opencl_compile_orbital_pgm(cl_context clctx, cl_device_id *cldevs, int &clerr) {
  cl_program clpgm = NULL;

  clpgm = clCreateProgramWithSource(clctx, 1, &clorbitalsrc, NULL, &clerr);
  CLERR

  char clcompileflags[4096];
  sprintf(clcompileflags, "-DUNROLLX=%d -DUNROLLY=%d -DBLOCKSIZEX=%d -DBLOCKSIZEY=%d -DBLOCKSIZE=%d -DSHAREDSIZE=%d -DMEMCOAMASK=%d -DMAXSHELLCOUNT=%d -DV4UNROLLX=%d -DV4UNROLLY=%d -DV4BLOCKSIZEX=%d -DV4BLOCKSIZEY=%d -cl-fast-relaxed-math -cl-single-precision-constant -cl-denorms-are-zero -cl-mad-enable -cl-no-signed-zeros", UNROLLX, UNROLLY, BLOCKSIZEX, BLOCKSIZEY, BLOCKSIZE, SHAREDSIZE, MEMCOAMASK, MAXSHELLCOUNT, V4UNROLLX, V4UNROLLY, V4BLOCKSIZEX, V4BLOCKSIZEY);
  clerr = clBuildProgram(clpgm, 0, NULL, clcompileflags, NULL, NULL);
  if (clerr != CL_SUCCESS)
    printf("  compilation failed!\n");

  if (cldevs) {
    char buildlog[8192];
    size_t len=0;
    clerr = clGetProgramBuildInfo(clpgm, cldevs[0], CL_PROGRAM_BUILD_LOG, sizeof(buildlog), buildlog, &len);
    if (len > 1) {
      printf("OpenCL compilation log:\n");
      printf("  '%s'\n", buildlog);
    }
    CLERR
  }

  return clpgm;
}


vmd_opencl_orbital_handle * vmd_opencl_create_orbital_handle(cl_context ctx, cl_command_queue cmdq, cl_device_id *devs) {
  vmd_opencl_orbital_handle * orbh;
  cl_int clerr = CL_SUCCESS;

  orbh = (vmd_opencl_orbital_handle *) malloc(sizeof(vmd_opencl_orbital_handle));
  orbh->ctx = ctx;
  orbh->cmdq = cmdq;
  orbh->devs = devs;

  orbh->pgm = vmd_opencl_compile_orbital_pgm(orbh->ctx, orbh->devs, clerr);
  CLERR
  orbh->kconstmem = clCreateKernel(orbh->pgm, "clorbitalconstmem", &clerr);
  CLERR
  orbh->kconstmem_vec4 = clCreateKernel(orbh->pgm, "clorbitalconstmem_vec4", &clerr);
  CLERR
  orbh->ktiledshared = clCreateKernel(orbh->pgm, "clorbitaltiledshared", &clerr);
  CLERR

  return orbh;
}


int vmd_opencl_destroy_orbital_handle(vmd_opencl_orbital_handle * orbh) {
  clReleaseKernel(orbh->kconstmem);
  clReleaseKernel(orbh->kconstmem_vec4);
  clReleaseKernel(orbh->ktiledshared);
  clReleaseProgram(orbh->pgm);
  free(orbh);

  return 0;
}



static int computepaddedsize(int orig, int tilesize) {
  int alignmask = tilesize - 1;
  int paddedsz = (orig + alignmask) & ~alignmask;  
//printf("orig: %d  padded: %d  tile: %d\n", orig, paddedsz, tilesize);
  return paddedsz;
}

static void * openclorbitalthread(void *voidparms) {
  size_t volsize[3], Gsz[3], Bsz[3];
  cl_mem d_wave_f = NULL;
  cl_mem d_basis_array = NULL;
  cl_mem d_atominfo = NULL;
  cl_mem d_shellinfo = NULL;
  cl_mem d_origin = NULL;
  cl_mem d_numvoxels = NULL;
  cl_mem d_orbitalgrid = NULL;
  cl_mem const_wave_f=NULL;
  cl_mem const_atompos=NULL;
  cl_mem const_atom_basis=NULL;
  cl_mem const_num_shells_per_atom=NULL;
  cl_mem const_basis_array=NULL;
  cl_mem const_num_prim_per_shell=NULL;
  cl_mem const_shell_symmetry=NULL;
  float *h_orbitalgrid = NULL;
  float *h_basis_array_exp2f = NULL;
  int numvoxels[3];
  float origin[3];
  orbthrparms *parms = NULL;
  int usefastconstkernel=0;
  int threadid=0;
  int tilesize = 1; // default tile size to use in absence of other info
  cl_int clerr = CL_SUCCESS;

#if defined(USE_OPENCLDEVPOOL)
  wkf_threadpool_worker_getid(voidparms, &threadid, NULL);
  wkf_threadpool_worker_getdata(voidparms, (void **) &parms);

  // scale tile size by device performance
  tilesize=4; // GTX 280, Tesla C1060 starting point tile size
  wkf_threadpool_worker_devscaletile(voidparms, &tilesize);
#else
  wkf_threadlaunch_getid(voidparms, &threadid, NULL);
  wkf_threadlaunch_getdata(voidparms, (void **) &parms);
#endif

  // get various existing OpenCL state from handle provided by caller
  vmd_opencl_orbital_handle * orbh = parms->orbh;
  cl_context clctx = parms->orbh->ctx;
  cl_command_queue clcmdq = parms->orbh->cmdq;

  numvoxels[0] = parms->numvoxels[0];
  numvoxels[1] = parms->numvoxels[1];
  numvoxels[2] = 1;

  origin[0] = parms->origin[0];
  origin[1] = parms->origin[1];

  int density = parms->density;
 
  // setup OpenCL grid and block sizes
  if (getenv("VMDMOVEC4")) {
    // setup energy grid size, padding out arrays for peak GPU memory performance
    volsize[0] = (parms->numvoxels[0] + V4GPU_X_ALIGNMASK) & ~(V4GPU_X_ALIGNMASK);
    volsize[1] = (parms->numvoxels[1] + V4GPU_Y_ALIGNMASK) & ~(V4GPU_Y_ALIGNMASK);
    volsize[2] = 1;      // we only do one plane at a time
    Bsz[0] = V4BLOCKSIZEX;
    Bsz[1] = V4BLOCKSIZEY;
    Bsz[2] = 1;
    Gsz[0] = volsize[0] / V4UNROLLX;
    Gsz[1] = volsize[1] / V4UNROLLY;
    Gsz[2] = 1;
  } else {
    volsize[0] = (parms->numvoxels[0] + GPU_X_ALIGNMASK) & ~(GPU_X_ALIGNMASK);
    volsize[1] = (parms->numvoxels[1] + GPU_Y_ALIGNMASK) & ~(GPU_Y_ALIGNMASK);
    volsize[2] = 1;      // we only do one plane at a time
    Bsz[0] = BLOCKSIZEX;
    Bsz[1] = BLOCKSIZEY;
    Bsz[2] = 1;
    Gsz[0] = volsize[0] / UNROLLX;
    Gsz[1] = volsize[1] / UNROLLY;
    Gsz[2] = 1;
  }
  int volmemsz = sizeof(float) * volsize[0] * volsize[1] * volsize[2];

  // determine which runtime strategy is workable
  // given the data sizes involved
  if ((parms->num_wave_f < MAX_WAVEF_SZ) &&
      (parms->numatoms < MAX_ATOM_SZ) &&
      (parms->numatoms < MAX_ATOMSHELL_SZ) &&
      (2*parms->num_basis < MAX_BASIS_SZ) &&
      (parms->num_shells < MAX_SHELL_SZ)) {
    usefastconstkernel=1;
  }

  // allow overrides for testing purposes
  if (getenv("VMDFORCEMOTILEDSHARED") != NULL) {
    usefastconstkernel=0; 
  }

  // allocate and copy input data to GPU arrays
  int padsz;
  padsz = computepaddedsize(2 * parms->num_basis, MEMCOALESCE);
  h_basis_array_exp2f = (float *) malloc(padsz * sizeof(float));
  float log2e = log2(2.718281828);
  for (int ll=0; ll<(2*parms->num_basis); ll+=2) {
#if 1
    // use exp2f() rather than expf()
    h_basis_array_exp2f[ll  ] = parms->basis_array[ll  ] * log2e;
#else
    h_basis_array_exp2f[ll  ] = parms->basis_array[ll  ];
#endif
    h_basis_array_exp2f[ll+1] = parms->basis_array[ll+1];
  }

  if (usefastconstkernel) {
    const_wave_f=clCreateBuffer(clctx, CL_MEM_READ_ONLY, parms->num_wave_f * sizeof(float), NULL, NULL);
    clEnqueueWriteBuffer(clcmdq, const_wave_f, CL_TRUE, 0, parms->num_wave_f * sizeof(float), parms->wave_f, 0, NULL, NULL);

    const_atompos=clCreateBuffer(clctx, CL_MEM_READ_ONLY, 3 * parms->numatoms * sizeof(float), NULL, NULL);
    clEnqueueWriteBuffer(clcmdq, const_atompos, CL_TRUE, 0, 3 * parms->numatoms * sizeof(float), parms->atompos, 0, NULL, NULL);
   
    const_atom_basis=clCreateBuffer(clctx, CL_MEM_READ_ONLY, parms->numatoms * sizeof(int), NULL, NULL);
    clEnqueueWriteBuffer(clcmdq, const_atom_basis, CL_TRUE, 0, parms->numatoms * sizeof(int), parms->atom_basis, 0, NULL, NULL);
   
    const_num_shells_per_atom=clCreateBuffer(clctx, CL_MEM_READ_ONLY, parms->numatoms * sizeof(int), NULL, NULL);
    clEnqueueWriteBuffer(clcmdq, const_num_shells_per_atom, CL_TRUE, 0,  parms->numatoms * sizeof(int), parms->num_shells_per_atom, 0, NULL, NULL);
    
    const_basis_array=clCreateBuffer(clctx, CL_MEM_READ_ONLY, 2 * parms->num_basis * sizeof(float), NULL, NULL);
    clEnqueueWriteBuffer(clcmdq, const_basis_array, CL_TRUE, 0, 2 * parms->num_basis * sizeof(float), h_basis_array_exp2f, 0, NULL, NULL);
    
    const_num_prim_per_shell=clCreateBuffer(clctx, CL_MEM_READ_ONLY, parms->num_shells * sizeof(int), NULL, NULL);
    clEnqueueWriteBuffer(clcmdq, const_num_prim_per_shell, CL_TRUE, 0, parms->num_shells * sizeof(int), parms->num_prim_per_shell, 0, NULL, NULL);
    
    const_shell_symmetry=clCreateBuffer(clctx, CL_MEM_READ_ONLY, parms->num_shells * sizeof(int), NULL, NULL);
    clEnqueueWriteBuffer(clcmdq, const_shell_symmetry, CL_TRUE, 0, parms->num_shells * sizeof(int), parms->shell_symmetry, 0, NULL, NULL);
  } else {
    padsz = computepaddedsize(parms->num_wave_f, MEMCOALESCE);
    d_wave_f = clCreateBuffer(clctx, CL_MEM_READ_ONLY, padsz * sizeof(float), NULL, NULL);
    clEnqueueWriteBuffer(clcmdq, d_wave_f, CL_TRUE, 0, parms->num_wave_f * sizeof(float), parms->wave_f, 0, NULL, NULL);

    // pack atom data into a tiled array
    padsz = computepaddedsize(16 * parms->numatoms, MEMCOALESCE);
    flint * h_atominfo = (flint *) calloc(1, padsz * sizeof(flint));
    d_atominfo = clCreateBuffer(clctx, CL_MEM_READ_ONLY, padsz * sizeof(flint), NULL, NULL);
    int ll;
    for (ll=0; ll<parms->numatoms; ll++) {
      int addr = ll * 16;
      h_atominfo[addr    ].f = parms->atompos[ll*3    ];
      h_atominfo[addr + 1].f = parms->atompos[ll*3 + 1];
      h_atominfo[addr + 2].f = parms->atompos[ll*3 + 2];
      h_atominfo[addr + 3].i = parms->atom_basis[ll];
      h_atominfo[addr + 4].i = parms->num_shells_per_atom[ll];
    }
    clEnqueueWriteBuffer(clcmdq, d_atominfo, CL_TRUE, 0, padsz * sizeof(flint), h_atominfo, 0, NULL, NULL);
    free(h_atominfo);

    padsz = computepaddedsize(16 * parms->num_shells, MEMCOALESCE);
    int * h_shellinfo = (int *) calloc(1, padsz * sizeof(int));
    d_shellinfo = clCreateBuffer(clctx, CL_MEM_READ_ONLY, padsz * sizeof(int), NULL, NULL);
    for (ll=0; ll<parms->num_shells; ll++) {
      h_shellinfo[ll*16    ] = parms->num_prim_per_shell[ll];
      h_shellinfo[ll*16 + 1] = parms->shell_symmetry[ll];
    }
    clEnqueueWriteBuffer(clcmdq, d_shellinfo, CL_TRUE, 0, padsz * sizeof(int), h_shellinfo, 0, NULL, NULL);
    free(h_shellinfo);

    d_basis_array = clCreateBuffer(clctx, CL_MEM_READ_ONLY, padsz * sizeof(float), NULL, NULL);
    clEnqueueWriteBuffer(clcmdq, d_basis_array, CL_TRUE, 0, 2 * parms->num_basis * sizeof(float), h_basis_array_exp2f, 0, NULL, NULL);
    
    padsz = computepaddedsize(3, MEMCOALESCE);
    d_origin = clCreateBuffer(clctx, CL_MEM_READ_ONLY, padsz * sizeof(float), NULL, NULL);
    clEnqueueWriteBuffer(clcmdq, d_origin, CL_TRUE, 0, 3 * sizeof(float), origin, 0, NULL, NULL);

    d_numvoxels = clCreateBuffer(clctx, CL_MEM_READ_ONLY, padsz * sizeof(int), NULL, NULL);
    clEnqueueWriteBuffer(clcmdq, d_numvoxels, CL_TRUE, 0, 3 * sizeof(int), numvoxels, 0, NULL, NULL);
  }

  // allocate and initialize the GPU output array
  d_orbitalgrid = clCreateBuffer(clctx, CL_MEM_READ_WRITE, volmemsz, NULL, NULL);
  CLERR // check and clear any existing errors

  // allocate working buffer
  h_orbitalgrid = (float *) malloc(volmemsz);

#if 0
  if (threadid == 0) {
    // inform on which kernel we're actually going to run
    printf("atoms[%d] ", parms->numatoms);
    printf("wavef[%d] ", parms->num_wave_f);
    printf("basis[%d] ", parms->num_basis);
    printf("shell[%d] ", parms->num_shells);
    if (usefastconstkernel) {
      printf("GPU constant memory");
    } else {
      printf("GPU tiled shared memory:");
    }
    printf(" Gsz:%dx%d\n", (int) Gsz[0], (int) Gsz[1]);
  }
#endif

  // loop over orbital planes
  wkf_tasktile_t tile;
  int planesize = numvoxels[0] * numvoxels[1];

#if defined(USE_OPENCLDEVPOOL)
  while (wkf_threadpool_next_tile(voidparms, tilesize, &tile) != WKF_SCHED_DONE) {
#else
  while (wkf_threadlaunch_next_tile(voidparms, tilesize, &tile) != WKF_SCHED_DONE) {
#endif
    int k;
    for (k=tile.start; k<tile.end; k++) {
      origin[2] = parms->origin[2] + parms->voxelsize * k;


      // RUN the kernel...
      if (usefastconstkernel) {
        cl_event event;
        if (getenv("VMDMOVEC4")) {
          clerr |= clSetKernelArg(orbh->kconstmem_vec4, 0, sizeof(int), &parms->numatoms);
          clerr |= clSetKernelArg(orbh->kconstmem_vec4, 1, sizeof(cl_mem), &const_atompos);
          clerr |= clSetKernelArg(orbh->kconstmem_vec4, 2, sizeof(cl_mem), &const_atom_basis);
          clerr |= clSetKernelArg(orbh->kconstmem_vec4, 3, sizeof(cl_mem), &const_num_shells_per_atom);
          clerr |= clSetKernelArg(orbh->kconstmem_vec4, 4, sizeof(cl_mem), &const_basis_array);
          clerr |= clSetKernelArg(orbh->kconstmem_vec4, 5, sizeof(cl_mem), &const_num_prim_per_shell);
          clerr |= clSetKernelArg(orbh->kconstmem_vec4, 6, sizeof(cl_mem), &const_shell_symmetry);
          clerr |= clSetKernelArg(orbh->kconstmem_vec4, 7, sizeof(cl_mem), &const_wave_f);
          clerr |= clSetKernelArg(orbh->kconstmem_vec4, 8, sizeof(float), &parms->voxelsize);
          clerr |= clSetKernelArg(orbh->kconstmem_vec4, 9, sizeof(float), &origin[0]);
          clerr |= clSetKernelArg(orbh->kconstmem_vec4, 10, sizeof(float), &origin[1]);
          clerr |= clSetKernelArg(orbh->kconstmem_vec4, 11, sizeof(float), &origin[2]);
          clerr |= clSetKernelArg(orbh->kconstmem_vec4, 12, sizeof(cl_int), &density);
          clerr |= clSetKernelArg(orbh->kconstmem_vec4, 13, sizeof(cl_mem), &d_orbitalgrid);
          clerr = clEnqueueNDRangeKernel(clcmdq, orbh->kconstmem_vec4, 2, NULL, Gsz, Bsz, 0, NULL, &event);
        } else {
          clerr |= clSetKernelArg(orbh->kconstmem, 0, sizeof(int), &parms->numatoms);
          clerr |= clSetKernelArg(orbh->kconstmem, 1, sizeof(cl_mem), &const_atompos);
          clerr |= clSetKernelArg(orbh->kconstmem, 2, sizeof(cl_mem), &const_atom_basis);
          clerr |= clSetKernelArg(orbh->kconstmem, 3, sizeof(cl_mem), &const_num_shells_per_atom);
          clerr |= clSetKernelArg(orbh->kconstmem, 4, sizeof(cl_mem), &const_basis_array);
          clerr |= clSetKernelArg(orbh->kconstmem, 5, sizeof(cl_mem), &const_num_prim_per_shell);
          clerr |= clSetKernelArg(orbh->kconstmem, 6, sizeof(cl_mem), &const_shell_symmetry);
          clerr |= clSetKernelArg(orbh->kconstmem, 7, sizeof(cl_mem), &const_wave_f);
          clerr |= clSetKernelArg(orbh->kconstmem, 8, sizeof(float), &parms->voxelsize);
          clerr |= clSetKernelArg(orbh->kconstmem, 9, sizeof(float), &origin[0]);
          clerr |= clSetKernelArg(orbh->kconstmem, 10, sizeof(float), &origin[1]);
          clerr |= clSetKernelArg(orbh->kconstmem, 11, sizeof(float), &origin[2]);
          clerr |= clSetKernelArg(orbh->kconstmem, 12, sizeof(cl_int), &density);
          clerr |= clSetKernelArg(orbh->kconstmem, 13, sizeof(cl_mem), &d_orbitalgrid);
          clerr = clEnqueueNDRangeKernel(clcmdq, orbh->kconstmem, 2, NULL, Gsz, Bsz, 0, NULL, &event);
        }
        CLERR // check and clear any existing errors
        clerr |= clWaitForEvents(1, &event);
        clerr |= clReleaseEvent(event);
        CLERR // check and clear any existing errors
      } else {
        clerr |= clSetKernelArg(orbh->ktiledshared, 0, sizeof(int), &parms->numatoms);
        clerr |= clSetKernelArg(orbh->ktiledshared, 1, sizeof(cl_mem), &d_wave_f);
        clerr |= clSetKernelArg(orbh->ktiledshared, 2, sizeof(cl_mem), &d_basis_array);
        clerr |= clSetKernelArg(orbh->ktiledshared, 3, sizeof(cl_mem), &d_atominfo);
        clerr |= clSetKernelArg(orbh->ktiledshared, 4, sizeof(cl_mem), &d_shellinfo);
        clerr |= clSetKernelArg(orbh->ktiledshared, 5, sizeof(float), &parms->voxelsize);
        clerr |= clSetKernelArg(orbh->ktiledshared, 6, sizeof(float), &origin[0]);
        clerr |= clSetKernelArg(orbh->ktiledshared, 7, sizeof(float), &origin[1]);
        clerr |= clSetKernelArg(orbh->ktiledshared, 8, sizeof(float), &origin[2]);
        clerr |= clSetKernelArg(orbh->ktiledshared, 9, sizeof(cl_int), &density);
        clerr |= clSetKernelArg(orbh->ktiledshared, 10, sizeof(cl_mem), &d_orbitalgrid);
        cl_event event;
        clerr = clEnqueueNDRangeKernel(clcmdq, orbh->ktiledshared, 2, NULL, Gsz, Bsz, 0, NULL, &event);
        CLERR // check and clear any existing errors
        clerr |= clWaitForEvents(1, &event);
        clerr |= clReleaseEvent(event);
        CLERR // check and clear any existing errors
      }
      CLERR // check and clear any existing errors

      // Copy the GPU output data back to the host and use/store it..
      clEnqueueReadBuffer(clcmdq, d_orbitalgrid, CL_TRUE, 0, volmemsz, h_orbitalgrid, 0, NULL, NULL);
      CLERR // check and clear any existing errors
      clFinish(clcmdq);

      // Copy GPU blocksize padded array back down to the original size
      int y;
      for (y=0; y<numvoxels[1]; y++) {
        long orbaddr = k*planesize + y*numvoxels[0];
        memcpy(&parms->orbitalgrid[orbaddr], &h_orbitalgrid[y*volsize[0]], numvoxels[0] * sizeof(float));
      }
    }
  }
  clFinish(clcmdq);

// printf("worker[%d] done.\n", threadid);

// printf("freeing host memory objects\n");
  free(h_basis_array_exp2f);
  free(h_orbitalgrid);

  if (usefastconstkernel) {
// printf("freeing device constant memory objects\n");
    clReleaseMemObject(const_wave_f);
    clReleaseMemObject(const_atompos);
    clReleaseMemObject(const_atom_basis);
    clReleaseMemObject(const_num_shells_per_atom);
    clReleaseMemObject(const_basis_array);
    clReleaseMemObject(const_num_prim_per_shell);
    clReleaseMemObject(const_shell_symmetry);
  } else {
// printf("freeing device global memory objects\n");
    clReleaseMemObject(d_wave_f);
    clReleaseMemObject(d_basis_array);
    clReleaseMemObject(d_atominfo);
    clReleaseMemObject(d_shellinfo);
    clReleaseMemObject(d_numvoxels);
    clReleaseMemObject(d_origin);
  }
  clReleaseMemObject(d_orbitalgrid);
  CLERR // check and clear any existing errors

  return NULL;
}



int vmd_opencl_evaluate_orbital_grid(wkf_threadpool_t *devpool,
                       vmd_opencl_orbital_handle *orbh,
                       int numatoms,
                       const float *wave_f, int num_wave_f,
                       const float *basis_array, int num_basis,
                       const float *atompos,
                       const int *atom_basis,
                       const int *num_shells_per_atom,
                       const int *num_prim_per_shell,
                       const int *shell_symmetry,
                       int num_shells,
                       const int *numvoxels,
                       float voxelsize,
                       const float *origin,
                       int density,
                       float *orbitalgrid) {
  int rc=0;
  orbthrparms parms;

  parms.numatoms = numatoms;
  parms.wave_f = wave_f;
  parms.num_wave_f = num_wave_f;
  parms.basis_array = basis_array;
  parms.num_basis = num_basis;
  parms.atompos = atompos;
  parms.atom_basis = atom_basis;
  parms.num_shells_per_atom = num_shells_per_atom;
  parms.num_prim_per_shell = num_prim_per_shell;
  parms.shell_symmetry = shell_symmetry;
  parms.num_shells = num_shells;
  parms.numvoxels = numvoxels;
  parms.voxelsize = voxelsize;
  parms.origin = origin;
  parms.density = density;
  parms.orbitalgrid = orbitalgrid;
  parms.orbh = orbh;

  wkf_tasktile_t tile;
  tile.start=0;
  tile.end=numvoxels[2];

  // XXX hard-coded for single GPU/CPU device
  // multi-GPU is disabled at present as we have to add a bunch of 
  // explicit infrastructure to gaurantee thread safety of various
  // OpenCL API calls among multiple worker threads.
#if defined(USE_OPENCLDEVPOOL)
  wkf_threadpool_sched_dynamic(devpool, &tile);
  wkf_threadpool_launch(devpool, openclorbitalthread, &parms, 1); 
#else
  /* spawn child threads to do the work */
  rc = wkf_threadlaunch(1, &parms, openclorbitalthread, &tile);
#endif

  return rc;
}


