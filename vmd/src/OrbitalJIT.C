/***************************************************************************
 *cr
 *cr            (C) Copyright 2008-2009 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: OrbitalJIT.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.3 $      $Date: 2010/06/11 21:47:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   This source file contains the just-in-time (JIT) code for generating
 *  CUDA and OpenCL kernels for computation of molecular orbitals on a 
 *  uniformly spaced grid, using one or more GPUs.
 *
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "WKFThreads.h"
#include "OrbitalJIT.h"

// Only enabled for testing an emulation of JIT code generation approach
// #define VMDMOJIT 1
// #define VMDMOJITSRC "/home/johns/mojit.cu"

#define ANGS_TO_BOHR 1.8897259877218677f

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

#define MEMCOALESCE  384

// orbital shell types 
#define S_SHELL 0
#define P_SHELL 1
#define D_SHELL 2
#define F_SHELL 3
#define G_SHELL 4
#define H_SHELL 5

//
// CUDA constant arrays 
//
#define MAX_ATOM_SZ 256

#define MAX_ATOMPOS_SZ (MAX_ATOM_SZ)
// __constant__ static float const_atompos[MAX_ATOMPOS_SZ * 3];

#define MAX_ATOM_BASIS_SZ (MAX_ATOM_SZ)
// __constant__ static int const_atom_basis[MAX_ATOM_BASIS_SZ];

#define MAX_ATOMSHELL_SZ (MAX_ATOM_SZ)
// __constant__ static int const_num_shells_per_atom[MAX_ATOMSHELL_SZ];

#define MAX_BASIS_SZ 6144 
// __constant__ static float const_basis_array[MAX_BASIS_SZ];

#define MAX_SHELL_SZ 1024
// __constant__ static int const_num_prim_per_shell[MAX_SHELL_SZ];
// __constant__ static int const_shell_types[MAX_SHELL_SZ];

#define MAX_WAVEF_SZ 6144
// __constant__ static float const_wave_f[MAX_WAVEF_SZ];


//
// CUDA JIT code generator for constant memory
//
int orbital_jit_generate(int jitlanguage,
                         const char * srcfilename, int numatoms,
                         const float *wave_f, const float *basis_array,
                         const int *atom_basis,
                         const int *num_shells_per_atom,
                         const int *num_prim_per_shell,
                         const int *shell_types) {
  FILE *ofp=NULL;
  if (srcfilename) 
    ofp=fopen(srcfilename, "w");

  if (ofp == NULL)
    ofp=stdout; 

  // calculate the value of the wavefunction of the
  // selected orbital at the current grid point
  int at;
  int prim, shell;

  // initialize the wavefunction and shell counters
  int shell_counter = 0;

  if (jitlanguage == ORBITAL_JIT_CUDA) {
    fprintf(ofp, 
      "__global__ static void cuorbitalconstmem_jit(int numatoms,\n"
      "                          float voxelsize,\n"
      "                          float originx,\n"
      "                          float originy,\n"
      "                          float grid_z, \n"
      "                          int density, \n"
      "                          float * orbitalgrid) {\n"
      "  unsigned int xindex  = __umul24(blockIdx.x, blockDim.x)\n"
      "                         + threadIdx.x;\n"
      "  unsigned int yindex  = __umul24(blockIdx.y, blockDim.y)\n"
      "                         + threadIdx.y;\n"
      "  unsigned int outaddr = __umul24(gridDim.x, blockDim.x) * yindex\n"
      "                         + xindex;\n"
    );
  } else if (jitlanguage == ORBITAL_JIT_OPENCL) {
    fprintf(ofp, 
      "// unit conversion                                                  \n"
      "#define ANGS_TO_BOHR 1.8897259877218677f                            \n"
    );

    fprintf(ofp, "__kernel __attribute__((reqd_work_group_size(%d, %d, 1)))\n",
            BLOCKSIZEX, BLOCKSIZEY);

    fprintf(ofp, 
      "void clorbitalconstmem_jit(int numatoms,                            \n"
      "                       __constant float *const_atompos,             \n"
      "                       __constant float *const_wave_f,              \n"
      "                       float voxelsize,                             \n"
      "                       float originx,                               \n"
      "                       float originy,                               \n"
      "                       float grid_z,                                \n"
      "                       int density,                                 \n"
      "                       __global float * orbitalgrid) {              \n"
      "  unsigned int xindex  = get_global_id(0);                          \n"
      "  unsigned int yindex  = get_global_id(1);                          \n"
      "  unsigned int outaddr = get_global_size(0) * yindex + xindex;      \n"
    );
  }

  fprintf(ofp, 
    "  float grid_x = originx + voxelsize * xindex;\n"
    "  float grid_y = originy + voxelsize * yindex;\n"
 
    "  // similar to C version\n"
    "  int at;\n"
    "  // initialize value of orbital at gridpoint\n"
    "  float value = 0.0f;\n"
    "  // initialize the wavefunction and shell counters\n"
    "  int ifunc = 0;\n"
    "  // loop over all the QM atoms\n"
    "  for (at = 0; at < numatoms; at++) {\n"
    "    // calculate distance between grid point and center of atom\n"
//    "    int maxshell = const_num_shells_per_atom[at];\n"
//    "    int prim_counter = const_atom_basis[at];\n"
    "    float xdist = (grid_x - const_atompos[3*at  ])*ANGS_TO_BOHR;\n"
    "    float ydist = (grid_y - const_atompos[3*at+1])*ANGS_TO_BOHR;\n"
    "    float zdist = (grid_z - const_atompos[3*at+2])*ANGS_TO_BOHR;\n"
    "    float xdist2 = xdist*xdist;\n"
    "    float ydist2 = ydist*ydist;\n"
    "    float zdist2 = zdist*zdist;\n"
    "    float dist2 = xdist2 + ydist2 + zdist2;\n"
    "    float contracted_gto=0.0f;\n"
    "    float tmpshell=0.0f;\n"
    "\n"
  );

#if 0
  // loop over all the QM atoms generating JIT code for each type
  for (at=0; at<numatoms; at++) {
#else
  // generate JIT code for one atom type and assume they are all the same
  for (at=0; at<1; at++) {
#endif
    int maxshell = num_shells_per_atom[at];
    int prim_counter = atom_basis[at];

    // loop over the shells belonging to this atom
    for (shell=0; shell < maxshell; shell++) {
      // Loop over the Gaussian primitives of this contracted
      // basis function to build the atomic orbital
      int maxprim = num_prim_per_shell[shell_counter];
      int shelltype = shell_types[shell_counter];
      for (prim=0; prim<maxprim; prim++) {
        float exponent       = basis_array[prim_counter    ];
        float contract_coeff = basis_array[prim_counter + 1];
#if 1
        if (jitlanguage == ORBITAL_JIT_CUDA) {
          if (prim == 0) {
            fprintf(ofp,"    contracted_gto = %ff * exp2f(-%ff*dist2);\n",
                    contract_coeff, exponent);
          } else {
            fprintf(ofp,"    contracted_gto += %ff * exp2f(-%ff*dist2);\n",
                    contract_coeff, exponent);
          }
        } else if (jitlanguage == ORBITAL_JIT_OPENCL) {
          if (prim == 0) {
            fprintf(ofp,"    contracted_gto = %ff * native_exp2(-%ff*dist2);\n",
                    contract_coeff, exponent);
          } else {
            fprintf(ofp,"    contracted_gto += %ff * native_exp2(-%ff*dist2);\n",
                    contract_coeff, exponent);
          }
        }
#else
        if (jitlanguage == ORBITAL_JIT_CUDA) {
          if (prim == 0) {
            fprintf(ofp,"    contracted_gto = %ff * expf(-%ff*dist2);\n",
                    contract_coeff, exponent);
          } else {
            fprintf(ofp,"    contracted_gto += %ff * expf(-%ff*dist2);\n",
                    contract_coeff, exponent);
          }
        } else if (jitlanguage == ORBITAL_JIT_OPENCL) {
          if (prim == 0) {
            fprintf(ofp,"    contracted_gto = %ff * native_exp(-%ff*dist2);\n",
                    contract_coeff, exponent);
          } else {
            fprintf(ofp,"    contracted_gto += %ff * native_exp(-%ff*dist2);\n",
                    contract_coeff, exponent);
          }
        }
#endif
        prim_counter += 2;
      }

      /* multiply with the appropriate wavefunction coefficient */
      switch (shelltype) {
        case S_SHELL:
          fprintf(ofp, 
            "    // S_SHELL\n"
            "    value += const_wave_f[ifunc++] * contracted_gto;\n");
          break;

        case P_SHELL:
          fprintf(ofp,
            "    // P_SHELL\n"
            "    tmpshell = const_wave_f[ifunc++] * xdist;\n"
            "    tmpshell += const_wave_f[ifunc++] * ydist;\n"
            "    tmpshell += const_wave_f[ifunc++] * zdist;\n"
            "    value += tmpshell * contracted_gto;\n"
          );
          break;

        case D_SHELL:
          fprintf(ofp,
            "    // D_SHELL\n"
            "    tmpshell = const_wave_f[ifunc++] * xdist2;\n"
            "    tmpshell += const_wave_f[ifunc++] * xdist * ydist;\n"
            "    tmpshell += const_wave_f[ifunc++] * ydist2;\n"
            "    tmpshell += const_wave_f[ifunc++] * xdist * zdist;\n"
            "    tmpshell += const_wave_f[ifunc++] * ydist * zdist;\n"
            "    tmpshell += const_wave_f[ifunc++] * zdist2;\n"
            "    value += tmpshell * contracted_gto;\n"
          );
          break;

        case F_SHELL:
          fprintf(ofp,
            "    // F_SHELL\n"
            "    tmpshell = const_wave_f[ifunc++] * xdist2 * xdist;\n"
            "    tmpshell += const_wave_f[ifunc++] * xdist2 * ydist;\n"
            "    tmpshell += const_wave_f[ifunc++] * ydist2 * xdist;\n"
            "    tmpshell += const_wave_f[ifunc++] * ydist2 * ydist;\n"
            "    tmpshell += const_wave_f[ifunc++] * xdist2 * zdist;\n"
            "    tmpshell += const_wave_f[ifunc++] * xdist * ydist * zdist;\n"
            "    tmpshell += const_wave_f[ifunc++] * ydist2 * zdist;\n"
            "    tmpshell += const_wave_f[ifunc++] * zdist2 * xdist;\n"
            "    tmpshell += const_wave_f[ifunc++] * zdist2 * ydist;\n"
            "    tmpshell += const_wave_f[ifunc++] * zdist2 * zdist;\n"
            "    value += tmpshell * contracted_gto;\n"
          );
          break;

        case G_SHELL:
          fprintf(ofp,
            "    // G_SHELL\n"
            "    tmpshell = const_wave_f[ifunc++] * xdist2 * xdist2;\n"
            "    tmpshell += const_wave_f[ifunc++] * xdist2 * xdist * ydist;\n"
            "    tmpshell += const_wave_f[ifunc++] * xdist2 * ydist2;\n"
            "    tmpshell += const_wave_f[ifunc++] * ydist2 * ydist * xdist;\n"
            "    tmpshell += const_wave_f[ifunc++] * ydist2 * ydist2;\n"
            "    tmpshell += const_wave_f[ifunc++] * xdist2 * xdist * zdist;\n"
            "    tmpshell += const_wave_f[ifunc++] * xdist2 * ydist * zdist;\n"
            "    tmpshell += const_wave_f[ifunc++] * ydist2 * xdist * zdist;\n"
            "    tmpshell += const_wave_f[ifunc++] * ydist2 * ydist * zdist;\n"
            "    tmpshell += const_wave_f[ifunc++] * xdist2 * zdist2;\n"
            "    tmpshell += const_wave_f[ifunc++] * zdist2 * xdist * ydist;\n"
            "    tmpshell += const_wave_f[ifunc++] * ydist2 * zdist2;\n"
            "    tmpshell += const_wave_f[ifunc++] * zdist2 * zdist * xdist;\n"
            "    tmpshell += const_wave_f[ifunc++] * zdist2 * zdist * ydist;\n"
            "    tmpshell += const_wave_f[ifunc++] * zdist2 * zdist2;\n"
            "    value += tmpshell * contracted_gto;\n"
          );
          break;

      } // end switch
      fprintf(ofp, "\n");

      shell_counter++;
    } // end shell
  } // end atom

  fprintf(ofp, 
    "  }\n"
    "\n"
    "  // return either orbital density or orbital wavefunction amplitude \n"
    "  if (density) { \n"
  );

  if (jitlanguage == ORBITAL_JIT_CUDA) {
    fprintf(ofp, "    orbitalgrid[outaddr] = copysignf(value*value, value);\n");
  } else if (jitlanguage == ORBITAL_JIT_OPENCL) {
    fprintf(ofp, "    orbitalgrid[outaddr] = copysign(value*value, value);\n");
  }

  fprintf(ofp, 
    "  } else { \n"
    "    orbitalgrid[outaddr] = value; \n"
    "  }\n"
    "}\n"
  );

  if (ofp != stdout)
    fclose(ofp);

  return 0;
}



