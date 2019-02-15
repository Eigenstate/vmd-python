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
 *	$RCSfile: Orbital.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.43 $	$Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * The Orbital class, which stores orbitals, for a
 * single timestep.
 *
 ***************************************************************************/
#ifndef ORBITAL_H
#define ORBITAL_H

#include <string.h>
#include "QMData.h"
#include "Molecule.h"
#include "ProfileHooks.h"

/// The Orbital class, which stores orbitals, SCF energies, etc. for a
/// single timestep.
class Orbital {
private:
  int numatoms;         ///< # of atom centers needed for this Orbital.
  const float *atompos; ///< pointer to the atom coordinates in the
                        ///< Timestep corresponding to this Orbital.

  int num_wave_f; ///< # of wave function coefficients, i.e. the
                  ///< # of cartesian contracted gaussian basis functions

  float *wave_f;            ///< expansion coefficients for wavefunction

  int num_basis_funcs;      ///< # of basis functions stored in basis_array

  const float *basis_array; ///< array of size 2*num_basis_funcs holding
                            ///< the contraction coeffients and exponents
                            ///< for the basis functions in the form
                            ///< {exp(1), c-coeff(1), exp(2), c-coeff(2), ....};
                            ///< SP-shells must be expanded into a separate S-
                            ///< and P-shell, respectively.

  int numtypes;
  const basis_atom_t *basis_set;  ///< hierarchical representation of the
                                  ///< basis set for each atom.
  const int *atom_types;          ///< maps atom indexes to atom types
  const int *atom_sort;           ///< atom indexes sorted by atom type

  const int *atom_basis;          ///< offset into basis_array for each atom

  const float **norm_factors;     ///< normalization factors for each shell type
                                  ///< and their cartesian functions.

  const int *num_shells_per_atom; ///< # shells per atom i
  const int *num_prim_per_shell;  ///< # primitives per shell i
  const int *shell_types;         ///< symmetry type (0, 1, 2, ..) per
                                  ///< (exp(),c-coeff()) pair in basis

  // grid related data
  int numvoxels[3];  ///< Number of voxels in each dimension
  float voxelsize;   ///< Side length of a voxel i.e. the grid resolution
  float origin[3];   ///< Origin of the grid
  float gridsize[3]; ///< Length of each grid dimension in Angstrom;
                     ///< Should be multiples of voxelsize.
  float *grid_data;  ///< Raw volumetric data;
                     ///< the grid is assumed to be orthogonal.


public:
  Orbital(const float *atompos,
          const float *wave_function,
          const float *basis,
          const basis_atom_t *bset,
          const int *types,
          const int *atom_sort,
          const int *atom_basis,
          const float **norm_factors,
          const int *num_shells_per_atom,
          const int *num_prim_per_shell,
          const int *orbital_symmetry,
          int numatoms, int numtypes, int num_wave_f,
          int num_basis_funcs, 
          int orbid); ///< constructor

  ~Orbital(void);               ///< destructor


  // Return array sizes need for GPU-acclerated versions
  int total_shells() {
    int shellcnt=0;
    for (int at=0; at<numatoms; at++) {
      for (int shell=0; shell < num_shells_per_atom[at]; shell++) { 
        shellcnt++;
      }
    }

    return shellcnt;
  }

  int num_types(void) { return numtypes; }

  /// Return the max number of primitives that occur in a basis function
  int max_primitives(void);

  /// Return maximum shell type contained in the orbital
  int max_shell_type(void);

  /// Count the max number of wave_f accesses for the shell types
  /// contained in this orbital 
  int max_wave_f_count(void);

  /// Get the grid origin
  const float* get_origin() { return origin; }

  /// Get the side lengths of the grid in Angstrom
  const float* get_gridsize() { return gridsize; }

  /// Get the number of voxels in each dimension
  const int* get_numvoxels() { return numvoxels; }

  /// Get the axes of the volumetric grid as defined in volumetric_t.
  void get_grid_axes(float xaxis[3], float yaxis[3], float zaxis[3]) {
    xaxis[0] = gridsize[0];
    yaxis[1] = gridsize[1];
    zaxis[2] = gridsize[2];
    xaxis[1] = xaxis[2] = yaxis[0] = yaxis[2] = zaxis[0] = zaxis[1] = 0.0;
  }

  /// Get the grid resolution, i.e. the side length of a voxel
  float get_resolution() { return voxelsize; }

  /// Set the grid size and resolution
  /// The given grid dimensions will be rounded to a multiple
  /// of the voxel size.
  void set_grid(float newori[3], float newdim[3], float voxelsize);

  /// Change the resolution of the grid
  void set_resolution(float voxelsize);

  /// Get a pointer to the raw volumetric data
  float* get_grid_data() { return grid_data; }

  /// Sets the grid dimensions to the bounding box of the given
  /// set of atoms *pos including a padding in all dimensions.
  /// The resulting grid dimensions will be rounded to a multiple
  /// of the voxel size.
  int set_grid_to_bbox(const float *pos, float padding,
                       float resolution);

  /// Optimize position and dimension of current grid so that
  /// all orbital values higher than threshold are contained
  /// in the grid.
  void find_optimal_grid(float threshold,
                         int minstepsize, int maxstepsize);

  /// Check if all values in the boundary plane given by dir 
  /// are below threshold.
  /// If not, jump back, decrease the stepsize and test again.
  /// Helper function for find_optimal_grid().
  int check_plane(int w, float threshold, int minstepsize, int &stepsize);

  /// Multiply wavefunction coefficients with the
  /// basis set normalization factors.
  void normalize_wavefunction(const float *wfn);

  /// Compute the volumetric data for the orbital
  int calculate_mo(DrawMolecule *mol, int density);

  /// Compute the volumetric data for given point in space
  float evaluate_grid_point(float grid_x, float grid_y, float grid_z);

  /// Compute total FLOPS executed for a single gridpoint
  double flops_per_gridpoint();

  void print_wavefunction();
};


// Compute the volumetric data for the whole grid
int evaluate_grid(int numatoms, 
                  const float *wave_f, 
                  const float *basis_array,
                  const float *atompos,
                  const int *atom_basis,
                  const int *num_shells_per_atom,
                  const int *num_prim_per_shell,
                  const int *shell_types,
                  const int *numvoxels,
                  float voxelsize,
                  const float *origin,
                  int density,
                  float * orbitalgrid); 

#define VMDORBUSETHRPOOL 1

// Multiprocessor implementation
int evaluate_grid_fast(
#if defined(VMDORBUSETHRPOOL)
                       wkf_threadpool_t *thrpool,
#else
                       int numcputhreads,
#endif
                       int numatoms, 
                       const float *wave_f,
                       const float *basis_array,
                       const float *atompos,
                       const int *atom_basis,
                       const int *num_shells_per_atom,
                       const int *num_prim_per_shell,
                       const int *shell_types,
                       const int *numvoxels,
                       float voxelsize,
                       const float *origin,
                       int density,
                       float * orbitalgrid); 

#endif

