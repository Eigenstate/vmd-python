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
 *	$RCSfile: Timestep.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.53 $	$Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * The Timestep class, which stores coordinates, energies, etc. for a
 * single timestep.
 *
 * Note: As more data is stored for each step, it should go in here.  For
 * example, H-Bonds could be calculated each step.
 ***************************************************************************/
#ifndef TIMESTEP_H
#define TIMESTEP_H

#include "ResizeArray.h"
#include "Matrix4.h"
#include "QMTimestep.h"

// Energy terms and temperature stored for each timestep
// TSENERGIES must be the last element.  It indicates the number
// energies.  (TSE_TOTAL is the total energy).  If you add fields here
// you should also add the lines in MolInfo.C so you can get access to
// the fields from Tcl.
enum { TSE_BOND, TSE_ANGLE, TSE_DIHE, TSE_IMPR, TSE_VDW, TSE_COUL,
       TSE_HBOND, TSE_KE, TSE_PE, TSE_TEMP, TSE_TOTAL, TSE_VOLUME,
       TSE_PRESSURE, TSE_EFIELD, TSE_UREY_BRADLEY, TSE_RESTRAINT,
       TSENERGIES};

/// Timesteps store coordinates, energies, etc. for one trajectory timestep
class Timestep {
public:
  int num;                  ///< number of atoms this timestep is for
  int page_align_sz;        ///< page alignment size for unbuffered kernel I/O
  float *pos;               ///< atom coords.     unit:Angstroms
  float *pos_ptr;           ///< non-block-aligned pointer to pos buffer
  float *vel;               ///< atom velocites.  unit: 
  float *force;             ///< atom forces.     unit:kcal/mol/A
  float *user;              ///< Demand-allocated 1-float-per-atom 'User' data
  float *user2;             ///< Demand-allocated 1-float-per-atom 'User' data
  float *user3;             ///< Demand-allocated 1-float-per-atom 'User' data
  float *user4;             ///< Demand-allocated 1-float-per-atom 'User' data
  QMTimestep *qm_timestep;  ///< QM timestep data (orbitals, wavefctns, etc)
  float energy[TSENERGIES]; ///< energies for this step. unit:kcal/mol
  int timesteps;            ///< timesteps elapsed so far (if known)
  double physical_time;     ///< physical time for this step. unit:femtoseconds

  /// Size and shape of unit cell 
  float a_length, b_length, c_length, alpha, beta, gamma;

  /// Get vectors corresponding to periodic image vectors
  void get_transform_vectors(float v1[3], float v2[3], float v3[3]) const;
 
  /// Compute transformations from current unit cell dimensions
  void get_transforms(Matrix4 &a, Matrix4 &b, Matrix4 &c) const;

  /// Convert (na, nb, nc) tuple to a transformation based on the current
  /// unit cell.
  void get_transform_from_cell(const int *cell, Matrix4 &trans) const;

  Timestep(int n, int pagealignsz=1); ///< constructor: # atoms, alignment req
  Timestep(const Timestep& ts);       ///< copy constructor
  ~Timestep(void);                    ///< destructor
  
  void zero_values();                 ///< set the coords to 0
};

#endif

