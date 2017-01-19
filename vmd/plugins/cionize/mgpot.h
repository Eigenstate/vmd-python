/*
 * Copyright (C) 2007 by David J. Hardy.  All rights reserved.
 *
 * mgpot.h
 */

#ifndef MGPOT_H
#define MGPOT_H

#ifdef __cplusplus
extern "C" {
#endif

  /* entry point for computing grid energies using mgpot */
  int calc_grid_energies_excl_mgpot(float* atoms, float* grideners,
      long int numplane, long int numcol, long int numpt, long int natoms,
      float gridspacing, unsigned char* excludepos, int maxnumprocs,
      int emethod);

#ifdef __cplusplus
}
#endif

#endif /* MGPOT_H */
