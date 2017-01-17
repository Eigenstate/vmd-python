/*
 * Copyright (C) 2008 by David J. Hardy.  All rights reserved.
 */

/**@file    moltypes/energy.h
 * @brief   Container for "energies."
 * @author  David J. Hardy
 * @date    Apr. 2008
 */

#ifndef MOLTYPES_ENERGY_H
#define MOLTYPES_ENERGY_H

#include "nlbase/nlbase.h"

#ifdef __cplusplus
extern "C" {
#endif

  enum {
    VIRIAL_XX = 0,
    VIRIAL_XY = 1,
    VIRIAL_XZ = 2,
    VIRIAL_YX = 1,
    VIRIAL_YY = 3,
    VIRIAL_YZ = 4,
    VIRIAL_ZX = 2,
    VIRIAL_ZY = 4,
    VIRIAL_ZZ = 5,
    NELEMS_VIRIAL = 6
  };

  typedef struct Energy_t {
    dreal pe;
    dreal pe_bond;
    dreal pe_angle;
    dreal pe_dihed;
    dreal pe_impr;
    dreal pe_elec;
    dreal pe_vdw;
    dreal pe_buck;
    dreal pe_boundary;

    dreal ke;

    dreal total;
    dreal total2;
    dreal total3;

    dreal temperature;
    dreal tempavg;
    dreal pressure;
    dreal gpressure;
    dreal pressavg;
    dreal gpressavg;
    dreal volume;

    dvec linmo;
    dvec angmo;

    dreal f_virial[NELEMS_VIRIAL];
    dreal k_virial[NELEMS_VIRIAL];
  } Energy;

  int Energy_init(Energy *);
  void Energy_done(Energy *);

#ifdef __cplusplus
}
#endif

#endif /* MOLTYPES_ENERGY_H */
