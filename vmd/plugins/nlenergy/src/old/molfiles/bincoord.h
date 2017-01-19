/*
 * Copyright (C) 2008 by David J. Hardy.  All rights reserved.
 */

/**@file    molfiles/bincoord.h
 * @brief   Read and write binary coordinate files.
 * @author  David J. Hardy
 * @date    Apr. 2008
 */

#ifndef MOLFILES_BINCOORD_H
#define MOLFILES_BINCOORD_H

#include "moltypes/moltypes.h"

#ifdef __cplusplus
extern "C" {
#endif

  typedef enum BinCoordType_t {
    BINCOORD_NONE    = 0,  /**< no unit conversion */
    BINCOORD_POS     = 0,  /**< Angstrom file, Angstrom array */
    BINCOORD_VEL     = 0,  /**< A/fs file, A/fs array */
    BINCOORD_NAMDPOS = 0,  /**< A file, A array */
    BINCOORD_NAMDVEL = 1,  /**< sqrt(kcal/mol/AMU) file, A/fs array */
    /*
     * Explanation:
     *
     * NAMD internal velocity units are sqrt(kcal/mol/AMU).
     * This comes about by scaling the fs time step dt by a
     * time factor T.  Units are chosen for T so that
     * multiplication of the force/mass = (kcal/mol/A) / AMU
     * by (dt/T)^2 gives position units A.
     *
     * http://www.ks.uiuc.edu/Research/namd/mailing_list/namd-l/4125.html
     */
  } BinCoordType;


  int Bincoord_write(const dvec *a, int32 natoms, BinCoordType atype,
      const char *fname);

  int Bincoord_read(dvec *a, int32 natoms, BinCoordType atype,
      const char *fname);

  int32 Bincoord_read_numatoms(const char *fname);


#ifdef __cplusplus
}
#endif

#endif /* MOLFILES_BINCOORD_H */
