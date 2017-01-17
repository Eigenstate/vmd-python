/*
 * Copyright (C) 2008 by David J. Hardy.  All rights reserved.
 */

/**@file    molfiles/psf.h
 * @brief   Read PSF into Topology data container.
 * @author  David J. Hardy
 * @date    Aug. 2007
 */

#ifndef MOLFILES_PSF_H
#define MOLFILES_PSF_H

#include "moltypes/moltypes.h"

#ifdef __cplusplus
extern "C" {
#endif

  int Topology_read_psf(Topology *, const char *fname);

#ifdef __cplusplus
}
#endif

#endif /* MOLFILES_PSF_H */
