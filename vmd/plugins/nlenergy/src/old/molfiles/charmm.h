/*
 * Copyright (C) 2008 by David J. Hardy.  All rights reserved.
 */

/**@file    molfiles/charmm.h
 * @brief   Read CHARMM force field parameters file into ForcePrm container.
 * @author  David J. Hardy
 * @date    Apr. 2008
 */

#ifndef MOLFILES_CHARMM_H
#define MOLFILES_CHARMM_H

#include "moltypes/moltypes.h"

#ifdef __cplusplus
extern "C" {
#endif

  int ForcePrm_read_charmm(ForcePrm *, const char *fname);

#ifdef __cplusplus
}
#endif

#endif /* MOLFILES_CHARMM_H */
