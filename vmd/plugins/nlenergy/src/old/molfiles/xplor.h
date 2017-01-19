/*
 * Copyright (C) 2008 by David J. Hardy.  All rights reserved.
 */

/**@file    molfiles/xplor.h
 * @brief   Read X-Plor force field parameters file into ForcePrm container.
 * @author  David J. Hardy
 * @date    Apr. 2008
 */

#ifndef MOLFILES_XPLOR_H
#define MOLFILES_XPLOR_H

#include "moltypes/moltypes.h"

#ifdef __cplusplus
extern "C" {
#endif

  int ForcePrm_read_xplor(ForcePrm *, const char *fname);

#ifdef __cplusplus
}
#endif

#endif /* MOLFILES_XPLOR_H */
