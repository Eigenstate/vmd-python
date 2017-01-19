/*
 * Copyright (C) 2008 by David J. Hardy.  All rights reserved.
 *
 * numeric.h - NAMD-Lite semi-numeric functions.
 */

#ifndef NLBASE_NUMERIC_H
#define NLBASE_NUMERIC_H

#include "nlbase/types.h"

#ifdef __cplusplus
extern "C" {
#endif

  /**@brief Return greatest common divisor. */
  uint32 Numeric_gcd(uint32 a, uint32 b);

#ifdef __cplusplus
}
#endif

#endif /* NLBASE_NUMERIC_H */
