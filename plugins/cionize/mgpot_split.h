/*
 * Copyright (C) 2007 by David J. Hardy.  All rights reserved.
 *
 * mgpot_split.h - evaluates polynomial part of parameterized splittings
 */

#ifndef MGPOT_SPLIT_H
#define MGPOT_SPLIT_H

#include "mgpot_error.h"

#ifdef __cplusplus
extern "C" {
#endif


  /* identifies different splittings */
  enum MgpotSplit_t {
    SPLIT_NONE=0,
    TAYLOR2,
    TAYLOR3,
    SPLIT_MAX
  };

  /*
   * G_NORMAL() computes the smoothed polynomial part of the
   * splitting functions, i.e. g_1((r/a)**2).
   *
   *   pg - float* with return value
   *   s - (ra/)**2, assumed to be between 0 and 1
   *   split - identifies which splitting function
   */
#define G_NORMAL(pg, s, split) \
  do { \
    double _s = s;  /* where s=(r/a)**2 */ \
    double _g = 0; \
    ASSERT(0 <= _s && _s <= 1); \
    switch (split) { \
      case TAYLOR2: \
	_g = 1 + (_s-1)*(-1./2 + (_s-1)*(3./8)); \
	break; \
      case TAYLOR3: \
	_g = 1 + (_s-1)*(-1./2 + (_s-1)*(3./8 + (_s-1)*(-5./16))); \
	break; \
      default: \
	ERROR("unknown splitting\n"); \
	return FAIL; \
    } \
    *(pg) = _g; \
  } while (0)
  /* closing ';' from use as function call */


#ifdef DEBUGGING
  /* when debugging, make this an actual function call */
  static int mgpot_split(float *gs, float s, int split) {
    G_NORMAL(gs, s, split);
  }
#else
  /* otherwise, expand in code as a macro */
#define mgpot_split  G_NORMAL
#endif


#ifdef __cplusplus
}
#endif

#endif /* MGPOT_SPLIT_H */
