/*
 * Copyright (C) 2007 by David J. Hardy.  All rights reserved.
 *
 * mgpot_longrng.c - compute long-range contribution to potentials
 */

#include "mgpot_defn.h"

int mgpot_longrng(Mgpot *mg) {
  switch (mg->interp) {
    case CUBIC:
      if (mgpot_longrng_cubic(mg)) {
        return ERROR("mgpot_longrng_cubic() failed\n");
      }
      break;
    case QUINTIC1:
      if (mgpot_longrng_quintic1(mg)) {
        return ERROR("mgpot_longrng_quintic1() failed\n");
      }
      break;
    default:
      return ERROR("unknown splitting\n");
  }
  return 0;
}


int mgpot_longrng_finish(Mgpot *mg) {
  if (mg->separate_longrng) {
    const float *longrng = mg->grideners_longrng;
    float *epotmap = mg->grideners;
    long int n = mg->numplane * mg->numcol * mg->numpt;
    long int i;
    for (i = 0;  i < n;  i++) {
      epotmap[i] += longrng[i];
    }
  }
  return 0;
}
