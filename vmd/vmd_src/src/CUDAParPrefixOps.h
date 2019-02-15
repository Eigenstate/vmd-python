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
 *      $RCSfile: CUDAParPrefixOps.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.7 $        $Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   GPU-accelerated parallel prefix operations (sum, min, max etc)
 ***************************************************************************/

#include "ProfileHooks.h"  // needed here for GTC profile tests

// force use of either CUB-based back-end implementation instead of 
// using Thrust, which is the default.  Thrust is shipped with CUDA 
// presently, but CUB as-yet, is not.  Unless we ship CUB with the
// VMD src, we'll need to retain the ability compile either way
// for a while yet. 
#if 0
#define VMDUSECUB 1
#endif

//
// Exclusive prefix sum
//
template <typename T>
long dev_excl_scan_sum_tmpsz(T *in_d, long nitems, T *out_d, T ival);

template <typename T>
void dev_excl_scan_sum(T *in_d, long nitems, T *out_d,
                       void *scanwork_d, long tsz, T ival);


//
// Inclusive prefix sum
//
template <typename T>
long dev_incl_scan_sum_tmpsz(T *in_d, long nitems, T *out_d);

template <typename T>
void dev_incl_scan_sum(T *in_d, long nitems, T *out_d,
                       void *scanwork_d, long tsz);

