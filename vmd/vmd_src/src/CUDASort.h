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
 *      $RCSfile: CUDASort.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.5 $        $Date: 2019/01/17 21:20:58 $
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
// Ascending key-value radix sort 
//
template <typename KeyT, typename ValT>
long dev_radix_sort_by_key_tmpsz(KeyT *keys_d, ValT *vals_d, long nitems);

template <typename KeyT, typename ValT>
int dev_radix_sort_by_key(KeyT *keys_d, ValT *vals_d, long nitems,
                          KeyT *keyswork_d, ValT *valswork_d,
                          void *sortwork_d, long tsz, 
                          KeyT min_key, KeyT max_key);


