/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2016 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: vmddlopen.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.11 $      $Date: 2016/11/28 03:05:08 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Routines for loading dynamic link libraries and shared object files
 *   on various platforms, abstracting from machine dependent APIs.
 *
 * LICENSE:
 *   UIUC Open Source License 
 *   http://www.ks.uiuc.edu/Research/vmd/plugins/pluginlicense.html
 *
 ***************************************************************************/

/*
 * vmddlopen: thin multi-platform wrapper around dlopen/LoadLibrary
 */

#ifndef VMD_DLOPEN__

#ifdef __cplusplus
extern "C" {
#endif

/* Try to open the specified library.  All symbols must be resolved or the 
 * load will fail (RTLD_NOW).  
 */
void *vmddlopen(const char *fname);

/* Try to load the specified symbol using the given handle.  Returns NULL if 
 * the symbol cannot be loaded.
 */
void *vmddlsym(void *h, const char *sym);

/* Unload the library.  Return 0 on success, nonzero on error. 
 */
int vmddlclose(void *h);

/* Return last error from any of the above functions.  Not thread-safe on
 * Windows due to static buffer in our code. 
 */ 
const char *vmddlerror(void);

#ifdef __cplusplus
}
#endif

#endif

