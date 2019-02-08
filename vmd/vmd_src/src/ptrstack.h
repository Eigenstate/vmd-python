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
 *      $RCSfile: ptrstack.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.3 $      $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Trivial stack implementation for use in eliminating recursion
 *   in molecule graph traversal algorithms.
 *
 ***************************************************************************/

#ifdef __cplusplus
extern "C" {
#endif

typedef void * PtrStackHandle;

PtrStackHandle ptrstack_create(int size);
void ptrstack_destroy(PtrStackHandle voidhandle);
int ptrstack_compact(PtrStackHandle voidhandle);
int ptrstack_push(PtrStackHandle voidhandle, void *p);
int ptrstack_pop(PtrStackHandle voidhandle, void **p);
int ptrstack_popall(PtrStackHandle voidhandle);
int ptrstack_empty(PtrStackHandle voidhandle);

#ifdef __cplusplus
}
#endif

