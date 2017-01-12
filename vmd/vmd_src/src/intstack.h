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
 *      $RCSfile: intstack.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.8 $      $Date: 2016/11/28 03:05:07 $
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

typedef void * IntStackHandle;

IntStackHandle intstack_create(int long);
void intstack_destroy(IntStackHandle voidhandle);
int intstack_compact(IntStackHandle voidhandle);
int intstack_push(IntStackHandle voidhandle, int i);
int intstack_pop(IntStackHandle voidhandle, int *i);
int intstack_popall(IntStackHandle voidhandle);
int intstack_empty(IntStackHandle voidhandle);

#ifdef __cplusplus
}
#endif

