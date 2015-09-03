/**
***  Copyright (c) 1995, 1996, 1997, 1998, 1999, 2000 by
***  The Board of Trustees of the University of Illinois.
***  All rights reserved.
**/

/* $Id: pub3dfft.h,v 1.2 2005/07/20 15:37:39 johns Exp $ */

#ifndef PUB3DFFT_H
#define PUB3DFFT_H

typedef struct { float r, i; } floatcomplex;

/* ntable should be 4*max(n1,n2,n3) +15 */
/* size of table should be 3*ntable floats */
/* size of work should be 2*max(n1,n2,n3) floats */

int pubz3di(int *n1, int *n2, int *n3, float *table, int *ntable);

int pubz3d(int *isign, int *n1, int *n2,
   int *n3, floatcomplex *w, int *ld1, int *ld2, float
   *table, int *ntable, floatcomplex *work);

/* for real to complex n1 and ld1 must be even */

int pubd3di(int n1, int n2, int n3, float *table, int ntable);

int pubdz3d(int isign, int n1, int n2,
   int n3, float *w, int ld1, int ld2, float
   *table, int ntable, float *work);

int pubzd3d(int isign, int n1, int n2,
   int n3, float *w, int ld1, int ld2, float
   *table, int ntable, float *work);

#endif

