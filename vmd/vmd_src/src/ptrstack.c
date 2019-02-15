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
 *      $RCSfile: ptrstack.c,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.3 $      $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Trivial stack implementation for use in eliminating recursion
 *   in molecule graph traversal algorithms.
 *
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "ptrstack.h"

typedef struct {
  int growthrate;
  int size;
  int top;
  void **s;
} ptrstack;

PtrStackHandle ptrstack_create(int size) {
  ptrstack *s;

  s = (ptrstack *) malloc(sizeof(ptrstack));
  if (s == NULL)
    return NULL;

  s->growthrate = 32768;
  s->top = -1;

  if (size > 0) {
    s->size = size;
    s->s = (void **) malloc(s->size * sizeof(void *));
  } else {
    s->size = 0;
    s->s = NULL;
  }

  return s;
}


void ptrstack_destroy(PtrStackHandle voidhandle) {
  ptrstack *s = (ptrstack *) voidhandle;
  free(s->s);
  s->s = NULL; /* prevent access after free */
  free(s);
}


int ptrstack_compact(PtrStackHandle voidhandle) {
  ptrstack *s = (ptrstack *) voidhandle;

  if (s->size > (s->top + 1)) {
    int newsize = s->top + 1;
    void **tmp = (void **) realloc(s->s, newsize * sizeof(void *));
    if (tmp == NULL)
      return -1;
    s->s = tmp;
    s->size = newsize; 
  }

  return 0;
}

int ptrstack_push(PtrStackHandle voidhandle, void *p) {
  ptrstack *s = (ptrstack *) voidhandle;

  s->top++;
  if (s->top >= s->size) {
    int newsize = s->size + s->growthrate; 
    void *tmp = (int *) realloc(s->s, newsize * sizeof(void *));
    if (tmp == NULL) {
      s->top--;
      return -1; /* out of space! */
    }
    s->s = tmp;
    s->size = newsize;
  }  

  s->s[s->top] = p; /* push onto the stack */

  return 0;
}


int ptrstack_pop(PtrStackHandle voidhandle, void **p) {
  ptrstack *s = (ptrstack *) voidhandle;
  if (s->top < 0)
    return -1;

  *p = s->s[s->top];
  s->top--;

  return 0;
}

int ptrstack_popall(PtrStackHandle voidhandle) {
  ptrstack *s = (ptrstack *) voidhandle;
  s->top = -1;

  return 0;
}

int ptrstack_empty(PtrStackHandle voidhandle) {
  ptrstack *s = (ptrstack *) voidhandle;
  if (s->top < 0) return 1;
  else return 0;
}


