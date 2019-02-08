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
 *      $RCSfile: intstack.c,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.10 $      $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Trivial stack implementation for use in eliminating recursion
 *   in molecule graph traversal algorithms.
 *
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "intstack.h"

typedef struct {
  long growthrate;
  long size;
  long top;
  int *s;
} intstack;

IntStackHandle intstack_create(long size) {
  intstack *s;

  s = (intstack *) malloc(sizeof(intstack));
  if (s == NULL)
    return NULL;

  s->growthrate = 32768;
  s->top = -1;

  if (size > 0) {
    s->size = size;
    s->s = (int *) malloc(s->size * sizeof(int));
  } else {
    s->size = 0;
    s->s = NULL;
  }

  return s;
}


void intstack_destroy(IntStackHandle voidhandle) {
  intstack *s = (intstack *) voidhandle;
  free(s->s);
  s->s = NULL; /* prevent access after free */
  free(s);
}


int intstack_compact(IntStackHandle voidhandle) {
  intstack *s = (intstack *) voidhandle;

  if (s->size > (s->top + 1)) {
    long newsize = s->top + 1L;
    int *tmp = (int *) realloc(s->s, newsize * sizeof(int));
    if (tmp == NULL)
      return -1;
    s->s = tmp;
    s->size = newsize; 
  }

  return 0;
}

int intstack_push(IntStackHandle voidhandle, int i) {
  intstack *s = (intstack *) voidhandle;

  s->top++;
  if (s->top >= s->size) {
    long newsize = s->size + s->growthrate; 
    int *tmp = (int *) realloc(s->s, newsize * sizeof(int));
    if (tmp == NULL) {
      s->top--;
      return -1; /* out of space! */
    }
    s->s = tmp;
    s->size = newsize;
  }  

  s->s[s->top] = i; /* push onto the stack */

  return 0;
}


int intstack_pop(IntStackHandle voidhandle, int *i) {
  intstack *s = (intstack *) voidhandle;
  if (s->top < 0)
    return -1;

  *i = s->s[s->top];
  s->top--;

  return 0;
}

int intstack_popall(IntStackHandle voidhandle) {
  intstack *s = (intstack *) voidhandle;
  s->top = -1;

  return 0;
}

int intstack_empty(IntStackHandle voidhandle) {
  intstack *s = (intstack *) voidhandle;
  if (s->top < 0) return 1;
  else return 0;
}

#if 0

#include <stdio.h>

int main() {
  int i;
  IntStackHandle stack;

printf("allocating stack...\n");
  stack = intstack_create(0);

printf("pushing data values onto the stack...\n");
  intstack_push(stack, 5);
  intstack_compact(stack);
  intstack_push(stack, 3);
  intstack_compact(stack);
  intstack_push(stack, 5);
  intstack_compact(stack);
  intstack_push(stack, 2);
  intstack_compact(stack);
  intstack_push(stack, 9);
  intstack_compact(stack);
  intstack_push(stack, 5);
  intstack_compact(stack);
  intstack_push(stack, 1);
  intstack_compact(stack);
  intstack_push(stack, 4);
  intstack_compact(stack);
  intstack_push(stack, 1);
  intstack_compact(stack);
  intstack_push(stack, 3);

printf("popping data values off the stack...\n");
  while (!intstack_pop(stack, &i)) {
    printf("%d\n", i);
  }

  intstack_destroy(stack);
 
  return 0;
}

#endif

