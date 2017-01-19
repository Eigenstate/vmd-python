/*
 * Copyright (C) 2007 by David J. Hardy.  All rights reserved.
 *
 * test_array.c - Demonstrate use of array class.
 */

#include "nlbase/nlbase.h"

int info(Array *a, const char *name) {
  NL_printf("info \"%s\":", name);
  NL_printf("  uselen=%lu", (unsigned long) a->uselen);
  NL_printf("  buflen=%lu", (unsigned long) a->buflen);
  NL_printf("\n");
  return OK;
}

int print(Array *a, const char *name) {
  int32 *p = Array_data(a);
  int32 max = Array_length(a);
  int32 i;
  NL_printf("data \"%s\":", name);
  for (i = 0;  i < max;  i++) {
    NL_printf(" %d", p[i]);
  }
  NL_printf("\n");
  return 0;
}

int main(void) {
#define MAXLEN 10
  Array a;
  int32 *p;
  int32 i;
  int status = OK;

  if ((status = Array_init(&a, sizeof(int32))) != OK) {
    return ERROR(status);
  }
  info(&a, "a");
  print(&a, "a");
  for (i = 0;  i < MAXLEN;  i++) {
    Array_append(&a, &i);
    info(&a, "a");
    print(&a, "a");
  }

  while (Array_length(&a) > 0) {
    Array_remove(&a, NULL);
    info(&a, "a");
    print(&a, "a");
  }
  Array_remove(&a, NULL);  /* should print error message! */
  info(&a, "a");
  print(&a, "a");

  Array_resize(&a, MAXLEN);
  p = Array_data(&a);
  for (i = 0;  i < Array_length(&a);  i++) {
    p[i] = 2*(i+1);
  }
  info(&a, "a");
  print(&a, "a");

  Array_done(&a);

  return 0;
}
