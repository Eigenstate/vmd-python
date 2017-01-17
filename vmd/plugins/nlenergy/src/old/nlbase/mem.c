/*
 * Copyright (C) 2007 by David J. Hardy.  All rights reserved.
 *
 * mem.c - NAMD-Lite low-level memory management routines.
 */

#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "nlbase/error.h"
#include "nlbase/io.h"


void *NL_malloc(size_t size) {
  void *p = NULL;
  if (0==size) return NULL;
  errno = 0;
  if ((p = malloc(size))==NULL || errno) {
    (void) ERROR_SYSCALL("malloc()", errno, ERR_MEMALLOC);
    return NULL;
  }
  return p;
}


void *NL_calloc(size_t nelem, size_t elsize) {
  void *p = NULL;
  if (0==nelem || 0==elsize) return NULL;
  errno = 0;
  if ((p = calloc(nelem, elsize))==NULL || errno) {
    (void) ERROR_SYSCALL("calloc()", errno, ERR_MEMALLOC);
    return NULL;
  }
  return p;
}


void NL_free(void *ptr) {
  free(ptr);
}


void *NL_realloc(void *ptr, size_t size) {
  void *pnew = NULL;
  if (0==size) {
    free(ptr);
    return NULL;
  }
  errno = 0;
  if ((pnew = realloc(ptr, size))==NULL || errno) {
    (void) ERROR_SYSCALL("realloc()", errno, ERR_MEMALLOC);
    return NULL;
  }
  return pnew;
}

char *NL_strdup(const char *s1) {
  char *s2 = NULL;
  if (NULL == s1) return NULL;
  errno = 0;
  if ((s2 = strdup(s1))==NULL || errno) {
    (void) ERROR_SYSCALL("strdup()", errno, ERR_MEMALLOC);
    return NULL;
  }
  return s2;
}
