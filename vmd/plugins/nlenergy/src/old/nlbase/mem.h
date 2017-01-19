/*
 * Copyright (C) 2007 by David J. Hardy.  All rights reserved.
 *
 * mem.h - NAMD-Lite low-level memory management routines.
 */

#ifndef NLBASE_MEM_H
#define NLBASE_MEM_H

#ifdef __cplusplus
extern "C" {
#endif

  /* Similar semantics to the C library routines. */
  void *NL_malloc(size_t size);
  void *NL_calloc(size_t nelem, size_t elsize);
  void NL_free(void *ptr);
  void *NL_realloc(void *ptr, size_t size);
  char *NL_strdup(const char *s1);

#ifdef __cplusplus
}
#endif

#endif /* NLBASE_MEM_H */
