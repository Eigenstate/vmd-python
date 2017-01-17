/*
 * Copyright (C) 2008 by David J. Hardy.  All rights reserved.
 *
 * array.c - Provides generic resize arrays.
 */

#include <string.h>
#include "nlbase/error.h"
#include "nlbase/io.h"
#include "nlbase/mem.h"
#include "nlbase/array.h"

int Array_init(Array *a, size_t elemsz) {
  ASSERT(a != NULL);
  memset(a, 0, sizeof(Array));
  a->elemsz = elemsz;
  return OK;
}

void Array_done(Array *a) {
  Array_setbuflen(a, 0);
}

int Array_copy(Array *dest, const Array *src) {
  int s;  /* error status */
  if (dest->elemsz != src->elemsz) return ERROR(ERR_VALUE);
  if ((s=Array_setbuflen(dest, src->buflen)) != OK) return ERROR(s);
  dest->uselen = src->uselen;
  memcpy(dest->buffer, src->buffer, src->uselen * src->elemsz);
  return OK;
}

int Array_setbuflen(Array *a, size_t buflen) {
  if (0 == buflen) {
    NL_free(a->buffer);
    a->buffer = NULL;
    a->buflen = 0;
  }
  else {
    void *tmp = NL_realloc(a->buffer, buflen * a->elemsz);
    if (NULL == tmp) return ERROR(ERR_MEMALLOC);
    a->buffer = tmp;
    a->buflen = buflen;
  }
  if (a->uselen > a->buflen) a->uselen = a->buflen;
  return OK;
}

int Array_resize(Array *a, size_t uselen) {
  int s;  /* error status */
  if (uselen > a->buflen && (s=Array_setbuflen(a, uselen)) != OK) {
    return ERROR(s);
  }
  a->uselen = uselen;
  ASSERT(a->uselen <= a->buflen);
  return OK;
}

int Array_append(Array *a, const void *pelem) {
  int s;  /* error status */
  if (0 == a->buflen && (s=Array_setbuflen(a, 1)) != OK) {
    return ERROR(s);
  }
  else if (a->uselen == a->buflen
      && (s=Array_setbuflen(a, a->buflen << 1)) != OK) {
    return ERROR(s);
  }
  ASSERT(a->uselen < a->buflen);
  ASSERT(pelem != NULL);
  memcpy( ((char *) a->buffer) + a->uselen * a->elemsz, pelem, a->elemsz);
  a->uselen++;
  return OK;
}

int Array_remove(Array *a, void *pelem) {
  int s;  /* error status */
  if (0 == a->uselen) return ERROR(ERR_RANGE);  /* empty array */
  if (pelem) {
    memcpy(pelem, ((char *) a->buffer) + (a->uselen-1)*a->elemsz, a->elemsz);
  }
  if (a->uselen <= (a->buflen >> 2)
      && (s=Array_setbuflen(a, a->buflen >> 1)) != OK) {
    return ERROR(s);
  }
  a->uselen--;
  return OK;
}

int Array_erase(Array *a) {
  memset(a->buffer, 0, a->buflen * a->elemsz);
  return OK;
}

void Array_unalias(Array *a) {
  a->buffer = NULL;
  a->buflen = 0;
  a->uselen = 0;
}

void *Array_data(Array *a) { return a->buffer; }

const void *Array_data_const(const Array *a) { return a->buffer; }

int32 Array_length(const Array *a) { return a->uselen; }

void *Array_elem(Array *a, size_t i) {
  if (i >= a->uselen) {
    (void) ERROR(ERR_RANGE);
    return NULL;
  }
  return (void *) (((char *) a->buffer) + i * a->elemsz);
}

const void *Array_elem_const(const Array *a, size_t i) {
  if (i >= a->uselen) {
    (void) ERROR(ERR_RANGE);
    return NULL;
  }
  return (void *) (((char *) a->buffer) + i * a->elemsz);
}


int Objarr_init(Objarr *a, size_t elemsz,
    int (*elem_init)(void *),
    void (*elem_done)(void *),
    int (*elem_copy)(void *, const void *),
    int (*elem_erase)(void *)) {
  ASSERT(a != NULL);
  memset(a, 0, sizeof(Objarr));
  a->elemsz = elemsz;
  a->elem_init = elem_init;
  a->elem_done = elem_done;
  a->elem_copy = elem_copy;
  a->elem_erase = elem_erase;
  return OK;
}

void Objarr_done(Objarr *a) {
  Objarr_setbuflen(a, 0);
}

int Objarr_copy(Objarr *dest, const Objarr *src) {
  int s;  /* error status */
  size_t i;
  if (dest->elemsz != src->elemsz) return ERROR(ERR_VALUE);
  if ((s=Objarr_setbuflen(dest, src->buflen)) != OK) return ERROR(s);
  dest->uselen = src->uselen;
  for (i = 0;  i < src->uselen;  i++) {
    if ((s=src->elem_copy( ((char *)dest->buffer) + i*src->elemsz,
            ((const char *)src->buffer) + i*src->elemsz)) != OK) {
      return ERROR(s);
    }
  }
  return OK;
}

int Objarr_setbuflen(Objarr *a, size_t buflen) {
  int s = OK;  /* error status */
  if (0 == buflen) {
    NL_free(a->buffer);
    a->buffer = NULL;
    a->buflen = 0;
  }
  else {
    void *tmp = NULL;
    size_t i;
    for (i = buflen;  i < a->buflen;  i++) {
      /* destroy elements new buflen up to old buflen */
      a->elem_done( ((char *) a->buffer) + i*a->elemsz);
    }
    tmp = NL_realloc(a->buffer, buflen * a->elemsz);
    if (NULL == tmp) return ERROR(ERR_MEMALLOC);
    a->buffer = tmp;
    for (i = a->buflen;  i < buflen;  i++) {
      /* create elements old buflen up to new buflen */
      if ((s=a->elem_init( ((char *) a->buffer) + i*a->elemsz)) != OK) {
        (void) ERROR(s);
        break;
      }
    }
    a->buflen = buflen;
  }
  if (a->uselen > a->buflen) a->uselen = a->buflen;
  return s;
}

int Objarr_resize(Objarr *a, size_t uselen) {
  int s;  /* error status */
  if (uselen > a->buflen && (s=Objarr_setbuflen(a, uselen)) != OK) {
    return ERROR(s);
  }
  a->uselen = uselen;
  ASSERT(a->uselen <= a->buflen);
  return OK;
}

int Objarr_append(Objarr *a, const void *pelem) {
  int s;  /* error status */
  if (0 == a->buflen && (s=Objarr_setbuflen(a, 1)) != OK) {
    return ERROR(s);
  }
  else if (a->uselen == a->buflen
      && (s=Objarr_setbuflen(a, a->buflen << 1)) != OK) {
    return ERROR(s);
  }
  ASSERT(a->uselen < a->buflen);
  ASSERT(pelem != NULL);
  if ((s=a->elem_copy(((char *)a->buffer)+a->uselen*a->elemsz, pelem)) != OK) {
    return ERROR(s);
  }
  a->uselen++;
  return OK;
}

int Objarr_remove(Objarr *a, void *pelem) {
  int s = OK;
  if (0 == a->uselen) return ERROR(ERR_RANGE);  /* emtpy array */
  if (pelem && (s=a->elem_copy(pelem, ((char *)a->buffer)+a->uselen-1)) != OK) {
    return ERROR(s);
  }
  if (a->uselen <= (a->buflen >> 2)
      && (s=Objarr_setbuflen(a, a->buflen >> 1)) != OK) {
    return ERROR(s);
  }
  a->uselen--;
  return OK;
}

int Objarr_erase(Objarr *a) {
  int s;  /* error status */
  size_t i;
  for (i = 0;  i < a->buflen;  i++) {
    if ((s=a->elem_erase( ((char *) a->buffer) + i*a->elemsz)) != OK) {
      return ERROR(s);
    }
  }
  return OK;
}

void Objarr_unalias(Objarr *a) {
  a->buffer = NULL;
  a->buflen = 0;
  a->uselen = 0;
}

void *Objarr_data(Objarr *a) { return a->buffer; }

const void *Objarr_data_const(const Objarr *a) { return a->buffer; }

int32 Objarr_length(const Objarr *a) { return a->uselen; }

void *Objarr_elem(Objarr *a, size_t i) {
  if (i >= a->uselen) {
    (void) ERROR(ERR_RANGE);
    return NULL;
  }
  return (void *) (((char *) a->buffer) + i * a->elemsz);
}

const void *Objarr_elem_const(const Objarr *a, size_t i) {
  if (i >= a->uselen) {
    (void) ERROR(ERR_RANGE);
    return NULL;
  }
  return (void *) (((char *) a->buffer) + i * a->elemsz);
}
