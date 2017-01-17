/*
 * Copyright (C) 2008 by David J. Hardy.  All rights reserved.
 *
 * array.h - Provides generic resize arrays.
 */

#ifndef NLBASE_ARRAY_H
#define NLBASE_ARRAY_H

#include "nlbase/types.h"

#ifdef __cplusplus
extern "C" {
#endif

  /**@ Generic resize array container.
   *
   * Invariants:  0 < elemsz
   *              0 <= uselen <= buflen
   *              size of allocation = (buflen * elemsz) bytes
   *              buffer == NULL if and only if buflen == 0
   *
   * Methods returning "int" indicate success by OK or failure by some
   * related error code (from error.h).
   *
   * Append and remove, respectively, grow and shrink the used array by one
   * element on the end, enabling a stack.  The memory allocation/deallocation
   * has O(1) amortized cost.
   */
  typedef struct Array_t {
    void *buffer;    /**< points to the memory allocation */
    size_t buflen;   /**< actual length of memory allocation in elemnts */
    size_t uselen;   /**< number of elements in use */
    size_t elemsz;   /**< size of an element in bytes */
  } Array;

  int Array_init(Array *, size_t elemsz);         /* constructor */
  void Array_done(Array *);                       /* destructor */
  int Array_copy(Array *dest, const Array *src);  /* deep copy */

  int Array_setbuflen(Array *, size_t buflen);
  int Array_resize(Array *, size_t uselen);
  int Array_append(Array *, const void *pelem);
  int Array_remove(Array *, void *pelem);

  int Array_erase(Array *);     /* erase all entries */
  void Array_unalias(Array *);  /* reset internals (for aliased array) */

  void *Array_data(Array *);            /* return buffer */
  const void *Array_data_const(const Array *);
  int32 Array_length(const Array *);    /* return uselen */
  void *Array_elem(Array *, size_t i);  /* range-checked, 0 <= i < uselen */
  const void *Array_elem_const(const Array *, size_t i);


  /**@ Object array needs constructor/destructor/copy for elements.
   *
   * Otherwise, same as above.
   */
  typedef struct Objarr_t {
    void *buffer;
    size_t buflen;
    size_t uselen;
    size_t elemsz;
    int (*elem_init)(void *pelem);
    void (*elem_done)(void *pelem);
    int (*elem_copy)(void *dest, const void *src);
    int (*elem_erase)(void *pelem);
  } Objarr;

  int Objarr_init(Objarr *, size_t elemsz,
      int (*elem_init)(void *),
      void (*elem_done)(void *),
      int (*elem_copy)(void *, const void *),
      int (*elem_erase)(void *));
  void Objarr_done(Objarr *);
  int Objarr_copy(Objarr *dest, const Objarr *src);

  int Objarr_setbuflen(Objarr *, size_t buflen);
  int Objarr_resize(Objarr *, size_t uselen);
  int Objarr_append(Objarr *, const void *pelem);
  int Objarr_remove(Objarr *, void *pelem);

  int Objarr_erase(Objarr *);     /* erase all entries */
  void Objarr_unalias(Objarr *);  /* reset internals (for aliased array) */

  void *Objarr_data(Objarr *);            /* return buffer */
  const void *Objarr_data_const(const Objarr *);
  int32 Objarr_length(const Objarr *);    /* return uselen */
  void *Objarr_elem(Objarr *, size_t i);  /* range-checked, 0 <= i < uselen */
  const void *Objarr_elem_const(const Objarr *, size_t i);

#ifdef __cplusplus
}
#endif

#endif /* NLBASE_ARRAY_H */
