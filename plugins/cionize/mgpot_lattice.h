/*
 * Copyright (C) 2007 by David J. Hardy.  All rights reserved.
 *
 * mgpot_lattice.h
 *
 * Manage and access storage for generic 3D lattice.  Indicate index
 * ranges ia..ib, ja..jb, ka..kb, allowing negative subscripting.
 *
 * Storage in column-major order like Fortran, access memory sequentially
 * by looping over k in outer loop, j in middle loop, i in inner loop:
 *
 *     for k in ka..kb {
 *       for j in ja..jb {
 *         for i in ia..ib {
 *           access lattice element (i,j,k)
 *         }
 *       }
 *     }
 */

#ifndef MGPOT_LATTICE_H
#define MGPOT_LATTICE_H

#include <stdlib.h>
#include <string.h>
#include "mgpot_error.h"

#ifdef __cplusplus
extern "C" {
#endif


  /**@
   * Generate typeLattice class definition.  (needs closing ';')
   */
#define LATTICE_CLASS(T) \
  typedef struct T##Lattice_t { \
  /* private data */ \
    T *buffer;            /**< start of allocated data buffer */ \
    long int nelems;      /**< total number of elements (ni*nj*nk) */ \
  /* public data - (treat as read-only) */ \
    long int ia, ib, ni;  /**< index range and length along i dimension */ \
    long int ja, jb, nj;  /**< index range and length along j dimension */ \
    long int ka, kb, nk;  /**< index range and length along k dimension */ \
  /* public methods */ \
    void (*zero)(struct T##Lattice_t *); \
      /**< sets all lattice elements to zero */ \
    T *(*elem)(struct T##Lattice_t *, long int i, long int j, long int k); \
      /**< returns pointer to lattice element (range-checked) */ \
    T *(*data)(struct T##Lattice_t *); \
      /**< returns pointer to transformed buffer for typical flat access,
       * e.g. g = L->data(L), g[k*nj*ni + j*ni + i] for ia<=i<=ib, etc. */ \
    long int (*index)(struct T##Lattice_t *, \
        long int i, long int j, long int k); \
      /**< returns flat index into data() (range-checked) \
       * - good to use within ASSERT() for debugging */ \
  } T##Lattice; \
  T##Lattice *new_##T##Lattice(long int ia, long int ib, \
      long int ja, long int jb, long int ka, long int kb); \
  void delete_##T##Lattice(T##Lattice *)
  /* end LATTICE_CLASS */


  /**@
   * Generate corresponding typeLattice class methods.
   * Should be expanded in .c file.  Only functions
   * new_typeLattice() and delete_typeLattice() have external linkage.
   * (does not need closing ';')
   */
#define LATTICE_CLASS_METHODS(T) \
  static void zero_##T##Lattice(T##Lattice *L) { \
    memset(L->buffer, 0, L->nelems * sizeof(T)); \
  } \
  static T *data_##T##Lattice(T##Lattice *L) { \
    return L->buffer + ((-L->ka * L->nj - L->ja) * L->ni - L->ia); \
  } \
  static long int index_##T##Lattice(T##Lattice *L, \
      long int i, long int j, long int k) { \
    if (i < L->ia || i > L->ib) { \
      ERROR("index i=%d outside of range %d..%d\n", i, L->ia, L->ib); \
      return FAIL; /* bad error indicator, since negative index is possible */\
    } \
    if (j < L->ja || j > L->jb) { \
      ERROR("index j=%d outside of range %d..%d\n", j, L->ja, L->jb); \
      return FAIL; \
    } \
    if (k < L->ka || k > L->kb) { \
      ERROR("index k=%d outside of range %d..%d\n", k, L->ka, L->kb); \
      return FAIL; \
    } \
    return ((k * L->nj + j) * L->ni + i); \
  } \
  static T *elem_##T##Lattice(T##Lattice *L, \
      long int i, long int j, long int k) { \
    return data_##T##Lattice(L) + index_##T##Lattice(L,i,j,k); \
  } \
  T##Lattice *new_##T##Lattice(long int ia, long int ib, \
      long int ja, long int jb, long int ka, long int kb) { \
    T##Lattice *L; \
    ASSERT(ia <= ib && ja <= jb && ka <= kb); \
    L = (T##Lattice *) calloc(1, sizeof(T##Lattice)); \
    if (NULL == L) { \
      ERROR("can\'t calloc() memory for %s\n", #T "Lattice"); \
      return NULL; \
    } \
    else { \
      L->ia = ia; \
      L->ib = ib; \
      L->ja = ja; \
      L->jb = jb; \
      L->ka = ka; \
      L->kb = kb; \
      L->ni = ib - ia + 1; \
      L->nj = jb - ja + 1; \
      L->nk = kb - ka + 1; \
      L->nelems = L->ni * L->nj * L->nk; \
      L->buffer = (T *) calloc(L->nelems, sizeof(T)); \
      if (NULL == L->buffer) { \
        ERROR("can\'t calloc() %d elements for %s\n", L->nelems, #T "Lattice");\
        return NULL; \
      } \
      L->zero = zero_##T##Lattice; \
      L->elem = elem_##T##Lattice; \
      L->data = data_##T##Lattice; \
      L->index = index_##T##Lattice; \
    } \
    return L; \
  } \
  void delete_##T##Lattice(T##Lattice *L) { \
    if (L) { \
      free(L->buffer); \
    } \
    free(L); \
  }
  /* end LATTICE_CLASS_METHODS */


/*
 * Type definitions and prototypes for lattice classes are expanded here.
 * The corresponding methods are expanded in .c file.
 */
  LATTICE_CLASS(float);


#ifdef __cplusplus
}
#endif

#endif /* MGPOT_LATTICE_H */
