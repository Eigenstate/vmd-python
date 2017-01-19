/*
 * Copyright (C) 2008 by David J. Hardy.  All rights reserved.
 *
 * arrmap.h - Provides generic mapping of an array through user-provided
 *   functions for field access, hashing, and equality check on keys.
 */

#ifndef NLBASE_ARRMAP_H
#define NLBASE_ARRMAP_H

#include "nlbase/types.h"
#include "nlbase/array.h"

#ifdef __cplusplus
extern "C" {
#endif

  /**@brief Arrmap container class.
   *
   * Maps array elements using a user-specified field from each element
   * and user-defined hash function.
   *
   * Similar in functionality to Strmap container, except that, instead of
   * explicitly passing keys and values, the key is taken from the ith
   * array element and the associated value is the index i.  This permits
   * the array to be alternatively indexed using the user-specified field
   * through the Arrmap_lookup() function.
   *
   * Note that Strmap cannot be attached to an array of static fields that
   * is then resized, because the pointers stored in the table might be
   * invalidated.  In this implementation, we store indices into the array
   * and a pointer to the array, so we can always recover the search keys,
   * even if the user resizes the array.  It is up to the user to leave
   * array elements untouched once they have been inserted into the hash
   * table.
   */
  typedef struct Arrmap_t {
    const Array *array;  /**< we don't modify the array, but user still can */
    const void *(*key)(const Array *, int32 i);
                         /**< access the key field from array element i */
    int32 (*hash)(const struct Arrmap_t *, const void *key);
                         /**< hashing function mapping keys to nonnegative
                          * integers, used as index into bucket array */
    int32 (*keycmp)(const void *key1, const void *key2);
                         /**< test equality of keys, returns 0 for equality
                          * just like strcmp(), otherwise returns nonzero */
    struct Arrmap_node_t **bucket;  /**< array of hash nodes */
    int32 size;                     /**< size of the array */
    int32 entries;                  /**< number of entries in hash table */
    int32 downshift;                /**< shift count, used for hashing */
    int32 mask;                     /**< used to select bits for hashing */
    char buf[64];                   /**< buffer for stats */
  } Arrmap;

  int Arrmap_init(Arrmap *, const Array *,
      const void *(*key)(const Array *, int32 i),
      int32 (*hash)(const Arrmap *, const void *key),
      int32 (*keycmp)(const void *key1, const void *key2),
      int32 nbuckets);
  void Arrmap_done(Arrmap *);

  int32 Arrmap_insert(Arrmap *, int32 i);
  int32 Arrmap_lookup(const Arrmap *, const void *key);
  int32 Arrmap_update(Arrmap *, int32 i);
  int32 Arrmap_remove(Arrmap *, int32 i);

  const char *Arrmap_stats(Arrmap *);

#ifdef __cplusplus
}
#endif

#endif /* NLBASE_ARRMAP_H */
