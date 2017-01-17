/*
 * Copyright (C) 2007 by David J. Hardy.  All rights reserved.
 */

/**@file    base/strmap.h
 * @brief   Mapping of strings to integers (i.e. array indices).
 * @author  John Stone and David J. Hardy
 * @date    2007
 *
 * The @c Strmap class provides a hash table that maps string keys to
 * integer values (i.e. any key has exactly one value associated with it).
 * The keys are nil-terminated strings and the values are nonnegative
 * integers (most likely an array index).  The @c Strmap supports fast
 * @f$ O(1) @f$ searching by applying a hash function to the key.
 *
 * The integer value associated with a key is intended to be an array index
 * (i.e. nonnegative).  Otherwise, the value will conflict with the
 * @c FAIL (-1) return value indicating an error.  
 * The @c const @c char @c * strings passed in as keys are aliased by the
 * hash table, so they must persist at least as long as the object does.
 * The easiest way is to use string literals as hash keys.
 *
 * Errors are generally indicated by a return value of @c FAIL,
 * due either to failed memory allocation or inability to find a key in
 * the table, depending on the function.
 * 
 * The hash table implementation was donated by John Stone,
 * and the interface was subsequently modified by David Hardy.
 */

#ifndef NLBASE_STRMAP_H
#define NLBASE_STRMAP_H

#include "nlbase/types.h"

#ifdef __cplusplus
extern "C" {
#endif

  /**@brief Strmap container class.
   */
  typedef struct Strmap_t {
    struct hash_node_t **bucket;  /**< array of hash nodes */
    int32 size;                   /**< size of the array */
    int32 entries;                /**< number of entries in hash table */
    int32 downshift;              /**< shift count, used for hashing */
    int32 mask;                   /**< used to select bits for hashing */
    char buf[64];                 /**< buffer for stats */
  } Strmap;


  int Strmap_init(Strmap *, int32 size);
  void Strmap_done(Strmap *);

  int32 Strmap_insert(Strmap *, const char *key, int32 value);
  int32 Strmap_lookup(const Strmap *, const char *key);
  int32 Strmap_update(Strmap *, const char *key, int32 newvalue);
  int32 Strmap_remove(Strmap *, const char *key);

  const char *Strmap_stats(Strmap *);

  int Strmap_print(const Strmap *);  /* for debugging */

#ifdef __cplusplus
}
#endif

#endif /* NLBASE_STRMAP_H */
