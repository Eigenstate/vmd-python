/*
 * Copyright (C) 2007 by David J. Hardy.  All rights reserved.
 *
 * idlist.h - Idlist container stores ordered list of indices
 */

#ifndef NLBASE_IDLIST_H
#define NLBASE_IDLIST_H

#include "nlbase/types.h"
#include "nlbase/array.h"

#ifdef __cplusplus
extern "C" {
#endif

  /**@brief Ordered list of unique indices.
   *
   * Implementation is a cursor linked list using resize array.
   * In order insertion maintains the memsorted condition,
   * i.e. element ordering is in array index order.
   * Removing the max also maintains this memsorted condition.
   * Fast _find() and _max() whenever memsort condition is true,
   * otherwise these are linear time searches; fast _min() regardless.
   */
  typedef struct Idlist_t {
    Array idnode;       /**< array of Idlist_node */
    boolean memsorted;  /**< is traversal aligned contiguously in memory? */
  } Idlist;

  int Idlist_init(Idlist *);
  void Idlist_done(Idlist *);
  int Idlist_copy(Idlist *dest, const Idlist *src);

  int Idlist_erase(Idlist *);    /* erase all entries */
  int Idlist_unalias(Idlist *);  /* set internals to zero (for aliased list) */

  int32 Idlist_length(const Idlist *);  /* list length, number of IDs */
  int Idlist_insert(Idlist *, int32 id);  /* insert ID into list */
  int Idlist_remove(Idlist *, int32 id);  /* remove ID from list */
  int32 Idlist_find(const Idlist *, int32 id);  /* return id or FAIL */
  int32 Idlist_max(const Idlist *);  /* return max ID in list */
  int32 Idlist_min(const Idlist *);  /* return min ID in list */

  int Idlist_merge(Idlist *dest, const Idlist *src1, const Idlist *src2);
    /* destructive merge of src1 and src2, does not preserve dest */

  int Idlist_merge_into(Idlist *dest, const Idlist *src);
    /* merges src into dest, preserving contents of dest */

  void Idlist_memsort(Idlist *);  /* sort layout for contiguous mem access */

  void Idlist_print(const Idlist *);  /* prints out list for debugging */
  void Idlist_print_internal(const Idlist *);  /* prints out internal data */

  /**@brief Sequencer (iterator) for Idlist
   *
   * Read-only sequential iteration through Idlist elements.
   * While using the iterator, you shouldn't do _memsort() or
   * do _remove() of current element.
   */
  typedef struct Idseq_t {
    const Idlist *idlist;
    int32 current;
  } Idseq;

  int Idseq_init(Idseq *, const Idlist *);
  void Idseq_done(Idseq *);

  int32 Idseq_getid(Idseq *);  /* returns next ID in list, FAIL at list end */

#ifdef __cplusplus
}
#endif

#endif /* NLBASE_LIST_H */
