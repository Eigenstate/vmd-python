/*
 * Copyright (C) 2000 by John Stone.  All rights reserved.
 *
 * arrmap.c - simple hash table container
 */

#include <string.h>
#include "nlbase/error.h"
#include "nlbase/io.h"
#include "nlbase/mem.h"
#include "nlbase/arrmap.h"

#undef   HASH_LIMIT
#define  HASH_LIMIT  0.5

static int rebuild(Arrmap *);
static double alos(Arrmap *);

/*
 *  Local types
 */
typedef struct Arrmap_node_t {
  int32 index;                  /* index into the array */
  struct Arrmap_node_t *next;   /* next node in hash chain */
} Arrmap_node;


int Arrmap_init(Arrmap *a, const Array *array,
    const void *(*key)(const Array *, int32 i),
    int32 (*hash)(const Arrmap *, const void *key),
    int32 (*keycmp)(const void *key1, const void *key2),
    int32 nbuckets) {

  /* make sure we allocate something */
  if (nbuckets == 0) nbuckets = 16;

  /* initialize the table */
  a->entries = 0;
  a->size = 2;
  a->mask = 1;
  a->downshift = 29;

  /* initialize the array and generic methods */
  a->array = array;
  a->key = key;
  a->hash = hash;
  a->keycmp = keycmp;

  /* ensure buckets is a power of 2 */
  while (a->size < nbuckets) {
    a->size <<= 1;
    a->mask = (a->mask << 1) + 1;
    a->downshift--;
  } /* while */

  /* allocate memory for hash table */
  a->bucket = (Arrmap_node **) NL_calloc(a->size, sizeof(Arrmap_node *));
  if (a->bucket == NULL) {
    a->size = 0;
    Arrmap_done(a);
    return ERROR(ERR_MEMALLOC);
  }

  return OK;
}


void Arrmap_done(Arrmap *a) {
  Arrmap_node *node, *last;
  int32 i;

  for (i = 0;  i < a->size;  i++) {
    node = a->bucket[i];
    while (node != NULL) { 
      last = node;   
      node = node->next;
      NL_free(last);
    }
  }     

  /* free the entire array of buckets */
  NL_free(a->bucket);
  memset(a, 0, sizeof(Arrmap));
}


int rebuild(Arrmap *a) {
  Arrmap_node **old_bucket, *old_hash, *tmp;
  const void *keyptr = NULL;
  int32 old_size, old_entries, old_downshift, old_mask, h, i;
  int status = OK;

  old_bucket = a->bucket;
  old_size = a->size;
  old_entries = a->entries;
  old_downshift = a->downshift;
  old_mask = a->mask;

  /* create a new hash table and rehash old buckets */
  if ((status = Arrmap_init(a, a->array, a->key, a->hash, a->keycmp,
          old_size << 1)) != OK) {
    a->bucket = old_bucket;
    a->size = old_size;
    a->entries = old_entries;
    a->downshift = old_downshift;
    a->entries = old_entries;
    return ERROR(status);
  }
  for (i = 0;  i < old_size;  i++) {
    old_hash = old_bucket[i];
    while (old_hash) {
      tmp = old_hash;
      old_hash = old_hash->next;
      if ((keyptr = a->key(a->array, tmp->index)) == NULL) {
        return ERROR(ERR_EXPECT);
      }
      h = a->hash(a, keyptr);
      tmp->next = a->bucket[h];
      a->bucket[h] = tmp;
      a->entries++;
    } /* while */
  } /* for */

  /* free memory used by old hash table */
  NL_free(old_bucket);

  return OK;
}


int32 Arrmap_lookup(const Arrmap *a, const void *inkey) {
  int32 h;
  const void *keyptr;
  Arrmap_node *node;

  ASSERT(inkey != NULL);

  /* find the entry in the hash table */
  h = a->hash(a, inkey);
  for (node = a->bucket[h];  node != NULL;  node = node->next) {
    if ((keyptr = a->key(a->array, node->index)) == NULL) {
      return ERROR(ERR_EXPECT);
    }
    if (a->keycmp(inkey, keyptr) == 0) {
      /* found entry, return data */
      return node->index;
    }
  }

  /* otherwise we failed */
  return FAIL;
}


int32 Arrmap_update(Arrmap *a, int32 i) {
  /*
   * here we have multiple array entries with the same key,
   * update key from a previous array entry to the ith one
   */
  int32 h, old_index;
  const void *inkey, *keyptr;
  Arrmap_node *node;

  /* find the entry in the hash table */
  if ((inkey = a->key(a->array, i)) == NULL) {
    return ERROR(ERR_EXPECT);
  }
  h = a->hash(a, inkey);
  for (node = a->bucket[h];  node != NULL;  node = node->next) {
    if ((keyptr = a->key(a->array, node->index)) == NULL) {
      return ERROR(ERR_EXPECT);
    }
    if (a->keycmp(inkey, keyptr) == 0) {
      /* found entry, update and return old index */
      old_index = node->index;
      node->index = i;
      return old_index;
    }
  }

  /* otherwise we failed */
  return FAIL;
}


int32 Arrmap_insert(Arrmap *a, int32 i) {
  Arrmap_node *node;
  const void *inkey, *keyptr;
  int32 h;

  /* find the entry in the hash table */
  if ((inkey = a->key(a->array, i)) == NULL) {
    return ERROR(ERR_EXPECT);
  }
  h = a->hash(a, inkey);
  for (node = a->bucket[h];  node != NULL;  node = node->next) {
    if ((keyptr = a->key(a->array, node->index)) == NULL) {
      return ERROR(ERR_EXPECT);
    }
    if (a->keycmp(inkey, keyptr) == 0) {
      /* found an entry with same key, return its index */
      return node->index;
    }
  }
  /* otherwise, we need to insert this index into the table */

  /* expand the table if needed */
  if (a->entries >= HASH_LIMIT * a->size) {
    do {
      if (rebuild(a)) return FAIL;
    } while (a->entries >= HASH_LIMIT * a->size);
    /* have to recompute hash function */
    h = a->hash(a, inkey);
  }

  /* insert the new entry */
  node = (Arrmap_node *) NL_malloc(sizeof(Arrmap_node));
  if (node == NULL) {
    return ERROR(ERR_MEMALLOC);
  }
  node->index = i;
  node->next = a->bucket[h];
  a->bucket[h] = node;
  a->entries++;

  return i;
}


int32 Arrmap_remove(Arrmap *a, int32 i) {
  Arrmap_node *node, *last;
  const void *inkey, *keyptr;
  int32 index;
  int32 h;

  /* find the node to remove */
  if ((inkey = a->key(a->array, i)) == NULL) {
    return ERROR(ERR_EXPECT);
  }
  h = a->hash(a, inkey);
  for (node = a->bucket[h];  node;  node = node->next) {
    if ((keyptr = a->key(a->array, node->index)) == NULL) {
      return ERROR(ERR_EXPECT);
    }
    if (a->keycmp(inkey, keyptr) == 0) {
      break;
    }
  }

  /* Didn't find anything, return FAIL */
  if (node == NULL) return FAIL;

  /* if node is at head of bucket, we have it easy */
  if (node == a->bucket[h]) {
    a->bucket[h] = node->next;
  }
  else {
    /* find the node before the node we want to remove */
    for (last = a->bucket[h];  last && last->next;  last = last->next) {
      if (last->next == node) break;
    }
    last->next = node->next;
  }

  /* do we find the correct index associated with this hash key? */
  index = node->index;
  if (i != index) return(index);  /* if not, return the correct index */

  /* free memory and return the index */
  a->entries--;
  NL_free(node);

  return(index);
}


double alos(Arrmap *a) {
  int32 i, j;
  double alos = 0;
  Arrmap_node *node;

  for (i = 0;  i < a->size;  i++) {
    for (node = a->bucket[i], j = 0;  node != NULL;  node = node->next, j++) ;
    if (j) alos += ((j*(j+1)) >> 1);  /* i.e., alos += 1+2+3+...+j */
  } /* for */

  return(a->entries ? alos / a->entries : 0);
}


const char *Arrmap_stats(Arrmap *a) {
  snprintf(a->buf, sizeof(a->buf),
      "%d slots, %d entries, and %1.2f ALOS",
      (int) a->size, (int) a->entries, (double) alos(a));
  return (const char *) a->buf;
}
