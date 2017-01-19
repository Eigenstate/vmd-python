/*
 * Copyright (C) 2000 by John Stone.  All rights reserved.
 *
 * strmap.c - simple hash table container
 * 
 * Uses null terminated strings as the keys for the hash table.
 * Stores an integer value with the string key.  It would
 * be easy to change to use void * values instead of int.
 * Maybe rewrite as a C++ template??
 *
 * Donated by John Stone.
 *
 * Modified by David Hardy.
 */

#include <string.h>
#include "nlbase/error.h"
#include "nlbase/io.h"
#include "nlbase/mem.h"
#include "nlbase/strmap.h"

#undef   HASH_LIMIT
#define  HASH_LIMIT  0.5

static int32 hash(const Strmap *tptr, const char *key);
static int rebuild(Strmap *tptr);
static double alos(Strmap *tptr);

/*
 *  Local types
 */
typedef struct hash_node_t {
  int32 data;                 /* data in hash node */
  const char *key;            /* key for hash lookup */
  struct hash_node_t *next;   /* next node in hash chain */
} hash_node_t;


/*
 *  hash() - Hash function returns a hash number for a given key.
 *
 *  tptr: Pointer to a hash table
 *  key: The key to create a hash number for
 */
int32 hash(const Strmap *tptr, const char *key) {
  int32 i=0;
  int32 hashvalue;
 
  while (*key != '\0')
    i=(i<<3)+(*key++ - '0');
 
  hashvalue = (((i*1103515249)>>tptr->downshift) & tptr->mask);
  if (hashvalue < 0) {
    hashvalue = 0;
  }

  return hashvalue;
}


/*
 * Strmap_done() - Delete the entire hash table and all remaining entries.
 * 
 */
void Strmap_done(Strmap *tptr) {
  hash_node_t *node, *last;
  int32 i;

  for (i=0; i<tptr->size; i++) {
    node = tptr->bucket[i];
    while (node != NULL) { 
      last = node;   
      node = node->next;
      NL_free(last);
    }
  }     

  /* free the entire array of buckets */
  NL_free(tptr->bucket);
  memset(tptr, 0, sizeof(Strmap));
}


/*
 *  Strmap_init() - Initialize a new hash table.
 *
 *  tptr: Pointer to the hash table to initialize
 *  buckets: The number of initial buckets to create
 */
int Strmap_init(Strmap *tptr, int32 buckets) {

  /* make sure we allocate something */
  if (buckets==0) buckets=16;

  /* initialize the table */
  tptr->entries=0;
  tptr->size=2;
  tptr->mask=1;
  tptr->downshift=29;

  /* ensure buckets is a power of 2 */
  while (tptr->size<buckets) {
    tptr->size<<=1;
    tptr->mask=(tptr->mask<<1)+1;
    tptr->downshift--;
  } /* while */

  /* allocate memory for hash table */
  tptr->bucket=(hash_node_t **) NL_calloc(tptr->size, sizeof(hash_node_t *));
  if (tptr->bucket == NULL) {
    tptr->size = 0;
    Strmap_done(tptr);
    return ERROR(ERR_MEMALLOC);
  }

  return OK;
}


/*
 *  rebuild() - Create new hash table when old one fills up.
 *
 *  tptr: Pointer to a hash table
 */
int rebuild(Strmap *tptr) {
  hash_node_t **old_bucket, *old_hash, *tmp;
  int32 old_size, old_entries, old_downshift, old_mask, h, i;
  int status = OK;

  old_bucket=tptr->bucket;
  old_size=tptr->size;
  old_entries=tptr->entries;
  old_downshift=tptr->downshift;
  old_mask=tptr->mask;

  /* create a new hash table and rehash old buckets */
  if ((status = Strmap_init(tptr, old_size<<1)) != OK) {
    tptr->bucket = old_bucket;
    tptr->size = old_size;
    tptr->entries = old_entries;
    tptr->downshift = old_downshift;
    tptr->entries = old_entries;
    return ERROR(status);
  }
  for (i=0; i<old_size; i++) {
    old_hash=old_bucket[i];
    while(old_hash) {
      tmp=old_hash;
      old_hash=old_hash->next;
      h=hash(tptr, tmp->key);
      tmp->next=tptr->bucket[h];
      tptr->bucket[h]=tmp;
      tptr->entries++;
    } /* while */
  } /* for */

  /* free memory used by old hash table */
  NL_free(old_bucket);

  return OK;
}


/*
 *  lookup() - Lookup an entry in the hash table and return a
 *  pointer to it or FAIL if it wasn't found.
 *
 *  tptr: Pointer to the hash table
 *  key: The key to lookup
 */
int32 Strmap_lookup(const Strmap *tptr, const char *key) {
  int32 h;
  hash_node_t *node;

  /* find the entry in the hash table */
  h=hash(tptr, key);
  for (node=tptr->bucket[h]; node!=NULL; node=node->next) {
    if (strcmp(node->key, key)==0) {
      /* found entry, return data */
      return node->data;
    }
  }

  /* otherwise we failed */
  return FAIL;
}


/*
 *  update() - Update int32 data for this key if the key is already
 *  in this table.  Return old data or FAIL if key isn't in table.
 *
 *  tptr: Pointer to the hash table
 *  key: The key to lookup
 *  data: The new data value for this key
 */
int32 Strmap_update(Strmap *tptr, const char *key, int32 data) {
  int32 h, olddata;
  hash_node_t *node;

  /* find the entry in the hash table */
  h = hash(tptr, key);
  for (node = tptr->bucket[h];  node != NULL;  node = node->next) {
    if (strcmp(node->key, key)==0) {
      /* found entry, update and return old data */
      olddata = node->data;
      node->data = data;
      return olddata;
    }
  }

  /* otherwise we failed */
  return FAIL;
}


/*
 *  insert() - Insert an entry into the hash table and return
 *  the data value.  If the entry already exists return a pointer to it,
 *  otherwise return FAIL.
 *
 *  tptr: A pointer to the hash table
 *  key: The key to insert into the hash table
 *  data: A pointer to the data to insert into the hash table
 */
int32 Strmap_insert(Strmap *tptr, const char *key, int32 data) {
  hash_node_t *node;
  int32 h;

  /* find the entry in the hash table */
  h=hash(tptr, key);
  for (node=tptr->bucket[h]; node!=NULL; node=node->next) {
    if (strcmp(node->key, key)==0) {
      /* found entry, return data value that we found */
      return node->data;
    }
  }
  /* otherwise, we need to insert this (key,data) pair into the table */

  /* expand the table if needed */
  if (tptr->entries>=HASH_LIMIT*tptr->size) {
    do {
      if (rebuild(tptr)) return FAIL;
    } while (tptr->entries>=HASH_LIMIT*tptr->size);
    /* have to recompute hash function */
    h=hash(tptr, key);
  }

  /* insert the new entry */
  node=(struct hash_node_t *) NL_malloc(sizeof(hash_node_t));
  if (node == NULL) {
    return ERROR(ERR_MEMALLOC);
  }
  node->data=data;
  node->key=key;
  node->next=tptr->bucket[h];
  tptr->bucket[h]=node;
  tptr->entries++;

  return data;
}


/*
 *  remove() - Remove an entry from a hash table and return a
 *  pointer to its data or FAIL if it wasn't found.
 *
 *  tptr: A pointer to the hash table
 *  key: The key to remove from the hash table
 */
int32 Strmap_remove(Strmap *tptr, const char *key) {
  hash_node_t *node, *last;
  int32 data;
  int32 h;

  /* find the node to remove */
  h=hash(tptr, key);
  for (node=tptr->bucket[h]; node; node=node->next) {
    if (strcmp(node->key, key)==0) {
      break;
    }
  }

  /* Didn't find anything, return FAIL */
  if (node==NULL) return FAIL;

  /* if node is at head of bucket, we have it easy */
  if (node==tptr->bucket[h])
    tptr->bucket[h]=node->next;
  else {
    /* find the node before the node we want to remove */
    for (last=tptr->bucket[h]; last && last->next; last=last->next) {
      if (last->next==node) break;
    }
    last->next=node->next;
  }

  /* free memory and return the data */
  tptr->entries--;
  data=node->data;
  NL_free(node);

  return(data);
}


/*
 *  alos() - Find the average length of search.
 *
 *  tptr: Pointer to a hash table
 */
double alos(Strmap *tptr) {
  int32 i,j;
  double alos=0;
  hash_node_t *node;

  for (i=0; i<tptr->size; i++) {
    for (node=tptr->bucket[i], j=0; node!=NULL; node=node->next, j++);
    if (j) alos += ((j*(j+1))>>1);
  } /* for */

  return(tptr->entries ? alos/tptr->entries : 0);
}


/*
 *  stats() - Return a string with stats about hash table.
 *
 *  tptr: A pointer to the hash table
 */
const char *Strmap_stats(Strmap *tptr) {
  snprintf(tptr->buf, sizeof(tptr->buf),
      "%d slots, %d entries, and %1.2f ALOS",
      (int)tptr->size, (int)tptr->entries, (double)alos(tptr));
  return (const char *) tptr->buf;
}


int Strmap_print(const Strmap *tp) {
  struct hash_node_t **bucket = tp->bucket;
  struct hash_node_t *node;
  int32 size = tp->size;
  int32 i;
  for (i = 0;  i < size;  i++) {
    node = bucket[i];
    if (NULL == node) {
      NL_printf("%3d   (empty)\n", i);
    }
    else {
      NL_printf("%3d   %20s   %3d\n", i, node->key, node->data);
      while (node->next) {
        node = node->next;
        NL_printf("      %20s   %3d\n", node->key, node->data);
      }
    }
  }
  return OK;
}
