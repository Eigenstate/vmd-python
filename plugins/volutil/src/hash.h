/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2004 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: hash.h,v $
 *      $Author: ltrabuco $        $Locker:  $             $State: Exp $
 *      $Revision: 1.1 $      $Date: 2009/08/06 20:58:45 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   A simple hash table implementation for strings, contributed by John Stone,
 *   derived from his ray tracer code.
 ***************************************************************************/

#ifndef _HASH_H
#define _HASH_H

/** hash table top level data structure */
typedef struct {
  struct hash_node_t **bucket;        /* array of hash nodes */
  int size;                           /* size of the array */
  int entries;                        /* number of entries in table */
  int downshift;                      /* shift cound, used in hash function */
  int mask;                           /* used to select bits for hashing */
} hash_t;

#define HASH_FAIL -1

/** initialize hash table for first use */
void hash_init(hash_t *, int);

/** lookup a string key in the hash table returning its integer key */
int hash_lookup (const hash_t *, const char *);

/** insert a string into the hash table, along with an integer key */
int hash_insert (hash_t *, const char *, int);

/** delete an string from the hash table, given its string name */
int hash_delete (hash_t *, const char *);

/** destroy the hash table completely, deallocate memory */
void hash_destroy(hash_t *);

/** print hash table vital stats */
char *hash_stats (hash_t *);



#endif
