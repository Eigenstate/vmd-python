/*
 * hash.h - A simple hash table
 * 
 * Uses null terminated strings as the keys for the table.
 * Stores an integer value with the string key.  It would be
 * easy to change to use void *'s instead of ints.  Maybe rewrite
 * as a C++ template??
 * 
 * Donated by John Stone
 */

#ifndef HASH_H
#define HASH_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hash_t {
  struct hash_node_t **bucket;        /* array of hash nodes */
  int size;                           /* size of the array */
  int entries;                        /* number of entries in table */
  int downshift;                      /* shift cound, used in hash function */
  int mask;                           /* used to select bits for hashing */
} hash_t;

#define HASH_FAIL -1

void hash_init(hash_t *, int);
int hash_lookup (hash_t *, const char *);
int hash_insert (hash_t *, const char *, int);
int hash_delete (hash_t *, const char *);
void hash_destroy(hash_t *);
char *hash_stats (hash_t *);

#ifdef __cplusplus
}
#endif

#endif

