
#include <stdlib.h>
#include <string.h>
#include "hash.h"
#include "memarena.h"
#include "hasharray.h"

struct hasharray {
  memarena *keyarena;
  hash_t hash;
  int count;
  int alloc;
  int itemsize;
  void **itemarray;
};

hasharray * hasharray_create(void **itemarray, int itemsize) {
  hasharray * a;
  if ( (a = (hasharray*) malloc(sizeof(hasharray))) ) {
    a->count = 0;
    a->alloc = 0;
    a->itemsize = itemsize;
    a->itemarray = itemarray;
    *(a->itemarray) = 0;
    if ( ! ( a->keyarena = memarena_create() ) ) {
      free((void*)a);
      return 0;
    }
    hash_init(&(a->hash),0);
  }
  return a;
}

int hasharray_clear(hasharray *a) {
  if ( ! a ) return HASHARRAY_FAIL;
  hash_destroy(&(a->hash));
  memarena_destroy(a->keyarena);
  if ( ! ( a->keyarena = memarena_create() ) ) {
    return HASHARRAY_FAIL;
  }
  hash_init(&(a->hash),0);
  return 0;
}

void hasharray_destroy(hasharray *a) {
  if ( ! a ) return;
  hash_destroy(&(a->hash));
  memarena_destroy(a->keyarena);
  if ( *(a->itemarray) ) {
    free(*(a->itemarray));
    *(a->itemarray) = 0;
  }
  free((void*)a);
}

int hasharray_reinsert(hasharray *a, const char *key, int pos) {
  int i;
  char *s;
  if ( ! a ) return HASHARRAY_FAIL;
  i = hash_lookup(&(a->hash),key);
  if ( i != HASH_FAIL ) return i;
  i = pos;
  if ( ! ( s = memarena_alloc(a->keyarena,strlen(key)+1) ) ) {
    return HASHARRAY_FAIL;
  }
  strcpy(s,key);
  hash_insert(&(a->hash),s,i);
  return i;
}

int hasharray_insert(hasharray *a, const char *key) {
  int i;
  int new_alloc;
  void *new_array;
  char *s;
  if ( ! a ) return HASHARRAY_FAIL;
  i = hash_lookup(&(a->hash),key);
  if ( i != HASH_FAIL ) return i;
  i = a->count;
  a->count++;
  if ( a->count > a->alloc ) {
    if ( a->alloc ) new_alloc = a->alloc * 2;
    else new_alloc = 8;
    new_array = realloc(*(a->itemarray), new_alloc * a->itemsize);
    if ( new_array ) {
      *(a->itemarray) = new_array;
      a->alloc = new_alloc;
    } else return HASHARRAY_FAIL;
  }
  if ( ! ( s = memarena_alloc(a->keyarena,strlen(key)+1) ) ) {
    return HASHARRAY_FAIL;
  }
  strcpy(s,key);
  hash_insert(&(a->hash),s,i);
  return i;
}

int hasharray_delete(hasharray *a, const char *key) {
  if (!a) return HASHARRAY_FAIL; /* I think this should be assert(a) */
  return hash_delete(&(a->hash), key);
}

int hasharray_index(hasharray *a, const char *key) {
  int i;
  if ( ! a ) return HASHARRAY_FAIL;
  i = hash_lookup(&(a->hash),key);
  if ( i == HASH_FAIL ) i = HASHARRAY_FAIL;
  return i;
}

int hasharray_count(hasharray *a) {
  if ( ! a ) return HASHARRAY_FAIL;
  return a->count;
}

