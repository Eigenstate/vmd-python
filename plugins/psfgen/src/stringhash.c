
#include <string.h>
#include <stdlib.h>
#include "memarena.h"
#include "hasharray.h"
#include "stringhash.h"

struct stringhash {
  memarena *datarena;
  char **datarray;
  hasharray *ha;
};

stringhash * stringhash_create(void) {
  stringhash *h;
  if ( (h = (stringhash*) malloc(sizeof(stringhash))) ) {
    if ( ! ( h->datarena = memarena_create() ) ) {
      free((void*)h);
      return 0;
    }
    if ( ! ( h->ha = hasharray_create((void**)&(h->datarray),sizeof(char*)) ) ) {
      memarena_destroy(h->datarena);
      free((void*)h);
      return 0;
    }
  }
  return h;
}

void stringhash_destroy(stringhash *h) {
  if ( ! h ) return;
  memarena_destroy(h->datarena);
  hasharray_destroy(h->ha);
  free((void*)h);
}

const char* stringhash_insert(stringhash *h, const char *key, const char *data) {
  int i;
  char *s;
  if ( ! h ) return STRINGHASH_FAIL;
  i = hasharray_insert(h->ha,key);
  if ( i == HASHARRAY_FAIL ) return STRINGHASH_FAIL;
  h->datarray[i] = s = memarena_alloc(h->datarena,strlen(data)+1);
  if ( ! s ) {
    h->datarray[i] = STRINGHASH_FAIL;  /* should always be 0 */
    return STRINGHASH_FAIL;
  }
  strcpy(s,data);
  return s;
}

const char* stringhash_lookup(stringhash *h, const char *key) {
  int i;
  if ( ! h ) return STRINGHASH_FAIL;
  i = hasharray_index(h->ha,key);
  if ( i == HASHARRAY_FAIL ) return STRINGHASH_FAIL;
  return h->datarray[i];
}

const char* stringhash_delete(stringhash *h, const char *key) {
  int i;
  char *s;
  if ( ! h ) return STRINGHASH_FAIL;
  i = hasharray_index(h->ha,key);
  if ( i == HASHARRAY_FAIL ) return STRINGHASH_FAIL;
  s = h->datarray[i];
  h->datarray[i] = STRINGHASH_FAIL;
  return s;
}


