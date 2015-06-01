
#include <stdlib.h>
#include "memarena.h"

struct memarena_stack_t;
typedef struct memarena_stack_t memarena_stack_t;
struct memarena_stack_t {
  memarena_stack_t * next;
  void * data;
};

struct memarena {
  memarena_stack_t * stack;
  int newblocksize;
  int size, used;
};

memarena * memarena_create(void) {
  memarena * a;
  if ( (a = (memarena*) malloc(sizeof(memarena))) ) {
    a->stack = 0;
    a->newblocksize = 128000;
    a->size = 0;
    a->used = 0;
  }
  return a;
}

void memarena_destroy(memarena *a) {
  memarena_stack_t * s;
  if ( ! a ) return;
  while ( a->stack ) {
    s = a->stack;
    a->stack = s->next;
    free((void*)s->data);
    free((void*)s);
  }
  free((void*)a);
}

void memarena_blocksize(memarena *a, int blocksize) {
  a->newblocksize = blocksize;
}

void * memarena_alloc(memarena *a, int size) {
  memarena_stack_t * s;
  void * m;
  if ( size > a->newblocksize / 2 ) {
    s = (memarena_stack_t*) malloc(sizeof(memarena_stack_t));
    if ( ! s ) return 0;
    s->data = malloc(size);
    if ( ! s->data ) {
      free((void*)s);
      return 0;
    }
    if ( a->stack ) {
      s->next = a->stack->next;
      a->stack->next = s;
    } else {
      s->next = 0;
      a->stack = s;
    }
    return s->data;
  } else if ( a->used + size > a->size ) {
    s = (memarena_stack_t*) malloc(sizeof(memarena_stack_t));
    if ( ! s ) return 0;
    s->next = a->stack;
    s->data = malloc(a->newblocksize);
    if ( ! s->data ) {
      free((void*)s);
      return 0;
    }
    a->stack = s;
    a->size = a->newblocksize;
    a->used = 0;
  }
  m = (void*) ( (char*) a->stack->data + a->used );
  a->used += size;
  return m;
}

void * memarena_alloc_aligned(memarena *a, int size, int alignment) {
  return 0;
}


