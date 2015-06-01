
#ifndef MEMARENA_H
#define MEMARENA_H

struct memarena;
typedef struct memarena memarena;

memarena * memarena_create(void);
void memarena_destroy(memarena *a);

void memarena_blocksize(memarena *a, int blocksize);
void * memarena_alloc(memarena *a, int size);
void * memarena_alloc_aligned(memarena *a, int size, int alignment);

#endif

