
#ifndef STRINGHASH_H
#define STRINGHASH_H

struct stringhash;
typedef struct stringhash stringhash;

stringhash * stringhash_create(void);
void stringhash_destroy(stringhash *h);

const char* stringhash_insert(stringhash *h, const char *key, const char *data);

#define STRINGHASH_FAIL 0

const char* stringhash_lookup(stringhash *h, const char *key);

const char* stringhash_delete(stringhash *h, const char *key);

#endif

