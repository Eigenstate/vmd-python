#include <stdlib.h>
#include "nlbase/sort.h"
#include "nlbase/nlbase.h"

#define DEFLEN   100
#define DEFRANGE  20
#define DEFSEED    0


int main(int argc, char *argv[]) {
  Array arr;
  Random ran;
  unsigned long seed;
  int len, range, i, s, debug;
  SortElem *a;

  if (argc > 5) {
    NL_fprintf(stderr, "syntax: %s [N RANGE SEED debug?]\n", argv[0]);
    exit(1);
  }
  len = (argc >= 2 ? atoi(argv[1]) : DEFLEN);
  range = (argc >= 3 ? atoi(argv[2]) : DEFRANGE);
  seed = (argc >= 4 ? atoi(argv[3]) : DEFSEED);
  debug = (argc >= 5);

  if (len <= 0) {
    NL_fprintf(stderr, "nelems must be positive\n");
    exit(1);
  }

  if ((s=Array_init(&arr, sizeof(SortElem))) != OK) {
    NL_fprintf(stderr, "Array_init() failed\n");
    exit(1);
  }
  if ((s=Array_resize(&arr, len)) != OK) {
    NL_fprintf(stderr, "Array_resize() failed for len=%d\n", len);
    exit(1);
  }

  if ((s=Random_initseed(&ran, seed)) != OK) {
    NL_fprintf(stderr, "Random_initseed() failed\n");
    exit(1);
  }

  a = (SortElem *) Array_data(&arr);
  NL_printf("creating random array of %d elements...\n", len);
  for (i = 0;  i < len;  i++) {
    a[i].key = (int32) (Random_uniform(&ran) * range);
    a[i].value = i;
  }
  if (debug) {
    for (i = 0;  i < len;  i++) {
      NL_printf(" %d", a[i].key);
    }
    NL_printf("\n");
  }

  NL_printf("calling quicksort...\n");
  Sort_quick(a, len);

  NL_printf("testing sort condition...\n");
  if (debug) {
    for (i = 0;  i < len;  i++) {
      NL_printf(" %d", a[i].key);
    }
    NL_printf("\n");
  }

  for (i = 1;  i < len;  i++) {
    if (a[i-1].key > a[i].key) {
      NL_fprintf(stderr, "failed sort condition for pair (%d,%d):\n"
          "a[%d]=%d  a[%d]=%d\n", i-1, i, i-1, a[i-1].key, i, a[i].key);
      exit(1);
    }
  }

  NL_printf("success!\n");
  return 0;
}
