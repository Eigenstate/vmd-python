/*
 * Copyright (C) 2007 by David J. Hardy.  All rights reserved.
 *
 * test_arrmap.c - Demonstrate use of Arrmap class.
 */

#include "nlbase/nlbase.h"

typedef struct Angle_t {
  int32 atomID[3];
  int32 sum;        /* just store sum of atomIDs here */
} Angle;

const void *angle_key(const Array *angle_array, int32 n) {
  const Angle *ap = Array_elem_const(angle_array, n);
  return (ap ? ap->atomID : NULL);
}

int32 angle_hash(const Arrmap *arrmap, const void *key) {
  const int32 *atomID = key;
  uint32 i = atomID[0];
  int32 k, hashvalue;
  for (k = 1;  k < 3;  k++) {
    i = (i << 5) + (atomID[k] - atomID[0]);
  }
  hashvalue = (((i*1103515249) >> arrmap->downshift) & arrmap->mask);
  return hashvalue;
}

int32 angle_keycmp(const void *key1, const void *key2) {
  const int32 *aid = key1;
  const int32 *bid = key2;
  int32 diff = aid[0] - bid[0];
  int32 j;
  for (j = 1;  diff == 0 && j < 3;  j++) {
    diff = aid[j] - bid[j];
  }
  return diff;
}


int main() {
  /* tested to work for up to N==10000000 */
#define N 1000
  Array a;
  Arrmap map;

  int32 n, s, prev;

  Array_init(&a, sizeof(Angle));
  Arrmap_init(&map, &a, angle_key, angle_hash, angle_keycmp, 0);

  /* store a lot of water angles */
  for (n = 0;  n < N;  n++) {
    Angle t;
    t.atomID[0] = 3*n;
    t.atomID[1] = 3*n + 1;
    t.atomID[2] = 3*n + 2;
    t.sum = 9*n + 3;
    Array_append(&a, &t);
    if ((s = Arrmap_insert(&map, n)) != n) {
      printf("Arrmap_insert failed: n=%d s=%d\n", n, s);
      exit(1);
    }
#if (N <= 10000)
    else {
      printf("n=%d, stats: %s\n", n, Arrmap_stats(&map));
    }
#endif
  }

  /* try to insert an existing water angle */
  {
    Angle t;
    t.atomID[0] = 3;
    t.atomID[1] = 4;
    t.atomID[2] = 5;
    t.sum = 12;
    Array_append(&a, &t);
    if ((s = Arrmap_insert(&map, N)) < 0) {
      printf("Arrmap_insert failed: s=%d N=%d\n", s, N);
      exit(1);
    }
    else if (s == N) {
      printf("allowed insert s=%d N=%d that should have faild\n", s, N);
      exit(1);
    }
    else {
      printf("Success: found existing angle index=%d\n", s);
    }
    prev = s;
    if ((s = Arrmap_update(&map, N)) != prev) {
      printf("Arrmap_update failed: s=%d prev=%d\n", s, prev);
      exit(1);
    }
    else {
      printf("Success: updated previous angle index=%d to %d\n", prev, N);
    }
  }

  /* lookup some assortment of angles */
  for (n = 10;  n < N;  n += 10) {
    Angle t;
    t.atomID[0] = 3*n;
    t.atomID[1] = 3*n + 1;
    t.atomID[2] = 3*n + 2;
    if ((s = Arrmap_lookup(&map, t.atomID)) != FAIL) {
      printf("Found angle index %d\n", s);
    }
    else {
      printf("Arrmap_lookup failed: unable to find angle index %d\n", s);
      exit(1);
    }
  }

  /* remove half of the angles */
  for (n = N;  n >= N/2;  n--) {
    if ((s = Arrmap_remove(&map, n)) != n) {
      printf("Arrmap_remove failed: s=%d n=%d\n", s, n);
      exit(1);
    }
#if (N <= 10000)
    else {
      printf("n=%d, stats: %s\n", n, Arrmap_stats(&map));
    }
#endif
    Array_remove(&a, NULL);
  }

  /* success, cleanup */
  printf("Success!\n");
  Arrmap_done(&map);
  Array_done(&a);

  return 0;
}
