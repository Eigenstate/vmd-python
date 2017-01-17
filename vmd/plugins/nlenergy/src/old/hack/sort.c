/*
 * An efficient, practical quicksort implementation from
 * Mark Allen Weiss, "Data Structures and Algorithm Analysis," 1992.
 */

#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include "nlbase/sort.h"
#include "nlbase/error.h"

#define MINDATA INT_MIN
#define CUTOFF  50 

#if 0
#define SWAP(a,b)  \
  do { \
    int32 tkey = (a).key;     \
    int32 tvalue = (a).value; \
    (a).key = (b).key;        \
    (a).value = (b).value;    \
    (b).key = tkey;           \
    (b).value = tvalue;       \
  } while (0)
#else
#define SWAP(x,y)  \
  do { \
    SortElem t = (x);  \
    (x) = (y);         \
    (y) = t;           \
  } while (0)
#endif

static void median3(SortElem a[], int32 left, int32 right, int32 *pivot);
static void qsorter(SortElem a[], int32 left, int32 right);

int Sort_quick(SortElem a[], int32 len) {
  qsorter(a, 0, len-1);           /* produces a mostly sorted array */
  return Sort_insertion(a, len);  /* clean it up */
}

void qsorter(SortElem a[], int32 left, int32 right) {
  int32 pivot;
  int32 i, j;
  if (left+CUTOFF <= right) {
    median3(a, left, right, &pivot);
    i = left;  j = right-1;
    while (1) {
      do { i++; } while (a[i].key < pivot);
      do { j--; } while (a[j].key > pivot);
      if (j <= i) break;
      SWAP(a[i], a[j]);
    }
    SWAP(a[i], a[right-1]);  /* restore pivot */
    qsorter(a, left, i);
    qsorter(a, i+1, right);
  } /* end if */
}

void median3(SortElem a[], int32 left, int32 right, int32 *pivot) {
  int32 center = (left + right) >> 1;  /* divide by 2 */
  if (a[left].key   > a[center].key) SWAP(a[left], a[center]);
  if (a[left].key   > a[right].key)  SWAP(a[left], a[right]);
  if (a[center].key > a[right].key)  SWAP(a[center], a[right]);
  /* invariant:  a[left] <= a[center] <= a[right] */
  *pivot = a[center].key;
  SWAP(a[center], a[right-1]);  /* hide pivot */
}

int Sort_insertion(SortElem a[], int32 len) {
  /* try to use a sentinel for efficiency */
  SortElem a0 = a[0], tmp;
  int32 j, k;
  a[0].key = MINDATA;  /* sentinel */
  for (k = 2;  k < len;  k++) {
    j = k;
    tmp = a[k];
    while (tmp.key < a[j-1].key) {
      a[j] = a[j-1];
      j--;
    }
    a[j] = tmp;
  }
#if 0
  /* find location for first element */
  if (len > CUTOFF) {
    /* guaranteed that there are larger elements than a[0] in array */
    j = 1;
    while (a0.key > a[j].key) {
      a[j-1] = a[j];
      j++;
    }
    a[j-1] = a0;
  }
  else {
    /* a[0] might be largest in array */
    for (j = 1;  j < len;  j++) {
      if (a0.key <= a[j].key) break;
      a[j-1] = a[j];
    }
    a[j-1] = a0;
  }
#else
  /* find location for first element, a[0] might be largest in array */
  for (j = 1;  j < len;  j++) {
    if (a0.key <= a[j].key) break;
    a[j-1] = a[j];
  }
  a[j-1] = a0;
#endif
  return OK;
}
