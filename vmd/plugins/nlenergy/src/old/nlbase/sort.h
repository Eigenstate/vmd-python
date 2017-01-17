#ifndef NLBASE_SORT_H
#define NLBASE_SORT_H

#include "nlbase/types.h"

#ifdef __cplusplus
extern "C" {
#endif

  /**@brief Give array of (key,value) pairs to sort routine.
   */
  typedef struct SortElem_t {
    int32 key;
    int32 value;
  } SortElem;

  /**@brief Use quicksort to sort SortElem array by key.
   *
   * Implementation uses median-of-3 pivot selection plus
   * cleanup with insertion sort.
   */
  int Sort_quick(SortElem a[], int len);

  /**@brief Use insertion sort to sort SortElem array by key.
   *
   * Although this is O(N^2) in general, it turns out to be O(N)
   * for almost sorted lists.
   */
  int Sort_insertion(SortElem a[], int len);


#ifdef __cplusplus
}
#endif

#endif /* NLBASE_SORT_H */
