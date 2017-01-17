/*
 * Copyright (C) 2007 by David J. Hardy.  All rights reserved.
 *
 * idlist.c - methods for Idlist container, cursor linked-list implementation
 */

#include <string.h>
#include "nlbase/error.h"
#include "nlbase/io.h"
#include "nlbase/mem.h"
#include "nlbase/idlist.h"


typedef struct Idnode_t {
  int32 id;
  int32 next;
} Idnode;


int Idlist_init(Idlist *p) {
  Idnode headnode = {-1, -1};
  int s;  /* error status */
  ASSERT(p != NULL);
  if ((s=Array_init(&(p->idnode), sizeof(Idnode))) != OK) return ERROR(s);
  if ((s=Array_append(&(p->idnode), &headnode)) != OK) return ERROR(s);
  p->memsorted = TRUE;
  return OK;
}


void Idlist_done(Idlist *p) {
  Array_done(&(p->idnode));
}


/* dest is always memsorted */
int Idlist_copy(Idlist *dest, const Idlist *src) {
  int s;  /* error status */
  if (src->memsorted) {
    if ((s=Array_copy(&(dest->idnode), &(src->idnode))) != OK) return ERROR(s);
  }
  else {
    const Idnode *asrc = Array_data_const(&(src->idnode));
    Idnode *adest;
    int32 ksrc, kdest;
    if ((s=Array_setbuflen(&(dest->idnode), src->idnode.buflen)) != OK) {
      return ERROR(s);
    }
    if ((s=Array_resize(&(dest->idnode), src->idnode.uselen)) != OK) {
      return ERROR(s);
    }
    adest = Array_data(&(dest->idnode));
    adest[0].id = -1;  /* sentinel value for head node */
    kdest = 0;
    for (ksrc = asrc[0].next;  ksrc != -1;  ksrc = asrc[ksrc].next) {
      adest[kdest].next = kdest+1;
      adest[kdest+1].id = asrc[ksrc].id;
      ASSERT(adest[kdest].id < adest[kdest+1].id);
      kdest++;
    }
    adest[kdest].next = -1;  /* mark end of list */
  }
  dest->memsorted = TRUE;
  return OK;
}


/* set list to empty */
int Idlist_erase(Idlist *p) {
  Idnode *arr = Array_data(&(p->idnode));
  int s;  /* error status */
  ASSERT(-1 == arr[0].id);
  arr[0].next = -1;  /* list ends after head node */
  if ((s=Array_resize(&(p->idnode), 1)) != OK) return ERROR(s);
  p->memsorted = TRUE;
  return OK;
}


/* unalias this list after shallow copy */
int Idlist_unalias(Idlist *p) {
  Idnode headnode = {-1, -1};
  int s;  /* error status */
  Array_unalias(&(p->idnode));
  if ((s=Array_append(&(p->idnode), &headnode)) != OK) return ERROR(s);
  p->memsorted = TRUE;
  return OK;
}


int32 Idlist_length(const Idlist *p) {
  return Array_length(&(p->idnode)) - 1;
}


/*
 * for memsorted list, return index of smallest element that is
 * greater than or equal to id (e.g. if id is greater than all
 * elements, return len)
 */
static int32 binary_search(const Idnode *arr, int32 len, int32 id) {
  int32 lo = 0;    /* lowest index of sorted subarray */
  int32 hi = len;  /* highest index plus one of sorted subarray */
  int32 mid;
  while (lo < hi) {
    mid = lo + ((hi - lo) >> 1);  /* avoid overflow */
    if (arr[mid].id < id) lo = mid+1;
    else hi = mid;
  }
  return lo;
}


int Idlist_insert(Idlist *p, int32 id) {
  Idnode newnode;
  Idnode *arr = Array_data(&(p->idnode));
  int32 len = Array_length(&(p->idnode));  /* index for newnode insertion */
  int s;  /* error status */

  if (id < 0) return ERROR(ERR_VALUE);  /* id must be non-negative */
  newnode.id = id;
  if (p->memsorted) {
    if (arr[len-1].id < id) {  /* O(1) insert max node into memsorted list */
      newnode.next = -1;
      arr[len-1].next = len;
    }
    else {                       /* O(log n) insert into memsorted list */
      int32 k = binary_search(arr, len, id);
      ASSERT(k > 0 && k < len);
      if (id == arr[k].id) return FAIL;  /* duplicate id value */
      arr[k-1].next = len;
      newnode.next = k;
      p->memsorted = FALSE;
    }
  }
  else {                         /* O(n) insert into non-memsorted list */
    int32 prev = 0;
    int32 curr = arr[0].next;
    while (curr != -1 && id > arr[curr].id) {
      prev = curr;
      curr = arr[curr].next;
    }
    if (curr != -1 && id == arr[curr].id) return FAIL;  /* dup id value */
    arr[prev].next = len;
    newnode.next = curr;
    p->memsorted = FALSE;
  }
  if ((s=Array_append(&(p->idnode), &newnode)) != OK) return ERROR(s);
  return OK;
}


int Idlist_remove(Idlist *p, int32 id) {
  Idnode *arr = Array_data(&(p->idnode));
  int32 len = Array_length(&(p->idnode));
  int s;  /* error status */

  if (id < 0) return ERROR(ERR_VALUE);  /* id must be non-negative */
  if (p->memsorted) {
    if (id == arr[len-1].id) {  /* O(1) remove max node from memsorted list */
      arr[len-2].next = -1;
    }
    else {                      /* O(log n) remove from memsorted list */
      int32 k = binary_search(arr, len, id);
      ASSERT(k > 0 && k <= len);
      if (k == len || id != arr[k].id) return FAIL;    /* no node to remove */
      if (len-2 == k) {   /* remove next to last node, keeps list memsorted */
        arr[k] = arr[k+1];          /* just slide last node down into place */
      }
      else {
        ASSERT(k < len-2);
        arr[k-1].next = arr[k].next;   /* unlink kth node */
        arr[k] = arr[len-1];           /* fill the hole with last element */
        arr[len-2].next = k;           /* relink this element */
        p->memsorted = FALSE;          /* list is no longer sorted */
      }
    }
  }
  else {                        /* O(n) remove from non-memsorted list */
    int32 prev = 0;
    int32 curr = arr[0].next;
    while (curr != -1 && id > arr[curr].id) {
      prev = curr;
      curr = arr[curr].next;
    }
    if (curr == -1 || id != arr[curr].id) return FAIL;  /* no node to remove */
    arr[prev].next = arr[curr].next;  /* unlink current node */
    if (curr != len-1) {  /* left hole in middle of array, need to fill it */
      int32 k;  /* find pointer to the last array element */
      for (k = 0;  k < len-1 && arr[k].next != len-1;  k++) ;
      if (len-1 == k) return ERROR(ERR_EXPECT);  /* list was inconsistent */
      arr[curr] = arr[len-1];  /* rotate last array element into the hole */
      arr[k].next = curr;      /* relink this element */
    }
    p->memsorted = (len <= 3); /* if so, list is now sorted length 0 or 1 */
  }
  if ((s=Array_remove(&(p->idnode), NULL)) != OK) return ERROR(s);
  return OK;
}


int32 Idlist_find(const Idlist *p, int32 id) {
  const Idnode *arr = Array_data_const(&(p->idnode));
  const int32 len = Array_length(&(p->idnode));

  if (id < 0) return ERROR(ERR_VALUE);  /* id must be non-negative */
  else if (p->memsorted) {
    int32 k = binary_search(arr, len, id);
    if (k == len || arr[k].id != id) return FAIL;
  }
  else {
    int32 k = arr[0].next;
    while (k != -1 && id > arr[k].id) k = arr[k].next;
    if (k == -1 || arr[k].id != id) return FAIL;
  }
  return id;
}


int32 Idlist_max(const Idlist *p) {
  const Idnode *arr = Array_data_const(&(p->idnode));
  const int32 len = Array_length(&(p->idnode));

  if (p->memsorted) {
    return arr[len-1].id;  /* this returns FAIL==-1 for empty list */
  }
  else {
    int32 k = 0;
    while (arr[k].next != -1) k = arr[k].next;
    return arr[k].id;  /* this returns FAIL==-1 for empty list */
  }
}


int32 Idlist_min(const Idlist *p) {
  const Idnode *arr = Array_data_const(&(p->idnode));
  const int32 len = Array_length(&(p->idnode));
  return (len > 1 ? arr[ arr[0].next ].id : FAIL);
}


/* destructive merge of src1 and src2, does not preserve dest */
int Idlist_merge(Idlist *dest, const Idlist *src1, const Idlist *src2) {
  Idnode *adest = Array_data(&(dest->idnode));
  const Idnode *arr1 = Array_data_const(&(src1->idnode));
  const Idnode *arr2 = Array_data_const(&(src2->idnode));
  int32 k1 = arr1[0].next;
  int32 k2 = arr2[0].next;
  int32 destlen = 1;  /* empty list has array length 1 */
  Idnode newnode;
  int s;  /* error status */

  if ((s=Idlist_erase(dest)) != OK) return ERROR(s);
  while (k1 != -1 && k2 != -1) {
    if (arr1[k1].id < arr2[k2].id) {
      newnode.id = arr1[k1].id;
      newnode.next = destlen++;
      if ((s=Array_append(&(dest->idnode), &newnode)) != OK) return ERROR(s);
      k1 = arr1[k1].next;
    }
    else if (arr1[k1].id > arr2[k2].id) {
      newnode.id = arr2[k2].id;
      newnode.next = destlen++;
      if ((s=Array_append(&(dest->idnode), &newnode)) != OK) return ERROR(s);
      k2 = arr2[k2].next;
    }
    else {  /* elements are equal, take one but increment both src pointers */
      newnode.id = arr1[k1].id;
      newnode.next = destlen++;
      if ((s=Array_append(&(dest->idnode), &newnode)) != OK) return ERROR(s);
      k1 = arr1[k1].next;
      k2 = arr2[k2].next;
    }
  }
  while (k1 != -1) {
    newnode.id = arr1[k1].id;
    newnode.next = destlen++;
    if ((s=Array_append(&(dest->idnode), &newnode)) != OK) return ERROR(s);
    k1 = arr1[k1].next;
  }
  while (k2 != -1) {
    newnode.id = arr2[k2].id;
    newnode.next = destlen++;
    if ((s=Array_append(&(dest->idnode), &newnode)) != OK) return ERROR(s);
    k2 = arr2[k2].next;
  }
  adest = Array_data(&(dest->idnode));
  adest[destlen-1].next = -1;
  return OK;
}


/* merges src into dest, preserving contents of dest */
/* memsorted if dest is empty */
int Idlist_merge_into(Idlist *dest, const Idlist *src) {
  const Idnode *asrc = Array_data_const(&(src->idnode));
  Idnode *adest = Array_data(&(dest->idnode));
  int32 destlen = Array_length(&(dest->idnode));
  int32 prev = 0;              /* in dest, link before node insertion point */
  int32 curr = adest[0].next;  /* in dest, link at node insertion point */
  int32 index = asrc[0].next;  /* link in src */
  Idnode newnode;
  int s;  /* error status */

  while (curr != -1 && index != -1) {
    if (adest[curr].id < asrc[index].id) {  /* increment dest */
      prev = curr;
      curr = adest[curr].next;
    }
    else if (adest[curr].id == asrc[index].id) {  /* incr dest and src */
      prev = curr;
      curr = adest[curr].next;
      index = asrc[index].next;
    }
    else {  /* put src node at end of dest array and link it */
      newnode.id = asrc[index].id;
      newnode.next = curr;
      if ((s=Array_append(&(dest->idnode), &newnode)) != OK) return ERROR(s);
      adest = Array_data(&(dest->idnode));
      adest[prev].next = destlen;
      prev = destlen++;  /* move prev to this new node and incr array len */
      dest->memsorted = FALSE;
      index = asrc[index].next;
    }
  }
  while (index != -1) {
    newnode.id = asrc[index].id;
    newnode.next = curr;
    if ((s=Array_append(&(dest->idnode), &newnode)) != OK) return ERROR(s);
    adest = Array_data(&(dest->idnode));
    adest[prev].next = destlen;
    prev = destlen++;  /* move prev to this new node and incr array len */
    index = asrc[index].next;
  }
  return OK;
}


static void insertion_sort(Idnode *arr, int32 len) {
  int32 i, j, val;
  for (i = 2;  i < len;  i++) {
    val = arr[i].id;
    j = i-1;
    while (arr[j].id > val) {
      arr[j+1].id = arr[j].id;
      j--;
    }
    arr[j+1].id = val;
  }
}


/* sort layout for contiguous mem access */
void Idlist_memsort(Idlist *p) {
  if ( !(p->memsorted) ) {
    Idnode *arr = Array_data(&(p->idnode));
    int32 len = Array_length(&(p->idnode));
    int32 k;
    insertion_sort(arr, len);         /* we can add quicksort later */
    for (k = 0;  k < len-1;  k++) {   /* repair the next pointers */
      arr[k].next = k+1;
    }
    arr[k].next = -1;
    p->memsorted = TRUE;
  }
}


void Idlist_print(const Idlist *p) {
  Idseq s;
  int32 id;
  Idseq_init(&s, p);
  while ((id = Idseq_getid(&s)) != FAIL) {
    printf("%d ", id);
  }
  printf("\n");
  Idseq_done(&s);
}


void Idlist_print_internal(const Idlist *p) {
  const Idnode *arr = Array_data_const(&(p->idnode));
  int32 len = Array_length(&(p->idnode));
  int32 k;
  printf("memsorted=%d  uselen=%d  buflen=%d\n",
      (int)(p->memsorted), (int)(p->idnode.uselen), (int)(p->idnode.buflen));
  for (k = 0;  k < len;  k++) {
    printf("(%d,%d) ", arr[k].id, arr[k].next);
  }
  printf("\n");
}


int Idseq_init(Idseq *q, const Idlist *p) {
  const Idnode *arr = Array_data_const(&(p->idnode));
  ASSERT(q != NULL);
  q->idlist = p;
  q->current = arr[0].next;
  return OK;
}

void Idseq_done(Idseq *q) {
  /* nothing to do! */
}

int32 Idseq_getid(Idseq *q) {
  const Idnode *arr = Array_data_const(&(q->idlist->idnode));
  int32 id = FAIL;
  if (q->current != -1) {
    id = arr[q->current].id;
    q->current = arr[q->current].next;
  }
  return id;
}
