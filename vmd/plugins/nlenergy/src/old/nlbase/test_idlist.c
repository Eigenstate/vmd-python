/*
 * Copyright (C) 2007 by David J. Hardy.  All rights reserved.
 *
 * test_array.c - Demonstrate use of array class.
 */

#include "nlbase/nlbase.h"

int main() {
  int s;

  Idlist idlist;
  Idlist *p = &idlist;

  Idlist_init(p);
  Idlist_print(p);

  Idlist_insert(p, 2);
  Idlist_print(p);

  Idlist_insert(p, 1);
  Idlist_print(p);

  Idlist_insert(p, 2);
  Idlist_print(p);

  Idlist_memsort(p);
  Idlist_print(p);

  Idlist_insert(p, 3);
  Idlist_print(p);

  if ((s=Idlist_remove(p, 1)) != OK) return ERROR(s);
  Idlist_print(p);

  if ((s=Idlist_remove(p, 3)) != OK) return ERROR(s);
  Idlist_print(p);

  Idlist_insert(p, 4);
  Idlist_insert(p, 5);
  Idlist_insert(p, 6);
  Idlist_print(p);

  Idlist_remove(p, 6);
  Idlist_print(p);

  Idlist_done(p);

  return 0;
}
