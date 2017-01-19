/*
 * Copyright (C) 2007 by David J. Hardy.  All rights reserved.
 *
 * test_strmap2.c - More extensive test of Strmap.
 */

#include "nlbase/nlbase.h"

#define NELEMS(x)  (sizeof(x) / sizeof(x[0]))

/* array of color names */
static const char *Color[] = {
  "red",
  "blue",
  "yellow",
  "green",
  "orange",
  "purple",
  "black",
  "white",
  "gray",
  "brown",
  "magenta",
  "violet",
  "pink",
  "tan",
  "burgundy",
  "maroon",
  "navy",
  "lime",
  "lavender",
  "turquoise",
};

int main() {
  Strmap strmap;
  Strmap *t = &strmap;
  int32 i;
  int status = OK;

  if ((status = Strmap_init(t, 0)) != OK) {
    return ERROR(status);
  }

  NL_printf("+++ Insert first half of the hash table entries\n");
  for (i = 0;  i < NELEMS(Color)/2;  i++) {
    if (i != Strmap_insert(t, Color[i], i)) {
      return ERROR(FAIL);
    }
  }
  Strmap_print(t);

  NL_printf("+++ Insert second half of the hash table entries\n");
  for ( ;  i < NELEMS(Color);  i++) {
    if (i != Strmap_insert(t, Color[i], i)) {
      return ERROR(FAIL);
    }
  }
  Strmap_print(t);

  NL_printf("+++ Strmap stats:  %s\n", Strmap_stats(t));

  NL_printf("+++ Look up all of the hash table entries\n");
  for (i = 0;  i < NELEMS(Color);  i++) {
    if (i != Strmap_lookup(t, Color[i])) {
      return ERROR(FAIL);
    }
    NL_printf("      %20s   %3d\n", Color[i], i);
  }

  NL_printf("+++ Update all of the hash table entries\n");
  for (i = 0;  i < NELEMS(Color);  i++) {
    if (i != Strmap_update(t, Color[i], 2*i)) {
      return ERROR(FAIL);
    }
  }
  Strmap_print(t);

  NL_printf("+++ Delete first half of the hash table entries\n");
  for (i = 0;  i < NELEMS(Color)/2;  i++) {
    if (2*i != Strmap_remove(t, Color[i])) {
      return ERROR(FAIL);
    }
  }
  Strmap_print(t);

  NL_printf("+++ Delete second half of the hash table entries\n");
  for ( ;  i < NELEMS(Color);  i++) {
    if (2*i != Strmap_remove(t, Color[i])) {
      return ERROR(FAIL);
    }
  }
  Strmap_print(t);

  Strmap_done(t);

  return OK;
}
