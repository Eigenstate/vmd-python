/* energy.c */

#include <string.h>
#include "moltypes/energy.h"


int Energy_init(Energy *e) {
  memset(e, 0, sizeof(Energy));
  return OK;
}

void Energy_done(Energy *e) {
  /* nothing to do! */
}
