#include "nlbase/nlbase.h"

int main() {
  int *pn;
  int i;
  if ((pn = NL_calloc(10, sizeof(int)))==NULL) {
    return ERROR(ERR_MEMALLOC);
  }
  pn[0] = 1;
  for (i = 1;  i < 10;  i++) {
    pn[i] = pn[i-1] + 1;
  }
  if ((pn = NL_realloc(pn, 20*sizeof(int)))==NULL) {
    return ERROR(ERR_MEMALLOC);
  }
  for (i = 10;  i < 20;  i++) {
    pn[i] = pn[i-1] + 1;
  }
  NL_printf("array: ");
  for (i = 0;  i < 20;  i++) {
    NL_printf("%d ", i);
  }
  NL_printf("\n");
  NL_free(pn);
  return OK;
}
