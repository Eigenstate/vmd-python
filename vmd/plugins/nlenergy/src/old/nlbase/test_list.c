#include "nlbase/list.h"

int print_list(const List *ls, const char *label) {
  int stat;
  ListIter li;
  const int32 *pelem;
  if ((stat=ListIter_init(&li, ls)) != OK) return ERROR(stat);
  NL_printf("%s(length=%2d):", label, ls->length(ls));
  while ((pelem=li.get(&li)) != NULL) {
    NL_printf(" %d", (int)(*pelem));
  }
  NL_printf("\n");
  ListIter_done(&li);
  return OK;
}


int main() {
  List a, b;
  int32 i, j;

  List_init(&a);
  List_init(&b);

  print_list(&a, "a");
  for (i = 0;  i < 10;  i++) {
    a.insert(&a, 2*i);
  }
  print_list(&a, "a");

  a.erase(&a);
  for (i = 0, j = 7;  i < 10;  i++, j = (j+7)%10) {
    print_list(&a, "a");
    a.insert(&a, 2*j);
  }
  print_list(&a, "a");

  print_list(&b, "b");
  for (i = 5;  i > 0;  i--) {
    b.insert(&b, i);
  }
  print_list(&b, "b");

  List_copy(&b, &a);
  print_list(&a, "a");
  print_list(&b, "b");

  a.remove(&a, 0);
  a.remove(&a, 18);
  a.remove(&a, 7);
  a.remove(&a, 8);
  print_list(&a, "a");
  print_list(&b, "b");

  b.erase(&b);
  print_list(&b, "b");

  List_done(&a);
  List_done(&b);
  return OK;
}
