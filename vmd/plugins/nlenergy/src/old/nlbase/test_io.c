#include "nlbase/nlbase.h"

int main() {
  INFO("\nThe meaning of %s, %s, and %s is %d.\n\n",
     "life", "the universe", "everything", 42);
  /* WARN("Look out, %s.", "something is wrong"); */
  (void) ERROR(ERR_MEMALLOC);
  return OK;
}
