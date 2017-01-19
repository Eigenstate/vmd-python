#include "nlbase/error.h"
#include "nlbase/numeric.h"


/*
 * Stein's algorithm.
 * http://en.wikipedia.org/wiki/Binary_GCD_algorithm
 */
uint32 Numeric_gcd(uint32 a, uint32 b) {
  int32 shift;

  /* GCD(0,x) := x */
  if (a==0 || b==0)  return (a | b);

  /* Let shift := lg K, where K is greatest power of 2 dividing both a and b */
  for (shift = 0;  ((a | b) & 1) == 0;  shift++) {
    a >>= 1;
    b >>= 1;
  }

  while ((a & 1) == 0)  a >>= 1;

  /* From here on, a is always odd */
  do {
    while ((b & 1) == 0)  b >>= 1;

    /* Now a and b are both odd, so diff(a, b) is even.
     * Let a = min(a, b) and b = diff(a, b) / 2 */
    if (a < b) {
      b -= a;
    }
    else {
      uint32 diff = a - b;
      a = b;
      b = diff;
    }
    b >>= 1;

  } while (b != 0);

  return a << shift;
}
