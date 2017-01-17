/*
 * Copyright (c) 1993 Martin Birgmeier
 * All rights reserved.
 *
 * You may redistribute unmodified or modified versions of this source
 * code provided that the above copyright notice and this and the
 * following conditions are retained.
 *
 * This software is provided ``as is'', and comes with no warranties
 * of any kind. I shall in no event be liable for anything that happens
 * to anyone/anything when using this software.
 */

/*
 * source from NAMD Random.h, modified from C++
 */

#include <math.h>
#include "nlbase/error.h"
#include "nlbase/random.h"

#ifdef _MSC_VER
#define INT64_LITERAL(X) X ## i64
#else
#define INT64_LITERAL(X) X ## LL
#endif

#define RAND48_SEED   INT64_LITERAL(0x00001234abcd330e)
#define RAND48_MULT   INT64_LITERAL(0x00000005deece66d)
#define RAND48_ADD    INT64_LITERAL(0x000000000000000b)
#define RAND48_MASK   INT64_LITERAL(0x0000ffffffffffff)

/* internal routines */
static void skip(Random *r);


/* default constructor */
int Random_init(Random *r)
{
  Random_initseed(r, 0);
  r->rand48_seed = RAND48_SEED;
  return OK;
}


/* reinitialize with seed */
int Random_initseed(Random *r, unsigned long seed)
{
  r->second_gaussian = 0;
  r->second_gaussian_waiting = 0;
  r->rand48_seed = seed & INT64_LITERAL(0x00000000ffffffff);
  r->rand48_seed = r->rand48_seed << 16;
  r->rand48_seed |= RAND48_SEED & INT64_LITERAL(0x0000ffff);
  r->rand48_mult = RAND48_MULT;
  r->rand48_add = RAND48_ADD;
  return OK;
}


/* destructor */
void Random_done(Random *r) {
  /* nothing to do! */
}


/* advance generator by one (seed = seed * mult + add, to 48 bits) */
void skip(Random *r)
{
  r->rand48_seed = ( r->rand48_seed * r->rand48_mult
      + r->rand48_add ) & RAND48_MASK;
}


/* split into numStreams different steams and take stream iStream */
void Random_split(Random *r, int iStream, int numStreams)
{
  int64 save_seed;
  int64 new_add;
  int i;

  /* make sure that numStreams is odd to ensure maximum period */
  numStreams |= 1;

  /* iterate to get to the correct stream */
  for ( i = 0; i < iStream; ++i ) skip(r);

  /* save seed and add so we can use skip() for our calculations */
  save_seed = r->rand48_seed;

  /* calculate c *= ( 1 + a + ... + a^(numStreams-1) ) */
  r->rand48_seed = r->rand48_add;
  for ( i = 1; i < numStreams; ++i ) skip(r);
  new_add = r->rand48_seed;

  /* calculate a = a^numStreams */
  r->rand48_seed = r->rand48_mult;
  r->rand48_add  = 0;
  for ( i = 1; i < numStreams; ++i ) skip(r);
  r->rand48_mult = r->rand48_seed;

  r->rand48_add  = new_add;
  r->rand48_seed = save_seed;

  r->second_gaussian = 0;
  r->second_gaussian_waiting = 0;
}


/* return a number uniformly distributed between 0 and 1 */
double Random_uniform(Random *r)
{
  const double exp48 = ( 1.0 / (double)(INT64_LITERAL(1) << 48) );

  skip(r);
  return ( (double) r->rand48_seed * exp48 );
}


/* return a number from a standard gaussian distribution */
double Random_gaussian(Random *r)
{
  double fac, rsq, v1, v2;

  if (r->second_gaussian_waiting) {
    r->second_gaussian_waiting = 0;
    return r->second_gaussian;
  }
  else {
    /*
     * rsq >= 1.523e-8 ensures abs result < 6
     * make sure we are within unit circle
     */
    do {
      v1 = 2.0 * Random_uniform(r) - 1.0;
      v2 = 2.0 * Random_uniform(r) - 1.0;
      rsq = v1*v1 + v2*v2;
    } while (rsq >=1. || rsq < 1.523e-8);
    fac = sqrt(-2.0 * log(rsq)/rsq);
    /*
     * Now make the Box-Muller transformation to get two normally
     * distributed random numbers.  Save one and return the other.
     */
    r->second_gaussian_waiting = 1;
    r->second_gaussian = v1 * fac;
    return v2 * fac;
  }
}


/* return a vector of gaussian random numbers */
dvec Random_gaussian_vector(Random *r)
{
  dvec v;

  v.x = Random_gaussian(r);
  v.y = Random_gaussian(r);
  v.z = Random_gaussian(r);
  return v;
}


/* return a random long within 0..2^31-1 */
long Random_integer(Random *r)
{
  skip(r);
  return ( ( r->rand48_seed >> 17 ) & INT64_LITERAL(0x000000007fffffff) );
}


#if 0
/* randomly order an array of whatever */
template <class Elem> void reorder(Elem *a, int n) {
  for ( int i = 0; i < (n-1); ++i ) {
    int ie = i + ( integer() % (n-i) );
    if ( ie == i ) continue;
    const Elem e = a[ie];
    a[ie] = a[i];
    a[i] = e;
  }
}
#endif
