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

/**@file    random.h
 * @brief   Random number library.
 * @author  Martin Birgmeier
 * @date    1993
 *
 * The @c Random_t class is a linear congruential generator for
 * a pseudo-random number stream.  The functional portion of the
 * code, originally by Martin Birgmeier, is taken from NAMD.
 * It has been turned into a class to provide thread-safe use.
 */


/*
 * source from NAMD Random.h, modified from C++
 */

#ifndef NLBASE_RANDOM_H
#define NLBASE_RANDOM_H

#include "nlbase/types.h"

#ifdef __cplusplus
extern "C" {
#endif


  /**@brief Random number class.
   *
   * Members should be treated as private.
   * Stores random sequence state.
   */
  typedef struct Random_t {
    double second_gaussian;
    int64 second_gaussian_waiting;
    int64 rand48_seed;
    int64 rand48_mult;
    int64 rand48_add;
  } Random;


  /**@brief Constructor.
   *
   * Initializes random sequence using a seed of 0.
   */
  int Random_init(Random *r);


  /**@brief Constructor.
   *
   * @param[in] seed  Seeds random sequence.
   *
   * Initializes random sequence using specified seed.
   */
  int Random_initseed(Random *r, unsigned long seed);


  /**@brief Destructor.
   */
  void Random_done(Random *r);


  /**@brief Split streams.
   *
   * @param[in] iStream     Identifies stream to take.
   * @param[in] numStreams  Indicates number of streams.
   *
   * Split into @c numStreams number of streams and take stream numbered
   * @c iStream.
   */
  void Random_split(Random *r, int iStream, int numStreams);


  /**@brief Uniform distribution.
   *
   * Determine a random number uniformly distributed between 0 and 1.
   *
   * @return the random number.
   */
  double Random_uniform(Random *r);


  /**@brief Gaussian distribution.
   *
   * Determine a random number from a standard Gaussian distribution.
   *
   * @return the random number.
   */
  double Random_gaussian(Random *r);


  /**@brief Vector of Gaussian random numbers.
   *
   * Generate a 3-vector of Gaussian random numbers.
   *
   * @return the random vector.
   */
  dvec Random_gaussian_vector(Random *r);


  /**@brief Random integer.
   *
   * Generate a random long integer in range 0..2^31-1.
   *
   * @return the random integer.
   */
  long Random_integer(Random *r);

#ifdef __cplusplus
}
#endif

#endif  /* NLBASE_RANDOM_H */
