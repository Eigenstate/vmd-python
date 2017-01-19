/*
 * Copyright (C) 2007 by David J. Hardy.  All rights reserved.
 *
 * types.h - Fundamental types.
 */

#ifndef NLBASE_TYPES_H
#define NLBASE_TYPES_H

#include <limits.h>

#ifdef __cplusplus
extern "C" {
#endif


  /**@brief Define types @c int32 and @c uint32
   * to be signed and unsigned 32-bit (4-byte) integers.
   * Define limit macros @c MAX_INT32 and @c MIN_INT32.
   * Define scanf() format specifier @c FMT_INT32.
   */
#if   ( INT_MAX == 2147483647 )

  typedef int int32;
  typedef unsigned int uint32;
#define MIN_INT32  INT_MIN
#define MAX_INT32  INT_MAX
#define MAX_UINT32 UINT_MAX
#define FMT_INT32  "%d"
#define FMT_UINT32 "%u"

#elif ( LONG_MAX == 2147483647 )

  typedef long int32;
  typedef unsigned long uint32;
#define MIN_INT32  LONG_MIN
#define MAX_INT32  LONG_MAX
#define MAX_UINT32 ULONG_MAX
#define FMT_INT32  "%ld"
#define FMT_UINT32 "%lu"

#elif ( SHRT_MAX == 2147483647 )

  typedef short int32;
  typedef unsigned short uint32;
#define MIN_INT32  SHRT_MIN
#define MAX_INT32  SHRT_MAX
#define MAX_UINT32 USHRT_MAX
#define FMT_INT32  "%hd"
#define FMT_UINT32 "%hu"

#endif


  /**@brief Define types @c int64 and @c uint64
   * to be signed and unsigned 64-bit (8-byte) integers.
   * Define limit macros @c MIN_INT64, @c MAX_INT64, and @c MAX_UINT64.
   * Define scanf() format specifier @c FMT_INT64.
   */
#ifdef _MSC_VER
  /* need to finish this for Microsoft C compilers */
  typedef __int64 int64;
#else
#if ( INT_MAX == 9223372036854775807LL )

  typedef int int64;
  typedef unsigned int uint64;
#define MIN_INT64  INT_MIN
#define MAX_INT64  INT_MAX
#define MAX_UINT64 UINT_MAX
#define FMT_INT64  "%d"
#define FMT_UINT64 "%u"

#elif ( LONG_MAX == 9223372036854775807LL )

  typedef long int64;
  typedef unsigned long uint64;
#define MIN_INT64  LONG_MIN
#define MAX_INT64  LONG_MAX
#define MAX_UINT64 ULONG_MAX
#define FMT_INT64  "%ld"
#define FMT_UINT64 "%lu"

#elif ( LLONG_MAX == 9223372036854775807LL )

  typedef long long int64;
  typedef unsigned long long uint64;
#define MIN_INT64  LLONG_MIN
#define MAX_INT64  LLONG_MAX
#define MAX_UINT64 ULLONG_MAX
#define FMT_INT64  "%lld"
#define FMT_UINT64 "%llu"

#endif
#endif


  /**@brief Define floating point types to better compare use of different
   * bit precision.  Define corresponding scanf() format specifiers.
   */
#if defined(ALL_SINGLE_PREC)

  typedef float  freal;
  typedef float  dreal;
#define FMT_FREAL  "%f"
#define FMT_DREAL  "%f"

#elif defined(ALL_DOUBLE_PREC)

  typedef double freal;
  typedef double dreal;
#define FMT_FREAL  "%lf"
#define FMT_DREAL  "%lf"

#else

  typedef float  freal;
  typedef double dreal;
#define FMT_FREAL  "%f"
#define FMT_DREAL  "%lf"

#endif


  /**@brief Boolean type is either @c TRUE or @c FALSE. */
  typedef enum boolean_t { FALSE=0, TRUE } boolean;
#define FMT_BOOLEAN  "%d"


  /**@brief Double-precision vector. */
  typedef struct dvec_t { dreal x, y, z; } dvec;

  /**@brief Single-precision vector. */
  typedef struct fvec_t { freal x, y, z; } fvec;

  /**@brief 32-bit integer vector. */
  typedef struct ivec_t { int32 x, y, z; } ivec;


#ifdef __cplusplus
}
#endif

#endif /* NLBASE_TYPES_H */
