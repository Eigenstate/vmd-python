/*
 * Copyright (C) 2007 by David J. Hardy.  All rights reserved.
 *
 * error.h - NAMD-Lite error reporting.
 */

#ifndef NLBASE_ERROR_H
#define NLBASE_ERROR_H

#include "nlbase/io.h"

#ifdef __cplusplus
extern "C" {
#endif

  enum {
    OK =             0,  /* not an error */
    FAIL =          -1,  /* generic error condition */
    ERR_MEMALLOC =  -2,  /* memory allocation */
    ERR_FOPEN =     -3,  /* file open */
    ERR_FCLOSE =    -4,  /* file close */
    ERR_FREAD =     -5,  /* file reading */
    ERR_FWRITE =    -6,  /* file writing */
    ERR_INPUT =     -7,  /* user input */
    ERR_RANGE =     -8,  /* out-of-range */
    ERR_VALUE =     -9,  /* bad value */
    ERR_EXPECT =   -10   /* unexpected condition */
  };

  /* prints error message to stderr and returns errcode < 0 */
  int NL_error_report(int errcode);

  /* prints system error message to stderr and returns errcode < 0 */
  int NL_error_report_syscall(const char *name, int syserrno, int errcode);

/*
 * Use ASSERT(expr) to assert truth (nonzero-ness) of expression "expr".
 * Expands into code only if DEBUG is defined.
 */
#ifdef DEBUG
#define ASSERT(expr) \
  do { \
    if ( !(expr) ) { \
      NL_fprintf(stderr, "*** ASSERTION FAILED (%s,%d): %s\n", \
          NL_shorten_filename(__FILE__), __LINE__, #expr); \
      return FAIL; \
    } \
  } while (0)
#define ASSERT_P(expr) \
  do { \
    if ( !(expr) ) { \
      NL_fprintf(stderr, "*** ASSERTION FAILED (%s,%d): %s\n", \
          NL_shorten_filename(__FILE__), __LINE__, #expr); \
      return NULL; \
    } \
  } while (0)
#define ASSERT_V(expr) \
  do { \
    if ( !(expr) ) { \
      NL_fprintf(stderr, "*** ASSERTION FAILED (%s,%d): %s\n", \
          NL_shorten_filename(__FILE__), __LINE__, #expr); \
      return; \
    } \
  } while (0)
#else
#define ASSERT(expr)
#define ASSERT_P(expr)
#define ASSERT_V(expr)
#endif

/*
 * Use the ERROR() macro to report an error has occurred.
 * Call with the error code ERR_* value to indicate the kind of error.
 * The return value is always the error code (< 0) whether or not
 * ERROR_REPORT is defined to have error reported, e.g.:
 *
 *   if (n > toobig) return ERROR(ERR_RANGE);
 */
#if defined(DEBUG) || (defined(ERROR_REPORT) && !defined(SILENT))
#define ERROR(errcode) ( \
  NL_fprintf(stderr, "*** ERROR (%s,%d): ", \
      NL_shorten_filename(__FILE__), __LINE__), \
  NL_error_report(errcode) )
#define ERROR_SYSCALL(name,syserrno,errcode) ( \
  NL_fprintf(stderr, "*** ERROR (%s,%d): ", \
      NL_shorten_filename(__FILE__), __LINE__), \
  NL_error_report_syscall(name, syserrno, errcode) )
#else
#define ERROR(errcode)  (errcode)
#define ERROR_SYSCALL(name,syserrno,errcode)  (errcode)
#endif

#ifdef __cplusplus
}
#endif

#endif /* NLBASE_ERROR_H */
