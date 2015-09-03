/*
 * Copyright (C) 2007 by David J. Hardy.  All rights reserved.
 *
 * mgpot_error.h
 */

#ifndef MGPOT_ERROR_H
#define MGPOT_ERROR_H

#include <stdarg.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

  /*
   * Return value to indicate failure of a function call.
   */
#undef FAIL
#define FAIL  (-1)

  /*
   * Use the ERROR() macro to print an error message to stderr.
   * The semantics for arguments are exactly like printf().
   * The return value is always FAIL, so it can be used concisely
   * for the code here, e.g.:
   *
   *   if (n > toobig) {
   *     return ERROR("n=%d is too big\n", n);
   *   }
   */
#undef ERROR
#define ERROR \
  mgpot_error_message("ERROR (%s, line %d): ", \
      mgpot_error_filename(__FILE__), __LINE__), \
  mgpot_error_message

  /*
   * Semantics are like printf().  Output is to stderr.  Always returns FAIL.
   * (Use through ERROR() macro for consistent appearance.)
   */
  int mgpot_error_message(const char *args, ...);

  /*
   * Get rid of full pathnames when identifying source files,
   * include only filename and its immediate subdirectory.
   * (Used to filter __FILE__ string by ERROR() and ASSERT() macros.)
   */
  const char *mgpot_error_filename(const char *fname);

  /*
   * Use ASSERT(expr) to assert truth (non-zero-ness) of expr.
   * Expands into code only if DEBUG macro is defined.
   *
   * If DEBUG macro is defined, then the redefined DEBUG() macro
   * will expand code placed inside, e.g. DEBUG( printf("val=%d\n", val); ).
   * Otherwise, it is redefined to make the code vanish.
   *
   * DEBUGGING macro is used as a flag for addition of code alternatives,
   * where debugging mode is determined by its definition.
   */
#if defined(DEBUG) || defined(DEBUGGING)
#undef DEBUG
#define DEBUG(X) X
#undef DEBUGGING
#define DEBUGGING
#undef ASSERT
#define ASSERT(expr) \
  do { \
    if ( !(expr) ) { \
      mgpot_error_message("ASSERTION FAILED (%s, line %d): \"%s\"\n", \
          mgpot_error_filename(__FILE__), __LINE__, #expr); \
      return FAIL; \
    } \
  } while (0)
#else
#undef DEBUG
#define DEBUG(X)
#undef DEBUGGING
#undef ASSERT
#define ASSERT(expr)
#endif

#ifdef __cplusplus
}
#endif

#endif /* MGPOT_ERROR_H */
