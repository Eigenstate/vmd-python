/*
 * Copyright (C) 2007 by David J. Hardy.  All rights reserved.
 *
 * mgpot_error.c
 */

#include <stdlib.h>
#include <string.h>
#include "mgpot_error.h"


/*
 * print error message to standard error, always returns "FAIL",
 * otherwise semantics just like printf()
 */
int mgpot_error_message(const char *args, ...) {
  va_list ap;
  va_start(ap, args);
  vfprintf(stderr, args, ap);
  va_end(ap);
  return FAIL;
}


/*
 * for improved appearance of filename on error output,
 * get rid of absolute prefix (which isn't really informative),
 * instead leaving just the first level source directory and filename
 *
 * no modification to fname is necessary, simply return pointer
 * closer to end of fname string
 */
const char *mgpot_error_filename(const char *fname) {
  const char *p = strrchr(fname, '/');
  while (p > fname) {
    p--;
    if ('/' == *p) {
      p++;
      break;
    }
  }
  return (p != NULL ? p : fname);
}
