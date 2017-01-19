/*
 * Copyright (C) 2007 by David J. Hardy.  All rights reserved.
 *
 * error.c - NAMD-Lite error reporting.
 */

#include <string.h>
#include "nlbase/error.h"


#undef NELEMS
#define NELEMS(a) (sizeof(a)/sizeof(a[0]))

const char *ErrorMessage[] = {
  "OK",
  "FAIL",
  "memory allocation",
  "can't open file",
  "can't close file",
  "can't read file",
  "can't write file",
  "illegal input",
  "out-of-range",
  "bad value",
  "unexpected condition",
};


/* prints error message to stderr and returns errcode < 0 */
int NL_error_report(int errcode) {
  ASSERT(errcode <= 0);
  if (errcode <= 0 && -errcode < NELEMS(ErrorMessage)) {
    NL_fprintf(stderr, "%s\n", ErrorMessage[-errcode]);
  }
  else {
    NL_fprintf(stderr, "invalid error code %d\n", errcode);
  }
  return errcode;
}


/* prints system error message to stderr and returns errcode < 0 */
int NL_error_report_syscall(const char *name, int syserrno, int errcode) {
  ASSERT(errcode <= 0);
  NL_fprintf(stderr, "%s: %s\n", name, strerror(syserrno));
  return errcode;
}

