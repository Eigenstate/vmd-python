/*
 * Copyright (C) 2007 by David J. Hardy.  All rights reserved.
 *
 * io.c - NAMD-Lite low-level I/O routines.
 */

#include <string.h>
#include <errno.h>
#include "nlbase/error.h"
#include "nlbase/io.h"


NL_FILE *NL_fopen(const char *filename, const char *mode)
{
  FILE *fp;
  errno = 0;
  if (NULL==(fp = fopen(filename, mode))) {
    (void) ERROR_SYSCALL("fopen()", errno, ERR_FOPEN);
    return NULL;
  }
  return fp;
}


int NL_fclose(NL_FILE *fp)
{
  errno = 0;
  if (fclose(fp)) {
    return ERROR_SYSCALL("fclose()", errno, ERR_FCLOSE);
  }
  return OK;
}


int NL_print_info(const char *format, ...) {
#define INFO_STREAM  stdout
#define INFO_RETVAL  OK
#define INFO_LINBRK  "\n# "
#define INFO_LINBRK_LEN  3
  char *p = strchr(format, '\n');
  va_list ap;

  va_start(ap, format);
  if (p != NULL && *(p+1)=='\0') {
    NL_vfprintf(INFO_STREAM, format, ap);
  }
  else {
    char fmtbuf[256];
    char *dest = fmtbuf;
    const char *src = format;
    int n = sizeof(fmtbuf)-1;

    while (p != NULL) {
      for ( ;  n > 0 && src < p;  n--, src++, dest++)  *dest = *src;
      src = p+1;
      if (*src=='\0' || n < INFO_LINBRK_LEN) break;
      strcpy(dest, INFO_LINBRK);
      dest += INFO_LINBRK_LEN;
      n -= INFO_LINBRK_LEN;
      p = strchr(src, '\n');
    }
    for ( ;  n > 0 && *src != '\0';  n--, src++, dest++)  *dest = *src;
    strcpy(dest, "\n");
    NL_vfprintf(INFO_STREAM, fmtbuf, ap);
  }
  va_end(ap);

  return INFO_RETVAL;
}


int NL_ignore_info(const char *format, ...) {
  va_list ap;
  va_start(ap, format);
  va_end(ap);
  return OK;
}


/*
 * for improved appearance of filename on error output,
 * get rid of absolute prefix (which isn't really informative),
 * instead leaving just the first level source directory and filename
 *
 * no modification to fname is necessary, simply return pointer
 * closer to end of fname string
 */
const char *NL_shorten_filename(const char *fname) {
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
