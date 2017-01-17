/*
 * Copyright (C) 2007 by David J. Hardy.  All rights reserved.
 *
 * io.h - NAMD-Lite low-level I/O routines.
 */

#ifndef NLBASE_IO_H
#define NLBASE_IO_H

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#ifdef __cplusplus
extern "C" {
#endif

  /* Process message for INFO() macro. */
  int NL_print_info(const char *format, ...);

  /* When keeping INFO() macro silent. */
  int NL_ignore_info(const char *format, ...);

  /*
   * Shorten full pathnames when identifying source files to
   * include only filename and its immediate subdirectory.
   * Used to filter __FILE__ string by ERROR() and ASSERT() macros.
   */
  const char *NL_shorten_filename(const char *fname);

/*
 * Invoke C library I/O routines through wrappers so that they
 * can be redefined if needed.
 */
#define NL_FILE      FILE
#define NL_feof      feof
#define NL_fgets     fgets
#define NL_fputs     fputs
#define NL_fread     fread
#define NL_fwrite    fwrite
#define NL_fprintf   fprintf
#define NL_printf    printf
#define NL_vfprintf  vfprintf
#define NL_chdir     chdir

  NL_FILE *NL_fopen(const char *filename, const char *mode);
  int NL_fclose(NL_FILE *);

/*
 * Use the INFO() macro to print information to stdout.  The semantics
 * for arguments are exactly like printf() except that it returns OK.
 * Expands into code unless SILENT is defined.
 */
#if defined(DEBUG) || !defined(SILENT)
#define INFO \
  NL_fprintf(stdout, "# "), \
  NL_print_info
#else
#define INFO NL_ignore_info
#endif

/*
 * Macros for displaying text or variable contents.  These vanish when
 * debug is turned off.
 */
#if defined(DEBUG) && !defined(SILENT)
#define TEXT(t) \
  NL_fprintf(stderr, "DEBUG (%s,%d): \"%s\"\n", \
      NL_shorten_filename(__FILE__), __LINE__, t)
#define STR(str) \
  do { \
    const char *_str = (str); \
    (_str ? \
      NL_fprintf(stderr, "DEBUG (%s,%d): (%s)=\"%s\"\n", \
        NL_shorten_filename(__FILE__), __LINE__, #str, _str) : \
      NL_fprintf(stderr, "DEBUG (%s,%d): (%s)=(NULL)\n", \
        NL_shorten_filename(__FILE__), __LINE__, #str)); \
  } while (0)
#define PTR(p) \
  NL_fprintf(stderr, "DEBUG (%s,%d): (%s)=%p\n", \
      NL_shorten_filename(__FILE__), __LINE__, #p, (void *)(p))
#define INT(i) \
  NL_fprintf(stderr, "DEBUG (%s,%d): (%s)=%d\n", \
      NL_shorten_filename(__FILE__), __LINE__, #i, (int)(i))
#define HEX(x) \
  NL_fprintf(stderr, "DEBUG (%s,%d): (%s)=%#x\n", \
      NL_shorten_filename(__FILE__), __LINE__, #x, (int)(x))
#define FLT(x) \
  NL_fprintf(stderr, "DEBUG (%s,%d): (%s)=%g\n", \
      NL_shorten_filename(__FILE__), __LINE__, #x, (double)(x))
#define VEC(v) \
  NL_fprintf(stderr, "DEBUG (%s,%d): (%s)=(%g %g %g)\n", \
      NL_shorten_filename(__FILE__), __LINE__, #v, \
      (double)((v).x), (double)((v).y), (double)((v).z))
#else
#define TEXT(t)
#define STR(s)
#define PTR(p)
#define INT(n)
#define HEX(n)
#define FLT(x)
#define VEC(v)
#endif

#ifdef __cplusplus
}
#endif

#endif /* NLBASE_IO_H */
