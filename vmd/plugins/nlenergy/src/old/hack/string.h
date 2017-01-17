/*
 * Copyright (C) 2008 by David J. Hardy.  All rights reserved.
 *
 * string.h - Provides a nicer interface to heap-allocated C-strings.
 */

#ifndef NLBASE_STRING_H
#define NLBASE_STRING_H

#include "nlbase/types.h"
#include "nlbase/array.h"

#ifdef __cplusplus
extern "C" {
#endif

  typedef struct String_t {
    char *p;
  } String;

  int String_init(String *);
  void String_done(String *);
  int String_copy(String *dest, const String *src);
  int String_erase(String *);  /* set to empty string */

  int32 String_length(const String *);
  const char *String_get(const String *);
  int String_set(String *, const char *);
  int String_set_substr(String *, const char *, int len);
  int String_append(String *, const char *);
  int String_prefix(String *, const char *);

  /* Retrieve numeric value from C-string. */
  int String_boolean(boolean *, const char *);
  int String_int32  (int32   *, const char *);
  int String_dreal  (dreal   *, const char *);
  int String_dvec   (dvec    *, const char *);


  typedef struct Strarray_t {
    Objarr sarr;
  } Strarray;

  int Strarray_init(Strarray *);
  void Strarray_done(Strarray *);

  int Strarray_append_cstr(Strarray *, const char *);
  int Strarray_append(Strarray *, const String *);
  int Strarray_remove(Strarray *, String *);

  String *Strarray_data(Strarray *);
  const String *Strarray_data_const(const Strarray *);
  int32 Strarray_length(const Strarray *);
  String *Strarray_elem(Strarray *, size_t i);
  const String *Strarray_elem_const(const Strarray *, size_t i);

#ifdef __cplusplus
}
#endif

#endif /* NLBASE_STRING_H */
