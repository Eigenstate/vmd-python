/*
 * Copyright (C) 2008 by David J. Hardy.  All rights reserved.
 *
 * string.c - Provides a nicer interface to heap-allocated C-strings.
 */

#include <string.h>
#include "nlbase/error.h"
#include "nlbase/io.h"
#include "nlbase/mem.h"
#include "nlbase/string.h"

int String_init(String *s) {
  s->p = NULL;
  return OK;
}

void String_done(String *s) {
  NL_free(s->p);
  s->p = NULL;
}

int String_copy(String *dest, const String *src) {
  NL_free(dest->p);
  dest->p = NL_strdup(src->p);
  if (NULL == dest->p && src->p != NULL) return ERROR(ERR_MEMALLOC);
  return OK;
}

int String_erase(String *s) {
  NL_free(s->p);
  s->p = NL_strdup("");
  if (NULL == s->p) return ERROR(ERR_MEMALLOC);
  return OK;
}

int32 String_length(const String *s) {
  if (NULL == s->p) return FAIL;  /* == -1 */
  return strlen(s->p);
}

const char *String_get(const String *s) {
  return s->p;
}

int String_set(String *s, const char *cstr) {
  NL_free(s->p);
  s->p = NL_strdup(cstr);
  if (NULL == s->p && cstr != NULL) return ERROR(ERR_MEMALLOC);
  return OK;
}

int String_set_substr(String *s, const char *cstr, int len) {
  if (NULL == cstr || len < 0) return ERROR(ERR_VALUE);
  NL_free(s->p);
  s->p = (char *) NL_malloc(len+1);  /* room for nil-terminator */
  if (NULL == s->p) return ERROR(ERR_MEMALLOC);
  strncpy(s->p, cstr, len);
  s->p[len] = 0;
  return OK;
}

int String_append(String *s, const char *cstr) {
  char *pold = s->p;
  int32 len = (s->p != NULL ? strlen(s->p) : 0) + strlen(cstr);
  s->p = (char *) NL_malloc(len+1);  /* room for nil-terminator */
  if (NULL == s->p) {
    NL_free(pold);
    return ERROR(ERR_MEMALLOC);
  }
  if (pold != NULL) strcpy(s->p, pold);
  else s->p[0] = 0;
  strcat(s->p, cstr);
  NL_free(pold);
  return OK;
}

int String_prefix(String *s, const char *cstr) {
  char *pold = s->p;
  int32 len = (s->p != NULL ? strlen(s->p) : 0) + strlen(cstr);
  s->p = (char *) NL_malloc(len+1);  /* room for nil-terminator */
  if (NULL == s->p) {
    NL_free(pold);
    return ERROR(ERR_MEMALLOC);
  }
  strcpy(s->p, cstr);
  if (pold != NULL) strcat(s->p, pold);
  NL_free(pold);
  return OK;
}


#define INT32  FMT_INT32
#define FREAL  FMT_FREAL
#define DREAL  FMT_DREAL
#define WS     " "
#define SEP    " , "
#define EXTRA  "%1s"

int String_boolean(boolean *b, const char *cstr) {
  if (NULL == cstr) return ERROR(ERR_EXPECT);
  if (strcasecmp(cstr, "on") == 0
      || strcasecmp(cstr, "yes") == 0) {
    *b = TRUE;
  }
  else if (strcasecmp(cstr, "off") == 0
      || strcasecmp(cstr, "no") == 0) {
    *b = FALSE;
  }
  else return ERROR(ERR_VALUE);
  return OK;
}

int String_int32(int32 *n, const char *cstr) {
  char extra[4];
  if (NULL == cstr) return ERROR(ERR_EXPECT);
  if (sscanf(cstr, INT32 EXTRA, n, extra) != 1) return ERROR(ERR_VALUE);
  return OK;
}

int String_dreal(dreal *r, const char *cstr) {
  char extra[4];
  if (NULL == cstr) return ERROR(ERR_EXPECT);
  if (sscanf(cstr, DREAL EXTRA, r, extra) != 1) return ERROR(ERR_VALUE);
  return OK;
}

int String_dvec(dvec *v, const char *cstr) {
  char extra[4];
  if (NULL == cstr) return ERROR(ERR_EXPECT);
  if (sscanf(cstr, DREAL WS DREAL WS DREAL EXTRA,
        &(v->x), &(v->y), &(v->z), extra) != 3
      && sscanf(cstr, DREAL SEP DREAL SEP DREAL EXTRA,
        &(v->x), &(v->y), &(v->z), extra) != 3) {
    return ERROR(ERR_VALUE);
  }
  return OK;
}


static int vs_init(void *v) {
  int s;
  if ((s=String_init((String *) v)) != OK) return ERROR(s);
  return OK;
}

static void vs_done(void *v) {
  String_done((String *) v);
}

static int vs_copy(void *vdest, const void *vsrc) {
  int s;
  if ((s=String_copy((String *) vdest, (const String *) vsrc)) != OK) {
    return ERROR(s);
  }
  return OK;
}

static int vs_erase(void *v) {
  int s;
  if ((s=String_erase((String *) v)) != OK) return ERROR(s);
  return OK;
}

int Strarray_init(Strarray *p) {
  int s;
  if ((s=Objarr_init(&(p->sarr), sizeof(String),
          vs_init, vs_done, vs_copy, vs_erase)) != OK) {
    return ERROR(s);
  }
  return OK;
}

void Strarray_done(Strarray *p) {
  Objarr_done(&(p->sarr));
}

int Strarray_append_cstr(Strarray *p, const char *cstr) {
  String str;
  int s;
  if ((s=String_init(&str)) != OK) return ERROR(s);
  else if ((s=String_set(&str, cstr)) != OK) {
    String_done(&str);
    return ERROR(s);
  }
  else if ((s=Objarr_append(&(p->sarr), &str)) != OK) {
    String_done(&str);
    return ERROR(s);
  }
  String_done(&str);
  return OK;
}

int Strarray_append(Strarray *p, const String *pstr) {
  return Objarr_append(&(p->sarr), pstr);
}

int Strarray_remove(Strarray *p, String *pstr) {
  return Objarr_remove(&(p->sarr), pstr);
}

String *Strarray_data(Strarray *p) {
  return Objarr_data(&(p->sarr));
}

const String *Strarray_data_const(const Strarray *p) {
  return Objarr_data_const(&(p->sarr));
}

int32 Strarray_length(const Strarray *p) {
  return Objarr_length(&(p->sarr));
}

String *Strarray_elem(Strarray *p, size_t i) {
  return Objarr_elem(&(p->sarr), i);
}

const String *Strarray_elem_const(const Strarray *p, size_t i) {
  return Objarr_elem_const(&(p->sarr), i);
}
