/*
 * Copyright (C) 2008 by David J. Hardy.  All rights reserved.
 */

#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "molfiles/config.h"

#undef  BUFLEN
#define BUFLEN  1024


static int remove_comment(char *line);
static int split_line(char **pkey, char **pval, char *line);


int Config_init(Config *c) {
  memset(c, 0, sizeof(Config));
  return OK;
}


void Config_done(Config *c) {
  if (c->file != NULL) {
    NL_fclose(c->file);
    c->file = NULL;
    c->pkey = NULL;
    c->pval = NULL;
  }
}


const char *Config_keyword(const Config *c) {
  return c->pkey;
}


const char *Config_value(const Config *c) {
  return c->pval;
}


int Config_read(Config *c, const char *fname) {
  if (NULL==(c->file = NL_fopen(fname, "r"))) return ERROR(ERR_FOPEN);
  return OK;
}


int Config_readline(Config *c) {
  const int32 bufsz = sizeof(c->buf);
  if (NULL == c->file) return FAIL;  /* end-of-file */
  /* make sure file line length does not exceed BUFLEN-2 characters */
  c->buf[bufsz-2] = c->buf[bufsz-3] = '\n';
  while (NL_fgets(c->buf, bufsz, c->file) != NULL) {
    if (c->buf[bufsz-2] != '\n' && c->buf[bufsz-3] != '\n') {
      return ERROR(ERR_INPUT);  /* input line length exceeded max */
    }
    else if (remove_comment(c->buf)) {
      return ERROR(ERR_INPUT);
    }
    else if (split_line(&(c->pkey), &(c->pval), c->buf)) {
      return OK;  /* found non-empty input line */
    }
  }
  if (NL_feof(c->file)) {
    int status = (NL_fclose(c->file) ? ERROR(ERR_FCLOSE) : FAIL);
    c->file = NULL;
    c->pkey = NULL;
    c->pval = NULL;
    return status;  /* FAIL indicates normal end-of-file */
  }
  return ERROR(ERR_FREAD);  /* otherwise a file reading error occurred */
}


int remove_comment(char *line)
{
  char *s;
  /* look for end of line comment, marked by '#' */
  if (NULL!=(s = strchr(line, (int)'#'))) {
    *s = '\0';
  }
  return OK;
}


int split_line(char **pkey, char **pval, char *line)
{
  char *key, *val, *eoln, *endkey;

  eoln = line + strlen(line);  /* points to nil-terminator */

  /* remove trailing white space */
  while (eoln > line && isspace(eoln[-1])) {
    eoln--;
  }
  *eoln = '\0';

  /* skip leading white space in front of keyword */
  key = line;
  while (isspace(*key) && *key != '\0') key++;

  /* find end of keyword */
  endkey = key;
  while (!isspace(*endkey) && *endkey != '=' && *endkey != '\0') endkey++;

  /* skip leading white space in front of value and optional assignment */
  val = endkey;
  while (isspace(*val) && *val != '\0') val++;
  if ('=' == *val) val++;
  while (isspace(*val) && *val != '\0') val++;

  *endkey = '\0';  /* mark end of keyword */

  /* return pointers to keyword and value */
  *pkey = key;
  *pval = val;

  return (key[0] != '\0');  /* return TRUE for non-empty line */
}


#define INT32  FMT_INT32
#define FREAL  FMT_FREAL
#define DREAL  FMT_DREAL
#define WS     " "
#define SEP    " , "
#define EXTRA  "%1s"

int Config_value_boolean(const Config *c, boolean *b) {
  if (NULL == c->pval) return ERROR(ERR_VALUE);
  if (strcasecmp(c->pval, "on") == 0
      || strcasecmp(c->pval, "yes") == 0) {
    *b = TRUE;
  }
  else if (strcasecmp(c->pval, "off") == 0
      || strcasecmp(c->pval, "no") == 0) {
    *b = FALSE;
  }
  else return FAIL;
  return OK;
}


int Config_value_int32(const Config *c, int32 *n) {
  char extra[4];
  if (NULL == c->pval) return ERROR(ERR_VALUE);
  if (sscanf(c->pval, INT32 EXTRA, n, extra) != 1) return FAIL;
  return OK;
}


int Config_value_dreal(const Config *c, dreal *r) {
  char extra[4];
  if (NULL == c->pval) return ERROR(ERR_VALUE);
  if (sscanf(c->pval, DREAL EXTRA, r, extra) != 1) return FAIL;
  return OK;
}


int Config_value_dvec(const Config *c, dvec *v) {
  char extra[4];
  if (NULL == c->pval) return ERROR(ERR_VALUE);
  if (sscanf(c->pval, DREAL WS DREAL WS DREAL EXTRA,
        &(v->x), &(v->y), &(v->z), extra) != 3
      && sscanf(c->pval, DREAL SEP DREAL SEP DREAL EXTRA,
        &(v->x), &(v->y), &(v->z), extra) != 3) {
    return FAIL;
  }
  return OK;
}
