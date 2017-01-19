/*
 * Copyright (C) 2008 by David J. Hardy.  All rights reserved.
 */

/**@file    molfiles/config.h
 * @brief   Read lines of a NAMD-style config file.
 * @author  David J. Hardy
 * @date    Apr. 2008
 *
 * Read NAMD-style config file ("keyword=value" lines),
 * perform lexical scan of value string.
 *
 * File format is simple:
 * - blank (and all white-space) lines are skipped
 * - '#' indicates comment through end of line
 * - meaningful lines contain:
 *     keyword = value
 *   where keyword is first non-white cluster of characters,
 *   the '=' is optional, and value is remaining characters
 *   with all leading and trailing white-space removed
 */

#ifndef MOLFILES_CONFIG_H
#define MOLFILES_CONFIG_H

#include "moltypes/moltypes.h"

#ifdef __cplusplus
extern "C" {
#endif

  typedef struct Config_t {
    char buf[512];
    char *pkey;
    char *pval;
    NL_FILE *file;
  } Config;

  int Config_init(Config *);
  void Config_done(Config *);

  int Config_read(Config *, const char *fname);

  /**@brief Obtain next keyword and value pair with return OK,
   * return FAIL for end-of-file or < FAIL for error. */
  int Config_readline(Config *);

  /* Access current keyword and value strings. */
  const char *Config_keyword(const Config *);
  const char *Config_value(const Config *);

  /* Retrieve numeric values from current value string. */
  int Config_value_boolean(const Config *, boolean *);
  int Config_value_int32  (const Config *, int32   *);
  int Config_value_dreal  (const Config *, dreal   *);
  int Config_value_dvec   (const Config *, dvec    *);

#ifdef __cplusplus
}
#endif

#endif /* MOLFILES_CONFIG_H */
