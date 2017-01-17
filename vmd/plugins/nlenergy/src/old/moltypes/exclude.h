/*
 * Copyright (C) 2008 by David J. Hardy.  All rights reserved.
 */

/**@file    moltypes/exclude.h
 * @brief   Determine all excluded interactions.
 * @author  David J. Hardy
 * @date    May 2008
 */

#ifndef MOLTYPES_EXCLUDE_H
#define MOLTYPES_EXCLUDE_H

#include "moltypes/topology.h"

#ifdef __cplusplus
extern "C" {
#endif

  typedef struct Exclude_t {
    const ForcePrm *fprm;
    const Topology *topo;

    Objarr exclx;   /**< explicit exclusions and self interactions */
    Objarr excl12;  /**< 1-2 interactions */
    Objarr excl13;  /**< 1-2, 1-3 interactions */
    Objarr excl14;  /**< 1-2, 1-3, 1-4 interactions */
    Objarr only14;  /**< 1-4 interactions only */

    Objarr excllist;    /**< final exclusion list based on policy */
    Objarr scal14list;  /**< final list of scaled 1-4 interactions */

  } Exclude;

  int Exclude_init(Exclude *, const Topology *);
  void Exclude_done(Exclude *);

  int Exclude_setup(Exclude *);

  const Idlist *Exclude_excllist(const Exclude *, int32);
  const Idlist *Exclude_scal14list(const Exclude *, int32);

  int Exclude_pair(const Exclude *, int32, int32);

#ifdef __cplusplus
}
#endif

#endif /* MOLTYPES_EXCLUDE_H */
