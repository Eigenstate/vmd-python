/*
 * Copyright (C) 2008 by David J. Hardy.  All rights reserved.
 */

#include <math.h>
#include "force/fbonded.h"


int Fbonded_init(Fbonded *p, const Topology *t) {
  if (NULL==t || NULL==t->fprm) return ERROR(ERR_EXPECT);
  p->fprm = t->fprm;
  p->topo = t;
  return OK;
}

void Fbonded_done(Fbonded *p) {
  /* nothing to do! */
}

int Fbonded_setup(Fbonded *p, const Domain *d) {
  if (NULL==d) return ERROR(ERR_EXPECT);
  p->domain = d;
  return OK;
}

int Fbonded_eval(Fbonded *p, const dvec *pos, dvec *f, Energy *e) {
  int s;  /* error status */
  if ((s=Fbonded_eval_bond(p, pos, f, e)) != OK) return ERROR(s);
  if ((s=Fbonded_eval_angle(p, pos, f, e)) != OK) return ERROR(s);
  if ((s=Fbonded_eval_dihed(p, pos, f, e)) != OK) return ERROR(s);
  if ((s=Fbonded_eval_impr(p, pos, f, e)) != OK) return ERROR(s);
  return OK;
}
