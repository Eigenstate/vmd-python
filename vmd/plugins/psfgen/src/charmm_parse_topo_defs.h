
#ifndef CHARMM_PARSE_TOPO_DEFS_H
#define CHARMM_PARSE_TOPO_DEFS_H

#include <stdio.h>
#include "topo_defs.h"

int charmm_parse_topo_defs(topo_defs *defs, FILE *file, int all_caps, void *v,
                                void (*print_msg)(void *,const char *));

#endif

