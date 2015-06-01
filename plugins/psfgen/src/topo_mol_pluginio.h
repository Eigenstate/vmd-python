
#ifndef TOPO_MOL_PLUGINIO_H
#define TOPO_MOL_PLUGINIO_H

#include <stdio.h>
#include "topo_mol.h"
#include "stringhash.h"

int topo_mol_read_plugin(topo_mol *mol, const char *pluginname,
                         const char *filename, 
                         const char *coorpluginname, const char *coorfilename,
                         const char *segid, stringhash *h, int all_caps,
                         int coordinatesonly, int residuesonly,
                         void *, void (*print_msg)(void *, const char *));

struct image_spec {
  int na, nb, nc;
  double ax, ay, az;
  double bx, by, bz;
  double cx, cy, cz;
};

int topo_mol_write_plugin(topo_mol *mol, const char *pluginname,
                          const char *filename, struct image_spec *images,
                          void *, void (*print_msg)(void *, const char *));

#endif

