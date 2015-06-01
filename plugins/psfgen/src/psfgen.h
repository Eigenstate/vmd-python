
/* psfgen.h
 * Defines set of data structures used in creation of molecule structures
 * Exported here so that new modules can be written to interface with psfgen
*/ 

#ifndef PSFGEN_H
#define PSFGEN_H

#include "topo_defs.h"
#include "topo_mol.h"
#include "stringhash.h"

/* psfgen-specific data */
struct psfgen_data {
  int id, in_use, all_caps;
  topo_defs *defs;
  topo_mol *mol;
  stringhash *aliases;
};
typedef struct psfgen_data psfgen_data;

#endif /* PSFGEN_H */
