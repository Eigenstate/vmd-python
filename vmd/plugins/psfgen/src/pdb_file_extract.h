
#ifndef PDB_FILE_EXTRACT_H
#define PDB_FILE_EXTRACT_H

#include <stdio.h>
#include "stringhash.h"
#include "topo_mol.h"

int pdb_file_extract_residues(topo_mol *mol, FILE *file, stringhash *h, int all_caps,
                                void *, void (*print_msg)(void *,const char *));

int pdb_file_extract_coordinates(topo_mol *mol, FILE *file, FILE *namdbinfile,
                                const char *segid, stringhash *h, int all_caps,
                                void *,void (*print_msg)(void *,const char *));

#endif

