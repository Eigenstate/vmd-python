
#ifndef TOPO_MOL_OUTPUT_H
#define TOPO_MOL_OUTPUT_H

#include <stdio.h>
#include "topo_mol.h"

int topo_mol_write_pdb(topo_mol *mol, FILE *file, void *, 
                                void (*print_msg)(void *, const char *));

int topo_mol_write_namdbin(topo_mol *mol, FILE *file, void *, 
                                void (*print_msg)(void *, const char *));

int topo_mol_write_psf(topo_mol *mol, FILE *file, int charmmfmt, int nocmap,
                        void *, void (*print_msg)(void *, const char *));

#endif

