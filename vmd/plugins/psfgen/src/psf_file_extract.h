
#ifndef PSF_FILE_READ_H 
#define PSF_FILE_READ_H 

#include <stdio.h>
#include "topo_mol.h"

int psf_file_extract(topo_mol *mol, FILE *file, FILE *pdbfile, FILE *namdbinfile, FILE *velnamdbinfile,
                                void *, void (*print_msg)(void *, const char *));

#endif

