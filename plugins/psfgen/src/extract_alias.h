
#ifndef EXTRACT_ALIAS_H
#define EXTRACT_ALIAS_H

#include "stringhash.h"

#define EXTRACT_ALIAS_FAIL -1

int extract_alias_residue_define(stringhash *h,
			const char *altres, const char *realres);

int extract_alias_atom_define(stringhash *h, const char *resname,
			const char *altatom, const char *realatom);

const char * extract_alias_residue_check(stringhash *h,
						const char *resname);

const char * extract_alias_atom_check(stringhash *h,
			const char *resname, const char *atomname);

#endif

