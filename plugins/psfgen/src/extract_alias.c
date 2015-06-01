
#include <string.h>
#include <stdio.h>
#include "stringhash.h"
#include "extract_alias.h"

int extract_alias_residue_define(stringhash *h,
			const char *altres, const char *realres) {
  if ( ! h || ! altres || ! realres ||
       stringhash_insert(h,altres,realres) == STRINGHASH_FAIL ) {
    return EXTRACT_ALIAS_FAIL;
  }
  return 0;
}

int extract_alias_atom_define(stringhash *h, const char *resname,
			const char *altatom, const char *realatom) {
  char resatom[24];
  const char *resname2;
  if ( ! h || ! resname || ! altatom || ! realatom ) return EXTRACT_ALIAS_FAIL;
  if ( strlen(resname) + strlen(altatom) > 20 ) return EXTRACT_ALIAS_FAIL;
  sprintf(resatom,"%s %s",resname,altatom);
  if ( stringhash_insert(h,resatom,realatom) == STRINGHASH_FAIL ) {
    return EXTRACT_ALIAS_FAIL;
  }
  resname2 = extract_alias_residue_check(h,resname);
  if ( resname == resname2 ) return 0;
  resname = resname2;
  if ( strlen(resname) + strlen(altatom) > 20 ) return EXTRACT_ALIAS_FAIL;
  sprintf(resatom,"%s %s",resname,altatom);
  if ( stringhash_insert(h,resatom,realatom) == STRINGHASH_FAIL ) {
    return EXTRACT_ALIAS_FAIL;
  }
  return 0;
}

const char * extract_alias_residue_check(stringhash *h,
						const char *resname) {
  const char *realres;
  if ( ! h || ! resname ) return resname;
  realres = stringhash_lookup(h,resname);
  if ( realres != STRINGHASH_FAIL ) return realres;
  return resname;
}

const char * extract_alias_atom_check(stringhash *h,
			const char *resname, const char *atomname) {
  char resatom[24];
  const char *realatom;
  if ( ! h || ! resname || ! atomname ) return atomname;
  if ( strlen(resname) + strlen(atomname) < 20 ) {
    sprintf(resatom,"%s %s",resname,atomname);
    realatom = stringhash_lookup(h,resatom);
    if ( realatom != STRINGHASH_FAIL ) return realatom;
  }
  resname = extract_alias_residue_check(h,resname);
  if ( strlen(resname) + strlen(atomname) < 20 ) {
    sprintf(resatom,"%s %s",resname,atomname);
    realatom = stringhash_lookup(h,resatom);
    if ( realatom != STRINGHASH_FAIL ) return realatom;
  }
  return atomname;
}

