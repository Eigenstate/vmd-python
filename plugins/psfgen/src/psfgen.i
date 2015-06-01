
%module _psfgen

%{
#include "pdb_file_extract.h"
#include "psf_file_extract.h"
#include "topo_defs.h"
#include "topo_mol.h"
#include "topo_mol_output.h"
#include "charmm_parse_topo_defs.h"
#include "stringhash.h"
#include "extract_alias.h"

static void stdout_msg(void *v, const char *s) {
  printf("%s\n", s);
}
%}

%typemap(python,in) (const topo_mol_ident_t *targets, int ntargets) {
  int i;
  if (!PyList_Check($input)) {
    PyErr_SetString(PyExc_ValueError, "Expected a list of ident_t.");
    return NULL;
  }
  $2 = PyList_Size($input);
  $1 = (topo_mol_ident_t *)malloc($2*sizeof(topo_mol_ident_t));
  for (i=0; i<$2; i++) {
	  PyObject *anameobj;
    PyObject *identobj = PyList_GET_ITEM($input, i);
    if (!PyDict_Check(identobj)) {
      free($1);
      PyErr_SetString(PyExc_ValueError, "Expected a list of dictionaries.");
      return NULL;
    }
    $1[i].segid = PyString_AsString(PyDict_GetItemString(identobj, "segid"));
    $1[i].resid = PyString_AsString(PyDict_GetItemString(identobj, "resid"));
    if ((anameobj = PyDict_GetItemString(identobj, "aname"))) {
    	$1[i].aname = PyString_AsString(anameobj);
	  } else {
			$1[i].aname = 0;
		}
  }
}
%typemap(python,freearg) (const topo_mol_ident_t *targets, int ntargets) {
  free($1);
}

%typemap(python,in) const topo_mol_ident_t *target {
  PyObject *tmpobj;
  if (!PyDict_Check($input)) {
    PyErr_SetString(PyExc_ValueError, "Expected a dictionary.");
    return NULL;
  }
  $1 = (topo_mol_ident_t *)malloc(sizeof(topo_mol_ident_t));
	tmpobj = PyDict_GetItemString($input, "segid");
	$1->segid = tmpobj ? PyString_AsString(tmpobj) : 0;
	tmpobj = PyDict_GetItemString($input, "resid");
	$1->resid = tmpobj ? PyString_AsString(tmpobj) : 0;
	tmpobj = PyDict_GetItemString($input, "aname");
	$1->aname = tmpobj ? PyString_AsString(tmpobj) : 0;
}
%typemap(freearg) const topo_mol_ident_t * {
  free($1);
}

%typemap(in) void (*print_msg)(void *, const char *) {
  $1=stdout_msg;
}

%include pdb_file_extract.h
%include psf_file_extract.h
%include topo_defs.h
%include topo_mol.h
%include topo_mol_output.h
%include charmm_parse_topo_defs.h
%include stringhash.h
%include extract_alias.h


FILE *fopen(const char*, const char *);
int fclose(FILE *);
