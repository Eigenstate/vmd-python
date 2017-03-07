/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2016 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: py_atomsel.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.23 $      $Date: 2016/11/28 03:05:08 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   New Python atom selection interface
 ***************************************************************************/

#include "py_commands.h"
#include "AtomSel.h"
#include "VMDApp.h"
#include "MoleculeList.h"
#include "SymbolTable.h"
#include "Measure.h"
#include "SpatialSearch.h"

// support for compiling against old (2.4.x and below) versions of Python
#if PY_VERSION_HEX < ((2<<24)|(5<<16))
typedef int Py_ssize_t;
#endif

typedef struct {
  PyObject_HEAD
  AtomSel *atomSel;
  VMDApp *app;
} PyAtomSelObject;

// forward declaration
static int atomsel_Check(PyObject *obj);

// return molecule for atomsel, or NULL and set exception
DrawMolecule *get_molecule(PyAtomSelObject *a) {
  int molid = a->atomSel->molid();
  DrawMolecule *mol = a->app->moleculeList->mol_from_id(molid);
  if (!mol) {
    PyErr_Format(PyExc_ValueError, "invalid molid '%d'", molid);
  }
  return mol;
}

static char *atomsel_doc = (char *)
    "atomsel( selection, molid = top, frame = now) -> new selection object\n"
    ;

static void
atomsel_dealloc( PyAtomSelObject *obj ) {
  delete obj->atomSel;
  ((PyObject *)(obj))->ob_type->tp_free((PyObject *)obj);
}

static PyObject *
atomsel_repr( PyAtomSelObject *obj ) {
  AtomSel *sel = obj->atomSel;
  char *s = new char[strlen(sel->cmdStr) + 100];
  sprintf(s, "atomsel('%s', molid=%d, frame=%d)",
      sel->cmdStr, sel->molid(), sel->which_frame);
#if PY_MAJOR_VERSION >= 3
  PyObject *result = PyUnicode_FromString(s);
#else
  PyObject *result = PyString_FromString(s);
#endif
  delete [] s;
  return result;
}

static PyObject *
atomsel_str( PyAtomSelObject *obj ) {
#if PY_MAJOR_VERSION >= 3
  return PyUnicode_FromString(obj->atomSel->cmdStr);
#else
  return PyString_FromString(obj->atomSel->cmdStr);
#endif
}

// Create a new atom selection
static PyObject *
atomsel_new( PyTypeObject *type, PyObject *args, PyObject *kwds ) {

  const char *sel = "all";
  int molid = -1;  // if not overridden, use top molecule
  int frame = -1;  // corresponds to current frame

  static char *kwlist[] = { (char *)"selection", (char *)"molid",
                            (char *)"frame", NULL };

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|sii", kwlist,
        &sel, &molid, &frame))
    return NULL;

  VMDApp *app = get_vmdapp();
  if (molid < 0) molid = app->molecule_top();
  DrawMolecule *mol = app->moleculeList->mol_from_id(molid);
  if (!mol) {
    PyErr_Format(PyExc_ValueError, "invalid molid '%d'", molid);
    return NULL;
  }

  AtomSel *atomSel = new AtomSel(app->atomSelParser, mol->id());
  atomSel->which_frame = frame;
  int parse_result;

  Py_BEGIN_ALLOW_THREADS
    parse_result = atomSel->change(sel, mol);
  Py_END_ALLOW_THREADS

  if (parse_result == AtomSel::NO_PARSE) {
    PyErr_Format(PyExc_ValueError, "cannot parse atom selection text '%s'", sel);
    delete atomSel;
    return NULL;
  }

  PyObject *obj = type->tp_alloc(type, 0);
  if (obj == NULL) {
    delete atomSel;
    return NULL; // memory allocation error
  }
  ((PyAtomSelObject *)obj)->app = app;
  ((PyAtomSelObject *)obj)->atomSel = atomSel;
  return obj;
}

static char *get_doc = (char *)
    "get( attrib ) -> corresponding attrib values for selected atoms."
    ;

static PyObject *atomsel_get(PyAtomSelObject *a, PyObject *keyobj) {

  // FIXME: make everything in this pipeline const so that it's
  // thread-safe.
  VMDApp *app = a->app;
  AtomSel *atomSel = a->atomSel;
  const int num_atoms = atomSel->num_atoms;
  DrawMolecule *mol;
  if (!(mol = get_molecule(a))) return NULL;

#if PY_MAJOR_VERSION >= 3
  const char *attr = PyUnicode_AsUTF8(keyobj);
#else
  const char *attr = PyString_AsString(keyobj);
#endif

  if (!attr) return NULL;

  //
  // Check for a valid attribute
  //
  SymbolTable *table = app->atomSelParser;
  int attrib_index = table->find_attribute(attr);
  if (attrib_index == -1) {
    PyErr_SetString(PyExc_ValueError, "unknown atom attribute");
    return NULL;
  }
  SymbolTableElement *elem = table->fctns.data(attrib_index);
  if (elem->is_a != SymbolTableElement::KEYWORD &&
      elem->is_a != SymbolTableElement::SINGLEWORD) {
    PyErr_SetString(PyExc_ValueError, "attribute is not a keyword or singleword");
    return NULL;
  }

  //
  // fetch the data
  //

  int *flgs = atomSel->on;
  atomsel_ctxt context(table, mol, atomSel->which_frame, attr);
  PyObject *newlist = PyList_New(atomSel->selected);

  if (elem->is_a == SymbolTableElement::SINGLEWORD) {
    int *tmp = new int[num_atoms];
    memcpy(tmp, atomSel->on, num_atoms*sizeof(int));
    elem->keyword_single(&context, num_atoms, tmp);
    int j=0;
    for (int i=0; i<num_atoms; i++) {
      if (flgs[i]) {
        PyObject *val = tmp[i] ? Py_True : Py_False;
        Py_INCREF(val);
        PyList_SET_ITEM(newlist, j++, val);
      }
    }
    delete [] tmp;
  } else {
    switch(table->fctns.data(attrib_index)->returns_a) {
      case (SymbolTableElement::IS_STRING):
      {
        const char **tmp= new const char *[num_atoms];
        elem->keyword_string(&context, num_atoms, tmp, flgs);
        int j=0;
        for (int i=0; i<num_atoms; i++) {
          if (flgs[i]) {
#if PY_MAJOR_VERSION >= 3
            PyList_SET_ITEM(newlist, j++, PyUnicode_FromString(tmp[i]));
#else
            PyList_SET_ITEM(newlist, j++, PyString_FromString(tmp[i]));
#endif
          }
        }
        delete [] tmp;
      }
      break;
      case (SymbolTableElement::IS_INT):
      {
        int *tmp = new int[num_atoms];
        elem->keyword_int(&context, num_atoms, tmp, flgs);
        int j=0;
        for (int i=0; i<num_atoms; i++) {
          if (flgs[i]) {
#if PY_MAJOR_VERSION >= 3
            PyList_SET_ITEM(newlist, j++, PyLong_FromLong(tmp[i]));
#else
            PyList_SET_ITEM(newlist, j++, PyInt_FromLong(tmp[i]));
#endif
          }
        }
        delete [] tmp;
      }
      break;
      case (SymbolTableElement::IS_FLOAT):
      {
        double *tmp = new double[num_atoms];
        elem->keyword_double(&context, num_atoms, tmp, flgs);
        int j=0;
        for (int i=0; i<num_atoms; i++) {
          if (flgs[i])
            PyList_SET_ITEM(newlist, j++, PyFloat_FromDouble(tmp[i]));
        }
        delete [] tmp;
      }
      break;
    } // end switch
  }   // end else
  return newlist;
}

static char *set_doc = (char *)
    "set( attrib, val ) -> set attrib values for selected atoms.\n"
    "  val must be either a single value, or a sequence of values, one for\n"
    "   each atom in selection.\n" ;

static PyObject *atomsel_set(PyAtomSelObject *a, PyObject *args) {

  // FIXME: make everything in this pipeline const so that it's
  // thread-safe.
  VMDApp *app = a->app;
  AtomSel *atomSel = a->atomSel;
  const int num_atoms = atomSel->num_atoms;
  DrawMolecule *mol;
  if (!(mol = get_molecule(a))) return NULL;

  const char *attr;
  PyObject *val;
  if (!PyArg_ParseTuple(args, "sO", &attr, &val)) return NULL;

  //
  // Check for a valid attribute
  //
  SymbolTable *table = app->atomSelParser;
  int attrib_index = table->find_attribute(attr);
  if (attrib_index == -1) {
    PyErr_SetString(PyExc_ValueError, "unknown atom attribute");
    return NULL;
  }
  SymbolTableElement *elem = table->fctns.data(attrib_index);
  if (elem->is_a != SymbolTableElement::KEYWORD &&
      elem->is_a != SymbolTableElement::SINGLEWORD) {
    PyErr_SetString(PyExc_ValueError, "attribute is not a keyword or singleword");
    return NULL;
  }
  if (!table->is_changeable(attrib_index)) {
    PyErr_SetString(PyExc_ValueError, "attribute is not modifiable");
    return NULL;
  }

  //
  // check that we have been given either one value or one for each selected
  // atom
  //

  const int num_selected = atomSel->selected;
  PyObject *fastval = NULL;
  int nvals = 1;
#if PY_MAJOR_VERSION >= 3
  if (PySequence_Check(val) && !PyUnicode_Check(val)) {
#else
  if (PySequence_Check(val) && !PyString_Check(val)) {
#endif
    nvals = PySequence_Size(val);
    if (nvals != 1 && nvals != num_selected) {
      PyErr_SetString(PyExc_ValueError, "wrong number of items");
      return NULL;
    }
    fastval = PySequence_Fast(val, "Invalid values");
    if (!fastval) return NULL;
  }

  int *flgs = atomSel->on;

  //
  // set the data
  //

  // singlewords can never be set, so macro is NULL.
  atomsel_ctxt context(table, mol, atomSel->which_frame, NULL);
  if (elem->returns_a == SymbolTableElement::IS_INT) {
    int *list = new int[num_atoms];
    if (fastval) {
      int j=0;
      for (int i=0; i<num_atoms; i++) {
        if (!flgs[i]) continue;
#if PY_MAJOR_VERSION >= 3
        int ival = (int)PyLong_AsLong(PySequence_Fast_GET_ITEM(fastval, j++));
#else
        int ival = (int)PyInt_AsLong(PySequence_Fast_GET_ITEM(fastval, j++));
#endif
        // FIXME: check for error!
        list[i] = ival;
      }
    } else {
#if PY_MAJOR_VERSION >= 3
      int ival = (int)PyLong_AsLong(val);
#else
      int ival = (int)PyInt_AsLong(val);
#endif
      // FIXME: check for error!
      for (int i=0; i<num_atoms; i++) {
        if (flgs[i]) list[i] = ival;
      }
    }
    elem->set_keyword_int(&context, num_atoms, list, flgs);
    delete [] list;

  } else if (elem->returns_a == SymbolTableElement::IS_FLOAT) {
    double *list = new double[num_atoms];
    if (fastval) {
      int j=0;
      for (int i=0; i<num_atoms; i++) {
        if (!flgs[i]) continue;
        float dval = (float)PyFloat_AsDouble(PySequence_Fast_GET_ITEM(fastval, j++));
        list[i] = dval;
      }
    } else {
      float dval = (float)PyFloat_AsDouble(val);
      for (int i=0; i<num_atoms; i++) {
        if (flgs[i]) list[i] = dval;
      }
    }
    elem->set_keyword_double(&context, num_atoms, list, flgs);
    delete [] list;


  } else if (elem->returns_a == SymbolTableElement::IS_STRING) {
    const char **list = new const char *[num_atoms];
    if (fastval) {
      int j=0;
      for (int i=0; i<num_atoms; i++) {
        if (!flgs[i]) continue;
#if PY_MAJOR_VERSION >= 3
        const char *sval = PyUnicode_AsUTF8(PySequence_Fast_GET_ITEM(fastval, j++));
#else
        const char *sval = PyString_AsString(PySequence_Fast_GET_ITEM(fastval, j++));
#endif
        list[i] = sval;
      }
    } else {
#if PY_MAJOR_VERSION >= 3
      const char *sval = PyUnicode_AsUTF8(val);
#else
      const char *sval = PyString_AsString(val);
#endif
      for (int i=0; i<num_atoms; i++) {
        if (flgs[i]) list[i] = sval;
      }
    }
    elem->set_keyword_string(&context, num_atoms, list, flgs);
    delete [] list;
  }

  // Recompute the color assignments if certain atom attributes are changed.
  if (!strcmp(attr, "name") ||
      !strcmp(attr, "type") ||
      !strcmp(attr, "resname") ||
      !strcmp(attr, "chain") ||
      !strcmp(attr, "segid") ||
      !strcmp(attr, "segname"))
    app->moleculeList->add_color_names(mol->id());

  mol->force_recalc(DrawMolItem::SEL_REGEN | DrawMolItem::COL_REGEN);

  Py_XDECREF(fastval);
  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject *getframe(PyAtomSelObject *a, void *) {

  AtomSel *atomSel = a->atomSel;
  DrawMolecule *mol;
  if (!(mol = get_molecule(a))) return NULL;

#if PY_MAJOR_VERSION >= 3
  return PyLong_FromLong(atomSel->which_frame);
#else
  return PyInt_FromLong(atomSel->which_frame);
#endif
}

static int setframe(PyAtomSelObject *a, PyObject *frameobj, void *) {

  AtomSel *atomSel = a->atomSel;
  DrawMolecule *mol;
  if (!(mol = get_molecule(a))) return -1;

#if PY_MAJOR_VERSION >= 3
  int frame = PyLong_AsLong(frameobj);
#else
  int frame = PyInt_AsLong(frameobj);
#endif
  if (frame < 0 && PyErr_Occurred()) return -1;
  if (frame != AtomSel::TS_LAST && frame != AtomSel::TS_NOW &&
      (frame < 0 || frame >= mol->numframes())) {
    PyErr_SetString(PyExc_ValueError, "Invalid frame");
    return -1;
  }
  atomSel->which_frame = frame;
  return 0;
}

static char *frame_doc = (char *)
    "frame -- the animation frame referenced by the selection.\n"
    "   Special values: -1 = current frame, -2 = last frame.\n"
    "NOTE: changing the frame does not update the selection;\n"
    "   use update() to do that.\n";

static char *update_doc = (char *)
    "update() -> Recompute atoms in selection; if the selection is\n"
       "distance-based (e.g., if it uses 'within'), changes to the frame\n"
       "will not be reflected in the selected atoms until this method\n"
       "is called.\n" ;

static PyObject *py_update(PyAtomSelObject *a) {

  AtomSel *atomSel = a->atomSel;
  DrawMolecule *mol;
  if (!(mol = get_molecule(a))) return NULL;

  Py_BEGIN_ALLOW_THREADS
    atomSel->change(NULL, mol);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}

static char *write_doc = (char *)
  "write(filetype, filename) -> None\n"
  "Write the atoms in the selection to a file of the given type.";

static PyObject *py_write(PyAtomSelObject *a, PyObject *args, PyObject *kwds) {

  AtomSel *atomSel = a->atomSel;
  DrawMolecule *mol;
  if (!(mol = get_molecule(a))) return NULL;
  const char *filetype, *filename;
  static char *kwlist[] = { (char *)"filetype", (char *)"filename", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "ss", kwlist,
        &filetype, &filename))
    return NULL;

  int frame=-1;
  switch (atomSel -> which_frame) {
    case AtomSel::TS_NOW:  frame = mol->frame(); break;
    case AtomSel::TS_LAST: frame = mol->numframes()-1; break;
    default:               frame = atomSel->which_frame; break;
  }
  if (frame < 0) frame = 0;
  else if (frame >= mol->numframes()) frame = mol->numframes()-1;
  // Write the requested atoms to the file
  FileSpec spec;
  spec.first = frame;
  spec.last = frame;
  spec.stride = 1;
  spec.waitfor = -1;
  spec.selection = atomSel->on;
  int rc;
  VMDApp *app = get_vmdapp();
  Py_BEGIN_ALLOW_THREADS
    rc = app->molecule_savetrajectory(mol->id(), filename, filetype, &spec);
  Py_END_ALLOW_THREADS

  if (rc < 0) {
    PyErr_SetString(PyExc_ValueError, "Unable to write selection to file.");
    return NULL;
  }
  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject *getbonds(PyAtomSelObject *a, void *) {

  AtomSel *atomSel = a->atomSel;
  DrawMolecule *mol;
  if (!(mol = get_molecule(a))) return NULL;

  PyObject *newlist = PyList_New(atomSel->selected);

  int k=0;
  for (int i=0; i< atomSel->num_atoms; i++) {
    if (!atomSel->on[i]) continue;
    const MolAtom *atom = mol->atom(i);
    PyObject *bondlist = PyList_New(atom->bonds);
    for (int j=0; j<atom->bonds; j++) {
#if PY_MAJOR_VERSION >= 3
      PyList_SET_ITEM(bondlist, j, PyLong_FromLong(atom->bondTo[j]));
#else
      PyList_SET_ITEM(bondlist, j, PyInt_FromLong(atom->bondTo[j]));
#endif
    }
    PyList_SET_ITEM(newlist, k++, bondlist);
  }
  return newlist;
}

static int setbonds(PyAtomSelObject *a, PyObject *obj, void *) {

  AtomSel *atomSel = a->atomSel;
  DrawMolecule *mol;
  if (!(mol = get_molecule(a))) return -1;

  if (!PySequence_Check(obj) || PySequence_Size(obj) != atomSel->selected) {
    PyErr_SetString(PyExc_ValueError,
      (char *)"setbonds: atomlist and bondlist must have the same size");
    return -1;
  }
  PyObject *fastbonds = PySequence_Fast(obj, "Argument to setbonds must be sequence");
  if (!fastbonds) return -1;

  int ibond = 0;
  mol->force_recalc(DrawMolItem::MOL_REGEN); // many reps ignore bonds
  for (int i=0; i<atomSel->num_atoms; i++) {
    if (!atomSel->on[i]) continue;
    MolAtom *atom = mol->atom(i);

    PyObject *atomids = PySequence_Fast_GET_ITEM(fastbonds, ibond++);
    if (!PyList_Check(atomids)) continue;
    int numbonds = PyList_Size(atomids);
    int k=0;
    for (int j=0; j<numbonds; j++) {
#if PY_MAJOR_VERSION >= 3
      int bond = PyLong_AsLong(PyList_GET_ITEM(atomids, j));
#else
      int bond = PyInt_AsLong(PyList_GET_ITEM(atomids, j));
#endif
      if (bond >= 0 && bond < mol->nAtoms) {
        atom->bondTo[k++] = bond;
      }
    }
    atom->bonds = k;
  }
  Py_DECREF(fastbonds);
  return 0;
}

static char *bonds_doc = (char *)
    "bonds - for each atom in selection, a list of the indices\n"
    "  of the bonded atoms.\n";

static PyObject *getmolid(PyAtomSelObject *a, void *) {
  AtomSel *atomSel = a->atomSel;
#if PY_MAJOR_VERSION >= 3
  return PyLong_FromLong(atomSel->molid());
#else
  return PyInt_FromLong(atomSel->molid());
#endif
}

static char *molid_doc = (char *)
    "molid - The id of the molecule this selection is associated with.\n";


static PyGetSetDef atomsel_getset[] = {
  { (char *)"frame", (getter)getframe, (setter)setframe, frame_doc, NULL },
  { (char *)"bonds", (getter)getbonds, (setter)setbonds, bonds_doc, NULL },
  { (char *)"molid", (getter)getmolid, (setter)NULL, molid_doc, NULL },
  { NULL },
};


// macro(name, selection)
static PyObject *macro(PyObject *self, PyObject *args, PyObject *keywds) {
  char *name = NULL, *selection = NULL;
  static char *kwlist[] = {
    (char *)"name", (char *)"selection", NULL
  };
  if (!PyArg_ParseTupleAndKeywords(args, keywds, (char *)"|ss:atomsel.macro", kwlist, &name, &selection))
    return NULL;

  if (selection && !name) {
    PyErr_SetString(PyExc_ValueError, (char *)"Must specify name for macro");
    return NULL;
  }
  SymbolTable *table = get_vmdapp()->atomSelParser;
  if (!name && !selection) {
    // return list of defined macros
    PyObject *macrolist = PyList_New(0);
    for (int i=0; i<table->num_custom_singleword(); i++) {
      const char *s = table->custom_singleword_name(i);
      if (s && strlen(s))
#if PY_MAJOR_VERSION >= 3
        PyList_Append(macrolist, PyUnicode_FromString(s));
#else
        PyList_Append(macrolist, PyString_FromString(s));
#endif
    }
    return macrolist;
  }
  if (name && !selection) {
    // return definition of macro
    const char *s = table->get_custom_singleword(name);
    if (!s) {
      PyErr_SetString(PyExc_ValueError, (char *)"No macro for given name");
      return NULL;
    }
#if PY_MAJOR_VERSION >= 3
    return PyUnicode_FromString(s);
#else
    return PyString_FromString(s);
#endif
  }
  // must have both and selection.  Define a new macro.
  if (!table->add_custom_singleword(name, selection)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Unable to create new macro");
    return NULL;
  }
  Py_INCREF(Py_None);
  return Py_None;
}

// delmacro(name)
static PyObject *delmacro(PyObject *self, PyObject *args) {
  char *name;
  if (!PyArg_ParseTuple(args, (char *)"s:atomsel.delmacro", &name))
    return NULL;
  if (!get_vmdapp()->atomSelParser->remove_custom_singleword(name)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Unable to remove macro");
    return NULL;
  }
  Py_INCREF(Py_None);
  return Py_None;
}

// extract vector from sequence object.  Return success.
static int py_get_vector(PyObject *matobj, int n, float *vec) {

  if (!PySequence_Check(matobj) || PySequence_Size(matobj) != n) {
    PyErr_SetString(PyExc_ValueError, "vector has incorrect size");
    return 0;
  }
  PyObject *fastval = PySequence_Fast(matobj, "Invalid sequence");
  if (!fastval) return 0;

  for (int i=0; i<n; i++) {
    vec[i] = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(fastval, i));
    if (PyErr_Occurred()) {
      Py_DECREF(fastval);
      return 0;
    }
  }
  Py_DECREF(fastval);
  return 1;
}

static PyObject *inverse(PyObject *self, PyObject *matobj) {
  Matrix4 mat;
  if (!py_get_vector(matobj, 16, mat.mat)) return NULL;
  if (mat.inverse()) {
    PyErr_SetString(PyExc_ValueError, "Matrix is singular.");
    return NULL;
  }
  PyObject *result = PyTuple_New(16);
  for (int i=0; i<16; i++) {
    PyTuple_SET_ITEM(result, i, PyFloat_FromDouble(mat.mat[i]));
  }
  return result;
}

#if PY_MAJOR_VERSION >= 3
#define SYMBOL_TABLE_FUNC(funcname, elemtype) \
static PyObject *funcname(PyObject *self) { \
  VMDApp *app = get_vmdapp(); \
  PyObject *result = PyList_New(0); \
  SymbolTable *table = app->atomSelParser; \
  int i, n = table->fctns.num(); \
  for (i=0; i<n; i++) \
    if (table->fctns.data(i)->is_a == elemtype) \
      PyList_Append(result, PyUnicode_FromString(table->fctns.name(i))); \
  return result; \
}
#else
#define SYMBOL_TABLE_FUNC(funcname, elemtype) \
static PyObject *funcname(PyObject *self) { \
  VMDApp *app = get_vmdapp(); \
  PyObject *result = PyList_New(0); \
  SymbolTable *table = app->atomSelParser; \
  int i, n = table->fctns.num(); \
  for (i=0; i<n; i++) \
    if (table->fctns.data(i)->is_a == elemtype) \
      PyList_Append(result, PyString_FromString(table->fctns.name(i))); \
  return result; \
}
#endif

SYMBOL_TABLE_FUNC(keywords, SymbolTableElement::KEYWORD)
SYMBOL_TABLE_FUNC(booleans, SymbolTableElement::SINGLEWORD)
SYMBOL_TABLE_FUNC(functions, SymbolTableElement::FUNCTION)
SYMBOL_TABLE_FUNC(stringfunctions, SymbolTableElement::STRINGFCTN)

#undef SYMBOL_TABLE_FUNC

// utility routine for parsing weight values.  Uses the sequence protocol
// so that sequence-type structure (list, tuple) will be accepted.
static float *parse_weight(AtomSel *sel, PyObject *wtobj) {
  float *weight = new float[sel->selected];
  if (!wtobj || wtobj == Py_None) {
    for (int i=0; i<sel->selected; i++) weight[i] = 1.0f;
    return weight;
  }

  PyObject *seq = PySequence_Fast(wtobj, (char *)"weight must be a sequence.");
  if (!seq) return NULL;
  if (PySequence_Size(seq) != sel->selected) {
    Py_DECREF(seq);
    PyErr_SetString(PyExc_ValueError, "weight must be same size as selection.");
    delete [] weight;
    return NULL;
  }
  for (int i=0; i<sel->selected; i++) {
    double tmp = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(seq, i));
    if (PyErr_Occurred()) {
      PyErr_SetString(PyExc_ValueError, "non-floating point value found in weight.");
      Py_DECREF(seq);
      delete [] weight;
      return NULL;
    }
    weight[i] = (float)tmp;
  }
  Py_DECREF(seq);
  return weight;
}

static char *minmax_doc = (char *)
  "minmax() -> (min, max)\n"
  "Return minimum and maximum coordinates for selected atoms.";

static PyObject *minmax(PyAtomSelObject *a, PyObject *withradii) {

  AtomSel *atomSel = a->atomSel;
  DrawMolecule *mol;
  if (!(mol = get_molecule(a))) return NULL;
  const float *radii = NULL;
  if (withradii && PyObject_IsTrue(withradii)) {
    radii = mol->extraflt.data("radius");
  }

  const float *pos = atomSel->coordinates(a->app->moleculeList);
  float min[3], max[3];
  int rc = measure_minmax(atomSel->num_atoms, atomSel->on, pos, radii, min, max);
  if (rc < 0) {
    PyErr_SetString(PyExc_ValueError, measure_error(rc));
    return NULL;
  }
  return Py_BuildValue("[f,f,f],[f,f,f]",
      min[0], min[1], min[2], max[0], max[1], max[2]);
}

static char *center_doc = (char *)
  "center(weight=None) -> (x, y, z)\n"
  "Return a tuple corresponding to the center of the selection,\n"
  "optionally weighted by weight.";

static PyObject *center(PyAtomSelObject *a, PyObject *args, PyObject *kwds) {

  AtomSel *atomSel = a->atomSel;
  DrawMolecule *mol;
  if (!(mol = get_molecule(a))) return NULL;
  PyObject *weightobj = NULL;

  static char *kwlist[] = { "weight", NULL };

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist, &weightobj))
    return NULL;
  float *weight = parse_weight(atomSel, weightobj);
  if (!weight) return NULL;

  float cen[3];
  // compute center
  int ret_val = measure_center(atomSel,
      atomSel->coordinates(a->app->moleculeList),
      weight, cen);
  delete [] weight;
  if (ret_val < 0) {
    PyErr_SetString(PyExc_ValueError, measure_error(ret_val));
    return NULL;
  }
  return Py_BuildValue("(f,f,f)", cen[0], cen[1], cen[2]);
}

static float *parse_two_selections_return_weight(PyAtomSelObject *a,
    PyObject *args, AtomSel **othersel) {

  PyObject *other;
  PyObject *weightobj = NULL;

  AtomSel *atomSel = a->atomSel;
  DrawMolecule *mol;
  if (!(mol = get_molecule(a))) return NULL;

  if (!PyArg_ParseTuple(args, "O|O", &other, &weightobj))
    return NULL;
  float *weight = parse_weight(atomSel, weightobj);
  if (!weight) return NULL;
  if (!atomsel_Check(other)) return NULL;
  AtomSel *sel2 = ((PyAtomSelObject *)other)->atomSel;
  if (atomSel->selected != sel2->selected) {
    PyErr_SetString(PyExc_ValueError, "Selections must have same number of atoms");
    return NULL;
  }
  *othersel = sel2;
  return weight;
}

static char *rmsd_doc = (char *)
  "rmsd(sel, weight=None) -> rms distance between selections.\n"
  "  Selections must have the same number of atoms.\n"
  "  Weight must be None or list of same size as selections.";

static PyObject *py_rmsd(PyAtomSelObject *a, PyObject *args) {

  AtomSel *sel2;
  float *weight = parse_two_selections_return_weight(a, args, &sel2);
  if (!weight) return NULL;

  float rmsd;
  int rc = measure_rmsd(a->atomSel, sel2, a->atomSel->selected,
      a->atomSel->coordinates(a->app->moleculeList),
      sel2->coordinates(a->app->moleculeList),
      weight, &rmsd);
  delete [] weight;
  if (rc < 0) {
    PyErr_SetString(PyExc_ValueError, measure_error(rc));
    return NULL;
  }
  return PyFloat_FromDouble(rmsd);
}
static char* rmsf_doc = (char *)
  "rmsf([first=0, last=-1, step=1])\n"
  " Measures the rmsf along a trajectory.\n"
  " By default, goes over all frames\n";

static PyObject *py_rmsf(PyAtomSelObject *a, PyObject *args, PyObject *kwds) {
  int first=0, last=-1, step=1;
  static char *kwlist[] = {(char *)"first",(char *)"last",(char *)"step", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iii", kwlist,
        &first, &last, &step)) { return NULL;}
  float *rmsf = new float[a->atomSel->selected];
  int ret_val = measure_rmsf(a->atomSel, a->app->moleculeList, first, last, step, rmsf);
  if (ret_val < 0) {
    PyErr_SetString(PyExc_RuntimeError, measure_error(ret_val));
    PyErr_Print();
    delete [] rmsf;
    return NULL;
  }
  //Build the python list.
  PyObject* returnlist = PyList_New(a->atomSel->selected);
  for (int i = 0; i < a->atomSel->selected; i++)
    PyList_SetItem(returnlist, i, Py_BuildValue("f", rmsf[i]));
  delete [] rmsf;
  return returnlist;
}

static char *rgyr_doc = (char *)
  "rgyr(sel, weight=None) -> radius of gyration of a selection.\n"
  "  Weight must be None or list of same size as selection.";

static PyObject *py_rgyr(PyAtomSelObject *a, PyObject *args, PyObject *kwds) {
  PyObject *weightobj = NULL;

  static char *kwlist[] = { "weight", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist, &weightobj))
    return NULL;
  float *weight = parse_weight(a->atomSel, weightobj);
  if (!weight) return NULL;

  float rgyr;
  int rc = measure_rgyr(a->atomSel, a->app->moleculeList, weight, &rgyr);
  delete [] weight;
  if (rc < 0) {
    PyErr_SetString(PyExc_ValueError, measure_error(rc));
    return NULL;
  }
  return PyFloat_FromDouble(rgyr);
}
static char *fit_doc = (char *)
    "fit(sel, weight=None) -> transformation matrix\n"
    "  Compute and return the transformation matrix for the RMS alignment\n"
    "  of the selection to sel.  The format of the matrix is a 16-element\n"
    "  tuple suitable for passing to the move() method\n"
    "  (Column major/fortran ordering).\n"
    "  Weight must be None or list of same size as selections.";


static PyObject *py_fit(PyAtomSelObject *a, PyObject *args) {

  AtomSel *sel2;
  float *weight = parse_two_selections_return_weight(a, args, &sel2);
  if (!weight) return NULL;

  Matrix4 mat;
  int rc = measure_fit(a->atomSel, sel2,
      a->atomSel->coordinates(a->app->moleculeList),
      sel2->coordinates(a->app->moleculeList),
      weight, NULL, &mat);
  delete [] weight;
  if (rc < 0) {
    PyErr_SetString(PyExc_ValueError, measure_error(rc));
    return NULL;
  }
  PyObject *result = PyTuple_New(16);
  for (int i=0; i<16; i++) {
    PyTuple_SET_ITEM(result, i, PyFloat_FromDouble(mat.mat[i]));
  }
  return result;
}

static char *moveby_doc = (char *)
    "moveby( vec ) -> shift the selection by the three-element vector vec.";

static PyObject *py_moveby(PyAtomSelObject *a, PyObject *vecobj) {

  AtomSel *atomSel = a->atomSel;
  DrawMolecule *mol;
  if (!(mol = get_molecule(a))) return NULL;

  float offset[3];
  if (!py_get_vector(vecobj, 3, offset)) return NULL;
  float *pos = atomSel->coordinates(a->app->moleculeList);
  if (!pos) {
    PyErr_SetString(PyExc_ValueError, "No coordinates");
    return NULL;
  }
  for (int i=0; i<atomSel->num_atoms; i++) {
    if (atomSel->on[i]) {
      vec_add(pos, pos, offset);
    }
    pos += 3;
  }
  mol->force_recalc(DrawMolItem::MOL_REGEN);
  Py_INCREF(Py_None);
  return Py_None;
}

static char *move_doc = (char *)
    "move( matrix ) -> apply coordinate transformation to selection.\n"
    "  matrix should be of the form returned by fit()\n"
    "  (Column major/fortran ordering)";

static PyObject *py_move(PyAtomSelObject *a, PyObject *matobj) {

  AtomSel *atomSel = a->atomSel;
  DrawMolecule *mol;
  if (!(mol = get_molecule(a))) return NULL;

  Matrix4 mat;
  if (!py_get_vector(matobj, 16, mat.mat)) return NULL;

  int err;
  if ((err = measure_move(
          atomSel,
          atomSel->coordinates(a->app->moleculeList),
          mat)) != MEASURE_NOERR) {
    PyErr_SetString(PyExc_ValueError, measure_error(err));
    return NULL;
  }
  mol->force_recalc(DrawMolItem::MOL_REGEN);
  Py_INCREF(Py_None);
  return Py_None;
}

static char *contacts_doc = (char *)
  "contacts(sel, cutoff) -> contact pairs\n"
  "Return two lists, whose corresponding elements contain atom indices\n"
  "in selection that are within cutoff of sel, but not directly bonded.\n";

// Find all atoms p in sel1 and q in sel2 within the cutoff.
static PyObject *contacts(PyAtomSelObject *a, PyObject *args) {
  AtomSel *sel1 = a->atomSel;
  DrawMolecule *mol;
  if (!(mol = get_molecule(a))) return NULL;

  PyObject *obj2;
  float cutoff;
  if (!PyArg_ParseTuple(args, "Of", &obj2, &cutoff))
    return NULL;
  if (!atomsel_Check(obj2)) return NULL;
  if (!(get_molecule((PyAtomSelObject *)obj2))) return NULL;
  AtomSel *sel2 = ((PyAtomSelObject *)obj2)->atomSel;

  const float *ts1 = sel1->coordinates(a->app->moleculeList);
  const float *ts2 = sel2->coordinates(a->app->moleculeList);
  if (!ts1 || !ts2) {
    PyErr_SetString(PyExc_ValueError, "No coordinates in selection");
    return NULL;
  }

  GridSearchPair *pairlist = vmd_gridsearch3(
      ts1, sel1->num_atoms, sel1->on,
      ts2, sel2->num_atoms, sel2->on,
      cutoff, -1, (sel1->num_atoms + sel2->num_atoms) * 27);

  GridSearchPair *p, *tmp;
  PyObject *list1 = PyList_New(0);
  PyObject *list2 = PyList_New(0);
  PyObject *tmp1;
  PyObject *tmp2;
  for (p=pairlist; p != NULL; p=tmp) {
    // throw out pairs that are already bonded
    MolAtom *a1 = mol->atom(p->ind1);
    if (sel1->molid() != sel2->molid() || !a1->bonded(p->ind2)) {
    //Needed to avoid a memory leak. Append increments the reference count of whatever gets added to it, but so does PyInt_FromLong.
    //Without a decref, the integers created never have their reference count go to zero, and you leak
    //memory. Really bad if you call contacts repeatedly and the result is large. :(
#if PY_MAJOR_VERSION >= 3
      tmp1 = PyLong_FromLong(p->ind1);
      tmp2 = PyLong_FromLong(p->ind2);
#else
      tmp1 = PyInt_FromLong(p->ind1);
      tmp2 = PyInt_FromLong(p->ind2);
#endif
      PyList_Append(list1, tmp1);
      PyList_Append(list2, tmp2);
      Py_DECREF(tmp1);
      Py_DECREF(tmp2);
    }
    tmp = p->next;
    free(p);
  }
  PyObject *result = PyList_New(2);
  PyList_SET_ITEM(result, 0, list1);
  PyList_SET_ITEM(result, 1, list2);
  return result;
}

static char *sasa_doc = (char *)
    "sasa(srad, sel, ... ) -> solvent accessible surface area.\n"
    "srad gives solvent radius.\n"
    "Optional keyword arguments:\n"
    "  samples=500 -- specifies number of sample points used per atom.\n"
    "  points=None -- If points is a list, coordinates of surface points\n"
    "    will be appended to the list.\n"
    "  restrict=None -- Pass an atom selection as argument to restrict\n"
    "    to find contributions coming from just atoms in restrict.\n";

static PyObject *sasa(PyAtomSelObject *a, PyObject *args, PyObject *keywds) {
  float srad = 0;
  int samples = -1;
  const int *sampleptr = NULL;
  PyObject *pointsobj = NULL;
  PyObject *restrictobj = NULL;

  AtomSel *sel = a->atomSel;
  DrawMolecule *mol;
  if (!(mol = get_molecule(a))) return NULL;

  static char *kwlist[] = {
    "srad", "samples", "points", "restrict", NULL
  };
  if (!PyArg_ParseTupleAndKeywords(args, keywds,
        "f|iOO:atomsel.sasa", kwlist,
        &srad, &samples, &pointsobj, &restrictobj))
    return NULL;

  // validate srad
  if (srad < 0) {
    PyErr_SetString(PyExc_ValueError, "atomselect.sasa: srad must be non-negative.");
    return NULL;
  }

  // fetch the radii and coordinates
  const float *radii = mol->extraflt.data("radius");
  const float *coords = sel->coordinates(a->app->moleculeList);

  // if samples was given and is valid, use it
  if (samples > 1) sampleptr = &samples;

  // if restrict is given, validate it
  AtomSel *restrictsel = NULL;
  if (restrictobj) {
      if (!atomsel_Check(restrictobj)) return NULL;
      if (!get_molecule((PyAtomSelObject *)restrictobj)) return NULL;
      restrictsel = ((PyAtomSelObject *)restrictobj)->atomSel;
  }

  // if points are requested, fetch them
  ResizeArray<float> sasapts;
  ResizeArray<float> *sasaptsptr = pointsobj ? &sasapts : NULL;

  // go!
  float sasa = 0;
  int rc = measure_sasa(sel, coords, radii, srad, &sasa,
        sasaptsptr, restrictsel, sampleptr);
  if (rc) {
    PyErr_SetString(PyExc_ValueError, measure_error(rc));
    return NULL;
  }

  // append surface points to the provided list object.
  if (pointsobj) {
    for (int i=0; i<sasapts.num(); i++) {
      PyList_Append(pointsobj, PyFloat_FromDouble(sasapts[i]));
    }
  }

  // return the total SASA.
  return PyFloat_FromDouble(sasa);
}

/*
 * Support for mapping protocol
 */

static Py_ssize_t
atomselection_length( PyObject *a ) {
  return ((PyAtomSelObject *)a)->atomSel->selected;
}

// for integer argument, return True or False if index in in selection
static PyObject *
atomselection_subscript( PyAtomSelObject * a, PyObject * keyobj ) {
#if PY_MAJOR_VERSION >= 3
  long ind = PyLong_AsLong(keyobj);
#else
  long ind = PyInt_AsLong(keyobj);
#endif
  if (ind < 0 && PyErr_Occurred()) return NULL;
  PyObject *result = Py_False;
  if (ind >= 0 && ind < a->atomSel->num_atoms &&
      a->atomSel->on[ind]) {
    result = Py_True;
  }
  Py_INCREF(result);
  return result;
}

static PyMappingMethods atomsel_mapping = {
  atomselection_length,
  (binaryfunc)atomselection_subscript,
  0
};

/* Methods on selection instances */
static PyMethodDef atomselection_methods[] = {
  { "get", (PyCFunction)atomsel_get, METH_O, get_doc  },
  { "set", (PyCFunction)atomsel_set, METH_VARARGS, set_doc },
  { "update", (PyCFunction)py_update, METH_NOARGS, update_doc },
  { "write", (PyCFunction)py_write, METH_VARARGS|METH_KEYWORDS, write_doc },
  { "minmax", (PyCFunction)minmax, METH_VARARGS, minmax_doc },
  { "center", (PyCFunction)center, METH_VARARGS|METH_KEYWORDS, center_doc },
  { "rmsd", (PyCFunction)py_rmsd, METH_VARARGS, rmsd_doc },
  { "rmsf", (PyCFunction)py_rmsf, METH_VARARGS|METH_KEYWORDS, rmsf_doc },
  { "rgyr", (PyCFunction)py_rgyr, METH_VARARGS|METH_KEYWORDS, rgyr_doc },
  { "fit", (PyCFunction)py_fit, METH_VARARGS, fit_doc },
  { "move", (PyCFunction)py_move, METH_O, move_doc },
  { "moveby", (PyCFunction)py_moveby, METH_O, moveby_doc },
  { "contacts", (PyCFunction)contacts, METH_VARARGS, contacts_doc },
  { "sasa", (PyCFunction)sasa, METH_VARARGS | METH_KEYWORDS, sasa_doc },
  {NULL, NULL, 0, NULL}
};

// Atom selection iterator

namespace {
  typedef struct {
    PyObject_HEAD
    int index;
    PyAtomSelObject * a;
  } iterobject;

  PyObject *atomsel_iter(PyObject *);

  PyObject *iter_next(iterobject *it) {
    for ( ; it->index < it->a->atomSel->num_atoms; ++it->index) {
      if (it->a->atomSel->on[it->index])
#if PY_MAJOR_VERSION >= 3
        return PyLong_FromLong(it->index++);
#else
        return PyInt_FromLong(it->index++);
#endif
    }
    return NULL;
  }

  void iter_dealloc(iterobject *it) {
    Py_XDECREF(it->a);
  }
  PyObject *iter_len(iterobject *it) {
#if PY_MAJOR_VERSION >= 3
    return PyLong_FromLong(it->a->atomSel->selected);
#else
    return PyInt_FromLong(it->a->atomSel->selected);
#endif
  }

  PyMethodDef iter_methods[] = {
    {"__length_hint__", (PyCFunction)iter_len, METH_NOARGS },
    {NULL, NULL}
  };

#if PY_MAJOR_VERSION >= 3
  PyTypeObject itertype = {
    PyObject_HEAD_INIT(&PyType_Type)
    "atomsel.iterator",
    sizeof(iterobject), 0, // basic, item size
    (destructor)iter_dealloc, // dealloc
    0, //tp_print
    0, 0, // tp get and setattr
    0, // tp_as_async
    0, // tp_repr
    0, 0, 0, // as number, sequence, mapping
    0, 0, 0, // hash, call, str
    PyObject_GenericGetAttr, 0, // getattro, setattro
    0, // tp_as_buffer
    Py_TPFLAGS_DEFAULT, // flags
    0, // docstring
    0, 0, 0, // traverse, clear, richcompare
    0, // tp_weaklistoffset
    PyObject_SelfIter, // tp_iter
    (iternextfunc)iter_next, // tp_iternext
    iter_methods, // tp_methods
    0, 0, 0, // members, getset, base
};

PyTypeObject atomsel_type = {
    PyObject_HEAD_INIT(0)
    "atomsel.atomsel",
    sizeof(PyAtomSelObject), 0, // basic, item size
    (destructor)atomsel_dealloc, //dealloc
    0, // tp_print
    0, 0, // tp get and set attr
    0, // tp_as_async
    (reprfunc)atomsel_repr, // tp_repr
    0, 0, &atomsel_mapping, // as number, sequence, mapping
    0, 0, (reprfunc)atomsel_str, // hash, call, str
    PyObject_GenericGetAttr, 0, // getattro, setattro
    0, // tp_as_buffer
    Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, // flags
    atomsel_doc, // docstring
    0, 0, 0, // traverse, clear, richcompare
    0, // tp_weaklistoffset
    atomsel_iter, // tp_iter
    0, // tp_iternext
    atomselection_methods, // tp_methods,
    0, atomsel_getset, 0, // members, getset, base
    0, 0, 0, // tp_dict, descr_get, descr_set
    0, 0, // dictoffset, init
    PyType_GenericAlloc, // tp_alloc
    atomsel_new, // tp_new
    PyObject_Del, // tp_free
};

#else
  PyTypeObject itertype = {
    PyObject_HEAD_INIT(&PyType_Type)
    0,
    "atomsel.iterator",
    sizeof(iterobject),
    0, // itemsize
    /* methods */
    (destructor)iter_dealloc,
	0,					/* tp_print */
	0,					/* tp_getattr */
	0,					/* tp_setattr */
	0,					/* tp_compare */
	0,					/* tp_repr */
	0,					/* tp_as_number */
	0,					/* tp_as_sequence */
	0,					/* tp_as_mapping */
	0,					/* tp_hash */
	0,					/* tp_call */
	0,					/* tp_str */
	PyObject_GenericGetAttr,		/* tp_getattro */
	0,					/* tp_setattro */
	0,					/* tp_as_buffer */
	Py_TPFLAGS_DEFAULT,                     /* tp_flags */
	0,					/* tp_doc */
	0,	                                /* tp_traverse */
	0,					/* tp_clear */
	0,					/* tp_richcompare */
	0,					/* tp_weaklistoffset */
	PyObject_SelfIter,			/* tp_iter */
	(iternextfunc)iter_next,		/* tp_iternext */
	iter_methods,			        /* tp_methods */
	0,					/* tp_members */
  };

PyTypeObject atomsel_type = {
  PyObject_HEAD_INIT(0) /* Must fill in type value later */
  0,          /* ob_size */
  "atomsel.atomsel",     /* tp_name */
  sizeof(PyAtomSelObject),   /* tp_basicsize */
  0,          /* tp_itemsize */
  (destructor)atomsel_dealloc,   /* tp_dealloc */
  0,          /* tp_print */
  0,          /* tp_getattr FIXME: can I override this? */
  0,          /* tp_setattr */
  0,          /* tp_compare */
  (reprfunc)atomsel_repr,      /* tp_repr */
  0,          /* tp_as_number */
  0,          /* tp_as_sequence */
  &atomsel_mapping,          /* tp_as_mapping */
  0,          /* tp_hash */
  0,          /* tp_call */
  (reprfunc)atomsel_str,          /* tp_str */
  PyObject_GenericGetAttr,    /* tp_getattro */
  0,          /* tp_setattro */
  0,          /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE|Py_TPFLAGS_HAVE_CLASS, /* tp_flags */
  atomsel_doc,       /* tp_doc */
  0,          /* tp_traverse */
  0,          /* tp_clear */
  0,          /* tp_richcompare */
  0,          /* tp_weaklistoffset */
  atomsel_iter,      /* tp_iter */
  0,          /* tp_iternext */
  atomselection_methods,       /* tp_methods */
  0,          /* tp_members */
  atomsel_getset,          /* tp_getset */
  0,          /* tp_base */
  0,          /* tp_dict */
  0,          /* tp_descr_get */
  0,          /* tp_descr_set */
  0,          /* tp_dictoffset */
  0,          /* FIXME: needed? tp_init */
  PyType_GenericAlloc,      /* tp_alloc */
  atomsel_new,       /* tp_new */
  _PyObject_Del,       /* tp_free */
};
#endif

  PyObject *atomsel_iter(PyObject *self) {
    iterobject * iter = PyObject_New(iterobject, &itertype);
    if (!iter) return NULL;
    Py_INCREF( iter->a = (PyAtomSelObject *)self );
    iter->index = 0;
    return (PyObject *)iter;
  }
}


static int atomsel_Check(PyObject *obj) {
  if (PyObject_TypeCheck(obj, &atomsel_type)) return 1;
  PyErr_SetString(PyExc_TypeError, "expected atomsel");
  return 0;
}

AtomSel * atomsel_AsAtomSel( PyObject *obj) {
  if (!atomsel_Check(obj)) return NULL;
  return ((PyAtomSelObject *)obj)->atomSel;
}

/* List of functions exported by this module */
static PyMethodDef atomsel_methods[] = {
  {"macro", (PyCFunction)macro, METH_VARARGS | METH_KEYWORDS,
    "macro(name=None, selection=None) -- create and query selection macros.\n"
    "If both name and selection are None, return list of macro names.\n"
    "If selection is None, return definition for name.\n"
    "If both name and selection are given, define new macro.\n" },
  {"delmacro", (vmdPyMethod)delmacro, METH_VARARGS,
    "delmacro(name) -> Delete atom selection macro with given name." },
  {"inverse", (PyCFunction)inverse, METH_O,
    "inverse(matrix) -> Inverse of matrix returned by atomsel.fit(...)"},
  {"keywords", (PyCFunction)keywords, METH_NOARGS,
    "keywords() -> List of available atom selection keywords."},
  {"booleans", (PyCFunction)booleans, METH_NOARGS,
    "booleans() -> List of available atom selection boolean tokens."},
  {"functions", (PyCFunction)functions, METH_NOARGS,
    "functions() -> List of available atom selection functions."},
  {"stringfunctions", (PyCFunction)stringfunctions, METH_NOARGS,
    "stringfunctions() -> List of available atom selection string functions."},
  { NULL, NULL }
};

static char *module_doc = (char *)
    "Methods for creating, updating, querying, and modifying\n"
    "selections of atoms.\n"
    "\n"
    "Example of usage:\n"
    ">>> from atomsel import *\n"
    ">>> s1 = atomsel('residue 1 to 10 and backbone')\n"
    ">>> s1.get('resid')\n"
    " <snip> \n"
    ">>> s1.set('beta', 5') # set B value to 5 for atoms in s1\n"
    ">>> # Mass-weighted RMS alignment:\n"
    ">>> mass = s1.get('mass')\n"
    ">>> s2 = atomsel('residue 21 to 30 and backbone')\n"
    ">>> mat = s1.fit(s2, mass)\n"
    ">>> s1.move(mat)\n"
    ">>> print s1.rmsd(s2)\n" ;

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef atomseldef = {
    PyModuleDef_HEAD_INIT,
    "atomsel",
    module_doc,
    -1, // global state, no sub-interpreters
    atomsel_methods,
};
#endif


PyObject* initatomsel(void) {
#if PY_MAJOR_VERSION >= 3
  PyObject *m = PyModule_Create(&atomseldef);
  ((PyObject*)(&atomsel_type))->ob_type = &PyType_Type;
#else
  PyObject *m = Py_InitModule3( "atomsel", atomsel_methods, module_doc );
  atomsel_type.ob_type = &PyType_Type;
#endif

  Py_INCREF((PyObject *)&atomsel_type);
  if (PyModule_AddObject(m, "atomsel",
      (PyObject *)&atomsel_type) !=0)
      return NULL;
  if (PyType_Ready(&atomsel_type) < 0)
      return NULL;

  return m;
}

