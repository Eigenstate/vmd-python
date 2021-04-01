/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2019 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: py_atomsel.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.29 $      $Date: 2019/01/23 22:59:15 $
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

// Helper function to check if something is an atomsel type
static int atomsel_Check(PyObject *obj) {
  if (PyObject_TypeCheck(obj, &Atomsel_Type))
    return 1;
  PyErr_SetString(PyExc_TypeError, "expected atomsel");
  return 0;
}

AtomSel *atomsel_AsAtomSel( PyObject *obj) {
  if (!atomsel_Check(obj))
    return NULL;
  return ((PyAtomSelObject *)obj)->atomSel;
}


// return molecule for atomsel, or NULL and set exception
DrawMolecule *get_molecule(PyAtomSelObject *a) {
  int molid = a->atomSel->molid();
  DrawMolecule *mol = a->app->moleculeList->mol_from_id(molid);
  if (!mol) {
    PyErr_Format(PyExc_ValueError, "invalid molid '%d'", molid);
    return NULL;
  }
  return mol;
}

static const char atomsel_doc[] =
"Create a new atom selection object\n\n"
"Args:\n"
"    selection (str): Atom selection string. Defaults to 'all'\n"
"    molid (int): Molecule ID to select from. Defaults to -1 (top molecule)\n"
"    frame (int): Frame to select on. Defaults to -1 (current)\n\n"
"Example of usage:\n"
"    >>> from vmd import atomsel\n"
"    >>> s1 = atomsel('residue 1 to 10 and backbone')\n"
"    >>> s1.resid\n"
"     <snip> \n"
"    >>> s1.beta = 5 # Set beta to 5 for all atoms in s1\n"
"    >>> # Mass-weighted RMS alignment:\n"
"    >>> mass = s1.get('mass')\n"
"    >>> s2 = atomsel('residue 21 to 30 and backbone')\n"
"    >>> mat = s1.fit(s2, mass)\n"
"    >>> s1.move(mat)\n"
"    >>> print s1.rmsd(s2)\n";

// __del__(self)
static void atomsel_dealloc( PyAtomSelObject *obj ) {
  delete obj->atomSel;
  ((PyObject *)(obj))->ob_type->tp_free((PyObject *)obj);
}

// __repr__(self)
static PyObject *atomsel_repr(PyAtomSelObject *obj) {
  PyObject *result;
  AtomSel *sel;
  char *s;

  sel = obj->atomSel;
  s = new char[strlen(sel->cmdStr) + 100];

  sprintf(s, "atomsel('%s', molid=%d, frame=%d)", sel->cmdStr, sel->molid(),
          sel->which_frame);
  result = as_pystring(s);

  delete [] s;
  return result;
}

// __str__(self)
static PyObject* atomsel_str(PyAtomSelObject *obj)
{
  return as_pystring(obj->atomSel->cmdStr);
}

// __init__(self, selection, molid, frame)
static PyObject *atomsel_new(PyTypeObject *type, PyObject *args,
                             PyObject *kwargs)
{
  const char *kwlist[] = {"selection", "molid", "frame", NULL};
  int molid = -1, frame = -1;
  const char *sel = "all";
  DrawMolecule *mol;
  int parse_result;
  AtomSel *atomSel;
  PyObject *obj;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|sii:atomsel",
                                   (char**) kwlist, &sel, &molid, &frame))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (molid < 0)
    molid = app->molecule_top();

  if (!valid_molid(molid, app))
    return NULL;

  mol = app->moleculeList->mol_from_id(molid);

  atomSel = new AtomSel(app->atomSelParser, mol->id());
  atomSel->which_frame = frame;

  Py_BEGIN_ALLOW_THREADS
    parse_result = atomSel->change(sel, mol);
  Py_END_ALLOW_THREADS

  if (parse_result == AtomSel::NO_PARSE) {
    PyErr_Format(PyExc_ValueError, "cannot parse atom selection text '%s'", sel);
    goto failure;
  }

  obj = type->tp_alloc(type, 0);
  if (!obj) {
    PyErr_Format(PyExc_MemoryError, "cannot allocate atomsel type");
    goto failure;
  }

  ((PyAtomSelObject *)obj)->app = app;
  ((PyAtomSelObject *)obj)->atomSel = atomSel;

  return obj;

failure:
  delete atomSel;
  return NULL;
}

static const char get_doc[] =
"Get attribute values for selected atoms\n\n"
"Args:\n"
"    attribute (str): Attribute to query\n"
"Returns:\n"
"    (list): Attribute value for each atom in selection";
static PyObject *atomsel_get(PyAtomSelObject *a, PyObject *keyobj) {

  // FIXME: make everything in this pipeline const so that it's
  // thread-safe.
  const VMDApp *app = a->app;
  const AtomSel *atomSel = a->atomSel;
  const int num_atoms = a->atomSel->num_atoms;

  PyObject *newlist = NULL;
  SymbolTableElement *elem;
  SymbolTable *table;
  DrawMolecule *mol;
  int attrib_index, i;
  int *flgs;
  int j = 0;

  if (!(mol = get_molecule(a)))
    return NULL;

  const char *attr = as_charptr(keyobj);
  if (!attr) return NULL;

  //
  // Check for a valid attribute
  //
  table = app->atomSelParser;
  attrib_index = table->find_attribute(attr);
  if (attrib_index == -1) {
    PyErr_Format(PyExc_AttributeError, "unknown atom attribute '%s'", attr);
    return NULL;
  }

  elem = table->fctns.data(attrib_index);
  if (elem->is_a != SymbolTableElement::KEYWORD &&
      elem->is_a != SymbolTableElement::SINGLEWORD) {
    PyErr_Format(PyExc_AttributeError, "attribute '%s' is not a keyword or "
                 "singleword", attr);
    return NULL;
  }

  //
  // fetch the data
  //
  flgs = atomSel->on;
  atomsel_ctxt context(table, mol, atomSel->which_frame, attr);

  if (!(newlist = PyList_New(atomSel->selected)))
    return NULL;

  // Singleword attributes (backbone, etc) return array of bools
  if (elem->is_a == SymbolTableElement::SINGLEWORD) {

    int *tmp = new int[num_atoms];
    memcpy(tmp, atomSel->on, num_atoms*sizeof(int));
    elem->keyword_single(&context, num_atoms, tmp);

    for (i = 0; i < num_atoms; i++) {
      if (flgs[i]) {
        PyObject *val = tmp[i] ? Py_True : Py_False;
        Py_INCREF(val); // SET_ITEM steals a reference so we make one to steal
        PyList_SET_ITEM(newlist, j++, val);

        if (PyErr_Occurred())
          goto failure;
      }
    }
    delete [] tmp;

  // String attributes
  } else if (table->fctns.data(attrib_index)->returns_a \
          == SymbolTableElement::IS_STRING) {

    const char **tmp= new const char *[num_atoms];
    elem->keyword_string(&context, num_atoms, tmp, flgs);

    for (i = 0; i < num_atoms; i++) {
      if (flgs[i]) {
        PyList_SET_ITEM(newlist, j++, as_pystring(tmp[i]));

        if (PyErr_Occurred()) {
          delete [] tmp;
          goto failure;
        }
      }
    }
    delete [] tmp;

  // Integer attributes
  } else if (table->fctns.data(attrib_index)->returns_a \
          == SymbolTableElement::IS_INT) {

    int *tmp = new int[num_atoms];
    elem->keyword_int(&context, num_atoms, tmp, flgs);

    for (i = 0; i < num_atoms; i++) {
      if (flgs[i]) {
        PyList_SET_ITEM(newlist, j++, as_pyint(tmp[i]));

        if (PyErr_Occurred()) {
          delete [] tmp;
          goto failure;
        }
      }
    }
    delete [] tmp;

  // Floating point attributes
  } else if (table->fctns.data(attrib_index)->returns_a \
          == SymbolTableElement::IS_FLOAT) {

    double *tmp = new double[num_atoms];
    elem->keyword_double(&context, num_atoms, tmp, flgs);

    for (i = 0; i < num_atoms; i++) {
      if (flgs[i]) {
        PyList_SET_ITEM(newlist, j++, PyFloat_FromDouble(tmp[i]));

        if (PyErr_Occurred()) {
          delete [] tmp;
          goto failure;
        }

      }
    }
    delete [] tmp;
  }

  return newlist;

failure:
  return NULL;
}

static const char listattrs_doc[] =
"List available atom attributes\n\n"
"Args:\n"
"    changeable (bool): If only user-changeable attributes should be listed\n"
"        Defaults to False\n"
"Returns:\n"
"    (list of str): Atom attributes. These attributes may be accessed or\n"
"        set with a . after the class name.\n"
"Example to list available attributes and get the x coordinate attribute:\n"
"    >>> sel = atomsel('protein')\n"
"    >>> sel.list_attributes()\n"
"    >>> sel.x";
static PyObject *py_list_attrs(PyAtomSelObject *obj, PyObject *args,
                               PyObject *kwargs)
{
  const char *kwlist[] = {"changeable", NULL};
  SymbolTableElement *elem;
  PyObject *result = NULL;
  int changeable = 0;
  SymbolTable *table;
  int i;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O&:atomsel.list_attributes",
                                   (char**) kwlist, convert_bool, &changeable))
    return NULL;

  table = obj->app->atomSelParser;
  result = PyList_New(0);

  if (!result)
    goto failure;

  for (i = 0; i< table->fctns.num(); i++) {
    // Only list singleword or keyword attributes that are gettable
    elem = table->fctns.data(i);
    if (elem->is_a != SymbolTableElement::KEYWORD
     && elem->is_a != SymbolTableElement::SINGLEWORD)
      continue;

    // Only list changeable attributes if requested
    if (changeable && !table->is_changeable(i))
      continue;

    // Don't list unnamed attributes
    if (!table->fctns.name(i) || !strlen(table->fctns.name(i)))
      continue;

    // Don't leak references with PyList_Append
    PyObject *attr = as_pystring(table->fctns.name(i));
    PyList_Append(result, attr);
    Py_XDECREF(attr);

    if (PyErr_Occurred())
      goto failure;
  }

  return result;

failure:
  PyErr_SetString(PyExc_RuntimeError, "Problem listing attributes");
  Py_XDECREF(result);
  return NULL;
}

static PyObject *legacy_atomsel_get(PyObject *o, PyObject *args,
                                    PyObject *kwargs)
{
  const char *kwlist[] = {"attribute", NULL};
  PyObject *attr, *result;

  PyErr_Warn(PyExc_DeprecationWarning, "atomsel.get is deprecated. You can\n"
             "directly query the attributes of this object instead");

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O:atomsel.get",
                                   (char**) kwlist, &attr))
    return NULL;

  // It sets an exception on error
  if (!(result = atomsel_get((PyAtomSelObject*) o, attr)))
    return NULL;

  return result;
}

// __getattr__, with support for defined methods too
static PyObject *atomsel_getattro(PyObject *o, PyObject *attr_name)
{
  // Check that there isn't a class method by this name first
  PyObject *tmp;
  if (!(tmp = PyObject_GenericGetAttr(o, attr_name))) {
    // If no method found, throws AttributeError that we clear.
    // If it threw a different exception though, we do fail.
    if (!PyErr_ExceptionMatches(PyExc_AttributeError))
      return NULL;
    PyErr_Clear();
  } else {
    return tmp;
  }

  return atomsel_get((PyAtomSelObject*) o, attr_name);
}

static void help_int(void* p, PyObject *obj)
{
  *((int*)p) = as_int(obj);
}

static void help_double(void* p, PyObject *obj)
{
  *((double*)p) = PyFloat_AsDouble(obj);
}

static void help_charptr(void* p, PyObject *obj)
{
  char *str = as_charptr(obj);
  *((char**)p) = str;
}

// Helper functions for set
static int build_set_values(const void* list, int num_atoms, PyObject* val,
                            int *flgs, int dtype_size,
                            void (*converter)(void*, PyObject*))
{
  int i, is_array;
  int j = 0;
  void *p;

  // Determine if passed PyObject is an array
  is_array = (PySequence_Check(val) && !is_pystring(val)) ? 1 : 0;

  //If it is an array, check to make sure it matches the length of the selection.
  if (is_array && PySequence_Length(val) > 1 && PySequence_Length(val) != atomSel->selected ) {
    PyErr_SetString(PyExc_ValueError, "sequence length does not match the number of selected atoms");
    return 1;
  }
  
  for (i = 0; i < num_atoms; i++) {
    // Continue if atom is not part of selection
    if (!flgs[i])
      continue;

    // Get the correct atom
    p = (char*) list + i*dtype_size;

    // Set ival to each value in array if we have an array
    // If array has length 1, repeat first array element*
    // *I think this shouldn't be supported, but keeping it for compatilibity
    if (is_array) {
      if (PySequence_Length(val) == 1)
        converter(p, PySequence_ITEM(val, 0));
      else
        converter(p, PySequence_ITEM(val, j++));

    // If not an array of values, unpack PyObject alone
    } else {
      converter(p, val);
    }
    if (PyErr_Occurred())
      goto failure;
  }

  return 0;

failure:
  PyErr_SetString(PyExc_ValueError, "sequence passed to set contains "
                  "the wrong type of data (integer, float, or str)");
  return 1;
}


static int atomsel_set(PyAtomSelObject *a, const char *name, PyObject *val)
{
  const VMDApp *app = a->app;
  const AtomSel *atomSel = a->atomSel;
  const int num_atoms = atomSel->num_atoms;
  SymbolTable *table = app->atomSelParser;
  int *flgs = atomSel->on;
  void* list = NULL;

  SymbolTableElement *elem;
  DrawMolecule *mol;
  int attrib_index;

  // Check for a valid molecule
  if (!(mol = get_molecule(a))) {
    PyErr_SetString(PyExc_ValueError, "selection is on a deleted molecule");
    return -1;
  }

  // Check for a valid attribute
  if ((attrib_index = table->find_attribute(name)) == -1) {
    PyErr_Format(PyExc_AttributeError, "unknown atom attribute '%s'", name);
    return -1;
  }

  elem = table->fctns.data(attrib_index);
  if (elem->is_a != SymbolTableElement::KEYWORD &&
      elem->is_a != SymbolTableElement::SINGLEWORD) {
    PyErr_Format(PyExc_AttributeError, "attribute '%s' is not a keyword or "
                 "singleword", name);
    return -1;
  }

  if (!table->is_changeable(attrib_index)) {
    PyErr_Format(PyExc_AttributeError, "attribute '%s' is not modifiable",
                 name);
    return -1;
  }

  // singlewords can never be set, so macro is NULL.
  atomsel_ctxt context(table, mol, atomSel->which_frame, NULL);

  // Integer type
  if (elem->returns_a == SymbolTableElement::IS_INT) {
    list = malloc(num_atoms * sizeof(int));
    if (build_set_values(list, num_atoms, val, flgs, sizeof(int), help_int))
      goto failure;

    elem->set_keyword_int(&context, num_atoms, (int*) list, flgs);

  // Float type
  } else if (elem->returns_a == SymbolTableElement::IS_FLOAT) {
    list = malloc(num_atoms * sizeof(double));
    if (build_set_values(list, num_atoms, val, flgs,
                         sizeof(double), help_double))
      goto failure;

    elem->set_keyword_double(&context, num_atoms, (double*) list, flgs);

  } else if (elem->returns_a == SymbolTableElement::IS_STRING) {
    list = malloc(num_atoms * sizeof(char*));
    if (build_set_values(list, num_atoms, val, flgs,
                         sizeof(char*), help_charptr))
        goto failure;

    elem->set_keyword_string(&context, num_atoms, (const char**) list, flgs);
    // No special memory management needed for strings as python API
    // as_charptr gives a pointer into PyObject* that Python is managing
  }

  free(list);
  list = NULL;

  // Recompute the color assignments if certain atom attributes are changed.
  if (!strcmp(name, "name") ||
      !strcmp(name, "type") ||
      !strcmp(name, "resname") ||
      !strcmp(name, "chain") ||
      !strcmp(name, "segid") ||
      !strcmp(name, "segname"))
    app->moleculeList->add_color_names(mol->id());

  mol->force_recalc(DrawMolItem::SEL_REGEN | DrawMolItem::COL_REGEN);

  return 0;

failure:
  // Exception already set by build_set_values or similar
  free(list);
  return 1;
}

// Legacy atomsel.set method
static const char set_doc[] =
"Set attributes for selected atoms\n\n"
"Args:\n"
"    attribute (str): Attribute to set\n"
"    value: Value for attribute. If single value, all atoms will be set to\n"
"        have the same value. Otherwise, pass a sequence (list or tuple) of\n"
"        values, one per atom in selection";
static PyObject* legacy_atomsel_set(PyObject *o, PyObject *args,
                                    PyObject *kwargs)
{
  const char *kwlist[] = {"attribute", "value", NULL};
  const char *attr;
  PyObject *val;

  PyErr_Warn(PyExc_DeprecationWarning, "atomsel.set is deprecated. You can\n"
             "directly assign to the attributes instead.");

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sO:atomsel.set",
                                  (char**) kwlist,  &attr, &val))
    return NULL;

  // It sets an exception on error
  if (atomsel_set((PyAtomSelObject*) o, attr, val))
    return NULL;

  Py_INCREF(Py_None);
  return Py_None;
}

// __setattr__, with support for defined methods too
static int atomsel_setattro(PyObject *o, PyObject *name, PyObject *value)
{
  // Check that there isn't a class method for this set first
  // GenericSetAttr returns 0 on success
  // If no setter method found, throws AttributeError that we clear
  // We do fail if a different exception is thrown
  if (PyObject_GenericSetAttr(o, name, value)) {
    if (!PyErr_ExceptionMatches(PyExc_AttributeError))
      return -1;
    PyErr_Clear();

  } else {
    return 0;
  }

  return atomsel_set((PyAtomSelObject*) o, as_charptr(name), value);
}

static const char frame_doc[] =
"Get the frame an atomsel object references. Changing the frame does not\n"
"immediately update the selection. Use `atomsel.update()` to do that.\n"
"Special frame values are -1 for the current frame and -2 for the last frame";
static PyObject *getframe(PyAtomSelObject *a, void *) {

  AtomSel *atomSel = a->atomSel;
  DrawMolecule *mol;

  if (!(mol = get_molecule(a)))
    return NULL;

  return as_pyint(atomSel->which_frame);
}

static int setframe(PyAtomSelObject *a, PyObject *frameobj, void *) {

  AtomSel *atomSel = a->atomSel;
  DrawMolecule *mol;
  int frame;
  if (!(mol = get_molecule(a)))
    return -1;

  frame = as_int(frameobj);
  if (PyErr_Occurred())
    return -1;

  if (frame != AtomSel::TS_LAST && frame != AtomSel::TS_NOW &&
      (frame < 0 || frame >= mol->numframes())) {
    PyErr_Format(PyExc_ValueError, "Frame '%d' invalid for this molecule",
                 frame);
    return -1;
  }

  atomSel->which_frame = frame;
  return 0;
}

static const char update_doc[] =
"Recompute which atoms in the molecule belong to this selection. For example\n"
"when the selection is distance base (e.g. it uses 'within'), changes to the\n"
"frame of this atom selection will not be reflected in the selected atoms\n"
"until this method is called";
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

static const char write_doc[] =
"Write atoms in this selection to a file of the given type\n\n"
"Args:\n"
"    filetype (str): File type to write, as defined by molfileplugin\n"
"    filename (str): Filename to write to";
static PyObject *py_write(PyAtomSelObject *a, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"filetype", "filename", NULL};
  const char *filetype, *filename;
  AtomSel *atomSel = a->atomSel;
  DrawMolecule *mol;
  int frame = -1;
  FileSpec spec;
  VMDApp *app;
  int rc;

  if (!(mol = get_molecule(a))) {
    PyErr_SetString(PyExc_ValueError, "molecule for this selection is deleted");
    return NULL;
  }

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ss:atomsel.write",
                                   (char**) kwlist, &filetype, &filename))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  switch (atomSel->which_frame) {
    case AtomSel::TS_NOW:  frame = mol->frame(); break;
    case AtomSel::TS_LAST: frame = mol->numframes()-1; break;
    default:               frame = atomSel->which_frame; break;
  }

  // If frame out of bounds, return error
  // (formerly this just truncated frame to being in the valid range)
  if (frame < 0 || frame >= mol->numframes()) {
    PyErr_Format(PyExc_ValueError, "frame '%d' out of bounds for this molecule",
                 frame);
    return NULL;
  }

  // Write the requested atoms to the file
  spec.first = frame;
  spec.last = frame;
  spec.stride = 1;
  spec.waitfor = FileSpec::WAIT_ALL;
  spec.selection = atomSel->on;

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

static const char bonds_doc[] =
"For each atom in selection, a list of the indices of atoms to which it is\n"
"bonded.\nTo set bonds, pass a sequence with length of the number of atoms in\n"
"the selection, with each entry in the sequence being a list or tuple\n"
"containing the atom indices to which that atom in the selection is bound\n\n"
"For example, for a water molecule with atoms H-O-H:\n"
">>> sel = atomsel('water and residue 0')\n"
">>> sel.bonds = [(1), (0,2), (1)]";
static PyObject *getbonds(PyAtomSelObject *a, void *)
{
  PyObject *bondlist = NULL, *newlist = NULL;
  AtomSel *atomSel = a->atomSel;
  DrawMolecule *mol;
  int i, j, k = 0;

  if (!(mol = get_molecule(a))) {
    PyErr_SetString(PyExc_ValueError, "selection is on deleted molecule");
    return NULL;
  }

  newlist = PyList_New(atomSel->selected);
  if (!newlist)
    goto failure;

  for (i = atomSel->firstsel; i <= atomSel->lastsel; i++) {

    if (!atomSel->on[i])
      continue;

    const MolAtom *atom = mol->atom(i);
    bondlist = PyList_New(atom->bonds);
    if (!bondlist)
      goto failure;

    for (j=0; j<atom->bonds; j++) {
      PyList_SET_ITEM(bondlist, j, as_pyint(atom->bondTo[j]));
      if (PyErr_Occurred())
        goto failure;
    }

    PyList_SET_ITEM(newlist, k++, bondlist);
    if (PyErr_Occurred())
      goto failure;
  }
  return newlist;

failure:
  Py_XDECREF(newlist);
  Py_XDECREF(bondlist);
  PyErr_SetString(PyExc_RuntimeError, "Problem getting bonds");
  return NULL;
}

static int setbonds(PyAtomSelObject *a, PyObject *obj, void *)
{
  AtomSel *atomSel = a->atomSel;
  DrawMolecule *mol;
  PyObject *atomids;
  int ibond = 0;
  int i, j, k;

  if (!(mol = get_molecule(a))) {
    PyErr_SetString(PyExc_ValueError, "selection is on deleted molecule");
    return -1;
  }

  if (!PySequence_Check(obj)) {
    PyErr_SetString(PyExc_TypeError, "Argument to setbonds must be a sequence");
    return -1;
  }

  if (PySequence_Size(obj) != atomSel->selected) {
    PyErr_SetString(PyExc_ValueError, "atomlist and bondlist must be the same "
                    "size");
    return -1;
  }

  mol->force_recalc(DrawMolItem::MOL_REGEN); // many reps ignore bonds
  for (i = atomSel->firstsel; i <= atomSel->lastsel; i++) {
    if (!atomSel->on[i])
      continue;

    MolAtom *atom = mol->atom(i);
    if (!(atomids = PySequence_GetItem(obj, ibond++))) {
      PyErr_Format(PyExc_RuntimeError, "Could not get bonds for atom %d",
                   ibond - 1);
      goto failure;
    }

    if (!PySequence_Check(atomids)) {
      PyErr_SetString(PyExc_TypeError, "Bonded atoms must be a sequence");
      return -1;
    }

    k = 0;
    for (j = 0; j < PySequence_Size(atomids); j++) {
      int bond = as_int(PySequence_GetItem(atomids, j));

      if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "atom indices to set bonds must be "
                        "integers");
        goto failure;
      }

      if (bond >= 0 && bond < mol->nAtoms) {
        atom->bondTo[k++] = bond;
      }
    }
    atom->bonds = k;
  }

  return 0;

failure:
  return -1;
}

static const char molid_doc[] =
"The molecule ID the selection is associated with";
static PyObject *getmolid(PyAtomSelObject *a, void *) {
  if (!a->app->moleculeList->mol_from_id(a->atomSel->molid())) {
    PyErr_SetString(PyExc_ValueError, "selection is on deleted molecule");
    return NULL;
  }
  return as_pyint(a->atomSel->molid());
}

static PyGetSetDef atomsel_getset[] = {
  {(char*) "frame", (getter)getframe, (setter)setframe, (char*) frame_doc, NULL},
  {(char*) "bonds", (getter)getbonds, (setter)setbonds, (char*) bonds_doc, NULL},
  {(char*) "molid", (getter)getmolid, (setter)NULL, (char*) molid_doc, NULL},
  {NULL },
};

// utility routine for parsing weight values.  Uses the sequence protocol
// so that sequence-type structure (list, tuple) will be accepted.
static float *parse_weight(AtomSel *sel, PyObject *wtobj)
{
  float *weight = new float[sel->selected];
  PyObject *seq = NULL;
  int i;

  // If no weights passed, set them all to 1.0
  if (!wtobj || wtobj == Py_None) {
    for (int i=0; i<sel->selected; i++)
      weight[i] = 1.0f;
    return weight;
  }

  if (!(seq = PySequence_Fast(wtobj, "weight must be a sequence.")))
    goto failure;

  if (PySequence_Size(seq) != sel->selected) {
    PyErr_SetString(PyExc_ValueError, "weight must be same size as selection.");
    goto failure;
  }

  for (i = 0; i < sel->selected; i++) {
    PyObject *w = PySequence_Fast_GET_ITEM(seq, i);
    if (!PyFloat_Check(w)) {
      PyErr_SetString(PyExc_TypeError, "Weights must be floating point numbers");
      goto failure;
    }

    double tmp = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(seq, i));
    weight[i] = (float)tmp;
  }

  Py_XDECREF(seq);
  return weight;

failure:
  Py_XDECREF(seq);
  delete [] weight;
  return NULL;
}

static const char minmax_doc[] =
"Get minimum and maximum coordinates for selected atoms\n\n"
"Args:\n"
"    radii (bool): If atomic radii should be included in calculation\n"
"        Defaults to False.\n"
"Returns:\n"
"    (2-tuple of tuples): (x,y,z) coordinate of minimum, then maximum atom";
static PyObject *minmax(PyAtomSelObject *a, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"radii", NULL};
  AtomSel *atomSel = a->atomSel;
  const float *radii;
  float min[3], max[3];
  DrawMolecule *mol;
  int withradii = 0;
  const float *pos;
  int rc;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O&:atomsel.minmax",
                                   (char**) kwlist, convert_bool, &withradii))
    return NULL;

  if (!(mol = get_molecule(a))) {
    PyErr_SetString(PyExc_ValueError, "selection is on a deleted molecule");
    return NULL;
  }

  radii = withradii ? mol->extraflt.data("radius") : NULL;

  pos = atomSel->coordinates(a->app->moleculeList);
  rc = measure_minmax(atomSel->num_atoms, atomSel->on, pos, radii, min, max);
  if (rc < 0) {
    PyErr_SetString(PyExc_ValueError, measure_error(rc));
    return NULL;
  }

  return Py_BuildValue("(f,f,f),(f,f,f)",
      min[0], min[1], min[2], max[0], max[1], max[2]);
}

static const char centerperres_doc[] =
"Get the coordinates of the center of each residue in the selection,\n"
"optionally weighted by weight\n\n"
"Args:\n"
"    weight (list of float): Weights for each atom in selection. Optional.\n"
"        weights cannot be 0 otherwise NaN will be returned.\n"
"Returns:\n"
"    (list of tuple): (x,y,z) coordinates of center of each residue";
static PyObject *centerperresidue(PyAtomSelObject *a, PyObject *args,
                                  PyObject *kwargs)
{
  PyObject *weightobj = NULL, *returnlist = NULL, *element = NULL;
  const char *kwlist[] = {"weight", NULL};
  AtomSel *atomSel = a->atomSel;
  float *weight, *cen;
  DrawMolecule *mol;
  int i, j, ret_val;

  if (!(mol = get_molecule(a))) {
    PyErr_SetString(PyExc_ValueError, "selection is on a deleted molecule");
    return NULL;
  }

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O:atomsel.centerperresidue",
                                   (char**) kwlist, &weightobj))
    return NULL;

  weight = parse_weight(atomSel, weightobj);
  if (!weight)
    return NULL;

  // compute center
  cen = new float[3*atomSel->selected];
  ret_val = measure_center_perresidue(a->app->moleculeList, atomSel,
                                      atomSel->coordinates(a->app->moleculeList),
                                      weight, cen);
  delete [] weight;

  if (ret_val < 0) {
    PyErr_SetString(PyExc_ValueError, measure_error(ret_val));
    goto failure;
    return NULL;
  }

  //Build the python list.
  returnlist = PyList_New(ret_val);
  for (i = 0; i < ret_val; i++) {
    element = PyTuple_New(3);

    for (j = 0; j < 3; j++) {
      PyTuple_SET_ITEM(element, j, PyFloat_FromDouble((double) cen[3*i+j]));
      if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_ValueError, "Problem building center list");
        goto failure;
      }
    }

    PyList_SET_ITEM(returnlist, i, element);
    if (PyErr_Occurred()) {
      PyErr_SetString(PyExc_ValueError, "Problem building center list");
      goto failure;
    }
  }

  delete [] cen;
  return returnlist;

failure:
  delete [] cen;
  delete [] weight;
  Py_XDECREF(returnlist);
  Py_XDECREF(element);
  return NULL;
}


static const char rmsfperres_doc[] =
"Measures the root-mean-square fluctuation (RMSF) along a trajectory on a\n"
"per-residue basis. RMSF is the mean deviation from the average position\n\n"
"Args:\n"
"    first (int): First frame to include. Defaults to 0 (beginning).\n"
"    last (int): Last frame to include. Defaults to -1 (end).\n"
"    step (int): Use every step-th frame. Defaults to 1 (all frames)\n"
"Returns:\n"
"    (list of float): RMSF for each residue in selection";
static PyObject *py_rmsfperresidue(PyAtomSelObject *a, PyObject *args,
                                   PyObject *kwargs)
{
  const char *kwlist[] = {"first", "last", "step", NULL};
  int first=0, last=-1, step=1;
  PyObject *returnlist = NULL;
  int ret_val, i;
  float *rmsf;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|iii:atomsel.rmsfperresidue",
                                   (char**) kwlist, &first, &last, &step))
    return NULL;

  // Check molecule is valid
  if (!get_molecule(a)) {
    PyErr_SetString(PyExc_ValueError, "selection is on a deleted molecule");
    return NULL;
  }

  rmsf = new float[a->atomSel->selected];
  ret_val = measure_rmsf_perresidue(a->atomSel, a->app->moleculeList,
                                    first, last, step, rmsf);
  if (ret_val < 0) {
    PyErr_SetString(PyExc_RuntimeError, measure_error(ret_val));
    goto failure;
  }

  //Build the python list.
  returnlist = PyList_New(ret_val);
  for (i = 0; i < ret_val; i++) {
    PyList_SetItem(returnlist, i, PyFloat_FromDouble(rmsf[i]));
    if (PyErr_Occurred()) {
      PyErr_SetString(PyExc_RuntimeError, "Problem building rmsf list");
      goto failure;
    }
  }

  delete [] rmsf;
  return returnlist;

failure:
  Py_XDECREF(returnlist);
  delete [] rmsf;
  return NULL;
}

static const char center_doc[] =
"Get the coordinates of the center of the selection, optionally with weights\n"
"on the selection atoms\n\n"
"Args:\n"
"    weight (list of float): Weight on each atom. Optional\n"
"Returns:\n"
"    (3-tuple of float): (x,y,z) coordinates of center of the selection";
static PyObject *center(PyAtomSelObject *a, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"weight", NULL};
  AtomSel *atomSel = a->atomSel;
  PyObject *weightobj = NULL;
  DrawMolecule *mol;
  float *weight;
  float cen[3];
  int ret_val;

  if (!(mol = get_molecule(a))) {
    PyErr_SetString(PyExc_ValueError, "selection is on a deleted molecule");
    return NULL;
  }

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O:atomsel.center",
                                   (char**) kwlist, &weightobj))
    return NULL;

  weight = parse_weight(atomSel, weightobj);
  if (!weight)
    return NULL;

  // compute center
  ret_val = measure_center(atomSel, atomSel->coordinates(a->app->moleculeList),
                           weight, cen);
  delete [] weight;

  if (ret_val < 0) {
    PyErr_SetString(PyExc_ValueError, measure_error(ret_val));
    return NULL;
  }

  return Py_BuildValue("(f,f,f)", cen[0], cen[1], cen[2]);
}

// helper function to validate weights with multiple atomsels
static float *parse_two_selections_return_weight(PyAtomSelObject *a,
                                                 PyObject *args,
                                                 PyObject *kwargs,
                                                 AtomSel **othersel)
{
  const char *kwlist[] = {"selection", "weight", NULL};
  AtomSel *atomSel = a->atomSel;
  AtomSel *sel2;
  PyObject *weightobj = NULL;
  PyObject *other;
  DrawMolecule *mol;
  float *weight;

  if (!(mol = get_molecule(a))){
    PyErr_SetString(PyExc_ValueError, "selection is on a deleted molecule");
    return NULL;
  }

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|O", (char**) kwlist,
                                   &Atomsel_Type, &other, &weightobj))
    return NULL;

  weight = parse_weight(atomSel, weightobj);
  if (!weight)
    return NULL;

  sel2 = ((PyAtomSelObject *)other)->atomSel;
  if (!get_molecule((PyAtomSelObject *)other)) {
    PyErr_SetString(PyExc_ValueError,
                    "selection argument is on a deleted molecule");
    delete [] weight;
    return NULL;
  }

  if (atomSel->selected != sel2->selected) {
    PyErr_SetString(PyExc_ValueError, "Selections must have same number of "
                    "atoms");
    delete [] weight;
    return NULL;
  }

  *othersel = sel2;
  return weight;
}

static const char rmsdperres_doc[] =
"Get the per-residue root-mean-square (RMS) distance between selections\n\n"
"Args:\n"
"    selection (atomsel): Selection to compute distance to. Must have the\n"
"        same number of atoms as this selection\n"
"    weight (list of float): Weight for atoms, one per atom in selection.\n"
"        Optional\n"
"Returns:\n"
"    (list of float): RMSD between each residue in selection";
static PyObject *py_rmsdperresidue(PyAtomSelObject *a, PyObject *args,
                                   PyObject *kwargs)
{
  PyObject *returnlist = NULL;
  float *weight, *rmsd;
  AtomSel *sel2;
  int i, rc;

  weight = parse_two_selections_return_weight(a, args, kwargs, &sel2);
  if (!weight)
    return NULL;

  rmsd = new float[a->atomSel->selected];
  rc = measure_rmsd_perresidue(a->atomSel, sel2, a->app->moleculeList,
                               a->atomSel->selected, weight, rmsd);
  delete [] weight;

  if (rc < 0) {
    PyErr_SetString(PyExc_ValueError, measure_error(rc));
    goto failure;
  }

  //Build the python list.
  returnlist = PyList_New(rc);
  for (i = 0; i < rc; i++) {
    PyList_SET_ITEM(returnlist, i, PyFloat_FromDouble(rmsd[i]));
    if (PyErr_Occurred()) {
      PyErr_SetString(PyExc_RuntimeError, "Problem building rmsd list");
      goto failure;
    }
  }

  delete [] rmsd;
  return returnlist;

failure:
  delete [] rmsd;
  Py_XDECREF(returnlist);
  return NULL;
}

static const char rmsd_doc[] =
"Calculate the root-mean-square distance (RMSD) between selections. Atoms\n"
"must be in the same order in each selection\n\n"
"Args:\n"
"    selection (atomsel): Other selection to compute RMSD to. Must have\n"
"        the same number of atoms as this selection\n"
"    weight (list of float): Weight per atom, optional\n"
"Returns:\n"
"    (float): RMSD between selections.";
static PyObject *py_rmsd(PyAtomSelObject *a, PyObject *args, PyObject *kwargs)
{
  AtomSel *sel2;
  float *weight;
  float rmsd;
  int rc;

  weight = parse_two_selections_return_weight(a, args, kwargs, &sel2);
  if (!weight)
    return NULL;

  rc = measure_rmsd(a->atomSel, sel2, a->atomSel->selected,
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

static const char rmsd_q_doc[] =
"Calculate the root-mean-square distance (RMSD) between selection after\n"
"rotating them optimally\n\n"
"Args:\n"
"    selection (atomsel): Other selection to compute RMSD to. Must have\n"
"        the same number of atoms as this selection\n"
"    weight (list of float): Weight per atom, optional\n"
"Returns:\n"
"    (float): RMSD between selections.";
static PyObject *py_rmsd_q(PyAtomSelObject *a, PyObject *args, PyObject *kwargs)
{
  AtomSel *sel2;
  float *weight;
  float rmsd;
  int rc;

  weight = parse_two_selections_return_weight(a, args, kwargs, &sel2);
  if (!weight)
    return NULL;

  rc = measure_rmsd_qcp(a->app, a->atomSel, sel2, a->atomSel->selected,
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

static const char rmsf_doc[] =
"Measures the root-mean-square fluctuation (RMSF) along a trajectory on a\n"
"per-atom basis. RMSF is the mean deviation from the average position\n\n"
"Args:\n"
"    first (int): First frame to include. Defaults to 0 (beginning).\n"
"    last (int): Last frame to include. Defaults to -1 (end).\n"
"    step (int): Use every step-th frame. Defaults to 1 (all frames)\n"
"Returns:\n"
"    (list of float): RMSF for each atom in selection";
static PyObject *py_rmsf(PyAtomSelObject *a, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"first", "last", "step", NULL};
  int first = 0, last = -1, step = 1;
  PyObject *returnlist = NULL;
  int i, ret_val;
  float *rmsf;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|iii:atomsel.rmsf",
                                   (char**) kwlist, &first, &last, &step))
    return NULL;

  // Check molecule is valid
  if (!get_molecule(a)) {
    PyErr_SetString(PyExc_ValueError, "selection is on a deleted molecule");
    return NULL;
  }

  rmsf = new float[a->atomSel->selected];
  ret_val = measure_rmsf(a->atomSel, a->app->moleculeList, first, last,
                         step, rmsf);
  if (ret_val < 0) {
    PyErr_SetString(PyExc_RuntimeError, measure_error(ret_val));
    goto failure;
  }

  returnlist = PyList_New(a->atomSel->selected);
  for (i = 0; i < a->atomSel->selected; i++) {
    PyList_SET_ITEM(returnlist, i, PyFloat_FromDouble(rmsf[i]));
    if (PyErr_Occurred()) {
      PyErr_SetString(PyExc_RuntimeError, "cannot build rmsd list");
      goto failure;
    }
  }

  delete [] rmsf;
  return returnlist;

failure:
  delete [] rmsf;
  Py_XDECREF(returnlist);
  return NULL;
}

static const char rgyr_doc[] =
"Calculate the radius of gyration of this selection\n\n"
"Args:\n"
"    weight (list of float): Per-atom weights to apply during calcuation\n"
"        Must be same size as selection. Optional\n"
"Returns:\n"
"    (float): Radius of gyration";
static PyObject *py_rgyr(PyAtomSelObject *a, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"weight", NULL};
  PyObject *weightobj = NULL;
  float *weight;
  float rgyr;
  int rc;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O:atomsel.rgyr",
                                   (char**) kwlist, &weightobj))
    return NULL;

  weight = parse_weight(a->atomSel, weightobj);
  if (!weight)
    return NULL;

  rc = measure_rgyr(a->atomSel, a->app->moleculeList, weight, &rgyr);
  delete [] weight;

  if (rc < 0) {
    PyErr_SetString(PyExc_ValueError, measure_error(rc));
    return NULL;
  }

  return PyFloat_FromDouble(rgyr);
}

static const char fit_doc[] =
"Compute the transformation matrix for the root-mean-square (RMS) alignment\n"
"of this selection to the one given. The format of the matrix is suitable\n"
"for passing to the `atomsel.move()` method\n\n"
"Args:\n"
"    selection (atomsel): Selection to compute fit to. Must have the same\n"
"        number of atoms as this selection\n"
"    weight (list of float): Per-atom weights to apply during calculation\n"
"        Must be the same size as this selection. Optional\n"
"Returns:\n"
"    (16-tuple of float): Transformation matrix, in column major / fortran\n"
"        ordering";
static PyObject *py_fit(PyAtomSelObject *a, PyObject *args, PyObject *kwargs)
{
  PyObject *result;
  AtomSel *sel2;
  float *weight;
  Matrix4 mat;
  int rc, i;

  weight = parse_two_selections_return_weight(a, args, kwargs, &sel2);
  if (!weight)
    return NULL;

  rc = measure_fit(a->atomSel, sel2,
                   a->atomSel->coordinates(a->app->moleculeList),
                   sel2->coordinates(a->app->moleculeList),
                   weight, NULL, &mat);
  delete [] weight;

  if (rc < 0) {
    PyErr_SetString(PyExc_ValueError, measure_error(rc));
    return NULL;
  }

  result = PyTuple_New(16);
  for (i=0; i<16; i++) {
    PyTuple_SET_ITEM(result, i, PyFloat_FromDouble(mat.mat[i]));

    if (PyErr_Occurred()) {
      PyErr_SetString(PyExc_RuntimeError, "problem building fit matrix");
      Py_XDECREF(result);
      return NULL;
    }
  }

  return result;
}

static const char moveby_doc[] =
"Shift the selection by a vector\n\n"
"Args:\n"
"    vector (3-tuple of float): (x, y, z) movement to apply";
static PyObject *py_moveby(PyAtomSelObject *a, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"vector", NULL};
  AtomSel *atomSel = a->atomSel;
  DrawMolecule *mol;
  float offset[3];
  float *pos;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "(fff):atomsel.moveby",
                                   (char**) kwlist, &offset[0], &offset[1],
                                   &offset[2]))
    return NULL;

  if (!(mol = get_molecule(a))) {
    PyErr_SetString(PyExc_ValueError, "selection is on a deleted molecule");
    return NULL;
  }

  pos = atomSel->coordinates(a->app->moleculeList);
  if (!atomSel->selected || !pos) {
    PyErr_Format(PyExc_ValueError, "No coordinates in selection '%s'"
                 " molid %d", atomSel->cmdStr, atomSel->molid());
    return NULL;
  }

  for (int i = atomSel->firstsel; i <= atomSel->lastsel; i++) {
    if (atomSel->on[i]) {
      float *npos = pos + i * 3;
      vec_add(npos, npos, offset);
    }
  }

  mol->force_recalc(DrawMolItem::MOL_REGEN);
  Py_INCREF(Py_None);
  return Py_None;
}

static const char move_doc[] =
"Apply a coordinate transformation to the selection. To undo the move,\n"
"calculate the inverse coordinate transformation matrix with\n"
"`numpy.linalg.inv(matrix)` and pass that to this method\n\n"
"Args:\n"
"    matrix (numpy 16, matrix): Coordinate transformation, in form returned\n"
"        by `atomsel.fit()`, column major / fortran ordering";
static PyObject *py_move(PyAtomSelObject *a, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"matrix", NULL};
  AtomSel *atomSel = a->atomSel;
  DrawMolecule *mol;
  PyObject *matobj;
  Matrix4 mat;
  int err;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O:atomsel.move",
                                   (char**) kwlist, &matobj))
    return NULL;

  if (!(mol = get_molecule(a))) {
    PyErr_SetString(PyExc_ValueError, "selection is on a deleted molecule");
    return NULL;
  }

  if (!atomSel->selected || !atomSel->coordinates(a->app->moleculeList)) {
    PyErr_Format(PyExc_ValueError, "No coordinates in selection '%s'"
                 " molid %d", atomSel->cmdStr, atomSel->molid());
    return NULL;
  }

  // Exception set inside function call if an error is returned
  if (!py_get_vector(matobj, 16, mat.mat))
    return NULL;

  err = measure_move(atomSel, atomSel->coordinates(a->app->moleculeList), mat);
  if (err) {
    PyErr_SetString(PyExc_ValueError, measure_error(err));
    return NULL;
  }

  mol->force_recalc(DrawMolItem::MOL_REGEN);
  Py_INCREF(Py_None);
  return Py_None;
}

static const char contacts_doc[] =
"Finds all atoms in selection within a given distance of any atom in the\n"
"given selection that are not directly bonded to it. Selections can be in\n"
"different molecules.\n\n"
"Args:\n"
"    selection (atomsel): Atom selection to compare against\n"
"    cutoff (float): Distance cutoff for atoms to be considered contacting\n"
"Returns:\n"
"    (2 lists): Atom indices in this selection, and in given selection\n"
"        that are within the cutoff.";
static PyObject *contacts(PyAtomSelObject *a, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"selection", "cutoff", NULL};
  PyObject *result = NULL, *list1 = NULL, *list2 = NULL, *obj2 = NULL;
  GridSearchPair *pairlist, *tmp;
  const float *ts1, *ts2;
  AtomSel *sel1, *sel2;
  DrawMolecule *mol;
  float cutoff;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!f:atomsel.contacts",
                                   (char**) kwlist, &Atomsel_Type, &obj2,
                                   &cutoff))
    return NULL;

  if (!(mol = get_molecule(a))) {
    PyErr_SetString(PyExc_ValueError,
                    "this selection is in a deleted molecule");
    return NULL;
  }

  if (!(get_molecule((PyAtomSelObject *)obj2))) {
    PyErr_SetString(PyExc_ValueError,
                    "other selection is in a deleted molecule");
    return NULL;
  }
  sel1 = a->atomSel;
  sel2 = ((PyAtomSelObject *)obj2)->atomSel;

  if (!(sel1->selected) ||
      !(ts1 = sel1->coordinates(a->app->moleculeList))) {
    PyErr_Format(PyExc_ValueError, "No coordinates in selection '%s'"
                 " molid %d", sel1->cmdStr, sel1->molid());
    return NULL;
  }

  if (!(sel2->selected) ||
      !(ts2 = sel2->coordinates(a->app->moleculeList))) {
    PyErr_Format(PyExc_ValueError, "No coordinates in selection '%s'"
                 " molid %d", sel2->cmdStr, sel2->molid());
    return NULL;
  }

  // Check cutoffs are valid
  if (cutoff <= 0) {
    PyErr_SetString(PyExc_ValueError, "cutoff must be > 0");
    return NULL;
  }

  pairlist = vmd_gridsearch3(ts1, sel1->num_atoms, sel1->on, ts2,
                             sel2->num_atoms, sel2->on, cutoff, -1,
                             (sel1->num_atoms + sel2->num_atoms) * 27);

  list1 = PyList_New(0);
  list2 = PyList_New(0);
  if (PyErr_Occurred())
    goto failure;

  for (; pairlist; pairlist = tmp) {
    // throw out pairs that are already bonded
    MolAtom *a1 = mol->atom(pairlist->ind1);
    if (sel1->molid() != sel2->molid() || !a1->bonded(pairlist->ind2)) {
      // Since PyList_Append does *not* steal a reference, we need to
      // keep the object then decref it after append, to not leak references
      PyObject *tmp1 = as_pyint(pairlist->ind1);
      PyObject *tmp2 = as_pyint(pairlist->ind2);

      PyList_Append(list1, tmp1);
      PyList_Append(list2, tmp2);

      Py_DECREF(tmp1);
      Py_DECREF(tmp2);

      if (PyErr_Occurred())
        goto failure;
    }
    tmp = pairlist->next;
    free(pairlist);
  }

  result = PyList_New(2);
  PyList_SET_ITEM(result, 0, list1);
  PyList_SET_ITEM(result, 1, list2);
  if (PyErr_Occurred())
    goto failure;

  return result;

failure:
  PyErr_SetString(PyExc_RuntimeError, "Problem building contacts lists");
  Py_XDECREF(list1);
  Py_XDECREF(list2);
  Py_XDECREF(result);

  // Free linked list of GridSearchPairs, if iteration was broken out of
  for (; pairlist; pairlist = tmp) {
    tmp = pairlist->next;
    free(pairlist);
  }
  return NULL;
}

static const char py_hbonds_doc[] =
"Get hydrogen bonds present in current frame of selection using simple\n"
"geometric criteria.\n\n"
"Args:\n"
"    cutoff (float): Distance cutoff between donor and acceptor atoms\n"
"    maxangle (float): Angle cutoff between donor, hydrogen, and acceptor.\n"
"        Angle must be less than this value from 180 degrees.\n"
"    acceptor (atomsel): If given, atomselection for selector atoms, and this\n"
"        selection is assumed to have donor atoms. Both selections must be in\n"
"        the same molecule. If there is overlap between donor and acceptor\n"
"        selection, the output may be inaccurate. Optional.\n"
"Returns:\n"
"    (list of 3 lists): Donor atom indices, acceptor atom indices, and\n"
"        proton atom indices of identified hydrogen bonds\n";
// TODO: make it return a dict
static PyObject *py_hbonds(PyAtomSelObject *a, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"cutoff", "maxangle", "acceptor", NULL};
  PyObject *newdonlist, *newacclist, *newhydlist;
  PyObject *obj2 = NULL, *result = NULL;
  AtomSel *sel1 = a->atomSel, *sel2;
  int *donlist, *hydlist, *acclist;
  double cutoff, maxangle;
  int maxsize, rc, i;
  DrawMolecule *mol;
  const float *pos;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "dd|O!:atomsel.hbonds",
                                   (char**) kwlist, &cutoff, &maxangle,
                                   &Atomsel_Type, &obj2))
    return NULL;

  if (!(mol = get_molecule(a))) {
    PyErr_SetString(PyExc_ValueError, "selection is on a deleted molecule");
    return NULL;
  }

  // Acceptor must be an atomsel
  if (obj2 && !atomsel_Check(obj2)) {
    PyErr_SetString(PyExc_TypeError, "acceptor must be an atom selection");
    return NULL;
  }
  sel2 = obj2 ? ((PyAtomSelObject *)obj2)->atomSel : NULL;

  if (sel2 && !get_molecule((PyAtomSelObject*) obj2)) {
    PyErr_SetString(PyExc_ValueError, "acceptor selection is on a deleted molecule");
    return NULL;
  }

  // Selections should be on same molecule
  if (obj2 && mol != get_molecule((PyAtomSelObject*) obj2)) {
    PyErr_SetString(PyExc_ValueError, "acceptor selection must be in the same "
                    "molecule as this selection");
    return NULL;
  }

  if (!(a->atomSel->selected)
   || !(pos = sel1->coordinates(a->app->moleculeList))) {
    PyErr_Format(PyExc_ValueError, "No coordinates in selection '%s'"
                 " molid %d", a->atomSel->cmdStr, a->atomSel->molid());
    return NULL;
  }

  // Check cutoffs are valid
  if (cutoff <= 0) {
    PyErr_SetString(PyExc_ValueError, "cutoff must be > 0");
    return NULL;
  }

  if (maxangle < 0) {
    PyErr_SetString(PyExc_ValueError, "maxangle must be non-negative");
    return NULL;
  }

  // This heuristic is based on ice, where there are < 2 hydrogen bonds per
  // atom if hydrogens are in the selection, and exactly 2 if hydrogens are
  // not considered.
  maxsize = 2 * sel1->num_atoms;
  donlist = new int[maxsize];
  hydlist = new int[maxsize];
  acclist = new int[maxsize];
  rc = measure_hbonds((Molecule *)mol, sel1, sel2, cutoff, maxangle, donlist,
                      hydlist, acclist, maxsize);
  if (rc > maxsize) {
    delete [] donlist;
    delete [] hydlist;
    delete [] acclist;
    maxsize = rc;
    donlist = new int[maxsize];
    hydlist = new int[maxsize];
    acclist = new int[maxsize];
    rc = measure_hbonds((Molecule *)mol, sel1, sel2, cutoff, maxangle, donlist,
                        hydlist, acclist, maxsize);
  }
  if (rc < 0) {
    PyErr_SetString(PyExc_RuntimeError, "Problem calculating hbonds");
    return NULL;
  }

  newdonlist = PyList_New(rc);
  newacclist = PyList_New(rc);
  newhydlist = PyList_New(rc);
  if (PyErr_Occurred())
    goto failure;

  for (i = 0; i < rc; i++) {
    PyList_SET_ITEM(newdonlist, i, as_pyint(donlist[i]));
    PyList_SET_ITEM(newacclist, i, as_pyint(acclist[i]));
    PyList_SET_ITEM(newhydlist, i, as_pyint(hydlist[i]));

    if (PyErr_Occurred())
      goto failure;
  }

  result = PyList_New(3);
  PyList_SET_ITEM(result, 0, newdonlist);
  PyList_SET_ITEM(result, 1, newacclist);
  PyList_SET_ITEM(result, 2, newhydlist);

  if (PyErr_Occurred())
    goto failure;

  delete [] donlist;
  delete [] hydlist;
  delete [] acclist;
  return result;

failure:
  PyErr_SetString(PyExc_RuntimeError, "Problem building hbonds result");
  delete [] donlist;
  delete [] hydlist;
  delete [] acclist;
  Py_XDECREF(newdonlist);
  Py_XDECREF(newacclist);
  Py_XDECREF(newhydlist);
  Py_XDECREF(result);
  return NULL;
}

static const char sasa_doc[] =
"Get solvent accessible surface area of selection\n\n"
"Args:\n"
"    srad (float): Solvent radius\n"
"    samples (int): Maximum number of sample points per atom. Defaults to 500\n"
"    points (bool): True to also return the coordinates of surface points.\n"
"        Defaults to True.\n"
"    restrict (atomsel): Calculate SASA contributions from atoms in this\n"
"        selection only. Useful for getting SASA of residues in the context\n"
"        of their environment. Optional\n"
"Returns:\n"
"    (float): Solvent accessible surface area\n"
"    OR (float, list of 3-tuple): SASA, points, if points=True\n"
"Example to get percent solvent accssibility of a ligand\n"
"    >>> big_sel = atomsel('protein or resname LIG')\n"
"    >>> lig_sel = atomsel('resname LIG')\n"
"    >>> ligand_in_protein_sasa = big_sel.sasa(srad=1.4, restrict=lig_sel)\n"
"    >>> ligand_alone_sasa, points = lig_sel.sasa(srad=1.4, points=True)\n"
"    >>> print(100. * ligand_in_protein_sasa / ligand_alone_sasa)";
static PyObject *sasa(PyAtomSelObject *a, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"srad", "samples", "points", "restrict", NULL};
  PyObject *restrictobj = NULL, *pointsobj = NULL;
  AtomSel *sel, *restrictsel;
  const float *radii, *coords;
  ResizeArray<float> sasapts;
  float srad = 0, sasa = 0;
  int samples = 500, points = 0;
  DrawMolecule *mol;
  int rc;

  if (!(mol = get_molecule(a)))
    return NULL;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "f|iO&O!:atomsel.sasa",
                                   (char**) kwlist, &srad, &samples,
                                   convert_bool, &points, &Atomsel_Type,
                                   &restrictobj))
    return NULL;

  if (srad < 0) {
    PyErr_SetString(PyExc_ValueError, "srad must be non-negative.");
    return NULL;
  }

  if (samples <= 0) {
    PyErr_SetString(PyExc_ValueError, "samples must be positive.");
    return NULL;
  }

  // fetch the radii and coordinates
  sel = a->atomSel;
  radii = mol->extraflt.data("radius");
  coords = sel->coordinates(a->app->moleculeList);

  // if restrict is given, validate it and pull the selection out
  if (restrictobj && !get_molecule((PyAtomSelObject*) restrictobj)) {
    PyErr_SetString(PyExc_ValueError, "restrict sel is on deleted molecule");
    return NULL;
  }
  restrictsel = restrictobj ? ((PyAtomSelObject*) restrictobj)->atomSel : NULL;

  // actually calculate sasa
  rc = measure_sasa(sel, coords, radii, srad, &sasa,
                    points ? &sasapts : NULL, restrictsel, &samples);
  if (rc) {
    PyErr_SetString(PyExc_ValueError, measure_error(rc));
    return NULL;
  }

  // append surface points to the provided list object.
  if (points) {
    pointsobj  = PyList_New(sasapts.num()/3);

    for (int i = 0; i < sasapts.num() / 3; i++) {
      PyObject *coord = Py_BuildValue("ddd", sasapts[3L*i], sasapts[3L*i+1],
                                      sasapts[3L*i+2]);
      PyList_SET_ITEM(pointsobj, i, coord);

      if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError, "Problem building points list");
        Py_XDECREF(pointsobj);
        return NULL;
      }
    }
    return Py_BuildValue("dO", sasa, pointsobj);
  }

  return PyFloat_FromDouble(sasa);
}

#if defined(VMDCUDA)
#include "CUDAMDFF.h"
static const char mdffsim_doc[] =
"Compute a density map\n\n"
"Args:\n"
"    resolution (float): Resolution. Defaults to 10.0\n"
"    spacing (float): Space between adjacent voxel points.\n"
"        Defaults to 0.3*resolution\n"
"Returns:\n"
"    (4-element list): Density map, consisting of 4 lists:\n"
"        1) A 1D list of the grid values at each point\n"
"        2) 3 integers describing the x, y, z lengths in number of grid cells\n"
"        3) 3 floats describing the (x,y,z) coordinates of the origin\n"
"        4) 9 floats describing the deltas for the x, y, and z axes\n"
"Example usage for export to numpy:\n"
"    >>> data, shape, origin, delta = asel.mdffsim(10,3)\n"
"    >>> data = np.array(data)\n"
"    >>> shape = np.array(shape)\n"
"    >>> data = data.reshape(shape, order='F')\n"
"    >>> delta = np.array(delta).reshape(3,3)\n"
"    >>> delta /= shape-1";
static PyObject *py_mdffsim(PyAtomSelObject *a, PyObject *args,
                            PyObject *kwargs)
{
  const char *kwlist[] = {"resolution", "spacing", NULL};
  PyObject *data, *deltas, *origin, *size, *result;
  double gspacing = 0, resolution = 10.0;
  VolumetricData *synthvol = NULL;
  AtomSel *sel = a->atomSel;
  int i, cuda_err, quality;
  Molecule *mymol;
  float radscale;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|dd:atomsel:mdffsim",
                                   (char**) kwlist, &resolution, &gspacing))
    return NULL;

  if (gspacing < 0) {
    PyErr_SetString(PyExc_ValueError, "Spacing must be non-negative");
    return NULL;
  }

  if (resolution < 0) {
    PyErr_SetString(PyExc_ValueError, "Resolution must be non-negative");
    return NULL;
  }

  if (!(mymol = a->app->moleculeList->mol_from_id(sel->molid()))) {
    PyErr_SetString(PyExc_ValueError, "Selection points to a deleted molecule.");
    return NULL;
  }

  quality = resolution >= 9 ? 0 : 3;
  radscale = .2*resolution;
  if (gspacing == 0) {
    gspacing = 1.5*radscale;
  }

  cuda_err = vmd_cuda_calc_density(sel, a->app->moleculeList, quality, radscale,
                                   gspacing, &synthvol, NULL, NULL, 1);
  if (cuda_err == -1) {
    PyErr_SetString(PyExc_ValueError, "CUDA Error, no map returned.");
    return NULL;
  }

  data = PyList_New(synthvol->xsize * synthvol->ysize * synthvol->zsize);
  for (i = 0; i < synthvol->xsize * synthvol->ysize * synthvol->zsize; i++) {
    PyList_SET_ITEM(data, i, PyFloat_FromDouble((double)synthvol->data[i]));
    if (PyErr_Occurred())
      goto failure;
  }

  deltas = PyList_New(9);
  origin = PyList_New(3);
  size = PyList_New(3);
  if (PyErr_Occurred())
    goto failure;

  // (x, y, z) size
  PyList_SET_ITEM(size, 0, as_pyint(synthvol->xsize));
  PyList_SET_ITEM(size, 1, as_pyint(synthvol->ysize));
  PyList_SET_ITEM(size, 2, as_pyint(synthvol->zsize));
  if (PyErr_Occurred())
    goto failure;

  // Build length 9 array of axis deltas
  for (i = 0; i < 3; i++) {
    PyList_SET_ITEM(deltas, i, PyFloat_FromDouble(synthvol->xaxis[i]));
    PyList_SET_ITEM(deltas, 3+i, PyFloat_FromDouble(synthvol->yaxis[i]));
    PyList_SET_ITEM(deltas, 6+i, PyFloat_FromDouble(synthvol->zaxis[i]));
    PyList_SET_ITEM(origin, i, PyFloat_FromDouble(synthvol->origin[i]));
    if (PyErr_Occurred())
      goto failure;
  }

  delete synthvol;
  synthvol = NULL;

  result = PyList_New(4);
  PyList_SET_ITEM(result, 0, data);
  PyList_SET_ITEM(result, 1, size);
  PyList_SET_ITEM(result, 2, origin);
  PyList_SET_ITEM(result, 3, deltas);
  if (PyErr_Occurred())
    goto failure;

  return result;

failure:
  PyErr_SetString(PyExc_RuntimeError, "Problem building grid");
  Py_XDECREF(data);
  Py_XDECREF(deltas);
  Py_XDECREF(origin);
  Py_XDECREF(size);
  Py_XDECREF(result);

  if(synthvol)
    delete synthvol;

  return NULL;
}

static const char mdffcc_doc[] =
"Get the cross correlation between a volumetric map and the current "
"selection.\n\n"
"Args:\n"
"    volid (int): Index of volumetric dataset\n"
"    resolution (float): Resolution. Defaults to 10.0\n"
"    spacing (float): Space between adjacent voxel points.\n"
"        Defaults to 0.3*resolution\n"
"Returns:\n"
"    (float): Cross correlation for a single frame";
static PyObject *py_mdffcc(PyAtomSelObject *a, PyObject *args,
                           PyObject *kwargs)
{
  const char *kwlist[] = {"volid", "resolution", "spacing", NULL};
  const VolumetricData *volmapA = NULL;
  int volid, quality, cuda_err;
  const AtomSel *sel = a->atomSel;
  float return_cc, radscale;
  double resolution = 10.0;
  double gspacing = 0;
  Molecule *mymol;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i|dd:atomsel.mdffcc",
                                   (char**) kwlist, &volid, &resolution,
                                   &gspacing))
    return NULL;

  if (gspacing < 0) {
    PyErr_SetString(PyExc_ValueError, "Spacing must be non-negative");
    return NULL;
  }

  if (resolution < 0) {
    PyErr_SetString(PyExc_ValueError, "Resolution must be non-negative");
    return NULL;
  }

  if (!(mymol = a->app->moleculeList->mol_from_id(sel->molid()))) {
    PyErr_SetString(PyExc_ValueError, "Selection points to a deleted molecule");
    return NULL;
  }

  if (!(volmapA = mymol->get_volume_data(volid))) {
    PyErr_SetString(PyExc_ValueError, "Invalid volume specified. Make sure it's"
                    " loaded into the same molecule as the selection");
    return NULL;
  }

  quality = resolution >= 9 ? 0 : 3;
  radscale = .2*resolution;
  if (!gspacing) {
    gspacing = 1.5*radscale;
  }


  cuda_err = vmd_cuda_compare_sel_refmap(sel, a->app->moleculeList, quality,
                                         radscale, gspacing, volmapA, NULL,
                                         NULL, NULL, &return_cc, 0.0f, 0);
  if (cuda_err == -1) {
    PyErr_SetString(PyExc_ValueError, "CUDA Error, no map returned.");
    return NULL;
  }

  return PyFloat_FromDouble(return_cc);
}
#endif

// __len__
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
  { "list_attributes", (PyCFunction)py_list_attrs, METH_VARARGS|METH_KEYWORDS, listattrs_doc },
  { "get", (PyCFunction)legacy_atomsel_get, METH_VARARGS|METH_KEYWORDS, get_doc  },
  { "set", (PyCFunction)legacy_atomsel_set, METH_VARARGS|METH_KEYWORDS, set_doc },
  { "update", (PyCFunction)py_update, METH_NOARGS, update_doc },
  { "write", (PyCFunction)py_write, METH_VARARGS|METH_KEYWORDS, write_doc },
  { "minmax", (PyCFunction)minmax, METH_VARARGS|METH_KEYWORDS, minmax_doc },
  { "center", (PyCFunction)center, METH_VARARGS|METH_KEYWORDS, center_doc },
  { "rmsd", (PyCFunction)py_rmsd, METH_VARARGS|METH_KEYWORDS, rmsd_doc },
  { "rmsdQCP", (PyCFunction)py_rmsd_q, METH_VARARGS|METH_KEYWORDS, rmsd_q_doc },
  { "rmsf", (PyCFunction)py_rmsf, METH_VARARGS|METH_KEYWORDS, rmsf_doc },
  { "centerperresidue", (PyCFunction)centerperresidue, METH_VARARGS|METH_KEYWORDS, centerperres_doc },
  { "rmsdperresidue", (PyCFunction)py_rmsdperresidue, METH_VARARGS|METH_KEYWORDS, rmsdperres_doc },
  { "rmsfperresidue", (PyCFunction)py_rmsfperresidue, METH_VARARGS|METH_KEYWORDS, rmsfperres_doc },
  { "rgyr", (PyCFunction)py_rgyr, METH_VARARGS|METH_KEYWORDS, rgyr_doc },
  { "fit", (PyCFunction)py_fit, METH_VARARGS|METH_KEYWORDS, fit_doc },
  { "move", (PyCFunction)py_move, METH_VARARGS|METH_KEYWORDS, move_doc },
  { "moveby", (PyCFunction)py_moveby, METH_VARARGS|METH_KEYWORDS, moveby_doc },
  { "contacts", (PyCFunction)contacts, METH_VARARGS|METH_KEYWORDS, contacts_doc },
  { "hbonds", (PyCFunction)py_hbonds, METH_VARARGS|METH_KEYWORDS, py_hbonds_doc },
  { "sasa", (PyCFunction)sasa, METH_VARARGS|METH_KEYWORDS, sasa_doc },
#if defined(VMDCUDA)
  { "mdffsim", (PyCFunction)py_mdffsim, METH_VARARGS|METH_KEYWORDS, mdffsim_doc },
  { "mdffcc", (PyCFunction)py_mdffcc, METH_VARARGS|METH_KEYWORDS, mdffcc_doc },
#endif
  { NULL, NULL }
};

// Atom selection iterator
//
typedef struct {
  PyObject_HEAD
  int index;
  PyAtomSelObject * a;
} atomsel_iterobject;

PyObject *atomsel_iter(PyObject *);

PyObject *iter_next(atomsel_iterobject *it) {
  for ( ; it->index < it->a->atomSel->num_atoms; ++it->index) {
    if (it->a->atomSel->on[it->index])
      return as_pyint(it->index++);
  }
  return NULL;
}

void iter_dealloc(atomsel_iterobject *it) {
  Py_XDECREF(it->a);
}

// Length
PyObject *iter_len(atomsel_iterobject *it) {
  return as_pyint(it->a->atomSel->selected);
}

PyMethodDef iter_methods[] = {
  {"__length_hint__", (PyCFunction)iter_len, METH_NOARGS },
  {NULL, NULL}
};

#if PY_MAJOR_VERSION >= 3
  PyTypeObject itertype = {
    PyObject_HEAD_INIT(&PyType_Type)
    "atomsel.iterator",
    sizeof(atomsel_iterobject), 0, // basic, item size
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

PyTypeObject Atomsel_Type = {
    PyObject_HEAD_INIT(0)
    "atomsel",
    sizeof(PyAtomSelObject), 0, // basic, item size
    (destructor)atomsel_dealloc, //dealloc
    0, // tp_print
    0, 0, // tp get and set attr
    0, // tp_as_async
    (reprfunc)atomsel_repr, // tp_repr
    0, 0, &atomsel_mapping, // as number, sequence, mapping
    0, 0, (reprfunc)atomsel_str, // hash, call, str
    (getattrofunc) atomsel_getattro, // getattro
    (setattrofunc) atomsel_setattro, // setattro
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
    sizeof(atomsel_iterobject),
    0, // itemsize
    /* methods */
    (destructor)iter_dealloc,
  0,          /* tp_print */
  0,          /* tp_getattr */
  0,          /* tp_setattr */
  0,          /* tp_compare */
  0,          /* tp_repr */
  0,          /* tp_as_number */
  0,          /* tp_as_sequence */
  0,          /* tp_as_mapping */
  0,          /* tp_hash */
  0,          /* tp_call */
  0,          /* tp_str */
  PyObject_GenericGetAttr,    /* tp_getattro */
  0,          /* tp_setattro */
  0,          /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT,                     /* tp_flags */
  atomsel_doc, /* tp_doc */
  0,          /* tp_traverse */
  0,          /* tp_clear */
  0,          /* tp_richcompare */
  0,          /* tp_weaklistoffset */
  PyObject_SelfIter,      /* tp_iter */
  (iternextfunc)iter_next,    /* tp_iternext */
  iter_methods,              /* tp_methods */
  0,          /* tp_members */
  };

PyTypeObject Atomsel_Type = {
  PyObject_HEAD_INIT(0) /* Must fill in type value later */
  0,          /* ob_size */
  "atomsel.atomsel",     /* tp_name */
  sizeof(PyAtomSelObject),   /* tp_basicsize */
  0,          /* tp_itemsize */
  (destructor)atomsel_dealloc,   /* tp_dealloc */
  0,          /* tp_print */
  0,          /* tp_getattr - getattro used instead */
  0,          /* tp_setattr  - getattro used instead */
  0,          /* tp_compare */
  (reprfunc)atomsel_repr,      /* tp_repr */
  0,          /* tp_as_number */
  0,          /* tp_as_sequence */
  &atomsel_mapping,          /* tp_as_mapping */
  0,          /* tp_hash */
  0,          /* tp_call */
  (reprfunc)atomsel_str,          /* tp_str */
  (getattrofunc) atomsel_getattro, //tp_getattro
  (setattrofunc) atomsel_setattro, // setattro
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
  0,          /* tp_init */
  PyType_GenericAlloc,      /* tp_alloc */
  atomsel_new,       /* tp_new */
  _PyObject_Del,       /* tp_free */
};
#endif

// tp_iter for atomsel type
PyObject *atomsel_iter(PyObject *self) {
  atomsel_iterobject *iter = PyObject_New(atomsel_iterobject, &itertype);
  if (!iter)
    return NULL;

  Py_INCREF( iter->a = (PyAtomSelObject *)self );
  iter->index = 0;
  return (PyObject *)iter;
}

// Atomsel is a type, not a module. So we just initialize the type
// and return it.
PyObject* initatomsel(void) {

#if PY_MAJOR_VERSION >= 3
  ((PyObject*)(&Atomsel_Type))->ob_type = &PyType_Type;
#else
  Atomsel_Type.ob_type = &PyType_Type;
#endif

  Py_INCREF((PyObject *)&Atomsel_Type);

  if (PyType_Ready(&Atomsel_Type) < 0) {
    PyErr_SetString(PyExc_RuntimeError, "Problem initializing atomsel type");
    return NULL;
  }

  return (PyObject*) &Atomsel_Type;
}

