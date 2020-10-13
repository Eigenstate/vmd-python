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
 *      $RCSfile: py_molecule.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.73 $       $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Python molecule manipulation interface
 ***************************************************************************/

#include "py_commands.h"
#include <ctype.h>
#include <stdlib.h>

#include "config.h"
#include "utilities.h"
#include "VMDApp.h"
#include "JString.h"
#include "Molecule.h"
#include "MoleculeList.h"

// Handle legacy keywords beg, end, skip, with DeprecationWarning
// that is printed out only once
void handle_legacy_keywords(PyObject *kwargs)
{
  PyObject *value;
  int warn = 0;

  // If no keyword arguments, there will be nothing to do
  if (!kwargs)
    return;

  // beg -> first
  if ((value = PyDict_GetItemString(kwargs, "beg"))) {
    PyDict_SetItemString(kwargs, "first", value);
    PyDict_DelItemString(kwargs, "beg");
    warn = 1;
  }

  // end -> last
  if ((value = PyDict_GetItemString(kwargs, "end"))) {
    PyDict_SetItemString(kwargs, "last", value);
    PyDict_DelItemString(kwargs, "end");
    warn = 1;
  }

  // skip -> stride
  if ((value = PyDict_GetItemString(kwargs, "skip"))) {
    PyDict_SetItemString(kwargs, "stride", value);
    PyDict_DelItemString(kwargs, "skip");
    warn = 1;
  }

  if (warn) {
    PyErr_Warn(PyExc_DeprecationWarning, "beg, end, skip keywords are now "
               "first, last, stride");
  }
}

static const char mol_num_doc[] =
"Get the number of loaded molecules\n\n"
"Returns:\n"
"    (int): Number of molecules";
static PyObject* py_mol_num(PyObject *self, PyObject *args)
{
  VMDApp *app;

  if (!(app = get_vmdapp()))
    return NULL;

  return as_pyint(app->num_molecules());
}

static const char mol_listall_doc[] =
"List all valid molecule IDs\n\n"
"Returns:\n"
"    (list of int): Molids";
static PyObject* py_mol_listall(PyObject *self, PyObject *args) {

  PyObject *newlist = NULL;
  VMDApp *app;
  int i, num;

  if (!(app = get_vmdapp()))
    return NULL;

  num = app->num_molecules();
  if (!(newlist = PyList_New(num)))
    goto failure;

  for (i = 0; i < num; i++) {
    PyList_SET_ITEM(newlist, i, as_pyint(app->molecule_id(i)));
    if (PyErr_Occurred())
      goto failure;
  }
  return newlist;

failure:
  Py_XDECREF(newlist);
  PyErr_SetString(PyExc_ValueError, "Problem listing molids");
  return NULL;
}

static const char mol_exists_doc[] =
"Check if a molecule ID is valid\n\n"
"Args:\n"
"    molid (int): Molecule ID to query\n"
"Returns:\n"
"    (bool) True if molid is valid";
static PyObject* py_mol_exists(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", NULL};
  VMDApp *app;
  int molid;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i:molecule.exists",
                                   (char**) kwlist, &molid))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  return as_pyint(app->molecule_valid_id(molid));
}

static const char mol_name_doc[] =
"Get name of a given molecule\n\n"
"Args:\n"
"    molid (int): Molecule ID to query\n"
"Returns:\n"
"    (str): Molecule name";
static PyObject* py_mol_name(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", NULL};
  VMDApp *app;
  int molid;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i:molecule.name",
                                  (char**) kwlist, &molid))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app))
    return NULL;

  return as_pystring(app->molecule_name(molid));
}

static const char mol_numatoms_doc[] =
"Get number of atoms in a given molecule\n\n"
"Args:\n"
"    molid (int): Molecule ID to query\n"
"Returns:\n"
"    (int): Number of atoms present in molecule";
static PyObject* py_mol_numatoms(PyObject *self, PyObject *args,
                                 PyObject *kwargs)
{
  const char *kwlist[] = {"molid", NULL};
  VMDApp *app;
  int molid;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i:molecule.numatoms",
                                  (char**) kwlist, &molid))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app))
    return NULL;

  return as_pyint(app->molecule_numatoms(molid));
}

static const char mol_new_doc[] =
"Create a new molecule, with optional number of 'empty' atoms\n\n"
"Args:\n"
"    name (str): Name of new molecule\n"
"    natoms (int): Number of empty atoms in new molecule, optional.\n"
"Returns:\n"
"    (int): Molecule ID of created molecule";
static PyObject* py_mol_new(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"name", "natoms", NULL};
  int natoms = 0;
  VMDApp *app;
  char *name;
  int molid;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|i:molecule.new",
                                   (char**) kwlist, &name, &natoms))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (natoms < 0) {
    PyErr_SetString(PyExc_ValueError, "natoms must be >= 0");
    return NULL;
  }

  molid = app->molecule_new(name,natoms);

  if (molid < 0) {
    PyErr_Format(PyExc_ValueError, "Unable to create molecule name '%s'", name);
    return NULL;
  }

  return as_pyint(molid);
}

static const char mol_load_doc[] =
"Load a molecule from a file. Can optionally read in coordinate files as\n"
"well. Coordinate data does not have to be present in structure file.\n\n"
"Args:\n"
"    struct_type (str): File type for structure data, or 'graphics' for\n"
"        a blank molecule to add graphics to.\n"
"    struct_file (str): Filename for structure data\n"
"    coord_type (str): File type for coordinate data. Optional.\n"
"    coord_file (str): Filename for coordinate data. Optional.\n"
"Returns:\n"
"    (int): Molecule ID of loaded molecule";
static PyObject* py_mol_load(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"struct_type", "struct_file", "coord_type",
                          "coord_file", NULL};
  char *coor = NULL, *cfname = NULL;
  char *structure, *sfname;
  FileSpec spec;
  VMDApp *app;
  int molid;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ss|zz:molecule.load",
                                   (char**) kwlist, &structure, &sfname,
                                   &coor, &cfname))
    return NULL;

  // if a coordinates file type was specified, a coordinate file must be given
  // as well, and vice versa.
  if (coor && !cfname) {
    PyErr_SetString(PyExc_ValueError, "No coordinate file specified");
    return NULL;
  }
  if (cfname && !coor) {
    PyErr_SetString(PyExc_ValueError, "No coordinate type specified");
    return NULL;
  }

  if (!(app = get_vmdapp()))
    return NULL;

  // Special-case graphics molecules to load as "blank" molecules
  if (!strcmp(structure, "graphics"))
    return as_pyint(app->molecule_new(sfname, 0));

  // Load all frames at once by default
  spec.waitfor = FileSpec::WAIT_ALL; // load all frames at once by default
  molid = app->molecule_load(-1, sfname, structure, &spec);
  if (molid < 0) {
    PyErr_Format(PyExc_RuntimeError, "Unable to load structure file '%s' "
                 "format '%s'", sfname, structure);
    return NULL;
  }

  // Load coordinate data if provided
  if (cfname && (app->molecule_load(molid, cfname, coor, &spec) < 0)) {
      PyErr_Format(PyExc_RuntimeError, "Unable to load coordinate file '%s' "
                   "format '%s'", cfname, coor);
  }

  return as_pyint(molid);
}

static const char mol_cancel_doc[] =
"Cancel background loading of files for given molid\n\n"
"Args:\n"
"    molid (int): Molecule ID to cancel loading on";
static PyObject* py_mol_cancel(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", NULL};
  VMDApp *app;
  int molid;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i:molecule.cancel",
                                   (char**) kwlist, &molid))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app))
    return NULL;

  app->molecule_cancel_io(molid);

  Py_INCREF(Py_None);
  return Py_None;
}

static const char mol_delete_doc[] =
"Delete a molecule\n\n"
"Args:\n"
"    molid (int): Molecule ID to delete";
static PyObject* py_mol_delete(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", NULL};
  VMDApp *app;
  int molid;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i:molecule.delete",
                                   (char**) kwlist,  &molid))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app))
    return NULL;

  app->molecule_delete(molid);

  Py_INCREF(Py_None);
  return Py_None;
}

static const char get_top_doc[] =
"Get the ID of the top molecule\n\n"
"Returns:\n"
"    (int): Molecule ID of top molecule";
static PyObject* py_get_top(PyObject *self, PyObject *args)
{
  VMDApp *app;

  if (!(app = get_vmdapp()))
    return NULL;

  return as_pyint(app->molecule_top());
}

static const char set_top_doc[] =
"Set the top molecule\n\n"
"Args:\n"
"    molid (int): Molecule ID of new top molecule";
static PyObject* py_set_top(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", NULL};
  VMDApp *app;
  int molid;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i:molecule.set_top",
                                   (char**) kwlist, &molid))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app))
    return NULL;

  app->molecule_make_top(molid);

  Py_INCREF(Py_None);
  return Py_None;
}

static const char mol_read_doc[] =
"Read coordinate data into an existing molecule\n\n"
"Args:\n"
"    molid (int): Molecule ID to read data into\n"
"    filetype (str): File type of coordinate data\n"
"    filename (str): Path to coordinate data\n"
"    first (int): First frame to read. Defaults to 0 (first frame)\n"
"    last (int): Last frame to read, or -1 for the end. Defaults to -1\n"
"    stride (int): Frame stride. Defaults to 1 (read all frames)\n"
"    waitfor (int): Number of frames to load before returning. Defaults to 1,\n"
"        then loads asyncronously in background. Set to -1 to load all frames\n"
"    volsets (list of int): Indices of volumetric datasets to read in from\n"
"        the file. Invalid indices will be ignored.";
static PyObject* py_mol_read(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "filetype", "filename", "first", "last",
                          "stride", "waitfor", "volsets", NULL};
  PyObject *volsets = NULL;
  char *filename, *type;
  int molid, rc;
  FileSpec spec;
  VMDApp *app;

  // Set default options
  spec.first = 0;
  spec.last = -1;
  spec.stride = 1;
  spec.waitfor = FileSpec::WAIT_BACK;

  // Handle legacy keywords beg, end, skip, but emit DeprecationWarning
  handle_legacy_keywords(kwargs);

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iss|iiiiO!:molecule.read",
                                   (char**) kwlist, &molid, &type, &filename,
                                   &spec.first, &spec.last, &spec.stride,
                                   &spec.waitfor, &PyList_Type, &volsets))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  // Read volumetric data if provided
  if (volsets) {
    spec.nvolsets = PyList_Size(volsets);
    spec.setids = new int[spec.nvolsets];

    for (int i=0; i<spec.nvolsets; i++) {
      spec.setids[i] = as_int(PyList_GET_ITEM(volsets, i));
      if (PyErr_Occurred())
        goto failure;
    }

  // If not volumetric data, have a default of {0} for setids so that it isn't
  // always necessary to specify the set id's. This should be ignored if the
  // file type can't read volumetric datasets
  } else {
    spec.nvolsets = 1;
    spec.setids = new int[1];
    spec.setids[0] = 0;
  }

  // Need to do more error checking here so we don't leak setids on error.
  rc = app->molecule_load(molid, filename, type, &spec);
  if (rc < 0) {
    PyErr_Format(PyExc_RuntimeError, "Unable to read filename '%s' "
                 "format '%s'", filename, type);
    goto failure;
  }

  return as_pyint(rc);

failure:
  delete [] spec.setids;
  spec.setids = NULL;
  return NULL;
}

static const char mol_write_doc[] =
"Write coordinate and topology data from a loaded molecule to a file\n\n"
"Args:\n"
"    molid (int): Molecule ID to read data into\n"
"    filetype (str): File type of coordinate data\n"
"    filename (str): Path to coordinate data\n"
"    first (int): First frame to read. Defaults to 0 (first frame)\n"
"    last (int): Last frame to read, or -1 for the end. Defaults to -1\n"
"    stride (int): Frame stride. Defaults to 1 (write all frames)\n"
"    waitfor (int): Number of frames to wait for before returning control. Defaults to -1 (wait for all frames)"
"    selection (atomsel): Atom indices to write. Defaults to all atoms."
"Returns:\n"
"    (int): Number of frames written";
static PyObject* py_mol_write(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "filetype", "filename", "first", "last",
                          "stride", "waitfor", "selection", NULL};
  PyObject *selobj = NULL;
  char *filename, *type;
  int numframes, molid;
  int *on = NULL;
  FileSpec spec;
  Molecule *mol;
  VMDApp *app;

  // Set default options
  spec.first = 0;
  spec.last = -1;
  spec.stride = 1;
  spec.waitfor = FileSpec::WAIT_ALL;

  // Handle legacy keywords beg, end, skip, but emit DeprecationWarning
  handle_legacy_keywords(kwargs);

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iss|iiiiO:molecule.write",
                                   (char**) kwlist, &molid, &type, &filename,
                                   &spec.first, &spec.last, &spec.stride, &spec.waitfor,
                                   &selobj))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app))
    return NULL;

  // Bounds check frames
  mol = app->moleculeList->mol_from_id(molid);
  if (spec.first >= mol->numframes() || spec.last >= mol->numframes()
   || spec.first < -1 || spec.last < -1) {
    PyErr_Format(PyExc_ValueError, "frames '%d-%d' out of bounds for molecule "
                "'%d'", spec.first, spec.last, molid);
    return NULL;
  }

  // Get list of atom indices from optional atop selection object
  if (selobj && selobj != Py_None) {
    AtomSel *sel = atomsel_AsAtomSel( selobj );
    if (!sel || sel->molid() != molid) {
      PyErr_SetString(PyExc_ValueError, "Atomsel must be valid and must "
                      "reference same molecule as coordinates");
      return NULL;
    }
    on = sel->on;
  }
  spec.selection = on;

  numframes = app->molecule_savetrajectory(molid, filename, type, &spec);
  if (numframes < 0) {
    PyErr_Format(PyExc_ValueError, "Unable to save file '%s'", filename);
    return NULL;
  }

  return as_pyint(numframes);
}

static const char numframes_doc[] =
"Get the number of loaded frames in a molecule\n\n"
"Args:\n"
"    molid (int): Molecule ID to query\n"
"Returns:\n"
"    (int): Number of loaded frames";
static PyObject* py_numframes(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", NULL};
  VMDApp *app;
  int molid;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i:molecule.numframes",
                                  (char**) kwlist, &molid))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app))
    return NULL;

  return as_pyint(app->molecule_numframes(molid));
}

static const char get_frame_doc[] =
"Get the current frame\n\n"
"Args:\n"
"    molid (int): Molecule ID to query\n"
"Returns:\n"
"    (int): Current frame of molecule";
static PyObject* py_get_frame(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", NULL};
  VMDApp *app;
  int molid;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i:molecule.get_frame",
                                  (char**) kwlist,  &molid))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app))
    return NULL;

  return as_pyint(app->molecule_frame(molid));
}

static const char set_frame_doc[] =
"Set the frame of a molecule\n\n"
"Args:\n"
"    molid (int): Molecule ID to set\n"
"    frame (int): New frame for molecule";
static PyObject* py_set_frame(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "frame", NULL};
  int molid, frame;
  Molecule *mol;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii:molecule.set_frame",
                                   (char**) kwlist, &molid, &frame))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app))
    return NULL;

  mol = app->moleculeList->mol_from_id(molid);

  // Ensure frame is sane
  if (frame < 0 || frame >= mol->numframes()) {
    PyErr_Format(PyExc_ValueError, "frame '%d' out of bounds for molecule '%d'",
                 frame, molid);
    return NULL;
  }

  mol->override_current_frame(frame);
  mol->change_ts();

  Py_INCREF(Py_None);
  return Py_None;
}

static const char delframe_doc[] =
"Delete a frame or frame(s) from a loaded molecule\n\n"
"Args:\n"
"    molid (int): Molecule ID to delete frames from\n"
"    first (int): First frame to delete, inclusive. Defaults to 0.\n"
"    last (int): Last frame to delete, inclusive. Defaults to -1 (last frame)\n"
"    stride (int): Keep every Nth frame in range to delete. Defaults to 0.";
static PyObject* py_delframe(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "first", "last", "stride", NULL};
  int molid = 0, beg=0, end=-1, skip=0;
  VMDApp *app;

  // Handle legacy keywords beg, end, skip, but emit DeprecationWarning
  handle_legacy_keywords(kwargs);

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i|iii:molecule.delframe",
                                   (char**) kwlist, &molid, &beg, &end, &skip))
    return NULL;

  // Handle legacy keywords beg, end, skip, but emit DeprecationWarning
  handle_legacy_keywords(kwargs);

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app))
    return NULL;

  if (!app->molecule_deleteframes(molid, beg, end, skip)) {
    PyErr_SetString(PyExc_ValueError, "Unable to delete frames");
    return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}

static const char dupframe_doc[] =
"Duplicate a frame.\n\n"
"Args:\n"
"    molid (int): Molecule ID to duplicate frames on. If there are no loaded\n"
"        frames in this molecule, a new frame with all zero coordinates will\n"
"        be created.\n"
"    frame (int): Frame to duplicate. Defaults to -1 (current frame)";
static PyObject* py_dupframe(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "frame", NULL};
  int frame = -1;
  VMDApp *app;
  int molid;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i|i:molecule.dupframe",
                                   (char**) kwlist, &molid, &frame))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app))
    return NULL;

  if (!app->molecule_dupframe(molid, frame)) {
    PyErr_Format(PyExc_ValueError, "Unable to duplicate frame '%d' in "
                 "molid '%d'", frame, molid);
    return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}

static const char mol_ssrecalc_doc[] =
"Recalculate secondary structure for molecule\n\n"
"Args:\n"
"    molid (int): Molecule ID\n"
"Raises:\n"
"    RuntimeError: If structure could not be calculated, usually due to the\n"
"        Stride program being unavailable";
static PyObject *py_mol_ssrecalc(PyObject *self, PyObject *args,
                                 PyObject *kwargs)
{
  const char *kwlist[] = {"molid", NULL};
  int molid, rc;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i:molecule.ssrecalc",
                                   (char**) kwlist, &molid))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app))
    return NULL;

  // Release the GIL for this calculation
  Py_BEGIN_ALLOW_THREADS
  rc = app->molecule_ssrecalc(molid);
  Py_END_ALLOW_THREADS

  if (!rc) {
    PyErr_Format(PyExc_RuntimeError, "Secondary structure could not be "
                 "calculated for molid '%d'", molid);
    return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}

static const char mol_rename_doc[] =
"Change the name of a molecule\n\n"
"Args:\n"
"    molid (int): Molecule ID to rename\n"
"    name (str): New name for molecule";
static PyObject* py_mol_rename(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "name", NULL};
  char *newname;
  VMDApp *app;
  int molid;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "is:molecule.rename",
                                   (char**) kwlist, &molid, &newname))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app))
    return NULL;

  if (!app->molecule_rename(molid, newname)) {
    PyErr_Format(PyExc_ValueError, "Unable to rename molecule '%d'", molid);
    return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}

static const char add_volumetric_doc[] =
"Add a new volumetric dataset to a molecule\n\n"
"Args:\n"
"    molid (int): Molecule ID to add dataset to\n"
"    name (str): Dataset name\n"
"    size (3-tuple of int): Size of grid data in X, Y, and Z dimensions\n"
"    data (list of float): Grid data, of length xsize * ysize * zsize\n"
"    origin (3-tuple of float): (x,y,z) coordinate of grid origin, at lower left\n"
"        rear corner of grid. Defaults to (0,0,0)\n"
"    xaxis (3-tuple of float): (x,y,z) vector for X axis. Defaults to (1,0,0)\n"
"    yaxis (3-tuple of float): (x,y,z) vector for Y axis. Defaults to (0,1,0)\n"
"    zaxis (3-tuple of float): (x,y,z) vector for Z axis. Defaults to (0,0,1)";
static PyObject *py_mol_add_volumetric(PyObject *self, PyObject *args,
                                       PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "name", "size", "data",
                          "origin", "xaxis", "yaxis", "zaxis", NULL};
  float forigin[3] = {0.f, 0.f, 0.f};
  float fxaxis[3] = {1.f, 0.f, 0.f};
  float fyaxis[3] = {0.f, 1.f, 0.f};
  float fzaxis[3] = {0.f, 0.f, 1.f};
  PyObject *seqdata = NULL;
  float *fdata = NULL;
  int totalsize, i;
  PyObject *data;
  int molid = -1;
  VMDApp *app;
  int size[3];
  char *name;

  // Handle deprecated xsize, ysize, zsize arguments
  if (kwargs && PyDict_GetItemString(kwargs, "xsize")
      && PyDict_GetItemString(kwargs, "ysize")
      && PyDict_GetItemString(kwargs, "zsize")) {

    data = Py_BuildValue("(OOO)", PyDict_GetItemString(kwargs, "xsize"),
                         PyDict_GetItemString(kwargs, "ysize"),
                         PyDict_GetItemString(kwargs, "zsize"));

    PyDict_DelItemString(kwargs, "xsize");
    PyDict_DelItemString(kwargs, "ysize");
    PyDict_DelItemString(kwargs, "zsize");

    PyDict_SetItemString(kwargs, "size", data);
    Py_DECREF(data);

    PyErr_Warn(PyExc_DeprecationWarning, "xsize, ysize, zsize keywords are "
               "deprecated, pass size as a tuple instead");
  }

  if (!PyArg_ParseTupleAndKeywords(args, kwargs,
                                   "is(iii)O|O&O&O&O&:molecule.add_volumetric",
                                   (char**) kwlist, &molid, &name, &size[0],
                                   &size[1], &size[2], &data,
                                   &py_array_from_obj, &forigin,
                                   &py_array_from_obj, &fxaxis,
                                   &py_array_from_obj, &fyaxis,
                                   &py_array_from_obj, &fzaxis))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app))
    return NULL;

  // Validate size parameters
  if (size[0] < 1 || size[1] < 1 || size[2] < 1) {
    PyErr_SetString(PyExc_ValueError, "Volumetric sizes must be >= 1");
    goto failure;
  }

  // Handle either list or tuple for data
  if (!(seqdata = PySequence_Fast(data, "data must be a list or tuple")))
    goto failure;

  totalsize = size[0] * size[1] * size[2];
  if (PyList_Size(data) != totalsize) {
    PyErr_Format(PyExc_ValueError, "size of data list '%d' does not match "
                 "expected grid size %d x %d x %d",
                 (int) PySequence_Fast_GET_SIZE(seqdata),
                 size[0], size[1], size[2]);
    goto failure;
  }

  // allocate float array here; pass it to molecule
  fdata = new float[totalsize];
  for (i = 0; i < totalsize; i++) {
    PyObject *elem = PySequence_Fast_GET_ITEM(seqdata, i);

    if (!PyFloat_Check(elem)) {
      PyErr_SetString(PyExc_TypeError, "Volumetric data must be floats");
      goto failure;
    }

    float tmp = PyFloat_AsDouble(elem);

    if (PyErr_Occurred())
      goto failure;

    fdata[i] = tmp;
  }

  if (!app->molecule_add_volumetric(molid, name, forigin, fxaxis, fyaxis,
       fzaxis, size[0], size[1], size[2], fdata)) {
    PyErr_Format(PyExc_ValueError, "Unable to add volumetric data to molid %d",
                 molid);
    goto failure;
  }

  Py_DECREF(seqdata);
  Py_INCREF(Py_None);
  return Py_None;

  failure:
    Py_XDECREF(seqdata);
    if (fdata)
      delete [] fdata;
    return NULL;
}

static const char del_volumetric_doc[] =
"Delete a volumetric dataset from a molecule\n\n"
"Args:\n"
"    molid (int): Molecule ID to remove dataset from\n"
"    dataindex (int): Volumetric dataset index to delete";
static PyObject* py_mol_del_volumetric(PyObject *self, PyObject *args,
                                       PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "dataindex", NULL};
  int molid, index;
  Molecule *mol;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii:molecule.delete_volumetric",
                                  (char**) kwlist,  &molid, &index))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app))
    return NULL;

  mol = app->moleculeList->mol_from_id(molid);
  if (index < 0 || index >= mol->num_volume_data()) {
    PyErr_Format(PyExc_ValueError, "Invalid volumetric index '%d'", index);
    return NULL;
  }

  mol->remove_volume_data(index);
  Py_INCREF(Py_None);
  return Py_None;
}

// Deprecated name for del_volumetric
static PyObject *py_mol_drm_volumetric(PyObject *self, PyObject *args,
                                       PyObject *kwargs)
{
  PyErr_Warn(PyExc_DeprecationWarning, "the 'remove_volumetric' method has "
             "been renamed to 'delete_volumetric'");
  return py_mol_del_volumetric(self, args, kwargs);
}

static const char get_volumetric_doc[] =
"Obtain a volumetric dataset loaded into a molecule\n\n"
"Args:\n"
"    molid (int): Molecule ID to query\n"
"    dataindex (int): Volumetric dataset index to obtain\n"
"Returns:\n"
"    (4-list): Dataset, consisting of 4 lists:\n"
"        1.) A 1-D list of the values at each point\n"
"        2.) 3-tuple of int describing the x, y, z lengths.\n"
"        3.) 3-tuple of float describing the position of the origin.\n"
"        4.) 9-tuple of float describing the deltas for the x, y, and z axes)."
"\n\n"
"Example usage for export to numpy:\n"
"    >>> data, shape, origin, delta = molecule.get_volumetric(0,0)\n"
"    >>> data = np.array(data)\n"
"    >>> shape = np.array(shape)\n"
"    >>> data = data.reshape(shape, order='F')\n"
"    >>> delta = np.array(delta).reshape(3,3)\n"
"    >>> delta /= shape-1";
static PyObject* py_mol_get_volumetric(PyObject *self, PyObject *args,
                                       PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "dataindex", NULL};
  PyObject *result = NULL, *deltas = NULL, *origin = NULL;
  PyObject *size = NULL, *data = NULL;
  const VolumetricData *synthvol;
  int molid, index, dsize, i;
  Molecule *mol;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii:molecule.get_volumetric",
                                   (char**) kwlist, &molid, &index))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app))
    return NULL;

  mol = app->moleculeList->mol_from_id(molid);
  if (index < 0 || index >= mol->num_volume_data()) {
    PyErr_Format(PyExc_ValueError, "Invalid volumetric index '%d'", index);
    return NULL;
  }

  // Calculate size for data array
  synthvol = mol->get_volume_data(index);
  dsize = synthvol->xsize * synthvol->ysize * synthvol->zsize;

  // Create data structures to be returned
  result = PyList_New(4);
  deltas = PyTuple_New(9);
  origin = PyTuple_New(3);
  size = PyTuple_New(3);
  data = PyList_New(dsize);

  if (!result || !deltas || !origin || !size || !data) {
    PyErr_Format(PyExc_MemoryError, "Cannot allocate volumetric data arrays "
                 "for molid %d volindex %d", molid, index);
    goto failure;
  }

  // Build data array list
  for (i = 0; i < dsize; i++) {
    if (PyList_SetItem(data, i, PyFloat_FromDouble(synthvol->data[i]))) {
      PyErr_Format(PyExc_RuntimeError, "Cannot build data array for molid %d "
                   "volindex %d", molid, index);
      goto failure;
    }
  }

  // Build box size list
  if (PyTuple_SetItem(size, 0, as_pyint(synthvol->xsize))
      || PyTuple_SetItem(size, 1, as_pyint(synthvol->ysize))
      || PyTuple_SetItem(size, 2, as_pyint(synthvol->zsize))) {
    PyErr_Format(PyExc_RuntimeError, "Cannot build box size list for molid %d "
                 "volindex %d", molid, index);
    goto failure;
  }

  // Build axis vector list
  for (i = 0; i < 3; i++) {
    if (PyTuple_SetItem(deltas, i, PyFloat_FromDouble(synthvol->xaxis[i]))
     || PyTuple_SetItem(deltas, 3+i, PyFloat_FromDouble(synthvol->yaxis[i]))
     || PyTuple_SetItem(deltas, 6+i, PyFloat_FromDouble(synthvol->zaxis[i]))
     || PyTuple_SetItem(origin, i, PyFloat_FromDouble(synthvol->origin[i]))) {
      PyErr_Format(PyExc_RuntimeError, "Cannot build axis vector list for molid"
                   " %d volindex %d", molid, index);
      goto failure;
    }
  }

  // Finally, build list of lists that's returned
  PyList_SET_ITEM(result, 0, data);
  PyList_SET_ITEM(result, 1, size);
  PyList_SET_ITEM(result, 2, origin);
  PyList_SET_ITEM(result, 3, deltas);

  if (PyErr_Occurred()) {
    PyErr_Format(PyExc_RuntimeError, "Problem building volumetric information "
                 "list for molid %d volindex %d", molid, index);
    goto failure;
  }

  return result;

failure:
  Py_XDECREF(data);
  Py_XDECREF(deltas);
  Py_XDECREF(origin);
  Py_XDECREF(size);
  Py_XDECREF(result);
  return NULL;
}

static const char num_volumetric_doc[] =
"Get the number of volumetric datasets loaded into a molecule\n\n"
"Args:\n"
"    molid (int): Molecule ID to query\n"
"Returns:\n"
"    (int): Number of volumetric datasets loaded";
static PyObject* py_mol_num_volumetric(PyObject *self, PyObject *args,
                                       PyObject *kwargs)
{
  const char *kwlist[] = {"molid", NULL};
  Molecule *mol;
  VMDApp *app;
  int molid;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i:molecule.num_volumetric",
                                   (char**) kwlist, &molid))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app))
    return NULL;

  mol = app->moleculeList->mol_from_id(molid);
  return as_pyint(mol->num_volume_data());
}


static const char filenames_doc[] =
"Get list of all files loaded into a molecule\n\n"
"Args:\n"
"    molid (int): Molecule ID to query\n"
"Returns:\n"
"    (list of str): Paths to files loaded in molecule";
static PyObject* py_get_filenames(PyObject *self, PyObject *args,
                                  PyObject *kwargs)
{
  const char *kwlist[] = {"molid", NULL};
  PyObject *result;
  Molecule *mol;
  VMDApp *app;
  int molid;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i:molecule.get_filenames",
                                   (char**) kwlist, &molid))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app))
    return NULL;

  mol = app->moleculeList->mol_from_id(molid);

  result = PyList_New(mol->num_files());
  for (int i=0; i<mol->num_files(); i++) {
    PyList_SET_ITEM(result, i, as_pystring(mol->get_file(i)));

    if (PyErr_Occurred()) {
      PyErr_Format(PyExc_RuntimeError, "Problem listing loaded files in molid "
                   "'%d'", molid);
      Py_DECREF(result);
      return NULL;
    }
  }
  return result;
}

static const char filetypes_doc[] =
"Get file types loaded into a molecule\n\n"
"Args:\n"
"    molid (int): Molecule ID to query\n"
"Returns:\n"
"    (list of str): File types loaded into molecule";
static PyObject* py_get_filetypes(PyObject *self, PyObject *args,
                                  PyObject *kwargs)
{
  const char *kwlist[] = {"molid", NULL};
  PyObject *result;
  Molecule *mol;
  VMDApp *app;
  int molid;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i:molecule.get_filetypes",
                                   (char**) kwlist, &molid))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app))
    return NULL;

  mol = app->moleculeList->mol_from_id(molid);
  result = PyList_New(mol->num_files());

  for (int i=0; i<mol->num_files(); i++) {
    PyList_SET_ITEM(result, i, as_pystring(mol->get_type(i)));

    // Delete built list on error
    if (PyErr_Occurred()) {
      PyErr_Format(PyExc_RuntimeError, "Problem listing filetypes in molid "
                   "'%d'", molid);
      Py_DECREF(result);
      return NULL;
    }

  }
  return result;
}

static const char databases_doc[] =
"Get databases associated with files loaded into a molecule\n\n"
"Args:\n"
"    molid (int): Molecule ID to query\n"
"Returns:\n"
"    (list of str): Databases loaded into molecule. If no database is\n"
"        associated with a given file loaded into the molecule, an empty\n"
"        list will be returned";
static PyObject* py_get_databases(PyObject *self, PyObject *args,
                                  PyObject *kwargs)
{
  const char *kwlist[] = {"molid", NULL};
  PyObject *result;
  Molecule *mol;
  VMDApp *app;
  int molid;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i:molecule.get_databases",
                                   (char**) kwlist, &molid))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app))
    return NULL;

  mol = app->moleculeList->mol_from_id(molid);
  result = PyList_New(mol->num_files());

  for (int i=0; i<mol->num_files(); i++) {
    PyList_SET_ITEM(result, i, as_pystring(mol->get_database(i)));

    // Delete built list on error
    if (PyErr_Occurred()) {
      PyErr_Format(PyExc_RuntimeError, "Problem listing databases in molid "
                   "'%d'", molid);
      Py_DECREF(result);
      return NULL;
    }
  }
  return result;
}
static const char accessions_doc[] =
"Get database accession codes associated with files loaded into a molecule\n\n"
"Args:\n"
"    molid (int): Molecule ID to query\n"
"Returns:\n"
"    (list of str): Accession codes loaded into molecule. If no accession is\n"
"        associated with a given file loaded into the molecule, an empty\n"
"        list will be returned.";
static PyObject* py_get_accessions(PyObject *self, PyObject *args,
                                   PyObject *kwargs)
{
  const char *kwlist[] = {"molid", NULL};
  PyObject *result;
  Molecule *mol;
  VMDApp *app;
  int molid;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i:molecule.get_accessions",
                                   (char**) kwlist, &molid))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app))
    return NULL;

  mol = app->moleculeList->mol_from_id(molid);
  result = PyList_New(mol->num_files());

  for (int i=0; i<mol->num_files(); i++) {
    PyList_SET_ITEM(result, i, as_pystring(mol->get_accession(i)));

    // Delete built list on error
    if (PyErr_Occurred()) {
      PyErr_Format(PyExc_RuntimeError, "Problem listing databases in molid "
                   "'%d'", molid);
      Py_DECREF(result);
      return NULL;
    }
  }
  return result;
}

static const char remarks_doc[] =
"Get remarks associated with files loaded into a molecule\n\n"
"Args:\n"
"    molid (int): Molecule ID to query\n"
"Returns:\n"
"    (list of str): Remarks present in the molecule, one string per loaded\n"
"        file with remark information. Multiple remarks in the same file will\n"
"        be separated by newline (\\n) characters. If no remarks are\n"
"        associated with a given file loaded into the molecule, an empty\n"
"        string will be present in the list. If no remarks are associated\n"
"        with the molecule, an empty list will be returned.";
static PyObject* py_get_remarks(PyObject *self, PyObject *args,
                                PyObject *kwargs)
{
  const char *kwlist[] = {"molid", NULL};
  PyObject *result;
  Molecule *mol;
  VMDApp *app;
  int molid;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i:molecule.get_remarks",
                                   (char**) kwlist, &molid))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app))
    return NULL;

  mol = app->moleculeList->mol_from_id(molid);
  result = PyList_New(mol->num_files());

  for (int i=0; i<mol->num_files(); i++) {
    PyList_SET_ITEM(result, i, as_pystring(mol->get_remarks(i)));

    // Delete built list on error
    if (PyErr_Occurred()) {
      PyErr_Format(PyExc_RuntimeError, "Problem listing remarks in molid '%d'",
                   molid);
      Py_DECREF(result);
      return NULL;
    }
  }
  return result;
}


static const char get_periodic_doc[] =
"Get the periodic cell layout\n\n"
"Args:\n"
"    molid (int): Molecule ID to query. Defaults to -1 (top molecule)\n"
"    frame (int): Frame to get box information from. Defaults to -1 (current)\n"
"Returns:\n"
"    (dict str->float): Periodic cell layout with values for keys 'a', 'b',\n"
"        and 'c' representing the lengths of the unit cell along the first,\n"
"        second, and third unit cell vectors, respectively. Values for keys\n"
"        'alpha', 'beta', and 'gamma' give the angle between sides B and C,\n"
"        A and C, and A and B, respectively.";
static PyObject* py_get_periodic(PyObject *self, PyObject *args,
                                 PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "frame", NULL};
  int molid=-1, frame=-1;
  PyObject *dict;
  Timestep *ts;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|ii:molecule.get_periodic",
                                   (char**) kwlist, &molid, &frame))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (molid == -1)
    molid = app->molecule_top();
  if (!valid_molid(molid, app))
    return NULL;

  // Get frame number
  ts = parse_timestep(app, molid, frame);
  if (!ts)
    return NULL;

  dict = PyDict_New();
  PyDict_SetItemString(dict, "a", PyFloat_FromDouble(ts->a_length));
  PyDict_SetItemString(dict, "b", PyFloat_FromDouble(ts->b_length));
  PyDict_SetItemString(dict, "c", PyFloat_FromDouble(ts->c_length));
  PyDict_SetItemString(dict, "alpha", PyFloat_FromDouble(ts->alpha));
  PyDict_SetItemString(dict, "beta", PyFloat_FromDouble(ts->beta));
  PyDict_SetItemString(dict, "gamma", PyFloat_FromDouble(ts->gamma));

  // Clean up dictionary if something went wrong
  if (PyErr_Occurred()) {
    PyErr_SetString(PyExc_RuntimeError, "Problem getting box info");
    Py_DECREF(dict);
    return NULL;
  }

  return dict;
}

static const char set_periodic_doc[] =
"Set the periodic cell layout for a particular frame. Any number of box\n"
"attributes may be changed at a time\n\n"
"Args:\n"
"    molid (int): Molecule ID. Defaults to -1 (top molecule)\n"
"    frame (int): Frame to change box info for. Defaults to -1 (current)\n"
"    a (float): Length of unit cell along first cell vector. Optional\n"
"    b (float): Length of unit cell along second cell vector. Optional\n"
"    c (float): Length of unit cell along third cell vector. Optional\n"
"    alpha (float): Angle between cell sides B and C. Optional\n"
"    beta (float): Angle between cell sides A and C. Optional\n"
"    gamma (float): Angle between cell sides A and B. Optional";
static PyObject *py_set_periodic(PyObject *self, PyObject *args,
                                 PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "frame", "a", "b", "c", "alpha", "beta",
                          "gamma", NULL};
  float a=-1, b=-1, c=-1, alpha=-1, beta=-1, gamma=-1;
  int molid = -1, frame = -1;
  Timestep *ts;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs,
                                   "|iiffffff:molecule.set_periodic",
                                   (char**)  kwlist, &molid, &frame, &a, &b,
                                   &c, &alpha, &beta, &gamma))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (molid == -1)
    molid = app->molecule_top();
  if (!valid_molid(molid, app))
    return NULL;

  ts = parse_timestep(app, molid, frame);
  if (!ts)
    return NULL;

  if (a >= 0)
    ts->a_length = a;
  if (b >= 0)
    ts->b_length = b;
  if (c >= 0)
    ts->c_length = c;
  if (alpha > 0)
    ts->alpha = alpha;
  if (beta > 0)
    ts->beta = beta;
  if (gamma > 0)
    ts->gamma = gamma;

  Py_INCREF(Py_None);
  return Py_None;
}

static const char get_visible_doc[] =
"Get if a molecule is visible\n\n"
"Args:\n"
"    molid (int): Molecule ID to query. Defaults to -1 (top molecule)\n"
"Returns:\n"
"    (bool): If molecule is visible";
static PyObject* py_get_visible(PyObject *self, PyObject *args,PyObject *kwargs)
{
  const char *kwlist[] = {"molid", NULL};
  int molid = -1;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|i:molecule.get_visible",
                                   (char**) kwlist, &molid))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (molid<0)
    molid = app->molecule_top();
  if (!valid_molid(molid, app))
    return NULL;

  return PyBool_FromLong(app->molecule_is_displayed(molid));
}

static const char set_visible_doc[] =
"Set the visiblity of a molecule\n\n"
"Args:\n"
"    molid (int): Molecule ID\n"
"    visible (bool): If molecule should be visible";
static PyObject* py_set_visible(PyObject *self, PyObject *args,
                                PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "visible", NULL};
  PyObject *value;
  int molid = -1;
  VMDApp *app;
  int visible;

  // Handle legacy "state" keyword -> visible
  if (kwargs && (value = PyDict_GetItemString(kwargs, "state"))) {
    PyDict_SetItemString(kwargs, "visible", value);
    PyDict_DelItemString(kwargs, "state");
    PyErr_Warn(PyExc_DeprecationWarning, "'state' keywords is 'visible' now");
  }

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i|O&:molecule.set_visible",
                                   (char**) kwlist, &molid, convert_bool,
                                   &visible))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app))
    return NULL;

  app->molecule_display(molid, visible);
  Py_INCREF(Py_None);
  return Py_None;
}

static const char time_doc[] =
"Get the physical time value for a frame, if set\n\n"
"Args:\n"
"    molid (int): Molecule ID to query\n"
"    frame (int): Frame to query. Defaults to -1 (current)\n"
"Returns:\n"
"    (float): Physical time value for frame, or 0.0f if unset";
static PyObject* py_get_physical_time(PyObject *self, PyObject *args,
                                      PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "frame", NULL};
  int molid = -1, frame = -1;
  Timestep *ts;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs,
                                   "i|i:molecule.get_physical_time",
                                   (char**) kwlist, &molid, &frame))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app))
    return NULL;

  ts = parse_timestep(app, molid, frame);
  if (!ts)
    return NULL;

  return PyFloat_FromDouble(ts->physical_time);
}

static const char settime_doc[] =
"Set the physical time value for a frame\n\n"
"Args:\n"
"    molid (int): Molecule ID to set time for.\n"
"    time (float): Time value\n"
"    frame (int): Frame to set time for. Defaults to -1 (current)\n";
static PyObject* py_set_physical_time(PyObject *self, PyObject *args,
                                      PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "time", "frame", NULL};
  int molid = -1, frame = -1;
  double value;
  VMDApp *app;
  Timestep *ts;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs,
                                   "id|i:molecule.set_physical_time",
                                   (char**)  kwlist, &molid, &value, &frame))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app))
    return NULL;

  ts = parse_timestep(app, molid, frame);
  if (!ts)
    return NULL;

  ts->physical_time = value;

  Py_INCREF(Py_None);
  return Py_None;
}

static PyMethodDef methods[] = {
  {"num", (PyCFunction)py_mol_num, METH_NOARGS, mol_num_doc },
  {"listall", (PyCFunction)py_mol_listall, METH_VARARGS | METH_KEYWORDS ,mol_listall_doc},
  {"exists", (PyCFunction)py_mol_exists, METH_VARARGS | METH_KEYWORDS, mol_exists_doc},
  {"name", (PyCFunction)py_mol_name, METH_VARARGS | METH_KEYWORDS, mol_name_doc},
  {"numatoms", (PyCFunction)py_mol_numatoms, METH_VARARGS | METH_KEYWORDS, mol_numatoms_doc},
  {"new", (PyCFunction)py_mol_new, METH_VARARGS | METH_KEYWORDS, mol_new_doc},
  {"load", (PyCFunction)py_mol_load, METH_VARARGS | METH_KEYWORDS, mol_load_doc},
  {"cancel", (PyCFunction)py_mol_cancel, METH_VARARGS | METH_KEYWORDS, mol_cancel_doc},
  {"delete", (PyCFunction)py_mol_delete, METH_VARARGS | METH_KEYWORDS, mol_delete_doc},
  {"get_top", (PyCFunction)py_get_top, METH_NOARGS, get_top_doc},
  {"set_top", (PyCFunction)py_set_top, METH_VARARGS | METH_KEYWORDS, set_top_doc},
  {"read", (PyCFunction)py_mol_read, METH_VARARGS | METH_KEYWORDS, mol_read_doc},
  {"write", (PyCFunction)py_mol_write, METH_VARARGS | METH_KEYWORDS, mol_write_doc},
  {"numframes", (PyCFunction)py_numframes, METH_VARARGS | METH_KEYWORDS, numframes_doc},
  {"get_frame", (PyCFunction)py_get_frame, METH_VARARGS | METH_KEYWORDS, get_frame_doc},
  {"set_frame", (PyCFunction)py_set_frame, METH_VARARGS | METH_KEYWORDS, set_frame_doc},
  {"delframe", (PyCFunction)py_delframe, METH_VARARGS | METH_KEYWORDS, delframe_doc},
  {"dupframe", (PyCFunction)py_dupframe, METH_VARARGS | METH_KEYWORDS, dupframe_doc},
  {"ssrecalc", (PyCFunction)py_mol_ssrecalc, METH_VARARGS | METH_KEYWORDS, mol_ssrecalc_doc},
  {"rename", (PyCFunction)py_mol_rename, METH_VARARGS | METH_KEYWORDS, mol_rename_doc},
  {"add_volumetric", (PyCFunction)py_mol_add_volumetric,
                      METH_VARARGS | METH_KEYWORDS, add_volumetric_doc},
  {"delete_volumetric", (PyCFunction)py_mol_del_volumetric,
                        METH_VARARGS | METH_KEYWORDS, del_volumetric_doc},
  {"get_volumetric", (PyCFunction)py_mol_get_volumetric,
                     METH_VARARGS | METH_KEYWORDS, get_volumetric_doc},
  {"num_volumetric", (PyCFunction)py_mol_num_volumetric,
                     METH_VARARGS | METH_KEYWORDS, num_volumetric_doc},
  {"get_filenames", (PyCFunction)py_get_filenames, METH_VARARGS | METH_KEYWORDS, filenames_doc},
  {"get_filetypes", (PyCFunction)py_get_filetypes, METH_VARARGS | METH_KEYWORDS, filetypes_doc},
  {"get_databases", (PyCFunction)py_get_databases, METH_VARARGS | METH_KEYWORDS, databases_doc},
  {"get_accessions", (PyCFunction)py_get_accessions, METH_VARARGS | METH_KEYWORDS, accessions_doc},
  {"get_remarks", (PyCFunction)py_get_remarks, METH_VARARGS | METH_KEYWORDS, remarks_doc},
  {"get_periodic", (PyCFunction)py_get_periodic, METH_VARARGS | METH_KEYWORDS, get_periodic_doc},
  {"set_periodic", (PyCFunction)py_set_periodic, METH_VARARGS | METH_KEYWORDS, set_periodic_doc},
  {"get_visible", (PyCFunction)py_get_visible, METH_VARARGS | METH_KEYWORDS, get_visible_doc},
  {"set_visible", (PyCFunction)py_set_visible, METH_VARARGS | METH_KEYWORDS, set_visible_doc},
  {"get_physical_time", (PyCFunction)py_get_physical_time, METH_VARARGS | METH_KEYWORDS, time_doc},
  {"set_physical_time", (PyCFunction)py_set_physical_time, METH_VARARGS | METH_KEYWORDS, settime_doc},

  // Deprecated names, but still work for backwards compatibility
  {"remove_volumetric", (PyCFunction)py_mol_drm_volumetric,
                        METH_VARARGS | METH_KEYWORDS, del_volumetric_doc},
  {NULL, NULL}
};

static const char mol_moddoc[] =
"Methods to interact with molecules, including loading topology, trajectory, "
"or volumetric data, query the attributes of a molecule such as the number "
"of frames or the physical time, or set the visibility of a molecule";

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moleculedef = {
  PyModuleDef_HEAD_INIT,
  "molecule",
  mol_moddoc,
  -1,
  methods,
};
#endif

PyObject* initmolecule(void) {
#if PY_MAJOR_VERSION >= 3
  PyObject *module = PyModule_Create(&moleculedef);
#else
  PyObject *module = Py_InitModule3("molecule", methods, mol_moddoc);
#endif
  return module;
}

