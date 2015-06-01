/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2011 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: py_molecule.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.68 $       $Date: 2010/12/16 04:08:57 $
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


static char mol_num_doc[] = 
  "num() -> int\n"
  "RAeturns the number of loaded molecules.";
static PyObject *mol_num(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, (char *)":molecule.num"))
    return NULL;

  return PyInt_FromLong(get_vmdapp()->num_molecules());
}

static char mol_listall_doc[] =
  "listall() -> list\n"
  "Returns list of all valid molid's.";
static PyObject *mol_listall(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, (char *)":molecule.listall"))
    return NULL;

  VMDApp *app = get_vmdapp();
  int num = app->num_molecules();
  PyObject *newlist = PyList_New(num);
  for (int i=0; i<num; i++)
    PyList_SET_ITEM(newlist, i, PyInt_FromLong(app->molecule_id(i)));

  return newlist;
}

static char mol_exists_doc[] = 
  "exists(molid) -> boolean\n"
  "Return True if molid is valid.";
static PyObject *mol_exists(PyObject *self, PyObject *args) {
  int molid;
  if (!PyArg_ParseTuple(args, (char *)"i:molecule.exists", &molid))
    return NULL;
  VMDApp *app = get_vmdapp();
  return PyInt_FromLong(app->molecule_valid_id(molid));
}

static char mol_name_doc[] = 
  "name(molid) -> string\n"
  "Returns name of given molecule";
static PyObject *mol_name(PyObject *self, PyObject *args) {
  int molid;

  if (!PyArg_ParseTuple(args, (char *)"i:molecule.name", &molid))
    return NULL;

  VMDApp *app = get_vmdapp();
  const char *name = app->molecule_name(molid);
  if (!name) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  }
  return PyString_FromString((char *)name);
}

static char mol_numatoms_doc[] = 
  "numatoms (molid) -> int\n"
  "Returns number of atoms in given molecule";
// numatoms(molid)
static PyObject *mol_numatoms(PyObject *self, PyObject *args) {
  int molid;

  if (!PyArg_ParseTuple(args, (char *)"i:molecule.numatoms", &molid))
    return NULL;

  VMDApp *app = get_vmdapp();
  if (!app->molecule_valid_id(molid)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  }
  return PyInt_FromLong(app->molecule_numatoms(molid));
}

static char mol_new_doc[] = 
  "new(name[,natoms]) -> int\n"
  "Creates a new molecule with given name and returns molid. An\n"
  "additional integer argument adds that number of 'empty' atoms to the molecule." ;
static PyObject *mol_new(PyObject *self, PyObject *args) {
  char *name;
  int  natoms=0;
  if (!PyArg_ParseTuple(args, (char *)"s|i:molecule.new", &name, &natoms))
    return NULL;
  VMDApp *app = get_vmdapp();
  int molid = app->molecule_new(name,natoms);
  if (molid < 0) {
    PyErr_SetString(PyExc_ValueError, (char *)"Unable to create molecule.");
    return NULL;
  }
  return PyInt_FromLong(molid);
}

static char mol_load_doc[] = 
  "load(structure, sfname, coor, cfname) -> molid\n"
  "Load new molecule with structure file type 'structure', \n"
  "  structure file name 'sfname', coordinate file type 'coor', and\n"
  "  coordinate file name 'cfname'.  'coor' and 'cfname' are optional.";

static PyObject *mol_load(PyObject *self, PyObject *args, PyObject *keywds) {
 
  char *structure = NULL, *coor = NULL, *sfname = NULL, *cfname = NULL;
 
  static char *kwlist[] = {
    (char *)"structure", (char *)"sfname", (char *)"coor", (char *)"cfname", 
    NULL
  };

  if (!PyArg_ParseTupleAndKeywords(args, keywds, (char *)"ss|ss:molecule.load", kwlist, 
        &structure, &sfname, &coor, &cfname))
    return NULL;

  // must specify structure and a structure file
  if (!structure) {
    PyErr_SetString(PyExc_ValueError, (char *)"No structure type specified");
    return NULL;
  }
  if (!sfname) {
    PyErr_SetString(PyExc_ValueError, (char *)"No structure file specified");
    return NULL;
  }

  // if a coordinates file type was specified, a coordinate file must be given
  // as well, and vice versa.
  if (coor && !cfname) {
    PyErr_SetString(PyExc_ValueError, (char *)"No coordinate file specified");
    return NULL;
  }
  if (cfname && !coor) {
    PyErr_SetString(PyExc_ValueError, (char *)"No coordinate type specified");
    return NULL;
  }

  // Get the VMDApp instance from the VMDApp module
  VMDApp *app = get_vmdapp();

  // Special-case graphics molecules to load as "blank" molecules
  if (!strcmp(structure, "graphics")) {
    return PyInt_FromLong(app->molecule_new(sfname,0));
  }
  FileSpec spec;
  spec.waitfor=-1; // load all frames at once by default
  int molid = app->molecule_load(-1, sfname, structure, &spec);
  if (molid < 0) {
    PyErr_SetString(PyExc_ValueError, (char *)"Unable to load structure file");
    return NULL;
  }
  if (cfname) {
    app->molecule_load(molid, cfname, coor, &spec);
  } 
  return PyInt_FromLong(molid); 
}

static char mol_cancel_doc[] =
  "cancel(molid) -> None\n"
  "Cancel background loading of files for given molid.";
static PyObject *mol_cancel(PyObject *self, PyObject *args) {
  // get one integer argument
  int molid;
  if (!PyArg_ParseTuple(args, (char *)"i:molecule.cancel", &molid))
    return NULL;

  VMDApp *app = get_vmdapp();
  if (!app->molecule_valid_id(molid)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  } 
  app->molecule_cancel_io(molid);

  Py_INCREF(Py_None);
  return Py_None;
}

static char mol_delete_doc[] =
  "delete(molid) -> None\n"
  "Delete molecule with given molid.";
static PyObject *mol_delete(PyObject *self, PyObject *args) {
  int molid;
  if (!PyArg_ParseTuple(args, (char *)"i:molecule.delete", &molid))
    return NULL;

  VMDApp *app = get_vmdapp();
  if (!app->molecule_valid_id(molid)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  } 
  app->molecule_delete(molid);
  
  Py_INCREF(Py_None);
  return Py_None;
}

static char get_top_doc[] =
  "get_top(molid) -> molid\n"
  "Return molid of top molecule.";
static PyObject *get_top(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, (char *)":molecule.get_top"))
    return NULL;

  return PyInt_FromLong(get_vmdapp()->molecule_top());
}

static char set_top_doc[] =
  "set_top(molid) -> None\n"
  "Make the molecule top.";
static PyObject *set_top(PyObject *self, PyObject *args) {
  int molid;
  if (!PyArg_ParseTuple(args, (char *)"i:molecule.set_top", &molid))
    return NULL;

  VMDApp *app = get_vmdapp();
  if (!app->molecule_valid_id(molid)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  } 
  app->molecule_make_top(molid);
  
  Py_INCREF(Py_None);
  return Py_None;
}


static PyObject *readorwrite(PyObject *self, PyObject *args, PyObject *keywds,
                            int do_read) {

  int molid = -1;
  char *filename = NULL, *type = NULL;
  int beg=0, end=-1, stride=1, waitfor = do_read ? 1 : -1;
  PyObject *volsets = NULL, *selobj = NULL;
  int *on = NULL;

  static char *kwlist[] = {
    (char *)"molid", (char *)"filetype", (char *)"filename", (char *)"beg", 
    (char *)"end", (char *)"skip", (char *)"waitfor", (char *)"volsets", 
    (char *)"selection", NULL
  };

  if (!PyArg_ParseTupleAndKeywords(args, keywds, (char *)"iss|iiiiO!O:read/write", kwlist,
    &molid, &type, &filename, &beg, &end, &stride, &waitfor, &PyList_Type, 
    &volsets, &selobj))
    return NULL;

  VMDApp *app = get_vmdapp();
 
  int numframes = 0;
  if (do_read) { 
    FileSpec spec;
    spec.first = beg;
    spec.last = end;
    spec.stride = stride;
    spec.waitfor = waitfor;
    if (volsets) {
      spec.nvolsets = PyList_Size(volsets);
      spec.setids = new int[spec.nvolsets];
      for (int i=0; i<spec.nvolsets; i++) 
        spec.setids[i] = PyInt_AsLong(PyList_GET_ITEM(volsets, i));
      if (PyErr_Occurred()) return NULL;
    } else {
      // Have a default of {0} for setids so that it isn't always necessary
      // to specify the set id's.  This should be ignored if the file type 
      // can't read volumetric datasets
      spec.nvolsets = 1;
      spec.setids = new int[1];
      spec.setids[0] = 0;
    }
    return PyInt_FromLong(app->molecule_load(molid, filename, type, &spec));
  }
  if (!app->molecule_valid_id(molid)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  }
  if (selobj && selobj != Py_None) {
    AtomSel *sel = atomsel_AsAtomSel( selobj );
    if (!sel) return NULL;
    if (sel->molid() != molid) {
      PyErr_SetString( PyExc_ValueError, 
          "atomsel must reference same molecule as coordinates" );
      return NULL;
    }
    on = sel->on;
  }
  FileSpec spec;
  spec.first = beg;
  spec.last = end;
  spec.stride = stride;
  spec.waitfor = waitfor;
  spec.selection = on;
  numframes = app->molecule_savetrajectory(molid, filename, type, &spec);
  if (numframes < 0) {
    PyErr_SetString(PyExc_ValueError, (char *)"Unable to save file");
    return NULL;
  }
  return PyInt_FromLong(numframes);
}

static char mol_read_doc[] =
  "read(molid, filetype, filename, beg=0, end=-1, skip=1, waitfor=1)\n"
  "Read data from file 'filename' of type 'filetype' into molecule with\n"
  "id 'molid'.  molid=-1 creates new molecule.  Return number of frames\n"
  "read (remaining frames will be loaded in the background.\n"
  "  First frame = beg; default 0\n"
  "  Last frame = end; default all frames\n"
  "  Frame stride = skip; default load all frames\n"
  "  Frames to load before returning = waitfor; default 1\n";
static PyObject *mol_read(PyObject *self, PyObject *args, PyObject *keywds) {
  return readorwrite(self, args, keywds, 1);
}

static char mol_write_doc[] =
  "write(molid, filetype, filename, beg=0, end=-1, skip=1, waitfor=-1,\n"
  "      selection=None)\n"
  "Write data into file 'filename' of type 'filetype' from molecule with\n"
  "id 'molid'.  Return number of frames written (remaining frames will be\n"
  "written in the background.\n"
  "  First frame = beg; default 0\n"
  "  Last frame = end; default all frames\n"
  "  Frame stride = skip; default load all frames\n"
  "  Frames to load before returning = waitfor; default 1\n"
  "  Specify selection of atoms to write with 'selection' keyword; pass\n"
  "    tuple of atom indexes as argument.";

static PyObject *mol_write(PyObject *self, PyObject *args, PyObject *keywds) {
  return readorwrite(self, args, keywds, 0);
}

static char numframes_doc[] =
  "numframes(molid) -> number of frames in molecule with id 'molid'.";
static PyObject *numframes(PyObject *self, PyObject *args) {
  int molid;

  if (!PyArg_ParseTuple(args, (char *)"i:molecule.numframes", &molid))
    return NULL;

  VMDApp *app = get_vmdapp();
  if (!app->molecule_valid_id(molid)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  }
  return PyInt_FromLong(app->molecule_numframes(molid));
}

static char get_frame_doc[] =
  "get_frame(molid) -> current frame of molecule with id 'molid'.";
static PyObject *get_frame(PyObject *self, PyObject *args) {
  int molid;

  if (!PyArg_ParseTuple(args, (char *)"i:molecule.get_frame", &molid))
    return NULL;

  VMDApp *app = get_vmdapp();
  if (!app->molecule_valid_id(molid)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  }
  return PyInt_FromLong(app->molecule_frame(molid));
}

static char set_frame_doc[] =
  "set_frame(molid, frame)\n"
  "Change frame of molecule with id 'molid' to 'frame'.";
// set_frame(molid, frame)
static PyObject *set_frame(PyObject *self, PyObject *args) {
  int molid, frame;

  if (!PyArg_ParseTuple(args, (char *)"ii:molecule.set_frame", &molid, &frame))
    return NULL;

  VMDApp *app = get_vmdapp();
  Molecule *mol = app->moleculeList->mol_from_id(molid);
  if (!mol) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  }
  mol->override_current_frame(frame);
  mol->change_ts();
  Py_INCREF(Py_None);
  return Py_None;
}

static char delframe_doc[] =
  "delframe(molid, beg=0, end=-1, skip=0)\n"
  "Delete frames in the range [beg, end] inclusive, keeping every\n"
  "  'skip' frame between beg and end.";
static PyObject *delframe(PyObject *self, PyObject *args, PyObject *keywds) {

  int molid = 0, beg=0, end=-1, skip=0;

  static char *kwlist[] = {
    (char *)"molid", (char *)"beg", (char *)"end", (char *)"skip", NULL
  };

  if (!PyArg_ParseTupleAndKeywords(args, keywds, (char *)"i|iii:molecule.delframe", kwlist,
    &molid, &beg, &end, &skip))
    return NULL;

  VMDApp *app = get_vmdapp();
  if (!app->molecule_valid_id(molid)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molid");
    return NULL;
  }

  if (!app->molecule_deleteframes(molid, beg, end, skip)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Unable to delete frames");
    return NULL;
  }
 
  Py_INCREF(Py_None);
  return Py_None;
}

static char dupframe_doc[] =
  "dupframe(molid, frame) -> Duplicate given frame.\n"
  "If frame is -1, duplicate the current frame; this also creates\n"
  "a new frame with coordinates all zero if there are no frames in\n"
  "the molecule.";
static PyObject *dupframe(PyObject *self, PyObject *args) {

  int molid, frame;
  if (!PyArg_ParseTuple(args, (char *)"ii:molecule.dupframe", &molid, &frame))
    return NULL;

  VMDApp *app = get_vmdapp();
  if (!app->molecule_valid_id(molid)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molid");
    return NULL;
  }

  if (!app->molecule_dupframe(molid, frame)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Unable to duplicate frame");
    return NULL;
  }
  Py_INCREF(Py_None);
  return Py_None;
}

static char mol_ssrecalc_doc[] =
  "ssrecalc(molid) -> recalculate secondary structure for molecule.\n"
  "raises RuntimeError if structure could not be calculated, usually\n"
  "due to the Stride program not being available.";
static PyObject *mol_ssrecalc(PyObject *self, PyObject *args) {
  int molid;
  if (!PyArg_ParseTuple(args, (char *)"i:molecule.ssrecalc", &molid))
    return NULL;

  VMDApp *app = get_vmdapp();
  if (!app->molecule_valid_id(molid)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  } 
  int rc;
  Py_BEGIN_ALLOW_THREADS
  rc = app->molecule_ssrecalc(molid);
  Py_END_ALLOW_THREADS
  if (!rc) {
    PyErr_SetString(PyExc_RuntimeError, 
        "Secondary structure could not be calculated");
    return NULL;
  }
  Py_INCREF(Py_None);
  return Py_None;
}

static char mol_rename_doc[] =
  "rename(molid, newname) -> rename molecule to 'newname'.";
static PyObject *mol_rename(PyObject *self, PyObject *args) {
  int molid;
  char *newname;
  if (!PyArg_ParseTuple(args, (char *)"is:molecule.rename", &molid, &newname))
    return NULL;

  VMDApp *app = get_vmdapp();
  if (!app->molecule_valid_id(molid)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  } 
  if (!app->molecule_rename(molid, newname)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Unable to rename molecule.");
    return NULL;
  }
  Py_INCREF(Py_None);
  return Py_None;
}
   
static char add_volumetric_doc[] = 
  "add_volumetric(molid, name, origin, xaxis, yaxis, zaxis,\n"
  "               xsize, ysize, zsize, data)\n"
  "  Add a new volumetric data set to a molecule.  origin, xaxis, yaxis,\n"
  "  and zaxis are 3-vectors represented as lists.  xsize, ysize, zsize\n"
  "  give the size of the volumetric grid data.  data must be a list of\n"
  "  floats of size xsize * ysize * zsize.  origin represents the point\n"
  "  at the lower left rear corner of the grid.";
static PyObject *mol_add_volumetric(PyObject *self, PyObject *args, PyObject *keywds) {

  int molid = -1;
  int xsize = -1, ysize = -1, zsize = -1;
  int size;
  char *name;
  PyObject *data = NULL, *origin = NULL, *xaxis = NULL, *yaxis = NULL, 
           *zaxis = NULL;
  float forigin[3], fxaxis[3], fyaxis[3], fzaxis[3];

  static char *kwlist[] = {
    (char *)"molid", (char *)"name", (char *)"origin",  (char *)"xaxis",
    (char *)"yaxis",  (char *)"zaxis",  (char *)"xsize", (char *)"ysize", 
    (char *)"zsize", (char *)"data", NULL
  };

  if (!PyArg_ParseTupleAndKeywords(args, keywds, 
       (char *)"isOOOOiiiO:molecule.add_volumetric", kwlist,
       &molid, &name, &origin, &xaxis, &yaxis, &zaxis, &xsize, 
       &ysize, &zsize, &data))
    return NULL;

  VMDApp *app = get_vmdapp();
  if (!app->molecule_valid_id(molid)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  }
  if (xsize < 1 || ysize < 1 || zsize < 1) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid size parameters");
    return NULL;
  }
  size = xsize * ysize * zsize;
  if (!PyList_Check(data)) {
    PyErr_SetString(PyExc_ValueError, (char *)"data must be a list");
    return NULL;
  }
  if (PyList_Size(data) != size) {
    PyErr_SetString(PyExc_ValueError, (char *)"size of list does not match specified sizes");
    return NULL;
  }
  // Check that any optional arguments are valid and set defaults.
  if (origin) {
    if (!py_array_from_obj(origin, forigin)) return NULL;
  } else {
    forigin[0] = forigin[1] = forigin[2] = 0;
  }
  if (xaxis) {
    if (!py_array_from_obj(xaxis, fxaxis)) return NULL;
  } else {
    fxaxis[0] = 1.0; fxaxis[1] = fxaxis[2] = 0;
  }
  if (yaxis) {
    if (!py_array_from_obj(yaxis, fyaxis)) return NULL;
  } else {
    fyaxis[1] = 1.0; fyaxis[0] = fyaxis[2] = 0;
  }
  if (zaxis) {
    if (!py_array_from_obj(zaxis, fzaxis)) return NULL;
  } else {
    fzaxis[2] = 1.0; fzaxis[0] = fzaxis[1] = 0;
  }

  // allocate float array here; pass it to molecule
  float *fdata = new float[size];
  for (int i=0; i<size; i++) {
    PyObject *elem = PyList_GET_ITEM(data, i);
    float tmp = PyFloat_AsDouble(elem);
    if (PyErr_Occurred()) {
      delete [] fdata;
      return NULL;
    }
    fdata[i] = tmp;
  }
  if (!app->molecule_add_volumetric(molid, name, forigin, fxaxis, fyaxis,
       fzaxis, xsize, ysize, zsize, fdata)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Unable to add volumetric data");
    delete [] fdata;
    return NULL;
  }
  Py_INCREF(Py_None);
  return Py_None;
}
   
static char filenames_doc[] = "get_filenames(molid): return list of files loaded in the molecule";
static char filetypes_doc[] = "get_filetypes(molid): returns list of corresponding file types.";
static char databases_doc[] = "get_databases(molid): returns list of databases of origin ";
static char accessions_doc[] = "get_accessions(molid): returns list of database accession codes";
static char remarks_doc[] = "get_remarks(molid): returns list of per-file remarks/comments";
static PyObject *get_filenames(PyObject *self, PyObject *args) {
  int molid;
  if (!PyArg_ParseTuple(args, (char *)"i:molecule.get_filenames", &molid))
    return NULL;

  VMDApp *app = get_vmdapp();
  Molecule *mol = app->moleculeList->mol_from_id(molid);
  if (!mol) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  }
  int num = mol->num_files();
  PyObject *result = PyList_New(num);
  for (int i=0; i<mol->num_files(); i++) {
    PyList_SET_ITEM(result, i, PyString_FromString(mol->get_file(i)));
  }
  return result;
}
static PyObject *get_filetypes(PyObject *self, PyObject *args) {
  int molid;
  if (!PyArg_ParseTuple(args, (char *)"i:molecule.get_filetypes", &molid))
    return NULL;

  VMDApp *app = get_vmdapp();
  Molecule *mol = app->moleculeList->mol_from_id(molid);
  if (!mol) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  }
  int num = mol->num_files();
  PyObject *result = PyList_New(num);
  for (int i=0; i<mol->num_files(); i++) {
    PyList_SET_ITEM(result, i, PyString_FromString(mol->get_type(i)));
  }
  return result;
}
static PyObject *get_databases(PyObject *self, PyObject *args) {
  int molid;
  if (!PyArg_ParseTuple(args, (char *)"i:molecule.get_databases", &molid))
    return NULL;

  VMDApp *app = get_vmdapp();
  Molecule *mol = app->moleculeList->mol_from_id(molid);
  if (!mol) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  }
  int num = mol->num_files();
  PyObject *result = PyList_New(num);
  for (int i=0; i<mol->num_files(); i++) {
    PyList_SET_ITEM(result, i, PyString_FromString(mol->get_database(i)));
  }
  return result;
}
static PyObject *get_accessions(PyObject *self, PyObject *args) {
  int molid;
  if (!PyArg_ParseTuple(args, (char *)"i:molecule.get_accessions", &molid))
    return NULL;

  VMDApp *app = get_vmdapp();
  Molecule *mol = app->moleculeList->mol_from_id(molid);
  if (!mol) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  }
  int num = mol->num_files();
  PyObject *result = PyList_New(num);
  for (int i=0; i<mol->num_files(); i++) {
    PyList_SET_ITEM(result, i, PyString_FromString(mol->get_accession(i)));
  }
  return result;
}
static PyObject *get_remarks(PyObject *self, PyObject *args) {
  int molid;
  if (!PyArg_ParseTuple(args, (char *)"i:molecule.get_remarks", &molid))
    return NULL;

  VMDApp *app = get_vmdapp();
  Molecule *mol = app->moleculeList->mol_from_id(molid);
  if (!mol) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  }
  int num = mol->num_files();
  PyObject *result = PyList_New(num);
  for (int i=0; i<mol->num_files(); i++) {
    PyList_SET_ITEM(result, i, PyString_FromString(mol->get_remarks(i)));
  }
  return result;
}


static char get_periodic_doc[] = 
  "get_periodic(molid=-1, frame=-1)\n"
  "return dict with keys 'a', 'b', 'c', 'alpha', 'beta', and 'gamma'\n"
  "representing the periodic cell layout for a particular frame.\n"
  "a, b, and c are the lengths of the unit cell along the first, second\n"
  "and third unit cell vectors, respectively.  alpha, beta, and gamma\n"
  "give the angle between sides B and C, A and C, and A and B,\n"
  "respectively.";
static PyObject *get_periodic(PyObject *self, PyObject *args, PyObject *kwds) {
  int molid=-1;
  int frame=-1;
  static char *kwlist[] = { (char *)"molid", (char *)"frame", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kwds, (char *)"|ii:molecule.get_periodic", kwlist, &molid, &frame))
    return NULL;

  Timestep *ts = parse_timestep(get_vmdapp(), molid, frame);
  if (!ts) return NULL;
  PyObject *dict = PyDict_New();
  PyDict_SetItemString(dict, (char *)"a", PyFloat_FromDouble(ts->a_length));
  PyDict_SetItemString(dict, (char *)"b", PyFloat_FromDouble(ts->b_length));
  PyDict_SetItemString(dict, (char *)"c", PyFloat_FromDouble(ts->c_length));
  PyDict_SetItemString(dict, (char *)"alpha", PyFloat_FromDouble(ts->alpha));
  PyDict_SetItemString(dict, (char *)"beta", PyFloat_FromDouble(ts->beta));
  PyDict_SetItemString(dict, (char *)"gamma", PyFloat_FromDouble(ts->gamma));
  return dict;
}

static char set_periodic_doc[] =
  "set_periodic(molid=-1, frame=-1, a, b, c, alpha, beta, gamma)\n"
  "Set the periodic cell layout for a particular frame.\n"
  "All keywords except molid are optional.  See get_periodic for\n"
  "descriptions of the keywords.";
static PyObject *set_periodic(PyObject *self, PyObject *args, PyObject *kwds) {
  int molid=-1;
  int frame=-1;
  float a=-1, b=-1, c=-1, alpha=-1, beta=-1, gamma=-1;
  static char *kwlist[] = { (char *)"molid", (char *)"frame", (char *)"a",
    (char *)"b", (char *)"c", (char *)"alpha", (char *)"beta", (char *)"gamma",
    NULL
  };
  if (!PyArg_ParseTupleAndKeywords(args, kwds, 
        (char *)"|iiffffff:molecule.set_periodic", kwlist, 
        &molid, &frame, &a, &b, &c, &alpha, &beta, &gamma))
    return NULL;
  Timestep *ts = parse_timestep(get_vmdapp(), molid, frame);
  if (!ts) return NULL;
  if (a >= 0) ts->a_length = a;
  if (b >= 0) ts->b_length = b;
  if (c >= 0) ts->c_length = c;
  if (alpha > 0) ts->alpha = alpha;
  if (beta > 0) ts->beta = beta;
  if (gamma > 0) ts->gamma = gamma;

  Py_INCREF(Py_None);
  return Py_None;
}

static char get_visible_doc[] =
  "get_visible(molid=-1) -> boolean\n"
  "  Get visibility state for selected molecule, default top molecule\n";
static char set_visible_doc[] =
  "set_visible(molid=-1, state)\n"
  "  Turn selected molecule on or off, default top molecule\n";

static PyObject *get_visible(PyObject *self, PyObject *args, PyObject *kwds) {
  int molid = -1;
  static char *kwlist[] = { "molid", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist, &molid))
    return NULL;
  VMDApp *app = get_vmdapp();
  if (!app) return NULL;
  if (molid<0) molid = app->molecule_top();
  if (!app->molecule_valid_id(molid)) {
    PyErr_SetString(PyExc_ValueError, "Invalid molid");
    return NULL;
  }
  return PyBool_FromLong( app->molecule_is_displayed( molid ) );
}
static PyObject *set_visible(PyObject *self, PyObject *args, PyObject *kwds) {
  int molid = -1;
  PyObject *obj=NULL;

  static char *kwlist[] = { "molid", "state", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iO", kwlist, &molid, &obj))
    return NULL;
  VMDApp *app = get_vmdapp();
  if (!app) return NULL;
  if (molid<0) molid = app->molecule_top();
  if (!app->molecule_valid_id(molid)) {
    PyErr_SetString(PyExc_ValueError, "Invalid molid");
    return NULL;
  }
  app->molecule_display( molid, PyObject_IsTrue(obj));
  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject *get_physical_time(PyObject *self, PyObject *args, PyObject *kwds) {
  int molid = -1;
  int frame = -1;
  static char *kwlist[] = { "molid", "frame", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ii", kwlist, &molid, &frame))
    return NULL;
  Timestep *ts = parse_timestep(get_vmdapp(), molid, frame);
  if (!ts) return NULL;
  return PyFloat_FromDouble(ts->physical_time);
}

static PyObject *set_physical_time(PyObject *self, PyObject *args, PyObject *kwds) {
  int molid = -1;
  int frame = -1;
  double value;
  static char *kwlist[] = { "value", "molid", "frame", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "d|ii", kwlist, &value, &molid, &frame))
    return NULL;
  Timestep *ts = parse_timestep(get_vmdapp(), molid, frame);
  if (!ts) return NULL;
  ts->physical_time = value;

  Py_INCREF(Py_None);
  return Py_None;
}

static PyMethodDef MolMethods[] = {
  {(char *)"num", (vmdPyMethod)mol_num, METH_VARARGS, mol_num_doc },
  {(char *)"listall", (vmdPyMethod)mol_listall, METH_VARARGS,mol_listall_doc},
  {(char *)"new", (vmdPyMethod)mol_new, METH_VARARGS, mol_new_doc},
  {(char *)"load", (PyCFunction)mol_load, METH_VARARGS | METH_KEYWORDS, mol_load_doc},
  {(char *)"cancel", (vmdPyMethod)mol_cancel, METH_VARARGS, mol_cancel_doc},
  {(char *)"delete", (vmdPyMethod)mol_delete, METH_VARARGS, mol_delete_doc},
  {(char *)"read", (PyCFunction)mol_read, METH_VARARGS | METH_KEYWORDS, mol_read_doc},
  {(char *)"write", (PyCFunction)mol_write, METH_VARARGS | METH_KEYWORDS, mol_write_doc},
  {(char *)"delframe", (PyCFunction)delframe, METH_VARARGS | METH_KEYWORDS, delframe_doc},
  {(char *)"dupframe", (vmdPyMethod)dupframe, METH_VARARGS, dupframe_doc},
  {(char *)"numframes", (vmdPyMethod)numframes, METH_VARARGS, numframes_doc},
  {(char *)"get_frame", (vmdPyMethod)get_frame, METH_VARARGS, get_frame_doc},
  {(char *)"set_frame", (vmdPyMethod)set_frame, METH_VARARGS, set_frame_doc},
  {(char *)"numatoms", (vmdPyMethod)mol_numatoms, METH_VARARGS, mol_numatoms_doc},
  {(char *)"exists", (vmdPyMethod)mol_exists, METH_VARARGS, mol_exists_doc},
  {(char *)"name", (vmdPyMethod)mol_name, METH_VARARGS, mol_name_doc},
  {(char *)"ssrecalc", (vmdPyMethod)mol_ssrecalc, METH_VARARGS, mol_ssrecalc_doc},
  {(char *)"rename", (vmdPyMethod)mol_rename, METH_VARARGS, mol_rename_doc},
  {(char *)"get_top", (vmdPyMethod)get_top, METH_VARARGS, get_top_doc},
  {(char *)"set_top", (vmdPyMethod)set_top, METH_VARARGS, set_top_doc},
  {(char *)"add_volumetric", (PyCFunction)mol_add_volumetric, 
                              METH_VARARGS | METH_KEYWORDS, add_volumetric_doc},
  {(char *)"get_filenames", (vmdPyMethod)get_filenames, METH_VARARGS, filenames_doc},
  {(char *)"get_filetypes", (vmdPyMethod)get_filetypes, METH_VARARGS, filetypes_doc},
  {(char *)"get_databases", (vmdPyMethod)get_databases, METH_VARARGS, databases_doc},
  {(char *)"get_accessions", (vmdPyMethod)get_accessions, METH_VARARGS, accessions_doc},
  {(char *)"get_remarks", (vmdPyMethod)get_remarks, METH_VARARGS, remarks_doc},
  {(char *)"get_periodic", (PyCFunction)get_periodic, METH_VARARGS | METH_KEYWORDS, get_periodic_doc},
  {(char *)"set_periodic", (PyCFunction)set_periodic, METH_VARARGS | METH_KEYWORDS, set_periodic_doc},
  {(char *)"get_visible", (PyCFunction)get_visible, METH_VARARGS | METH_KEYWORDS, get_visible_doc},
  {(char *)"set_visible", (PyCFunction)set_visible, METH_VARARGS | METH_KEYWORDS, set_visible_doc},
  {(char *)"get_physical_time", (PyCFunction)get_physical_time, METH_VARARGS | METH_KEYWORDS },
  {(char *)"set_physical_time", (PyCFunction)set_physical_time, METH_VARARGS | METH_KEYWORDS },
  {NULL, NULL}
};

void initmolecule() {
  (void) Py_InitModule((char *)"molecule", MolMethods);
}

