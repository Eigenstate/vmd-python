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
 *      $RCSfile: py_commands.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.42 $       $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Core VMD Python interface.
 ***************************************************************************/

#ifndef PY_COMMANDS_H
#define PY_COMMANDS_H

#include "Python.h"

#if (PY_MAJOR_VERSION == 2) && (PY_MINOR_VERSION < 5)
#define CAST_HACK (char *)
#else
#define CAST_HACK
#endif

class VMDApp;
class Timestep;
class AtomSel;

// store/retrieve the VMDApp instance from the __builtins__ module. 
extern VMDApp *get_vmdapp();
void set_vmdapp(VMDApp *);

// turn a PyObject into an array of three floats, if possible
// The object must be a tuple of size 3
// return 1 on success, 0 on error
extern int py_array_from_obj(PyObject *obj, float *arr);

// Turn PyObject to and from strings or ints, with ifdefs for python 2/3
char* as_charptr(PyObject *target);
PyObject* as_pystring(const char *target);
int is_pystring(const PyObject *target);
PyObject* as_pyint(int target);
int as_int(PyObject *target);
int is_pyint(PyObject *target);
int valid_molid(int molid, VMDApp *app);
int py_get_vector(PyObject *matobj, int n, float *vec);
int convert_bool(PyObject *obj, void *boolval);

// Get the timestep corresponding to the given molid and frame.
// If molid is -1, the top molecule will be used.
// If frame is -1, the current timestep is used.
// if frame is -2, the last timestep is used.
// Otherwise, if the molid or frame are not valid, an exception is set
// and NULL is returned.
Timestep *parse_timestep(VMDApp *app, int molid, int frame);

// Return the underlying AtomSel object.  Raise PyError and return
// NULL on failure if the object is not an instance of atomsel.
// Does not check if the molid referenced by the underlying AtomSel 
// is still valid.
AtomSel * atomsel_AsAtomSel( PyObject *obj );

// Atomsel type
extern PyTypeObject Atomsel_Type;

// VMD main initialization function, with no name mangling
#if PY_MAJOR_VERSION >= 3
extern "C" PyObject* PyInit_vmd();
#else
extern "C" void initvmd();
#endif

extern PyObject* initanimate();
extern PyObject* initatomsel();
extern PyObject* initaxes();
extern PyObject* initcolor();
extern PyObject* initdisplay();
extern PyObject* initgraphics();
extern PyObject* initimd();
extern PyObject* initlabel();
extern PyObject* initmaterial();
extern PyObject* initmolecule();
extern PyObject* initmolrep();
extern PyObject* initmouse();
extern PyObject* initrender();
extern PyObject* inittrans();
extern PyObject* initvmdmenu();
extern PyObject* initvmdcallbacks();
extern PyObject* initmeasure();
extern PyObject* inittopology();
extern PyObject* initselection();

#ifdef VMDNUMPY
extern PyObject* initvmdnumpy();
#endif

// Contains submodule initialization functions with submodule names
// Each initialization function returns a PyObject*
struct _py3_inittab {
    const char *name;
    PyObject* (*initfunc)(void);
};

extern _py3_inittab py_initializers[];
#endif

