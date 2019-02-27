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
 *      $RCSfile: py_commands.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.16 $       $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Core VMD Python interface
 ***************************************************************************/

#include "py_commands.h"
#include "VMDApp.h"
#include "Molecule.h"
#include "MoleculeList.h"

/*

Some distributed versions of VMD are linked against the Python 2.0 
library.  The following BeOpen license agreement permits us to use the 
Python 2.0 libraries in this fashion.  The BeOpen license agreement is in
no way applicable to the license under which VMD itself is distributed;
persuant to item 2 below, we merely include a copy of the BeOpen license
to indicate our use of the BeOpen software.

HISTORY OF THE SOFTWARE
=======================

Python was created in the early 1990s by Guido van Rossum at Stichting
Mathematisch Centrum (CWI) in the Netherlands as a successor of a
language called ABC.  Guido is Python's principal author, although it
includes many contributions from others.  The last version released
from CWI was Python 1.2.  In 1995, Guido continued his work on Python
at the Corporation for National Research Initiatives (CNRI) in Reston,
Virginia where he released several versions of the software.  Python
1.6 was the last of the versions released by CNRI.  In 2000, Guido and
the Python core developement team moved to BeOpen.com to form the
BeOpen PythonLabs team (www.pythonlabs.com).  Python 2.0 is the first
release from PythonLabs.  Thanks to the many outside volunteers who
have worked under Guido's direction to make this release possible.



BEOPEN.COM TERMS AND CONDITIONS FOR PYTHON 2.0
==============================================

BEOPEN PYTHON OPEN SOURCE LICENSE AGREEMENT VERSION 1
-----------------------------------------------------

1. This LICENSE AGREEMENT is between BeOpen.com ("BeOpen"), having an
office at 160 Saratoga Avenue, Santa Clara, CA 95051, and the
Individual or Organization ("Licensee") accessing and otherwise using
this software in source or binary form and its associated
documentation ("the Software").

2. Subject to the terms and conditions of this BeOpen Python License
Agreement, BeOpen hereby grants Licensee a non-exclusive,
royalty-free, world-wide license to reproduce, analyze, test, perform
and/or display publicly, prepare derivative works, distribute, and
otherwise use the Software alone or in any derivative version,
provided, however, that the BeOpen Python License is retained in the
Software, alone or in any derivative version prepared by Licensee.

3. BeOpen is making the Software available to Licensee on an "AS IS"
basis.  BEOPEN MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR
IMPLIED.  BY WAY OF EXAMPLE, BUT NOT LIMITATION, BEOPEN MAKES NO AND
DISCLAIMS ANY REPRESENTATION OR WARRANTY OF MERCHANTABILITY OR FITNESS
FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF THE SOFTWARE WILL NOT
INFRINGE ANY THIRD PARTY RIGHTS.

4. BEOPEN SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF THE
SOFTWARE FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR LOSS
AS A RESULT OF USING, MODIFYING OR DISTRIBUTING THE SOFTWARE, OR ANY
DERIVATIVE THEREOF, EVEN IF ADVISED OF THE POSSIBILITY THEREOF.

5. This License Agreement will automatically terminate upon a material
breach of its terms and conditions.

6. This License Agreement shall be governed by and interpreted in all
respects by the law of the State of California, excluding conflict of
law provisions.  Nothing in this License Agreement shall be deemed to
create any relationship of agency, partnership, or joint venture
between BeOpen and Licensee.  This License Agreement does not grant
permission to use BeOpen trademarks or trade names in a trademark
sense to endorse or promote products or services of Licensee, or any
third party.  As an exception, the "BeOpen Python" logos available at
http://www.pythonlabs.com/logos.html may be used according to the
permissions granted on that web page.

7. By copying, installing or otherwise using the software, Licensee
agrees to be bound by the terms and conditions of this License
Agreement.


*/

// Wrapper function for getting char* from a Python string, with 2/3 ifdefs
char* as_charptr(PyObject *target)
{
    char *result;
#if PY_MAJOR_VERSION >= 3
    result = (char*) PyUnicode_AsUTF8(target);
#else
    result = PyString_AsString(target);
#endif

    if (! result || PyErr_Occurred()) {
        PyErr_SetString(PyExc_ValueError, "cannot convert PyObject to char*");
        return NULL;
    }
    return result;
}

// Wrapper function for turning char* into a Python string, with 2/3 ifdefs
PyObject* as_pystring(const char *target)
{
    PyObject *result;
    if (!target) {
        PyErr_SetString(PyExc_ValueError, "cannot return null string");
        return NULL;
    }

#if PY_MAJOR_VERSION >= 3
    result = PyUnicode_FromString(target);
#else
    result = PyString_FromString(target);
#endif

    if (!result) {
        PyErr_Format(PyExc_ValueError, "cannot convert char* '%s'", target);
        return NULL;
    }

    // Separate this so it gives a more helpful error message
    if (PyErr_Occurred())
        return NULL;

    return result;
}

// Wrapper function for checking if a PyObject is a string, with 2/3 ifdefs
int is_pystring(const PyObject *target)
{
#if PY_MAJOR_VERSION >= 3
    return PyUnicode_Check(target);
#else
    return PyString_Check(target);
#endif
}

// Wrapper function for turning int into a Python int/long, with 2/3 ifdefs
PyObject* as_pyint(int target)
{
    PyObject *result;
#if PY_MAJOR_VERSION >= 3
    result = PyLong_FromLong((long) target);
#else
    result = PyInt_FromLong((long) target);
#endif

    if (!result) {
        PyErr_Format(PyExc_ValueError, "cannot convert int %d", target);
        return NULL;
    }

    // Separate this so it gives a more helpful error message
    if (PyErr_Occurred())
        return NULL;

    return result;
}

// Wrapper function for getting int from a Python object, with 2/3 ifdefs
int as_int(PyObject *target)
{
    int result;
#if PY_MAJOR_VERSION >= 3
    if (!PyLong_Check(target)) {
#else
    if (!PyInt_Check(target)) {
#endif
        PyErr_SetString(PyExc_ValueError, "Non-integer Python object in as_int");
        return -1;
    }

#if PY_MAJOR_VERSION >= 3
    result = (int) PyLong_AsLong(target);
#else
    result = (int) PyInt_AsLong(target);
#endif

    return result;
}

// Wrapper function to check if a python object is an int
int is_pyint(PyObject *target)
{
#if PY_MAJOR_VERSION >= 3
    return PyLong_Check(target);
#else
    return PyInt_Check(target);
#endif
}

// The VMDApp instance will be found in the VMDApp module, under the
// VMDApp dictionary entry.  Got it?  

VMDApp *get_vmdapp() {

  PyObject *module_dict = PyEval_GetBuiltins();
  if (!module_dict)
      return NULL;

// Python 3 uses a "capsule" to store C pointers
#if PY_MAJOR_VERSION >= 3
    PyObject *c_obj = PyDict_GetItemString(module_dict, "-vmdapp-");
    if (!c_obj || PyErr_Occurred())
        return NULL;

    if (PyCapsule_CheckExact(c_obj))
        return (VMDApp *)PyCapsule_GetPointer(c_obj, "-vmdapp-");

// Python 2 instead uses the idea of a "C Object"
#else
    PyObject *c_obj = PyDict_GetItemString(module_dict, (char *)"-vmdapp-");
    if (!c_obj || PyErr_Occurred())
        return NULL;

    if (PyCObject_Check(c_obj))
        return (VMDApp *)PyCObject_AsVoidPtr(c_obj);
#endif
    return NULL;
}

void set_vmdapp(VMDApp *app) {

  PyObject *module_dict = PyEval_GetBuiltins();
  PyObject *cap;

#if PY_MAJOR_VERSION >= 3
  cap = PyCapsule_New(app, "-vmdapp-", NULL);
#else
  cap = PyCObject_FromVoidPtr(app, NULL);
#endif
  PyDict_SetItemString(module_dict, (const char *)"-vmdapp-", cap);
}

int py_array_from_obj(PyObject *obj, float *arr) {

   PyObject *seqdata = NULL;
   PyObject *elem;
   int i;

  if (!(seqdata = PySequence_Fast(obj, "Coordinate argument must be a sequence")))
    goto failure;

  if (PySequence_Fast_GET_SIZE(seqdata) != 3) {
    PyErr_SetString(PyExc_ValueError, "Coordinate must have length 3");
    goto failure;
  }

  for (i = 0; i < 3; i++) {
    elem = PySequence_Fast_GET_ITEM(seqdata, i);

    arr[i] = PyFloat_AsDouble(elem);
    if (PyErr_Occurred()) {
      PyErr_SetString(PyExc_ValueError, "Problem unpacking coordinate");
      goto failure;
    }
  }
  return 1;  // successful return

failure:
  Py_XDECREF(seqdata);
  return 0;
}

Timestep *parse_timestep(VMDApp *app, int molid, int frame) {

  Timestep *ts = NULL;
  Molecule *mol;

  // Get molecule from molid or top molecule
  if (molid < 0)
    molid = app->molecule_top();
  mol = app->moleculeList->mol_from_id(molid);

  if (!mol) {
    PyErr_Format(PyExc_ValueError, "Invalid molecule id '%d'", molid);
    return NULL;
  }

  // Get frame number
  if (frame == -1)
    ts = mol->current();
  else if (frame == -2)
    ts = mol->get_last_frame();
  else
    ts = mol->get_frame(frame);

  if (!ts) {
    PyErr_Format(PyExc_ValueError, "Invalid frame '%d'", frame);
    return NULL;
  }

  return ts;
}

// Helper function to check if molid is valid and set exception if not
int valid_molid(int molid, VMDApp *app)
{
  if (!app->molecule_valid_id(molid)) {
    PyErr_Format(PyExc_ValueError, "Invalid molecule id '%d'", molid);
    return 0;
  }
  return 1;
}


// extract vector from sequence object.  Return success.
int py_get_vector(PyObject *matobj, int n, float *vec) {

  PyObject *fastval = NULL;
  PyObject *fval;
  int i;

  if (!PySequence_Check(matobj) || PySequence_Size(matobj) != n) {
    PyErr_SetString(PyExc_ValueError, "vector has incorrect size");
    return 0;
  }

  if (!(fastval = PySequence_Fast(matobj, "Invalid sequence")))
    goto failure;

  for (i = 0; i < n; i++) {
    fval = PySequence_Fast_GET_ITEM(fastval, i);

    if (!PyFloat_Check(fval)) {
      PyErr_SetString(PyExc_TypeError, "vector must contain only floats");
      goto failure;
    }

    vec[i] = PyFloat_AsDouble(fval);
    if (PyErr_Occurred())
      goto failure;
  }

  Py_DECREF(fastval);
  return 1;

failure:
  Py_XDECREF(fastval);
  return 0;
}

// Converter function for boolean arguments
int convert_bool(PyObject *obj, void *boolval)
{
  if (!PyObject_TypeCheck(obj, &PyBool_Type)) {
    PyErr_SetString(PyExc_TypeError, "expected a boolean");
    return 0;
  }

  *((int*)(boolval)) = PyObject_IsTrue(obj);
  return 1; // success
}

_py3_inittab py_initializers[] = {
   {"animate", initanimate},
   {"atomsel", initatomsel},
   {"axes", initaxes},
   {"color", initcolor},
   {"display", initdisplay},
   {"graphics", initgraphics},
#ifdef VMDIMD
   {"imd", initimd},
#endif
   {"label", initlabel},
   {"material", initmaterial},
   {"molecule", initmolecule},
   {"molrep", initmolrep},
   {"mouse", initmouse},
   {"render", initrender},
   {"trans", inittrans},
   {"measure", initmeasure},
   {"topology", inittopology},
   {"selection", initselection},
   {"vmdcallbacks", initvmdcallbacks},
   {"vmdmenu", initvmdmenu},
#ifdef VMDNUMPY
   {"vmdnumpy", initvmdnumpy},
#endif
   {NULL, NULL},
};

