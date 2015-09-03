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
 *      $RCSfile: py_commands.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.14 $       $Date: 2010/12/16 04:08:56 $
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


// The VMDApp instance will be found in the VMDApp module, under the
// VMDApp dictionary entry.  Got it?  

VMDApp *get_vmdapp() {
  PyObject *module = PyImport_ImportModule((char *)"__builtin__");
  if (!module) return NULL;
  PyObject *module_dict = PyModule_GetDict(module);
  if (!module_dict) return NULL;
  PyObject *c_obj = PyDict_GetItemString(module_dict, (char *)"-vmdapp-");
  if (!c_obj) return NULL;
  if (PyCObject_Check(c_obj))
    return (VMDApp *)PyCObject_AsVoidPtr(c_obj);
  return NULL;
}

void set_vmdapp(VMDApp *app) {
  PyObject *mod = PyImport_ImportModule((char *)"__builtin__");
  PyObject_SetAttrString(mod, (char *)"-vmdapp-", 
      PyCObject_FromVoidPtr(app, NULL));
  Py_DECREF(mod);
}

int py_array_from_obj(PyObject *obj, float *arr) {
  if (PyTuple_Check(obj)) {
    if (PyTuple_Size(obj) != 3) {
      PyErr_SetString(PyExc_ValueError, (char *)"Tuple must have length 3");
      return 0;
    }
    for (int i=0; i<3; i++) {
      PyObject *elem = PyTuple_GET_ITEM(obj, i);
      arr[i] = PyFloat_AsDouble(elem);
      if (PyErr_Occurred())
        return 0;
    }
    return 1;  // successful return
  }
  PyErr_SetString(PyExc_ValueError, (char *)"Invalid tuple");
  return 0;
}

Timestep *parse_timestep(VMDApp *app, int molid, int frame) {
  if (molid < 0) molid = app->molecule_top();
  Molecule *mol = app->moleculeList->mol_from_id(molid);
  if (!mol) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  }
  Timestep *ts = NULL;
  if (frame == -1) {
    ts = mol->current();
  } else if (frame == -2) {
    ts = mol->get_last_frame();
  } else {
    ts = mol->get_frame(frame);
  }
  if (!ts) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid frame");
    return NULL;
  }
  return ts;
}

