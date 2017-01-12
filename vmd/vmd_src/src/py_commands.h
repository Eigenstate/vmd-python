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
 *      $RCSfile: py_commands.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.41 $       $Date: 2016/11/28 03:05:08 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Core VMD Python interface.
 ***************************************************************************/

#ifndef PY_COMMANDS_H
#define PY_COMMANDS_H

#if defined(__APPLE__)
// use the Apple-provided Python framework
#include "Python/Python.h"
#else
#include "Python.h"
#endif
#include "bytesobject.h"

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

#if PY_MAJOR_VERSION < 3
extern void initanimate();
extern void initatomselection();
extern void initatomsel();
extern void initaxes();
extern void initcolor();
extern void initdisplay();
extern void initgraphics();
extern void initimd();
extern void initlabel();
extern void initmaterial();
extern void initmolecule();
extern void initmolrep();
extern void initmouse();
extern void initrender();
extern void inittrans();
extern void initvmdmenu();
extern void initmeasure();
extern void inittopology();

#ifdef VMDNUMPY
extern void initvmdnumpy();
#endif
#endif

// use this typedef so that we can define our Python methods as static 
// functions, then cast them to the proper type, rather than declaring the
// functions extern "C", which can lead to namespace collision. 

extern "C" {
  typedef PyObject *(*vmdPyMethod)(PyObject *, PyObject *);
}

#endif

