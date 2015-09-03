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
 *      $RCSfile: py_graphics.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.24 $       $Date: 2011/02/01 16:46:37 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Python interface to user-provided graphics objects
 ***************************************************************************/

#include "py_commands.h"
#include "Molecule.h"
#include "MoleculeList.h"
#include "MaterialList.h"
#include "VMDApp.h"
#include "Scene.h"
#include "MoleculeGraphics.h"

// helper function to get a molecule.  Raises a ValueError exception on error
static MoleculeGraphics *mol_from_id(int id) {
  VMDApp *app = get_vmdapp();
  MoleculeList *mlist = app->moleculeList;
  Molecule *mol = mlist->mol_from_id(id);
  if (!mol) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid graphics molecule");
    return NULL;
  }
  return mol->moleculeGraphics();
}
 
// 
// the rest of the commands define graphics primitives.  They all take an
// id as the first argument, and optional keyword arguments to define the
// the properties of the primitive
//  

// triangle: three vertices as tuples
static PyObject *graphics_triangle(PyObject *self, PyObject *args) {
  int id;
  PyObject *v1, *v2, *v3;
  float arr1[3], arr2[3], arr3[3];
 
  if (!PyArg_ParseTuple(args, (char *)"iOOO:graphics.triangle", &id, &v1, &v2, &v3))
    return NULL;
 
  MoleculeGraphics *mol = mol_from_id(id);
  if (!mol)
    return NULL;
 
  if (!py_array_from_obj(v1, arr1) ||
      !py_array_from_obj(v2, arr2) ||
      !py_array_from_obj(v3, arr3)) 
     return NULL;

  
  int result = mol->add_triangle(arr1, arr2, arr3);
  return PyInt_FromLong(result);
}

// trinorm: three vertices, followed by three normals
static PyObject *graphics_trinorm(PyObject *self, PyObject *args) {
  int id;
  PyObject *v1, *v2, *v3, *n1, *n2, *n3;
  float vert1[3], vert2[3], vert3[3], norm1[3], norm2[3], norm3[3];

  if (!PyArg_ParseTuple(args, (char *)"iOOOOOO:graphics.trinorm", &id, &v1, &v2, &v3, &n1, &n2, &n3))
    return NULL;
  
  MoleculeGraphics *mol = mol_from_id(id);
  if (!mol)
    return NULL;

  if (!py_array_from_obj(v1, vert1) ||
      !py_array_from_obj(v2, vert2) ||
      !py_array_from_obj(v3, vert3) ||
      !py_array_from_obj(n1, norm1) ||
      !py_array_from_obj(n2, norm2) ||
      !py_array_from_obj(n2, norm3)) 
    return NULL;

  int result = mol->add_trinorm(vert1,vert2,vert3,norm1,norm2,norm3);
  return PyInt_FromLong(result);
}

// cylinder: require two vertices
// optional keywords, radius, resolution, filled
static PyObject *graphics_cylinder(PyObject *self, PyObject *args, PyObject *keywds) {
  int id;
  PyObject *v1, *v2;
  float radius = 1.0;
  int resolution = 6;
  int filled = 0;
  float vert1[3], vert2[3];

  static char *kwlist[] = {
    (char *)"id", (char *)"v1", (char *)"v2", (char *)"radius",
    (char *)"resolution", (char *)"filled", NULL
  };

  if (!PyArg_ParseTupleAndKeywords(args, keywds, (char *)"iOO|fii:graphics.cylinder", kwlist,
    &id, &v1, &v2, &radius, &resolution, &filled))
    return NULL;

  MoleculeGraphics *mol = mol_from_id(id);
  if (!mol) 
    return NULL;
   
  if (!py_array_from_obj(v1, vert1) ||
      !py_array_from_obj(v2, vert2))
    return NULL;

  int result = mol->add_cylinder(vert1, vert2, (float)radius, resolution, 
                                 filled);
  return PyInt_FromLong(result);
}

// point: one vertex
static PyObject *graphics_point(PyObject *self, PyObject *args) {
  int id;
  PyObject *v;
  float vert[3];
  
  if (!PyArg_ParseTuple(args, (char *)"iO:graphics.point", &id, &v))
    return NULL;

  MoleculeGraphics *mol = mol_from_id(id);
  if (!mol) 
    return NULL;
  
  if (!py_array_from_obj(v, vert))
    return NULL;

  int result = mol->add_point(vert);
  return PyInt_FromLong(result);
}

// line: two vertices
// optional: style = "solid" | "dashed"
//           width is an integer
static PyObject *graphics_line(PyObject *self, PyObject *args, PyObject *keywds) {
  int id;
  PyObject *v1, *v2;
  float vert1[3], vert2[3];
  int width = 1;
  char *style = NULL;
  int line_style = ::SOLIDLINE;

  static char *kwlist[] = {
    (char *)"id", (char *)"v1", (char *)"v2", (char *)"style", (char *)"width",
    NULL
  };

  if (!PyArg_ParseTupleAndKeywords(args, keywds, (char *)"iOO|si:graphics.line", kwlist,
    &id, &v1, &v2, &style, &width))
    return NULL; 

  MoleculeGraphics *mol = mol_from_id(id);
  if (!mol) 
    return NULL;
 
  if (!py_array_from_obj(v1, vert1) ||
      !py_array_from_obj(v2, vert2))
    return NULL;

  if (style) {
    if (!strcmp(style, "solid"))
      line_style = ::SOLIDLINE;
    else if (!strcmp(style, "dashed"))
      line_style = ::DASHEDLINE;
    else {
      PyErr_SetString(PyExc_ValueError, (char *)"invalid line style");
      return NULL;
    }
  }
  
  // don't check the line width; I don't know what values different display 
  // devices will accept
  int result = mol->add_line(vert1, vert2, line_style, width);
  return PyInt_FromLong(result);
}

// materials: this just turns materials on or off.  Takes an integer
static PyObject *graphics_materials(PyObject *self, PyObject *args) {
  int id;
  int onoff;
  
  if (!PyArg_ParseTuple(args, (char *)"ii:graphics.materials", &id, &onoff))
    return NULL;

  MoleculeGraphics *mol = mol_from_id(id);
  if (!mol) 
    return NULL;
 
  int result = mol->use_materials(onoff);
  return PyInt_FromLong(result);
}

// material(name)
static PyObject *graphics_material(PyObject *self, PyObject *args) {
  int id;
  char *name;
  if (!PyArg_ParseTuple(args, (char *)"is:graphics.material", &id, &name))
     return NULL;

  MoleculeGraphics *mol = mol_from_id(id);
  if (!mol)
    return NULL;

  MaterialList *mlist = get_vmdapp()->materialList;
  int matindex = mlist->material_index(name);
  if (matindex < 0) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid material name");
    return NULL;
  }
  int result = mol->use_material(mlist->material(matindex));
  return PyInt_FromLong(result);
}
   
// colors: Takes one argument, either a tuple of three floats, a string 
// color name, or an integer color index 
static PyObject *graphics_color(PyObject *self, PyObject *args) {
  int id;
  PyObject *obj;
  int result = -1;
  
  if (!PyArg_ParseTuple(args, (char *)"iO:graphics.color", &id, &obj))
    return NULL;

  MoleculeGraphics *mol = mol_from_id(id);
  if (!mol) 
    return NULL;
  
  if (PyInt_Check(obj)) {
    int index = PyInt_AsLong(obj);
    if (index >= 0 && index < MAXCOLORS) 
      result = mol->use_color(index);

  } else if (PyString_Check(obj)) { 
    char *name = PyString_AsString(obj);
    VMDApp *app = get_vmdapp();
    int index = app->color_index(name);
    if (index >= 0) 
      result = mol->use_color(index);
  }

  if (result >= 0)
    return PyInt_FromLong(result);

  PyErr_SetString(PyExc_ValueError, (char *)"Invalid color");
  return NULL;
}  

// cone: just like cylinder, except that it's always filled
static PyObject *graphics_cone(PyObject *self, PyObject *args, PyObject *keywds) {
  int id;
  PyObject *v1, *v2;
  float radius = 1.0;
  float radius2 = 0.0;
  int resolution = 6;
  float vert1[3], vert2[3];

  static char *kwlist[] = {
    (char *)"id", (char *)"v1", (char *)"v2", (char *)"radius",
    (char *)"resolution", NULL
  };

  if (!PyArg_ParseTupleAndKeywords(args, keywds, (char *)"iOO|fi:graphics.cone", kwlist,
    &id, &v1, &v2, &radius, &resolution))
    return NULL;

  MoleculeGraphics *mol = mol_from_id(id);
  if (!mol)
    return NULL;
  
  if (!py_array_from_obj(v1, vert1) ||
      !py_array_from_obj(v2, vert2))
    return NULL;

  int result = mol->add_cone(vert1, vert2, (float)radius, (float)radius2, resolution);
  return PyInt_FromLong(result);
}

// sphere: center (tuple), radius (float), resolution (integer)
static PyObject *graphics_sphere(PyObject *self, PyObject *args, PyObject *keywds) {
  
  int id;
  PyObject *center = NULL;
  float radius = 1;    // default value
  int resolution = 6;  // default value
  float arr[3];        // default will be 0,0,0

  static char *kwlist[] = {
    (char *)"id",(char *)"center", (char *)"radius", (char *)"resolution", NULL
  };

  if (!PyArg_ParseTupleAndKeywords(args, keywds, (char *)"i|Ofi:graphics.sphere", kwlist,
    &id, &center, &radius, &resolution))
    return NULL;
      
  MoleculeGraphics *mol = mol_from_id(id);
  if (!mol) 
    return NULL;

  if (center == NULL) {
    // no center set, so use 0,0,0
    arr[0] = arr[1] = arr[2] = 0.0f;
  } else {
    // try to get center from the passed in object
    if (!py_array_from_obj(center, arr))   
      return NULL;
  }
  int result = mol->add_sphere(arr, radius, resolution);
  return PyInt_FromLong(result);
}

// text: requires position and string arguments
// optional: keyword argument size = float
static PyObject *graphics_text(PyObject *self, PyObject *args, PyObject *keywds) {
  int id;
  PyObject *v;
  float arr[3];
  char *text;
  float size = 1.0f;

  static char *kwlist[] = {
    (char *)"id", (char *)"pos", (char *)"text", (char *)"size", NULL
  };

  if (!PyArg_ParseTupleAndKeywords(args, keywds, (char *)"iOs|f:graphics.text", kwlist,
    &id, &v, &text, &size))
    return NULL;

  MoleculeGraphics *mol = mol_from_id(id);
  if (!mol) 
    return NULL;
  
  if (!py_array_from_obj(v, arr))
    return NULL;

  int result = mol->add_text(arr, text, size, 1.0f);
  return PyInt_FromLong(result);
}

// delete: takes either an integer argument, which is the index to delete,
// or a string argument "all", which deletes everything
// returns nothing
static PyObject *graphics_delete(PyObject *self, PyObject *args) {
  int id;
  PyObject *obj;
  
  if (!PyArg_ParseTuple(args, (char *)"iO:graphics.delete", &id, &obj))
    return NULL;

  MoleculeGraphics *mol = mol_from_id(id);
  if (!mol) 
    return NULL;

  if (PyInt_Check(obj)) {
    int index = PyInt_AsLong(obj);
    mol->delete_id(index);
  } else if (PyString_Check(obj)) {
    char *s = PyString_AsString(obj);
    if (!strcmp(s, "all")) 
      mol->delete_all();
    else {
      PyErr_SetString(PyExc_ValueError, (char *)"delete: invalid value");
      return NULL;
    }
  }

  Py_INCREF(Py_None);
  return Py_None;
}

// replace: delete the given id and have the next element replace this one
static PyObject *graphics_replace(PyObject *self, PyObject *args) {
  int id;
  int index;

  if (!PyArg_ParseTuple(args, (char *)"ii:graphics.replace", &id, &index))
    return NULL;

  MoleculeGraphics *mol = mol_from_id(id);
  if (!mol) 
    return NULL;
   
  mol->replace_id(index);

  Py_INCREF(Py_None);
  return Py_None;
}

// info: return a string describing the graphics object with the given index
// if out of range, or the object no longer exists, raise IndexError 
static PyObject *graphics_info(PyObject *self, PyObject *args) {
  int id;
  int index;

  if (!PyArg_ParseTuple(args, (char *)"ii:graphics.info", &id, &index))
    return NULL;

  MoleculeGraphics *mol = mol_from_id(id);
  if (!mol) 
    return NULL;

  if (mol->index_id(index) == -1) {
    PyErr_SetString(PyExc_IndexError, "Invalid graphics object");
    return NULL;
  } 
  const char *info = mol->info_id(index);
  return PyString_FromString(info);
}
 
// list: return a list of the valid graphics objects
static PyObject *graphics_listall(PyObject *self, PyObject *args) {
  int id;
  
  if (!PyArg_ParseTuple(args, (char *)"i:graphics.listall", &id))
    return NULL;

  MoleculeGraphics *mol = mol_from_id(id);
  if (!mol) 
    return NULL;

  PyObject *newlist = PyList_New(0);
  int num = mol->num_elements();
  for (int i=0; i<num; i++) {
    int index = mol->element_id(i);
    if (index >= 0) {
      if (PyList_Append(newlist, PyInt_FromLong(index)) < 0)  
        return NULL;
    }
  }
  return newlist;
}

static PyMethodDef GraphicsMethods[] = {
  {(char *)"sphere", (PyCFunction)graphics_sphere,METH_VARARGS | METH_KEYWORDS},
  {(char *)"triangle",(vmdPyMethod)graphics_triangle,METH_VARARGS},
  {(char *)"trinorm",(vmdPyMethod)graphics_trinorm,METH_VARARGS},
  {(char *)"cylinder", (PyCFunction)graphics_cylinder,METH_VARARGS | METH_KEYWORDS},
  {(char *)"point",(vmdPyMethod)graphics_point,METH_VARARGS},
  {(char *)"line", (PyCFunction)graphics_line,METH_VARARGS | METH_KEYWORDS},
  {(char *)"materials",(vmdPyMethod)graphics_materials,METH_VARARGS},
  {(char *)"material",(vmdPyMethod)graphics_material,METH_VARARGS},
  {(char *)"color",(vmdPyMethod)graphics_color,METH_VARARGS},
  {(char *)"cone", (PyCFunction)graphics_cone,METH_VARARGS | METH_KEYWORDS},
  {(char *)"sphere", (PyCFunction)graphics_sphere,METH_VARARGS | METH_KEYWORDS},
  {(char *)"text", (PyCFunction)graphics_text,METH_VARARGS | METH_KEYWORDS},
  {(char *)"delete",(vmdPyMethod)graphics_delete,METH_VARARGS},
  {(char *)"replace",(vmdPyMethod)graphics_replace,METH_VARARGS},
  {(char *)"info",(vmdPyMethod)graphics_info,METH_VARARGS},
  {(char *)"listall",(vmdPyMethod)graphics_listall,METH_VARARGS},
  {NULL, NULL}
};

void initgraphics() {
  (void) Py_InitModule((char *)"graphics", GraphicsMethods);
}
 
