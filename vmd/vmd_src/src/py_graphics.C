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
 *      $RCSfile: py_graphics.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.26 $       $Date: 2019/01/17 21:21:03 $
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
static MoleculeGraphics *mol_from_id(int id)
{
  MoleculeList *mlist;
  Molecule *mol;
  VMDApp *app;

  if (!(app = get_vmdapp()))
    return NULL;

  mlist = app->moleculeList;
  if (!(mol = mlist->mol_from_id(id))) {
    PyErr_Format(PyExc_ValueError, "Invalid graphics molecule '%d'", id);
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
static char triangle_doc[] =
"Draws a triangle\n\n"
"Args:\n"
"    molid (int): Molecule ID to draw on\n"
"    v1 (3-tuple of float): (x,y,z) coordinates of first vertex\n"
"    v2 (3-tuple of float): (x,y,z) coordinates of second vertex\n"
"    v3 (3-tuple of float): (x,y,z) coordinates of third vertex\n"
"Returns:\n"
"    (int): ID of drawn triangle";
static PyObject* py_triangle(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "v1", "v2", "v3", NULL};
  float arr1[3], arr2[3], arr3[3];
  PyObject *v1, *v2, *v3;
  MoleculeGraphics *mol;
  int id;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iOOO:graphics.triangle",
                                   (char**) kwlist, &id, &v1, &v2, &v3))
    return NULL;

  if (!(mol = mol_from_id(id)))
    return NULL;

  if (!py_array_from_obj(v1, arr1) ||
      !py_array_from_obj(v2, arr2) ||
      !py_array_from_obj(v3, arr3))
     return NULL;


  return as_pyint(mol->add_triangle(arr1, arr2, arr3));
}

static char trinorm_doc[] =
"Draws a triangle with given vertices and vertex normals\n\n"
"Args:\n"
"    molid (int): Molecule ID to draw on\n"
"    v1 (3-tuple of float): (x,y,z) coordinates of first vertex\n"
"    v2 (3-tuple of float): (x,y,z) coordinates of second vertex\n"
"    v3 (3-tuple of float): (x,y,z) coordinates of third vertex\n"
"    n1 (3-tuple of float): (x,y,z) normal of first vertex\n"
"    n2 (3-tuple of float): (x,y,z) normal of second vertex\n"
"    n3 (3-tuple of float): (x,y,z) normal of third vertex\n"
"Returns:\n"
"    (int): ID of drawn triangle";
static PyObject* py_trinorm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "v1", "v2", "v3", "n1", "n2", "n3", NULL};
  float vert1[3], vert2[3], vert3[3], norm1[3], norm2[3], norm3[3];
  PyObject *v1, *v2, *v3, *n1, *n2, *n3;
  MoleculeGraphics *mol;
  int id;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iOOOOOO:graphics.trinorm",
                                   (char**) kwlist, &id, &v1, &v2, &v3, &n1,
                                   &n2, &n3))
    return NULL;

  if (!(mol = mol_from_id(id)))
    return NULL;

  if (!py_array_from_obj(v1, vert1) ||
      !py_array_from_obj(v2, vert2) ||
      !py_array_from_obj(v3, vert3) ||
      !py_array_from_obj(n1, norm1) ||
      !py_array_from_obj(n2, norm2) ||
      !py_array_from_obj(n2, norm3))
    return NULL;

  return as_pyint(mol->add_trinorm(vert1,vert2,vert3,norm1,norm2,norm3));
}

static char cylinder_doc[] =
"Draws a cylinder\n\n"
"Args:\n"
"    molid (int): Molecule ID to draw on \n"
"    v1 (3-tuple of float): (x,y,z) coordinates of first vertex\n"
"    v2 (3-tuple of float): (x,y,z) coordinates of second vertex\n"
"    radius (float): Cylinder radius, defaults to 1.0\n"
"    resolution (int): Number of sides of cylinder, defaults to 6\n"
"    filled (bool): If cylinder ends should be capped\n"
"Returns:\n"
"    (int): ID of drawn cylinder";
static PyObject* py_cylinder(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "v1", "v2", "radius", "resolution", "filled",
                          NULL};
  int resolution = 6, filled = 0;
  float vert1[3], vert2[3];
  MoleculeGraphics *mol;
  float radius = 1.0;
  PyObject *v1, *v2;
  int id;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iOO|fiO&:graphics.cylinder",
                                   (char**) kwlist, &id, &v1, &v2, &radius,
                                   &resolution, convert_bool, &filled))
    return NULL;

  if (!(mol = mol_from_id(id)))
    return NULL;

  if (!py_array_from_obj(v1, vert1) ||
      !py_array_from_obj(v2, vert2))
    return NULL;

  return as_pyint(mol->add_cylinder(vert1, vert2, radius, resolution, filled));
}

static char point_doc[] =
"Draw a point at the given vertex\n\n"
"Args:\n"
"    molid (int): Molecule ID to draw on\n"
"    v1 (3-tuple of float): (x,y,z) coordinates of point\n"
"Returns:\n"
"    (int): ID of drawn point";
static PyObject *py_point(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "v1", NULL};
  MoleculeGraphics *mol;
  float vert[3];
  PyObject *v;
  int id;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iO:graphics.point",
                                   (char**) kwlist, &id, &v))
    return NULL;

  if (!(mol = mol_from_id(id)))
    return NULL;

  if (!py_array_from_obj(v, vert))
    return NULL;

  return as_pyint(mol->add_point(vert));
}

static char line_doc[] =
"Draw a line between the given vertices\n\n"
"Args:\n"
"    molid (int): Molecule ID to draw on\n"
"    v1 (3-tuple of float): (x,y,z) coordinates of first vertex\n"
"    v2 (3-tuple of float): (x,y,z) coordinates of second vertex\n"
"    style (str): Either 'solid' or 'dashed'. Defaults to solid line\n"
"    width (int): Width of line. Defaults to 1\n"
"Returns:\n"
"    (int): ID of drawn line";
static PyObject* py_line(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "v1", "v2", "style", "width", NULL};
  int line_style = ::SOLIDLINE;
  float vert1[3], vert2[3];
  MoleculeGraphics *mol;
  char *style = NULL;
  PyObject *v1, *v2;
  int width = 1;
  int id;


  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iOO|zi:graphics.line",
                                   (char**) kwlist, &id, &v1, &v2, &style,
                                   &width))
    return NULL;

  if (!(mol = mol_from_id(id)))
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
      PyErr_Format(PyExc_ValueError, "invalid line style '%s'", style);
      return NULL;
    }
  }

  // don't check the line width; I don't know what values different display
  // devices will accept
  return as_pyint(mol->add_line(vert1, vert2, line_style, width));
}

// materials: this just turns materials on or off.  Takes an integer
static char mats_doc[] =
"Turns materials on or off for subsequent graphics primitives. Already drawn\n"
"graphics objects are not affected.\n\n"
"Args:\n"
"    molid (int): Molecule ID to affect\n"
"    on (bool): If materials should be on";
static PyObject* py_materials(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "on", NULL};
  MoleculeGraphics *mol;
  int id, onoff;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iO&:graphics.materials",
                                   (char**) kwlist, &id, convert_bool, &onoff))
    return NULL;

  if (!(mol = mol_from_id(id)))
    return NULL;

  mol->use_materials(onoff);
  Py_INCREF(Py_None);
  return Py_None;
}

// material(name)
static char mat_doc[] =
"Set material for all graphics in this molecule\n\n"
"Args:\n"
"    molid (int): Molecule ID to affect\n"
"    name (str): Material name. Must be one returned by `material.listall()`";
static PyObject* py_material(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "name", NULL};
  MoleculeGraphics *mol;
  MaterialList *mlist;
  int id, matindex;
  char *name;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "is:graphics.material",
                                   (char**) kwlist, &id, &name))
     return NULL;

  if (!(mol = mol_from_id(id)))
    return NULL;

  mlist = get_vmdapp()->materialList;
  matindex = mlist->material_index(name);
  if (matindex < 0) {
    PyErr_Format(PyExc_ValueError, "Invalid material name '%s'", name);
    return NULL;
  }

  mol->use_material(mlist->material(matindex));
  Py_INCREF(Py_None);
  return Py_None;
}

static char color_doc[] =
"Set color for subsequent graphics primitives in this molecule.\n\n"
"Args:\n"
"    molid (int): Molecule ID to affect\n"
"    color (int, 3-tuple of float, or str): Color, either as color ID,\n"
"        or name.";
static PyObject* py_color(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "color", NULL};
  MoleculeGraphics *mol;
  int id, index;
  PyObject *obj;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iO:graphics.color",
                                   (char**) kwlist, &id, &obj))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!(mol = mol_from_id(id)))
    return NULL;

  // Get index of passed integer color
  if (is_pyint(obj)) {
    index = as_int(obj);

  // Get index of passed color as string
  } else if (is_pystring(obj)) {
    index = app->color_index(as_charptr(obj));

  // Error for unsupported type
  } else {
    PyErr_SetString(PyExc_TypeError, "Wrong type for color object");
    return NULL;
  }

  // Check retrieved color index is sane
  if (index < 0 || index >= MAXCOLORS) {
    PyErr_Format(PyExc_ValueError, "Color index '%d' out of bounds", index);
    return NULL;
  }

  // Actually set the color
  if (mol->use_color(index) < 0) {
    PyErr_SetString(PyExc_ValueError, "Error using color");
    return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}

static char cone_doc[] =
"Draws a cone. Base of cone is always filled\n\n"
"Args:\n"
"    molid (int): Molecule ID to draw on \n"
"    v1 (3-tuple of float): (x,y,z) coordinates of first vertex\n"
"    v2 (3-tuple of float): (x,y,z) coordinates of second vertex\n"
"    radius (float): Cone radius at base, defaults to 1.0\n"
"    radius2 (float): Cone radius at end. Defaults to 0.0 for pointy cone.\n"
"      If nonzero, end will not appear as filled."
"    resolution (int): Number of sides of cone , defaults to 6\n"
"Returns:\n"
"    (int): ID of drawn cone";
static PyObject *graphics_cone(PyObject *self, PyObject *args,
                               PyObject *kwargs)
{
  float vert1[3], vert2[3];
  MoleculeGraphics *mol;
  PyObject *v1, *v2;
  float radius = 1.0;
  float radius2 = 0.0;
  int resolution = 6;
  int id;

  const char *kwlist[] = {"molid", "v1", "v2", "radius", "radius2",
                          "resolution", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iOO|ffi:graphics.cone",
                                  (char**)  kwlist, &id, &v1, &v2, &radius,
                                   &radius2, &resolution))
    return NULL;

  if (!(mol = mol_from_id(id)))
    return NULL;

  if (!py_array_from_obj(v1, vert1) ||
      !py_array_from_obj(v2, vert2))
    return NULL;

  return as_pyint(mol->add_cone(vert1, vert2, radius, radius2, resolution));
}

static char sphere_doc[] =
"Draws a sphere\n\n"
"Args:\n"
"    molid (int): Molecule ID to draw on \n"
"    center (3-tuple of float): (x,y,z) coordinates to center sphere.\n"
"        Defaults to the origin.\n"
"    radius (float): Sphere radius. Defaults to 1.0\n"
"    resolution (int): Sphere resolution. Defaults to 6\n"
"Returns:\n"
"    (int): ID of drawn sphere";
static PyObject *graphics_sphere(PyObject *self, PyObject *args,
                                 PyObject *kwargs)
{

  const char *kwlist[] = {"molid","center", "radius", "resolution", NULL};
  PyObject *center = NULL;
  MoleculeGraphics *mol;
  int resolution = 6;
  float radius = 1;
  float arr[3];
  int id;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i|Ofi:graphics.sphere",
                                   (char**) kwlist, &id, &center, &radius,
                                   &resolution))
    return NULL;

  if (!(mol = mol_from_id(id)))
    return NULL;

  // If no center set, default to origin
  if (center == NULL || center == Py_None) {
    arr[0] = arr[1] = arr[2] = 0.0f;

  // Otherwise, try to unpack tuple
  } else if (!py_array_from_obj(center, arr)) {
    PyErr_SetString(PyExc_ValueError, "Sphere center must be a 3-tuple");
    return NULL;
  }

  return as_pyint(mol->add_sphere(arr, radius, resolution));
}

static char text_doc[] =
"Draw text\n\n"
"Args:\n"
"    molid (int): Molecule ID to draw on\n"
"    position (3-tuple of float): (x,y,z) coordinates to center text on\n"
"    text (str): Text to display\n"
"    size (float): Text size. Defaults to 1.0\n"
"    width (float): Text width. Defaults to 1.0\n"
"Returns:\n"
"    (int): ID of drawn text\n";
static PyObject *graphics_text(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  const char *kwlist[] = {"molid", "position", "text", "size", "width", NULL};
  MoleculeGraphics *mol;
  float size = 1.0f, width = 1.0f;
  float arr[3];
  PyObject *v;
  char *text;
  int id;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iOs|ff:graphics.text",
                                   (char**) kwlist, &id, &v, &text, &size,
                                   &width))
    return NULL;

  if (!(mol = mol_from_id(id)))
    return NULL;

  if (!py_array_from_obj(v, arr)) {
    PyErr_SetString(PyExc_ValueError, "Text position must be a 3-tuple");
    return NULL;
  }

  return as_pyint(mol->add_text(arr, text, size, width));
}

// delete: takes either an integer argument, which is the index to delete,
// or a string argument "all", which deletes everything
// returns nothing
static char delete_doc[] =
"Deletes a specified graphics object, or all graphics at a given molid\n\n"
"Args:\n"
"    molid (int): Molecule ID to delete graphics from\n"
"    which (int or str): Graphics ID to delete, or 'all' for all objects.\n";
static PyObject *graphics_delete(PyObject *self, PyObject *args,
                                 PyObject *kwargs)
{
  const char* kwlist[] = {"molid", "which", NULL};
  MoleculeGraphics *mol;
  PyObject *obj;
  int id;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iO:graphics.delete",
                                   (char**) kwlist, &id, &obj))
    return NULL;

  if (!(mol = mol_from_id(id)))
    return NULL;

  // Handle integer object to delete
  if (is_pyint(obj)) {

    int index = as_int(obj);
    mol->delete_id(index);

  // Or, string
  } else if (is_pystring(obj)) {

    char *s = as_charptr(obj);

    if (!strcmp(s, "all")) {
      mol->delete_all();
    } else {
      PyErr_Format(PyExc_ValueError, "Invalid delete string '%s'", s);
      return NULL;
    }

  // Otherwise, invalid input
  } else {
      PyErr_SetString(PyExc_TypeError, "Invalid argument for 'which'. Must "
                      "be int or str");
      return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}

static char replace_doc[] =
"Delete a graphics object and have the next element replace this one.\n\n"
"Args:\n"
"    molid (int): Molecule ID containing graphics object\n"
"    graphic (int): Graphics object ID to delete\n";
static PyObject *graphics_replace(PyObject *self, PyObject *args,
                                  PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "graphic", NULL};
  MoleculeGraphics *mol;
  int index;
  int id;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii:graphics.replace",
                                   (char**) kwlist, &id, &index))
    return NULL;

  if (!(mol = mol_from_id(id)))
    return NULL;

  mol->replace_id(index);

  Py_INCREF(Py_None);
  return Py_None;
}

static char info_doc[] =
"Describe a graphics object with given index\n\n"
"Args:\n"
"    molid (int): Molecule ID containing graphics object\n"
"    graphic (int): Graphics object ID to describe\n"
"Returns:\n"
"    (str): Description of graphics object\n"
"Raises:\n"
"    IndexError: If object does not exist\n";
static PyObject *graphics_info(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "graphic", NULL};
  MoleculeGraphics *mol;
  int index;
  int id;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii:graphics.info",
                                   (char**) kwlist, &id, &index))
    return NULL;

  if (!(mol = mol_from_id(id)))
    return NULL;

  if (mol->index_id(index) == -1) {
    PyErr_SetString(PyExc_IndexError, "Invalid graphics object");
    return NULL;
  }

  return as_pystring(mol->info_id(index));
}

static char list_doc[] =
"List all drawn graphics objects on a given molecule\n\n"
"Args:\n"
"    molid (int): Molecule ID to query\n"
"Returns:\n"
"    (list of int): Graphics object IDs present in molecule";
static PyObject *graphics_listall(PyObject *self, PyObject *args,
                                  PyObject *kwargs)
{
  const char *kwlist[] = {"molid", NULL};
  MoleculeGraphics *mol;
  PyObject *newlist;
  int id;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i:graphics.listall",
                                   (char**) kwlist, &id))
    return NULL;

  if (!(mol = mol_from_id(id)))
    return NULL;

  newlist = PyList_New(0);
  for (int i=0; i< mol->num_elements(); i++) {
    int index = mol->element_id(i);
    if (index >= 0) {
      // PyList_Append does not steal a reference
      PyObject *tmp = as_pyint(index);
      int rc = PyList_Append(newlist, tmp);
      Py_XDECREF(tmp);

      if (rc || PyErr_Occurred())
        goto failure;
    }
  }

  return newlist;

failure:
  PyErr_SetString(PyExc_ValueError, "Problem listing graphics objects");
  Py_DECREF(newlist);
  return NULL;
}

static PyMethodDef GraphicsMethods[] = {
  {"cone", (PyCFunction)graphics_cone,METH_VARARGS | METH_KEYWORDS, cone_doc},
  {"sphere", (PyCFunction)graphics_sphere, METH_VARARGS | METH_KEYWORDS, sphere_doc},
  {"triangle",(PyCFunction)py_triangle, METH_VARARGS | METH_KEYWORDS, triangle_doc},
  {"trinorm",(PyCFunction)py_trinorm, METH_VARARGS | METH_KEYWORDS, trinorm_doc},
  {"point",(PyCFunction)py_point, METH_VARARGS | METH_KEYWORDS, point_doc},
  {"line", (PyCFunction)py_line,METH_VARARGS | METH_KEYWORDS, line_doc},
  {"cylinder", (PyCFunction)py_cylinder, METH_VARARGS | METH_KEYWORDS, cylinder_doc},
  {"materials",(PyCFunction)py_materials, METH_VARARGS | METH_KEYWORDS, mats_doc},
  {"material",(PyCFunction)py_material, METH_VARARGS | METH_KEYWORDS, mat_doc},
  {"color",(PyCFunction) py_color, METH_VARARGS | METH_KEYWORDS, color_doc},
  {"sphere", (PyCFunction)graphics_sphere,METH_VARARGS | METH_KEYWORDS, sphere_doc},
  {"text", (PyCFunction)graphics_text,METH_VARARGS | METH_KEYWORDS, text_doc},
  {"delete",(PyCFunction)graphics_delete, METH_VARARGS | METH_KEYWORDS, delete_doc},
  {"replace",(PyCFunction)graphics_replace, METH_VARARGS | METH_KEYWORDS, replace_doc},
  {"info",(PyCFunction)graphics_info, METH_VARARGS | METH_KEYWORDS, info_doc},
  {"listall",(PyCFunction)graphics_listall, METH_VARARGS | METH_KEYWORDS, list_doc},
  {NULL, NULL}
};

static const char graphics_moddoc[] =
"Methods for drawing graphics primitives in the render window";

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef graphicsdef = {
  PyModuleDef_HEAD_INIT,
  "graphics",
  graphics_moddoc,
  -1,
  GraphicsMethods,
};
#endif

PyObject* initgraphics(void) {
#if PY_MAJOR_VERSION >= 3
  PyObject *m = PyModule_Create(&graphicsdef);
#else
  PyObject *m = Py_InitModule3("graphics", GraphicsMethods, graphics_moddoc);
#endif
  return m;
}

