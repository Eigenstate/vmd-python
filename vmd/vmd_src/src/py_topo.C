#include "py_commands.h"
#include "VMDApp.h"
#include "Molecule.h"
#include "MoleculeList.h"
#include "DrawMolecule.h"

static const char bond_doc[] = "Get all unique bonds within a specified molecule. Optionally, can get bond\n"
"type and order by modifying the type parameter\n\n"
"Args:\n"
"    molid (int): Molecule ID to query. Defaults to top molecule\n"
"    type (bool): Whether to include bond type information in the result\n"
"        Defaults to False.\n"
"    orders (bool): Whether to include bond order information in the result\n"
"        Defaults to False.\n"
"Returns:\n"
"    (list of lists) Information about each bond in the system. Each bond\n"
"        will be a list with the indices of the two atoms in the\n"
"        bond, followed by bond type (as a string) and order (as a float) if\n"
"        requested";
static PyObject* topo_get_bond(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "type", "order", NULL};
  PyObject *returnlist = NULL, *bond = NULL;
  int b_types = 0, b_orders = 0, molid = -1;
  PyObject *obtypes = Py_False;
  int i, j, types;
  Molecule *mol;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|iOO&:topology.bonds",
                                     (char**) kwlist, &molid, &obtypes,
                                     convert_bool, &b_orders))
    return NULL;

  // Check molid is valid
  app = get_vmdapp();
  if (molid == -1)
    molid = app->molecule_top();

  if (!(mol = app->moleculeList->mol_from_id(molid))) {
    PyErr_Format(PyExc_ValueError, "invalid molid '%d'", molid);
    return NULL;
  }

  if (PyBool_Check(obtypes)) {
  b_types = (obtypes == Py_True) ? 1 : 0;

  // Handle deprecated types argument
  } else if (is_pyint(obtypes)) {

    // Check regular argument wasn't also passed
    if (kwargs && PyDict_GetItemString(kwargs, "order")) {
      PyErr_SetString(PyExc_ValueError, "Cannot specify deprecated type "
                      "argument with new or orders arguments");
      return NULL;
    }
    types = as_int(obtypes);

    if (types == 3)
      b_types = b_orders = 1;
    else if (types == 2)
      b_orders = 1;
    else if (types == 1)
      b_types = 1;

    PyErr_Warn(PyExc_DeprecationWarning, "type int keyword is now replaced by "
               "type and order booleans");
  // Handle error if it's not a boolean
  } else {
    PyErr_SetString(PyExc_TypeError, "type keyword expected a bool");
    return NULL;
  }


  // Assume no bonds, and build the list from empty to avoid traversing
  // the entire bond list twice
  if (!(returnlist = PyList_New(0)))
    goto failure;

  for (i = 0; i < mol->nAtoms - 1; i++) { // Last atom can add no new bonds.
    const MolAtom *atom = mol->atom(i);
    for (j = 0; j < atom->bonds; j++) {
      if (i < atom->bondTo[j]) {

        if (b_types && b_orders)
          bond = Py_BuildValue("iisf", i, atom->bondTo[j],
                                   mol->bondTypeNames.name(mol->getbondtype(i, j)),
                                   mol->getbondorder(i, j));
        else if (b_types)
          bond = Py_BuildValue("iis", i, atom->bondTo[j],
                                   mol->bondTypeNames.name(mol->getbondtype(i, j)));
        else if (b_orders)
          bond = Py_BuildValue("iif", i, atom->bondTo[j],
                                   mol->getbondorder(i, j));
        else
          bond = Py_BuildValue("ii", i, atom->bondTo[j]);

        // Don't leak references with PyList_Append
        PyList_Append(returnlist, bond);
        Py_XDECREF(bond);
        if (PyErr_Occurred())
          goto failure;
      }
    }
  }
  return returnlist;

failure:
  PyErr_SetString(PyExc_RuntimeError, "problem building bond list");
  Py_XDECREF(returnlist);
  return NULL;
}

static const char angle_doc[] =
"Get all unique angles within a specified molecule. Optionally, can get angle\n"
"type as well\n\n"
"Args:\n"
"    molid (int): Molecule ID to query. Defaults to top molecule\n"
"    type (bool): Whether to include angle type information in the result\n"
"        Defaults to False.\n"
"Returns:\n"
"    (list of lists) Information about each angle in the system. Each angle \n"
"        will be a list with the indices of the three atoms comprising the\n"
"        angle, followed by angle type (as a string) if requested.";
static PyObject* topo_get_angle(PyObject *self, PyObject *args,
                                PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "type", NULL};
  PyObject *returnlist = NULL;
  int molid = -1, types = 0;
  Molecule *mol;
  VMDApp *app;
  int i;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|ii:topology.angle",
                                   (char**) kwlist, &molid, &types))
        return NULL;

  app = get_vmdapp();
  if (molid == -1)
    molid = app->molecule_top();

  if (!(mol = app->moleculeList->mol_from_id(molid))) {
    PyErr_Format(PyExc_ValueError, "invalid molid '%d'", molid);
    return NULL;
  }

  if (!(returnlist = PyList_New(mol->num_angles())))
    goto failure;

  for (i=0; i<mol->num_angles(); i++) {
    PyObject* angle;
    if (types)
      angle = Py_BuildValue("iiis", mol->angles[3*i], mol->angles[3*i+1],
                            mol->angles[3*i+2],
                            mol->angleTypeNames.name(mol->get_angletype(i)));
    else
      angle = Py_BuildValue("iii", mol->angles[3*i], mol->angles[3*i+1],
                            mol->angles[3*i+2]);

    PyList_SET_ITEM(returnlist, i, angle);
    if (PyErr_Occurred())
      goto failure;
  }
  return returnlist;

failure:
  PyErr_SetString(PyExc_RuntimeError, "problem building angle list");
  Py_XDECREF(returnlist);
  return NULL;
}

static const char dihed_doc[] =
"Get all unique dihedrals within a specified molecule. Optionally, can get\n"
"dihedral type as well\n\n"
"Args:\n"
"    molid (int): Molecule ID to query. Defaults to top molecule\n"
"    type (bool): Whether to include dihedral type information in the result\n"
"        Defaults to False.\n"
"Returns:\n"
"    (list of lists) Information about each dihedral in the system. Each\n"
"        dihedral will be a list with the indices of the four atoms\n"
"        comprising the dihedral, followed by dihedral type (as a string) if\n"
"        requested.";
static PyObject* topo_get_dihed(PyObject *self, PyObject *args,
                                PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "type", NULL};
  PyObject *returnlist = NULL;
  int molid = -1, types = 0;
  Molecule *mol;
  VMDApp *app;
  int i;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|ii:topology.dihedral",
                                   (char**) kwlist, &molid, &types))
        return NULL;

  app = get_vmdapp();
  if (molid == -1)
    molid = app->molecule_top();

  if (!(mol = app->moleculeList->mol_from_id(molid))) {
    PyErr_Format(PyExc_ValueError, "invalid molid '%d'", molid);
    return NULL;
  }

  if (!(returnlist = PyList_New(mol->num_dihedrals())))
    goto failure;

  for (i = 0; i < mol->num_dihedrals(); i++) {
    PyObject* dihed;
    if (types)
      dihed = Py_BuildValue("iiiis",
                            mol->dihedrals[4*i], mol->dihedrals[4*i+1],
                            mol->dihedrals[4*i+2], mol->dihedrals[4*i+3],
                            mol->dihedralTypeNames.name(mol->get_dihedraltype(i)));
    else
      dihed = Py_BuildValue("iiii",
                            mol->dihedrals[4*i], mol->dihedrals[4*i+1],
                            mol->dihedrals[4*i+2], mol->dihedrals[4*i+3]);

    PyList_SET_ITEM(returnlist, i, dihed);
    if (PyErr_Occurred())
      goto failure;
  }
  return returnlist;

failure:
  PyErr_SetString(PyExc_RuntimeError, "Problem building dihedral list");
  Py_XDECREF(returnlist);
  return NULL;
}

static const char impropers_doc[] =
"Get all unique impropers within a specified molecule. Optionally, can get\n"
"improper type as well\n\n"
"Args:\n"
"    molid (int): Molecule ID to query. Defaults to top molecule\n"
"    type (bool): Whether to include improper type information in the result\n"
"        Defaults to False.\n"
"Returns:\n"
"    (list of lists) Information about each improper in the system. Each\n"
"        improper will be a list with the indices of the four atoms\n"
"        comprising the improper, followed by improper type (as a string) if\n"
"        requested.";
static PyObject* topo_get_impro(PyObject *self, PyObject *args,
                                PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "type", NULL};
  PyObject *returnlist = NULL;
  int molid = -1, types = 0;
  Molecule *mol;
  VMDApp *app;
  int i;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|ii:topology.impropers",
                                   (char**) kwlist, &molid, &types))
        return NULL;

  app = get_vmdapp();
  if (molid == -1)
    molid = app->molecule_top();

  if (!(mol = app->moleculeList->mol_from_id(molid))) {
    PyErr_Format(PyExc_ValueError, "invalid molid '%d'", molid);
    return NULL;
  }

  if (!(returnlist = PyList_New(mol->num_impropers())))
    goto failure;

  for (i = 0; i < mol->num_impropers(); i++) {
    PyObject* improper;
    if (types)
      improper = Py_BuildValue("iiiis",
                               mol->impropers[4*i], mol->impropers[4*i+1],
                               mol->impropers[4*i+2], mol->impropers[4*i+3],
                               mol->improperTypeNames.name(mol->get_impropertype(i)));
    else
      improper = Py_BuildValue("iiii",
                               mol->impropers[4*i], mol->impropers[4*i+1],
                               mol->impropers[4*i+2], mol->impropers[4*i+3]);
    PyList_SetItem(returnlist, i, improper);
  }
  return returnlist;

failure:
  PyErr_SetString(PyExc_RuntimeError, "Problem building improper list");
  Py_XDECREF(returnlist);
  return NULL;
}

static const char addbond_doc[] =
"Add a bond between two atoms with given indices. If bond is already present,\n"
"nothing will be done.\n\n"
"Args:\n"
"    i (int): Index of first atom\n"
"    j (int): Index of second atom\n"
"    molid (int): Molecule ID to add bond to. Defaults to top molecule.\n"
"    order (float): Bond order. Defaults to 1.0\n"
"    type (str): Bond type. Can be from output of `topology.bondtypes()`,\n"
"        or can define a new bond type. Defaults to None";
static PyObject* topo_add_bond(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"i", "j", "molid", "order", "type", NULL};
  int molid = -1, type = -1;
  char *bondtype = NULL;
  float order = 1.0;
  Molecule *mol;
  VMDApp *app;
  int i, j;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii|ifz:topology.addbond",
                                   (char**) kwlist, &i, &j, &molid, &order,
                                   &bondtype))
        return NULL;

  app = get_vmdapp();
  if (molid == -1)
    molid = app->molecule_top();

  if (!(mol = app->moleculeList->mol_from_id(molid))) {
    PyErr_Format(PyExc_ValueError, "invalid molid '%d'", molid);
    return NULL;
  }

  // Handle bond type
  if (bondtype) {
    type = mol->bondTypeNames.add_name(bondtype, 0);
    mol->set_dataset_flag(BaseMolecule::BONDTYPES);
  }

  mol->set_dataset_flag(BaseMolecule::BONDS);
  mol->set_dataset_flag(BaseMolecule::BONDORDERS);

  if (mol->add_bond_dupcheck(i, j, order, type)) {
    PyErr_Format(PyExc_RuntimeError, "Problem adding bond between atoms %d "
                 "and %d on molid %d", i, j, molid);
    return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}

static const char addangle_doc[] =
"Add an angle between three atoms with given indices. No checking for\n"
"duplicate angles is performed\n\n"
"Args:\n"
"    i (int): Index of first atom\n"
"    j (int): Index of second atom\n"
"    k (int): Index of third atom\n"
"    molid (int): Molecule ID to add angle to. Defaults to top molecule.\n"
"    type (str): Angle type. Can be from output of `topology.getangletypes()`,\n"
"        or can define a new angle type. Defaults to None\n"
"Returns:\n"
"    (int) Index of new angle in system";
static PyObject* topo_add_angle(PyObject *self, PyObject *args,
                                PyObject *kwargs)
{
  const char *kwlist[] = {"i", "j", "k", "molid", "type", NULL};
  int molid = -1, type = -1;
  char *angletype = NULL;
  Molecule *mol;
  VMDApp *app;
  int i, j, k;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iii|iz:topology.addangle",
                                   (char**) kwlist, &i, &j, &k, &molid,
                                   &angletype))
        return NULL;

  app = get_vmdapp();
  if (molid == -1)
    molid = app->molecule_top();

  if (!(mol = app->moleculeList->mol_from_id(molid))) {
    PyErr_Format(PyExc_ValueError, "invalid molid '%d'", molid);
    return NULL;
  }

  if (angletype) {
    type = mol->angleTypeNames.add_name(angletype, 0);
    // This is indirectly set in add_angle, but do it explictly to be clear
    mol->set_dataset_flag(BaseMolecule::ANGLETYPES);
  }
  mol->set_dataset_flag(BaseMolecule::ANGLES);

  return as_pyint(mol->add_angle(i, j, k, type));
}

static const char adddihed_doc[] =
"Add a dihedral between four atoms with the given indices. No checking for\n"
"duplicate dihedrals is performed.\n\n"
"Args:\n"
"    i (int): Index of first atom\n"
"    j (int): Index of second atom\n"
"    k (int): Index of third atom\n"
"    l (int): Index of fourth atom\n"
"    molid (int): Molecule ID to add dihedral in. Defaults to top molecule.\n"
"    type (str): Angle type. Can be from output of `topology.getangletypes()`,\n"
"        or can define a new angle type. Defaults to None\n"
"Returns:\n"
"    (int) New number of dihedrals defined in system";
static PyObject* topo_add_dihed(PyObject *self, PyObject *args,
                                PyObject *kwargs)
{
  const char *kwlist[] = {"i", "j", "k", "l", "molid", "type", NULL};
  int molid = -1, type = -1;
  char *dihetype = NULL;
  int i, j, k, l;
  Molecule *mol;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiii|iz:topology.adddihedral",
                                   (char**) kwlist, &i, &j, &k, &l, &molid,
                                   &dihetype))
        return NULL;

  app = get_vmdapp();
  if (molid == -1)
    molid = app->molecule_top();

  if (!(mol = app->moleculeList->mol_from_id(molid))) {
    PyErr_Format(PyExc_ValueError, "invalid molid '%d'", molid);
    return NULL;
  }

  if (dihetype) {
    type = mol->dihedralTypeNames.add_name(dihetype, 0);
    // This is indirectly set in add_dihedral, but do it explictly to be clear
    // Yes, it is supposed to be the ANGLETYPES flag
    mol->set_dataset_flag(BaseMolecule::ANGLETYPES);
  }
  mol->set_dataset_flag(BaseMolecule::ANGLES);

  return as_pyint(mol->add_dihedral(i, j, k, l, type));
}

static const char addimproper_doc[] =
"Add an improper dihedral between four atoms with the given indices. No\n"
"checking for duplicate impropers is performed.\n\n"
"Args:\n"
"    i (int): Index of first atom\n"
"    j (int): Index of second atom\n"
"    k (int): Index of third atom\n"
"    l (int): Index of fourth atom\n"
"    molid (int): Molecule ID to add dihedral in. Defaults to top molecule.\n"
"    type (str): Angle type. Can be from output of `topology.getangletypes()`,\n"
"        or can define a new angle type. Defaults to None\n"
"Returns:\n"
"    (int) New number of improper dihedrals defined in system";
static PyObject* topo_add_improp(PyObject *self, PyObject *args,
                                 PyObject *kwargs)
{
  const char *kwlist[] = {"i", "j", "k", "l", "molid", "type", NULL};
  int molid = -1, type = -1;
  char *dihetype = NULL;
  int i, j, k, l;
  Molecule *mol;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiii|iz:topology.addimproper",
                                   (char**) kwlist, &i, &j, &k, &l, &molid,
                                   &dihetype))
        return NULL;

  app = get_vmdapp();
  if (molid == -1)
    molid = app->molecule_top();

  if (!(mol = app->moleculeList->mol_from_id(molid))) {
    PyErr_Format(PyExc_ValueError, "invalid molid '%d'", molid);
    return NULL;
  }

  if (dihetype) {
    type = mol->improperTypeNames.add_name(dihetype, 0);
    // This is indirectly set in add_dihedral, but do it explictly to be clear
    // Yes, it is supposed to be the ANGLETYPES flag
    mol->set_dataset_flag(BaseMolecule::ANGLETYPES);
  }
  mol->set_dataset_flag(BaseMolecule::ANGLES);

  return as_pyint(mol->add_improper(i, j, k, l, type));
}

static const char delbond_doc[] =
"Delete a bond between atoms with the given indices. If the bond does not\n"
"exist, does nothing.\n\n"
"Args:\n"
"    i (int): Index of first atom\n"
"    j (int): Index of second atom\n"
"    molid (int): Molecule ID to delete bond from. Defaults to top molecule.\n"
"Returns:\n"
"    (bool) True if bond exists and was deleted";
static PyObject* topo_del_bond(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"i", "j", "molid", NULL};
  PyObject *result = Py_False;
  MolAtom *atomi, *atomj;
  float *bondOrders;
  int molid = -1;
  int *bondTypes;
  Molecule *mol;
  int i, j, tmp;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii|i:topology.delbond",
                                   (char**) kwlist, &i, &j, &molid))
        return NULL;

  app = get_vmdapp();
  if (molid == -1)
    molid = app->molecule_top();

  if (!(mol = app->moleculeList->mol_from_id(molid))) {
    PyErr_Format(PyExc_ValueError, "invalid molid '%d'", molid);
    return NULL;
  }

  if (i < 0 || j < 0 || i >= mol->nAtoms || j >= mol->nAtoms) {
    PyErr_Format(PyExc_ValueError, "invalid bond atom indices '%d'-'%d'", i, j);
    return NULL;
  }

  bondOrders = mol->extraflt.data("bondorders");
  bondTypes = mol->extraint.data("bondtypes");
  atomi = mol->atom(i);
  atomj = mol->atom(j);

  for (tmp = 0; tmp < atomi->bonds && j != atomi->bondTo[tmp]; tmp++);

  // Match is found
  if (tmp < atomi->bonds) {
    atomi->bondTo[tmp] = atomi->bondTo[--atomi->bonds];
    mol->set_dataset_flag(BaseMolecule::BONDS);

    if (bondOrders) {
      bondOrders[i * MAXATOMBONDS + tmp] = bondOrders[i * MAXATOMBONDS + atomi->bonds];
      bondOrders[i * MAXATOMBONDS + atomi->bonds] = 1.0;
      mol->set_dataset_flag(BaseMolecule::BONDORDERS);
    }

    if (bondTypes) {
      bondTypes[i * MAXATOMBONDS + tmp] = bondTypes[i * MAXATOMBONDS + atomi->bonds];
      bondTypes[i * MAXATOMBONDS + atomi->bonds] = -1;
      mol->set_dataset_flag(BaseMolecule::BONDTYPES);
    }

    for (tmp = 0; tmp < atomj->bonds && i != atomj->bondTo[tmp]; tmp++);
    atomj->bondTo[tmp] = atomj->bondTo[--atomj->bonds];

    result = Py_True;
  }

  Py_INCREF(result);
  return result;
}

static const char delangle_doc[] =
"Delete an angle between atoms with the given indices. If the angle does not\n"
"exist, does nothing.\n\n"
"Args:\n"
"    i (int): Index of first atom\n"
"    j (int): Index of second atom\n"
"    k (int): Index of third atom\n"
"    molid (int): Molecule ID to delete from. Defaults to top molecule.\n"
"Returns:\n"
"    (bool) True if angle exists and was deleted";
static PyObject* topo_del_angle(PyObject *self, PyObject *args,
                                PyObject *kwargs)
{
  const char *kwlist[] = {"i", "j", "k", "molid", NULL};
  PyObject *result = Py_False;
  int molid = -1;
  int i, j, k, s;
  Molecule *mol;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iii|i:topology:delangle",
                                   (char**) kwlist, &i, &j, &k, &molid))
        return NULL;

  app = get_vmdapp();
  if (molid == -1)
    molid = app->molecule_top();

  if (!(mol = app->moleculeList->mol_from_id(molid))) {
    PyErr_Format(PyExc_ValueError, "invalid molid '%d'", molid);
    return NULL;
  }

  if (i < 0 || j < 0 || k < 0 \
   || k >= mol->nAtoms || i >= mol->nAtoms || j >= mol->nAtoms) {
    PyErr_Format(PyExc_ValueError, "invalid angle atom indices %d-%d-%d",
                 i, j, k);
    return NULL;
  }

  // Reorder so i < k to follow convention for atom ordering in angles
  if (i > k) {
    s = k;
    k = i;
    i = s;
  }

  s = mol->num_angles();
  for (int tmp = 0; tmp < s; tmp++) {
    if (i == mol->angles[3*tmp] && j == mol->angles[3*tmp+1] \
     && k == mol->angles[3*tmp+2]) {
      mol->angles[3*tmp] = mol->angles[3*s-3];
      mol->angles[3*tmp+1] = mol->angles[3*s-2];
      mol->angles[3*tmp+2] = mol->angles[3*s-1];

      if (mol->angleTypes.num() > tmp) {
        if (mol->angleTypes.num() == s)
          mol->angleTypes[tmp] = mol->angleTypes[s-1];
        else
          mol->angleTypes[tmp] = -1;
      }

      mol->angles.pop();
      mol->angles.pop();
      mol->angles.pop();
      mol->set_dataset_flag(BaseMolecule::ANGLES);
      result = Py_True;
      break;
    }
  }

  Py_INCREF(result);
  return result;
}

static const char deldihed_doc[] =
"Delete a dihedral angle between atoms with the given indices. If the\n"
"dihedral does not exist, does nothing.\n\n"
"Args:\n"
"    i (int): Index of first atom\n"
"    j (int): Index of second atom\n"
"    k (int): Index of third atom\n"
"    l (int): Index of fourth atom\n"
"    molid (int): Molecule ID to delete from. Defaults to top molecule.\n"
"Returns:\n"
"    (bool) True if dihedral exists and was deleted";
static PyObject* topo_del_dihed(PyObject *self, PyObject *args,
                                PyObject *kwargs)
{
  const char *kwlist[] = {"i", "j", "k", "l", "molid", NULL};
  PyObject *result = Py_False;
  int i, j, k, l, tmp, s;
  int molid = -1;
  Molecule *mol;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiii|i:topology.deldihedral",
                                   (char**) kwlist, &i, &j, &k, &l, &molid))
        return NULL;

  app = get_vmdapp();
  if (molid == -1)
    molid = app->molecule_top();

  if (!(mol = app->moleculeList->mol_from_id(molid))) {
    PyErr_Format(PyExc_ValueError, "invalid molid '%d'", molid);
    return NULL;
  }

  if (i < 0 || j < 0 || k < 0 || l < 0 || k >= mol->nAtoms \
   || i >= mol->nAtoms || j >= mol->nAtoms || l >= mol->nAtoms) {
    PyErr_Format(PyExc_ValueError, "invalid dihedral atom indices %d-%d-%d-%d",
                 i, j, k, l);
    return NULL;
  }

  //Reorder so i < l as is convention for listing dihedrals
  if (i > l) {
    tmp = l;
    l = i;
    i = tmp;
    tmp = k;
    k = j;
    j = tmp;
  }

  s = mol->num_dihedrals();
  for (tmp = 0; tmp < s; tmp++) {
    if (i == mol->dihedrals[4*tmp] && j == mol->dihedrals[4*tmp+1] \
     && k == mol->dihedrals[4*tmp+2] && l == mol->dihedrals[4*tmp+3]) {

      mol->dihedrals[4*tmp] = mol->dihedrals[4*s-4];
      mol->dihedrals[4*tmp+1] = mol->dihedrals[4*s-3];
      mol->dihedrals[4*tmp+2] = mol->dihedrals[4*s-2];
      mol->dihedrals[4*tmp+3] = mol->dihedrals[4*s-1];
      if (mol->dihedralTypes.num() > tmp) {
        if (mol->dihedralTypes.num() == s)
          mol->dihedralTypes[tmp] = mol->dihedralTypes[s-1];
        else
          mol->dihedralTypes[tmp] = -1;
      }
      mol->dihedrals.pop();
      mol->dihedrals.pop();
      mol->dihedrals.pop();
      mol->dihedrals.pop();
      mol->set_dataset_flag(BaseMolecule::ANGLES);

      result = Py_True;
      break;
    }
  }

  Py_INCREF(result);
  return result;
}

static const char delimproper_doc[] =
"Delete an improper dihedral angle between atoms with the given indices. If\n"
"the improper does not exist, does nothing.\n\n"
"Args:\n"
"    i (int): Index of first atom\n"
"    j (int): Index of second atom\n"
"    k (int): Index of third atom\n"
"    l (int): Index of fourth atom\n"
"    molid (int): Molecule ID to delete from. Defaults to top molecule.\n"
"Returns:\n"
"    (bool) True if improper dihedral exists and was deleted";
static PyObject* topo_del_improper(PyObject *self, PyObject *args,
                                   PyObject *kwargs)
{
  const char *kwlist[] = {"i", "j", "k", "l", "molid", NULL};
  PyObject *result = Py_False;
  int i, j, k, l, tmp, s;
  int molid = -1;
  Molecule *mol;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiii|i:topology.delimproper",
                                   (char**) kwlist, &i, &j, &k, &l, &molid))
        return NULL;

  app = get_vmdapp();
  if (molid == -1)
    molid = app->molecule_top();

  if (!(mol = app->moleculeList->mol_from_id(molid))) {
    PyErr_Format(PyExc_ValueError, "invalid molid '%d'", molid);
    return NULL;
  }

  if (i < 0 || j < 0 || k < 0 || l < 0 || k >= mol->nAtoms \
   || i >= mol->nAtoms || j >= mol->nAtoms || l >= mol->nAtoms) {

    PyErr_Format(PyExc_ValueError, "invalid imporper atom indices %d-%d-%d-%d",
                 i, j, k, l);
    return NULL;
  }

  //Reorder so i < l for convention with dihedral order
  if (i > l) {
    tmp = l;
    l = i;
    i = tmp;
    tmp = k;
    k = j;
    j = tmp;
  }

  s = mol->num_impropers();
  for (tmp = 0; tmp < s; tmp++) {
    if (i == mol->impropers[4*tmp] && j == mol->impropers[4*tmp+1] \
     && k == mol->impropers[4*tmp+2] && l == mol->impropers[4*tmp+3]) {

      mol->impropers[4*tmp] = mol->impropers[4*s-4];
      mol->impropers[4*tmp+1] = mol->impropers[4*s-3];
      mol->impropers[4*tmp+2] = mol->impropers[4*s-2];
      mol->impropers[4*tmp+3] = mol->impropers[4*s-1];

      if (mol->improperTypes.num() > tmp) {
        if (mol->improperTypes.num() == s)
          mol->improperTypes[tmp] = mol->improperTypes[s-1];
        else
          mol->improperTypes[tmp] = -1;
      }
      mol->impropers.pop();
      mol->impropers.pop();
      mol->impropers.pop();
      mol->impropers.pop();
      mol->set_dataset_flag(BaseMolecule::ANGLES);

      result = Py_True;
      break;
    }
  }

  Py_INCREF(result);
  return result;
}

static const char del_all_bond_doc[] =
"Delete all bonds in a given molecule\n\n"
"Args:\n"
"    molid (int): Molecule ID to delete bonds in. Defaults to top molecule.\n"
"Returns:\n"
"    (int) Number of bonds deleted";
static PyObject* topo_del_all_bonds(PyObject *self, PyObject *args,
                                    PyObject *kwargs)
{
  const char *kwlist[] = {"molid", NULL};
  int molid = -1;
  Molecule *mol;
  VMDApp *app;
  int retval;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|i:topology.delallbonds",
                                   (char**) kwlist, &molid))
        return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (molid == -1)
    molid = app->molecule_top();

  if (!(mol = app->moleculeList->mol_from_id(molid))) {
    PyErr_Format(PyExc_ValueError, "invalid molid '%d'", molid);
    return NULL;
  }

  retval = mol->count_bonds();
  mol->clear_bonds();

  return as_pyint(retval);
}

static const char del_all_angle_doc[] =
"Delete all angles in a given molecule\n\n"
"Args:\n"
"    molid (int): Molecule ID to delete angles in. Defaults to top molecule.\n"
"Returns:\n"
"    (int) Number of angles deleted";
static PyObject* topo_del_all_angles(PyObject *self, PyObject *args,
                                     PyObject *kwargs) {
  const char *kwlist[] = {"molid", NULL};
  int molid = -1;
  Molecule *mol;
  VMDApp *app;
  int retval;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|i:topology.delallangles",
                                   (char**) kwlist, &molid))
        return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (molid == -1)
    molid = app->molecule_top();

  if (!(mol = app->moleculeList->mol_from_id(molid))) {
    PyErr_Format(PyExc_ValueError, "invalid molid '%d'", molid);
    return NULL;
  }

  retval = mol->num_angles();
  mol->angles.clear();
  mol->angleTypes.clear();

  return as_pyint(retval);
}

static const char del_all_dihed_doc[] =
"Delete all dihedrals in a given molecule\n\n"
"Args:\n"
"    molid  (int): Molecule ID to delete in. Defaults to top molecule\n"
"Returns:\n"
"    (int) Number of dihedrals deleted";
static PyObject* topo_del_all_dihed(PyObject *self, PyObject *args,
                                    PyObject *kwargs)
{
  const char *kwlist[] = {"molid", NULL};
  int molid = -1;
  Molecule *mol;
  VMDApp *app;
  int retval;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|i:topology.delalldihedrals",
                                   (char**) kwlist, &molid))
        return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (molid == -1)
    molid = app->molecule_top();

  if (!(mol = app->moleculeList->mol_from_id(molid))) {
    PyErr_Format(PyExc_ValueError, "invalid molid '%d'", molid);
    return NULL;
  }

  retval = mol->num_dihedrals();
  mol->dihedrals.clear();
  mol->dihedralTypes.clear();

  return as_pyint(retval);
}

static const char del_all_improper_doc[] =
"Delete all improper dihedrals in a given molecule\n\n"
"Args:\n"
"    molid  (int): Molecule ID to delete in. Defaults to top molecule\n"
"Returns:\n"
"    (int) Number of impropers deleted";
static PyObject* topo_del_all_impropers(PyObject *self, PyObject *args,
                                        PyObject *kwargs)
{
  const char *kwlist[] = {"molid", NULL};
  int molid = -1;
  Molecule *mol;
  VMDApp *app;
  int retval;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|i:topology.delallimpropers",
                                   (char**) kwlist, &molid))
        return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (molid == -1)
    molid = app->molecule_top();

  if (!(mol = app->moleculeList->mol_from_id(molid))) {
    PyErr_Format(PyExc_ValueError, "invalid molid '%d'", molid);
    return NULL;
  }

  retval = mol->num_impropers();
  mol->impropers.clear();
  mol->improperTypes.clear();

  return as_pyint(retval);
}

static const char btypes_doc[] =
"Get all bond types defined in the molecule. If molecule does not have\n"
"bond types, will return an empty list.\n\n"
"Args:\n"
"    molid (int): Molecule ID to query. Defaults to top molecule\n"
"Returns:\n"
"    (list of str): Bond types in molecule";
static PyObject* topo_bondtypes(PyObject *self, PyObject *args,
                                PyObject *kwargs)
{
  const char *kwlist[] = {"molid", NULL};
  PyObject *returnlist = NULL;
  int molid = -1;
  Molecule *mol;
  VMDApp *app;
  int i;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|i:topology.bondtypes",
                                   (char**) kwlist, &molid))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (molid == -1)
    molid = app->molecule_top();

  if (!(mol = app->moleculeList->mol_from_id(molid))) {
    PyErr_Format(PyExc_ValueError, "Invalid molecule id '%d'", molid);
    return NULL;
  }

  if (!(returnlist = PyList_New(mol->bondTypeNames.num())))
    goto failure;

  for (i = 0; i < mol->bondTypeNames.num(); i++) {
    PyList_SET_ITEM(returnlist, i, as_pystring(mol->bondTypeNames.name(i)));
    if (PyErr_Occurred())
      goto failure;
  }

  return returnlist;

failure:
  PyErr_Format(PyExc_RuntimeError, "Problem building bond type list for molid "
               "%d", molid);
  Py_XDECREF(returnlist);
  return NULL;
}

static const char atypes_doc[] =
"Get all angle types defined in the molecule. If molecule does not have angle\n"
"types, will return an empty list.\n\n"
"Args:\n"
"    molid (int): Molecule ID to query. Defaults to top molecule\n"
"Returns:\n"
"    (list of str): Angle types in molecule";
static PyObject* topo_angletypes(PyObject *self, PyObject *args,
                                PyObject *kwargs)
{
  const char *kwlist[] = {"molid", NULL};
  PyObject *returnlist = NULL;
  int molid = -1;
  Molecule *mol;
  VMDApp *app;
  int i;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|i:topology.angletypes",
                                   (char**) kwlist, &molid))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (molid == -1)
    molid = app->molecule_top();

  if (!(mol = app->moleculeList->mol_from_id(molid))) {
    PyErr_Format(PyExc_ValueError, "Invalid molecule id '%d'", molid);
    return NULL;
  }

  if (!(returnlist = PyList_New(mol->angleTypeNames.num())))
    goto failure;

  for (i = 0; i < mol->angleTypeNames.num(); i++) {
    PyList_SET_ITEM(returnlist, i, as_pystring(mol->angleTypeNames.name(i)));
    if (PyErr_Occurred())
      goto failure;
  }

  return returnlist;

failure:
  PyErr_Format(PyExc_RuntimeError, "Problem building angle type list for molid "
               "%d", molid);
  Py_XDECREF(returnlist);
  return NULL;
}

static const char dtypes_doc[] =
"Get all dihedral types defined in the molecule. If the molecule does not\n"
"have dihedral types, will return an empty list.\n\n"
"Args:\n"
"    molid (int): Molecule ID to query. Defaults to top molecule\n"
"Returns:\n"
"    (list of str): Dihedral types in molecule";
static PyObject* topo_dihetypes(PyObject *self, PyObject *args,
                                 PyObject *kwargs)
{
  const char *kwlist[] = {"molid", NULL};
  PyObject *returnlist = NULL;
  int molid = -1;
  Molecule *mol;
  VMDApp *app;
  int i;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|i:topology.dihedraltypes",
                                   (char**) kwlist, &molid))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (molid == -1)
    molid = app->molecule_top();

  if (!(mol = app->moleculeList->mol_from_id(molid))) {
    PyErr_Format(PyExc_ValueError, "Invalid molecule id '%d'", molid);
    return NULL;
  }

  if (!(returnlist = PyList_New(mol->dihedralTypeNames.num())))
    goto failure;

  for (i = 0; i < mol->dihedralTypeNames.num(); i++) {
    PyList_SET_ITEM(returnlist, i, as_pystring(mol->dihedralTypeNames.name(i)));
    if (PyErr_Occurred())
      goto failure;
  }

  return returnlist;

failure:
  PyErr_Format(PyExc_RuntimeError, "Problem building dihedral type list for "
               "molid %d", molid);
  Py_XDECREF(returnlist);
  return NULL;
}

static const char itypes_doc[] =
"Get all improper dihedral types defined in the molecule. If the molecule\n"
"not have improper types, will return an empty list.\n\n"
"Args:\n"
"    molid (int): Molecule ID to query. Defaults to top molecule\n"
"Returns:\n"
"    (list of str): Improper dihedral types in molecule";
static PyObject* topo_imptypes(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", NULL};
  PyObject *returnlist = NULL;
  int molid = -1;
  Molecule *mol;
  VMDApp *app;
  int i;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|i:topology.impropertypes",
                                   (char**) kwlist, &molid))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (molid == -1)
    molid = app->molecule_top();

  if (!(mol = app->moleculeList->mol_from_id(molid))) {
    PyErr_Format(PyExc_ValueError, "Invalid molecule id '%d'", molid);
    return NULL;
  }

  if (!(returnlist = PyList_New(mol->improperTypeNames.num())))
    goto failure;

  for (i = 0; i < mol->improperTypeNames.num(); i++) {
    PyList_SET_ITEM(returnlist, i, as_pystring(mol->improperTypeNames.name(i)));
    if (PyErr_Occurred())
      goto failure;
  }

  return returnlist;

failure:
  PyErr_Format(PyExc_RuntimeError, "Problem building improper dihedral type "
               "list for molid %d", molid);
  Py_XDECREF(returnlist);
  return NULL;
}


static PyMethodDef methods[] = {
  {"bonds", (PyCFunction)topo_get_bond, METH_VARARGS | METH_KEYWORDS, bond_doc},
  {"angles", (PyCFunction)topo_get_angle, METH_VARARGS | METH_KEYWORDS, angle_doc},
  {"dihedrals", (PyCFunction)topo_get_dihed, METH_VARARGS | METH_KEYWORDS, dihed_doc},
  {"impropers", (PyCFunction)topo_get_impro, METH_VARARGS | METH_KEYWORDS, impropers_doc},
  {"addbond", (PyCFunction)topo_add_bond, METH_VARARGS | METH_KEYWORDS, addbond_doc},
  {"addangle", (PyCFunction)topo_add_angle, METH_VARARGS | METH_KEYWORDS, addangle_doc},
  {"adddihedral", (PyCFunction)topo_add_dihed, METH_VARARGS | METH_KEYWORDS, adddihed_doc},
  {"addimproper", (PyCFunction)topo_add_improp, METH_VARARGS | METH_KEYWORDS, addimproper_doc},
  {"delbond", (PyCFunction)topo_del_bond, METH_VARARGS | METH_KEYWORDS, delbond_doc},
  {"delangle", (PyCFunction)topo_del_angle, METH_VARARGS | METH_KEYWORDS, delangle_doc},
  {"deldihedral", (PyCFunction)topo_del_dihed, METH_VARARGS | METH_KEYWORDS, deldihed_doc},
  {"delimproper", (PyCFunction)topo_del_improper, METH_VARARGS | METH_KEYWORDS, delimproper_doc},
  {"delallbonds", (PyCFunction)topo_del_all_bonds, METH_VARARGS | METH_KEYWORDS, del_all_bond_doc},
  {"delallangles", (PyCFunction)topo_del_all_angles, METH_VARARGS | METH_KEYWORDS, del_all_angle_doc},
  {"delalldihedrals", (PyCFunction)topo_del_all_dihed, METH_VARARGS | METH_KEYWORDS, del_all_dihed_doc},
  {"delallimpropers", (PyCFunction)topo_del_all_impropers, METH_VARARGS | METH_KEYWORDS, del_all_improper_doc},
  {"bondtypes", (PyCFunction)topo_bondtypes, METH_VARARGS | METH_KEYWORDS, btypes_doc},
  {"angletypes", (PyCFunction)topo_angletypes, METH_VARARGS | METH_KEYWORDS, atypes_doc},
  {"dihedraltypes", (PyCFunction)topo_dihetypes, METH_VARARGS | METH_KEYWORDS, dtypes_doc},
  {"impropertypes", (PyCFunction)topo_imptypes, METH_VARARGS | METH_KEYWORDS, itypes_doc},
  {NULL, NULL}
};

static const char topo_moddoc[] =
"Methods for querying or modifying the topology of a molecule, which consists "
"of all defined bonds, angles, dihedrals, and impropers, for applicable "
"topology file formats.";

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef topologydef = {
  PyModuleDef_HEAD_INIT,
  "topology",
  topo_moddoc,
  -1,
  methods,
};
#endif

PyObject* inittopology(void) {
#if PY_MAJOR_VERSION >= 3
  PyObject *module = PyModule_Create(&topologydef);
#else
  PyObject *module = Py_InitModule3("topology", methods, topo_moddoc);
#endif
  return module;
}
