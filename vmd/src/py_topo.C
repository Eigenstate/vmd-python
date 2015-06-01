#include "py_commands.h"
#include "VMDApp.h"
#include "Molecule.h"
#include "MoleculeList.h"
//#include "Measure.h"
#include "DrawMolecule.h"
static char* bond_doc = (char *)
"bond([molid=top, type=0]) -> list of bonds\n"
"	Returns all unique bonds within the structure of the specified molid.\n"
"	Each bond will be its own 2-element list within the list. Optionally,\n"
"	the bond type and order can be returned by modifying the type parameter.\n"
"	0=bond only, 1 adds bond type information, 2 adds bond order, and 3 adds both.\n";
static PyObject* topo_get_bond(PyObject *self, PyObject *args, PyObject *kwds) {
	int molid = -1, i, j, types = 0;
	static char *kwlist[] = { (char *)"molid", (char *)"type", NULL };
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ii", kwlist, &molid, &types))
        return NULL;
	VMDApp *app = get_vmdapp();
	if (molid == -1)
		molid = app->molecule_top();
	Molecule *mol = app->moleculeList->mol_from_id(molid);
	if (!mol) {
		PyErr_SetString(PyExc_ValueError, "invalid molid");
		PyErr_Print();
		return Py_BuildValue("i", -1);
	}
	//I have no idea how big the return list should be.
	//Technically a mol->count_bonds() would work, but that actually cause me to traverse the whole bond list twice. That's stupid. :(
	//Assume no bonds, and build the list from empty.
	PyObject* returnlist = PyList_New(0);
	for (i = 0; i < mol->nAtoms - 1; i++) { //Last atom can add no new bonds.
		const MolAtom *atom = mol->atom(i);
		for (j = 0; j < atom->bonds; j++) {
			if (i < atom->bondTo[j]) {
				PyObject* bondpair;
				switch (types) {
					case 1: //Types only.
					
					//mol->bondTypeNames.name(mol->getbondtype(i, j))
					bondpair = Py_BuildValue("iis", i, atom->bondTo[j], mol->bondTypeNames.name(mol->getbondtype(i, j)));
					break;
					case 2: //Orders only
					bondpair = Py_BuildValue("iif", i, atom->bondTo[j], mol->getbondorder(i, j));
					break;
					case 3: //Both
					bondpair = Py_BuildValue("iisf", i, atom->bondTo[j], mol->bondTypeNames.name(mol->getbondtype(i, j)), mol->getbondorder(i, j));
					break;
					default:
					bondpair = Py_BuildValue("ii", i, atom->bondTo[j]);
				}
				if (PyList_Append(returnlist, bondpair)) {// Returns -1 on failure, 0 on success
					PyErr_SetString(PyExc_Exception, "Failed to append list");
					PyErr_Print();
					return Py_BuildValue("i", -1);
				}
				Py_DECREF(bondpair);
			}
		}
	}
	return returnlist;
}
static char* angle_doc = (char *)
"angles([molid=top, type=0]) -> list of angles\n"
"	Returns all unique angles within the structure of the specified molid.\n"
"	Each angle will be its own 3-element list within the list. Optionally,\n"
"	the angle type can be returned by modifying the type parameter.\n"
"	0=angle only, 1 adds angle type information.\n";
static PyObject* topo_get_angle(PyObject *self, PyObject *args, PyObject *kwds) {
	int molid = -1, i, types = 0;
	static char *kwlist[] = { (char *)"molid", (char *)"type", NULL };
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ii", kwlist, &molid, &types))
        return NULL;
	VMDApp *app = get_vmdapp();
	if (molid == -1)
		molid = app->molecule_top();
	Molecule *mol = app->moleculeList->mol_from_id(molid);
	if (!mol) {
		PyErr_SetString(PyExc_ValueError, "invalid molid");
		PyErr_Print();
		return Py_BuildValue("i", -1);
	}
	PyObject* returnlist = PyList_New(mol->num_angles());
	for (i=0; i<mol->num_angles(); i++) {
		PyObject* angle;
		if (types)
			angle = Py_BuildValue("iiis", mol->angles[3*i], mol->angles[3*i+1], mol->angles[3*i+2], mol->angleTypeNames.name(mol->get_angletype(i)));
		else
			angle = Py_BuildValue("iii", mol->angles[3*i], mol->angles[3*i+1], mol->angles[3*i+2]);
		PyList_SetItem(returnlist, i, angle);
	}
	return returnlist;
}
static char* dihed_doc = (char *)
"dihedrals([molid=top, type=0]) -> list of dihedrals\n"
"	Returns all unique dihedrals within the structure of the specified molid.\n"
"	Each dihedral will be its own 4-element list within the list. Optionally,\n"
"	the dihedral type can be returned by modifying the type parameter.\n"
"	0=angle only, 1 adds dihedral type information.\n";
static PyObject* topo_get_dihed(PyObject *self, PyObject *args, PyObject *kwds) {
	int molid = -1, i, types = 0;
	static char *kwlist[] = { (char *)"molid", (char *)"type", NULL };
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ii", kwlist, &molid, &types))
        return NULL;
	VMDApp *app = get_vmdapp();
	if (molid == -1)
		molid = app->molecule_top();
	Molecule *mol = app->moleculeList->mol_from_id(molid);
	if (!mol) {
		PyErr_SetString(PyExc_ValueError, "invalid molid");
		PyErr_Print();
		return Py_BuildValue("i", -1);
	}
	PyObject* returnlist = PyList_New(mol->num_dihedrals());
	for (i=0; i<mol->num_dihedrals(); i++) {
		PyObject* dihed;
		if (types)
			dihed = Py_BuildValue("iiiis", mol->dihedrals[4*i], mol->dihedrals[4*i+1], mol->dihedrals[4*i+2], mol->dihedrals[4*i+3], mol->dihedralTypeNames.name(mol->get_dihedraltype(i)));
		else
			dihed = Py_BuildValue("iiii", mol->dihedrals[4*i], mol->dihedrals[4*i+1], mol->dihedrals[4*i+2], mol->dihedrals[4*i+3]);
		PyList_SetItem(returnlist, i, dihed);
	}
	return returnlist;
}
static char* impropers_doc = (char *)
"impropers([molid=top, type=0]) -> list of impropers\n"
"	Returns all unique impropers within the structure of the specified molid.\n"
"	Each improper will be its own 4-element list within the list. Optionally,\n"
"	the improper type can be returned by modifying the type parameter.\n"
"	0=angle only, 1 adds improper type information.\n";
static PyObject* topo_get_impro(PyObject *self, PyObject *args, PyObject *kwds) {
	int molid = -1, i, types = 0;
	static char *kwlist[] = { (char *)"molid", (char *)"type", NULL };
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ii", kwlist, &molid, &types))
        return NULL;
	VMDApp *app = get_vmdapp();
	if (molid == -1)
		molid = app->molecule_top();
	Molecule *mol = app->moleculeList->mol_from_id(molid);
	if (!mol) {
		PyErr_SetString(PyExc_ValueError, "invalid molid");
		PyErr_Print();
		return Py_BuildValue("i", -1);
	}
	PyObject* returnlist = PyList_New(mol->num_impropers());
	for (i=0; i<mol->num_impropers(); i++) {
		PyObject* improper;
		if (types)
			improper = Py_BuildValue("iiiis", mol->impropers[4*i], mol->impropers[4*i+1], mol->impropers[4*i+2], mol->impropers[4*i+3], mol->improperTypeNames.name(mol->get_impropertype(i)));
		else
			improper = Py_BuildValue("iiii", mol->impropers[4*i], mol->impropers[4*i+1], mol->impropers[4*i+2], mol->impropers[4*i+3]);
		PyList_SetItem(returnlist, i, improper);
	}
	return returnlist;
}
static char* addbond_doc = (char *)
"addbond(i, j [,molid=top, order=1.0, type=None]) -> 0 on success,-1 on failure.\n"
"	Adds one bond between the atoms at index i and j of the molecule given by molid,\n"
"	with bondorder order and bondtype type. Bonds cannot be added twice, and will be ignored if present.\n";
static PyObject* topo_add_bond(PyObject *self, PyObject *args, PyObject *kwds) {
	int molid = -1, i, j, type = -1;
	float order = 1.0;
	char* tmp = NULL;
	static char *kwlist[] = {(char*)"i", (char*)"j", (char*)"molid", (char*)"order", (char*)"type", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "ii|ifs", kwlist, &i, &j, &molid, &order, &tmp))
        return NULL;
    VMDApp *app = get_vmdapp();
	if (molid == -1)
		molid = app->molecule_top();
	Molecule *mol = app->moleculeList->mol_from_id(molid);
	if (!mol) {
		PyErr_SetString(PyExc_ValueError, "invalid molid");
		PyErr_Print();
		return Py_BuildValue("i", -1);
	}
    if (tmp != NULL) {//Type was set
    	type = mol->bondTypeNames.add_name(tmp, 0);
    	mol->set_dataset_flag(BaseMolecule::BONDTYPES);
    }
    mol->set_dataset_flag(BaseMolecule::BONDS);
    mol->set_dataset_flag(BaseMolecule::BONDORDERS);
    
    return Py_BuildValue("i", mol->add_bond_dupcheck(i, j, order, type));
}
static char* addangle_doc = (char *)
"addangle(i, j, k [,molid=top, type=None]) -> the index of the added angle.\n"
"	Adds one angle between the atoms at index i, j and k of the molecule given by molid,\n"
"	with angletype type. Duplicates are not checked!!!\n";
static PyObject* topo_add_angle(PyObject *self, PyObject *args, PyObject *kwds) {
	int molid = -1, i, j, k, type = -1;
	char* tmp = NULL;
	static char *kwlist[] = {(char*)"i", (char*)"j", (char*)"k", (char*)"molid", (char*)"type", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "iii|is", kwlist, &i, &j, &k, &molid, &tmp))
        return NULL;
    VMDApp *app = get_vmdapp();
	if (molid == -1)
		molid = app->molecule_top();
	Molecule *mol = app->moleculeList->mol_from_id(molid);
	if (!mol) {
		PyErr_SetString(PyExc_ValueError, "invalid molid");
		PyErr_Print();
		return Py_BuildValue("i", -1);
	}
    if (tmp != NULL) {//Type was set
    	type = mol->angleTypeNames.add_name(tmp, 0);
    	//Set by add_angle (indirectly)
    	//mol->set_dataset_flag(BaseMolecule::ANGLETYPES);
    }
    mol->set_dataset_flag(BaseMolecule::ANGLES);
    return Py_BuildValue("i", mol->add_angle(i, j, k, type));
}
static char* adddihed_doc = (char *)
"adddihedral(i, j, k, l [,molid=top, type=None]) -> the index of the added dihedral.\n"
"	Adds one dihedral between the atoms at index i, j, k, and l of the molecule given by molid,\n"
"	with dihedraltype type. Duplicates are not checked!!!\n";
static PyObject* topo_add_dihed(PyObject *self, PyObject *args, PyObject *kwds) {
	int molid = -1, i, j, k, l, type = -1;
	char* tmp = NULL;
	static char *kwlist[] = {(char*)"i", (char*)"j", (char*)"k", (char*)"l", (char*)"molid", (char*)"type", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiii|is", kwlist, &i, &j, &k, &l, &molid, &tmp))
        return NULL;
    VMDApp *app = get_vmdapp();
	if (molid == -1)
		molid = app->molecule_top();
	Molecule *mol = app->moleculeList->mol_from_id(molid);
	if (!mol) {
		PyErr_SetString(PyExc_ValueError, "invalid molid");
		PyErr_Print();
		return Py_BuildValue("i", -1);
	}
    if (tmp != NULL) {//Type was set
    	type = mol->dihedralTypeNames.add_name(tmp, 0);
    	//Set by add_dihedral
    	//mol->set_dataset_flag(BaseMolecule::ANGLETYPES);
    }
    mol->set_dataset_flag(BaseMolecule::ANGLES);
    return Py_BuildValue("i", mol->add_dihedral(i, j, k, l, type));
}
static char* addimproper_doc = (char *)
"addimproper(i, j, k, l [,molid=top, type=None]) -> the index of the added improper.\n"
"	Adds one improper between the atoms at index i, j, k, and l of the molecule given by molid,\n"
"	with impropertype type. Duplicates are not checked!!!\n";
static PyObject* topo_add_improp(PyObject *self, PyObject *args, PyObject *kwds) {
	int molid = -1, i, j, k, l, type = -1;
	char* tmp = NULL;
	static char *kwlist[] = {(char*)"i", (char*)"j", (char*)"k", (char*)"l", (char*)"molid", (char*)"type", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiii|is", kwlist, &i, &j, &k, &l, &molid, &tmp))
        return NULL;
    VMDApp *app = get_vmdapp();
	if (molid == -1)
		molid = app->molecule_top();
	Molecule *mol = app->moleculeList->mol_from_id(molid);
	if (!mol) {
		PyErr_SetString(PyExc_ValueError, "invalid molid");
		PyErr_Print();
		return Py_BuildValue("i", -1);
	}
    if (tmp != NULL) {//Type was set
    	type = mol->improperTypeNames.add_name(tmp, 0);
    	//Set by add_dihedral
    	//mol->set_dataset_flag(BaseMolecule::ANGLETYPES);
    }
    mol->set_dataset_flag(BaseMolecule::ANGLES);
    return Py_BuildValue("i", mol->add_improper(i, j, k, l, type));
}

static char* delbond_doc = (char *)
"delbond(i, j [,molid=top]) -> number of bonds deleted.\n"
"	Deletes the bond between the atoms at index i and j of the specified molid,\n"
"	assuming the bond exists.\n";
static PyObject* topo_del_bond(PyObject *self, PyObject *args, PyObject *kwds) {
	int molid = -1, i, j, tmp;
	static char *kwlist[] = {(char*)"i", (char*)"j", (char*)"molid", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "ii|i", kwlist, &i, &j, &molid))
        return NULL;
    VMDApp *app = get_vmdapp();
	if (molid == -1)
		molid = app->molecule_top();
	Molecule *mol = app->moleculeList->mol_from_id(molid);
	if (!mol) {
		PyErr_SetString(PyExc_ValueError, "invalid molid");
		PyErr_Print();
		return Py_BuildValue("i", -1);
	}
	if (i < 0 || j < 0 || i >= mol->nAtoms || j >= mol->nAtoms) {
		PyErr_SetString(PyExc_ValueError, "invalid index");
		PyErr_Print();
		return Py_BuildValue("i", -1);
	}
	float *bondOrders = mol->extraflt.data("bondorders");
	int *bondTypes = mol->extraint.data("bondtypes");
	MolAtom* atomi = mol->atom(i);
	MolAtom* atomj = mol->atom(j);
	for (tmp = 0; tmp < atomi->bonds && j != atomi->bondTo[tmp]; tmp++);
	if (tmp < atomi->bonds) {//We found a match!
		atomi->bondTo[tmp] = atomi->bondTo[--atomi->bonds];
		mol->set_dataset_flag(BaseMolecule::BONDS);
		if (bondOrders != NULL) {
			bondOrders[i * MAXATOMBONDS + tmp] = bondOrders[i * MAXATOMBONDS + atomi->bonds];
			bondOrders[i * MAXATOMBONDS + atomi->bonds] = 1.0;
			mol->set_dataset_flag(BaseMolecule::BONDORDERS);
		}
		if (bondTypes != NULL) {
			bondTypes[i * MAXATOMBONDS + tmp] = bondTypes[i * MAXATOMBONDS + atomi->bonds];
			bondTypes[i * MAXATOMBONDS + atomi->bonds] = -1;
			mol->set_dataset_flag(BaseMolecule::BONDTYPES);
		}
		for (tmp = 0; tmp < atomj->bonds && i != atomj->bondTo[tmp]; tmp++);
		atomj->bondTo[tmp] = atomj->bondTo[--atomj->bonds];
		
		return Py_BuildValue("i", 1);
	}
	return Py_BuildValue("i", 0);
}
static char* delangle_doc = (char *)
"delangle(i, j, k [,molid=top]) -> number of angles deleted.\n"
"	Deletes the angle specified between the atoms at index i, j, k of the specified molid,\n"
"	assuming such a term exists exists.\n";
static PyObject* topo_del_angle(PyObject *self, PyObject *args, PyObject *kwds) {
	int molid = -1, i, j, k, tmp, s;
	static char *kwlist[] = {(char*)"i", (char*)"j", (char*)"k", (char*)"molid", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "iii|i", kwlist, &i, &j, &k, &molid))
        return NULL;
    VMDApp *app = get_vmdapp();
	if (molid == -1)
		molid = app->molecule_top();
	Molecule *mol = app->moleculeList->mol_from_id(molid);
	if (!mol) {
		PyErr_SetString(PyExc_ValueError, "invalid molid");
		PyErr_Print();
		return Py_BuildValue("i", -1);
	}
	if (i < 0 || j < 0 || k < 0 || k >= mol->nAtoms || i >= mol->nAtoms || j >= mol->nAtoms) {
		PyErr_SetString(PyExc_ValueError, "invalid index");
		PyErr_Print();
		return Py_BuildValue("i", -1);
	}
	//Reorder so i < k.
	if (i > k) {
		tmp = k;
		k = i;
		i = tmp;
	}
	s = mol->num_angles();
	for (tmp = 0; tmp < s; tmp++) {
		if (i == mol->angles[3*tmp] && j == mol->angles[3*tmp+1] && k == mol->angles[3*tmp+2]) {
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
			return Py_BuildValue("i", 1);
		}
	}
	return Py_BuildValue("i", 0);
}
static char* deldihed_doc = (char *)
"deldihedral(i, j, k, l [,molid=top]) -> number of dihedrals deleted.\n"
"	Deletes the dihedral specified between the atoms at index i, j, k, and l of the specified molid,\n"
"	assuming such a term exists exists.\n";
static PyObject* topo_del_dihed(PyObject *self, PyObject *args, PyObject *kwds) {
	int molid = -1, i, j, k, l, tmp, s;
	static char *kwlist[] = {(char*)"i", (char*)"j", (char*)"k", (char*)"l", (char*)"molid", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiii|i", kwlist, &i, &j, &k, &l, &molid))
        return NULL;
    VMDApp *app = get_vmdapp();
	if (molid == -1)
		molid = app->molecule_top();
	Molecule *mol = app->moleculeList->mol_from_id(molid);
	if (!mol) {
		PyErr_SetString(PyExc_ValueError, "invalid molid");
		PyErr_Print();
		return Py_BuildValue("i", -1);
	}
	if (i < 0 || j < 0 || k < 0 || l < 0 || k >= mol->nAtoms || i >= mol->nAtoms || j >= mol->nAtoms || l >= mol->nAtoms) {
		PyErr_SetString(PyExc_ValueError, "invalid index");
		PyErr_Print();
		return Py_BuildValue("i", -1);
	}
	//Reorder so i < l.
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
		if (i == mol->dihedrals[4*tmp] && j == mol->dihedrals[4*tmp+1] && k == mol->dihedrals[4*tmp+2] && l == mol->dihedrals[4*tmp+3]) {
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
			return Py_BuildValue("i", 1);
		}
	}
	return Py_BuildValue("i", 0);
}
static char* delimproper_doc = (char *)
"delimproper(i, j, k, l [,molid=top]) -> number of impropers deleted.\n"
"	Deletes the improper specified between the atoms at index i, j, k, and l of the specified molid,\n"
"	assuming such a term exists exists.\n";
static PyObject* topo_del_improper(PyObject *self, PyObject *args, PyObject *kwds) {
	int molid = -1, i, j, k, l, tmp, s;
	static char *kwlist[] = {(char*)"i", (char*)"j", (char*)"k", (char*)"l", (char*)"molid", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiii|i", kwlist, &i, &j, &k, &l, &molid))
        return NULL;
    VMDApp *app = get_vmdapp();
	if (molid == -1)
		molid = app->molecule_top();
	Molecule *mol = app->moleculeList->mol_from_id(molid);
	if (!mol) {
		PyErr_SetString(PyExc_ValueError, "invalid molid");
		PyErr_Print();
		return Py_BuildValue("i", -1);
	}
	if (i < 0 || j < 0 || k < 0 || l < 0 || k >= mol->nAtoms || i >= mol->nAtoms || j >= mol->nAtoms || l >= mol->nAtoms) {
		PyErr_SetString(PyExc_ValueError, "invalid index");
		PyErr_Print();
		return Py_BuildValue("i", -1);
	}
	//Reorder so i < l.
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
		if (i == mol->impropers[4*tmp] && j == mol->impropers[4*tmp+1] && k == mol->impropers[4*tmp+2] && l == mol->impropers[4*tmp+3]) {
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
			return Py_BuildValue("i", 1);
		}
	}
	return Py_BuildValue("i", 0);
}
static char* del_all_bond_doc = (char *)
"delallbonds([molid=top]) -> number of bonds deleted.\n"
"	Deletes all bonds associated with a molid.\n";
static PyObject* topo_del_all_bonds(PyObject *self, PyObject *args, PyObject *kwds) {
	int molid = -1, retval;
	static char *kwlist[] = { (char*)"molid", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist, &molid))
        return NULL;
    VMDApp *app = get_vmdapp();
	if (molid == -1)
		molid = app->molecule_top();
	Molecule *mol = app->moleculeList->mol_from_id(molid);
	if (!mol) {
		PyErr_SetString(PyExc_ValueError, "invalid molid");
		PyErr_Print();
		return Py_BuildValue("i", -1);
	}
	retval = mol->count_bonds();
	mol->clear_bonds();
	return Py_BuildValue("i", retval);
}
static char* del_all_angle_doc = (char *)
"delallangles([molid=top]) -> number of angles deleted.\n"
"	Deletes all angles associated with a molid.\n";
static PyObject* topo_del_all_angles(PyObject *self, PyObject *args, PyObject *kwds) {
	int molid = -1, retval;
	static char *kwlist[] = { (char*)"molid", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist, &molid))
        return NULL;
    VMDApp *app = get_vmdapp();
	if (molid == -1)
		molid = app->molecule_top();
	Molecule *mol = app->moleculeList->mol_from_id(molid);
	if (!mol) {
		PyErr_SetString(PyExc_ValueError, "invalid molid");
		PyErr_Print();
		return Py_BuildValue("i", -1);
	}
	retval = mol->num_angles();
	mol->angles.clear();
	mol->angleTypes.clear();
	return Py_BuildValue("i", retval);
}
static char* del_all_dihed_doc = (char *)
"delalldihedrals([molid=top]) -> number of dihedrals deleted.\n"
"	Deletes all dihedrals associated with a molid.\n";
static PyObject* topo_del_all_dihed(PyObject *self, PyObject *args, PyObject *kwds) {
	int molid = -1, retval;
	static char *kwlist[] = { (char*)"molid", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist, &molid))
        return NULL;
    VMDApp *app = get_vmdapp();
	if (molid == -1)
		molid = app->molecule_top();
	Molecule *mol = app->moleculeList->mol_from_id(molid);
	if (!mol) {
		PyErr_SetString(PyExc_ValueError, "invalid molid");
		PyErr_Print();
		return Py_BuildValue("i", -1);
	}
	retval = mol->num_dihedrals();
	mol->dihedrals.clear();
	mol->dihedralTypes.clear();
	return Py_BuildValue("i", retval);
}
static char* del_all_improper_doc = (char *)
"delallimpropers([molid=top]) -> number of impropers deleted.\n"
"	Deletes all impropers associated with a molid.\n";
static PyObject* topo_del_all_impropers(PyObject *self, PyObject *args, PyObject *kwds) {
	int molid = -1, retval;
	static char *kwlist[] = { (char*)"molid", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist, &molid))
        return NULL;
    VMDApp *app = get_vmdapp();
	if (molid == -1)
		molid = app->molecule_top();
	Molecule *mol = app->moleculeList->mol_from_id(molid);
	if (!mol) {
		PyErr_SetString(PyExc_ValueError, "invalid molid");
		PyErr_Print();
		return Py_BuildValue("i", -1);
	}
	retval = mol->num_impropers();
	mol->impropers.clear();
	mol->improperTypes.clear();
	return Py_BuildValue("i", retval);
}

static PyMethodDef Methods[] = {
  {(char *)"bonds", (PyCFunction)topo_get_bond, METH_VARARGS | METH_KEYWORDS, bond_doc},
  {(char *)"angles", (PyCFunction)topo_get_angle, METH_VARARGS | METH_KEYWORDS, angle_doc},
  {(char *)"dihedrals", (PyCFunction)topo_get_dihed, METH_VARARGS | METH_KEYWORDS, dihed_doc},
  {(char *)"impropers", (PyCFunction)topo_get_impro, METH_VARARGS | METH_KEYWORDS, impropers_doc},
  {(char *)"addbond", (PyCFunction)topo_add_bond, METH_VARARGS | METH_KEYWORDS, addbond_doc},
  {(char *)"addangle", (PyCFunction)topo_add_angle, METH_VARARGS | METH_KEYWORDS, addangle_doc},
  {(char *)"adddihedral", (PyCFunction)topo_add_dihed, METH_VARARGS | METH_KEYWORDS, adddihed_doc},
  {(char *)"addimproper", (PyCFunction)topo_add_improp, METH_VARARGS | METH_KEYWORDS, addimproper_doc},
  {(char *)"delbond", (PyCFunction)topo_del_bond, METH_VARARGS | METH_KEYWORDS, delbond_doc},
  {(char *)"delangle", (PyCFunction)topo_del_angle, METH_VARARGS | METH_KEYWORDS, delangle_doc},
  {(char *)"deldihedral", (PyCFunction)topo_del_dihed, METH_VARARGS | METH_KEYWORDS, deldihed_doc},
  {(char *)"delimproper", (PyCFunction)topo_del_improper, METH_VARARGS | METH_KEYWORDS, delimproper_doc},
  {(char *)"delallbonds", (PyCFunction)topo_del_all_bonds, METH_VARARGS | METH_KEYWORDS, del_all_bond_doc},
  {(char *)"delallangles", (PyCFunction)topo_del_all_angles, METH_VARARGS | METH_KEYWORDS, del_all_angle_doc},
  {(char *)"delalldihedrals", (PyCFunction)topo_del_all_dihed, METH_VARARGS | METH_KEYWORDS, del_all_dihed_doc},
  {(char *)"delallimpropers", (PyCFunction)topo_del_all_impropers, METH_VARARGS | METH_KEYWORDS, del_all_improper_doc},
  {NULL, NULL}
};

void inittopology() {
  (void)Py_InitModule((char *)"topology", Methods);
}
