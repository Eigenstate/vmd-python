#include "py_commands.h"
#include "VMDApp.h"
#include "Measure.h"

static char* bond_doc = (char *)"bond(atom1, atom2[, molid=top, molid2=molid, frame=now, first, last]) -> list of floats\n"
"	Measures the distance between the atom1 and atom2 over the trajectory,\n"
"	either at the given frame, or over frames from first to last.\n"
"	Optionally, the distance between atoms of different molecules can be measured, by specifying specific molids.\n";
static PyObject *measure_bond(PyObject *self, PyObject *args, PyObject *kwds) {
	int first=-1, last=-1, frame=-1;
	int molid[2] = {-1, -1};
	int atmid[2];
	VMDApp *app = get_vmdapp();
	static char *kwlist[] = { (char *)"atom1",(char *)"atom2",(char *)"molid", (char *)"molid2", 
                            (char *)"frame", (char *)"first", (char *)"last", NULL };

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "ii|iiiii", kwlist, 
        &atmid[0], &atmid[1], &molid[0], &molid[1], &frame, &first, &last)) { return NULL;}
	if (molid[0] == -1)
		molid[0] = app->molecule_top();
    if (molid[1] == -1)
    	molid[1] = molid[0];
    if (frame != -1 && (first != -1 || last != -1)) {
    	PyErr_Warn(PyExc_SyntaxWarning, "Frame, as well as first or last were specified. Returning value for just the frame");
    	first = -1;
    	last = -1;
    }
    //Calculate the bond length
    ResizeArray<float> gValues(1024);
	int ret_val;
	ret_val = measure_geom(app->moleculeList, molid, atmid, &gValues, frame, first, last,
			 molid[0], MEASURE_BOND);
	//Check for errors
	if (ret_val<0) {
		PyErr_SetString(PyExc_RuntimeError, measure_error(ret_val));
		PyErr_Print();
		return NULL;
	}
	//Build the python list.
    int numvalues = gValues.num();
    PyObject* returnlist = PyList_New(numvalues);
    for (int i = 0; i < numvalues; i++)
    	PyList_SetItem(returnlist, i, Py_BuildValue("f", gValues[i]));
    
    return returnlist;
}
static char* angle_doc = (char *)
"angle(atom1, atom2, atom3[, molid=top, molid2=molid, molid3=molid, frame=now, first, last]) -> list of floats\n"
"	Measures the angle between the atoms over the trajectory,\n"
"	either at the given frame, or over frames from first to last.\n"
"	Angle is measured from atom1 to atom2 to atom3.\n"
"	Optionally, the angle between atoms of different molecules can be measured, by specifying specific molids.\n";
static PyObject *measure_angle(PyObject *self, PyObject *args, PyObject *kwds) {
	int first=-1, last=-1, frame=-1;
	int molid[3] = {-1, -1, -1};
	int atmid[3];
	VMDApp *app = get_vmdapp();
	static char *kwlist[] = { (char *)"atom1",(char *)"atom2",(char *)"atom3",(char *)"molid",
				(char *)"molid2", (char *)"molid3", (char *)"frame", (char *)"first", (char *)"last", NULL };

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "iii|iiiiii", kwlist, 
        &atmid[0], &atmid[1], &atmid[2], &molid[0], &molid[1], &molid[2], &frame, &first, &last)) { return NULL;}
	if (molid[0] == -1)
		molid[0] = app->molecule_top();
	if (molid[1] == -1)
    	molid[1] = molid[0];
    if (molid[2] == -1)
    	molid[2] = molid[0];
    if (frame != -1 && (first != -1 || last != -1)) {
    	PyErr_Warn(PyExc_SyntaxWarning, "Frame, as well as first or last were specified. Returning value for just the frame");
    	first = -1;
    	last = -1;
    }
    //Calculate the angle
    ResizeArray<float> gValues(1024);
	int ret_val;
	ret_val = measure_geom(app->moleculeList, molid, atmid, &gValues, frame, first, last,
			 molid[0], MEASURE_ANGLE);
	if (ret_val<0) {
		PyErr_SetString(PyExc_RuntimeError, measure_error(ret_val));
		PyErr_Print();
		return NULL;
	}
	//Build the python list.
    int numvalues = gValues.num();
    PyObject* returnlist = PyList_New(numvalues);
    for (int i = 0; i < numvalues; i++)
    	PyList_SetItem(returnlist, i, Py_BuildValue("f", gValues[i]));
    return returnlist;
}

static char* dihed_doc = (char *)
"dihedral(atom1, atom2, atom3, atom4[, molid=top, molid2=molid, molid3=molid, molid4=molid, frame=now, first, last]) -> list of floats\n"
"	Measures the dihedral angle between the atoms over the trajectory,\n"
"	either at the given frame, or over frames from first to last.\n"
"	Dihedral angle is measured from atom1 to atom2 to atom3 to atom4.\n"
"	Optionally, the angle between atoms of different molecules can be measured, by specifying specific molids.\n";
static PyObject *measure_dihed(PyObject *self, PyObject *args, PyObject *kwds) {
	int first=-1, last=-1, frame=-1;
	int molid[4] = {-1, -1, -1, -1};
	int atmid[4];
	VMDApp *app = get_vmdapp();
	static char *kwlist[] = { (char *)"atom1",(char *)"atom2",(char *)"atom3",(char *)"atom4",(char *)"molid",
				(char *)"molid2", (char *)"molid3", (char *)"molid4", (char *)"frame", (char *)"first", (char *)"last", NULL };

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiii|iiiiiii", kwlist, 
        &atmid[0], &atmid[1], &atmid[2], &atmid[3], &molid[0], &molid[1], &molid[2], &molid[3], &frame, &first, &last)) { return NULL;}
	if (molid[0] == -1)
		molid[0] = app->molecule_top();
	if (molid[1] == -1)
    	molid[1] = molid[0];
    if (molid[2] == -1)
    	molid[2] = molid[0];
    if (molid[3] == -1)
    	molid[3] = molid[0];
    if (frame != -1 && (first != -1 || last != -1)) {
    	PyErr_Warn(PyExc_SyntaxWarning, "Frame, as well as first or last were specified. Returning value for just the frame");
    	first = -1;
    	last = -1;
    }
    //Calculate the angle
    ResizeArray<float> gValues(1024);
	int ret_val;
	ret_val = measure_geom(app->moleculeList, molid, atmid, &gValues, frame, first, last,
			 molid[0], MEASURE_DIHED);
	if (ret_val<0) {
		PyErr_SetString(PyExc_RuntimeError, measure_error(ret_val));
		return NULL;
	}
	//Build the python list.
    int numvalues = gValues.num();
    PyObject* returnlist = PyList_New(numvalues);
    for (int i = 0; i < numvalues; i++)
    	PyList_SetItem(returnlist, i, Py_BuildValue("f", gValues[i]));
    return returnlist;
}

static PyMethodDef Methods[] = {
  {(char *)"bond", (PyCFunction)measure_bond, METH_VARARGS | METH_KEYWORDS, bond_doc},
  {(char *)"angle", (PyCFunction)measure_angle, METH_VARARGS | METH_KEYWORDS, angle_doc},
  {(char *)"dihedral", (PyCFunction)measure_dihed, METH_VARARGS | METH_KEYWORDS, dihed_doc},
  {NULL, NULL}
};

void initmeasure() {
  (void)Py_InitModule((char *)"measure", Methods);
}
