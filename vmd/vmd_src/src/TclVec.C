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
 *      $RCSfile: TclVec.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.42 $      $Date: 2010/12/16 04:08:43 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   A C-based implementation of some performance-critical Tcl callable
 *   routines in VMD.  The C implementation outperforms a raw Tcl version
 *   by a factor of three or so.  The performance advantage helps 
 *   significantly when doing analysis in VMD.
 ***************************************************************************/

#include <tcl.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "TclCommands.h"
#include "Matrix4.h"
#include "utilities.h"

/***************** override some of the vector routines for speed ******/
/* These should be the exact C equivalent to the corresponding Tcl    */
/* vector commands */

// Function:  vecadd v1 v2 {v3 ...}
//  Returns: the sum of vectors; must all be the same length
//  The increase in speed from Tcl to C++ is 4561 / 255 == 18 fold
static int obj_vecadd(ClientData, Tcl_Interp *interp, int argc, 
		       Tcl_Obj * const objv[]) {
  if (argc < 3) {
    Tcl_WrongNumArgs(interp, 1, objv, (char *)"vec1 vec2 ?vec3? ?vec4? ...");
    return TCL_ERROR;
  }
  int num;
  Tcl_Obj **data;
  if (Tcl_ListObjGetElements(interp, objv[1], &num, &data) != TCL_OK) {
    return TCL_ERROR;
  }
  double *sum = new double[num];
  int i;
  for (i=0; i<num; i++) {
    if (Tcl_GetDoubleFromObj(interp, data[i], sum+i) != TCL_OK) {
      delete [] sum;
      return TCL_ERROR;
    }
  }
  // do the sums on the rest
  int num2;
  for (int term=2; term < argc; term++) {
    if (Tcl_ListObjGetElements(interp, objv[term], &num2, &data) != TCL_OK) {
      delete [] sum;
      return TCL_ERROR;
    }
    if (num != num2) {
      Tcl_SetResult(interp, (char *) "vecadd: two vectors don't have the same size", TCL_STATIC);
      delete [] sum;
      return TCL_ERROR;
    }
    for (i=0; i<num; i++) {
      double df;
      if (Tcl_GetDoubleFromObj(interp, data[i], &df) != TCL_OK) {
	delete [] sum;
	return TCL_ERROR;
      }
      sum[i] += df;
    }
  }

  
  // and return the result
  Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
  for (i=0; i<num; i++) {
    Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewDoubleObj(sum[i]));
  }
  Tcl_SetObjResult(interp, tcl_result);
  delete [] sum;
  return TCL_OK;
}

// Function:  vecsub  v1 v2
//  Returns:   v1 - v2


static int obj_vecsub(ClientData, Tcl_Interp *interp, int argc, Tcl_Obj *const objv[])
{
  if (argc != 3) {
    Tcl_WrongNumArgs(interp, 1, objv, (char *)"?x? ?y?");
    return TCL_ERROR;
  }
  int num1=0, num2=0;
  Tcl_Obj **data1, **data2;
  if (Tcl_ListObjGetElements(interp, objv[1], &num1, &data1) != TCL_OK)
    return TCL_ERROR;
  if (Tcl_ListObjGetElements(interp, objv[2], &num2, &data2) != TCL_OK)
    return TCL_ERROR;

  if (num1 != num2) {
    Tcl_SetResult(interp, (char *)"vecsub: two vectors don't have the same size", TCL_STATIC);
    return TCL_ERROR;
  }

  Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
  for (int i=0; i<num1; i++) {
    double d1=0, d2=0;
    if (Tcl_GetDoubleFromObj(interp, data1[i], &d1) != TCL_OK) {
      Tcl_SetResult(interp, (char *)"vecsub: non-numeric in first argument", TCL_STATIC);
      return TCL_ERROR; 
    }
    if (Tcl_GetDoubleFromObj(interp, data2[i], &d2) != TCL_OK) {
      Tcl_SetResult(interp, (char *)"vecsub: non-numeric in second argument", TCL_STATIC);
      return TCL_ERROR; 
    }
    Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewDoubleObj(d1-d2));
  }
  Tcl_SetObjResult(interp, tcl_result);
  return TCL_OK;
}


// Function: vecscale
//  Returns: scalar * vector or vector * scalar
// speedup is 1228/225 = 5.5 fold
static int obj_vecscale(ClientData, Tcl_Interp *interp, int argc, 
		       Tcl_Obj * const objv[]) {
  if (argc != 3) {
    Tcl_WrongNumArgs(interp, 1, objv, (char *)"?c? ?v?");
    return TCL_ERROR;
  }
    
  int num1, num2;
  Tcl_Obj **data1, **data2;
  if (Tcl_ListObjGetElements(interp, objv[1], &num1, &data1) != TCL_OK) {
    return TCL_ERROR;
  }
  if (Tcl_ListObjGetElements(interp, objv[2], &num2, &data2) != TCL_OK) {
    return TCL_ERROR;
  }
  if (num1 == 0 || num2 == 0) {
    Tcl_SetResult(interp, (char *) "vecscale: parameters must have data", TCL_STATIC);
    return TCL_ERROR;
  } else if (num1 != 1 && num2 != 1) {
    Tcl_SetResult(interp, (char *) "vecscale: one parameter must be a scalar value", TCL_STATIC);
    return TCL_ERROR;
  }
  
  int num;
  Tcl_Obj *scalarobj, **vector;
  if (num1 == 1) {
    scalarobj = data1[0];
    vector = data2;
    num = num2;
  } else {
    scalarobj = data2[0];
    vector = data1;
    num = num1;
  }
 
  double scalar;
  if (Tcl_GetDoubleFromObj(interp, scalarobj, &scalar) != TCL_OK)
    return TCL_ERROR;

  Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
  for (int i=0; i<num; i++) {
    double val;
    if (Tcl_GetDoubleFromObj(interp, vector[i], &val) != TCL_OK) {
      Tcl_SetResult(interp, (char *) "vecscale: non-numeric in vector", TCL_STATIC);
      return TCL_ERROR;
    }
    val *= scalar;
    Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewDoubleObj(val));
  }
  Tcl_SetObjResult(interp, tcl_result);
  return TCL_OK;
}

/// Given a string with a matrix in it, return the matrix
// returns TCL_OK if good
// If bad, returns TCL_ERROR and sets the Tcl result to the error message
// The name of the function should be passed in 'fctn' so the error message
// can be constructed correctly
int tcl_get_matrix(const char *fctn, Tcl_Interp *interp, 
			  Tcl_Obj *s, float *mat)
{ 
  int num_rows;
  Tcl_Obj **data_rows;
  if (Tcl_ListObjGetElements(interp, s, &num_rows, &data_rows) != TCL_OK) {
    char tmpstring[1024];
    sprintf(tmpstring, "%s: badly formed matrix", fctn);
    Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
    return TCL_ERROR;
  }
  if (num_rows != 4) {
    char tmpstring[1024];
    sprintf(tmpstring, "%s: need a 4x4 matrix", fctn);
    Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
    return TCL_ERROR;
  }
  int num_row[4];
  Tcl_Obj **data_row[4];
  if (Tcl_ListObjGetElements(interp, data_rows[0], num_row+0, data_row+0) != TCL_OK ||
      num_row[0] != 4 ||
      Tcl_ListObjGetElements(interp, data_rows[1], num_row+1, data_row+1) != TCL_OK ||
      num_row[1] != 4 ||
      Tcl_ListObjGetElements(interp, data_rows[2], num_row+2, data_row+2) != TCL_OK ||
      num_row[2] != 4 ||
      Tcl_ListObjGetElements(interp, data_rows[3], num_row+3, data_row+3) != TCL_OK ||
      num_row[3] != 4) {
    Tcl_AppendResult(interp, fctn, ": poorly formed matrix", NULL);
    return TCL_ERROR;
  }
  // now get the numbers
  for (int i=0; i<4; i++) {
    for (int j=0; j<4; j++) {
      double tmp;
      if (Tcl_GetDoubleFromObj(interp, data_row[i][j], &tmp) != TCL_OK) {
        char tmpstring[1024];
	sprintf(tmpstring, "%s: non-numeric in matrix", fctn);
        Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
        return TCL_ERROR;
      } else {
	mat[4*j+i] = (float) tmp;  // Matrix4 is transpose of Tcl's matrix
      }
    }
  }
  return TCL_OK;
}

int tcl_get_vector(const char *s, float *val, Tcl_Interp *interp)
{
  int num;
  const char **pos;
  if (Tcl_SplitList(interp, s, &num, &pos) != TCL_OK) {
    Tcl_SetResult(interp, (char *) "need three data elements for a vector", 
                  TCL_STATIC);
    return TCL_ERROR;
  }
  if (num != 3) {
    Tcl_SetResult(interp, (char *) "need three numbers for a vector", TCL_STATIC);
    return TCL_ERROR;
  }
  double a[3];
  if (Tcl_GetDouble(interp, pos[0], a+0) != TCL_OK ||
      Tcl_GetDouble(interp, pos[1], a+1) != TCL_OK ||
      Tcl_GetDouble(interp, pos[2], a+2) != TCL_OK) {
    ckfree((char *) pos); // free of tcl data
    return TCL_ERROR;
  }
  val[0] = (float) a[0];
  val[1] = (float) a[1];
  val[2] = (float) a[2];
  ckfree((char *) pos); // free of tcl data
  return TCL_OK;
}

// append the matrix into the Tcl result
void tcl_append_matrix(Tcl_Interp *interp, const float *mat) {
  Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
  for (int i=0; i<4; i++) {
    Tcl_Obj *m = Tcl_NewListObj(0, NULL);
    for (int j=0; j<4; j++) 
      Tcl_ListObjAppendElement(interp, m, Tcl_NewDoubleObj(mat[4*j+i]));
    Tcl_ListObjAppendElement(interp, tcl_result, m);
  }
  Tcl_SetObjResult(interp, tcl_result);
}

// speed up the matrix * vector routines -- DIFFERENT ERROR MESSAGES
// THAN THE TCL VERSION
// speedup is nearly 25 fold
static int obj_vectrans(ClientData, Tcl_Interp *interp, int argc, 
		  Tcl_Obj * const objv[])
{
  if (argc != 3) {
    Tcl_WrongNumArgs(interp, 1, objv, (char *)"?matrix? ?vector?");
    return TCL_ERROR;
  }

  // get the matrix data
  float mat[16];
  if (tcl_get_matrix(
    Tcl_GetStringFromObj(objv[0],NULL), interp, objv[1], mat) != TCL_OK) {
    return TCL_ERROR;
  }
  
  // for the vector
  Tcl_Obj **vec;
  int vec_size;
  if (Tcl_ListObjGetElements(interp, objv[2], &vec_size, &vec) != TCL_OK)
    return TCL_ERROR;

  if (vec_size != 3 && vec_size != 4) {
    Tcl_SetResult(interp, (char *) "vectrans: vector must be of size 3 or 4",
                  TCL_STATIC);
    return TCL_ERROR;
  }

  float opoint[4];
  opoint[3] = 0;
  for (int i=0; i<vec_size; i++) {
    double tmp;
    if (Tcl_GetDoubleFromObj(interp, vec[i], &tmp) != TCL_OK) {
      Tcl_SetResult(interp, (char *) "vectrans: non-numeric in vector", TCL_STATIC);
      return TCL_ERROR;
    }
    opoint[i] = (float)tmp;
  }
  // vector data is in vec_data
  float npoint[4];
 
  npoint[0]=opoint[0]*mat[0]+opoint[1]*mat[4]+opoint[2]*mat[8]+opoint[3]*mat[12]
;
  npoint[1]=opoint[0]*mat[1]+opoint[1]*mat[5]+opoint[2]*mat[9]+opoint[3]*mat[13]
;
  npoint[2]=opoint[0]*mat[2]+opoint[1]*mat[6]+opoint[2]*mat[10]+opoint[3]*mat[14
];
  npoint[3]=opoint[0]*mat[3]+opoint[1]*mat[7]+opoint[2]*mat[11]+opoint[3]*mat[15
];
  // return it

  {
  Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
  for (int i=0; i<vec_size; i++) 
    Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewDoubleObj(npoint[i]));
  Tcl_SetObjResult(interp, tcl_result);
  }
  return TCL_OK;
}


// Function: transmult m1 m2 ... mn
//  Returns: the product of the matricies
// speedup is 136347 / 1316 = factor of 104
static int obj_transmult(ClientData, Tcl_Interp *interp, int argc, 
		   Tcl_Obj * const objv[]) {
  // make there there are at least two values
  if (argc < 3) {
    Tcl_WrongNumArgs(interp, 1, objv, (char *)"mx my ?m1? ?m2? ...");
    return TCL_ERROR;
  }
  // Get the first matrix
  float mult[16];
  if (tcl_get_matrix("transmult: ", interp, objv[1], mult) != TCL_OK) {
    return TCL_ERROR;
  }
  int i = 2;
  float pre[16];
  while (i < argc) {
    if (tcl_get_matrix("transmult: ", interp, objv[i], pre) != TCL_OK) {
      return TCL_ERROR;
    }
    // premultiply mult by tmp
    float tmp[4];
    for (int k=0; k<4; k++) {
      tmp[0] = mult[k];
      tmp[1] = mult[4+k];
      tmp[2] = mult[8+k];
      tmp[3] = mult[12+k];
      for (int j=0; j<4; j++) {
        mult[4*j+k] = pre[4*j]*tmp[0] + pre[4*j+1]*tmp[1] +
          pre[4*j+2]*tmp[2] + pre[4*j+3]*tmp[3];
      }
    }
    i++;
  }
  tcl_append_matrix(interp, mult);
  return TCL_OK;
}

static int obj_transvec(ClientData, Tcl_Interp *interp, int argc, 
		   Tcl_Obj * const objv[]) {
  if (argc != 2) {
    Tcl_WrongNumArgs(interp, 1, objv, (char *)"?vector?");
    return TCL_ERROR;
  }
  
  int num;
  Tcl_Obj **data;
  if (Tcl_ListObjGetElements(interp, objv[1], &num, &data) != TCL_OK) 
    return TCL_ERROR;
  if (num != 3) {
    Tcl_AppendResult(interp, "transvec: vector must have three elements",NULL);
    return TCL_ERROR;
  }
  double x,y,z;
  if (Tcl_GetDoubleFromObj(interp, data[0], &x) != TCL_OK ||
      Tcl_GetDoubleFromObj(interp, data[1], &y) != TCL_OK ||
      Tcl_GetDoubleFromObj(interp, data[2], &z) != TCL_OK) {
    Tcl_SetResult(interp, (char *)"transvec: non-numeric in vector", TCL_STATIC);
    return TCL_ERROR;
  }
  Matrix4 mat;
  mat.transvec((float) x,(float) y,(float) z);
  tcl_append_matrix(interp, mat.mat);
  return TCL_OK;
}

static int obj_transvecinv(ClientData, Tcl_Interp *interp, int argc, 
		   Tcl_Obj * const objv[]) {
  if (argc != 2) {
    Tcl_WrongNumArgs(interp, 1, objv, (char *)"?vector?");
    return TCL_ERROR;
  }
  
  int num;
  Tcl_Obj **data;
  if (Tcl_ListObjGetElements(interp, objv[1], &num, &data) != TCL_OK) 
    return TCL_ERROR;
  if (num != 3) {
    Tcl_AppendResult(interp, "transvecinv: vector must have three elements",NULL);
    return TCL_ERROR;
  }
  double x,y,z;
  if (Tcl_GetDoubleFromObj(interp, data[0], &x) != TCL_OK ||
      Tcl_GetDoubleFromObj(interp, data[1], &y) != TCL_OK ||
      Tcl_GetDoubleFromObj(interp, data[2], &z) != TCL_OK) {
    Tcl_SetResult(interp, (char *)"transvecinv: non-numeric in vector", TCL_STATIC);
    return TCL_ERROR;
  }
  Matrix4 mat;
  mat.transvecinv((float) x,(float) y,(float) z);
  tcl_append_matrix(interp, mat.mat);
  return TCL_OK;
}

// Returns the transformation matrix needed to rotate by a certain
// angle around a given axis.
// Tcl syntax:
// transabout v amount [deg|rad|pi]
// The increase in speed from Tcl to C++ is 15 fold
static int obj_transabout(ClientData, Tcl_Interp *interp, int argc, 
		   Tcl_Obj * const objv[]) {
  if (argc != 3 && argc != 4) {
    Tcl_WrongNumArgs(interp, 1, objv, (char *)"axis amount [deg|rad|pi]");
    return TCL_ERROR;
  }
  
  int num;
  Tcl_Obj **data;
  // get the axis
  if (Tcl_ListObjGetElements(interp, objv[1], &num, &data) != TCL_OK) 
    return TCL_ERROR;
  if (num != 3) {
    Tcl_AppendResult(interp, "transabout: vector must have three elements",NULL);
    return TCL_ERROR;
  }
  double x,y,z;
  if (Tcl_GetDoubleFromObj(interp, data[0], &x) != TCL_OK ||
      Tcl_GetDoubleFromObj(interp, data[1], &y) != TCL_OK ||
      Tcl_GetDoubleFromObj(interp, data[2], &z) != TCL_OK) {
    Tcl_SetResult(interp, (char *)"transabout: non-numeric in vector", TCL_STATIC);
    return TCL_ERROR;
  }

  // get the amount
  double amount;
  if (Tcl_GetDoubleFromObj(interp, objv[2], &amount) != TCL_OK) {
    Tcl_SetResult(interp, (char *)"transabout: non-numeric angle", TCL_STATIC);
    return TCL_ERROR;
  }

  // get units
  if (argc == 4) {
    if (!strcmp(Tcl_GetStringFromObj(objv[3], NULL), "deg")) {
      amount = DEGTORAD(amount);
    } else if (!strcmp(Tcl_GetStringFromObj(objv[3], NULL), "rad")) {
      // amount = amount; 
    } else if (!strcmp(Tcl_GetStringFromObj(objv[3], NULL), "pi")) {
      amount = amount*VMD_PI;
    } else {
      Tcl_AppendResult(interp, "transabout: unit must be deg|rad|pi",NULL);
      return TCL_ERROR;
    }
  } else {
    // If no unit was specified assume that we have degrees
    amount = DEGTORAD(amount);
  }

  float axis[3];
  axis[0] = (float) x;
  axis[1] = (float) y;
  axis[2] = (float) z;

  Matrix4 mat;
  mat.rotate_axis(axis, (float) amount);
  tcl_append_matrix(interp, mat.mat);
  return TCL_OK;
}

static int obj_veclength(ClientData, Tcl_Interp *interp, int argc, Tcl_Obj *const objv[]) {

  if (argc != 2) {
    Tcl_WrongNumArgs(interp, 1, objv, (char *)"?vector?");
    return TCL_ERROR;
  }

  int num;
  Tcl_Obj **data;
  if (Tcl_ListObjGetElements(interp, objv[1], &num, &data) != TCL_OK) 
    return TCL_ERROR;

  double length = 0.;
  for (int i=0; i<num; i++) {
    double tmp;
    if (Tcl_GetDoubleFromObj(interp, data[i], &tmp) != TCL_OK) {
      Tcl_SetResult(interp, (char *) "veclength: non-numeric in vector", TCL_STATIC);
      return TCL_ERROR;
    } else {
      length += tmp*tmp;
    }
  }

  length = sqrt(length);
  Tcl_Obj *tcl_result = Tcl_GetObjResult(interp);
  Tcl_SetDoubleObj(tcl_result, length);
  return TCL_OK; 
}


static double* obj_getdoublearray(Tcl_Interp *interp, Tcl_Obj *const objv[], int *len) {
  int num;

  Tcl_Obj **data;
  if (Tcl_ListObjGetElements(interp, objv[1], &num, &data) != TCL_OK)
    return NULL;
 
  double *list = (double*) malloc(num*sizeof(double));
  if (list == NULL)
    return NULL;

  for (int i=0; i<num; i++) {
    double tmp;
    if (Tcl_GetDoubleFromObj(interp, data[i], &tmp) != TCL_OK) {
      Tcl_SetResult(interp, (char *) "veclength: non-numeric in vector", TCL_STATIC);
      free(list);
      return NULL;
    }
    list[i] = tmp;
  }

  *len = num;

  return list;
}


static int obj_vecsum(ClientData, Tcl_Interp *interp, int argc, Tcl_Obj *const objv[]) {
  if (argc != 2) {
    Tcl_WrongNumArgs(interp, 1, objv, (char *)"?vector?");
    return TCL_ERROR;
  }

  int num;
  double *list = obj_getdoublearray(interp, objv, &num);
  if (list == NULL) 
    return TCL_ERROR;

  double sum = 0.;
  for (int i=0; i<num; i++) {
    sum += list[i];
  }
  free(list);

  Tcl_Obj *tcl_result = Tcl_GetObjResult(interp);
  Tcl_SetDoubleObj(tcl_result, sum);
  return TCL_OK;
}


static int obj_vecmean(ClientData, Tcl_Interp *interp, int argc, Tcl_Obj *const objv[]) {
  if (argc != 2) {
    Tcl_WrongNumArgs(interp, 1, objv, (char *)"?vector?");
    return TCL_ERROR;
  }

  int num;
  double *list = obj_getdoublearray(interp, objv, &num);
  if (list == NULL) 
    return TCL_ERROR;

  double sum = 0.;
  for (int i=0; i<num; i++) {
    sum += list[i];
  }
  sum /= (double) num;
  free(list);

  Tcl_Obj *tcl_result = Tcl_GetObjResult(interp);
  Tcl_SetDoubleObj(tcl_result, sum);
  return TCL_OK;
}


static int obj_vecstddev(ClientData, Tcl_Interp *interp, int argc, Tcl_Obj *const objv[]) {
  if (argc != 2) {
    Tcl_WrongNumArgs(interp, 1, objv, (char *)"?vector?");
    return TCL_ERROR;
  }

  int i, num;
  double* list = obj_getdoublearray(interp, objv, &num);
  if (list == NULL) 
    return TCL_ERROR;

  double mean = 0.;
  for (i=0; i<num; i++) {
    mean += list[i];
  }
  mean /= (double) num;

  double stddev = 0.;
  for (i=0; i<num; i++) {
    double tmp = list[i] - mean;
    stddev += tmp * tmp; 
  }
  stddev /= (double) num;
  stddev = sqrt(stddev);
  free(list);

  Tcl_Obj *tcl_result = Tcl_GetObjResult(interp);
  Tcl_SetDoubleObj(tcl_result, stddev);
  return TCL_OK;
}


int Vec_Init(Tcl_Interp *interp) {
  Tcl_CreateObjCommand(interp, "vecadd", obj_vecadd,
                    (ClientData) NULL, (Tcl_CmdDeleteProc *) NULL);
  Tcl_CreateObjCommand(interp, "vecsub", obj_vecsub,
                    (ClientData) NULL, (Tcl_CmdDeleteProc *) NULL);
  Tcl_CreateObjCommand(interp, "vecscale", obj_vecscale,
                    (ClientData) NULL, (Tcl_CmdDeleteProc *) NULL);
  Tcl_CreateObjCommand(interp, "transmult", obj_transmult,
                    (ClientData) NULL, (Tcl_CmdDeleteProc *) NULL);
  Tcl_CreateObjCommand(interp, "vectrans", obj_vectrans,
                    (ClientData) NULL, (Tcl_CmdDeleteProc *) NULL);
  Tcl_CreateObjCommand(interp, "veclength", obj_veclength,
                    (ClientData) NULL, (Tcl_CmdDeleteProc *) NULL);
  Tcl_CreateObjCommand(interp, "vecmean", obj_vecmean,
                    (ClientData) NULL, (Tcl_CmdDeleteProc *) NULL);
  Tcl_CreateObjCommand(interp, "vecstddev", obj_vecstddev,
                    (ClientData) NULL, (Tcl_CmdDeleteProc *) NULL);
  Tcl_CreateObjCommand(interp, "vecsum", obj_vecsum,
                    (ClientData) NULL, (Tcl_CmdDeleteProc *) NULL);
  Tcl_CreateObjCommand(interp, "transvec", obj_transvec,
                    (ClientData) NULL, (Tcl_CmdDeleteProc *) NULL);
  Tcl_CreateObjCommand(interp, "transvecinv", obj_transvecinv,
                    (ClientData) NULL, (Tcl_CmdDeleteProc *) NULL);
  Tcl_CreateObjCommand(interp, "transabout", obj_transabout,
                    (ClientData) NULL, (Tcl_CmdDeleteProc *) NULL);
  return TCL_OK;
}
 
