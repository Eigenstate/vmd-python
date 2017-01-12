/* 
 * tcl interface for the sgsmooth plugin 
 * 
 * Copyright (c) 2006-2009 akohlmey@cmm.chem.upenn.edu
 */

#include <stdio.h>
#include <string.h>
#include <tcl.h>

#include "sgsmooth.h"
#include "openmp-util.h"

/* pointer to current interpreter 
 * tcl currently allows only one interpreter
 * across threads, so we are safe for now.
 */
static Tcl_Interp *sgs_interp = NULL;

void sgs_error(const char *msg) 
{
    char *buffer;
    
    if (sgs_interp == NULL) return;
    
    buffer = (char *)Tcl_Alloc((strlen(msg)+15)*sizeof(char));
    sprintf(buffer, "vmdcon -error {%s}", msg);
    
    Tcl_Eval(sgs_interp,buffer);
    Tcl_Free(buffer);
    return;
}


/* this is the actual interface to the 'sgsmooth' command.
 * it parses the arguments, calls the calculation subroutine
 * and then passes the result to the interpreter. */
int tcl_sgsmooth(ClientData nodata, Tcl_Interp *interp,
             int objc, Tcl_Obj *const objv[]) 
{
    double *input, *output;
    int ndat, window, order;

    Tcl_Obj *resultPtr, **data;
    int i;

    sgs_interp=interp;
    
    /* Parse arguments:
     *
     * usage: sgsmooth <data> <window> <order>
     * 
     * the first is a list floating point numbers with the data
     * <window>     the size of the sliding window
     * <order>      the order of the polynomial to fit to
     */
    if (objc != 4) {
        Tcl_WrongNumArgs(interp, 1, objv, "<data> window order");
        return TCL_ERROR;
    }
    
    /* parse required arguments one by one. */
    /* time series data */
    Tcl_IncrRefCount(objv[1]);
    if (Tcl_ListObjGetElements(interp, objv[1], &ndat, &data) != TCL_OK) {
        return TCL_ERROR;
    }
    if (Tcl_GetIntFromObj(interp, objv[2], &window) != TCL_OK) {
        return TCL_ERROR;
    }
    if (Tcl_GetIntFromObj(interp, objv[3], &order) != TCL_OK) {
        return TCL_ERROR;
    }

    check_thread_count(interp,"sgsmooth");

    /* allocate temporary data */
    input = (double *)Tcl_Alloc(ndat*sizeof(double));
   
    /* parse list of lists for time series */ 
    for (i=0; i<ndat; ++i) {
        Tcl_GetDoubleFromObj(interp, data[i], (input + i));
    }

    /* free reference on the input data */
    Tcl_DecrRefCount(objv[1]);
    
    /* do the calculation */
    output=calc_sgsmooth(ndat, input, window, order);
    if (output == NULL) {
        Tcl_AppendResult(interp, "sgsmooth: error in calculation", NULL);
        return TCL_ERROR;
    }
    
    /* prepare results */
    resultPtr = Tcl_NewListObj(0, NULL);
    for (i=0; i < ndat; ++i) {
        Tcl_ListObjAppendElement(interp, resultPtr, Tcl_NewDoubleObj(output[i]));
    }
    Tcl_SetObjResult(interp, resultPtr);

    /* free intermediate storage */
    Tcl_Free((char *)input);

    return TCL_OK;
}

/* this is the actual interface to the 'sgsderiv' command.
 * it parses the arguments, calls the calculation subroutine
 * and then passes the result to the interpreter. */
int tcl_sgsderiv(ClientData nodata, Tcl_Interp *interp,
                 int objc, Tcl_Obj *const objv[]) 
{
    double *input, *output, delta;
    int ndat, window, order;

    Tcl_Obj *resultPtr, **data;
    int i;

    sgs_interp=interp;

    /* set defaults */
    delta = 1.0;
    
    /* Parse arguments:
     *
     * usage: sgsderiv <data> window order ?delta?
     * 
     * the first is a list floating point numbers with the data
     * <window>     the size of the sliding window
     * <order>      the order of the polynomial to fit to
     * <delta>      the distance between two data points. defaults to 1.0.
     */
    if ((objc < 4) || (objc > 5)) {
        Tcl_WrongNumArgs(interp, 1, objv, "<data> window order ?delta?");
        return TCL_ERROR;
    }
    
    /* parse required arguments one by one. */
    /* time series data */
    Tcl_IncrRefCount(objv[1]);
    if (Tcl_ListObjGetElements(interp, objv[1], &ndat, &data) != TCL_OK) {
        return TCL_ERROR;
    }
    if (Tcl_GetIntFromObj(interp, objv[2], &window) != TCL_OK) {
        return TCL_ERROR;
    }
    if (Tcl_GetIntFromObj(interp, objv[3], &order) != TCL_OK) {
        return TCL_ERROR;
    }

    if(objc == 5) {
        if (Tcl_GetDoubleFromObj(interp, objv[4], &delta) != TCL_OK) {
            return TCL_ERROR;
        }
    }

    check_thread_count(interp,"sgsderiv");

    /* allocate temporary data */
    input = (double *)Tcl_Alloc(ndat*sizeof(double));
   
    /* parse list of lists for time series */ 
    for (i=0; i<ndat; ++i) {
        Tcl_GetDoubleFromObj(interp, data[i], (input + i));
    }

    /* free reference on the input data */
    Tcl_DecrRefCount(objv[1]);
    
    /* do the calculation */
    output=calc_sgsderiv(ndat, input, window, order, delta);
    if (output == NULL) {
        Tcl_AppendResult(interp, "sgsderiv: error in calculation", NULL);
        return TCL_ERROR;
    }
    
    /* prepare results */
    resultPtr = Tcl_NewListObj(0, NULL);
    for (i=0; i < ndat; ++i) {
        Tcl_ListObjAppendElement(interp, resultPtr, Tcl_NewDoubleObj(output[i]));
    }
    Tcl_SetObjResult(interp, resultPtr);

    /* free intermediate storage */
    Tcl_Free((char *)input);

    return TCL_OK;
}

/* register the plugin with the tcl interpreters */
#if defined(SGSMOOTHTCLDLL_EXPORTS) && defined(_WIN32)
#  undef TCL_STORAGE_CLASS
#  define TCL_STORAGE_CLASS DLLEXPORT

#define WIN32_LEAN_AND_MEAN // Exclude rarely-used stuff from Window s headers
#include <windows.h>

BOOL APIENTRY DllMain( HANDLE hModule, 
                       DWORD  ul_reason_for_call, 
                       LPVOID lpReserved )
{
    return TRUE;
}

EXTERN int Sgsmooth_Init(Tcl_Interp *interp)

#else

int Sgsmooth_Init(Tcl_Interp *interp)   

#endif
{
  Tcl_CreateObjCommand(interp,"sgsmooth",tcl_sgsmooth,
                       (ClientData)NULL, (Tcl_CmdDeleteProc*)NULL);
  Tcl_CreateObjCommand(interp,"sgsderiv",tcl_sgsderiv,
                       (ClientData)NULL, (Tcl_CmdDeleteProc*)NULL);

  Tcl_PkgProvide(interp, "sgsmooth", "1.1");

  check_thread_count(interp,"sgsmooth");

  sgs_interp=interp;
  
  return TCL_OK;
}

