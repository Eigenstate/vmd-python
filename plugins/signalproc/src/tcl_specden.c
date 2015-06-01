/* 
 * tcl interface for the specden plugin 
 * 
 * Copyright (c) 2006-2009 akohlmey@cmm.chem.upenn.edu
 */

#include <stdio.h>
#include <string.h>
#include <tcl.h>

#include "specden.h"
#include "openmp-util.h"

/* this is the actual interface to the 'specden' command.
 * it parses the arguments, calls the calculation subroutine
 * and then passes the result to the interpreter. */
int tcl_specden(ClientData nodata, Tcl_Interp *interp,
             int objc, Tcl_Obj *const objv[]) 
{
    double *input, *output, maxfreq, deltat, temp;
    int ndat, nn, specr;
    const char *norm;

    Tcl_Obj *resultPtr, *freq, *spec, **tdata;
    int i, normtype;
    double avg[3];

    /* defaults */
    specr = 1;
    normtype = NORM_HARMONIC;
    temp  = 300;

    /* Parse arguments:
     *
     * usage: specden <x-,y-,z-data> <deltat> <maxfreq> ?<norm>? ?<temp>? ?<specr>?
     * 
     * the first is a list containing a time series of 1-3 dim vectors.
     * <deltat>     time between data points in a.u.
     * <maxfreq>    max. frequency in wavenumbers
     * <norm>       correction method. (default = harmonic) others are: fourier, classic, kubo, schofield
     * <specr>      spectrum resolution factor (default 1 = max resolution and noise)
     * <temp>       temperature for corrections in kelvin (default: 300K)
     */
    if (objc < 4) {
        Tcl_WrongNumArgs(interp, 1, objv, "<x-,y-,z-data> deltat maxfreq ?norm? ?temp? ?specr?");
        return TCL_ERROR;
    }
    
    /* parse required arguments one by one. */
    /* time series data */
    Tcl_IncrRefCount(objv[1]);
    if (Tcl_ListObjGetElements(interp, objv[1], &ndat, &tdata) != TCL_OK) {
        return TCL_ERROR;
    }
    if (Tcl_GetDoubleFromObj(interp, objv[2], &deltat) != TCL_OK) {
        return TCL_ERROR;
    }
    if (Tcl_GetDoubleFromObj(interp, objv[3], &maxfreq) != TCL_OK) {
        return TCL_ERROR;
    }

    /* optional arguments */
    if (objc > 4) {
        norm = Tcl_GetString(objv[4]); 
        if (strncmp(norm,"harm",4) == 0) {
            normtype = NORM_HARMONIC;
        } else if (strncmp(norm,"four",4) == 0) {
            normtype = NORM_FOURIER;
        } else if (strncmp(norm,"clas",4) == 0) {
            normtype = NORM_CLASSIC;
        } else if (strncmp(norm,"kubo",4) == 0) {
            normtype = NORM_KUBO;
        } else if (strncmp(norm,"scho",4) == 0) {
            normtype = NORM_SCHOFIELD;
        } else {
            Tcl_AppendResult(interp, "specden: unknown correction scheme: ", norm, NULL);
            return TCL_ERROR;
        }
    }
    if (objc > 5) {
        if (Tcl_GetDoubleFromObj(interp, objv[5], &temp) != TCL_OK) {
            return TCL_ERROR;
        }
    }
    if (objc > 6) {
        if (Tcl_GetIntFromObj(interp, objv[6], &specr) != TCL_OK) {
            return TCL_ERROR;
        }
    }
    if (objc > 7) {
        Tcl_WrongNumArgs(interp, 1, objv, "<x-,y-,z-data> deltat maxfreq ?norm? ?temp? ?specr?");
        return TCL_ERROR;
    }
    
    check_thread_count(interp,"specden");

    input = (double *)Tcl_Alloc((ndat+2)*3*sizeof(double));
    input[0]=input[1]=input[2]=0.0; /* internally we have fortran style array indices :-( */
   
    /* parse list of lists for time series */ 
    avg[0]=avg[1]=avg[2]=0.0;
    for (i=0; i<ndat; ++i) {
        int num_coords, j;
        Tcl_Obj **clist;

        input[3*i+0]=input[3*i+1]=input[3*i+2]=0.0; 
        if (Tcl_ListObjGetElements(interp, tdata[i], &num_coords, &clist) != TCL_OK) {
            return TCL_ERROR;
        }
        for (j=0; j<num_coords; ++j) {
            Tcl_GetDoubleFromObj(interp, clist[j], (input + 3*i+j));
        }
        avg[0] += input[3*i+0];
        avg[1] += input[3*i+1];
        avg[2] += input[3*i+2];
    }
    /* apply shift, so that data has a zero mean and we get a cleaner spectrum */
    avg[0] /= (double) ndat;
    avg[1] /= (double) ndat;
    avg[2] /= (double) ndat;
    for (i=0; i<ndat; ++i) {
        input[3*i+0] -= avg[0];
        input[3*i+1] -= avg[1];
        input[3*i+2] -= avg[2];
    }

    /* free references on the time series list */
    Tcl_DecrRefCount(objv[1]);
    
    nn = (int) ((double)ndat)*maxfreq/219474.0*deltat/(2.0*M_PI);
    output = (double *)Tcl_Alloc((nn+1)*2*sizeof(double));
    
    /* do the calculation */
    nn=calc_specden(ndat, input, output, normtype, specr, maxfreq, deltat, temp);
    if (nn < 0) {
        Tcl_AppendResult(interp, "specden: error in calculation", NULL);
        return TCL_ERROR;
    }
    
    /* free intermediate storage */
    Tcl_Free((char *)input);

    /* prepare results */
    freq = Tcl_NewListObj(0, NULL);
    spec = Tcl_NewListObj(0, NULL);
    for (i=0; i<nn+1; ++i) {
        Tcl_ListObjAppendElement(interp, freq, Tcl_NewDoubleObj(output[2*i]));
        Tcl_ListObjAppendElement(interp, spec, Tcl_NewDoubleObj(output[2*i+1]));
    }
    resultPtr = Tcl_NewListObj(0, NULL);
    Tcl_ListObjAppendElement(interp, resultPtr, freq);
    Tcl_ListObjAppendElement(interp, resultPtr, spec);
    Tcl_SetObjResult(interp, resultPtr);

    /* free intermediate storage */
    Tcl_Free((char *)output);

    return TCL_OK;
}


/* register the plugin with the tcl interpreters */
#if defined(SPECDENTCLDLL_EXPORTS) && defined(_WIN32)
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

EXTERN int Specden_Init(Tcl_Interp *interp)

#else

int Specden_Init(Tcl_Interp *interp)   

#endif
{
  Tcl_CreateObjCommand(interp,"specden",tcl_specden,
        (ClientData)NULL, (Tcl_CmdDeleteProc*)NULL);

  Tcl_PkgProvide(interp, "specden", "1.1");

  check_thread_count(interp,"specden");

  return TCL_OK;
}

