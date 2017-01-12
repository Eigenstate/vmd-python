/* 
 * tcl interface for the fftcmds plugin 
 * 
 * Copyright 2008-2009 Axel Kohlmeyer <akohlmey@cmm.chem.upenn.edu>
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <tcl.h>

#include "kiss_fft.h"
#include "openmp-util.h"

#define FFT_MAX_DIM 4

TCL_DECLARE_MUTEX(myFftMutex)

/* helper function: read number from list into complex */
static int read_list_cpx(Tcl_Interp *interp, Tcl_Obj *tdata, kiss_fft_cpx *num)
{
    Tcl_Obj **clist;
    int num_el;
    double tmp;
    
    if (Tcl_ListObjGetElements(interp, tdata, &num_el, &clist) != TCL_OK) {
        return TCL_ERROR;
    }
    if (num_el == 2) {        /* complex data */
        Tcl_GetDoubleFromObj(interp, clist[0], &tmp);
        num->r=(kiss_fft_scalar)tmp;
        Tcl_GetDoubleFromObj(interp, clist[1], &tmp);
        num->i=(kiss_fft_scalar)tmp;
    } else if (num_el == 1) { /* real only data. set phase to zero. */
        Tcl_GetDoubleFromObj(interp, clist[0], &tmp);
        num->r=(kiss_fft_scalar)tmp;
        num->i=0.0;
    } else {                  /* unknown data format */
        return TCL_ERROR;
    }
    return TCL_OK;
}

/* helper function: create tcl list from complex number */
static Tcl_Obj *make_list_cpx(Tcl_Interp *interp, Tcl_Obj *list, kiss_fft_cpx *num)
{
    Tcl_Obj *cmplx;
    cmplx = Tcl_NewListObj(0, NULL);
    Tcl_ListObjAppendElement(interp, cmplx, Tcl_NewDoubleObj(num->r));
    Tcl_ListObjAppendElement(interp, cmplx, Tcl_NewDoubleObj(num->i));
    Tcl_ListObjAppendElement(interp, list, cmplx);
    return list;
}


/* helper function: recurse through lists to get to the data */
static int read_list_list(Tcl_Interp *interp, Tcl_Obj *tdata, int curdim, int ndim, 
                         int *ndat, kiss_fft_cpx *input, int *alldim) 
{
    int i,num_el;
    Tcl_Obj **clist;
        
    if (Tcl_ListObjGetElements(interp, tdata, &num_el, &clist) != TCL_OK) {
        return TCL_ERROR;
    }
    if (num_el != ndat[curdim]) { /* consistency check. all lists must be the same length */
        return TCL_ERROR;
    }
    if (ndim == curdim+1) {     /* end of recursion. read numbers and increment counter accordingly */
        for (i=0; i<num_el; ++i) {
            if (read_list_cpx(interp, clist[i], input + *alldim) != TCL_OK) {
                return TCL_ERROR;
            }
            ++(*alldim);
        }
    } else {  /* recurse into next dimension after consistency check. */
        if (curdim+1 > ndim) return TCL_ERROR;
        for (i=0; i<num_el; ++i) {
            if (read_list_list(interp, clist[i], curdim+1, ndim, ndat, input, alldim) != TCL_OK) {
                return TCL_ERROR;
            }
        }
    }
    return TCL_OK;
}

/* helper function: build tcl list of lists recursively */
static Tcl_Obj *make_list_list(Tcl_Interp *interp, Tcl_Obj *list, int curdim, 
                               int ndim, int *ndat, kiss_fft_cpx *output, int *alldim) 
{
    Tcl_Obj *clist;
    int i;
    clist = Tcl_NewListObj(0, NULL);
        
    if (ndim == curdim+1) {
        for (i=0; i<ndat[curdim]; ++i) {
            make_list_cpx(interp, clist, output + *alldim);
            ++(*alldim);
        }
    } else {
        for (i=0; i<ndat[curdim]; ++i) {
            make_list_list(interp, clist, curdim+1, ndim, ndat, output, alldim);
        }
    }
    Tcl_ListObjAppendElement(interp, list, clist);
    return TCL_OK;
}

/* generic complex 1d-transform. */
int tcl_cfft_1d(ClientData nodata, Tcl_Interp *interp,
                int objc, Tcl_Obj *const objv[]) 
{
    Tcl_Obj *result, **tdata;
    
    const char *name;
    kiss_fft_cpx *input;
    kiss_fft_cpx *output;
    kiss_fft_cfg work;
    
    int dir, ndat, k;

    /* thread safety */
    Tcl_MutexLock(&myFftMutex);

    /* set defaults: */
    dir   = FFT_FORWARD;
    ndat  = -1;
    
    /* Parse arguments:
     *
     * usage: cfftf_1d <data>
     *    or: cfftb_1d <data>
     * 
     * cfftf_1d   : is the 1d complex forward transform.
     * cfftb_1d   : is the 1d complex backward transform.
     * <data>     : list containing data to be transformed. this can either a real 
     *              or a list with two reals interpreted as complex.
     */

    name = Tcl_GetString(objv[0]);
    if (strcmp(name,"cfftf_1d") == 0) {
        dir = FFT_FORWARD;
    } else if (strcmp(name,"cfftb_1d") == 0) {
        dir = FFT_BACKWARD;
    } else {
        Tcl_AppendResult(interp, name, ": unknown fft command.", NULL);
        Tcl_MutexUnlock(&myFftMutex);
        return TCL_ERROR;
    }

    if (objc != 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "<data>");
        Tcl_MutexUnlock(&myFftMutex);
        return TCL_ERROR;
    }
    
    /* get handle on data  and check */
    Tcl_IncrRefCount(objv[1]);
    if (Tcl_ListObjGetElements(interp, objv[1], &ndat, &tdata) != TCL_OK) {
        Tcl_DecrRefCount(objv[1]);
        Tcl_MutexUnlock(&myFftMutex);
        return TCL_ERROR;
    }
    if (ndat < 0) { /* this should not happen, but... */
        Tcl_AppendResult(interp, name, ": illegal data array.", NULL);
        Tcl_DecrRefCount(objv[1]);
        Tcl_MutexUnlock(&myFftMutex);
        return TCL_ERROR;
    }
    if ((ndat == 0) || (ndat == 1)) { /* no effect for zero or one element */
        Tcl_DecrRefCount(objv[1]);
        Tcl_SetObjResult(interp, objv[1]);
        Tcl_MutexUnlock(&myFftMutex);
        return TCL_OK;
    }
    
    check_thread_count(interp,"fftcmds");

    /* get dynamic storage for passing data to the lowlevel code. */
    input  = (void *)Tcl_Alloc(ndat*sizeof(kiss_fft_cpx));
    output = (void *)Tcl_Alloc(ndat*sizeof(kiss_fft_cpx));
    work   = kiss_fft_alloc(ndat, dir, NULL, NULL);
    
    /* parse/copy data list */
    for (k=0; k<ndat; ++k) {
        if (read_list_cpx(interp, tdata[k], input + k) != TCL_OK) {
            Tcl_AppendResult(interp, name, ": illegal data array.", NULL);
            Tcl_DecrRefCount(objv[1]);
            Tcl_MutexUnlock(&myFftMutex);
            return TCL_ERROR;
        }
    }
    Tcl_DecrRefCount(objv[1]);
    
    /* finally run the transform */
    kiss_fft(work, input, output);

    /* prepare results */
    result = Tcl_NewListObj(0, NULL);
    for (k=0; k<ndat; ++k) {
        make_list_cpx(interp, result, output + k);
    }
    Tcl_SetObjResult(interp, result);

    /* free intermediate storage */
    Tcl_Free((char *)input);
    Tcl_Free((char *)output);
    kiss_fft_free(work);
    kiss_fft_cleanup();
    
    Tcl_MutexUnlock(&myFftMutex);
    return TCL_OK;
}

/* generic complex <N>d-transform. */
int tcl_cfft_nd(ClientData nodata, Tcl_Interp *interp,
                int objc, Tcl_Obj *const objv[]) 
{
    Tcl_Obj *result, **tdata[FFT_MAX_DIM];
    
    const char *name;
    kiss_fft_cpx *input;
    kiss_fft_cpx *output;
    kiss_fftnd_cfg work;
    
    int dir, ndim, alldim, ndat[FFT_MAX_DIM];
    int i;

    Tcl_MutexLock(&myFftMutex);

    /* set defaults: */
    dir   = FFT_FORWARD;
    ndim  = -1;
        
    /* Parse arguments:
     *
     * usage: cfftf_nd <data>
     *    or: cfftb_nd <data>
     * 
     * cfftf_nd   : is the Nd complex forward transform.
     * cfftb_nd   : is the Nd complex backward transform.
     * <data>     : list containing data to be transformed. this can either a real 
     *              or a list with two reals interpreted as complex.
     */

    name = Tcl_GetString(objv[0]);
    if (strcmp(name,"cfftf_2d") == 0) {
        dir = FFT_FORWARD;
        ndim = 2;
    } else if (strcmp(name,"cfftb_2d") == 0) {
        dir = FFT_BACKWARD;
        ndim = 2;
    } else if (strcmp(name,"cfftf_3d") == 0) {
        dir = FFT_FORWARD;
        ndim = 3;
    } else if (strcmp(name,"cfftb_3d") == 0) {
        dir = FFT_BACKWARD;
        ndim = 3;
    } else if (strcmp(name,"cfftf_4d") == 0) {
        dir = FFT_FORWARD;
        ndim = 4;
    } else if (strcmp(name,"cfftb_4d") == 0) {
        dir = FFT_BACKWARD;
        ndim = 4;
    } else {
        Tcl_AppendResult(interp, name, ": unknown fft command.", NULL);
        Tcl_MutexUnlock(&myFftMutex);
        return TCL_ERROR;
    }

    if (objc != 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "<data>");
        Tcl_MutexUnlock(&myFftMutex);
        return TCL_ERROR;
    }
    
    /* mark data as busy and check */
    Tcl_IncrRefCount(objv[1]);
    if (Tcl_ListObjGetElements(interp, objv[1], &(ndat[0]), &(tdata[0])) != TCL_OK) {
        Tcl_DecrRefCount(objv[1]);
        Tcl_MutexUnlock(&myFftMutex);
        return TCL_ERROR;
    }
    if ((ndat[0] < 0) || (ndim > FFT_MAX_DIM)) { /* this should not happen, but... */
        Tcl_AppendResult(interp, name, ": illegal or unsupported data array.", NULL);
        Tcl_DecrRefCount(objv[1]);
        Tcl_MutexUnlock(&myFftMutex);
        return TCL_ERROR;
    }
    if (ndat[0] == 0) {         /* no effect for empty array */
        Tcl_DecrRefCount(objv[1]);
        Tcl_SetObjResult(interp, objv[1]);
        Tcl_MutexUnlock(&myFftMutex);
        return TCL_OK;
    }

    check_thread_count(interp,"fftcmds");

    /* determine size of each dimension for storage size and parsing/checking. */
    alldim=ndat[0];
    for (i=1; i<ndim; ++i) { 
        if (Tcl_ListObjGetElements(interp, tdata[i-1][0], &(ndat[i]), &(tdata[i])) != TCL_OK) {
            Tcl_DecrRefCount(objv[1]);
            Tcl_MutexUnlock(&myFftMutex);
            return TCL_ERROR;
        }
        alldim *= ndat[i];
    }
    input  = (void *)Tcl_Alloc(alldim*sizeof(kiss_fft_cpx));
    output = (void *)Tcl_Alloc(alldim*sizeof(kiss_fft_cpx));
    work   = kiss_fftnd_alloc(ndat, ndim, dir, NULL, NULL);

    /* parse/copy data list through recursive function and release original data. */
    alldim=0;
    for (i=0; i<ndat[0]; ++i) {
        if (read_list_list(interp, tdata[0][i], 1, ndim, ndat, input, &alldim) != TCL_OK) {
            Tcl_AppendResult(interp, name, ": illegal data array.", NULL);
            Tcl_DecrRefCount(objv[1]);
            Tcl_MutexUnlock(&myFftMutex);
            return TCL_ERROR;
        }
    }
    Tcl_DecrRefCount(objv[1]);
    
    /* finally run the transform */
    kiss_fftnd(work, input, output);
    
    /* build result list(s) recursively */
    result = Tcl_NewListObj(0, NULL);
    alldim = 0;
    for (i=0; i<ndat[0]; ++i) {
        make_list_list(interp, result, 1, ndim, ndat, output, &alldim);
    }
    Tcl_SetObjResult(interp, result);

    /* free intermediate storage */
    Tcl_Free((char *)input);
    Tcl_Free((char *)output);
    kiss_fft_free(work);
    kiss_fft_cleanup();

    Tcl_MutexUnlock(&myFftMutex);
    return TCL_OK;
}


/* real-to-complex transform in 1 dimension */
int tcl_rfft_1d(ClientData nodata, Tcl_Interp *interp,
                int objc, Tcl_Obj *const objv[]) 
{
    Tcl_Obj *result, **tdata;
    
    const char *name;
    kiss_fft_scalar *timed;
    kiss_fft_cpx    *freqd;
    kiss_fftr_cfg    work;
    
    int dir, ndat, k;

    /* thread safety */
    Tcl_MutexLock(&myFftMutex);

    /* set defaults: */
    dir   = FFT_FORWARD;
    ndat  = -1;
    
    /* Parse arguments:
     *
     * usage: r2cfft_1d <data>
     *    or: c2rfft_1d <data>
     * 
     * r2cfftf_1d : is the 1d real-to-complex forward transform.
     * c2rfftb_1d : is the 1d complex-to-real backward transform.
     * <data>     : list containing data to be transformed. this can either a real 
     *              or a list with two reals interpreted as complex.
     */

    name = Tcl_GetString(objv[0]);
    if (strcmp(name,"r2cfft_1d") == 0) {
        dir = FFT_FORWARD;
    } else if (strcmp(name,"c2rfft_1d") == 0) {
        dir = FFT_BACKWARD;
    } else {
        Tcl_AppendResult(interp, name, ": unknown fft command.", NULL);
        Tcl_MutexUnlock(&myFftMutex);
        return TCL_ERROR;
    }

    if (objc != 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "<data>");
        Tcl_MutexUnlock(&myFftMutex);
        return TCL_ERROR;
    }
    
    /* get handle on data */
    Tcl_IncrRefCount(objv[1]);
    if (Tcl_ListObjGetElements(interp, objv[1], &ndat, &tdata) != TCL_OK) {
        Tcl_DecrRefCount(objv[1]);
        Tcl_MutexUnlock(&myFftMutex);
        return TCL_ERROR;
    }
    if (ndat < 0) {             /* this should not happen, but... */
        Tcl_AppendResult(interp, name, ": illegal data array.", NULL);
        Tcl_DecrRefCount(objv[1]);
        Tcl_MutexUnlock(&myFftMutex);
        return TCL_ERROR;
    }
    /* no effect for zero or one element */
    if ((ndat == 0) || (ndat == 1)) {
        Tcl_DecrRefCount(objv[1]);
        Tcl_SetObjResult(interp, objv[1]);
        Tcl_MutexUnlock(&myFftMutex);
        return TCL_OK;
    }

    /* we need an even number of data points for the forward transform */
    if (ndat & 1) {
        if (dir == FFT_FORWARD) {
            Tcl_AppendResult(interp, name, " needs an even number of data points.", NULL);
            Tcl_DecrRefCount(objv[1]);
            Tcl_MutexUnlock(&myFftMutex);
            return TCL_ERROR;
        }
    }

    check_thread_count(interp,"fftcmds");

    /* size of data arrays for backward transform */
    if (dir == FFT_BACKWARD) ndat = (ndat-1)*2;
    
    /* get dynamic storage for passing data to the lowlevel code. */
    timed = (void *)Tcl_Alloc(ndat*sizeof(kiss_fft_scalar));
    freqd = (void *)Tcl_Alloc((ndat/2+1)*sizeof(kiss_fft_cpx));
    work  = kiss_fftr_alloc(ndat, dir, NULL, NULL);
    
    /* parse/copy data list */
    if (dir == FFT_FORWARD) {
        for (k=0; k<ndat; ++k) {
            if (Tcl_GetDoubleFromObj(interp, tdata[k], timed + k) != TCL_OK) {
                Tcl_AppendResult(interp, name, ": illegal data array.", NULL);
                Tcl_DecrRefCount(objv[1]);
                Tcl_MutexUnlock(&myFftMutex);
                return TCL_ERROR;
            }
        }
    } else {
        for (k=0; k<(ndat/2)+1; ++k) {
            if (read_list_cpx(interp, tdata[k], freqd + k) != TCL_OK) {
                Tcl_AppendResult(interp, name, ": illegal data array.", NULL);
                Tcl_DecrRefCount(objv[1]);
                Tcl_MutexUnlock(&myFftMutex);
                return TCL_ERROR;
            }
        }
    }
    Tcl_DecrRefCount(objv[1]);
    
    /* finally run the transform */
    if (dir == FFT_FORWARD) {
        kiss_fftr(work, timed, freqd);
    } else {
        kiss_fftri(work, freqd, timed);
    }

    /* prepare results */
    result = Tcl_NewListObj(0, NULL);
    if (dir == FFT_FORWARD) {
        for (k=0; k<(ndat/2)+1; ++k) {
            make_list_cpx(interp, result, freqd + k);
        }
    } else {
        for (k=0; k<ndat; ++k) {
            Tcl_ListObjAppendElement(interp, result, Tcl_NewDoubleObj(timed[k]));
        }
    }
    Tcl_SetObjResult(interp, result);

    /* free intermediate storage */
    Tcl_Free((char *)timed);
    Tcl_Free((char *)freqd);
    kiss_fft_free(work);
    kiss_fft_cleanup();

    Tcl_MutexUnlock(&myFftMutex);
    return TCL_OK;
}

/* register the plugin with the tcl interpreters */
#if defined(FFTCMDSTCLDLL_EXPORTS) && defined(_WIN32)
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

EXTERN int Fftcmds_Init(Tcl_Interp *interp)

#else

int Fftcmds_Init(Tcl_Interp *interp)   

#endif
{
  Tcl_CreateObjCommand(interp,"r2cfft_1d",tcl_rfft_1d,
        (ClientData)NULL, (Tcl_CmdDeleteProc*)NULL);
  Tcl_CreateObjCommand(interp,"c2rfft_1d",tcl_rfft_1d,
        (ClientData)NULL, (Tcl_CmdDeleteProc*)NULL);
  Tcl_CreateObjCommand(interp,"cfftf_1d",tcl_cfft_1d,
        (ClientData)NULL, (Tcl_CmdDeleteProc*)NULL);
  Tcl_CreateObjCommand(interp,"cfftb_1d",tcl_cfft_1d,
        (ClientData)NULL, (Tcl_CmdDeleteProc*)NULL);
  Tcl_CreateObjCommand(interp,"cfftf_2d",tcl_cfft_nd,
        (ClientData)NULL, (Tcl_CmdDeleteProc*)NULL);
  Tcl_CreateObjCommand(interp,"cfftb_2d",tcl_cfft_nd,
        (ClientData)NULL, (Tcl_CmdDeleteProc*)NULL);
  Tcl_CreateObjCommand(interp,"cfftf_3d",tcl_cfft_nd,
        (ClientData)NULL, (Tcl_CmdDeleteProc*)NULL);
  Tcl_CreateObjCommand(interp,"cfftb_3d",tcl_cfft_nd,
        (ClientData)NULL, (Tcl_CmdDeleteProc*)NULL);
  Tcl_CreateObjCommand(interp,"cfftf_4d",tcl_cfft_nd,
        (ClientData)NULL, (Tcl_CmdDeleteProc*)NULL);
  Tcl_CreateObjCommand(interp,"cfftb_4d",tcl_cfft_nd,
        (ClientData)NULL, (Tcl_CmdDeleteProc*)NULL);

  Tcl_PkgProvide(interp, "fftcmds", "1.1");

  check_thread_count(interp,"fftcmds");

  return TCL_OK;
}
