/* 
 * tcl interface for the hesstrans plugin 
 * 
 * Copyright (c) 2007 saam@charite.de
 */

#include <stdio.h>
#include <string.h>
#include <tcl.h>


#include "hesstrans.h"

// It is necessary to declare this as C code because looks for a C-style
// linked object called Hesstrans_Init in hesstrans.so


/* this is the actual interface to the 'hesstrans' command.
 * it parses the arguments, calls the calculation subroutine
 * and then passes the result to the interpreter. */
int tcl_hesstrans(ClientData nodata, Tcl_Interp *interp,
             int objc, Tcl_Obj *const objv[]) 
{
    int ncoor, ndim, nrows, ncols, nbonds, nangles, ndiheds, nimprps;

    Tcl_Obj **coorListObj, **hessListObj, **bondListObj, **angleListObj, **dihedListObj, **imprpListObj;
    int i;


    /* Parse arguments:
     *
     * usage: hesstrans <coords> <hessian> <bonds> ?<angles>? ?<diheds>? ?<imprps>?
     * 
     * <coords>     list of {x y z} coordinates
     * <hessian>    cartesian hessian
     * <bonds>      list of index pairs
     * <angles>     list of index triples
     * <diheds>     list of index quadruples
     * <imprps>     list of index quadruples
     */
    if (objc < 4 || objc > 7) {
        Tcl_WrongNumArgs(interp, 1, objv,
			 "<coords> <hessian> <bonds> ?<angles>? ?<diheds>? ?<imprps>?");
        return TCL_ERROR;
    }
    
    /* parse required arguments one by one. */

    /* cartesian coordinates */
    if (Tcl_ListObjGetElements(interp, objv[1], &ncoor, &coorListObj) != TCL_OK) {
        return TCL_ERROR;
    }
    if (Tcl_ListObjGetElements(interp, objv[2], &nrows, &hessListObj) != TCL_OK) {
        return TCL_ERROR;
    }

    if (3*ncoor!=nrows) {
      char errstring[200];
      sprintf(errstring, "Number of rows %i in Hessian doesn't match number of coordinates, %i", nrows, 3*ncoor);
      Tcl_AppendResult(interp, errstring, NULL);
      return TCL_ERROR;
    }

    if (Tcl_ListObjGetElements(interp, objv[3], &nbonds, &bondListObj) != TCL_OK) {
        return TCL_ERROR;
    }

    /* optional arguments */
    if (objc > 4) {
      if (Tcl_ListObjGetElements(interp, objv[4], &nangles, &angleListObj) != TCL_OK) {
        return TCL_ERROR;
      }
    }

    if (objc > 5) {
      if (Tcl_ListObjGetElements(interp, objv[5], &ndiheds, &dihedListObj) != TCL_OK) {
        return TCL_ERROR;
      }
    }

    if (objc > 6) {
      if (Tcl_ListObjGetElements(interp, objv[6], &nimprps, &imprpListObj) != TCL_OK) {
        return TCL_ERROR;
      }
    }


    /* parse list of cartesian coordinates */ 
    double **cartcoor = new double*[ncoor];
      for (i=0; i<ncoor; ++i) {
      Tcl_Obj **coorObj;
      if (Tcl_ListObjGetElements(interp, coorListObj[i], &ndim, &coorObj) != TCL_OK) {
	return TCL_ERROR;
      }
      int j;
      cartcoor[i] = new double[3];
      //Tcl_GetDoubleFromObj(interp, coorObj[0], cartcoor[i]);
      //Tcl_GetDoubleFromObj(interp, coorObj[1], cartcoor[i]+1);
      //Tcl_GetDoubleFromObj(interp, coorObj[2], cartcoor[i]+2);
      for (j=0; j<ndim; j++) {
      //Tcl_GetDoubleFromObj(interp, coorObj[j], (cartcoor + 3*i+j));
      Tcl_GetDoubleFromObj(interp, coorObj[j], (cartcoor[i] + j));
      printf("%f ", cartcoor[i][j]);
      }
      printf("\n");
    }

    /* parse cartesian hessian matrix */ 
    double *hessian = (double *)Tcl_Alloc(ncoor*3*ncoor*3*sizeof(double));
    Matrix Hc(ncoor*3,ncoor*3);
    for (i=0; i<3*ncoor; i++) {
      Tcl_Obj **hessObj;
      if (Tcl_ListObjGetElements(interp, hessListObj[i], &ncols, &hessObj) != TCL_OK) {
	return TCL_ERROR;
      }
      if (ncols<=i) {
	char errstring[200];
	sprintf(errstring, "Number of elements in row %i of Hessian, %i, doesn't match number of coordinates, %i", i, ncols, 3*ncoor);
	Tcl_AppendResult(interp, errstring, NULL);
	return TCL_ERROR;
      }
      int j;
      for (j=0; j<=i; j++) {
	if (Tcl_GetDoubleFromObj(interp, hessObj[j], &hessian[3*ncoor*i+j]) != TCL_OK) {
	  return TCL_ERROR;
	}      
	if (Tcl_GetDoubleFromObj(interp, hessObj[j], &hessian[3*ncoor*j+i]) != TCL_OK) {
	  return TCL_ERROR;
	}      
      }
    }

    /* parse list of pairs for bonds */ 
    int *bondlist = (int *)Tcl_Alloc(nbonds*2*sizeof(int));
    for (i=0; i<nbonds; ++i) {
        int j;
        Tcl_Obj **bondObj;

        if (Tcl_ListObjGetElements(interp, bondListObj[i], &ndim, &bondObj) != TCL_OK) {
            return TCL_ERROR;
        }
	printf("bond %i: ndim=%i\n", i, ndim);
	if (ndim!=2) {
	  Tcl_AppendResult(interp, "hesstrans: bonds must be defined by 2 atoms", NULL);
	  return TCL_ERROR;
	}
	Tcl_GetIntFromObj(interp, bondObj[0], (bondlist + 2*i));
	Tcl_GetIntFromObj(interp, bondObj[1], (bondlist + 2*i+1));
	for (j=0; j<2; ++j) {
	    printf("%i ", *(bondlist + 2*i+j));
        }
	printf("\n");
    }

    /* parse list of pairs for angles */ 
    int *anglelist = (int *)Tcl_Alloc(nangles*3*sizeof(int));
    for (i=0; i<nangles; ++i) {
        int j;
        Tcl_Obj **angleObj;

        if (Tcl_ListObjGetElements(interp, angleListObj[i], &ndim, &angleObj) != TCL_OK) {
            return TCL_ERROR;
        }
	printf("angle %i: ndim=%i\n", i, ndim);
        for (j=0; j<ndim; ++j) {
            Tcl_GetIntFromObj(interp, angleObj[j], (anglelist + 3*i+j));
	    printf("%i ", *(anglelist + 3*i+j));
        }
	printf("\n");
	if (ndim!=3) {
	  Tcl_AppendResult(interp, "hesstrans: angles must be defined by 3 atoms", NULL);
	  return TCL_ERROR;
	}
    }

    /* parse list of pairs for diheds */ 
    int *dihedlist = (int *)Tcl_Alloc(ndiheds*4*sizeof(int));
    for (i=0; i<ndiheds; ++i) {
        int j;
        Tcl_Obj **dihedObj;

        if (Tcl_ListObjGetElements(interp, dihedListObj[i], &ndim, &dihedObj) != TCL_OK) {
            return TCL_ERROR;
        }
	printf("dihed %i: ndim=%i\n", i, ndim);
	if (ndim!=4) {
	  Tcl_AppendResult(interp, "hesstrans: dihedrals must be defined by 4 atoms", NULL);
	  return TCL_ERROR;
	}

        for (j=0; j<4; ++j) {
            Tcl_GetIntFromObj(interp, dihedObj[j], (dihedlist + 4*i+j));
	    printf("%i ", *(dihedlist + 4*i+j));
        }
 	printf("\n");
   }

    /* parse list of pairs for imprps */ 
    int *imprplist = (int *)Tcl_Alloc(nimprps*4*sizeof(int));
    for (i=0; i<nimprps; ++i) {
        int j;
        Tcl_Obj **imprpObj;

        if (Tcl_ListObjGetElements(interp, imprpListObj[i], &ndim, &imprpObj) != TCL_OK) {
            return TCL_ERROR;
        }
	printf("imprp %i: ndim=%i\n", i, ndim);
	if (ndim!=4) {
	  Tcl_AppendResult(interp, "hesstrans: impropers must be defined by 4 atoms", NULL);
	  return TCL_ERROR;
	}
        for (j=0; j<4; ++j) {
            Tcl_GetIntFromObj(interp, imprpObj[j], (imprplist + 4*i+j));
  	    printf("%i ", *(imprplist + 4*i+j));
      }
 	printf("\n");
    }


    int nintcoor=nbonds+nangles+ndiheds+nimprps;
    double *internalhessian = new double[nintcoor*nintcoor];

    /* do the calculation */
    int ret_val;
    ret_val = getInternalHessian(cartcoor, hessian, bondlist, anglelist, dihedlist, imprplist,
		       ncoor, nbonds, nangles, ndiheds, nimprps, internalhessian); 

    if (ret_val < 0) {
        Tcl_AppendResult(interp, "hesstrans: error in calculation", NULL);
        return TCL_ERROR;
    }
    
    /* free intermediate storage */
    for (i=0; i<ncoor; i++) delete [] cartcoor[i];
    delete [] cartcoor;
    Tcl_Free((char *)hessian);
    Tcl_Free((char *)bondlist);
    Tcl_Free((char *)anglelist);
    Tcl_Free((char *)dihedlist);
    Tcl_Free((char *)imprplist);

    /* prepare results */
    Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
    for (i=0; i<nintcoor; ++i) {
      int j;
      Tcl_Obj *inthessListObj = Tcl_NewListObj(0, NULL);
      for (j=0; j<nintcoor; ++j) {
        Tcl_ListObjAppendElement(interp, inthessListObj, Tcl_NewDoubleObj(internalhessian[nintcoor*i+j]));
      }
      Tcl_ListObjAppendElement(interp, tcl_result, inthessListObj);
    }

    Tcl_SetObjResult(interp, tcl_result);


    /* free intermediate storage */
    delete [] internalhessian;

    return TCL_OK;
}

int tcl_normalmodes(ClientData nodata, Tcl_Interp *interp,
             int objc, Tcl_Obj *const objv[]) 
{
    int ncoor, ndim, nrows, ncols, nmass;

    Tcl_Obj **coorListObj, **hessListObj, **massListObj;
    int i;


    /* Parse arguments:
     *
     * usage: normalmodes <coords> <hessian> <masses>
     * 
     * <coords>     list of {x y z} coordinates
     * <hessian>    cartesian hessian
     * <masses>     list of index pairs
     */
    if (objc < 4 || objc >4) {
        Tcl_WrongNumArgs(interp, 1, objv,
			 "<coords> <hessian> <masses>");
        return TCL_ERROR;
    }
    
    /* parse required arguments one by one. */

    /* cartesian coordinates */
    if (Tcl_ListObjGetElements(interp, objv[1], &ncoor, &coorListObj) != TCL_OK) {
        return TCL_ERROR;
    }
    if (Tcl_ListObjGetElements(interp, objv[2], &nrows, &hessListObj) != TCL_OK) {
        return TCL_ERROR;
    }

    if (3*ncoor!=nrows) {
      char errstring[200];
      sprintf(errstring, "Number of rows %i in Hessian doesn't match number of coordinates, %i", nrows, 3*ncoor);
      Tcl_AppendResult(interp, errstring, NULL);
      return TCL_ERROR;
    }

    if (Tcl_ListObjGetElements(interp, objv[3], &nmass, &massListObj) != TCL_OK) {
        return TCL_ERROR;
    }


    /* parse list of cartesian coordinates */ 
    double **cartcoor = new double*[ncoor];
      for (i=0; i<ncoor; ++i) {
      Tcl_Obj **coorObj;
      if (Tcl_ListObjGetElements(interp, coorListObj[i], &ndim, &coorObj) != TCL_OK) {
	return TCL_ERROR;
      }
      cartcoor[i] = new double[3];
      Tcl_GetDoubleFromObj(interp, coorObj[0], cartcoor[i]);
      Tcl_GetDoubleFromObj(interp, coorObj[1], cartcoor[i]+1);
      Tcl_GetDoubleFromObj(interp, coorObj[2], cartcoor[i]+2);
    }

    /* parse cartesian hessian matrix */ 
    double *hessian = (double *)Tcl_Alloc(ncoor*3*ncoor*3*sizeof(double));
    Matrix Hc(ncoor*3,ncoor*3);
    for (i=0; i<3*ncoor; i++) {
      Tcl_Obj **hessObj;
      if (Tcl_ListObjGetElements(interp, hessListObj[i], &ncols, &hessObj) != TCL_OK) {
	return TCL_ERROR;
      }
      if (ncols<=i) {
	char errstring[200];
	sprintf(errstring, "Number of elements in row %i of Hessian, %i, doesn't match number of coordinates, %i", i, ncols, 3*ncoor);
	Tcl_AppendResult(interp, errstring, NULL);
	return TCL_ERROR;
      }
      int j;
      for (j=0; j<=i; j++) {
	if (Tcl_GetDoubleFromObj(interp, hessObj[j], &hessian[3*ncoor*i+j]) != TCL_OK) {
	  return TCL_ERROR;
	}      
	if (Tcl_GetDoubleFromObj(interp, hessObj[j], &hessian[3*ncoor*j+i]) != TCL_OK) {
	  return TCL_ERROR;
	}      
      }
    }

    /* parse list of atom masses */ 
    double *masslist = (double *)Tcl_Alloc(nmass*sizeof(double));
    for (i=0; i<nmass; ++i) {
      if (Tcl_GetDoubleFromObj(interp, massListObj[i], (masslist + i)) != TCL_OK) {
	return TCL_ERROR;
      }
    }

    double *frequencies = new double[ncoor];
    double *normalmodes = new double[ncoor*ncoor];

    /* do the calculation */
    int ret_val, Nvib=0;
    ret_val = getNormalModes(cartcoor, hessian, masslist, ncoor, frequencies, normalmodes, Nvib, 1); 

    if (ret_val < 0) {
        Tcl_AppendResult(interp, "normalmodes: error in calculation", NULL);
        return TCL_ERROR;
    }
    
    /* free intermediate storage */
    for (i=0; i<ncoor; i++) delete [] cartcoor[i];
    delete [] cartcoor;
    Tcl_Free((char *)hessian);
    Tcl_Free((char *)masslist);


    /* prepare results */
    Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
    Tcl_Obj *freqListObj = Tcl_NewListObj(0, NULL);
    for (i=0; i<Nvib; ++i) {
      Tcl_ListObjAppendElement(interp, freqListObj, Tcl_NewDoubleObj(frequencies[i]));
    }
    Tcl_ListObjAppendElement(interp, tcl_result, freqListObj);

#if 0
    for (i=0; i<ncoor; ++i) {
      int j;
      Tcl_Obj *nmListObj = Tcl_NewListObj(0, NULL);
      for (j=0; j<ncoor; ++j) {
        Tcl_ListObjAppendElement(interp, nmListObj, Tcl_NewDoubleObj(normalmodes[ncoor*i+j]));
      }
      Tcl_ListObjAppendElement(interp, tcl_result, nmListObj);
    }
#endif
    Tcl_SetObjResult(interp, tcl_result);


    /* free intermediate storage */
    delete [] normalmodes;
    if (frequencies) delete [] frequencies;

    return TCL_OK;
}


extern "C" {

/* register the plugin with the tcl interpreters */
#if defined(HESSTRANSTCLDLL_EXPORTS) && defined(_WIN32)
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

EXTERN int Hesstrans_Init(Tcl_Interp *interp)

#else

int Hesstrans_Init(Tcl_Interp *interp)   

#endif
{
  Tcl_CreateObjCommand(interp, "hesstrans", tcl_hesstrans,
        (ClientData)NULL, (Tcl_CmdDeleteProc*)NULL);

  Tcl_CreateObjCommand(interp, "normalmodes", tcl_normalmodes,
        (ClientData)NULL, (Tcl_CmdDeleteProc*)NULL);

  Tcl_PkgProvide(interp, "hesstrans", "1.1");

  return TCL_OK;
}

} // extern "C"
