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
 *	$RCSfile: TclVolMap.C,v $
 *	$Author: saam $	$Locker:  $		$State: Exp $
 *	$Revision: 1.115 $	$Date: 2011/03/05 21:32:15 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * These are essentially just Tcl wrappers for the volmap commands in
 * VolMapCreate.C.
 *
 ***************************************************************************/

#include <math.h>
#include <stdlib.h>
#include <tcl.h>
#include "TclCommands.h"
#include "AtomSel.h"
#include "VMDApp.h"
#include "MoleculeList.h"
#include "config.h"
#include "Measure.h"
#include "VolumetricData.h"
#include "VolMapCreate.h"
#include "Inform.h"
#include "MeasureSymmetry.h"

// XXX Todo List: 
// - Make this into an independent Tcl function 
//  (e.g.: "volmap new density <blah...>"
// - Make friends with as-of-yet non-existing VMD volumetric map I/O
// - parse and allow user to set useful params

// Function: volmap <maptype> <selection> [-weight <none|atom value|string array>] 
// Options for all maptypes:
//   -allframes
//   -res <res>
// Options for "density":
//   -weight <none|atom value|string array>
//   -scale <radius scale>


static int parse_minmax_args(Tcl_Interp *interp, int arg_minmax,
                             Tcl_Obj * const objv[], double *minmax) {
  int num, i;
  Tcl_Obj **data, **vecmin, **vecmax;
    
  // get the list containing minmax
  if (Tcl_ListObjGetElements(interp, objv[arg_minmax], &num, &data) != TCL_OK) {
    Tcl_SetResult(interp, (char *)"volmap: could not read parameter (-minmax)", TCL_STATIC);
    return TCL_ERROR;
  }
  if (num != 2) {
    Tcl_SetResult(interp, (char *)"volmap: minmax requires a list with two vectors (-minmax)", TCL_STATIC);
    return TCL_ERROR;
  }
  // get the list containing min
  if (Tcl_ListObjGetElements(interp, data[0], &num, &vecmin) != TCL_OK) {
    return TCL_ERROR;
  }
  if (num != 3) {
    Tcl_SetResult(interp, (char *)"volmap: the first vector does not contain 3 elements (-minmax)", TCL_STATIC);
    return TCL_ERROR;
  }
  // get the list containing max
  if (Tcl_ListObjGetElements(interp, data[1], &num, &vecmax) != TCL_OK) {
    return TCL_ERROR;
  }
  if (num != 3) {
    Tcl_SetResult(interp, (char *)"volmap: the second vector does not contain 3 elements (-minmax)", TCL_STATIC);
    return TCL_ERROR;
  }

  // read min
  for (i=0; i<3; i++)
    if (Tcl_GetDoubleFromObj(interp, vecmin[i], minmax+i) != TCL_OK)
      return TCL_ERROR;
    
  // read max
  for (i=0; i<3; i++)
    if (Tcl_GetDoubleFromObj(interp, vecmax[i], minmax+i+3) != TCL_OK)
      return TCL_ERROR;
    
  // Check that range is valid...
  if (minmax[0] >= minmax[3] || minmax[1] >= minmax[4] || minmax[2] >= minmax[5]) {
    Tcl_SetResult(interp, (char *)"volmap: invalid minmax range (-minmax)", TCL_STATIC);
    return TCL_ERROR;
  }

  return TCL_OK;
}


static int vmd_volmap_ils(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  bool bad_usage = false;
  bool bailout = false;
  double cutoff       = 10.;
  double resolution   = 1.;
  double minmax[6];
  bool pbc = false;
  bool pbcbox = false;
  float *pbccenter = NULL;

  int maskonly = 0;
  bool export_to_file = false;
  int molid = -1;
  char *filebase = NULL;
  char *filename = NULL;

  int    nprobecoor   = 0;
  int    nprobevdw    = 0;
  int    nprobecharge = 0;
  float *probe_vdwrmin = NULL;
  float *probe_vdweps  = NULL;
  float *probe_coords  = NULL;
  float *probe_charge  = NULL;
  double temperature = 300.0;
  double maxenergy   = 150.0;
  int subres = 3;
  int num_conf = 0;  // number of ligand rotamers
  AtomSel *probesel = NULL;
  AtomSel *alignsel = NULL;
  Matrix4 *transform = NULL;

  if (argc<3) {
    Tcl_WrongNumArgs(interp, 2, objv-1, (char *)"<molid> <minmax> [options...]");
    return TCL_ERROR;
  }

  // Get the molecule ID
  if (!strcmp(Tcl_GetStringFromObj(objv[1], NULL), "top"))
    molid = app->molecule_top(); 
  else 
    Tcl_GetIntFromObj(interp, objv[1], &molid);
  
  if (!app->molecule_valid_id(molid)) {
    Tcl_AppendResult(interp, "volmap: molecule specified for ouput is invalid. (-mol)", NULL);
    return TCL_ERROR;
  }

  // Get the minmax box or the pbcbox keyword
  if (!strcmp(Tcl_GetStringFromObj(objv[2], NULL), "pbcbox")) {
    pbcbox = true;
  }
  else if (parse_minmax_args(interp, 2, objv, minmax)!=TCL_OK) {
    Tcl_AppendResult(interp, "volmap: no atoms selected.", NULL);
  }

  int first = 0;
  int nframes = app->molecule_numframes(molid);
  int last = nframes-1;

  // Parse optional arguments
  int arg;
  for (arg=3; arg<argc && !bad_usage && !bailout; arg++) {
    // Arguments common to all volmap types
    if (!strcmp(Tcl_GetStringFromObj(objv[arg], NULL), "-res")) {
      if (arg+1>=argc) { bad_usage=true; break; }
      Tcl_GetDoubleFromObj(interp, objv[arg+1], &resolution);
      if (resolution <= 0.f) {
        Tcl_AppendResult(interp, "volmap ils: resolution must be positive. (-res)", NULL);
        bailout = true; break;
      }
      arg++;
    }
    else if (!strcmp(Tcl_GetStringFromObj(objv[arg], NULL), "-probesel")) {
      if (arg+1>=argc) { bad_usage=true; break; }
      // Get atom selection for probe
      probesel = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[arg+1],NULL));
      if (!probesel) {
        Tcl_AppendResult(interp, "volmap ils -probesel: no atom selection.", NULL);
        bailout = true; break;
      }
      if (!probesel->selected) {
        Tcl_AppendResult(interp, "volmap ils -probesel: no atoms selected.", NULL);
        bailout = true; break;
      }
      if (!app->molecule_valid_id(probesel->molid())) {
        Tcl_AppendResult(interp, "volmap ils -probesel: ",
                         measure_error(MEASURE_ERR_NOMOLECULE), NULL);
        bailout = true; break;
      }
      arg++;
    }
    // ILS specific arguments
    else if (!strcmp(Tcl_GetStringFromObj(objv[arg], NULL), "-first")) {
      if (arg+1>=argc) { bad_usage=true; break; }
      Tcl_GetIntFromObj(interp, objv[arg+1], &first);
      if (first < 0) {
        Tcl_AppendResult(interp, "volmap ils: Frame specified with -first must be positive.", NULL);
        bailout = true; break;
      }
      if (first >= nframes) {
        Tcl_AppendResult(interp, "volmap ils: Frame specified with -first exceeds number of existing frames.", NULL);
        bailout = true; break;
      }
      arg++;
    }
    else if (!strcmp(Tcl_GetStringFromObj(objv[arg], NULL), "-last")) {
      if (arg+1>=argc) { bad_usage=true; break; }
      Tcl_GetIntFromObj(interp, objv[arg+1], &last);
      if (last < 0) {
        Tcl_AppendResult(interp, "volmap ils: Frame specified with -last must be positive.", NULL);
        bailout = true; break;
      }
      if (last >= nframes) {
        Tcl_AppendResult(interp, "volmap ils: Frame specified with -last exceeds number of existing frames.", NULL);
        bailout = true; break;
      }
      arg++;
    }
    else if (!strcmp(Tcl_GetStringFromObj(objv[arg], NULL), "-cutoff")) {
      if (arg+1 >= argc) { bad_usage=true; break; }
      Tcl_GetDoubleFromObj(interp, objv[arg+1], &cutoff);
      if (cutoff <= 0.) {
        Tcl_AppendResult(interp, "volmap ils: cutoff must be positive. (-cutoff)", NULL);
        bailout = true;	break;
      }
      arg++;
    }
    else if (!strcmp(Tcl_GetStringFromObj(objv[arg], NULL), "-dx") ||
             !strcmp(Tcl_GetStringFromObj(objv[arg], NULL), "-o")) {
      if (arg+1>=argc) { bad_usage=true; break; }
      const char* outputfilename = Tcl_GetString(objv[arg+1]);
      if (outputfilename) {
        filebase = new char[strlen(outputfilename)+1];
        strcpy(filebase, outputfilename);
        export_to_file = true;
      }
      arg++;
    }

    else if (!strcmp(Tcl_GetStringFromObj(objv[arg], NULL), "-probecoor")) {
      if (arg+1>=argc) { bad_usage=true; break; }
      int i;
      double tmp;
      Tcl_Obj **listObj;
      if (Tcl_ListObjGetElements(interp, objv[arg+1], &nprobecoor, &listObj) != TCL_OK) {
        Tcl_AppendResult(interp, " volmap ils: bad syntax in probecoor!", NULL);
        bailout = true;	break;
      }
      if (!nprobecoor) {
        Tcl_AppendResult(interp, " volmap ils: Empty probecoor list!", NULL);
        bailout = true;	break;
      }

      probe_coords = new float[3*nprobecoor];

      for (i=0; i<nprobecoor; i++) {
        Tcl_Obj **coorObj;
        int j, ndim = 0;
        if (Tcl_ListObjGetElements(interp, listObj[i], &ndim, &coorObj) != TCL_OK) {
          Tcl_AppendResult(interp, " volmap ils: bad syntax in probecoor!", NULL);
          bailout = true;	break;
        }

        if (ndim!=3) {
          Tcl_AppendResult(interp, " volmap ils: need three values for each probecoor vector", NULL);
          bailout = true;	break;
        }
      
        for (j=0; j<3; j++) {
          if (Tcl_GetDoubleFromObj(interp, coorObj[j], &tmp) != TCL_OK) {
            Tcl_AppendResult(interp, " volmap ils: non-numeric in probecoor", NULL);
            bailout = true;	break;
          }
          probe_coords[3*i+j] = (float)tmp;
        }
      }
      arg++;
    }
    else if (!strcmp(Tcl_GetStringFromObj(objv[arg], NULL), "-probevdw")) {
      if (arg+1>=argc) { bad_usage=true; break; }
      double tmp;
      Tcl_Obj **listObj;
      if (Tcl_ListObjGetElements(interp, objv[arg+1], &nprobevdw, &listObj) != TCL_OK) {
        Tcl_AppendResult(interp, " volmap ils: bad syntax in probevdw", NULL);
        bailout = true;	break;
      }

      probe_vdweps  = new float[nprobevdw];
      probe_vdwrmin = new float[nprobevdw];

      int i;
      for (i=0; i<nprobevdw; i++) {
        Tcl_Obj **vdwObj;
        int ndim = 0;
        if (Tcl_ListObjGetElements(interp, listObj[i], &ndim, &vdwObj) != TCL_OK) {
          Tcl_AppendResult(interp, " volmap ils: bad syntax in probevdw", NULL);
          bailout = true;	break;
        }

        if (ndim!=2) {
          Tcl_AppendResult(interp, " volmap ils: Need two probevdw values (eps, rmin) for each atom", NULL);
          bailout = true;	break;
        }
      
        if (Tcl_GetDoubleFromObj(interp, vdwObj[0], &tmp) != TCL_OK) {
          Tcl_AppendResult(interp, " volmap ils: Non-numeric in probevdw (eps)", NULL);
          bailout = true;	break;
        }
        probe_vdweps[i] = (float)tmp;
        
        if (Tcl_GetDoubleFromObj(interp, vdwObj[1], &tmp) != TCL_OK) {
          Tcl_AppendResult(interp, " volmap ils: Non-numeric in probevdw (rmin)", NULL);
          bailout = true;	break;
        }
        probe_vdwrmin[i] = (float)tmp;        
      }
      arg++;
    }

    else if (!strcmp(Tcl_GetStringFromObj(objv[arg], NULL), "-probecharge")) {
      if (arg+1>=argc) { bad_usage=true; break; }
      int i;
      double tmp;
      Tcl_Obj **listObj;
      if (Tcl_ListObjGetElements(interp, objv[arg+1], &nprobecharge, &listObj) != TCL_OK) {
        Tcl_AppendResult(interp, " volmap ils: bad syntax in probecharge", NULL);
        bailout = true;	break;
      }

      probe_charge = new float[nprobecharge];

      for (i=0; i<nprobecharge; i++) {
        if (Tcl_GetDoubleFromObj(interp, listObj[0], &tmp) != TCL_OK) {
          Tcl_AppendResult(interp, " volmap ils: non-numeric in probecharge", NULL);
          bailout = true; break;
        }
        probe_charge[i] = (float)tmp;
      }
      arg++;
    }

    else if (!strcmp(Tcl_GetStringFromObj(objv[arg], NULL), "-orient") ||
             !strcmp(Tcl_GetStringFromObj(objv[arg], NULL), "-conf")) {
      if (arg+1>=argc) { bad_usage=true; break; }
      Tcl_GetIntFromObj(interp, objv[arg+1], &num_conf);
      if (num_conf < 0) {
        Tcl_AppendResult(interp, "volmap ils: invalid -orient parameter", NULL);
        bailout = true; break;
      }
      arg++;
    }
    else if (!strcmp(Tcl_GetStringFromObj(objv[arg], NULL), "-subres")) {
      if (arg+1>=argc) { bad_usage=true; break; }
      Tcl_GetIntFromObj(interp, objv[arg+1], &subres);
      if (subres < 1) {
        Tcl_AppendResult(interp, "volmap ils: invalid -subres parameter", NULL);
        bailout = true; break;
      }
      arg++;
    }
    else if (!strcmp(Tcl_GetStringFromObj(objv[arg], NULL), "-T")) {
      if (arg+1>=argc) { bad_usage=true; break; }
      Tcl_GetDoubleFromObj(interp, objv[arg+1], &temperature);
      arg++;
    }    
    else if (!strcmp(Tcl_GetStringFromObj(objv[arg], NULL), "-maxenergy")) {
      if (arg+1>=argc) { bad_usage=true; break; }
      if (Tcl_GetDoubleFromObj(interp, objv[arg+1], &maxenergy) != TCL_OK) {
        Tcl_AppendResult(interp, "volmap ils: invalid -maxenergy parameter", NULL);
        bailout = true; break;
      }
      arg++;
    }    
    else if (!strcmp(Tcl_GetStringFromObj(objv[arg], NULL), "-alignsel")) {
      if (arg+1>=argc) { bad_usage=true; break; }
      alignsel = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[arg+1],NULL));
      if (!alignsel) {
        Tcl_AppendResult(interp, "volmap ils: no atom selection for alignment.", NULL);
        bailout = true; break;
      }
      if (!alignsel->selected) {
        Tcl_AppendResult(interp, "volmap ils: no atoms selected for alignment.", NULL);
        bailout = true; break;
      }
      arg++;
    }    
    else if (!strcmp(Tcl_GetStringFromObj(objv[arg], NULL), "-transform")) {
      if (arg+1>=argc) { bad_usage=true; break; }
      transform = new Matrix4;
      tcl_get_matrix("volmap ils: ", interp, objv[arg+1], transform->mat);
      arg++;
    }

    else if (!strcmp(Tcl_GetStringFromObj(objv[arg], NULL), "-pbc")) {
      pbc = true;
    }

    else if (!strcmp(Tcl_GetStringFromObj(objv[arg], NULL), "-pbccenter")) {
      if (arg+1>=argc) { bad_usage=true; break; }
      int i, ndim = 0;
      double tmp;
      Tcl_Obj **centerObj;
      if (Tcl_ListObjGetElements(interp, objv[arg+1], &ndim, &centerObj) != TCL_OK) {
        Tcl_AppendResult(interp, " volmap ils: bad syntax in pbccenter", NULL);
        bailout = true; break;
      }
      
      if (ndim!=3) {
        Tcl_AppendResult(interp, " volmap ils: need three values for vector pbccenter", NULL);
        bailout = true; break;
      }
      
      pbccenter = new float[3];
      for (i=0; i<3; i++) {
        if (Tcl_GetDoubleFromObj(interp, centerObj[i], &tmp) != TCL_OK) {
          Tcl_AppendResult(interp, " volmap ils: non-numeric in pbccenter", NULL);
          bailout = true; break;
        }
        pbccenter[i] = (float)tmp;
      }
      arg++;
      pbc = true;
    }

    else if (!strcmp(Tcl_GetStringFromObj(objv[arg], NULL), "-maskonly")) {
      maskonly = 1;
    }

    else {
      // unknown arg
      Tcl_AppendResult(interp, " volmap ils: unknown argument ",
                       Tcl_GetStringFromObj(objv[arg], NULL), NULL);
      bailout = true;
    }

  }
  
  if (bad_usage) {
    Tcl_WrongNumArgs(interp, 2, objv-1, (char *)"<molid> <minmax> [options...]");
    bailout = true;
  }

  int num_probe_atoms = 0;
  if (!bailout) {
    if (probesel) {
      if (nprobecoor) {
        Tcl_AppendResult(interp, "volmap ils: Must specify either -probesel or -probecoor not both.",
                         NULL);
        bailout = true;
      }
      
      // The probe was specified in form of an atomselection
      Molecule *probemol = app->moleculeList->mol_from_id(probesel->molid());
      const float *radius = probemol->extraflt.data("radius");
      if (!radius) {
        Tcl_AppendResult(interp, "volmap ils: probe selection contains no VDW radii", NULL);
        bailout = true;
      }
      
      const float *occupancy = probemol->extraflt.data("occupancy");
      if (!occupancy) {
        Tcl_AppendResult(interp, "volmap ils: probe selection contains no VDW epsilon values (in occupancy field)", NULL);
        bailout = true;
      }
      
      if (!bailout) {
        const float *charge = probemol->extraflt.data("charge");
        
        num_probe_atoms = probesel->selected;
        if (!nprobevdw) {
          probe_vdwrmin = new float[num_probe_atoms];
          probe_vdweps  = new float[num_probe_atoms];
        }
        probe_charge  = new float[num_probe_atoms];
        probe_coords  = new float[3*num_probe_atoms];
        const float *coords = probesel->coordinates(app->moleculeList);
        int i, j=0;
        for (i=0; i<probesel->num_atoms; i++) {
          if (probesel->on[i]) {
            vec_copy(&probe_coords[3*j], &coords[3*i]);

            if (!nprobevdw) {
              probe_vdweps[j]  = -fabsf(occupancy[i]);
              probe_vdwrmin[j] = radius[i];
            }
            if (!nprobecharge) {
              if (charge) probe_charge[j] = charge[i];
              else        probe_charge[j] = 0.f;
            }
            j++;
          }
        }  
      }
      
    } else {
      // The probe was specified through a coordinate list
      if (nprobecoor==0 && nprobevdw==1) {
        // No need to specify coodinates of monoatomic probe
        num_probe_atoms = 1;
        probe_coords = new float[3];
        probe_coords[0] = probe_coords[1] = probe_coords[2] = 0.f;
      } else {
        num_probe_atoms = nprobecoor;
      }

      if (!nprobevdw) {
        Tcl_AppendResult(interp, "volmap ils: No probe VDW parameters specified.", NULL);
        bailout = true;      
      }

      if (nprobecharge && nprobecharge!=num_probe_atoms && !bailout) {
        Tcl_AppendResult(interp, "volmap ils: # probe charges doesn't match # probe atoms", NULL);
        bailout = true;
      }
    }

    if (num_probe_atoms==0 && !bailout) {
      Tcl_AppendResult(interp, "volmap: No probe coordinates specified.", NULL);
      bailout = true;
    }

    if (nprobevdw && nprobevdw!=num_probe_atoms && !bailout) {
      Tcl_AppendResult(interp, "volmap ils: # probe VDW params doesn't match # probe atoms", NULL);
      bailout = true;
    }

    if (pbc && !alignsel) {
      Tcl_AppendResult(interp, "volmap ils: You cannot use -pbc without also "
                       " providing -alignsel.", NULL);
      bailout = true;
    }

    //if (num_probe_atoms==1 && num_conf>1 && !bailout) {
      //Tcl_AppendResult(interp, "volmap: Specifying -orient for monoatomic probes makes no sense.", NULL);
      //bailout = true;
    //}
  }

  if (bailout) {
    if (transform) delete transform;
    if (pbccenter) delete [] pbccenter;
    if (filebase)  delete [] filebase;
    if (probe_vdwrmin) delete [] probe_vdwrmin;
    if (probe_vdweps)  delete [] probe_vdweps;
    if (probe_charge)  delete [] probe_charge;
    if (probe_coords)  delete [] probe_coords;
    return TCL_ERROR;
  }

  // If the probe was provided in form of an atom selection
  // determine the symmetry of the probe. We need to know if
  // the probe has a tetrahedral point group, the highest
  // rotary axis and any C2 axis orthogonal to it.
  // If the probe was specified through parameter list instead
  // of a selection then the ILS code will try a simple
  // symmetry check itself
  int tetrahedral_symm = 0;
  int order1=0, order2=0;
  float symmaxis1[3];
  float symmaxis2[3];
  if (probesel) {
    msgInfo << "Determining probe symmetry:" << sendmsg;

    // Create Symmetry object, verbosity level = 1
    Symmetry sym = Symmetry(probesel, app->moleculeList, 1);
    
    // Set tolerance for atomic overlapping
    sym.set_overlaptol(0.05f);
    
    // Take bonds into account
    sym.set_checkbonds(1);

    // Guess the symmetry
    int ret_val = sym.guess(0.05f);
    if (ret_val < 0) {
      Tcl_AppendResult(interp, "measure symmetry: ", measure_error(ret_val), NULL);
      return TCL_ERROR;
    }
    int pgorder;
    char pointgroup[6];
    sym.get_pointgroup(pointgroup, &pgorder);

    if (pointgroup[0]=='T') tetrahedral_symm = 1;

    if (sym.numaxes()) {
      // First symmetry axis
      vec_copy(symmaxis1, sym.axis(0));
      order1 = sym.get_axisorder(0);

      int i;
      for (i=1; i<sym.numaxes(); i++) {
        vec_copy(symmaxis2, sym.axis(i));
        if (fabs(dot_prod(symmaxis1, symmaxis2)) < DEGTORAD(1.f)) {
          // Orthogonal second axis found
          order2 = sym.get_axisorder(i);
          break;
        }
      }
    }
  }

  // 6. Create the volmap
  VolMapCreateILS vol(app, molid, first, last, (float)temperature,
                      (float)resolution, subres, (float)cutoff,
                      maskonly);

  vol.set_probe(num_probe_atoms, num_conf, probe_coords,
                probe_vdwrmin, probe_vdweps, probe_charge);
  vol.set_maxenergy(float(maxenergy));

  if (pbc)       vol.set_pbc(pbccenter, pbcbox);
  if (transform) vol.set_transform(transform);
  if (alignsel)  vol.set_alignsel(alignsel);
  if (!pbcbox) {
    vol.set_minmax(float(minmax[0]), float(minmax[1]), float(minmax[2]),
                   float(minmax[3]), float(minmax[4]), float(minmax[5]));
  }

  if (tetrahedral_symm || order1 || order2) {
    // Provide info about probe symmetry
    vol.set_probe_symmetry(order1, symmaxis1, order2, symmaxis2, tetrahedral_symm);
  }

  // Create map...
  int ret_val = vol.compute();

  if (ret_val < 0) {
    Tcl_AppendResult(interp, "\nvolmap: ERROR ", measure_error(ret_val), NULL);

    if (transform) delete transform;
    if (pbccenter) delete [] pbccenter;
    if (filebase)  delete [] filebase;
    if (probe_vdwrmin) delete [] probe_vdwrmin;
    if (probe_vdweps)  delete [] probe_vdweps;
    if (probe_charge)  delete [] probe_charge;
    if (probe_coords)  delete [] probe_coords;
    return TCL_ERROR;
  }

  int numconf, numorient, numrot;
  vol.get_statistics(numconf, numorient, numrot);

  // If the probe was specified in form of a selection and a separate
  // molecule then we append a frame for each conformer and set the
  // probe coordinates accordingly.
  if (probesel && probesel->molid()!=molid) {
    float *conformers = NULL;
    int numconf = vol.get_conformers(conformers);
    Molecule *pmol = app->moleculeList->mol_from_id(probesel->molid());
    int i, j;
    for (i=0; i<numconf; i++) {
      if (i>0) {
        pmol->duplicate_frame(pmol->current());
      }
      float *coor = pmol->get_frame(i)->pos;
      int k=0;
      for (j=0; j<probesel->num_atoms; j++) { 
        if (!probesel->on[j]) continue; //atom is not selected

	vec_copy(&coor[3*j], &conformers[i*3*num_probe_atoms+3*k]);
	k++;
      }
    }
  }

  // Export volmap to a file or just add it to the molecule:
  if (export_to_file) {
    // Add .dx suffix to filebase if it is missing
    filename = new char[strlen(filebase)+16];
    strcpy(filename, filebase);
    char *suffix = strrchr(filename, '.'); // beginning of .dx
    if (strcmp(suffix, ".dx")) strcat(filename, ".dx");

    // Write tha map into a dx file
    if (!vol.write_map(filename)) {
      Tcl_AppendResult(interp, "\nvolmap: ERROR Could not write ils map to file", NULL);
    }

    delete[] filename;

  } else {
    if (!vol.add_map_to_molecule()) {
      Tcl_AppendResult(interp, "\nvolmap: ERROR Could not add ils map to molecule", NULL);
    }
  }
    
  Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewStringObj("numconf", -1));
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewDoubleObj(numconf));
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewStringObj("numorient", -1));
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewDoubleObj(numorient));
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewStringObj("numrot", -1));
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewDoubleObj(numrot));
  Tcl_SetObjResult(interp, tcl_result);

  if (transform) delete transform;
  if (pbccenter) delete [] pbccenter;
  if (filebase)  delete [] filebase;
  if (probe_vdwrmin) delete [] probe_vdwrmin;
  if (probe_vdweps)  delete [] probe_vdweps;
  if (probe_charge)  delete [] probe_charge;
  if (probe_coords)  delete [] probe_coords;

  return TCL_OK;
}


static int vmd_volmap_new_fromtype(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {

  // 1. Figure out which map type we are dealing with:
  enum {UNDEF_MAP, DENS_MAP, INTERP_MAP, DIST_MAP, OCCUP_MAP, MASK_MAP,
        CPOTENTIAL_MAP, CPOTENTIALMSM_MAP } maptype=UNDEF_MAP;

  char *maptype_string=Tcl_GetString(objv[0]); 

  if      (!strcmp(maptype_string, "density"))    maptype=DENS_MAP;
  else if (!strcmp(maptype_string, "interp"))     maptype=INTERP_MAP;
  else if (!strcmp(maptype_string, "distance"))   maptype=DIST_MAP;
  else if (!strcmp(maptype_string, "occupancy"))  maptype=OCCUP_MAP; 
  else if (!strcmp(maptype_string, "mask"))       maptype=MASK_MAP; 
  else if (!strcmp(maptype_string, "coulomb"))    maptype=CPOTENTIAL_MAP;
  else if (!strcmp(maptype_string, "coulombpotential")) maptype=CPOTENTIAL_MAP;
  else if (!strcmp(maptype_string, "coulombmsm"))  maptype=CPOTENTIALMSM_MAP;
 
  // 2. Get atom selection
  if (argc < 2) { 
    Tcl_AppendResult(interp, "volmap: no atom selection.", NULL);
    return TCL_ERROR;
  }

  AtomSel *sel = NULL;
  sel = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[1],NULL));
  if (!sel) {
    Tcl_AppendResult(interp, "volmap: no atom selection.", NULL);
    return TCL_ERROR;
  }
  if (!sel->selected) {
    Tcl_AppendResult(interp, "volmap: no atoms selected.", NULL);
    return TCL_ERROR;
  }
  if (!app->molecule_valid_id(sel->molid())) {
    Tcl_AppendResult(interp, "volmap: ",
                     measure_error(MEASURE_ERR_NOMOLECULE), NULL);
    return TCL_ERROR;
  }


  // 3. Define and initialize the optional arguments
  bool accept_weight = false; // allow a user-specified weight for each atom
  bool accept_cutoff = false; // parse a user-specified cutoff distance
  bool accept_radius = false; // allow radius multiplicator for density
  bool accept_usepoints = false; // allows use of point particles

  bool use_point_particles = false;  // for MASK map
  bool export_to_file = false;
  bool use_all_frames = false;
  bool bad_usage = false;

  int export_molecule = -1;
  double cutoff        = 5.;
  double radius_factor = 1.;
  double resolution    = 1.;
  double minmax[6];
  
  char *filebase = NULL;
  char *filename = NULL;

  
  // File export options
  int checkpoint_freq = 500;
      

  // Specify required/accepted options for each maptype as well as default values.
  switch(maptype) {
    case DENS_MAP:
      accept_weight = true;
      accept_radius = true;
      break;
    case INTERP_MAP:
      accept_weight = true; 
      break;
    case DIST_MAP:
      accept_cutoff = true;
      cutoff = 3.;
      break;
    case OCCUP_MAP:
      accept_usepoints = true;
      break;
    case MASK_MAP:
      accept_cutoff = true;
      cutoff = 4.;
      break;
    case CPOTENTIAL_MAP:
    case CPOTENTIALMSM_MAP:
      break; 
    case UNDEF_MAP:
      bad_usage = true;
      break;   
  }
 

  // 4. Parse the command-line
  int arg_weight=0, arg_combine=0, arg_minmax=0;

  // Parse the arguments
  for (int arg=2; arg<argc && !bad_usage; arg++) {
    if (!strcmp(Tcl_GetStringFromObj(objv[arg], NULL), "-res")) {
      if (arg+1>=argc) bad_usage=true;
      Tcl_GetDoubleFromObj(interp, objv[arg+1], &resolution);
      if (resolution <= 0.f) {
        Tcl_AppendResult(interp, "volmap: resolution must be positive. (-res)", NULL);
        return TCL_ERROR;
      }
      arg++;
    }
    else if (!strcmp(Tcl_GetStringFromObj(objv[arg], NULL), "-mol")) { // add volmap to mol
      if (arg+1 >= argc) bad_usage=true;
      if (!strcmp(Tcl_GetStringFromObj(objv[arg+1], NULL), "top"))
        export_molecule = app->molecule_top(); 
      else 
        Tcl_GetIntFromObj(interp, objv[arg+1], &export_molecule);
      
      if (!app->molecule_valid_id(export_molecule)) {
        Tcl_AppendResult(interp, "volmap: molecule specified for ouput is invalid. (-mol)", NULL);
        return TCL_ERROR;
      }
      arg++;
    }
    else if (!strcmp(Tcl_GetStringFromObj(objv[arg], NULL), "-minmax")) {
      arg_minmax=arg+1;
      arg++;
      if (arg_minmax>=argc) bad_usage=true;
      if (parse_minmax_args(interp, arg_minmax, objv, minmax)!=TCL_OK) {
        return TCL_ERROR;
      }
    }
    else if (!strcmp(Tcl_GetStringFromObj(objv[arg], NULL), "-checkpoint")) {
      if (arg+1 >= argc) bad_usage=true;
      Tcl_GetIntFromObj(interp, objv[arg+1], &checkpoint_freq);
      if (checkpoint_freq < 0) {
        Tcl_AppendResult(interp, "volmap: invalid -checkpoint parameter", NULL);
        return TCL_ERROR;
      }
      arg++;
    }
    else if (!strcmp(Tcl_GetStringFromObj(objv[arg], NULL), "-allframes")) {
      use_all_frames=true;
    }
    else if (!strcmp(Tcl_GetStringFromObj(objv[arg], NULL), "-dx") ||
             !strcmp(Tcl_GetStringFromObj(objv[arg], NULL), "-o")) {
      if (arg+1>=argc) {bad_usage=true; break;}
      const char* outputfilename = Tcl_GetString(objv[arg+1]);
      if (outputfilename) {
        filebase = new char[strlen(outputfilename)+1];
        strcpy(filebase, outputfilename);
        export_to_file = true;
      }
      arg++;
    }
    else if (!strcmp(Tcl_GetStringFromObj(objv[arg], NULL), "-combine")) {
      arg_combine=arg+1;
      arg++;
      if (arg_combine>=argc) bad_usage=true;
    }
    else if (accept_usepoints && !strcmp(Tcl_GetStringFromObj(objv[arg], NULL), "-points")) {
      use_point_particles=true;
    }
    else if (accept_cutoff && !strcmp(Tcl_GetStringFromObj(objv[arg], NULL), "-cutoff")) {
      if (arg+1 >= argc) bad_usage=true;
      Tcl_GetDoubleFromObj(interp, objv[arg+1], &cutoff);
      if (cutoff <= 0.) {
        Tcl_AppendResult(interp, "volmap: cutoff must be positive. (-cutoff)", NULL);
        return TCL_ERROR;
      }
      arg++;
    }
    else if (accept_radius && !strcmp(Tcl_GetStringFromObj(objv[arg], NULL), "-radscale")) {
      if (arg+1 >= argc) bad_usage=true;
      Tcl_GetDoubleFromObj(interp, objv[arg+1], &radius_factor);
      if (radius_factor < 0.f) {
        Tcl_AppendResult(interp, "volmap: radscale must be positive. (-radscale)", NULL);
        return TCL_ERROR;
      }
      arg++;
    }
    else if (accept_weight && !strcmp(Tcl_GetStringFromObj(objv[arg], NULL), "-weight")) {
      if (arg+1>=argc) bad_usage=true;
      arg_weight = arg+1;      
      arg++;
    }
    
    else //unknown arg
      bad_usage=true; 
  }
  
    
  if (bad_usage) {
    if (maptype == UNDEF_MAP)
      Tcl_AppendResult(interp, "volmap: unknown map type.", NULL);
    else
      Tcl_WrongNumArgs(interp, 2, objv-1, (char *)"<selection> [options...]");

    return TCL_ERROR;
  }
  
  
  // 5. Assign some other optional parameters
     
  // parse map combination type
  VolMapCreate::CombineType combine_type=VolMapCreate::COMBINE_AVG;
  if (arg_combine) {
    char *combine_str=Tcl_GetString(objv[arg_combine]);
    if (!strcmp(combine_str, "avg") || !strcmp(combine_str, "average")) 
      combine_type=VolMapCreate::COMBINE_AVG;
    else if (!strcmp(combine_str, "max") || !strcmp(combine_str, "maximum")) 
      combine_type=VolMapCreate::COMBINE_MAX;
    else if (!strcmp(combine_str, "min") || !strcmp(combine_str, "minimum")) 
      combine_type=VolMapCreate::COMBINE_MIN;
    else if (!strcmp(combine_str, "stdev")) 
      combine_type=VolMapCreate::COMBINE_STDEV;
    else if (!strcmp(combine_str, "pmf")) 
      combine_type=VolMapCreate::COMBINE_PMF;
    else {
      Tcl_AppendResult(interp, "volmap: -combine argument must be: avg, min, \
                                max, stdev, pmf", NULL);
      return TCL_ERROR;
    }
  }
 
  // parse weights
  int ret_val=0;
  float *weights = NULL;
  if (accept_weight) {
    weights = new float[sel->selected];

    if (arg_weight) 
      ret_val = tcl_get_weights(interp, app, sel, objv[arg_weight], weights);
    else
      ret_val = tcl_get_weights(interp, app, sel, NULL, weights);
    
    if (ret_val < 0) {
      Tcl_AppendResult(interp, "volmap: ", measure_error(ret_val), NULL);
      delete [] weights;
      return TCL_ERROR;
    }
  }
  


  // 6. Create the volmap
  VolMapCreate *volcreate = NULL;   

  // init the map creator objects and set default filenames
  switch(maptype) {
    case DENS_MAP: 
      volcreate = new VolMapCreateDensity(app, sel, (float)resolution, weights, (float)radius_factor);
      if (!filebase) {
        filebase = new char[strlen("density_out.dx")+1];
        strcpy(filebase, "density_out.dx");
      }
      break;

    case INTERP_MAP: 
      volcreate = new VolMapCreateInterp(app, sel, (float)resolution, weights);
      if (!filebase) {
        filebase = new char[strlen("interp_out.dx")+1];
        strcpy(filebase, "interp_out.dx");
      }
      break;

    case DIST_MAP:
      volcreate = new VolMapCreateDistance(app, sel, (float)resolution, (float)cutoff);
      if (!filebase) {
        filebase = new char[strlen("distance_out.dx")+1];
        strcpy(filebase, "distance_out.dx");
      }
      break;

    case OCCUP_MAP:
      volcreate = new VolMapCreateOccupancy(app, sel, (float)resolution, use_point_particles);
      if (!filebase) {
        filebase = new char[strlen("occupancy_out.dx")+1];
        strcpy(filebase, "occupancy_out.dx");
      }
      break;

    case MASK_MAP:
      volcreate = new VolMapCreateMask(app, sel, (float)resolution, (float)cutoff);
      if (!filebase) {
        filebase = new char[strlen("mask_out.dx")+1];
        strcpy(filebase, "mask_out.dx");
      }
      break;

    case CPOTENTIAL_MAP:
      volcreate = new VolMapCreateCoulombPotential(app, sel, (float)resolution);
      if (!filebase) {
        filebase = new char[strlen("coulomb_out.dx")+1];
        strcpy(filebase, "coulomb_out.dx");
      }
      break;

    case CPOTENTIALMSM_MAP:
      volcreate = new VolMapCreateCoulombPotentialMSM(app, sel, (float)resolution);
      if (!filebase) {
        filebase = new char[strlen("coulombmsm_out.dx")+1];
        strcpy(filebase, "coulombmsm_out.dx");
      }
      break;


    default:  // silence compiler warnings
      break;  
  }

  // generate and write out volmap
  if (volcreate) {
    // Pass parameters common to all volmap types
    if (arg_minmax)
      volcreate->set_minmax(float(minmax[0]), float(minmax[1]),
                            float(minmax[2]), float(minmax[3]), 
                            float(minmax[4]), float(minmax[5]));
    
    // Setup checkpointing
    if (checkpoint_freq) {
      char *checkpointname = new char[32+strlen(filebase)+1];
#if defined(_MSC_VER)
      char slash = '\\';
#else
      char slash = '/';
#endif
      char *tailname = strrchr(filebase, slash);
      if (!tailname) tailname = filebase;
      else tailname = tailname+1;
      char *dirname = new char[strlen(filebase)+1];
      strcpy(dirname, filebase);
      char *sep = strrchr(dirname, slash);

      if (sep) {
        *sep = '\0';
        sprintf(checkpointname, "%s%ccheckpoint:%s", dirname, slash, tailname);
      }
      else {
        *dirname = '\0';
        sprintf(checkpointname, "checkpoint:%s", tailname);
      }
      delete[] dirname;

      Tcl_AppendResult(interp, "CHECKPOINTNAME = ", checkpointname, NULL);
      volcreate->set_checkpoint(checkpoint_freq, checkpointname);
      delete[] checkpointname;
    }
    
    // Create map...
    ret_val = volcreate->compute_all(use_all_frames, combine_type, NULL);
    if (ret_val < 0) {
      delete volcreate;
      if (weights)  delete [] weights;
      if (filebase) delete [] filebase;

      Tcl_AppendResult(interp, "\nvolmap: ERROR ", measure_error(ret_val), NULL);
      return TCL_ERROR;
    }
    
    // Export volmap to a file:
    if (export_to_file || export_molecule < 0) {
      // add .dx suffix to filebase if it is missing
      filename = new char[strlen(filebase)+16];
      strcpy(filename,filebase);
      char *suffix = strrchr(filename, '.');
      if (!strcmp(suffix,".dx")) *suffix = '\0';
      strcat(filename, ".dx");
      volcreate->write_map(filename);
      delete[] filename;
    }
    
    // Export volmap to a molecule:
    if (export_molecule >= 0) {
      VolumetricData *volmap = volcreate->volmap;
      float origin[3], xaxis[3], yaxis[3], zaxis[3];
      int i;
      for (i=0; i<3; i++) {
        origin[i] = (float) volmap->origin[i];
        xaxis[i] = (float) volmap->xaxis[i];
        yaxis[i] = (float) volmap->yaxis[i];
        zaxis[i] = (float) volmap->zaxis[i];
      }
      int err = app->molecule_add_volumetric(export_molecule, 
         (volmap->name) ? volmap->name : "(no name)",
         origin, xaxis, yaxis, zaxis,
         volmap->xsize, volmap->ysize, volmap->zsize, volmap->data);
      if (err != 1) {
        Tcl_AppendResult(interp, "ERROR: export of volmap into molecule was unsuccessful!", NULL);
      }
      else volmap->data=NULL; // avoid data being deleted by volmap's destructor (it is now owned by the molecule)
    }

    delete volcreate;
  }
  
  if (weights) delete [] weights;
  if (filebase) delete [] filebase;

  return TCL_OK;
}


// vec_sub() from utilities.h works with float* only
// here I needed doubles.
#define DOUBLE_VSUB(D, X, Y) \
  D[0] = float(X[0]-Y[0]); \
  D[1] = float(X[1]-Y[1]); \
  D[2] = float(X[2]-Y[2]); 

// Compare two volumetric maps:
// volmap compare <molid1> <volid1> <molid2> <volid2>
// The two maps must be specified by their molID and volID.
// Prints the min/max vales, the largest difference, the RMSD,
// the RMSD computed inly for the elements that differ and the
// RMSD weighted by the magnitude of the elements compared so
// that smaller values receive a larger weight.
// (For ILS we are interested mainly in the smaller energies)
static int vmd_volmap_compare(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp)
{
  int molid1 = -1;
  int molid2 = -1;
  int mapid1 = -1;
  int mapid2 = -1;

  // Get the molecule IDs
  if (!strcmp(Tcl_GetStringFromObj(objv[1], NULL), "top"))
    molid1 = app->molecule_top(); 
  else 
    Tcl_GetIntFromObj(interp, objv[1], &molid1);
  
  if (!strcmp(Tcl_GetStringFromObj(objv[3], NULL), "top"))
    molid2 = app->molecule_top(); 
  else 
    Tcl_GetIntFromObj(interp, objv[3], &molid2);

  if (!app->molecule_valid_id(molid1) || !app->molecule_valid_id(molid2)) {
    Tcl_AppendResult(interp, "volmap compare: molecule specified for ouput is invalid. (-mol)", NULL);
    return TCL_ERROR;
  }

  Molecule *mol1 = app->moleculeList->mol_from_id(molid1);
  Molecule *mol2 = app->moleculeList->mol_from_id(molid2);

  // Get volmap IDs
  Tcl_GetIntFromObj(interp, objv[2], &mapid1);
  Tcl_GetIntFromObj(interp, objv[4], &mapid2);

  if (mapid1<0 || mapid2<0) {
    Tcl_AppendResult(interp, "volmap compare: Volmap ID must be positive.", NULL);
    return TCL_ERROR;
  }
  if (mapid1 >= mol1->num_volume_data() ||
      mapid2 >= mol2->num_volume_data()) {
    Tcl_AppendResult(interp, "volmap compare: Volmap ID does not exist.", NULL);
    return TCL_ERROR;
  }

  // Parse optional arguments
  bool bad_usage = false;
  double histinterval = 2.5;
  int numbins = 9;
  int arg;
  for (arg=5; arg<argc && !bad_usage; arg++) {
    if (!strcmp(Tcl_GetStringFromObj(objv[arg], NULL), "-interval")) {
      if (arg+1>=argc) { bad_usage=true; break; }
      Tcl_GetDoubleFromObj(interp, objv[arg+1], &histinterval);
      if (histinterval <= 0.f) {
        Tcl_AppendResult(interp, "volmap compare: histogram interval must be positive. (-interval)", NULL);
        return TCL_ERROR;
      }
      arg++;
    }
    else if (!strcmp(Tcl_GetStringFromObj(objv[arg], NULL), "-numbins")) {
      if (arg+1>=argc) { bad_usage=true; break; }
      Tcl_GetIntFromObj(interp, objv[arg+1], &numbins);
      if (numbins <= 0) {
        Tcl_AppendResult(interp, "volmap compare: histogram bin size must be positive. (-interval)", NULL);
        return TCL_ERROR;
      }
      arg++;
    }
    else {
      // unknown arg
      Tcl_AppendResult(interp, " volmap compare: unknown argument ",
                       Tcl_GetStringFromObj(objv[arg], NULL), NULL);
      return TCL_ERROR;
    }

  }
  if (bad_usage) {
    Tcl_WrongNumArgs(interp, 2, objv-1, (char *)"<molid> <minmax> [options...]");
    return TCL_ERROR;
  }
  
  const VolumetricData *vol1 = mol1->get_volume_data(mapid1);
  const VolumetricData *vol2 = mol2->get_volume_data(mapid2);

  if (vol1->xsize != vol2->xsize ||
      vol1->ysize != vol2->ysize ||
      vol1->zsize != vol2->zsize) {
    Tcl_AppendResult(interp, "volmap compare: maps have different dimensions.", NULL);
    return TCL_ERROR;
  }

  float vdiff[3];
  DOUBLE_VSUB(vdiff, vol1->origin, vol2->origin);
  if (norm(vdiff)>1e-6) {
    Tcl_AppendResult(interp, "volmap compare: maps have different origin.", NULL);
  }

  DOUBLE_VSUB(vdiff, vol1->xaxis, vol2->xaxis);
  if (norm(vdiff)>1e-6) {
    Tcl_AppendResult(interp, "volmap compare: maps have different x-axis.", NULL);
  }
  DOUBLE_VSUB(vdiff, vol1->yaxis, vol2->yaxis);
  if (norm(vdiff)>1e-6) {
    Tcl_AppendResult(interp, "volmap compare: maps have different y-axis.", NULL);
  }
  DOUBLE_VSUB(vdiff, vol1->zaxis, vol2->zaxis);
  if (norm(vdiff)>1e-6) {
    Tcl_AppendResult(interp, "volmap compare: maps have different z-axis.", NULL);
  }

  int i;
  int numdiff = 0;
  float sqsum = 0.f;
  float sqsumd = 0.f;
  float min1 = 0.f, min2 = 0.f;
  float max1 = 0.f, max2 = 0.f;
  float maxdiff = 0.f;
  int indexmaxdiff = 0;

  for (i=0; i<vol1->gridsize(); i++) {
    float v1 = vol1->data[i];
    float v2 = vol2->data[i];
    float diff = v1-v2;
    sqsum += diff*diff;
    if (v1<min1) min1 = v1;
    if (v2<min2) min2 = v2;
    if (v1>max1) max1 = v1;
    if (v2>max2) max2 = v2;
    if (fabsf(diff)>maxdiff) {
      maxdiff = fabsf(diff);
      indexmaxdiff = i;
    }
    if (diff) {
      //printf("%g - %g = %g\n", v1, v2, diff);
      sqsumd += diff*diff;
      numdiff++;
    }
  }
  float rmsd = 0.f;
  float diffrmsd = 0.f;
  if (sqsum)  rmsd     = sqrtf(sqsum/vol1->gridsize());
  if (sqsumd) diffrmsd = sqrtf(sqsumd/numdiff);

  char tmpstr[128];
  msgInfo << "volmap compare" << sendmsg;
  msgInfo << "--------------" << sendmsg;
  msgInfo << "Comparing mol "<<molid1<<" -> map "<<mapid1<<"/"<<mol1->num_volume_data()<<sendmsg;
  msgInfo << "       to mol "<<molid2<<" -> map "<<mapid2<<"/"<<mol2->num_volume_data()<<sendmsg;
  msgInfo << "Statistics:" << sendmsg;
  sprintf(tmpstr, "            %12s  |  %12s", "MAP 1", "MAP 2");
  msgInfo << tmpstr << sendmsg;
  sprintf(tmpstr, "min value = %12g  |  %12g", min1, min2);
  msgInfo << tmpstr << sendmsg;
  sprintf(tmpstr, "max value = %12g  |  %12g", max1, max2);
  msgInfo << tmpstr << sendmsg;
  msgInfo << sendmsg;
  sprintf(tmpstr, "# differing elements = %d/%d", numdiff, vol1->gridsize());
  msgInfo << tmpstr << sendmsg;
  msgInfo << "max difference:" << sendmsg;
  sprintf(tmpstr, "   map1[%d] = %g   map2[%d] = %g   diff = %g",
      indexmaxdiff, vol1->data[indexmaxdiff],
      indexmaxdiff, vol2->data[indexmaxdiff], maxdiff);
  msgInfo << tmpstr << sendmsg;

  // Statistics for the differing elements only:
  msgInfo << sendmsg;
  sprintf(tmpstr, "         RMSD = %12.6f", rmsd);
  msgInfo << tmpstr << sendmsg;
  sprintf(tmpstr, "     diffRMSD = %12.6f  (for the set of differing elements)", diffrmsd);
  msgInfo << tmpstr << sendmsg;

  // Get weighted RMSD where the differences of smaller values
  // are weighted more because the low enegy values are what is 
  // of interest in ILS free energy maps.
  float wsum = 0.f;
  float range = max2-min2;
  float wdiffrmsd = 0.f;
  if (range) {
    sqsum = 0.f;
    for (i=0; i<vol1->gridsize(); i++) {
      float diff = vol1->data[i]-vol2->data[i];
      if (diff) {
        float weight = 1.f-(vol2->data[i]-min2)/range;
        wsum += weight;
        sqsum += diff*diff*weight;
      }
    }
    if (sqsum) wdiffrmsd = sqrtf(sqsum/wsum);
  }
 

  sprintf(tmpstr, "weighted RMSD = %12.6f  (for the set of differing elements)", wdiffrmsd);
  msgInfo << tmpstr << sendmsg;
  msgInfo <<      "     weight factor w = 1-(E_i-E_min)/(E_max-E_min)" << sendmsg;
  msgInfo << sendmsg;

  // Compare error of map 1 relative to map 2, create histogram of error:
  //   | E_approx - E_exact | / ( E_exact - E_exact_min + 1 ),
  // where map 1 is considered the approximation and map 2 is considered exact.
  //
  // Since energy is arbitrary, we shift from the minimum value 
  // (usually around -11 to -6) so that the dominator is always
  // greater than or equal to 1.  This weights the lower values
  // more heavily than the upper values, which is intentional.
  //
  // The histogram is summed for the intervals
  // (-\inf,10) [10,20) [20,30) [30,40) [40,+\inf)


  // total accumulated error for each bin
  float *histo = new float[numbins*sizeof(float)];
  // count number of samples per bin
  int *num = new int[numbins*sizeof(float)];
  // max error for each bin
  float *maxEntry = new float[numbins*sizeof(float)];
  float *binrmsd = new float[numbins*sizeof(float)];
  memset(histo,    0, numbins*sizeof(float));
  memset(num,      0, numbins*sizeof(int));
  memset(maxEntry, 0, numbins*sizeof(float));
  memset(binrmsd,  0, numbins*sizeof(float));

  for (i = 0;  i < vol1->gridsize();  i++) {
    float e1 = vol1->data[i];
    float e2 = vol2->data[i];
    float err = fabsf(e1 - e2) / (e2 - min2 + 1);
    int index = (int) floorf((e2 - min2) / float(histinterval));
    if      (index < 0)        index = 0;
    else if (index >= numbins) index = numbins - 1;

    // check to see if we need to update the max for this bin
    if (err > maxEntry[index]) { 
       maxEntry[index] = err; 
    }
    histo[index] += err;
    num[index]++;
    binrmsd[index] += (e2-e1)*(e2-e1);
  }
  for (i=0; i<numbins; i++) {
    if (binrmsd[i]) binrmsd[i] = sqrtf(binrmsd[i]/num[i]);
  }

  // lower boundary of the first bin
  float firstbin = floorf(min2/float(histinterval))*float(histinterval);
  Tcl_Obj *caption = Tcl_NewListObj(0, NULL);
  Tcl_Obj *numEntries = Tcl_NewListObj(0, NULL);
  Tcl_Obj *objHisto   = Tcl_NewListObj(0, NULL);
  Tcl_Obj *objAverage = Tcl_NewListObj(0, NULL);
  Tcl_Obj *objMax     = Tcl_NewListObj(0, NULL);
  Tcl_Obj *objBinRMSD = Tcl_NewListObj(0, NULL);
  char label[64];
  msgInfo << "Histogram of error in map 1 relative to map 2 " << sendmsg;
  msgInfo << sendmsg;
  msgInfo << "     interval   # samples    total error      avg error"
          << "      max error           rmsd" << sendmsg;
  msgInfo << "---------------------------------------------------------"
          << "----------------------------" << sendmsg;

  sprintf(label, "[%g,%g)", (double) firstbin, (double) (firstbin+histinterval));
  sprintf(tmpstr, "%14s %10d %14.6f %14.6f %14.6f %14.6f", label, num[0],
          (double) histo[0], (double) (!num[0]?0:histo[0]/num[0]),
          (double) maxEntry[0], (double) binrmsd[0]);
  msgInfo << tmpstr << sendmsg;
  Tcl_ListObjAppendElement(interp, caption,    Tcl_NewStringObj(label, -1));
  Tcl_ListObjAppendElement(interp, numEntries, Tcl_NewIntObj(num[0]));
  Tcl_ListObjAppendElement(interp, objHisto,   Tcl_NewDoubleObj(histo[0]));
  Tcl_ListObjAppendElement(interp, objAverage, Tcl_NewDoubleObj(!num[0]?0:histo[0]/num[0]));
  Tcl_ListObjAppendElement(interp, objMax,     Tcl_NewDoubleObj(maxEntry[0]));
  Tcl_ListObjAppendElement(interp, objBinRMSD, Tcl_NewDoubleObj(binrmsd[0]));
  for (i = 1;  i < numbins-1;  i++) {
    sprintf(label, "[%g,%g)", (double)(firstbin+i*histinterval), (double)(firstbin+(i+1)*histinterval));
    sprintf(tmpstr, "%14s %10d %14.6f %14.6f %14.6f %14.6f", label, num[i],
            (double) histo[i], (double) (!num[i]?0:histo[i]/num[i]),
            (double) maxEntry[i], (double) binrmsd[i]);
    Tcl_ListObjAppendElement(interp, caption,    Tcl_NewStringObj(label, -1));
    Tcl_ListObjAppendElement(interp, numEntries, Tcl_NewIntObj(num[i]));
    Tcl_ListObjAppendElement(interp, objHisto,   Tcl_NewDoubleObj(histo[i]));
    Tcl_ListObjAppendElement(interp, objAverage, Tcl_NewDoubleObj(!num[i]?0:histo[i]/num[i]));
    Tcl_ListObjAppendElement(interp, objMax,     Tcl_NewDoubleObj(maxEntry[i]));
    Tcl_ListObjAppendElement(interp, objBinRMSD, Tcl_NewDoubleObj(binrmsd[i]));
    msgInfo << tmpstr << sendmsg;
  }
  sprintf(label, "[%g,+infty)", (double)(firstbin+i*histinterval));
  sprintf(tmpstr, "%14s %10d %14.6f %14.6f %14.6f %14.6f", label, num[i],
          (double) histo[i], (double) (!num[i]?0:histo[i]/num[i]),
          (double) maxEntry[i], (double) binrmsd[i]);
  Tcl_ListObjAppendElement(interp, caption,    Tcl_NewStringObj(label, -1));
  Tcl_ListObjAppendElement(interp, numEntries, Tcl_NewIntObj(num[i]));
  Tcl_ListObjAppendElement(interp, objHisto,   Tcl_NewDoubleObj(histo[i]));
  Tcl_ListObjAppendElement(interp, objAverage, Tcl_NewDoubleObj(!num[i]?0:histo[i]/num[i]));
  Tcl_ListObjAppendElement(interp, objMax,     Tcl_NewDoubleObj(maxEntry[i]));
  Tcl_ListObjAppendElement(interp, objBinRMSD, Tcl_NewDoubleObj(binrmsd[i]));
  msgInfo << tmpstr << sendmsg;
  msgInfo << sendmsg;

  Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewStringObj("rmsd", -1));
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewDoubleObj(rmsd));
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewStringObj("diffrmsd", -1));
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewDoubleObj(diffrmsd));
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewStringObj("weightedrmsd", -1));
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewDoubleObj(wdiffrmsd));
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewStringObj("maxdiff", -1));
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewDoubleObj(maxdiff));
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewStringObj("numdiff", -1));
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewIntObj(numdiff));
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewStringObj("caption", -1));
  Tcl_ListObjAppendElement(interp, tcl_result, caption);
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewStringObj("numEntries", -1));
  Tcl_ListObjAppendElement(interp, tcl_result, numEntries);
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewStringObj("histo", -1));
  Tcl_ListObjAppendElement(interp, tcl_result, objHisto);
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewStringObj("avgErr", -1));
  Tcl_ListObjAppendElement(interp, tcl_result, objAverage);
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewStringObj("maxError", -1));
  Tcl_ListObjAppendElement(interp, tcl_result, objMax);
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewStringObj("binRMSD", -1));
  Tcl_ListObjAppendElement(interp, tcl_result, objBinRMSD);
  Tcl_SetObjResult(interp, tcl_result);

  delete [] histo;
  delete [] num;
  delete [] maxEntry;
  delete [] binrmsd;
  return TCL_OK;
}

int obj_volmap(ClientData cd, Tcl_Interp *interp, int argc, Tcl_Obj * const objv[]) {
    
  if (argc < 2) {
    Tcl_SetResult(interp, (char *)
      "usage: volmap <command> <args...>\n"
      "\nVolmap Creation:\n"
      " volmap <maptype> <selection> [opts...]   -- create a new volmap file\n"
      " maptypes:\n"
      "   density    -- arbitrary-weight density map [atoms/A^3]\n"
      "   interp     -- arbitrary-weight interpolation map [atoms/A^3]\n"
      "   distance   -- distance nearest atom surface [A]\n"
      "   occupancy  -- percent atomic occupancy of gridpoints [%]\n"
      "   mask       -- binary mask by painting spheres around atoms\n"
      "   coulomb    -- Coulomb electrostatic potential [kT/e] (slow)\n"
      "   coulombmsm -- Coulomb electrostatic potential [kT/e] (fast)\n"
      "   ils        -- free energy map [kT] computed by implicit ligand sampling\n"
      " options common to all maptypes:\n"
      "   -o <filename>           -- output DX format file name (use .dx extension)\n"
      "   -mol <molid>            -- export volmap into the specified mol\n"
      "   -res <float>            -- resolution in A of smallest cube\n"
      "   -allframes              -- compute for all frames of the trajectory\n"
      "   -combine <arg>          -- rule for combining the different frames\n"
      "                              <arg> = avg, min, max, stdev or pmf\n"
      "   -minmax <list of 2 vectors>   -- specify boundary of output grid\n"
      " options specific to certain maptypes:\n"
      "   -points                 -- use point particles for occupancy\n"
      "   -cutoff <float>         -- distance cutoff for calculations [A]\n"
      "   -radscale <float>       -- premultiply all atomic radii by a factor\n"
      "   -weight <str/list>      -- per atom weights for calculation\n"
      " options for ils:\n"
      "   see documentation\n", NULL);
    return TCL_ERROR;
  }

  VMDApp *app = (VMDApp *)cd;
  char *arg1 = Tcl_GetStringFromObj(objv[1],NULL);

  // If maptype is recognized, proceed with the map creation (vs. yet-unimplemented map operations)...
  if (argc > 1 && !strupncmp(arg1, "occupancy", CMDLEN))
    return vmd_volmap_new_fromtype(app, argc-1, objv+1, interp);
  if (argc > 1 && !strupncmp(arg1, "density", CMDLEN))
    return vmd_volmap_new_fromtype(app, argc-1, objv+1, interp);
  if (argc > 1 && !strupncmp(arg1, "interp", CMDLEN))
    return vmd_volmap_new_fromtype(app, argc-1, objv+1, interp);
  if (argc > 1 && !strupncmp(arg1, "distance", CMDLEN))
    return vmd_volmap_new_fromtype(app, argc-1, objv+1, interp);
  if (argc > 1 && !strupncmp(arg1, "mask", CMDLEN))
    return vmd_volmap_new_fromtype(app, argc-1, objv+1, interp);
  if (argc > 1 && (!strupncmp(arg1, "coulombpotential", CMDLEN) || 
                   !strupncmp(arg1, "coulomb", CMDLEN) ||
                   !strupncmp(arg1, "coulombmsm", CMDLEN)))
    return vmd_volmap_new_fromtype(app, argc-1, objv+1, interp);   

  if (argc > 1 && !strupncmp(arg1, "compare", CMDLEN))
    return vmd_volmap_compare(app, argc-1, objv+1, interp);

  if (argc > 1 && !strupncmp(arg1, "ils", CMDLEN))
    return vmd_volmap_ils(app, argc-1, objv+1, interp);
  if (argc > 1 && !strupncmp(arg1, "ligand", CMDLEN))
    return vmd_volmap_ils(app, argc-1, objv+1, interp);

  Tcl_SetResult(interp, (char *)"Type 'volmap' for summary of usage\n",NULL);
  return TCL_ERROR;
}
