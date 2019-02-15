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
 *      $RCSfile: TclMDFF.C,v $
 *      $Author: ryanmcgreevy $        $Locker:  $             $State: Exp $
 *      $Revision: 1.30 $      $Date: 2019/01/30 18:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Tcl bindings for MDFF functions
 *
 ***************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h> // FLT_MAX etc

#include "Inform.h"
#include "utilities.h"

#include "AtomSel.h"
#include "VMDApp.h"
#include "MoleculeList.h"
#include "VolumetricData.h"
#include "VolMapCreate.h" // volmap_write_dx_file()

#include "CUDAMDFF.h"
#include "MDFF.h"
#include <math.h>
#include <tcl.h>
#include "TclCommands.h"
#include "TclMDFF.h"
#include "Voltool.h"


int mdff_sim(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  int verbose = 0;
  if (argc < 3) {
     // "     options: --allframes (average over all frames)\n"
    Tcl_SetResult(interp, (char *) "usage: voltool "
      "sim: <selection> [options]\n"
//      "              --weight (weight density with atomic numbers)\n"
      "              -o <output map> \n"
      "              -res <target resolution in Angstroms> (default 10.0)\n"
      "              -spacing <grid spacing in Angstroms> (default based on resolution)\n",
      TCL_STATIC);
    return TCL_ERROR;
  }

  //atom selection
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
    Tcl_AppendResult(interp, "invalide mol id.", NULL);
    return TCL_ERROR;
  }

//  int ret_val=0;
  float radscale;
  double gspacing = 0;
  double resolution = 10.0;
//  bool use_all_frames = false;
//  bool useweight = false;
//  VolumetricData *volmap = NULL;
  MoleculeList *mlist = app->moleculeList;
  Molecule *mymol = mlist->mol_from_id(sel->molid());
//  const char *outputmap = Tcl_GetStringFromObj(objv[2], NULL);
  const char *outputmap = NULL;

  for (int i=0; i < argc; i++) {
    char *opt = Tcl_GetStringFromObj(objv[i], NULL);
//    if (!strcmp(opt, "--weight")) {useweight = true;}
//    if (!strcmp(opt, "--allframes")) {use_all_frames = true;}
    if (!strcmp(opt, "-res")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No resolution specified",NULL);
        return TCL_ERROR;
      } else if (Tcl_GetDoubleFromObj(interp, objv[i+1], &resolution) != TCL_OK) { 
        Tcl_AppendResult(interp, "\nResolution incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }
    if (!strcmp(opt, "-o")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No output file specified",NULL);
        return TCL_ERROR;
      } else {
        outputmap = Tcl_GetStringFromObj(objv[i+1], NULL);
      }
    }
    if (!strcmp(opt, "-spacing")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No spacing specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetDoubleFromObj(interp, objv[i+1], &gspacing) != TCL_OK) {
        Tcl_AppendResult(interp, "\nspacing incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }

    if (!strcmp(opt, "-verbose") || (getenv("VMDMDFFVERBOSE") != NULL)) {
      verbose = 1;
    }
  }

  // use quicksurf to compute simulated density map
  const float *framepos = sel->coordinates(app->moleculeList);
  const float *radii = mymol->radius();
  radscale = .2*resolution;

  if (gspacing == 0) {
    gspacing = 1.5*radscale;
  }

  int quality = 0;
  if (resolution >= 9)
    quality = 0;
  else
    quality = 3;

  if (verbose)
    printf("MDFF dens: radscale %f gspacing %f\n", radscale, gspacing);

  int cuda_err = -1;
#if defined(VMDCUDA)
  VolumetricData *synthvol=NULL;
  if (getenv("VMDNOCUDA") == NULL) {
    cuda_err = vmd_cuda_calc_density(sel, app->moleculeList, quality, radscale, gspacing, &synthvol, NULL, NULL, verbose);
    init_new_volume_molecule(app, synthvol, "sim_map");
    if (outputmap != NULL) volmap_write_dx_file(synthvol, outputmap);
  }
#endif

  // If CUDA failed, we use CPU fallback, and we have to prevent QuickSurf
  // from using the GPU either...
  if (cuda_err == -1) {
    const int force_cpu_calc=1;
    QuickSurf *qs = new QuickSurf(force_cpu_calc);
    VolumetricData *volmap = NULL;
    volmap = qs->calc_density_map(sel, mymol, framepos, radii,
                                  quality, (float)radscale, (float)gspacing);
    init_new_volume_molecule(app, volmap, "sim_map");
    if (outputmap != NULL) volmap_write_dx_file(volmap, outputmap);
    delete qs;
  }

  return TCL_OK;
}



int mdff_cc(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  if (argc < 3) {
    Tcl_SetResult(interp,
      (char *) "usage: voltool "
      "cc: <selection> -res <target resolution in A> [options]\n"
      "     options: --allframes (average over all frames)\n"
    //  "              --weight (weight simulated density with atomic numbers)\n"
      "              -i <input map> specifies new target density filename to load.\n"
      "              -mol <molid> specifies an already loaded density's molid for use as target\n"
      "              -thresholddensity <x> (ignores voxels with values below x threshold)\n"
      "              -vol <volume id> specifies an already loaded density's volume id for use as target. Defaults to 0.\n"
      "              -spacing <grid spacing in Angstroms> (default based on resolution)\n"
      "              -calcsynthmap append the simulated density map to the molecule.\n" 
      "              -calcdiffmap append a difference map (between the simulated and target densities) to the molecule.(CUDA only)\n" 
      "              -calcspatialccmap append to the molecule a spatial map of the correlations between 8x8 voxel regions of the simulated and target densities.(CUDA only)\n" 
      "              -savesynthmap <output file> save the simulated density to a file.\n"
      "              -savespatialccmap <output file> save the spatial correlaiton map to a file.(CUDA only)\n"
      ,
      TCL_STATIC);
    return TCL_ERROR;
  }

  //atom selection
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
    Tcl_AppendResult(interp, "invalide mol id.", NULL);
    return TCL_ERROR;
  }

//  int ret_val=0;
//  bool useweight = false;
  int verbose = 0;
  float radscale;
  int calcsynthmap = 0;
  const char *savesynthmap = NULL;
  int calcdiffmap = 0;
  int calcspatialccmap = 0;
  const char *savespatialccmap = NULL;
  double gspacing = 0;
  double thresholddensity = -FLT_MAX;

  int molid = -1;
  int volid = 0;
  double resolution;
  const char *input_map = NULL;
  bool use_all_frames = false;
//  VolumetricData *volmap = NULL;
  MoleculeList *mlist = app->moleculeList;
  Molecule *mymol = mlist->mol_from_id(sel->molid());

#if 0
  if ( Tcl_GetDoubleFromObj(interp, objv[2], &resolution) != TCL_OK){
    Tcl_AppendResult(interp, "\nResolution incorrectly specified",NULL);
    return TCL_ERROR;
  }
#endif

  for (int i=0; i < argc; i++) {
    char *opt = Tcl_GetStringFromObj(objv[i], NULL);
//    if (!strcmp(opt, "--weight")) {useweight = true;}
    if (!strcmp(opt, "--allframes")) {use_all_frames = true;}
    if (!strcmp(opt, "-i")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No input map specified",NULL);
        return TCL_ERROR;
      }

      FileSpec spec;
      spec.waitfor = FileSpec::WAIT_BACK; // shouldn't this be waiting for all?
      input_map = Tcl_GetStringFromObj(objv[1+i], NULL);
      molid = app->molecule_new(input_map,0,1);
      //sel->molid()
      int ret_val = app->molecule_load(molid, input_map,app->guess_filetype(input_map),&spec);
      if (ret_val < 0) return TCL_ERROR;
    }

    if (!strcmp(opt, "-res")) {
      if (i == argc-1){
        Tcl_AppendResult(interp, "No resolution specified",NULL);
        return TCL_ERROR;
      } else if (Tcl_GetDoubleFromObj(interp, objv[i+1], &resolution) != TCL_OK){ 
        Tcl_AppendResult(interp, "\n resolution incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }

    if (!strcmp(opt, "-spacing")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No spacing specified",NULL);
        return TCL_ERROR;
      } else if (Tcl_GetDoubleFromObj(interp, objv[i+1], &gspacing) != TCL_OK) {
          Tcl_AppendResult(interp, "\n spacing incorrectly specified",NULL);
          return TCL_ERROR;
      }
    }

    if (!strcmp(opt, "-mol")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No molid specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetIntFromObj(interp, objv[i+1], &molid) != TCL_OK) {
        Tcl_AppendResult(interp, "\n molid incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }

    if (!strcmp(opt, "-vol")) {
      if (i == argc-1){
        Tcl_AppendResult(interp, "No volume id specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetIntFromObj(interp, objv[i+1], &volid) != TCL_OK) {
        Tcl_AppendResult(interp, "\n volume id incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }

    if (!strcmp(opt, "-thresholddensity")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No threshold specified",NULL);
        return TCL_ERROR;
      } else if (Tcl_GetDoubleFromObj(interp, objv[i+1], &thresholddensity) != TCL_OK) { 
        Tcl_AppendResult(interp, "\nthreshold incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }

    // calculate (and/or save) simulated density map
    if (!strcmp(opt, "-calcsynthmap")) {
      calcsynthmap = 1;
    }
    if (!strcmp(opt, "-savesynthmap")) {
      calcsynthmap = 1;
      if (i == argc-1){
        Tcl_AppendResult(interp, "No output dx file specified", NULL);
        return TCL_ERROR;
      } else {
        savesynthmap = Tcl_GetStringFromObj(objv[i+1], NULL);
#if 0
        Tcl_AppendResult(interp, "\n volume id incorrectly specified",NULL);
        return TCL_ERROR;
#endif
      }
    }


    if (!strcmp(opt, "-calcdiffmap")) {
      calcdiffmap = 1;
    }


    // calculate (and/or save) spatialcc map
    if (!strcmp(opt, "-calcspatialccmap")) {
      calcspatialccmap = 1;
    }
    if (!strcmp(opt, "-savespatialccmap")) {
      calcspatialccmap = 1;
      if (i == argc-1){
        Tcl_AppendResult(interp, "No output dx file specified", NULL);
        return TCL_ERROR;
      } else {
        savespatialccmap = Tcl_GetStringFromObj(objv[i+1], NULL);
#if 0
        Tcl_AppendResult(interp, "\n volume id incorrectly specified",NULL);
        return TCL_ERROR;
#endif
      }
    }


    if (!strcmp(opt, "-verbose") || (getenv("VMDMDFFVERBOSE") != NULL)) {
      verbose = 1;
    }
  }


  const VolumetricData *volmapA = NULL;
  if (molid > -1) {
    Molecule *volmol = mlist->mol_from_id(molid);
    if (volmol == NULL) {
      Tcl_AppendResult(interp, "\n invalid molecule specified",NULL);
      return TCL_ERROR;
    }

    if (volmapA == NULL) 
      volmapA = volmol->get_volume_data(volid);
  } else {
    Tcl_AppendResult(interp, "\n no target volume specified",NULL);
    return TCL_ERROR;
  }
  if (volmapA == NULL) {
    Tcl_AppendResult(interp, "\n no target volume correctly specified",NULL);
    return TCL_ERROR;
  }

  float return_cc = 0;
  int start, end;
  if (use_all_frames) {
    start = 0;
    end = mymol->numframes()-1;
  } else {
    // start = mymol->frame();
    // end = mymol->frame();
    start = sel->which_frame;
    end = sel->which_frame;
  }

  PROFILE_PUSH_RANGE("MDFF cross correlation", 5);

  // use quicksurf density map algorithm
  for (int frame = start; frame <= end; frame++) {
    const float *framepos = new float [sel->num_atoms*3L];
 
    // frame is -2 or -1, meaning last or current but 
    // either way not allframes, so current sel frame is fine
    if (frame < 0)
      framepos = sel->coordinates(app->moleculeList);
    else
      framepos = (mymol->get_frame(frame))->pos;
    const float *radii = mymol->radius();

    radscale = .2*resolution;
 //   if(gspacing == 0){gspacing = 1.5*radscale;}
    int quality = 0;
    if (resolution >=9){quality = 0;}
    else{quality = 3;}

    VolumetricData *synthvol=NULL;
    VolumetricData *diffvol=NULL;

#if 0
    double cc=0.0;
    if (threshold != 0.0) {
      printf("Calculating threshold...\n");
      int size = synthvol->xsize * synthvol->ysize * synthvol->zsize;
      double mean = 0.;
      int i;
      for (i=0; i<size; i++)
        mean += synthvol->data[i];
      mean /= size;

      double sigma = 0.;
      for (i=0; i<size; i++)
        sigma += (synthvol->data[i] - mean)*(synthvol->data[i] - mean);
      sigma /= size;
      sigma = sqrt(sigma);

      threshold = mean + threshold*sigma;
      printf("Threshold: %f\n", threshold);
    }
#endif

    VolumetricData *synthmap = NULL;
    VolumetricData *diffmap = NULL;
    VolumetricData *spatialccmap = NULL;

    if (verbose) {
      if (calcsynthmap)
        printf("MDFF calculating simulated map\n");
      if (calcdiffmap)
        printf("MDFF calculating difference map\n");
      if (calcspatialccmap)
        printf("MDFF calculating spatial CC map\n");
    }
    

    // 
    // Try computing with CUDA first
    //
    int cuda_err = -1;
#if defined(VMDCUDA)
    VolumetricData **synthpp = NULL;
    VolumetricData **diffpp = NULL;
    VolumetricData **spatialccpp = NULL;

    if (calcsynthmap) {
      synthpp = &synthmap;
    }

    if (calcdiffmap) {
      diffpp = &diffmap;
    }

    if (calcspatialccmap) {
      spatialccpp = &spatialccmap;
    }

    if (gspacing == 0 && (getenv("VMDNOCUDA") == NULL)) {
      if (verbose)
        printf("Computing CC on GPU...\n");

#if 0
      if (verbose) {
        printf("TclMDFF: prep for vmd_cuda_compare_sel_refmap():\n");
        printf("  refmap xaxis: %lf %lf %lf\n",
               volmapA->xaxis[0], volmapA->xaxis[1], volmapA->xaxis[2]);
        printf("  refmap size: %d %d %d\n",
               volmapA->xsize, volmapA->ysize, volmapA->zsize);
        printf("  gridspacing (orig): %f\n", gspacing);
      }
#endif

      cuda_err = vmd_cuda_compare_sel_refmap(sel, app->moleculeList, quality, 
                                    radscale, gspacing, 
                                    volmapA, synthpp, diffpp, spatialccpp, 
                                    &return_cc, thresholddensity, verbose);
    }
#endif

    // If CUDA failed, we use CPU fallback, and we have to prevent QuickSurf
    // from using the GPU either...
    if (cuda_err == -1) {
      const int force_cpu_calc=1;
      if (verbose)
        printf("Computing CC on CPUs...\n");

      if (gspacing == 0) {
        gspacing = 1.5*radscale;
      }

      QuickSurf *qs = new QuickSurf(force_cpu_calc);
      VolumetricData *volmap = NULL;
      volmap = qs->calc_density_map(sel, mymol, framepos, radii,
                                    quality, (float)radscale, (float)gspacing);
      double cc = 0.0;

#if 0
      // this is 'old style' sigma threshold 
      if (thresholddensity != -FLT_MAX) {
        printf("Calculating threshold...\n"); 
        int size = volmap->xsize*volmap->ysize*volmap->zsize;
        double mean = 0.;
        int i;
        for (i=0; i<size; i++)
          mean += volmap->data[i];
        mean /= size;
 
        double sigma = 0.;
        for (i=0; i<size; i++)
          sigma += (volmap->data[i] - mean)*(volmap->data[i] - mean);
        sigma /= size;
        sigma = sqrt(sigma);
 
        thresholddensity = mean + thresholddensity*sigma;
        printf("Threshold: %f\n", thresholddensity);
      }
#endif
  
      cc_threaded(volmap, volmapA, &cc, thresholddensity);
      return_cc += cc;
      delete qs;
  
      if (start != end) {
        return_cc /= mymol->numframes();
      }
      Tcl_SetObjResult(interp, Tcl_NewDoubleObj(return_cc));
  
      VolumetricData *vol = NULL;
      if (calcsynthmap) {
        vol=volmap;
        if (savesynthmap) {
          printf("Writing simulated map to '%s'\n", savesynthmap);
          volmap_write_dx_file(vol, savesynthmap);
        } else {
          printf("Adding simulated map to molecule\n");
          app->molecule_add_volumetric(molid, vol->name, vol->origin, 
                                       vol->xaxis, vol->yaxis, vol->zaxis,
                                       vol->xsize, vol->ysize, vol->zsize, 
                                       vol->data);
          vol->data = NULL; // prevent destruction of density array;
        }
      }

      delete volmap;

      PROFILE_POP_RANGE(); // first return point
      return TCL_OK;
    }

    VolumetricData *v = NULL;
    if (calcsynthmap) {
      v=synthmap;
      if (savesynthmap) {
        printf("Writing simulated map to '%s'\n", savesynthmap);
        volmap_write_dx_file(v, savesynthmap);
      } else {
        printf("Adding simulated map to molecule\n");
        app->molecule_add_volumetric(molid, v->name, v->origin, 
                                     v->xaxis, v->yaxis, v->zaxis,
                                     v->xsize, v->ysize, v->zsize, v->data);
        v->data = NULL; // prevent destruction of density array;
      }
      delete v;
    }
 
    if (calcdiffmap) {
      printf("Adding difference map to molecule\n");
      v=diffmap;
      app->molecule_add_volumetric(molid, v->name, v->origin, 
                                   v->xaxis, v->yaxis, v->zaxis,
                                   v->xsize, v->ysize, v->zsize, v->data);
      v->data = NULL; // prevent destruction of density array;
      delete v;
    }

    if (calcspatialccmap) {
      v=spatialccmap;
      if (savespatialccmap) {
        printf("Writing spatialcc map to '%s'\n", savespatialccmap);
        volmap_write_dx_file(v, savespatialccmap);
      } else {
        printf("Adding spatial CC map to molecule\n");
        app->molecule_add_volumetric(molid, v->name, v->origin, 
                                     v->xaxis, v->yaxis, v->zaxis,
                                     v->xsize, v->ysize, v->zsize, v->data);
        v->data = NULL; // prevent destruction of density array;
      }
      delete v;
    }

    delete synthvol;
    delete diffvol;
  }

  if (start != end)
    return_cc /= mymol->numframes(); 

  PROFILE_POP_RANGE(); // second return point

  Tcl_SetObjResult(interp, Tcl_NewDoubleObj(return_cc));
  return TCL_OK;
}





int obj_mdff_cc(ClientData cd, Tcl_Interp *interp, int argc,
                            Tcl_Obj * const objv[]){
  if (argc < 2) {
    Tcl_SetResult(interp,
    (char *) "Usage: mdffi <command> [args...]\n"
      "Commands:\n"
      "cc   -- calculates the cross-correlation coefficient\n"
      "sim  -- creates a simulated map from an atomic structure\n"
      ,
      TCL_STATIC);
    return TCL_ERROR;
  }
  char *argv1 = Tcl_GetStringFromObj(objv[1],NULL);

  VMDApp *app = (VMDApp *)cd;
  if (!strupncmp(argv1, "cc", CMDLEN))
    return mdff_cc(app, argc-1, objv+1, interp);
  if (!strupncmp(argv1, "sim", CMDLEN))
    return mdff_sim(app, argc-1, objv+1, interp);

  Tcl_SetResult(interp, (char *) "Type 'mdffi' for summary of usage\n", TCL_VOLATILE);
  return TCL_OK;
}


