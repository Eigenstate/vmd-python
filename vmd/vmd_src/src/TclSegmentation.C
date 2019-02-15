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
 *      $RCSfile: TclSegmentation.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.14 $      $Date: 2019/01/17 21:38:55 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Tcl bindings for density map segmentation functions
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

#include "Segmentation.h"
#include "MDFF.h"
#include <math.h>
#include <tcl.h>
#include "TclCommands.h"


int return_segment_usage(Tcl_Interp *interp) {
  Tcl_SetResult(interp, (char *) "usage: segmentation "
    "segment: -mol <mol> -vol <vol> [options]\n"
    "  options:\n"
    "    -mol <molid> specifies an already loaded density's molid\n"
    "         for use as the segmentation volume source\n"
    "    -vol <volume id> specifies an already loaded density's\n"
    "         volume id for use as the volume source. Defaults to 0.\n"
    "    -groups <count> iterate merging groups until no more than the\n"
    "         target number of segment groups remain\n"
    "    -watershed_blur <sigma> initial Gaussian blur sigma to use\n"
    "         prior to the first pass of the Watershed algorithm\n"
    "    -starting_sigma <sigma> initial Gaussian blur sigma to use\n"
    "         for the iterative scale-space segmentation algorithm\n"
    "    -blur_multiple <multfactor> multiplicative constant to apply\n"
    "         to the scale-space blur sigma at each iteration\n"
    "    -separate_groups flag causes each segment group to be emitted\n"
    "         as an additional density map, with all other group voxels\n"
    "         masked out, maintaining the dimensions of the original map\n"
    ,
    TCL_STATIC);

  return 0;
}


int segment_volume(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  int verbose = 0;
  if (argc < 3) {
    return_segment_usage(interp);
    return TCL_ERROR;
  }

  int seg_def_groups = 128;
  float def_blur_sigma = 1.0f;
  float def_initial_sigma = 1.0f;
  float def_blur_multiple = 1.5f;
  int def_separate_groups = 0;

  int num_groups_target = seg_def_groups;
  double watershed_blur_sigma = def_blur_sigma;
  double blur_starting_sigma = def_initial_sigma;
  double blur_multiple = def_blur_multiple;
  int separate_groups = def_separate_groups;

  int molid=-1;
  int volid=0;
  for (int i=0; i < argc; i++) {
    char *opt = Tcl_GetStringFromObj(objv[i], NULL);

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

    if (!strcmp(opt, "-groups")) {
      if (i == argc-1){
        Tcl_AppendResult(interp, "No target group count specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetIntFromObj(interp, objv[i+1], &num_groups_target) != TCL_OK) {
        Tcl_AppendResult(interp, "\n target group count incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }
 
    if (!strcmp(opt, "-watershed_blur")) {
      if (i == argc-1){
        Tcl_AppendResult(interp, "No target blur sigma specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetDoubleFromObj(interp, objv[i+1], &watershed_blur_sigma) != TCL_OK) {
        Tcl_AppendResult(interp, "\n watershed blur sigma incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }

    if (!strcmp(opt, "-starting_sigma")) {
      if (i == argc-1){
        Tcl_AppendResult(interp, "No starging sigma specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetDoubleFromObj(interp, objv[i+1], &watershed_blur_sigma) != TCL_OK) {
        Tcl_AppendResult(interp, "\n starting sigma incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }

    if (!strcmp(opt, "-blur_multiple")) {
      if (i == argc-1){
        Tcl_AppendResult(interp, "No blur multiple specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetDoubleFromObj(interp, objv[i+1], &blur_multiple) != TCL_OK) {
        Tcl_AppendResult(interp, "\n blur multiple incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }

    if (!strcmp(opt, "-separate_groups")) {
      separate_groups = 1;
    }

    if (!strcmp(opt, "-verbose") || (getenv("VMDSEGMENTATIONVERBOSE") != NULL)) {
      verbose = 1;
    }
  }

  if (verbose)
    msgInfo << "Verbose segmentation diagnostic output enabled." << sendmsg;

  MoleculeList *mlist = app->moleculeList;
  const VolumetricData *map = NULL;

  if (molid > -1) {
    Molecule *volmol = mlist->mol_from_id(molid);
    if (volmol != NULL)
      map = volmol->get_volume_data(volid);
  } else {
    Tcl_AppendResult(interp, "\n no target volume specified",NULL);
    return TCL_ERROR;
  }

  if (map == NULL) {
    Tcl_AppendResult(interp, "\n no target volume correctly specified",NULL);
    return TCL_ERROR;
  }

  // perform the segmentation
  msgInfo << "Performing scale-space image segmentation on mol[" 
          << molid << "], volume[" << volid << "]..." 
          << sendmsg;

  PROFILE_PUSH_RANGE("Segmentation (All Steps)", 6);

  // initialize a segmentation object, perform memory allocations, etc. 
  Segmentation seg(map, 1);

  // compute the segmentation
  seg.segment(num_groups_target, watershed_blur_sigma, blur_starting_sigma, blur_multiple, MERGE_HILL_CLIMB);

  // get the group results back

  // allocate output array for volumetric map of segment group indices, and 
  // perform any required type conversions and changes in storage locality
  float *fp32seggrpids = new float[map->gridsize()];
  seg.get_results<float>(fp32seggrpids);
  
  PROFILE_POP_RANGE();

  // propagate results to persistent storage
  app->molecule_add_volumetric(molid, "Segmentation", map->origin,
                               map->xaxis, map->yaxis, map->zaxis,
                               map->xsize, map->ysize, map->zsize,
                               fp32seggrpids);

  // Use segmentation group information to mask and split the 
  // segmented density map into multiple multiple density maps
  if (separate_groups) {
    PROFILE_PUSH_RANGE("Segmentation split groups", 1);
    msgInfo << "Splitting density map into individual group density maps" << sendmsg;
    // allocate output array for volumetric map of segment group indices, and 
    // perform any required type conversions and changes in storage locality
    long *l64seggrpids = new long[map->gridsize()];
    seg.get_results<long>(l64seggrpids);

    char title[4096];
    int nGroups = seg.get_num_groups();
    long nVoxels = map->gridsize();
    for (int i=0; i<nGroups; i++) {
      // each new density map will be owned/managed by VMD after addition 
      // to the target molecule
      float *output = new float[nVoxels];
      for (long v=0; v<nVoxels; v++) {
        output[v] = l64seggrpids[v] == i ? map->data[v] : 0.0f;
      }
      printf("Masking density map by individual groups: %.2f%% \r", 100 * i / (double)nGroups);
      snprintf(title, sizeof(title), "Segment Group %d", i);

      // propagate results to persistent storage
      app->molecule_add_volumetric(molid, "Segmentation", map->origin,
                                   map->xaxis, map->yaxis, map->zaxis,
                                   map->xsize, map->ysize, map->zsize,
                                   output);
    }

    msgInfo << "Finished processing segmentation groups." << sendmsg;
    PROFILE_POP_RANGE();
  }

  msgInfo << "Segmentation tasks complete." << sendmsg;
  return TCL_OK;
}


int obj_segmentation(ClientData cd, Tcl_Interp *interp, int argc, Tcl_Obj * const objv[]){
  if (argc < 2) {
    return_segment_usage(interp);
    return TCL_ERROR;
  }
  char *argv1 = Tcl_GetStringFromObj(objv[1],NULL);

  VMDApp *app = (VMDApp *)cd;
  if (!strupncmp(argv1, "segment", CMDLEN))
    return segment_volume(app, argc-1, objv+1, interp);

  return_segment_usage(interp);
  return TCL_OK;
}


