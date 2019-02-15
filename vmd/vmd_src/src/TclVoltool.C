/***************************************************************************
 *cr
 *cr            (C) Copyright 2007-2011 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: TclVoltool.C,v $
 *      $Author: ryanmcgreevy $        $Locker:  $             $State: Exp $
 *      $Revision: 1.4 $      $Date: 2019/01/25 20:27:47 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  General volumetric data processing routines, particularly supporting MDFF
 *
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h> // FLT_MAX etc
#include "Inform.h"
#include "utilities.h"
//#include "SymbolTable.h"

#include "AtomSel.h"
#include "VMDApp.h"
#include "MoleculeList.h"
#include "VolumetricData.h"
#include "VolMapCreate.h" // volmap_write_dx_file()

#include "CUDAMDFF.h"
#include "MDFF.h"
#include "TclMDFF.h"
#include <math.h>
#include <tcl.h>
#include "TclCommands.h"
#include "Measure.h"
#include "MolFilePlugin.h"
#include "Voltool.h"

#include <iostream>
#include <string>
#include <sstream>

void moveby(AtomSel *sel, float *vect, MoleculeList *mlist, float *framepos){
  
  for (int i=sel->firstsel; i<=sel->lastsel; i++) {
    if (sel->on[i]) {
      vec_add(framepos + 3L*i, framepos + 3L*i, vect);
    }
  }
}

void move(AtomSel *sel, Matrix4 mat, MoleculeList *mlist, float *framepos){
  measure_move(sel, framepos, mat);
}


double density_calc_cc(VolumetricData *volmapA, VolumetricData *volmapB, double thresholddensity) {

  double cc = 0.0;
  cc_threaded(volmapA, volmapB, &cc, thresholddensity);
  return cc;
}

double calc_cc (AtomSel *sel, VolumetricData *volmapA, float resolution, MoleculeList *mlist, float *framepos) {
  
  float radscale;
  double gspacing = 0;
  double thresholddensity = 0.1;
  int verbose = 0;
  float return_cc = 0;

  radscale = .2*resolution;
  gspacing = 1.5*radscale;

  int quality = 0;
  if (resolution >= 9)
    quality = 0;
  else
    quality = 3;
  
  Molecule *mymol = mlist->mol_from_id(sel->molid());
  const float *radii = mymol->radius();
  
  int cuda_err = -1;
#if defined(VMDCUDA)
  VolumetricData **synthpp = NULL;
  VolumetricData **diffpp = NULL;
  VolumetricData **spatialccpp = NULL;

  if (getenv("VMDNOCUDA") == NULL) {

    cuda_err = vmd_cuda_compare_sel_refmap(sel, mlist, quality, 
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
    cc_threaded(volmap, volmapA, &cc, thresholddensity);
    return_cc += cc;
    delete qs;
  }
  return return_cc;
}

void do_rotate(int stride, float *com, AtomSel *sel, int amt, float *axis, MoleculeList *mlist, float *framepos){
  double amount = DEGTORAD(stride*amt);
  Matrix4 mat;
  mat.rotate_axis(axis, (float) amount);
  move(sel, mat, mlist, framepos);
}

void do_density_rotate(int stride, int amt, float *axis, VolumetricData *synthvol){
//  float move1[3];
//  vec_scale(move1, -1.0, com);
  double amount = DEGTORAD(stride*amt);
  Matrix4 mat;
  mat.rotate_axis(axis, (float) amount);
//  moveby(sel, move1, mlist, framepos);
  vol_move(synthvol, mat.mat);
//  move(sel, mat, mlist, framepos);
//  moveby(sel, com, mlist, framepos);

}

void rotate(int stride, int max_rot, float *com, float *returncc, float *bestpos, AtomSel *sel, MoleculeList *mlist, VolumetricData *volmapA, float resolution, float *origin, float *framepos) {
  
 // float *framepos = sel->coordinates(mlist);
 // float bestpos[sel->selected]; 
  //float best_cc = -1;

  float move1[3];
  vec_scale(move1, -1.0, com);

  for( int x = 0; x < max_rot; x++) {
    for( int y = 0; y < max_rot; y++) {
      for( int z = 0; z < max_rot; z++) {
        
        //move sel to vmd origin
        moveby(sel, move1, mlist, framepos);
        //rotate x
        float axisx [3] = {1, 0, 0};
        do_rotate(stride, com, sel, x, axisx, mlist, framepos);
        //rotate y
        float axisy [3] = {0, 1, 0};
        do_rotate(stride, com, sel, y, axisy, mlist, framepos);
        //rotate z
        float axisz [3] = {0, 0, 1};
        do_rotate(stride, com, sel, z, axisz, mlist, framepos);
        //move sel back to its original com
        moveby(sel, com, mlist, framepos);
        
        float cc = calc_cc(sel, volmapA, resolution, mlist, framepos); 
        if (cc > *returncc) {
          *returncc = cc;
          for (int i=0; i<sel->selected*3L; i++) {
           // if (sel->on[i]) {
              bestpos[i] = framepos[i];
           // }
          }
        } else if (cc < *returncc) {
//          x++;
//          y++;
          z++;
        }
        for (int i=0; i<sel->selected*3L; i++) {
         // if (sel->on[i]) {
            framepos[i] = origin[i];
         // }
        }

      }
    } 
  }
}

void density_rotate(int stride, int max_rot, float *com, float *returncc, int *bestrot, VolumetricData *volmapA, VolumetricData *synthvol, float resolution, double *origin, double *dx, double *dy, double *dz) {
  
  int num = 0;
  float move1[3] = {0, 0, 0};
  for( int x = 0; x < max_rot; x++) {
    for( int y = 0; y < max_rot; y++) {
      for( int z = 0; z < max_rot; z++) {
        
        vol_moveto(synthvol, com, move1);
        //rotate x
        float axisx [3] = {1, 0, 0};
        do_density_rotate(stride, x, axisx, synthvol);  
       //do_rotate(stride, com, sel, x, axisx, mlist, framepos);
        //rotate y
        float axisy [3] = {0, 1, 0};
        //do_rotate(stride, com, sel, y, axisy, mlist, framepos);
        do_density_rotate(stride, y, axisy, synthvol);  
        //rotate z
        float axisz [3] = {0, 0, 1};
        //do_rotate(stride, com, sel, z, axisz, mlist, framepos);
        do_density_rotate(stride, z, axisz, synthvol);  
        
        float currcom[3];        
        vol_com(synthvol, currcom);
        vol_moveto(synthvol, currcom, com);
       /* 
        printf("origin %f %f %f\n", 
        synthvol->origin[0],
        synthvol->origin[1],
        synthvol->origin[2]);
        
        printf("delta %f %f %f\n", 
        synthvol->xaxis[0]/(synthvol->xsize - 1), 
        synthvol->xaxis[1]/(synthvol->xsize - 1),
        synthvol->xaxis[2]/(synthvol->xsize - 1));
        printf("delta %f %f %f\n", 
        synthvol->yaxis[0]/(synthvol->ysize - 1), 
        synthvol->yaxis[1]/(synthvol->ysize - 1),
        synthvol->yaxis[2]/(synthvol->ysize - 1));
        printf("delta %f %f %f\n", 
        synthvol->zaxis[0]/(synthvol->zsize - 1), 
        synthvol->zaxis[1]/(synthvol->zsize - 1),
        synthvol->zaxis[2]/(synthvol->zsize - 1));
        
        volmap_write_dx_file(synthvol, "test.dx");
*/
       // std::string filename = "output/map-" + std::to_string(num) + ".dx";
        //printf("filename: %s\n", filename.c_str());
        //volmap_write_dx_file(synthvol, filename.c_str());
        num++;
        float cc = density_calc_cc(volmapA, synthvol, 0.1);
  //      printf ("CC: %f\n", cc); 
        if (cc > *returncc) {
          *returncc = cc;
          bestrot[0] = x;
          bestrot[1] = y;  
          bestrot[2] = z;
        //  volmap_write_dx_file(synthvol, "best.dx");
            
        }
        synthvol->origin[0] = origin[0];
        synthvol->origin[1] = origin[1];
        synthvol->origin[2] = origin[2];
        synthvol->xaxis[0] = dx[0]; 
        synthvol->xaxis[1] = dx[1]; 
        synthvol->xaxis[2] = dx[2]; 
        synthvol->yaxis[0] = dy[0]; 
        synthvol->yaxis[1] = dy[1]; 
        synthvol->yaxis[2] = dy[2]; 
        synthvol->zaxis[0] = dz[0]; 
        synthvol->zaxis[1] = dz[1]; 
        synthvol->zaxis[2] = dz[2]; 
        
      
      }
    } 
  }
}

void reset_density(int stride, int *bestrot, VolumetricData *synthvol, float *synthcom, float *framepos, float *com, AtomSel *sel, MoleculeList *mlist) {

  float move1[3];
  vec_scale(move1, -1.0, synthcom);
  vol_moveto(synthvol, synthcom, move1);

  //rotate density x 
  float axisx [3] = {1, 0, 0};
  do_density_rotate(stride, bestrot[0], axisx, synthvol);  
  //rotate density y
  float axisy [3] = {0, 1, 0};
  do_density_rotate(stride, bestrot[1], axisy, synthvol);  
  //rotate density z
  float axisz [3] = {0, 0, 1};
  do_density_rotate(stride, bestrot[2], axisz, synthvol);  
  
  float currcom[3];
  vol_com(synthvol, currcom);
  vol_moveto(synthvol, currcom, com);
  
  vec_scale(move1, -1.0, com);
  //move sel to vmd origin
  moveby(sel, move1, mlist, framepos);
  //rotate sel x
  do_rotate(stride, com, sel, bestrot[0], axisx, mlist, framepos);
  //rotate sel y
  do_rotate(stride, com, sel, bestrot[1], axisy, mlist, framepos);
  //rotate sel z
  do_rotate(stride, com, sel, bestrot[2], axisz, mlist, framepos);
  //move sel back to its original com
  moveby(sel, com, mlist, framepos);

}
void reset_origin(float *origin, float *newpos, AtomSel *sel) {
  for (int i=0; i<sel->num_atoms*3L; i++) {
      origin[i] = newpos[i];
  }
}


int density_com(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  if (argc < 3) {
    Tcl_SetResult(interp, (char *) "usage: voltool "
      "com [options]\n"
      "    options:  -i <input map> specifies new density filename to load.\n"
      "              -mol <molid> specifies an already loaded density's molid for use as target\n"
      "              -vol <volume id> specifies an already loaded density's volume id for use as target. Defaults to 0.\n",
      TCL_STATIC);
    return TCL_ERROR;
  }

  int molid = -1;
  int volid = 0;
  const char *input_map = NULL;
  MoleculeList *mlist = app->moleculeList;
  
  for (int i=0; i < argc; i++) {
    char *opt = Tcl_GetStringFromObj(objv[i], NULL);
    if (!strcmp(opt, "-i")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No input map specified",NULL);
        return TCL_ERROR;
      }

      FileSpec spec;
      spec.waitfor = FileSpec::WAIT_BACK; // shouldn't this be waiting for all?
      input_map = Tcl_GetStringFromObj(objv[1+i], NULL);
      molid = app->molecule_new(input_map,0,1);
      int ret_val = app->molecule_load(molid, input_map,app->guess_filetype(input_map),&spec);
      if (ret_val < 0) return TCL_ERROR;
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

  }

  VolumetricData *volmapA = NULL;
  if (molid > -1) {
    Molecule *volmol = mlist->mol_from_id(molid);
    if (volmol == NULL) {
      Tcl_AppendResult(interp, "\n invalid molecule specified",NULL);
      return TCL_ERROR;
    }

    if (volmapA == NULL) 
      volmapA = volmol->modify_volume_data(volid);
  } else {
    Tcl_AppendResult(interp, "\n no target volume specified",NULL);
    return TCL_ERROR;
  }
  if (volmapA == NULL) {
    Tcl_AppendResult(interp, "\n no target volume correctly specified",NULL);
    return TCL_ERROR;
  }
  
  
  float com[3] = {0.0,0.0,0.0};
  vol_com(volmapA, com);  
 
  Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewDoubleObj(com[0]));
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewDoubleObj(com[1]));
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewDoubleObj(com[2]));

  Tcl_SetObjResult(interp, tcl_result);
  return TCL_OK;
}

int density_move(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  if (argc < 3) {
    Tcl_SetResult(interp, (char *) "usage: voltool "
      "move -mat <4x4 transform matrix to apply to density> [options]\n"
      "    options:  -i <input map> specifies new density filename to load.\n"
      "              -mol <molid> specifies an already loaded density's molid for use as target\n"
      "              -vol <volume id> specifies an already loaded density's volume id for use as target. Defaults to 0.\n"
      "              -o <filename> write density to file.\n",
      TCL_STATIC);
    return TCL_ERROR;
  }


  int molid = -1;
  int volid = 0;
  float mat[16];
  const char *input_map = NULL;
  const char *outputmap = NULL;
  MoleculeList *mlist = app->moleculeList;
  
  for (int i=0; i < argc; i++) {
    char *opt = Tcl_GetStringFromObj(objv[i], NULL);
    if (!strcmp(opt, "-i")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No input map specified",NULL);
        return TCL_ERROR;
      }

      FileSpec spec;
      spec.waitfor = FileSpec::WAIT_BACK; // shouldn't this be waiting for all?
      input_map = Tcl_GetStringFromObj(objv[1+i], NULL);
      molid = app->molecule_new(input_map,0,1);
      int ret_val = app->molecule_load(molid, input_map,app->guess_filetype(input_map),&spec);
      if (ret_val < 0) return TCL_ERROR;
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
    
    if (!strcmp(opt, "-mat")) {
      if (i == argc-1){
        Tcl_AppendResult(interp, "No matrix specified",NULL);
        return TCL_ERROR;
      } else if (tcl_get_matrix(Tcl_GetStringFromObj(objv[0],NULL), interp, objv[i+1], mat) != TCL_OK) {
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

  }
  Molecule *volmol = NULL;
  VolumetricData *volmapA = NULL;
  if (molid > -1) {
    volmol = mlist->mol_from_id(molid);
    if (volmol == NULL) {
      Tcl_AppendResult(interp, "\n invalid molecule specified",NULL);
      return TCL_ERROR;
    }

    if (volmapA == NULL) 
      volmapA = volmol->modify_volume_data(volid);
  } else {
    Tcl_AppendResult(interp, "\n no target volume specified",NULL);
    return TCL_ERROR;
  }
  if (volmapA == NULL) {
    Tcl_AppendResult(interp, "\n no target volume correctly specified",NULL);
    return TCL_ERROR;
  }
  
 
  vol_move(volmapA, mat); 
  volmol->force_recalc(DrawMolItem::MOL_REGEN);

  if (outputmap != NULL) {
    volmap_write_dx_file(volmapA, outputmap);
  }
  return TCL_OK;

}

int density_moveto(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  if (argc < 3) {
    Tcl_SetResult(interp, (char *) "usage: voltool "
      "moveto -pos {x y z} coordinates to move com to> [options]\n"
      "    options:  -i <input map> specifies new density filename to load.\n"
      "              -mol <molid> specifies an already loaded density's molid for use as target\n"
      "              -vol <volume id> specifies an already loaded density's volume id for use as target. Defaults to 0.\n"
      "              -o <filename> write density to file.\n",
      TCL_STATIC);
    return TCL_ERROR;
  }


  int molid = -1;
  int volid = 0;
  double pos[3] = {0.0, 0.0, 0.0};
  const char *input_map = NULL;
  const char *outputmap = NULL;
  MoleculeList *mlist = app->moleculeList;
  
  for (int i=0; i < argc; i++) {
    char *opt = Tcl_GetStringFromObj(objv[i], NULL);
    if (!strcmp(opt, "-i")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No input map specified",NULL);
        return TCL_ERROR;
      }

      FileSpec spec;
      spec.waitfor = FileSpec::WAIT_BACK; // shouldn't this be waiting for all?
      input_map = Tcl_GetStringFromObj(objv[1+i], NULL);
      molid = app->molecule_new(input_map,0,1);
      int ret_val = app->molecule_load(molid, input_map,app->guess_filetype(input_map),&spec);
      if (ret_val < 0) return TCL_ERROR;
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
    
    int num1;
    Tcl_Obj **vector;
    if (!strcmp(opt, "-pos")) {
      if (i == argc-1){
        Tcl_AppendResult(interp, "No position coordinate specified",NULL);
        return TCL_ERROR;
      } else if (Tcl_ListObjGetElements(interp, objv[i+1], &num1, &vector) != TCL_OK) {
      return TCL_ERROR;
      }
    
      for (int i=0; i<num1; i++) {
        if (Tcl_GetDoubleFromObj(interp, vector[i], &pos[i]) != TCL_OK) {
          Tcl_SetResult(interp, (char *) "vecscale: non-numeric in vector", TCL_STATIC);
          return TCL_ERROR;
        }
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
  }
  Molecule *volmol = NULL;
  VolumetricData *volmapA = NULL;
  if (molid > -1) {
    volmol = mlist->mol_from_id(molid);
    if (volmol == NULL) {
      Tcl_AppendResult(interp, "\n invalid molecule specified",NULL);
      return TCL_ERROR;
    }

    if (volmapA == NULL) 
      volmapA = volmol->modify_volume_data(volid);
  } else {
    Tcl_AppendResult(interp, "\n no target volume specified",NULL);
    return TCL_ERROR;
  }
  if (volmapA == NULL) {
    Tcl_AppendResult(interp, "\n no target volume correctly specified",NULL);
    return TCL_ERROR;
  }
  
 
  float com[3] = {0.0,0.0,0.0};
  float newpos[3] = {(float)pos[0], (float)pos[1], (float)pos[2]};
  vol_com(volmapA, com);  
  vol_moveto(volmapA, com, newpos);
  volmol->force_recalc(DrawMolItem::MOL_REGEN);

  if (outputmap != NULL) {
    volmap_write_dx_file(volmapA, outputmap);
  }
  return TCL_OK;

}

int fit(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  if (argc < 3) {
    Tcl_SetResult(interp, (char *) "usage: voltool "
      "fit <selection> -res <resolution of map in A> [options]\n"
      "    options:  -i <input map> specifies new target density filename to load.\n"
      "              -mol <molid> specifies an already loaded density's molid for use as target\n"
      "              -thresholddensity <x> (ignores voxels with values below x threshold)\n"
      "              -vol <volume id> specifies an already loaded density's volume id for use as target. Defaults to 0.\n",
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

  int ret_val=0;
  int molid = -1;
  int volid = 0;
  double resolution = 0;
  const char *input_map = NULL;
  MoleculeList *mlist = app->moleculeList;
  Molecule *mymol = mlist->mol_from_id(sel->molid());
  
  //parse arguments
  for (int i=0; i < argc; i++) {
    char *opt = Tcl_GetStringFromObj(objv[i], NULL);
    if (!strcmp(opt, "-i")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No input map specified",NULL);
        return TCL_ERROR;
      }

      FileSpec spec;
      spec.waitfor = FileSpec::WAIT_BACK; // shouldn't this be waiting for all?
      input_map = Tcl_GetStringFromObj(objv[1+i], NULL);
      molid = app->molecule_new(input_map,0,1);
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
  }

  VolumetricData *volmapA = NULL;
  if (molid > -1) {
    Molecule *volmol = mlist->mol_from_id(molid);
    if (volmol == NULL) {
      Tcl_AppendResult(interp, "\n invalid molecule specified",NULL);
      return TCL_ERROR;
    }

    if (volmapA == NULL) 
      volmapA = volmol->modify_volume_data(volid);
  } else {
    Tcl_AppendResult(interp, "\n no target volume specified",NULL);
    return TCL_ERROR;
  }
  if (volmapA == NULL) {
    Tcl_AppendResult(interp, "\n no target volume correctly specified",NULL);
    return TCL_ERROR;
  }
  
  
  float *framepos = sel->coordinates(app->moleculeList);
  //compute center of mass 
  float com[3];
  // get atom masses
  const float *weight = mymol->mass();
  ret_val = measure_center(sel, framepos, weight, com);
  if (ret_val < 0) {
    Tcl_AppendResult(interp, "measure center failed",
         NULL);
    return TCL_ERROR;
  }
 
  //move sel com to 
  float dencom[3] = {0.0,0.0,0.0};
  float move1[3];
  vol_com(volmapA, dencom);
  vec_sub(move1, dencom, com);
  moveby(sel, move1, mlist, framepos);
  //recalc sel com
  ret_val = measure_center(sel, framepos, weight, com);
  if (ret_val < 0) {
    Tcl_AppendResult(interp, "measure center failed",
         NULL);
    return TCL_ERROR;
  }
  
  //set up for rotational search
  float cc = -1;
  float *bestpos = new float [sel->num_atoms*3L];
  float *origin= new float [sel->num_atoms*3L];
  for (int k=0; k<sel->num_atoms*3L; k++) {
      origin[k] = framepos[k];
      bestpos[k] = framepos[k];
  }
    

/*
  //fit with map
  // use quicksurf to compute simulated density map
  float radscale;
  double gspacing = 0;
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

  VolumetricData *synthvol=NULL;
  int cuda_err = -1;
#if defined(VMDCUDA)
  if (getenv("VMDNOCUDA") == NULL) {
    cuda_err = vmd_cuda_calc_density(sel, app->moleculeList, quality, radscale, gspacing, &synthvol, NULL, NULL, verbose);
  //  delete synthvol;
  }
#endif

  // If CUDA failed, we use CPU fallback, and we have to prevent QuickSurf
  // from using the GPU either...
  if (cuda_err == -1) {
    const int force_cpu_calc=1;
    QuickSurf *qs = new QuickSurf(force_cpu_calc);
    synthvol = qs->calc_density_map(sel, mymol, framepos, radii,
                                  quality, (float)radscale, (float)gspacing);
//    volmap_write_dx_file(volmap, outputmap);
   // delete synthvol;
    delete qs;
  }
  
  float synthcom[3];
  vol_com(synthvol, synthcom);


  int bestrot[3];
  double *synthorigin = synthvol->origin;
  double *dx = synthvol->xaxis;
  double *dy = synthvol->yaxis;
  double *dz = synthvol->zaxis;

  int stride = 5;
  int max_rot = 360/stride;
  density_rotate(stride, max_rot, synthcom, &cc, bestrot, volmapA, synthvol, resolution, synthorigin, dx, dy, dz);
  
  reset_density(stride, bestrot, synthvol, synthcom, framepos, com, sel, mlist);
  
  int stride2 = 6;
  max_rot = stride/stride2;
  density_rotate(stride2, max_rot, synthcom, &cc, bestrot, volmapA, synthvol, resolution, synthorigin, dx, dy, dz);
  density_rotate(-stride2, max_rot, synthcom, &cc, bestrot, volmapA, synthvol, resolution, synthorigin, dx, dy, dz);

  int stride3 = 1;
  max_rot = stride2/stride3;
  density_rotate(stride3, max_rot, synthcom, &cc, bestrot, volmapA, synthvol, resolution, synthorigin, dx, dy, dz);
  reset_density(stride3, bestrot, synthvol, synthcom, framepos, com, sel, mlist);
  density_rotate(-stride3, max_rot, synthcom, &cc, bestrot, volmapA, synthvol, resolution, synthorigin, dx, dy, dz);

  reset_density(stride3, bestrot, synthvol, synthcom, framepos, com, sel, mlist);

*/


  //fit with struct
  int stride = 24;
  int max_rot = 360/stride;
  rotate(stride, max_rot, com, &cc, bestpos, sel, mlist, volmapA, resolution, origin, framepos);
  reset_origin(origin, bestpos, sel); 

  int stride2 = 4;
  max_rot = stride/stride2;
  rotate(stride2, max_rot, com, &cc, bestpos, sel, mlist, volmapA, resolution, origin, framepos);
  rotate(-stride2, max_rot, com, &cc, bestpos, sel, mlist, volmapA, resolution, origin, framepos);
  reset_origin(origin, bestpos, sel); 
  
  int stride3 = 1;
  max_rot = stride2/stride3;
  rotate(stride3, max_rot, com, &cc, bestpos, sel, mlist, volmapA, resolution, origin, framepos);
  rotate(-stride3, max_rot, com, &cc, bestpos, sel, mlist, volmapA, resolution, origin, framepos);
  
  for (int j=0; j<sel->num_atoms*3L; j++) {
      framepos[j] = bestpos[j];
  }
  
  // notify molecule that coordinates changed.
  mymol->force_recalc(DrawMolItem::MOL_REGEN);
 
 // int frame = app->molecule_frame(sel->molid());
 // FileSpec speco;
 // speco.first = frame;                // write current frame only
 // speco.last = frame;                 // write current frame only
 // speco.stride = 1;                   // write all selected frames
 // speco.waitfor = FileSpec::WAIT_ALL; // wait for all frames to be written
 // speco.selection = sel->on;      // write only selected atoms
 // app->molecule_savetrajectory(sel->molid(), "fittedi.pdb", "pdb", &speco);
  printf("Best CC:%f\n", cc);
  return TCL_OK;
}

int mask(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  if (argc < 3) {
    Tcl_SetResult(interp, (char *) "usage: voltool "
      "mask <selection> [options]\n"
      "    options:  -res <resolution of map in A> (Default: 5) \n"
      "              -cutoff <cutoff mask distance in A> (Default: 5) \n"
      "              -i <input map> specifies new target density filename to load.\n"
      "              -mol <molid> specifies an already loaded density's molid for use as target\n"
      "              -vol <volume id> specifies an already loaded density's volume id for use as target. Defaults to 0.\n"
      "              -o <filename> write density to file.\n",
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

  int molid = -1;
  int volid = 0;
  double resolution = 5;
  double cutoff = 5;
  const char *input_map = NULL;
  const char *outputmap = NULL;
  MoleculeList *mlist = app->moleculeList;
  
  //parse arguments
  for (int i=0; i < argc; i++) {
    char *opt = Tcl_GetStringFromObj(objv[i], NULL);
    if (!strcmp(opt, "-i")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No input map specified",NULL);
        return TCL_ERROR;
      }

      FileSpec spec;
      spec.waitfor = FileSpec::WAIT_BACK; // shouldn't this be waiting for all?
      input_map = Tcl_GetStringFromObj(objv[1+i], NULL);
      molid = app->molecule_new(input_map,0,1);
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
    
    if (!strcmp(opt, "-cutoff")) {
      if (i == argc-1){
        Tcl_AppendResult(interp, "No cutoff specified",NULL);
        return TCL_ERROR;
      } else if (Tcl_GetDoubleFromObj(interp, objv[i+1], &cutoff) != TCL_OK){ 
        Tcl_AppendResult(interp, "\n cutoff incorrectly specified",NULL);
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
    
    if (!strcmp(opt, "-o")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No output file specified",NULL);
        return TCL_ERROR;
      } else {
        outputmap = Tcl_GetStringFromObj(objv[i+1], NULL);
      }
    }
  }

  VolumetricData *volmapA = NULL;
  if (molid > -1) {
    Molecule *volmol = mlist->mol_from_id(molid);
    if (volmol == NULL) {
      Tcl_AppendResult(interp, "\n invalid molecule specified",NULL);
      return TCL_ERROR;
    }

    if (volmapA == NULL) 
      volmapA = volmol->modify_volume_data(volid);
  } else {
    Tcl_AppendResult(interp, "\n no target volume specified",NULL);
    return TCL_ERROR;
  }
  if (volmapA == NULL) {
    Tcl_AppendResult(interp, "\n no target volume correctly specified",NULL);
    return TCL_ERROR;
  }
  
  
  VolMapCreate *volcreate = new VolMapCreateMask(app, sel, (float)resolution, (float)cutoff);
  volcreate->compute_all(0, VolMapCreate::COMBINE_AVG, NULL);
  VolumetricData *mask = volcreate->volmap;
  VolumetricData *newvol = init_new_volume();
  bool USE_UNION = false;
  bool USE_INTERP = true;
  multiply(volmapA, mask, newvol, USE_INTERP, USE_UNION);
  
  init_new_volume_molecule(app, newvol, "masked_map");

  if (outputmap != NULL) {
    volmap_write_dx_file(newvol, outputmap);
  }
   
  return TCL_OK;
}

int density_trim(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  if (argc < 3) {
    Tcl_SetResult(interp, (char *) "usage: voltool "
      "trim -amt {x1 x2 y1 y2 z1 z2} amount to trim from each end in x, y, z axes> [options]\n"
      "    options:  -i <input map> specifies new density filename to load.\n"
      "              -mol <molid> specifies an already loaded density's molid for use as target\n"
      "              -vol <volume id> specifies an already loaded density's volume id for use as target. Defaults to 0.\n"
      "              -o <filename> write density to file.\n",
      TCL_STATIC);
    return TCL_ERROR;
  }


  int molid = -1;
  int volid = 0;
  int trim[6] = {0, 0, 0, 0, 0, 0};
  const char *input_map = NULL;
  const char *outputmap = NULL;
  MoleculeList *mlist = app->moleculeList;
  
  for (int i=0; i < argc; i++) {
    char *opt = Tcl_GetStringFromObj(objv[i], NULL);
    if (!strcmp(opt, "-i")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No input map specified",NULL);
        return TCL_ERROR;
      }

      FileSpec spec;
      spec.waitfor = FileSpec::WAIT_BACK; // shouldn't this be waiting for all?
      input_map = Tcl_GetStringFromObj(objv[1+i], NULL);
      molid = app->molecule_new(input_map,0,1);
      int ret_val = app->molecule_load(molid, input_map,app->guess_filetype(input_map),&spec);
      if (ret_val < 0) return TCL_ERROR;
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
    
    int num1;
    Tcl_Obj **vector;
    if (!strcmp(opt, "-amt")) {
      if (i == argc-1){
        Tcl_AppendResult(interp, "No trim amounts specified",NULL);
        return TCL_ERROR;
      } else if (Tcl_ListObjGetElements(interp, objv[i+1], &num1, &vector) != TCL_OK) {
      return TCL_ERROR;
      }
    
      for (int i=0; i<num1; i++) {
        if (Tcl_GetIntFromObj(interp, vector[i], &trim[i]) != TCL_OK) {
          Tcl_SetResult(interp, (char *) "amt: non-numeric in vector", TCL_STATIC);
          return TCL_ERROR;
        }
      }
      if (num1 != 6) {
        Tcl_SetResult(interp, (char *) "amt: incorrect number of values in vector", TCL_STATIC);
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
  }
  Molecule *volmol = NULL;
  VolumetricData *volmapA = NULL;
  if (molid > -1) {
    volmol = mlist->mol_from_id(molid);
    if (volmol == NULL) {
      Tcl_AppendResult(interp, "\n invalid molecule specified",NULL);
      return TCL_ERROR;
    }

    if (volmapA == NULL) 
      volmapA = volmol->modify_volume_data(volid);
  } else {
    Tcl_AppendResult(interp, "\n no target volume specified",NULL);
    return TCL_ERROR;
  }
  if (volmapA == NULL) {
    Tcl_AppendResult(interp, "\n no target volume correctly specified",NULL);
    return TCL_ERROR;
  }
  volmapA->pad(-trim[0], -trim[1], -trim[2], -trim[3], -trim[4], -trim[5]); 
  volmol->force_recalc(DrawMolItem::MOL_REGEN);
  
  if (outputmap != NULL) {
    volmap_write_dx_file(volmapA, outputmap);
  }
  return TCL_OK;

}

int density_crop(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  if (argc < 3) {
    Tcl_SetResult(interp, (char *) "usage: voltool "
      "crop -amt {minx miny minz maxx maxy maxz} minmax values given in coordinate space.> [options]\n"
      "    options:  -i <input map> specifies new density filename to load.\n"
      "              -mol <molid> specifies an already loaded density's molid for use as target\n"
      "              -vol <volume id> specifies an already loaded density's volume id for use as target. Defaults to 0.\n"
      "              -o <filename> write density to file.\n",
      TCL_STATIC);
    return TCL_ERROR;
  }


  int molid = -1;
  int volid = 0;
  int minmax[6] = {0, 0, 0, 0, 0, 0};
  const char *input_map = NULL;
  const char *outputmap = NULL;
  MoleculeList *mlist = app->moleculeList;
  
  for (int i=0; i < argc; i++) {
    char *opt = Tcl_GetStringFromObj(objv[i], NULL);
    if (!strcmp(opt, "-i")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No input map specified",NULL);
        return TCL_ERROR;
      }

      FileSpec spec;
      spec.waitfor = FileSpec::WAIT_BACK; // shouldn't this be waiting for all?
      input_map = Tcl_GetStringFromObj(objv[1+i], NULL);
      molid = app->molecule_new(input_map,0,1);
      int ret_val = app->molecule_load(molid, input_map,app->guess_filetype(input_map),&spec);
      if (ret_val < 0) return TCL_ERROR;
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
    
    int num1;
    Tcl_Obj **vector;
    if (!strcmp(opt, "-amt")) {
      if (i == argc-1){
        Tcl_AppendResult(interp, "No trim amounts specified",NULL);
        return TCL_ERROR;
      } else if (Tcl_ListObjGetElements(interp, objv[i+1], &num1, &vector) != TCL_OK) {
      return TCL_ERROR;
      }
    
      for (int i=0; i<num1; i++) {
        if (Tcl_GetIntFromObj(interp, vector[i], &minmax[i]) != TCL_OK) {
          Tcl_SetResult(interp, (char *) "amt: non-numeric in vector", TCL_STATIC);
          return TCL_ERROR;
        }
      }
      if (num1 != 6) {
        Tcl_SetResult(interp, (char *) "amt: incorrect number of values in vector", TCL_STATIC);
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
  }
  Molecule *volmol = NULL;
  VolumetricData *volmapA = NULL;
  if (molid > -1) {
    volmol = mlist->mol_from_id(molid);
    if (volmol == NULL) {
      Tcl_AppendResult(interp, "\n invalid molecule specified",NULL);
      return TCL_ERROR;
    }

    if (volmapA == NULL) 
      volmapA = volmol->modify_volume_data(volid);
  } else {
    Tcl_AppendResult(interp, "\n no target volume specified",NULL);
    return TCL_ERROR;
  }
  if (volmapA == NULL) {
    Tcl_AppendResult(interp, "\n no target volume correctly specified",NULL);
    return TCL_ERROR;
  }
  volmapA->crop(minmax[0], minmax[1], minmax[2], minmax[3], minmax[4], minmax[5]); 
  volmol->force_recalc(DrawMolItem::MOL_REGEN);
  
  if (outputmap != NULL) {
    volmap_write_dx_file(volmapA, outputmap);
  }
  return TCL_OK;

}

int density_clamp(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  if (argc < 3) {
    Tcl_SetResult(interp, (char *) "usage: voltool "
      "clamp [options]\n"
      "    options:  -min <min voxel value> Defaults to existing min.\n"
      "              -max <max voxel value> Defaults to existing max.\n"
      "              -i <input map> specifies new density filename to load.\n"
      "              -mol <molid> specifies an already loaded density's molid for use as target\n"
      "              -vol <volume id> specifies an already loaded density's volume id for use as target. Defaults to 0.\n"
      "              -o <filename> write density to file.\n",
      TCL_STATIC);
    return TCL_ERROR;
  }


  int molid = -1;
  int volid = 0;
  double min = -FLT_MIN;
  double max = FLT_MAX;
  const char *input_map = NULL;
  const char *outputmap = NULL;
  MoleculeList *mlist = app->moleculeList;
  
  for (int i=0; i < argc; i++) {
    char *opt = Tcl_GetStringFromObj(objv[i], NULL);
    if (!strcmp(opt, "-i")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No input map specified",NULL);
        return TCL_ERROR;
      }

      FileSpec spec;
      spec.waitfor = FileSpec::WAIT_BACK; // shouldn't this be waiting for all?
      input_map = Tcl_GetStringFromObj(objv[1+i], NULL);
      molid = app->molecule_new(input_map,0,1);
      int ret_val = app->molecule_load(molid, input_map,app->guess_filetype(input_map),&spec);
      if (ret_val < 0) return TCL_ERROR;
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
    
    if (!strcmp(opt, "-min")) {
      if (i == argc-1){
        Tcl_AppendResult(interp, "No minimum voxel specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetDoubleFromObj(interp, objv[i+1], &min) != TCL_OK) {
        Tcl_AppendResult(interp, "\n minimum voxel incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }
    
    if (!strcmp(opt, "-max")) {
      if (i == argc-1){
        Tcl_AppendResult(interp, "No maximum voxel specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetDoubleFromObj(interp, objv[i+1], &max) != TCL_OK) {
        Tcl_AppendResult(interp, "\n maximum voxel incorrectly specified",NULL);
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
  }
  Molecule *volmol = NULL;
  VolumetricData *volmapA = NULL;
  if (molid > -1) {
    volmol = mlist->mol_from_id(molid);
    if (volmol == NULL) {
      Tcl_AppendResult(interp, "\n invalid molecule specified",NULL);
      return TCL_ERROR;
    }

    if (volmapA == NULL) 
      volmapA = volmol->modify_volume_data(volid);
  } else {
    Tcl_AppendResult(interp, "\n no target volume specified",NULL);
    return TCL_ERROR;
  }
  if (volmapA == NULL) {
    Tcl_AppendResult(interp, "\n no target volume correctly specified",NULL);
    return TCL_ERROR;
  }
  volmapA->clamp(min, max); 
  volmol->force_recalc(DrawMolItem::MOL_REGEN);
  
  if (outputmap != NULL) {
    volmap_write_dx_file(volmapA, outputmap);
  }
  return TCL_OK;

}

int density_smult(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  if (argc < 3) {
    Tcl_SetResult(interp, (char *) "usage: voltool "
      "smult -amt <x> multiply every voxel by x. [options]\n"
      "    options:  -i <input map> specifies new density filename to load.\n"
      "              -mol <molid> specifies an already loaded density's molid for use as target\n"
      "              -vol <volume id> specifies an already loaded density's volume id for use as target. Defaults to 0.\n"
      "              -o <filename> write density to file.\n",
      TCL_STATIC);
    return TCL_ERROR;
  }


  int molid = -1;
  int volid = 0;
  double amt = 1;
  const char *input_map = NULL;
  const char *outputmap = NULL;
  MoleculeList *mlist = app->moleculeList;
  
  for (int i=0; i < argc; i++) {
    char *opt = Tcl_GetStringFromObj(objv[i], NULL);
    if (!strcmp(opt, "-i")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No input map specified",NULL);
        return TCL_ERROR;
      }

      FileSpec spec;
      spec.waitfor = FileSpec::WAIT_BACK; // shouldn't this be waiting for all?
      input_map = Tcl_GetStringFromObj(objv[1+i], NULL);
      molid = app->molecule_new(input_map,0,1);
      int ret_val = app->molecule_load(molid, input_map,app->guess_filetype(input_map),&spec);
      if (ret_val < 0) return TCL_ERROR;
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
    
    if (!strcmp(opt, "-amt")) {
      if (i == argc-1){
        Tcl_AppendResult(interp, "No scaling amount specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetDoubleFromObj(interp, objv[i+1], &amt) != TCL_OK) {
        Tcl_AppendResult(interp, "\n scaling amount incorrectly specified",NULL);
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
  }
  Molecule *volmol = NULL;
  VolumetricData *volmapA = NULL;
  if (molid > -1) {
    volmol = mlist->mol_from_id(molid);
    if (volmol == NULL) {
      Tcl_AppendResult(interp, "\n invalid molecule specified",NULL);
      return TCL_ERROR;
    }

    if (volmapA == NULL) 
      volmapA = volmol->modify_volume_data(volid);
  } else {
    Tcl_AppendResult(interp, "\n no target volume specified",NULL);
    return TCL_ERROR;
  }
  if (volmapA == NULL) {
    Tcl_AppendResult(interp, "\n no target volume correctly specified",NULL);
    return TCL_ERROR;
  }
  volmapA->scale_by((float)amt); 
  volmol->force_recalc(DrawMolItem::MOL_REGEN);
  
  if (outputmap != NULL) {
    volmap_write_dx_file(volmapA, outputmap);
  }
  return TCL_OK;

}

int density_smooth(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  if (argc < 3) {
    Tcl_SetResult(interp, (char *) "usage: voltool "
      "smooth -sigma <x> radius of guassian blur in x sigmas. [options]\n"
      "    options:  -i <input map> specifies new density filename to load.\n"
      "              -mol <molid> specifies an already loaded density's molid for use as target\n"
      "              -vol <volume id> specifies an already loaded density's volume id for use as target. Defaults to 0.\n"
      "              -o <filename> write density to file.\n",
      TCL_STATIC);
    return TCL_ERROR;
  }


  int molid = -1;
  int volid = 0;
  double sigma = 0;
  const char *input_map = NULL;
  const char *outputmap = NULL;
  MoleculeList *mlist = app->moleculeList;
  
  for (int i=0; i < argc; i++) {
    char *opt = Tcl_GetStringFromObj(objv[i], NULL);
    if (!strcmp(opt, "-i")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No input map specified",NULL);
        return TCL_ERROR;
      }

      FileSpec spec;
      spec.waitfor = FileSpec::WAIT_BACK; // shouldn't this be waiting for all?
      input_map = Tcl_GetStringFromObj(objv[1+i], NULL);
      molid = app->molecule_new(input_map,0,1);
      int ret_val = app->molecule_load(molid, input_map,app->guess_filetype(input_map),&spec);
      if (ret_val < 0) return TCL_ERROR;
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
    
    if (!strcmp(opt, "-sigma")) {
      if (i == argc-1){
        Tcl_AppendResult(interp, "No scaling amount specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetDoubleFromObj(interp, objv[i+1], &sigma) != TCL_OK) {
        Tcl_AppendResult(interp, "\n scaling amount incorrectly specified",NULL);
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
  }
  Molecule *volmol = NULL;
  VolumetricData *volmapA = NULL;
  if (molid > -1) {
    volmol = mlist->mol_from_id(molid);
    if (volmol == NULL) {
      Tcl_AppendResult(interp, "\n invalid molecule specified",NULL);
      return TCL_ERROR;
    }

    if (volmapA == NULL) 
      volmapA = volmol->modify_volume_data(volid);
  } else {
    Tcl_AppendResult(interp, "\n no target volume specified",NULL);
    return TCL_ERROR;
  }
  if (volmapA == NULL) {
    Tcl_AppendResult(interp, "\n no target volume correctly specified",NULL);
    return TCL_ERROR;
  }
  volmapA->gaussian_blur(sigma);
  volmol->force_recalc(DrawMolItem::MOL_REGEN);
  
  if (outputmap != NULL) {
    volmap_write_dx_file(volmapA, outputmap);
  }
  return TCL_OK;

}

int density_sadd(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  if (argc < 3) {
    Tcl_SetResult(interp, (char *) "usage: voltool "
      "sadd -amt <x> add x to every voxel. [options]\n"
      "    options:  -i <input map> specifies new density filename to load.\n"
      "              -mol <molid> specifies an already loaded density's molid for use as target\n"
      "              -vol <volume id> specifies an already loaded density's volume id for use as target. Defaults to 0.\n"
      "              -o <filename> write density to file.\n",
      TCL_STATIC);
    return TCL_ERROR;
  }


  int molid = -1;
  int volid = 0;
  double amt = 0;
  const char *input_map = NULL;
  const char *outputmap = NULL;
  MoleculeList *mlist = app->moleculeList;
  
  for (int i=0; i < argc; i++) {
    char *opt = Tcl_GetStringFromObj(objv[i], NULL);
    if (!strcmp(opt, "-i")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No input map specified",NULL);
        return TCL_ERROR;
      }

      FileSpec spec;
      spec.waitfor = FileSpec::WAIT_BACK; // shouldn't this be waiting for all?
      input_map = Tcl_GetStringFromObj(objv[1+i], NULL);
      molid = app->molecule_new(input_map,0,1);
      int ret_val = app->molecule_load(molid, input_map,app->guess_filetype(input_map),&spec);
      if (ret_val < 0) return TCL_ERROR;
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
    
    if (!strcmp(opt, "-amt")) {
      if (i == argc-1){
        Tcl_AppendResult(interp, "No scaling amount specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetDoubleFromObj(interp, objv[i+1], &amt) != TCL_OK) {
        Tcl_AppendResult(interp, "\n scaling amount incorrectly specified",NULL);
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
  }
  Molecule *volmol = NULL;
  VolumetricData *volmapA = NULL;
  if (molid > -1) {
    volmol = mlist->mol_from_id(molid);
    if (volmol == NULL) {
      Tcl_AppendResult(interp, "\n invalid molecule specified",NULL);
      return TCL_ERROR;
    }

    if (volmapA == NULL) 
      volmapA = volmol->modify_volume_data(volid);
  } else {
    Tcl_AppendResult(interp, "\n no target volume specified",NULL);
    return TCL_ERROR;
  }
  if (volmapA == NULL) {
    Tcl_AppendResult(interp, "\n no target volume correctly specified",NULL);
    return TCL_ERROR;
  }
  volmapA->scalar_add((float)amt); 
  volmol->force_recalc(DrawMolItem::MOL_REGEN);
  
  if (outputmap != NULL) {
    volmap_write_dx_file(volmapA, outputmap);
  }
  return TCL_OK;

}

int density_range(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  if (argc < 3) {
    Tcl_SetResult(interp, (char *) "usage: voltool "
      "range -minmax {min max} minmax voxel values> [options]\n"
      "    options:  -i <input map> specifies new density filename to load.\n"
      "              -mol <molid> specifies an already loaded density's molid for use as target\n"
      "              -vol <volume id> specifies an already loaded density's volume id for use as target. Defaults to 0.\n"
      "              -o <filename> write density to file.\n",
      TCL_STATIC);
    return TCL_ERROR;
  }


  int molid = -1;
  int volid = 0;
  int minmax[2] = {0, 0};
  const char *input_map = NULL;
  const char *outputmap = NULL;
  MoleculeList *mlist = app->moleculeList;
  
  for (int i=0; i < argc; i++) {
    char *opt = Tcl_GetStringFromObj(objv[i], NULL);
    if (!strcmp(opt, "-i")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No input map specified",NULL);
        return TCL_ERROR;
      }

      FileSpec spec;
      spec.waitfor = FileSpec::WAIT_BACK; // shouldn't this be waiting for all?
      input_map = Tcl_GetStringFromObj(objv[1+i], NULL);
      molid = app->molecule_new(input_map,0,1);
      int ret_val = app->molecule_load(molid, input_map,app->guess_filetype(input_map),&spec);
      if (ret_val < 0) return TCL_ERROR;
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
    
    int num1;
    Tcl_Obj **vector;
    if (!strcmp(opt, "-minmax")) {
      if (i == argc-1){
        Tcl_AppendResult(interp, "No trim amounts specified",NULL);
        return TCL_ERROR;
      } else if (Tcl_ListObjGetElements(interp, objv[i+1], &num1, &vector) != TCL_OK) {
      return TCL_ERROR;
      }
    
      for (int i=0; i<num1; i++) {
        if (Tcl_GetIntFromObj(interp, vector[i], &minmax[i]) != TCL_OK) {
          Tcl_SetResult(interp, (char *) "minmax: non-numeric in vector", TCL_STATIC);
          return TCL_ERROR;
        }
      }
      if (num1 != 2) {
        Tcl_SetResult(interp, (char *) "minmax: incorrect number of values in vector", TCL_STATIC);
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
  }
  Molecule *volmol = NULL;
  VolumetricData *volmapA = NULL;
  if (molid > -1) {
    volmol = mlist->mol_from_id(molid);
    if (volmol == NULL) {
      Tcl_AppendResult(interp, "\n invalid molecule specified",NULL);
      return TCL_ERROR;
    }

    if (volmapA == NULL) 
      volmapA = volmol->modify_volume_data(volid);
  } else {
    Tcl_AppendResult(interp, "\n no target volume specified",NULL);
    return TCL_ERROR;
  }
  if (volmapA == NULL) {
    Tcl_AppendResult(interp, "\n no target volume correctly specified",NULL);
    return TCL_ERROR;
  }
  
  volmapA->rescale_voxel_value_range(minmax[0], minmax[1]); 
  volmol->force_recalc(DrawMolItem::MOL_REGEN);
  
  if (outputmap != NULL) {
    volmap_write_dx_file(volmapA, outputmap);
  }
  return TCL_OK;

}

int density_downsample(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  if (argc < 3) {
    Tcl_SetResult(interp, (char *) "usage: voltool "
      "downsample [options]\n"
      "    options:  -i <input map> specifies new density filename to load.\n"
      "              -mol <molid> specifies an already loaded density's molid for use as target\n"
      "              -vol <volume id> specifies an already loaded density's volume id for use as target. Defaults to 0.\n"
      "              -o <filename> write density to file.\n",
      TCL_STATIC);
    return TCL_ERROR;
  }


  int molid = -1;
  int volid = 0;
  const char *input_map = NULL;
  const char *outputmap = NULL;
  MoleculeList *mlist = app->moleculeList;
  
  for (int i=0; i < argc; i++) {
    char *opt = Tcl_GetStringFromObj(objv[i], NULL);
    if (!strcmp(opt, "-i")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No input map specified",NULL);
        return TCL_ERROR;
      }

      FileSpec spec;
      spec.waitfor = FileSpec::WAIT_BACK; // shouldn't this be waiting for all?
      input_map = Tcl_GetStringFromObj(objv[1+i], NULL);
      molid = app->molecule_new(input_map,0,1);
      int ret_val = app->molecule_load(molid, input_map,app->guess_filetype(input_map),&spec);
      if (ret_val < 0) return TCL_ERROR;
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
    
    if (!strcmp(opt, "-o")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No output file specified",NULL);
        return TCL_ERROR;
      } else {
        outputmap = Tcl_GetStringFromObj(objv[i+1], NULL);
      }
    }
  }
  Molecule *volmol = NULL;
  VolumetricData *volmapA = NULL;
  if (molid > -1) {
    volmol = mlist->mol_from_id(molid);
    if (volmol == NULL) {
      Tcl_AppendResult(interp, "\n invalid molecule specified",NULL);
      return TCL_ERROR;
    }

    if (volmapA == NULL) 
      volmapA = volmol->modify_volume_data(volid);
  } else {
    Tcl_AppendResult(interp, "\n no target volume specified",NULL);
    return TCL_ERROR;
  }
  if (volmapA == NULL) {
    Tcl_AppendResult(interp, "\n no target volume correctly specified",NULL);
    return TCL_ERROR;
  }

  volmapA->downsample(); 
  volmol->force_recalc(DrawMolItem::MOL_REGEN);
  
  if (outputmap != NULL) {
    volmap_write_dx_file(volmapA, outputmap);
  }
  return TCL_OK;

}

int density_supersample(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  if (argc < 3) {
    Tcl_SetResult(interp, (char *) "usage: voltool "
      "downsample [options]\n"
      "    options:  -i <input map> specifies new density filename to load.\n"
      "              -mol <molid> specifies an already loaded density's molid for use as target\n"
      "              -vol <volume id> specifies an already loaded density's volume id for use as target. Defaults to 0.\n"
      "              -o <filename> write density to file.\n",
      TCL_STATIC);
    return TCL_ERROR;
  }


  int molid = -1;
  int volid = 0;
  const char *input_map = NULL;
  const char *outputmap = NULL;
  MoleculeList *mlist = app->moleculeList;
  
  for (int i=0; i < argc; i++) {
    char *opt = Tcl_GetStringFromObj(objv[i], NULL);
    if (!strcmp(opt, "-i")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No input map specified",NULL);
        return TCL_ERROR;
      }

      FileSpec spec;
      spec.waitfor = FileSpec::WAIT_BACK; // shouldn't this be waiting for all?
      input_map = Tcl_GetStringFromObj(objv[1+i], NULL);
      molid = app->molecule_new(input_map,0,1);
      int ret_val = app->molecule_load(molid, input_map,app->guess_filetype(input_map),&spec);
      if (ret_val < 0) return TCL_ERROR;
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
    
    if (!strcmp(opt, "-o")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No output file specified",NULL);
        return TCL_ERROR;
      } else {
        outputmap = Tcl_GetStringFromObj(objv[i+1], NULL);
      }
    }
  }
  Molecule *volmol = NULL;
  VolumetricData *volmapA = NULL;
  if (molid > -1) {
    volmol = mlist->mol_from_id(molid);
    if (volmol == NULL) {
      Tcl_AppendResult(interp, "\n invalid molecule specified",NULL);
      return TCL_ERROR;
    }

    if (volmapA == NULL) 
      volmapA = volmol->modify_volume_data(volid);
  } else {
    Tcl_AppendResult(interp, "\n no target volume specified",NULL);
    return TCL_ERROR;
  }
  if (volmapA == NULL) {
    Tcl_AppendResult(interp, "\n no target volume correctly specified",NULL);
    return TCL_ERROR;
  }

  volmapA->supersample(); 
  volmol->force_recalc(DrawMolItem::MOL_REGEN);
  
  if (outputmap != NULL) {
    volmap_write_dx_file(volmapA, outputmap);
  }
  return TCL_OK;

}

int density_sigma(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  if (argc < 3) {
    Tcl_SetResult(interp, (char *) "usage: voltool "
      "sigma [options]\n"
      "    options:  -i <input map> specifies new density filename to load.\n"
      "              -mol <molid> specifies an already loaded density's molid for use as target\n"
      "              -vol <volume id> specifies an already loaded density's volume id for use as target. Defaults to 0.\n"
      "              -o <filename> write density to file.\n",
      TCL_STATIC);
    return TCL_ERROR;
  }


  int molid = -1;
  int volid = 0;
  const char *input_map = NULL;
  const char *outputmap = NULL;
  MoleculeList *mlist = app->moleculeList;
  
  for (int i=0; i < argc; i++) {
    char *opt = Tcl_GetStringFromObj(objv[i], NULL);
    if (!strcmp(opt, "-i")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No input map specified",NULL);
        return TCL_ERROR;
      }

      FileSpec spec;
      spec.waitfor = FileSpec::WAIT_BACK; // shouldn't this be waiting for all?
      input_map = Tcl_GetStringFromObj(objv[1+i], NULL);
      molid = app->molecule_new(input_map,0,1);
      int ret_val = app->molecule_load(molid, input_map,app->guess_filetype(input_map),&spec);
      if (ret_val < 0) return TCL_ERROR;
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
    
    if (!strcmp(opt, "-o")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No output file specified",NULL);
        return TCL_ERROR;
      } else {
        outputmap = Tcl_GetStringFromObj(objv[i+1], NULL);
      }
    }
  }
  Molecule *volmol = NULL;
  VolumetricData *volmapA = NULL;
  if (molid > -1) {
    volmol = mlist->mol_from_id(molid);
    if (volmol == NULL) {
      Tcl_AppendResult(interp, "\n invalid molecule specified",NULL);
      return TCL_ERROR;
    }

    if (volmapA == NULL) 
      volmapA = volmol->modify_volume_data(volid);
  } else {
    Tcl_AppendResult(interp, "\n no target volume specified",NULL);
    return TCL_ERROR;
  }
  if (volmapA == NULL) {
    Tcl_AppendResult(interp, "\n no target volume correctly specified",NULL);
    return TCL_ERROR;
  }

  volmapA->sigma_scale(); 
  volmol->force_recalc(DrawMolItem::MOL_REGEN);
  
  if (outputmap != NULL) {
    volmap_write_dx_file(volmapA, outputmap);
  }
  return TCL_OK;

}

int density_mdff_potential(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  if (argc < 3) {
    Tcl_SetResult(interp, (char *) "usage: voltool "
      "pot [options]\n"
      "    options:  -threshold <x> clamp density to minimum value of x.\n"
      "              -i <input map> specifies new density filename to load.\n"
      "              -mol <molid> specifies an already loaded density's molid for use as target\n"
      "              -vol <volume id> specifies an already loaded density's volume id for use as target. Defaults to 0.\n"
      "              -o <filename> write potential to file.\n",
      TCL_STATIC);
    return TCL_ERROR;
  }


  int molid = -1;
  int volid = 0;
  double threshold = 0;
  const char *input_map = NULL;
  const char *outputmap = NULL;
  MoleculeList *mlist = app->moleculeList;
  
  for (int i=0; i < argc; i++) {
    char *opt = Tcl_GetStringFromObj(objv[i], NULL);
    if (!strcmp(opt, "-i")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No input map specified",NULL);
        return TCL_ERROR;
      }

      FileSpec spec;
      spec.waitfor = FileSpec::WAIT_BACK; // shouldn't this be waiting for all?
      input_map = Tcl_GetStringFromObj(objv[1+i], NULL);
      molid = app->molecule_new(input_map,0,1);
      int ret_val = app->molecule_load(molid, input_map,app->guess_filetype(input_map),&spec);
      if (ret_val < 0) return TCL_ERROR;
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
    
    if (!strcmp(opt, "-threshold")) {
      if (i == argc-1){
        Tcl_AppendResult(interp, "No scaling amount specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetDoubleFromObj(interp, objv[i+1], &threshold) != TCL_OK) {
        Tcl_AppendResult(interp, "\n scaling amount incorrectly specified",NULL);
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
  }
  Molecule *volmol = NULL;
  VolumetricData *volmapA = NULL;
  if (molid > -1) {
    volmol = mlist->mol_from_id(molid);
    if (volmol == NULL) {
      Tcl_AppendResult(interp, "\n invalid molecule specified",NULL);
      return TCL_ERROR;
    }

    if (volmapA == NULL) 
      volmapA = volmol->modify_volume_data(volid);
  } else {
    Tcl_AppendResult(interp, "\n no target volume specified",NULL);
    return TCL_ERROR;
  }
  if (volmapA == NULL) {
    Tcl_AppendResult(interp, "\n no target volume correctly specified",NULL);
    return TCL_ERROR;
  }
  volmapA->mdff_potential(threshold); 
  volmol->force_recalc(DrawMolItem::MOL_REGEN);
  
  if (outputmap != NULL) {
    volmap_write_dx_file(volmapA, outputmap);
  }
  return TCL_OK;

}

int density_histogram(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  if (argc < 3) {
    Tcl_SetResult(interp, (char *) "usage: voltool "
      "hist [options]\n"
      "    options:  -nbins <x> number of histogram bins. Defaults to 10.\n"
      "              -i <input map> specifies new density filename to load.\n"
      "              -mol <molid> specifies an already loaded density's molid for use as target\n"
      "              -vol <volume id> specifies an already loaded density's volume id for use as target. Defaults to 0.\n",
      TCL_STATIC);
    return TCL_ERROR;
  }


  int molid = -1;
  int volid = 0;
  int nbins = 10;
  const char *input_map = NULL;
  MoleculeList *mlist = app->moleculeList;
  
  for (int i=0; i < argc; i++) {
    char *opt = Tcl_GetStringFromObj(objv[i], NULL);
    if (!strcmp(opt, "-i")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No input map specified",NULL);
        return TCL_ERROR;
      }

      FileSpec spec;
      spec.waitfor = FileSpec::WAIT_BACK; // shouldn't this be waiting for all?
      input_map = Tcl_GetStringFromObj(objv[1+i], NULL);
      molid = app->molecule_new(input_map,0,1);
      int ret_val = app->molecule_load(molid, input_map,app->guess_filetype(input_map),&spec);
      if (ret_val < 0) return TCL_ERROR;
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
    
    if (!strcmp(opt, "-nbins")) {
      if (i == argc-1){
        Tcl_AppendResult(interp, "No number of bins specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetIntFromObj(interp, objv[i+1], &nbins) != TCL_OK) {
        Tcl_AppendResult(interp, "\n number of bins incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }
    
  }
  Molecule *volmol = NULL;
  VolumetricData *volmapA = NULL;
  if (molid > -1) {
    volmol = mlist->mol_from_id(molid);
    if (volmol == NULL) {
      Tcl_AppendResult(interp, "\n invalid molecule specified",NULL);
      return TCL_ERROR;
    }

    if (volmapA == NULL) 
      volmapA = volmol->modify_volume_data(volid);
  } else {
    Tcl_AppendResult(interp, "\n no target volume specified",NULL);
    return TCL_ERROR;
  }
  if (volmapA == NULL) {
    Tcl_AppendResult(interp, "\n no target volume correctly specified",NULL);
    return TCL_ERROR;
  }
  
  int *bins = new int[nbins]; 
  float *midpts = new float[nbins]; 
  histogram(volmapA, nbins, bins, midpts); 
  
  // convert the results of the lowlevel call to tcl lists
  // and build a list from them as return value.
  Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
  for (int j=0; j < nbins; j++) {
      Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewDoubleObj(midpts[j]));
      Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewIntObj(bins[j]));
  }
  Tcl_SetObjResult(interp, tcl_result);
  delete[] bins;
  delete[] midpts;
  return TCL_OK;

}

int density_info(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  if (argc < 3) {
    Tcl_SetResult(interp, (char *) "usage: voltool "
      "info [origin | xsize | ysize | zsize | minmax ] [options]\n"
      "    options:  -i <input map> specifies new density filename to load.\n"
      "              -mol <molid> specifies an already loaded density's molid for use as target\n"
      "              -vol <volume id> specifies an already loaded density's volume id for use as target. Defaults to 0.\n",
      TCL_STATIC);
    return TCL_ERROR;
  }


  int molid = -1;
  int volid = 0;
  enum INFO_MODE {
    NONE,
    origin,
    xsize,
    ysize,
    zsize,
    minmax
  };
  INFO_MODE mode = NONE;

  const char *input_map = NULL;
  MoleculeList *mlist = app->moleculeList;
  
  for (int i=0; i < argc; i++) {
    char *opt = Tcl_GetStringFromObj(objv[i], NULL);
    if (!strcmp(opt, "-i")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No input map specified",NULL);
        return TCL_ERROR;
      }

      FileSpec spec;
      spec.waitfor = FileSpec::WAIT_BACK; // shouldn't this be waiting for all?
      input_map = Tcl_GetStringFromObj(objv[1+i], NULL);
      molid = app->molecule_new(input_map,0,1);
      int ret_val = app->molecule_load(molid, input_map,app->guess_filetype(input_map),&spec);
      if (ret_val < 0) return TCL_ERROR;
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
    
    if (!strcmp(opt, "origin")) {
      mode = origin;
    }
    if (!strcmp(opt, "xsize")) {
      mode = xsize;
    }
    if (!strcmp(opt, "ysize")) {
      mode = ysize;
    }
    if (!strcmp(opt, "zsize")) {
      mode = zsize;
    }
    if (!strcmp(opt, "minmax")) {
      mode = minmax;
    }
    
  }
  Molecule *volmol = NULL;
  VolumetricData *volmapA = NULL;
  if (molid > -1) {
    volmol = mlist->mol_from_id(molid);
    if (volmol == NULL) {
      Tcl_AppendResult(interp, "\n invalid molecule specified",NULL);
      return TCL_ERROR;
    }

    if (volmapA == NULL) 
      volmapA = volmol->modify_volume_data(volid);
  } else {
    Tcl_AppendResult(interp, "\n no target volume specified",NULL);
    return TCL_ERROR;
  }
  if (volmapA == NULL) {
    Tcl_AppendResult(interp, "\n no target volume correctly specified",NULL);
    return TCL_ERROR;
  }
  
  switch(mode)
  {
    case origin: { 
      // convert the results of the lowlevel call to tcl lists
      // and build a list from them as return value.
      Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
      for (int j=0; j < 3; j++) {
          Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewDoubleObj(volmapA->origin[j]));
      }
      Tcl_SetObjResult(interp, tcl_result);
      break;
    }
    case xsize: {
      Tcl_SetObjResult(interp, Tcl_NewIntObj(volmapA->xsize));
      break;
    }
    case ysize: {
      Tcl_SetObjResult(interp, Tcl_NewIntObj(volmapA->ysize));
      break;
     }
    case zsize: {
      Tcl_SetObjResult(interp, Tcl_NewIntObj(volmapA->zsize));
      break;
    }
    case minmax: {
      float min, max;
      volmapA->datarange(min, max);
      Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
      Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewDoubleObj(min));
      Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewDoubleObj(max));
      Tcl_SetObjResult(interp, tcl_result);
      break;
    }
    case NONE: {
      Tcl_AppendResult(interp, "No mode correctly specified",NULL);
      return TCL_ERROR;
    }
  }
  return TCL_OK;      
}

int density_binmask(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  if (argc < 3) {
    Tcl_SetResult(interp, (char *) "usage: voltool "
      "binmask [options]\n"
      "    options:  -threshold <thresold value> set values above threshold to 1. Defaults to 0.\n"
      "              -i <input map> specifies new density filename to load.\n"
      "              -mol <molid> specifies an already loaded density's molid for use as target\n"
      "              -vol <volume id> specifies an already loaded density's volume id for use as target. Defaults to 0.\n"
      "              -o <filename> write density to file.\n",
      TCL_STATIC);
    return TCL_ERROR;
  }

  
  int molid = -1;
  int volid = 0;
  double threshold = 0.0;
  const char *input_map = NULL;
  const char *outputmap = NULL;
  MoleculeList *mlist = app->moleculeList;
  
  for (int i=0; i < argc; i++) {
    char *opt = Tcl_GetStringFromObj(objv[i], NULL);
    if (!strcmp(opt, "-i")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No input map specified",NULL);
        return TCL_ERROR;
      }

      FileSpec spec;
      spec.waitfor = FileSpec::WAIT_BACK; // shouldn't this be waiting for all?
      input_map = Tcl_GetStringFromObj(objv[1+i], NULL);
      molid = app->molecule_new(input_map,0,1);
      int ret_val = app->molecule_load(molid, input_map,app->guess_filetype(input_map),&spec);
      if (ret_val < 0) return TCL_ERROR;
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
    
    if (!strcmp(opt, "-o")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No output file specified",NULL);
        return TCL_ERROR;
      } else {
        outputmap = Tcl_GetStringFromObj(objv[i+1], NULL);
      }
    }
    
    if (!strcmp(opt, "-threshold")) {
      if (i == argc-1){
        Tcl_AppendResult(interp, "No threshold specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetDoubleFromObj(interp, objv[i+1], &threshold) != TCL_OK) {
        Tcl_AppendResult(interp, "\n threshold incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }
  }
  
  
  Molecule *volmol = NULL;
  VolumetricData *volmapA = NULL;
  if (molid > -1) {
    volmol = mlist->mol_from_id(molid);
    if (volmol == NULL) {
      Tcl_AppendResult(interp, "\n invalid molecule specified",NULL);
      return TCL_ERROR;
    }

    if (volmapA == NULL) 
      volmapA = volmol->modify_volume_data(volid);
  } else {
    Tcl_AppendResult(interp, "\n no target volume specified",NULL);
    return TCL_ERROR;
  }
  if (volmapA == NULL) {
    Tcl_AppendResult(interp, "\n no target volume correctly specified",NULL);
    return TCL_ERROR;
  }

  volmapA->binmask(threshold); 
  volmol->force_recalc(DrawMolItem::MOL_REGEN);
  
  if (outputmap != NULL) {
    volmap_write_dx_file(volmapA, outputmap);
  }
  return TCL_OK;

}

int density_add(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  if (argc < 3) {
    Tcl_SetResult(interp, (char *) "usage: voltool "
      "add [options]\n"
      "    input options:  -i1 <input map> specifies new density filename to load.\n"
      "              -mol1 <molid> specifies an already loaded density's molid for use as target\n"
      "              -vol1 <volume id> specifies an already loaded density's volume id for use as target. Defaults to 0.\n"
      "              -i2 <input map> specifies new density filename to load.\n"
      "              -mol2 <molid> specifies an already loaded density's molid for use as target\n"
      "              -vol2 <volume id> specifies an already loaded density's volume id for use as target. Defaults to 0.\n"
      "    options: \n"
      "              -union use union of input maps for operation\n"
      "              -nointerp do not use interpolation for operation\n"
      "              -o <filename> write density to file.\n",
      TCL_STATIC);
    return TCL_ERROR;
  }


  int molid1 = -1;
  int volid1 = 0;
  const char *input_map1 = NULL;
  int molid2 = -1;
  int volid2 = 0;
  const char *input_map2 = NULL;
  
  bool USE_UNION = false;
  bool USE_INTERP = true; 
  const char *outputmap = NULL;
  MoleculeList *mlist = app->moleculeList;
  
  for (int i=0; i < argc; i++) {
    char *opt = Tcl_GetStringFromObj(objv[i], NULL);
    if (!strcmp(opt, "-i1")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No input map specified",NULL);
        return TCL_ERROR;
      }

      FileSpec spec;
      spec.waitfor = FileSpec::WAIT_BACK; // shouldn't this be waiting for all?
      input_map1 = Tcl_GetStringFromObj(objv[1+i], NULL);
      molid1 = app->molecule_new(input_map1,0,1);
      int ret_val = app->molecule_load(molid1, input_map1,app->guess_filetype(input_map1),&spec);
      if (ret_val < 0) return TCL_ERROR;
    }


    if (!strcmp(opt, "-mol1")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No molid specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetIntFromObj(interp, objv[i+1], &molid1) != TCL_OK) {
        Tcl_AppendResult(interp, "\n molid incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }

    if (!strcmp(opt, "-vol1")) {
      if (i == argc-1){
        Tcl_AppendResult(interp, "No volume id specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetIntFromObj(interp, objv[i+1], &volid1) != TCL_OK) {
        Tcl_AppendResult(interp, "\n volume id incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }
    
    if (!strcmp(opt, "-i2")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No input map specified",NULL);
        return TCL_ERROR;
      }

      FileSpec spec;
      spec.waitfor = FileSpec::WAIT_BACK; // shouldn't this be waiting for all?
      input_map2 = Tcl_GetStringFromObj(objv[1+i], NULL);
      molid2 = app->molecule_new(input_map2,0,1);
      int ret_val = app->molecule_load(molid2, input_map2,app->guess_filetype(input_map2),&spec);
      if (ret_val < 0) return TCL_ERROR;
    }


    if (!strcmp(opt, "-mol2")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No molid specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetIntFromObj(interp, objv[i+1], &molid2) != TCL_OK) {
        Tcl_AppendResult(interp, "\n molid incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }

    if (!strcmp(opt, "-vol2")) {
      if (i == argc-1){
        Tcl_AppendResult(interp, "No volume id specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetIntFromObj(interp, objv[i+1], &volid2) != TCL_OK) {
        Tcl_AppendResult(interp, "\n volume id incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }
    
    if (!strcmp(opt, "-union")) {
      USE_UNION = true;
    }
    
    if (!strcmp(opt, "-nointerp")) {
      USE_INTERP = false;
    }
    
    if (!strcmp(opt, "-o")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No output file specified",NULL);
        return TCL_ERROR;
      } else {
        outputmap = Tcl_GetStringFromObj(objv[i+1], NULL);
      }
    }
  }
  
  Molecule *volmol1 = NULL;
  VolumetricData *volmapA = NULL;
  if (molid1 > -1) {
    volmol1 = mlist->mol_from_id(molid1);
    if (volmol1 == NULL) {
      Tcl_AppendResult(interp, "\n invalid molecule specified",NULL);
      return TCL_ERROR;
    }

    if (volmapA == NULL) 
      volmapA = volmol1->modify_volume_data(volid1);
  } else {
    Tcl_AppendResult(interp, "\n no target volume specified",NULL);
    return TCL_ERROR;
  }
  if (volmapA == NULL) {
    Tcl_AppendResult(interp, "\n no target volume correctly specified",NULL);
    return TCL_ERROR;
  }
 
  Molecule *volmol2 = NULL;
  VolumetricData *volmapB = NULL;
  if (molid2 > -1) {
    volmol2 = mlist->mol_from_id(molid2);
    if (volmol2 == NULL) {
      Tcl_AppendResult(interp, "\n invalid molecule specified",NULL);
      return TCL_ERROR;
    }

    if (volmapB == NULL) 
      volmapB = volmol2->modify_volume_data(volid2);
  } else {
    Tcl_AppendResult(interp, "\n no target volume specified",NULL);
    return TCL_ERROR;
  }
  if (volmapB == NULL) {
    Tcl_AppendResult(interp, "\n no target volume correctly specified",NULL);
    return TCL_ERROR;
  }
  
  VolumetricData *newvol = init_new_volume();
  add(volmapA, volmapB, newvol, USE_INTERP, USE_UNION);
  init_new_volume_molecule(app, newvol, "add_map");

  if (outputmap != NULL) {
    volmap_write_dx_file(newvol, outputmap);
  }
  return TCL_OK;

}

int density_subtract(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  if (argc < 3) {
    Tcl_SetResult(interp, (char *) "usage: voltool "
      "diff [options]\n"
      "    input options:  -i1 <input map> specifies new density filename to load.\n"
      "              -mol1 <molid> specifies an already loaded density's molid for use as target\n"
      "              -vol1 <volume id> specifies an already loaded density's volume id for use as target. Defaults to 0.\n"
      "              -i2 <input map> specifies new density filename to load.\n"
      "              -mol2 <molid> specifies an already loaded density's molid for use as target\n"
      "              -vol2 <volume id> specifies an already loaded density's volume id for use as target. Defaults to 0.\n"
      "    options: \n"
      "              -union use union of input maps for operation\n"
      "              -nointerp do not use interpolation for operation\n"
      "              -o <filename> write density to file.\n",
      TCL_STATIC);
    return TCL_ERROR;
  }


  int molid1 = -1;
  int volid1 = 0;
  const char *input_map1 = NULL;
  int molid2 = -1;
  int volid2 = 0;
  const char *input_map2 = NULL;
  
  bool USE_UNION = false;
  bool USE_INTERP = true; 
  const char *outputmap = NULL;
  MoleculeList *mlist = app->moleculeList;
  
  for (int i=0; i < argc; i++) {
    char *opt = Tcl_GetStringFromObj(objv[i], NULL);
    if (!strcmp(opt, "-i1")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No input map specified",NULL);
        return TCL_ERROR;
      }

      FileSpec spec;
      spec.waitfor = FileSpec::WAIT_BACK; // shouldn't this be waiting for all?
      input_map1 = Tcl_GetStringFromObj(objv[1+i], NULL);
      molid1 = app->molecule_new(input_map1,0,1);
      int ret_val = app->molecule_load(molid1, input_map1,app->guess_filetype(input_map1),&spec);
      if (ret_val < 0) return TCL_ERROR;
    }


    if (!strcmp(opt, "-mol1")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No molid specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetIntFromObj(interp, objv[i+1], &molid1) != TCL_OK) {
        Tcl_AppendResult(interp, "\n molid incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }

    if (!strcmp(opt, "-vol1")) {
      if (i == argc-1){
        Tcl_AppendResult(interp, "No volume id specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetIntFromObj(interp, objv[i+1], &volid1) != TCL_OK) {
        Tcl_AppendResult(interp, "\n volume id incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }
    
    if (!strcmp(opt, "-i2")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No input map specified",NULL);
        return TCL_ERROR;
      }

      FileSpec spec;
      spec.waitfor = FileSpec::WAIT_BACK; // shouldn't this be waiting for all?
      input_map2 = Tcl_GetStringFromObj(objv[1+i], NULL);
      molid2 = app->molecule_new(input_map2,0,1);
      int ret_val = app->molecule_load(molid2, input_map2,app->guess_filetype(input_map2),&spec);
      if (ret_val < 0) return TCL_ERROR;
    }


    if (!strcmp(opt, "-mol2")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No molid specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetIntFromObj(interp, objv[i+1], &molid2) != TCL_OK) {
        Tcl_AppendResult(interp, "\n molid incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }

    if (!strcmp(opt, "-vol2")) {
      if (i == argc-1){
        Tcl_AppendResult(interp, "No volume id specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetIntFromObj(interp, objv[i+1], &volid2) != TCL_OK) {
        Tcl_AppendResult(interp, "\n volume id incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }
    
    if (!strcmp(opt, "-union")) {
      USE_UNION = true;
    }
    
    if (!strcmp(opt, "-nointerp")) {
      USE_INTERP = false;
    }
    
    if (!strcmp(opt, "-o")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No output file specified",NULL);
        return TCL_ERROR;
      } else {
        outputmap = Tcl_GetStringFromObj(objv[i+1], NULL);
      }
    }
  }
  
  Molecule *volmol1 = NULL;
  VolumetricData *volmapA = NULL;
  if (molid1 > -1) {
    volmol1 = mlist->mol_from_id(molid1);
    if (volmol1 == NULL) {
      Tcl_AppendResult(interp, "\n invalid molecule specified",NULL);
      return TCL_ERROR;
    }

    if (volmapA == NULL) 
      volmapA = volmol1->modify_volume_data(volid1);
  } else {
    Tcl_AppendResult(interp, "\n no target volume specified",NULL);
    return TCL_ERROR;
  }
  if (volmapA == NULL) {
    Tcl_AppendResult(interp, "\n no target volume correctly specified",NULL);
    return TCL_ERROR;
  }
 
  Molecule *volmol2 = NULL;
  VolumetricData *volmapB = NULL;
  if (molid2 > -1) {
    volmol2 = mlist->mol_from_id(molid2);
    if (volmol2 == NULL) {
      Tcl_AppendResult(interp, "\n invalid molecule specified",NULL);
      return TCL_ERROR;
    }

    if (volmapB == NULL) 
      volmapB = volmol2->modify_volume_data(volid2);
  } else {
    Tcl_AppendResult(interp, "\n no target volume specified",NULL);
    return TCL_ERROR;
  }
  if (volmapB == NULL) {
    Tcl_AppendResult(interp, "\n no target volume correctly specified",NULL);
    return TCL_ERROR;
  }
  
  VolumetricData *newvol = init_new_volume();
  subtract(volmapA, volmapB, newvol, USE_INTERP, USE_UNION);
  init_new_volume_molecule(app, newvol, "diff_map");

  if (outputmap != NULL) {
    volmap_write_dx_file(newvol, outputmap);
  }
  return TCL_OK;

}

int density_multiply(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  if (argc < 3) {
    Tcl_SetResult(interp, (char *) "usage: voltool "
      "mult [options]\n"
      "    input options:  -i1 <input map> specifies new density filename to load.\n"
      "              -mol1 <molid> specifies an already loaded density's molid for use as target\n"
      "              -vol1 <volume id> specifies an already loaded density's volume id for use as target. Defaults to 0.\n"
      "              -i2 <input map> specifies new density filename to load.\n"
      "              -mol2 <molid> specifies an already loaded density's molid for use as target\n"
      "              -vol2 <volume id> specifies an already loaded density's volume id for use as target. Defaults to 0.\n"
      "    options: \n"
      "              -union use union of input maps for operation\n"
      "              -nointerp do not use interpolation for operation\n"
      "              -o <filename> write density to file.\n",
      TCL_STATIC);
    return TCL_ERROR;
  }


  int molid1 = -1;
  int volid1 = 0;
  const char *input_map1 = NULL;
  int molid2 = -1;
  int volid2 = 0;
  const char *input_map2 = NULL;
  
  bool USE_UNION = false;
  bool USE_INTERP = true; 
  const char *outputmap = NULL;
  MoleculeList *mlist = app->moleculeList;
  
  for (int i=0; i < argc; i++) {
    char *opt = Tcl_GetStringFromObj(objv[i], NULL);
    if (!strcmp(opt, "-i1")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No input map specified",NULL);
        return TCL_ERROR;
      }

      FileSpec spec;
      spec.waitfor = FileSpec::WAIT_BACK; // shouldn't this be waiting for all?
      input_map1 = Tcl_GetStringFromObj(objv[1+i], NULL);
      molid1 = app->molecule_new(input_map1,0,1);
      int ret_val = app->molecule_load(molid1, input_map1,app->guess_filetype(input_map1),&spec);
      if (ret_val < 0) return TCL_ERROR;
    }


    if (!strcmp(opt, "-mol1")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No molid specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetIntFromObj(interp, objv[i+1], &molid1) != TCL_OK) {
        Tcl_AppendResult(interp, "\n molid incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }

    if (!strcmp(opt, "-vol1")) {
      if (i == argc-1){
        Tcl_AppendResult(interp, "No volume id specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetIntFromObj(interp, objv[i+1], &volid1) != TCL_OK) {
        Tcl_AppendResult(interp, "\n volume id incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }
    
    if (!strcmp(opt, "-i2")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No input map specified",NULL);
        return TCL_ERROR;
      }

      FileSpec spec;
      spec.waitfor = FileSpec::WAIT_BACK; // shouldn't this be waiting for all?
      input_map2 = Tcl_GetStringFromObj(objv[1+i], NULL);
      molid2 = app->molecule_new(input_map2,0,1);
      int ret_val = app->molecule_load(molid2, input_map2,app->guess_filetype(input_map2),&spec);
      if (ret_val < 0) return TCL_ERROR;
    }


    if (!strcmp(opt, "-mol2")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No molid specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetIntFromObj(interp, objv[i+1], &molid2) != TCL_OK) {
        Tcl_AppendResult(interp, "\n molid incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }

    if (!strcmp(opt, "-vol2")) {
      if (i == argc-1){
        Tcl_AppendResult(interp, "No volume id specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetIntFromObj(interp, objv[i+1], &volid2) != TCL_OK) {
        Tcl_AppendResult(interp, "\n volume id incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }
    
    if (!strcmp(opt, "-union")) {
      USE_UNION = true;
    }
    
    if (!strcmp(opt, "-nointerp")) {
      USE_INTERP = false;
    }
    
    if (!strcmp(opt, "-o")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No output file specified",NULL);
        return TCL_ERROR;
      } else {
        outputmap = Tcl_GetStringFromObj(objv[i+1], NULL);
      }
    }
  }
  
  Molecule *volmol1 = NULL;
  VolumetricData *volmapA = NULL;
  if (molid1 > -1) {
    volmol1 = mlist->mol_from_id(molid1);
    if (volmol1 == NULL) {
      Tcl_AppendResult(interp, "\n invalid molecule specified",NULL);
      return TCL_ERROR;
    }

    if (volmapA == NULL) 
      volmapA = volmol1->modify_volume_data(volid1);
  } else {
    Tcl_AppendResult(interp, "\n no target volume specified",NULL);
    return TCL_ERROR;
  }
  if (volmapA == NULL) {
    Tcl_AppendResult(interp, "\n no target volume correctly specified",NULL);
    return TCL_ERROR;
  }
 
  Molecule *volmol2 = NULL;
  VolumetricData *volmapB = NULL;
  if (molid2 > -1) {
    volmol2 = mlist->mol_from_id(molid2);
    if (volmol2 == NULL) {
      Tcl_AppendResult(interp, "\n invalid molecule specified",NULL);
      return TCL_ERROR;
    }

    if (volmapB == NULL) 
      volmapB = volmol2->modify_volume_data(volid2);
  } else {
    Tcl_AppendResult(interp, "\n no target volume specified",NULL);
    return TCL_ERROR;
  }
  if (volmapB == NULL) {
    Tcl_AppendResult(interp, "\n no target volume correctly specified",NULL);
    return TCL_ERROR;
  }
  
  VolumetricData *newvol = init_new_volume();
  multiply(volmapA, volmapB, newvol, USE_INTERP, USE_UNION);
  init_new_volume_molecule(app, newvol, "mult_map");

  if (outputmap != NULL) {
    volmap_write_dx_file(newvol, outputmap);
  }
  return TCL_OK;

}

int density_average(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  if (argc < 3) {
    Tcl_SetResult(interp, (char *) "usage: voltool "
      "average [options]\n"
      "    input options:  -i1 <input map> specifies new density filename to load.\n"
      "              -mol1 <molid> specifies an already loaded density's molid for use as target\n"
      "              -vol1 <volume id> specifies an already loaded density's volume id for use as target. Defaults to 0.\n"
      "              -i2 <input map> specifies new density filename to load.\n"
      "              -mol2 <molid> specifies an already loaded density's molid for use as target\n"
      "              -vol2 <volume id> specifies an already loaded density's volume id for use as target. Defaults to 0.\n"
      "    options: \n"
      "              -union use union of input maps for operation\n"
      "              -nointerp do not use interpolation for operation\n"
      "              -o <filename> write density to file.\n",
      TCL_STATIC);
    return TCL_ERROR;
  }


  int molid1 = -1;
  int volid1 = 0;
  const char *input_map1 = NULL;
  int molid2 = -1;
  int volid2 = 0;
  const char *input_map2 = NULL;
  
  bool USE_UNION = false;
  bool USE_INTERP = true; 
  const char *outputmap = NULL;
  MoleculeList *mlist = app->moleculeList;
  
  for (int i=0; i < argc; i++) {
    char *opt = Tcl_GetStringFromObj(objv[i], NULL);
    if (!strcmp(opt, "-i1")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No input map specified",NULL);
        return TCL_ERROR;
      }

      FileSpec spec;
      spec.waitfor = FileSpec::WAIT_BACK; // shouldn't this be waiting for all?
      input_map1 = Tcl_GetStringFromObj(objv[1+i], NULL);
      molid1 = app->molecule_new(input_map1,0,1);
      int ret_val = app->molecule_load(molid1, input_map1,app->guess_filetype(input_map1),&spec);
      if (ret_val < 0) return TCL_ERROR;
    }


    if (!strcmp(opt, "-mol1")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No molid specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetIntFromObj(interp, objv[i+1], &molid1) != TCL_OK) {
        Tcl_AppendResult(interp, "\n molid incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }

    if (!strcmp(opt, "-vol1")) {
      if (i == argc-1){
        Tcl_AppendResult(interp, "No volume id specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetIntFromObj(interp, objv[i+1], &volid1) != TCL_OK) {
        Tcl_AppendResult(interp, "\n volume id incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }
    
    if (!strcmp(opt, "-i2")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No input map specified",NULL);
        return TCL_ERROR;
      }

      FileSpec spec;
      spec.waitfor = FileSpec::WAIT_BACK; // shouldn't this be waiting for all?
      input_map2 = Tcl_GetStringFromObj(objv[1+i], NULL);
      molid2 = app->molecule_new(input_map2,0,1);
      int ret_val = app->molecule_load(molid2, input_map2,app->guess_filetype(input_map2),&spec);
      if (ret_val < 0) return TCL_ERROR;
    }


    if (!strcmp(opt, "-mol2")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No molid specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetIntFromObj(interp, objv[i+1], &molid2) != TCL_OK) {
        Tcl_AppendResult(interp, "\n molid incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }

    if (!strcmp(opt, "-vol2")) {
      if (i == argc-1){
        Tcl_AppendResult(interp, "No volume id specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetIntFromObj(interp, objv[i+1], &volid2) != TCL_OK) {
        Tcl_AppendResult(interp, "\n volume id incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }
    
    if (!strcmp(opt, "-union")) {
      USE_UNION = true;
    }
    
    if (!strcmp(opt, "-nointerp")) {
      USE_INTERP = false;
    }
    
    if (!strcmp(opt, "-o")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No output file specified",NULL);
        return TCL_ERROR;
      } else {
        outputmap = Tcl_GetStringFromObj(objv[i+1], NULL);
      }
    }
  }
  
  Molecule *volmol1 = NULL;
  VolumetricData *volmapA = NULL;
  if (molid1 > -1) {
    volmol1 = mlist->mol_from_id(molid1);
    if (volmol1 == NULL) {
      Tcl_AppendResult(interp, "\n invalid molecule specified",NULL);
      return TCL_ERROR;
    }

    if (volmapA == NULL) 
      volmapA = volmol1->modify_volume_data(volid1);
  } else {
    Tcl_AppendResult(interp, "\n no target volume specified",NULL);
    return TCL_ERROR;
  }
  if (volmapA == NULL) {
    Tcl_AppendResult(interp, "\n no target volume correctly specified",NULL);
    return TCL_ERROR;
  }
 
  Molecule *volmol2 = NULL;
  VolumetricData *volmapB = NULL;
  if (molid2 > -1) {
    volmol2 = mlist->mol_from_id(molid2);
    if (volmol2 == NULL) {
      Tcl_AppendResult(interp, "\n invalid molecule specified",NULL);
      return TCL_ERROR;
    }

    if (volmapB == NULL) 
      volmapB = volmol2->modify_volume_data(volid2);
  } else {
    Tcl_AppendResult(interp, "\n no target volume specified",NULL);
    return TCL_ERROR;
  }
  if (volmapB == NULL) {
    Tcl_AppendResult(interp, "\n no target volume correctly specified",NULL);
    return TCL_ERROR;
  }
  
  VolumetricData *newvol = init_new_volume();
  average(volmapA, volmapB, newvol, USE_INTERP, USE_UNION);
  init_new_volume_molecule(app, newvol, "avg_map");
  if (outputmap != NULL) {
    volmap_write_dx_file(newvol, outputmap);
  }
  return TCL_OK;

}

int density_correlate(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  if (argc < 3) {
    Tcl_SetResult(interp, (char *) "usage: voltool "
      "correlate [options] \n"
      "    input options:  -i1 <input map> specifies new density filename to load.\n"
      "              -mol1 <molid> specifies an already loaded density's molid for use as target\n"
      "              -vol1 <volume id> specifies an already loaded density's volume id for use as target. Defaults to 0.\n"
      "              -i2 <input map> specifies new density filename to load.\n"
      "              -mol2 <molid> specifies an already loaded density's molid for use as target\n"
      "              -vol2 <volume id> specifies an already loaded density's volume id for use as target. Defaults to 0.\n"
      "    options: \n"
      "              -thresholddensity <x> (ignores voxels with values below x threshold)\n",
      TCL_STATIC);
    return TCL_ERROR;
  }


  int molid1 = -1;
  int volid1 = 0;
  const char *input_map1 = NULL;
  int molid2 = -1;
  int volid2 = 0;
  const char *input_map2 = NULL;
  double thresholddensity = -FLT_MAX;
  
  MoleculeList *mlist = app->moleculeList;
  
  for (int i=0; i < argc; i++) {
    char *opt = Tcl_GetStringFromObj(objv[i], NULL);
    if (!strcmp(opt, "-i1")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No input map specified",NULL);
        return TCL_ERROR;
      }

      FileSpec spec;
      spec.waitfor = FileSpec::WAIT_BACK; // shouldn't this be waiting for all?
      input_map1 = Tcl_GetStringFromObj(objv[1+i], NULL);
      molid1 = app->molecule_new(input_map1,0,1);
      int ret_val = app->molecule_load(molid1, input_map1,app->guess_filetype(input_map1),&spec);
      if (ret_val < 0) return TCL_ERROR;
    }


    if (!strcmp(opt, "-mol1")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No molid specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetIntFromObj(interp, objv[i+1], &molid1) != TCL_OK) {
        Tcl_AppendResult(interp, "\n molid incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }

    if (!strcmp(opt, "-vol1")) {
      if (i == argc-1){
        Tcl_AppendResult(interp, "No volume id specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetIntFromObj(interp, objv[i+1], &volid1) != TCL_OK) {
        Tcl_AppendResult(interp, "\n volume id incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }
    
    if (!strcmp(opt, "-i2")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No input map specified",NULL);
        return TCL_ERROR;
      }

      FileSpec spec;
      spec.waitfor = FileSpec::WAIT_BACK; // shouldn't this be waiting for all?
      input_map2 = Tcl_GetStringFromObj(objv[1+i], NULL);
      molid2 = app->molecule_new(input_map2,0,1);
      int ret_val = app->molecule_load(molid2, input_map2,app->guess_filetype(input_map2),&spec);
      if (ret_val < 0) return TCL_ERROR;
    }


    if (!strcmp(opt, "-mol2")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No molid specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetIntFromObj(interp, objv[i+1], &molid2) != TCL_OK) {
        Tcl_AppendResult(interp, "\n molid incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }

    if (!strcmp(opt, "-vol2")) {
      if (i == argc-1){
        Tcl_AppendResult(interp, "No volume id specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetIntFromObj(interp, objv[i+1], &volid2) != TCL_OK) {
        Tcl_AppendResult(interp, "\n volume id incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }
    
    if (!strcmp(opt, "-thresholddensity")) {
      if (i == argc-1){
        Tcl_AppendResult(interp, "No threshold specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetDoubleFromObj(interp, objv[i+1], &thresholddensity) != TCL_OK) {
        Tcl_AppendResult(interp, "\n threshold incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }
    
  }
  
  Molecule *volmol1 = NULL;
  VolumetricData *volmapA = NULL;
  if (molid1 > -1) {
    volmol1 = mlist->mol_from_id(molid1);
    if (volmol1 == NULL) {
      Tcl_AppendResult(interp, "\n invalid molecule specified",NULL);
      return TCL_ERROR;
    }

    if (volmapA == NULL) 
      volmapA = volmol1->modify_volume_data(volid1);
  } else {
    Tcl_AppendResult(interp, "\n no target volume specified",NULL);
    return TCL_ERROR;
  }
  if (volmapA == NULL) {
    Tcl_AppendResult(interp, "\n no target volume correctly specified",NULL);
    return TCL_ERROR;
  }
 
  Molecule *volmol2 = NULL;
  VolumetricData *volmapB = NULL;
  if (molid2 > -1) {
    volmol2 = mlist->mol_from_id(molid2);
    if (volmol2 == NULL) {
      Tcl_AppendResult(interp, "\n invalid molecule specified",NULL);
      return TCL_ERROR;
    }

    if (volmapB == NULL) 
      volmapB = volmol2->modify_volume_data(volid2);
  } else {
    Tcl_AppendResult(interp, "\n no target volume specified",NULL);
    return TCL_ERROR;
  }
  if (volmapB == NULL) {
    Tcl_AppendResult(interp, "\n no target volume correctly specified",NULL);
    return TCL_ERROR;
  }
  
  double return_cc = 0.0;
  cc_threaded(volmapA, volmapB, &return_cc, thresholddensity);

  Tcl_SetObjResult(interp, Tcl_NewDoubleObj(return_cc));
  return TCL_OK;
}

int obj_voltool(ClientData cd, Tcl_Interp *interp, int argc,
                            Tcl_Obj * const objv[]){
  if (argc < 2) {
    Tcl_SetResult(interp,
    (char *) "Usage: voltool <command> [args...]\n"
      "Commands:\n"
      "map operations using an atomic structure:\n"
      "fit          -- rigid body fitting\n"
      "cc           -- calculates the cross-correlation coefficient between a map and structure\n"
      "sim          -- creates a simulated map from an atomic structure\n"
      "mask         -- masks a map around an atomic structure\n"
      "operations on one map:\n"
      "com          -- get center of mass of density\n"
      "moveto       -- move density com to a specified coordinate\n"
      "move         -- apply specified 4x4 transformation matrix to density\n"
      "trim         -- trim edges of a density\n"
      "crop         -- crop density to values given in coordinate space\n"
      "clamp        -- clamp out of range voxel values\n"
      "smult        -- multiply every voxel by a scaling factor\n"
      "sadd         -- add a scaling factor to every voxel\n"
      "range        -- fit voxel values to a given range\n"
      "downsample   -- downsample by x2 (x8 total reduction)\n"
      "supersample  -- supersample by x2 (x8 total increase)\n"
      "sigma        -- transform map to sigma scale\n"
      "binmask      -- make a binary mask of the map\n"
      "smooth       -- 3D gaussian blur\n"
      "pot          -- convert a density map to an MDFF potential\n"  
      "hist         -- calculate a histogram of the density map\n"  
      "info         -- get information about the density map\n"  
      "operations on two maps:\n"
      "add          -- add two maps together\n"
      "diff         -- subtract map2 from map1\n"
      "mult         -- multiply map1 and map2\n"
      "avg          -- average two input maps into one\n"
      "correlate    -- calculates the cross-correlation coefficient between two maps\n"
      ,
      TCL_STATIC);
    return TCL_ERROR;
  }
  char *argv1 = Tcl_GetStringFromObj(objv[1],NULL);

  VMDApp *app = (VMDApp *)cd;
  if (!strupncmp(argv1, "fit", CMDLEN))
    return fit(app, argc-1, objv+1, interp);
  if (!strupncmp(argv1, "sim", CMDLEN))
    return mdff_sim(app, argc-1, objv+1, interp);
  if (!strupncmp(argv1, "cc", CMDLEN))
    return mdff_cc(app, argc-1, objv+1, interp);
  if (!strupncmp(argv1, "com", CMDLEN))
    return density_com(app, argc-1, objv+1, interp);
  if (!strupncmp(argv1, "moveto", CMDLEN))
    return density_moveto(app, argc-1, objv+1, interp);
  if (!strupncmp(argv1, "move", CMDLEN))
    return density_move(app, argc-1, objv+1, interp);
  if (!strupncmp(argv1, "trim", CMDLEN))
    return density_trim(app, argc-1, objv+1, interp);
  if (!strupncmp(argv1, "crop", CMDLEN))
    return density_crop(app, argc-1, objv+1, interp);
  if (!strupncmp(argv1, "clamp", CMDLEN))
    return density_clamp(app, argc-1, objv+1, interp);
  if (!strupncmp(argv1, "smult", CMDLEN))
    return density_smult(app, argc-1, objv+1, interp);
  if (!strupncmp(argv1, "sadd", CMDLEN))
    return density_sadd(app, argc-1, objv+1, interp);
  if (!strupncmp(argv1, "range", CMDLEN))
    return density_range(app, argc-1, objv+1, interp);
  if (!strupncmp(argv1, "downsample", CMDLEN))
    return density_downsample(app, argc-1, objv+1, interp);
  if (!strupncmp(argv1, "supersample", CMDLEN))
    return density_supersample(app, argc-1, objv+1, interp);
  if (!strupncmp(argv1, "sigma", CMDLEN))
    return density_sigma(app, argc-1, objv+1, interp);
  if (!strupncmp(argv1, "binmask", CMDLEN))
    return density_binmask(app, argc-1, objv+1, interp);
  if (!strupncmp(argv1, "smooth", CMDLEN))
    return density_smooth(app, argc-1, objv+1, interp);
  if (!strupncmp(argv1, "add", CMDLEN))
    return density_add(app, argc-1, objv+1, interp);
  if (!strupncmp(argv1, "diff", CMDLEN))
    return density_subtract(app, argc-1, objv+1, interp);
  if (!strupncmp(argv1, "mult", CMDLEN))
    return density_multiply(app, argc-1, objv+1, interp);
  if (!strupncmp(argv1, "avg", CMDLEN))
    return density_average(app, argc-1, objv+1, interp);
  if (!strupncmp(argv1, "correlate", CMDLEN))
    return density_correlate(app, argc-1, objv+1, interp);
  if (!strupncmp(argv1, "pot", CMDLEN))
    return density_mdff_potential(app, argc-1, objv+1, interp);
  if (!strupncmp(argv1, "hist", CMDLEN))
    return density_histogram(app, argc-1, objv+1, interp);
  if (!strupncmp(argv1, "info", CMDLEN))
    return density_info(app, argc-1, objv+1, interp);
  if (!strupncmp(argv1, "mask", CMDLEN))
    return mask(app, argc-1, objv+1, interp);

  Tcl_SetResult(interp, (char *) "Type 'voltool' for summary of usage\n", TCL_VOLATILE);
  return TCL_OK;
}
