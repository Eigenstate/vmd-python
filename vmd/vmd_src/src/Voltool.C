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
 *      $RCSfile: Voltool.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.6 $      $Date: 2019/01/17 21:38:55 $
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
#include <math.h>
#include <tcl.h>
#include "TclCommands.h"
#include "Measure.h"
#include "MolFilePlugin.h"

#include <iostream>
#include <string>
#include <sstream>
#include "Voltool.h"

/// creates axes, bounding box and allocates data based on 
/// geometrical intersection of A and B
void init_from_intersection(VolumetricData *mapA, const VolumetricData *mapB, VolumetricData *newvol) {
  int d;
  
  // Find intersection of A and B
  // The following has been verified for orthog. cells
  // (Does not work for non-orthog cells)
  
  for (d=0; d<3; d++) {
    newvol->origin[d] = MAX(mapA->origin[d], mapB->origin[d]);
    newvol->xaxis[d] = MAX(MIN(mapA->origin[d]+mapA->xaxis[d], mapB->origin[d]+mapB->xaxis[d]), newvol->origin[d]);
    newvol->yaxis[d] = MAX(MIN(mapA->origin[d]+mapA->yaxis[d], mapB->origin[d]+mapB->yaxis[d]), newvol->origin[d]);
    newvol->zaxis[d] = MAX(MIN(mapA->origin[d]+mapA->zaxis[d], mapB->origin[d]+mapB->zaxis[d]), newvol->origin[d]);
  }
    
  vec_sub(newvol->xaxis, newvol->xaxis, newvol->origin);
  vec_sub(newvol->yaxis, newvol->yaxis, newvol->origin);
  vec_sub(newvol->zaxis, newvol->zaxis, newvol->origin);
  
  newvol->xsize = (int) MAX(dot_prod(newvol->xaxis,mapA->xaxis)*mapA->xsize/dot_prod(mapA->xaxis,mapA->xaxis), \
                    dot_prod(newvol->xaxis,mapB->xaxis)*mapB->xsize/dot_prod(mapB->xaxis,mapB->xaxis));
  newvol->ysize = (int) MAX(dot_prod(newvol->yaxis,mapA->yaxis)*mapA->ysize/dot_prod(mapA->yaxis,mapA->yaxis), \
                    dot_prod(newvol->yaxis,mapB->yaxis)*mapB->ysize/dot_prod(mapB->yaxis,mapB->yaxis));
  newvol->zsize = (int) MAX(dot_prod(newvol->zaxis,mapA->zaxis)*mapA->zsize/dot_prod(mapA->zaxis,mapA->zaxis), \
                    dot_prod(newvol->zaxis,mapB->zaxis)*mapB->zsize/dot_prod(mapB->zaxis,mapB->zaxis));
  // Create map...
  if (newvol->data) delete[] newvol->data;
  newvol->data = new float[newvol->gridsize()];
}

/// creates axes, bounding box and allocates data based on 
/// geometrical intersection of A and B
void init_from_intersection(VolumetricData *mapA, VolumetricData *mapB, VolumetricData *newvol) {
  int d;
  
  // Find intersection of A and B
  // The following has been verified for orthog. cells
  // (Does not work for non-orthog cells)
  
  for (d=0; d<3; d++) {
    newvol->origin[d] = MAX(mapA->origin[d], mapB->origin[d]);
    newvol->xaxis[d] = MAX(MIN(mapA->origin[d]+mapA->xaxis[d], mapB->origin[d]+mapB->xaxis[d]), newvol->origin[d]);
    newvol->yaxis[d] = MAX(MIN(mapA->origin[d]+mapA->yaxis[d], mapB->origin[d]+mapB->yaxis[d]), newvol->origin[d]);
    newvol->zaxis[d] = MAX(MIN(mapA->origin[d]+mapA->zaxis[d], mapB->origin[d]+mapB->zaxis[d]), newvol->origin[d]);
  }
    
  vec_sub(newvol->xaxis, newvol->xaxis, newvol->origin);
  vec_sub(newvol->yaxis, newvol->yaxis, newvol->origin);
  vec_sub(newvol->zaxis, newvol->zaxis, newvol->origin);
  
  newvol->xsize = (int) MAX(dot_prod(newvol->xaxis,mapA->xaxis)*mapA->xsize/dot_prod(mapA->xaxis,mapA->xaxis), \
                    dot_prod(newvol->xaxis,mapB->xaxis)*mapB->xsize/dot_prod(mapB->xaxis,mapB->xaxis));
  newvol->ysize = (int) MAX(dot_prod(newvol->yaxis,mapA->yaxis)*mapA->ysize/dot_prod(mapA->yaxis,mapA->yaxis), \
                    dot_prod(newvol->yaxis,mapB->yaxis)*mapB->ysize/dot_prod(mapB->yaxis,mapB->yaxis));
  newvol->zsize = (int) MAX(dot_prod(newvol->zaxis,mapA->zaxis)*mapA->zsize/dot_prod(mapA->zaxis,mapA->zaxis), \
                    dot_prod(newvol->zaxis,mapB->zaxis)*mapB->zsize/dot_prod(mapB->zaxis,mapB->zaxis));
  // Create map...
  if (newvol->data) delete[] newvol->data;
  newvol->data = new float[newvol->gridsize()];
}


/// creates axes, bounding box and allocates data based on 
/// geometrical union of A and B
void init_from_union(VolumetricData *mapA, const VolumetricData *mapB, VolumetricData *newvol) {
  // Find union of A and B
  // The following has been verified for orthog. cells
  // (Does not work for non-orthog cells)
  
  vec_zero(newvol->xaxis);
  vec_zero(newvol->yaxis);
  vec_zero(newvol->zaxis);
  
  int d;
  for (d=0; d<3; d++) {
    newvol->origin[d] = MIN(mapA->origin[d], mapB->origin[d]);
  }
  d=0;
  newvol->xaxis[d] = MAX(MAX(mapA->origin[d]+mapA->xaxis[d], mapB->origin[d]+mapB->xaxis[d]), newvol->origin[d]);
  d=1;
  newvol->yaxis[d] = MAX(MAX(mapA->origin[d]+mapA->yaxis[d], mapB->origin[d]+mapB->yaxis[d]), newvol->origin[d]);
  d=2;
  newvol->zaxis[d] = MAX(MAX(mapA->origin[d]+mapA->zaxis[d], mapB->origin[d]+mapB->zaxis[d]), newvol->origin[d]);
  
  newvol->xaxis[0] -= newvol->origin[0];
  newvol->yaxis[1] -= newvol->origin[1];
  newvol->zaxis[2] -= newvol->origin[2];
  
  newvol->xsize = (int) MAX(dot_prod(newvol->xaxis,mapA->xaxis)*mapA->xsize/dot_prod(mapA->xaxis,mapA->xaxis), \
                    dot_prod(newvol->xaxis,mapB->xaxis)*mapB->xsize/dot_prod(mapB->xaxis,mapB->xaxis));
  newvol->ysize = (int) MAX(dot_prod(newvol->yaxis,mapA->yaxis)*mapA->ysize/dot_prod(mapA->yaxis,mapA->yaxis), \
                    dot_prod(newvol->yaxis,mapB->yaxis)*mapB->ysize/dot_prod(mapB->yaxis,mapB->yaxis));
  newvol->zsize = (int) MAX(dot_prod(newvol->zaxis,mapA->zaxis)*mapA->zsize/dot_prod(mapA->zaxis,mapA->zaxis), \
                    dot_prod(newvol->zaxis,mapB->zaxis)*mapB->zsize/dot_prod(mapB->zaxis,mapB->zaxis));
  // Create map...
  if (newvol->data) delete[] newvol->data;
  newvol->data = new float[newvol->gridsize()];
}

void init_from_identity(VolumetricData *mapA, VolumetricData *newvol) {

  vec_copy(newvol->origin, mapA->origin);
  vec_copy(newvol->xaxis, mapA->xaxis);
  vec_copy(newvol->yaxis, mapA->yaxis);
  vec_copy(newvol->zaxis, mapA->zaxis); 
  
  newvol->xsize = mapA->xsize;
  newvol->ysize = mapA->ysize;
  newvol->zsize = mapA->zsize;
  // Create map...
  if (newvol->data) delete[] newvol->data;
  newvol->data = new float[newvol->gridsize()];
}


/// creates axes, bounding box and allocates data based on 
/// geometrical union of A and B
void init_from_union(VolumetricData *mapA, VolumetricData *mapB, VolumetricData *newvol) {
  // Find union of A and B
  // The following has been verified for orthog. cells
  // (Does not work for non-orthog cells)
  
  vec_zero(newvol->xaxis);
  vec_zero(newvol->yaxis);
  vec_zero(newvol->zaxis);
  
  int d;
  for (d=0; d<3; d++) {
    newvol->origin[d] = MIN(mapA->origin[d], mapB->origin[d]);
  }
  d=0;
  newvol->xaxis[d] = MAX(MAX(mapA->origin[d]+mapA->xaxis[d], mapB->origin[d]+mapB->xaxis[d]), newvol->origin[d]);
  d=1;
  newvol->yaxis[d] = MAX(MAX(mapA->origin[d]+mapA->yaxis[d], mapB->origin[d]+mapB->yaxis[d]), newvol->origin[d]);
  d=2;
  newvol->zaxis[d] = MAX(MAX(mapA->origin[d]+mapA->zaxis[d], mapB->origin[d]+mapB->zaxis[d]), newvol->origin[d]);
  
  newvol->xaxis[0] -= newvol->origin[0];
  newvol->yaxis[1] -= newvol->origin[1];
  newvol->zaxis[2] -= newvol->origin[2];
  
  newvol->xsize = (int) MAX(dot_prod(newvol->xaxis,mapA->xaxis)*mapA->xsize/dot_prod(mapA->xaxis,mapA->xaxis), \
                    dot_prod(newvol->xaxis,mapB->xaxis)*mapB->xsize/dot_prod(mapB->xaxis,mapB->xaxis));
  newvol->ysize = (int) MAX(dot_prod(newvol->yaxis,mapA->yaxis)*mapA->ysize/dot_prod(mapA->yaxis,mapA->yaxis), \
                    dot_prod(newvol->yaxis,mapB->yaxis)*mapB->ysize/dot_prod(mapB->yaxis,mapB->yaxis));
  newvol->zsize = (int) MAX(dot_prod(newvol->zaxis,mapA->zaxis)*mapA->zsize/dot_prod(mapA->zaxis,mapA->zaxis), \
                    dot_prod(newvol->zaxis,mapB->zaxis)*mapB->zsize/dot_prod(mapB->zaxis,mapB->zaxis));
  // Create map...
  if (newvol->data) delete[] newvol->data;
  newvol->data = new float[newvol->gridsize()];
}

VolumetricData * init_new_volume(){
  double origin[3] = {0., 0., 0.};
  double xaxis[3] = {0., 0., 0.};
  double yaxis[3] = {0., 0., 0.};
  double zaxis[3] = {0., 0., 0.};
  int numvoxels [3] = {0, 0, 0};
  float *data = NULL;
  VolumetricData *newvol  = new VolumetricData("density map", origin, xaxis, yaxis, zaxis,
                                 numvoxels[0], numvoxels[1], numvoxels[2],
                                 data);
  return newvol; 
}


void init_new_volume_molecule(VMDApp *app, VolumetricData *newvol, const char *name){
  int newvolmolid = app->molecule_new(name,0,1);
  
  app->molecule_add_volumetric(newvolmolid, "density newvol", newvol->origin, 
                              newvol->xaxis, newvol->yaxis, newvol->zaxis, newvol->xsize, newvol->ysize, 
                              newvol->zsize, newvol->data);
  app->molecule_set_style("Isosurface");
  app->molecule_addrep(newvolmolid);

}


void vol_com(VolumetricData *vol, float *com){
  float ix,iy,iz;
  
  vec_zero(com);
  double mass = 0.0;

  for (int i = 0; i < vol->gridsize(); i++) {
    float m = vol->data[i];
    vol->voxel_coord(i, ix, iy, iz);
    float coord[3] = {ix,iy,iz};    
    mass = mass+m;
    vec_scale(coord, m, coord);
    vec_add(com, com, coord); 
     
  }
  
  float scale = 1.0/mass;
  vec_scale(com, scale, com);
}

void add(VolumetricData *mapA, VolumetricData  *mapB, VolumetricData *newvol, bool interp, bool USE_UNION) {

  // adding maps by spatial coords is slower than doing it directly, but
  // allows for precisely subtracting unaligned maps, and/or maps of
  // different resolutions

  if ( USE_UNION) {
    // UNION VERSION
    init_from_union(mapA, mapB, newvol);
  } else {
    // INTERSECTION VERSION
    init_from_intersection(mapA, mapB, newvol);
  }
  for (long i=0; i<newvol->gridsize(); i++){
    float x, y, z;
    newvol->voxel_coord(i, x, y, z);

    if (interp) newvol->data[i] = \
      mapA->voxel_value_interpolate_from_coord_safe(x,y,z) + \
      mapB->voxel_value_interpolate_from_coord_safe(x,y,z);
    else newvol->data[i] = \
      mapA->voxel_value_from_coord_safe(x,y,z) + \
      mapB->voxel_value_from_coord_safe(x,y,z);
  } 

}

void subtract(VolumetricData *mapA, VolumetricData  *mapB, VolumetricData *newvol, bool interp, bool USE_UNION) {

  // adding maps by spatial coords is slower than doing it directly, but
  // allows for precisely subtracting unaligned maps, and/or maps of
  // different resolutions

  if ( USE_UNION) {
    // UNION VERSION
    init_from_union(mapA, mapB, newvol);
  } else {
    // INTERSECTION VERSION
    init_from_intersection(mapA, mapB, newvol);
  }
  for (long i=0; i<newvol->gridsize(); i++){
    float x, y, z;
    newvol->voxel_coord(i, x, y, z);

    if (interp) newvol->data[i] = \
      mapA->voxel_value_interpolate_from_coord_safe(x,y,z) - \
      mapB->voxel_value_interpolate_from_coord_safe(x,y,z);
    else newvol->data[i] = \
      mapA->voxel_value_from_coord_safe(x,y,z) - \
      mapB->voxel_value_from_coord_safe(x,y,z);
  } 

}

void multiply(VolumetricData *mapA, VolumetricData  *mapB, VolumetricData *newvol, bool interp, bool USE_UNION) {

  // adding maps by spatial coords is slower than doing it directly, but
  // allows for precisely subtracting unaligned maps, and/or maps of
  // different resolutions

  if ( USE_UNION) {
    // UNION VERSION
    init_from_union(mapA, mapB, newvol);
    for (long i=0; i<newvol->gridsize(); i++){
      float x, y, z;
      float voxelA, voxelB;
      newvol->voxel_coord(i, x, y, z);
      if (interp) {
        voxelA = mapA->voxel_value_interpolate_from_coord(x,y,z);
        voxelB = mapB->voxel_value_interpolate_from_coord(x,y,z);
      } else {
        voxelA = mapA->voxel_value_from_coord(x,y,z);
        voxelB = mapB->voxel_value_from_coord(x,y,z);
      }

     if (!myisnan(voxelA) && !myisnan(voxelB))
       newvol->data[i] = voxelA * voxelB;
     else if (!myisnan(voxelA) && myisnan(voxelB))
       newvol->data[i] = voxelA;
     else if (myisnan(voxelA) && !myisnan(voxelB))
       newvol->data[i] = voxelB;
     else
       newvol->data[i] = 0.;
    }
  } else {
    // INTERSECTION VERSION
    init_from_intersection(mapA, mapB, newvol);
    
    for (long i=0; i<newvol->gridsize(); i++){
      float x, y, z;
      newvol->voxel_coord(i, x, y, z);

      if (interp) newvol->data[i] = \
        mapA->voxel_value_interpolate_from_coord(x,y,z) * \
        mapB->voxel_value_interpolate_from_coord(x,y,z);
      else newvol->data[i] = \
        mapA->voxel_value_from_coord(x,y,z) * \
        mapB->voxel_value_from_coord(x,y,z);
    } 

  }
}

void average(VolumetricData *mapA, VolumetricData  *mapB, VolumetricData *newvol, bool interp, bool USE_UNION) {

  // adding maps by spatial coords is slower than doing it directly, but
  // allows for precisely subtracting unaligned maps, and/or maps of
  // different resolutions

  if ( USE_UNION) {
    // UNION VERSION
    init_from_union(mapA, mapB, newvol);
  } else {
    // INTERSECTION VERSION
    init_from_intersection(mapA, mapB, newvol);
  }
  for (long i=0; i<newvol->gridsize(); i++){
    float x, y, z;
    newvol->voxel_coord(i, x, y, z);

    if (interp) newvol->data[i] = \
      (mapA->voxel_value_interpolate_from_coord_safe(x,y,z) + \
      mapB->voxel_value_interpolate_from_coord_safe(x,y,z))*0.5;
    else newvol->data[i] = \
      (mapA->voxel_value_from_coord_safe(x,y,z) + \
      mapB->voxel_value_from_coord_safe(x,y,z))*0.5;
  } 

}

void vol_moveto(VolumetricData *vol, float *com, float *pos){
  float origin[3] = {0.0, 0.0, 0.0};
  origin[0] = (float)vol->origin[0];
  origin[1] = (float)vol->origin[1];
  origin[2] = (float)vol->origin[2];

  float transvector[3];
  vec_sub(transvector, pos, com);
  vec_add(origin, origin, transvector);   
  vol->origin[0] = origin[0];
  vol->origin[1] = origin[1];
  vol->origin[2] = origin[2];
}
/*
void vectrans(float *npoint, float *mat, double *vec){
  npoint[0]=vec[0]*mat[0]+vec[1]*mat[4]+vec[2]*mat[8];
  npoint[1]=vec[0]*mat[1]+vec[1]*mat[5]+vec[2]*mat[9];
  npoint[2]=vec[0]*mat[2]+vec[1]*mat[6]+vec[2]*mat[10];
}
*/
void vol_move(VolumetricData *vol,  float *mat){
  float origin[3] = {0.0, 0.0, 0.0};
  origin[0] = (float)vol->origin[0];
  origin[1] = (float)vol->origin[1];
  origin[2] = (float)vol->origin[2];
 
  float transvector[3] = {mat[12], mat[13], mat[14]};
 
  float npoint[3];
  
  //deal with origin transformation
  //vectrans  
  npoint[0]=origin[0]*mat[0]+origin[1]*mat[4]+origin[2]*mat[8];
  npoint[1]=origin[0]*mat[1]+origin[1]*mat[5]+origin[2]*mat[9];
  npoint[2]=origin[0]*mat[2]+origin[1]*mat[6]+origin[2]*mat[10];

  vec_add(origin, npoint, transvector);
  vol->origin[0] = origin[0]; 
  vol->origin[1] = origin[1]; 
  vol->origin[2] = origin[2]; 
     
  //deal with delta transformation
  double deltax[3] = {vol->xaxis[0],vol->xaxis[1],vol->xaxis[2]};
  double deltay[3] = {vol->yaxis[0],vol->yaxis[1],vol->yaxis[2]};
  double deltaz[3] = {vol->zaxis[0],vol->zaxis[1],vol->zaxis[2]};
 
  float npointx[3];
  float npointy[3];
  float npointz[3];
  vectrans(npointx, mat, deltax);
  vectrans(npointy, mat, deltay);
  vectrans(npointz, mat, deltaz);
  
  for (int i = 0; i<3; i++){
    vol->xaxis[i] = npointx[i];
    vol->yaxis[i] = npointy[i];
    vol->zaxis[i] = npointz[i];
  }

}

/// Calculate histogram of map. bins and midpts are return
/// arrays for the counts and midpoints of the bins, respectively
/// and must be the size of nbins.
void histogram( VolumetricData *vol, int nbins, int *bins, float *midpts) {
  //get minmax values of map
  float min, max;
  vol->datarange(min, max);
  // Calculate the width of each bin
  double binwidth = (max-min)/nbins;
  //precompute inverse
  double binwidthinv = 1/binwidth;
  // Allocate array that will contain the number of voxels in each bin
  //int *bins = (int*) malloc(nbins*sizeof(int));
  memset(bins, 0, nbins*sizeof(int));

  // Calculate histogram
  for (long i=0; i<vol->gridsize(); i++) 
    bins[int((vol->data[i]-min)*binwidthinv)]++;
  
  for (int j = 0; j < nbins; j++)
      midpts[j] = min + (0.5*binwidth) + (j*binwidth);
}
