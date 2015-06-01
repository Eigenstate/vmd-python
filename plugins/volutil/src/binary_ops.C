/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2009 The Board of Trustees of the
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile: binary_ops.C,v $
 *	$Author: ltrabuco $	$Locker:  $		$State: Exp $
 *	$Revision: 1.4 $	$Date: 2009/08/19 22:57:06 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 *
 ***************************************************************************/

#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#include "volmap.h"
#include "vec.h"

#define VOLUTIL_COMPARE_TOL 1e-6

// From sysexits.h
#ifndef EX_SUCCESS
#define EX_SUCCESS 0
#endif
#ifndef EX_USAGE
#define EX_USAGE 64
#endif

/* BINARY OPERATIONS */

//////////////////////////////////////////////////////////////


/// creates axes, bounding box and allocates data based on 
/// geometrical intersection of A and B
void VolMap::init_from_intersection(VolMap *mapA, VolMap *mapB) {
  int d;
  
  // Find intersection of A and B
  // The following has been verified for orthog. cells
  // (Does not work for non-orthog cells)
  
  for (d=0; d<3; d++) {
    origin[d] = MAX(mapA->origin[d], mapB->origin[d]);
    xaxis[d] = MAX(MIN(mapA->origin[d]+mapA->xaxis[d], mapB->origin[d]+mapB->xaxis[d]), origin[d]);
    yaxis[d] = MAX(MIN(mapA->origin[d]+mapA->yaxis[d], mapB->origin[d]+mapB->yaxis[d]), origin[d]);
    zaxis[d] = MAX(MIN(mapA->origin[d]+mapA->zaxis[d], mapB->origin[d]+mapB->zaxis[d]), origin[d]);
  }
    
  vsub(xaxis, xaxis, origin);
  vsub(yaxis, yaxis, origin);
  vsub(zaxis, zaxis, origin);
  
  xsize = (int) MAX(vdot(xaxis,mapA->xaxis)*mapA->xsize/vdot(mapA->xaxis,mapA->xaxis), \
                    vdot(xaxis,mapB->xaxis)*mapB->xsize/vdot(mapB->xaxis,mapB->xaxis));
  ysize = (int) MAX(vdot(yaxis,mapA->yaxis)*mapA->ysize/vdot(mapA->yaxis,mapA->yaxis), \
                    vdot(yaxis,mapB->yaxis)*mapB->ysize/vdot(mapB->yaxis,mapB->yaxis));
  zsize = (int) MAX(vdot(zaxis,mapA->zaxis)*mapA->zsize/vdot(mapA->zaxis,mapA->zaxis), \
                    vdot(zaxis,mapB->zaxis)*mapB->zsize/vdot(mapB->zaxis,mapB->zaxis));
    
  for (d=0; d<3; d++) {
    xdelta[d] = xaxis[d]/(xsize-1);
    ydelta[d] = yaxis[d]/(ysize-1);
    zdelta[d] = zaxis[d]/(zsize-1);
  }
  
  // Create map...
  if (data) delete[] data;
  data = new float[xsize*ysize*zsize];
}




/// creates axes, bounding box and allocates data based on 
/// geometrical union of A and B
void VolMap::init_from_union(VolMap *mapA, VolMap *mapB) {
  // Find union of A and B
  // The following has been verified for orthog. cells
  // (Does not work for non-orthog cells)
  
  vset(xaxis, 0., 0., 0.);
  vset(yaxis, 0., 0., 0.);
  vset(zaxis, 0., 0., 0.);
  
  int d;
  
  for (d=0; d<3; d++) {
    origin[d] = MIN(mapA->origin[d], mapB->origin[d]);
  }
  d=0;
  xaxis[d] = MAX(MAX(mapA->origin[d]+mapA->xaxis[d], mapB->origin[d]+mapB->xaxis[d]), origin[d]);
  d=1;
  yaxis[d] = MAX(MAX(mapA->origin[d]+mapA->yaxis[d], mapB->origin[d]+mapB->yaxis[d]), origin[d]);
  d=2;
  zaxis[d] = MAX(MAX(mapA->origin[d]+mapA->zaxis[d], mapB->origin[d]+mapB->zaxis[d]), origin[d]);
  
  xaxis[0] -= origin[0];
  yaxis[1] -= origin[1];
  zaxis[2] -= origin[2];
  
  xsize = (int) MAX(vdot(xaxis,mapA->xaxis)*mapA->xsize/vdot(mapA->xaxis,mapA->xaxis), \
                    vdot(xaxis,mapB->xaxis)*mapB->xsize/vdot(mapB->xaxis,mapB->xaxis));
  ysize = (int) MAX(vdot(yaxis,mapA->yaxis)*mapA->ysize/vdot(mapA->yaxis,mapA->yaxis), \
                    vdot(yaxis,mapB->yaxis)*mapB->ysize/vdot(mapB->yaxis,mapB->yaxis));
  zsize = (int) MAX(vdot(zaxis,mapA->zaxis)*mapA->zsize/vdot(mapA->zaxis,mapA->zaxis), \
                    vdot(zaxis,mapB->zaxis)*mapB->zsize/vdot(mapB->zaxis,mapB->zaxis));
  
  for (d=0; d<3; d++) {
    xdelta[d] = xaxis[d]/(xsize-1);
    ydelta[d] = yaxis[d]/(ysize-1);
    zdelta[d] = zaxis[d]/(zsize-1);
  }
  
  // Create map...
  if (data) delete[] data;
  data = new float[xsize*ysize*zsize];
}



void VolMap::init_from_identity(VolMap *mapA) {

  vcopy(origin, mapA->origin);
  vcopy(xaxis, mapA->xaxis);
  vcopy(yaxis, mapA->yaxis);
  vcopy(zaxis, mapA->zaxis); 
  
  xsize = mapA->xsize;
  ysize = mapA->ysize;
  zsize = mapA->zsize;
    
  int d;
  for (d=0; d<3; d++) {
    xdelta[d] = xaxis[d]/(xsize-1);
    ydelta[d] = yaxis[d]/(ysize-1);
    zdelta[d] = zaxis[d]/(zsize-1);
  }
  
  // Create map...
  if (data) delete[] data;
  data = new float[xsize*ysize*zsize];
}



// Recursively perform a binary operation...
void VolMap::perform_recursively(char **files, int numfiles, unsigned int flagsbits, void (VolMap::*func)(VolMap*, VolMap*, unsigned int, Ops), Ops optype) {
 // printf("Performing recursive operation:\n");

  if (numfiles < 2) {
    fprintf(stderr, "ERROR: Need at least 2 files for operation.\n");
    exit(EX_USAGE);
  }
  
  VolMap *potsave = new VolMap(files[0]);
  printf("MAP <- \"%s\"\n", files[0]);
  potsave->set_refname("MAP");

  int i;
  for (i=1; i < numfiles; i++) {
    VolMap *pot = new VolMap(files[i]);
    
    (this->*func)(potsave, pot, flagsbits, optype);
    
    delete pot;
    potsave->clone(this);
  }  

  delete potsave;
}




void VolMap::add(VolMap *mapA, VolMap *mapB, unsigned int flagsbits, Ops optype) {

  int gx, gy, gz;

  // adding maps by spatial coords is slower than doing it directly, but
  // allows for precisely subtracting unaligned maps, and/or maps of
  // different resolutions

  bool interp = false;
  if (flagsbits & USE_INTERP) interp = true;

  if (flagsbits & USE_UNION) {

    // UNION VERSION
      
    if (interp) printf("%s <- add (%s, %s) [using union and interpolation]\n", get_refname(), mapA->get_refname(), mapB->get_refname());
    else printf("%s <- add (%s, %s) [using union]\n", get_refname(), mapA->get_refname(), mapB->get_refname());

    init_from_union(mapA, mapB);

    // use a 'safe' version of voxel_value_interpolate_from_coord
    // that returns zero if coordinate is outside the map

    for (gx=0; gx<xsize; gx++)
      for (gy=0; gy<ysize; gy++)
        for (gz=0; gz<zsize; gz++) {
          float x, y, z;
          voxel_coord(gx, gy, gz, x, y, z);

          if (interp) data[gz*xsize*ysize + gy*xsize + gx] = \
            mapA->voxel_value_interpolate_from_coord_safe(x,y,z) + \
            mapB->voxel_value_interpolate_from_coord_safe(x,y,z);
          else data[gz*xsize*ysize + gy*xsize + gx] = \
            mapA->voxel_value_from_coord_safe(x,y,z) + \
            mapB->voxel_value_from_coord_safe(x,y,z);
        }

  } else {
  
    // INTERSECTION VERSION
    
    if (interp) printf("%s <- add (%s, %s) [using intersection and interpolation]\n", get_refname(), mapA->get_refname(), mapB->get_refname());
    else printf("%s <- add (%s, %s) [using intersection]\n", get_refname(), mapA->get_refname(), mapB->get_refname());
    
    init_from_intersection(mapA, mapB);
  
    for (gx=0; gx<xsize; gx++)
      for (gy=0; gy<ysize; gy++)
        for (gz=0; gz<zsize; gz++) {
          float x, y, z;
          voxel_coord(gx, gy, gz, x, y, z);

          if (interp) data[gz*xsize*ysize + gy*xsize + gx] = \
            mapA->voxel_value_interpolate_from_coord(x,y,z) + \
            mapB->voxel_value_interpolate_from_coord(x,y,z);
          else data[gz*xsize*ysize + gy*xsize + gx] = \
            mapA->voxel_value_from_coord(x,y,z) + \
            mapB->voxel_value_from_coord(x,y,z);
        }

  }

}




void VolMap::multiply(VolMap *mapA, VolMap *mapB, unsigned int flagsbits, Ops optype) {

  int gx, gy, gz;

  // multiplying maps by spatial coords is slower than doing it
  // directly, but allows for precisely subtracting unaligned maps,
  // and/or maps of different resolutions
  //
  bool interp = false;
  if (flagsbits & USE_INTERP) interp = true;
  
  if (flagsbits & USE_UNION) {

    // UNION VERSION
    
    if (interp) printf("%s <- multiply (%s, %s) [using union and interpolation]\n", get_refname(), mapA->get_refname(), mapB->get_refname());
    else printf("%s <- multiply (%s, %s) [using union]\n", get_refname(), mapA->get_refname(), mapB->get_refname());
    
    init_from_union(mapA, mapB);

    // XXX - This is a simple implementation that can be made more 
    //       efficient by decomposing the regions and avoiding the 
    //       inner-loop conditionals.
    for (gx=0; gx<xsize; gx++)
      for (gy=0; gy<ysize; gy++)
        for (gz=0; gz<zsize; gz++) {
          float x, y, z;
          float voxelA, voxelB;
          voxel_coord(gx, gy, gz, x, y, z);

          if (interp) {
            voxelA = mapA->voxel_value_interpolate_from_coord(x,y,z);
            voxelB = mapB->voxel_value_interpolate_from_coord(x,y,z);
          } else {
            voxelA = mapA->voxel_value_from_coord(x,y,z);
            voxelB = mapB->voxel_value_from_coord(x,y,z);
          }

         if (!ISNAN(voxelA) && !ISNAN(voxelB))
           data[gz*xsize*ysize + gy*xsize + gx] = voxelA * voxelB;
         else if (!ISNAN(voxelA) && ISNAN(voxelB))
           data[gz*xsize*ysize + gy*xsize + gx] = voxelA;
         else if (ISNAN(voxelA) && !ISNAN(voxelB))
           data[gz*xsize*ysize + gy*xsize + gx] = voxelB;
         else
           data[gz*xsize*ysize + gy*xsize + gx] = 0.;
        }

  } else {

    // INTERSECTION VERSION
    
    if (interp) printf("%s <- multiply (%s, %s) [using intersection and interpolation]\n", get_refname(), mapA->get_refname(), mapB->get_refname());
    else printf("%s <- multiply (%s, %s) [using intersection]\n", get_refname(), mapA->get_refname(), mapB->get_refname());

    init_from_intersection(mapA, mapB);
    
    for (gx=0; gx<xsize; gx++)
      for (gy=0; gy<ysize; gy++)
        for (gz=0; gz<zsize; gz++) {
          float x, y, z;
          voxel_coord(gx, gy, gz, x, y, z);

          if (interp) data[gz*xsize*ysize + gy*xsize + gx] = \
            mapA->voxel_value_interpolate_from_coord(x,y,z) * \
            mapB->voxel_value_interpolate_from_coord(x,y,z);
          else data[gz*xsize*ysize + gy*xsize + gx] = \
            mapA->voxel_value_from_coord(x,y,z) * \
            mapB->voxel_value_from_coord(x,y,z);
    }

  }
  
}



void VolMap::average(VolMap *mapA, VolMap *mapB, unsigned int flagsbits, Ops optype) {

  int gx, gy, gz;
  Operation *ops = GetOps(optype);

  bool interp = false;
  if (flagsbits & USE_INTERP) interp = true;

  if (flagsbits & USE_UNION) {

    // UNION VERSION
    
    if (interp) printf("%s <- average (%s, %s) [using union and interpolation]\n", get_refname(), mapA->get_refname(), mapB->get_refname());
    else printf("%s <- average (%s, %s) [using union]\n", get_refname(), mapA->get_refname(), mapB->get_refname());

    init_from_union(mapA, mapB);

    // use a 'safe' version of voxel_value_interpolate_from_coord
    // that returns zero if coordinate is outside the map

    for (gx=0; gx<xsize; gx++)
      for (gy=0; gy<ysize; gy++)
        for (gz=0; gz<zsize; gz++) {
          float x, y, z;
          voxel_coord(gx, gy, gz, x, y, z);

          if (interp) data[gz*xsize*ysize + gy*xsize + gx] = \
            ops->ConvertAverage((mapA->weight * ops->ConvertValue(mapA->voxel_value_interpolate_from_coord_safe(x,y,z)) + \
                                 mapB->weight * ops->ConvertValue(mapB->voxel_value_interpolate_from_coord_safe(x,y,z))) /(mapA->weight+mapB->weight));
          else data[gz*xsize*ysize + gy*xsize + gx] = \
            ops->ConvertAverage((mapA->weight * ops->ConvertValue(mapA->voxel_value_from_coord_safe(x,y,z)) + \
                                 mapB->weight * ops->ConvertValue(mapB->voxel_value_from_coord_safe(x,y,z))) /(mapA->weight+mapB->weight));
        }

  } else {

    // INTERSECTION VERSION
    
    if (interp) printf("%s <- average (%s, %s) [using intersection and interpolation]\n", get_refname(), mapA->get_refname(), mapB->get_refname());
    else printf("%s <- average (%s, %s) [using intersection]\n", get_refname(), mapA->get_refname(), mapB->get_refname());

    init_from_intersection(mapA, mapB);
    
    for (gx=0; gx<xsize; gx++)
    for (gy=0; gy<ysize; gy++)
    for (gz=0; gz<zsize; gz++) {
      float x, y, z;
      voxel_coord(gx, gy, gz, x, y, z);

      if (interp) data[gz*xsize*ysize + gy*xsize + gx] = \
          ops->ConvertAverage((mapA->weight * ops->ConvertValue(mapA->voxel_value_interpolate_from_coord(x,y,z)) + \
                               mapB->weight * ops->ConvertValue(mapB->voxel_value_interpolate_from_coord(x,y,z))) /(mapA->weight+mapB->weight));
      else data[gz*xsize*ysize + gy*xsize + gx] = \
          ops->ConvertAverage((mapA->weight * ops->ConvertValue(mapA->voxel_value_from_coord(x,y,z)) + \
                               mapB->weight * ops->ConvertValue(mapB->voxel_value_from_coord(x,y,z))) /(mapA->weight+mapB->weight));

    }

  }
    
  weight = mapA->weight + mapB->weight;

}




void VolMap::subtract(VolMap *mapA, VolMap *mapB, unsigned int flagsbits, Ops optype) {

  int gx, gy, gz;

  // subtracting maps by spatial coords is slower than doing it
  // directly, but allows for precisely subtracting unaligned maps,
  // and/or maps of different resolutions
    
  bool interp = false;
  if (flagsbits & USE_INTERP) interp = true;

  if (flagsbits & USE_UNION) {
  
    // UNION VERSION
    
    if (interp) printf("%s <- subtract (%s, %s) [using union and interpolation]\n", get_refname(), mapA->get_refname(), mapB->get_refname());
    else printf("%s <- subtract (%s, %s) [using union]\n", get_refname(), mapA->get_refname(), mapB->get_refname());

    init_from_union(mapA, mapB);

    // use a 'safe' version of voxel_value_interpolate_from_coord
    // that returns zero if coordinate is outside the map
    for (gx=0; gx<xsize; gx++)
      for (gy=0; gy<ysize; gy++)
        for (gz=0; gz<zsize; gz++) {
          float x, y, z;
          voxel_coord(gx, gy, gz, x, y, z);

          if (interp) data[gz*xsize*ysize + gy*xsize + gx] = \
            mapA->voxel_value_interpolate_from_coord_safe(x,y,z) - \
            mapB->voxel_value_interpolate_from_coord_safe(x,y,z);
          else data[gz*xsize*ysize + gy*xsize + gx] = \
            mapA->voxel_value_from_coord_safe(x,y,z) - \
            mapB->voxel_value_from_coord_safe(x,y,z);
        }

  } else {

    // INTERSECTION VERSION
    
    if (interp) printf("%s <- subtract (%s, %s) [using intersection and interpolation]\n", get_refname(), mapA->get_refname(), mapB->get_refname());
    else printf("%s <- subtract (%s, %s) [using intersection]\n", get_refname(), mapA->get_refname(), mapB->get_refname());

    init_from_intersection(mapA, mapB);
  
    for (gx=0; gx<xsize; gx++)
      for (gy=0; gy<ysize; gy++)
        for (gz=0; gz<zsize; gz++) {
          float x, y, z;
          voxel_coord(gx, gy, gz, x, y, z);
          
          if (interp) data[gz*xsize*ysize + gy*xsize + gx] = \
            mapA->voxel_value_interpolate_from_coord(x,y,z) - \
            mapB->voxel_value_interpolate_from_coord(x,y,z);
          else data[gz*xsize*ysize + gy*xsize + gx] = \
            mapA->voxel_value_from_coord(x,y,z) - \
            mapB->voxel_value_from_coord(x,y,z);
        }

  }

}



void VolMap::correlate(VolMap *mapA, VolMap *mapB, unsigned int flagsbits, Ops optype) {

  int gx, gy, gz;

  // correlating maps by spatial coords is slower than doing it
  // directly, but allows for precisely correlating unaligned maps,
  // and/or maps of different resolutions

  // Here we calculate the correlation coefficient using
  // \sum _i \frac{(mapA(i) - <mapA>)(mapB(i) - <mapB>)}{N sigma_A sigma_B}
  // where N is the number of voxels of the intersection.

  clone(mapA);

  bool interp = false;
  if (flagsbits & USE_INTERP) interp = true;

  // This condition is already being test in params.C
  if (flagsbits & USE_UNION) {
    fprintf(stderr, "Correlation calculation using union is not supported.\n");
    exit(1);
  }

  bool safe = false;
  if (flagsbits & USE_SAFE) safe = true;

  // INTERSECTION VERSION
  
  if (interp) printf("%s <- correlate (%s, %s) [using intersection and interpolation]\n", get_refname(), mapA->get_refname(), mapB->get_refname());
  else printf("%s <- correlate (%s, %s) [using intersection]\n", get_refname(), mapA->get_refname(), mapB->get_refname());

  // XXX - This is not memory efficient; it is preferable to use a
  //       version of init_from_intersection that does not allocate data.
  VolMap *inter = new VolMap();
  inter->init_from_intersection(mapA, mapB);

  // Calculate mapA and mapB means in the intersection
  double mapA_mean = 0.;
  double mapB_mean = 0.;
  int inter_size = 0;
  if (!safe) {
    inter_size = inter->xsize*inter->ysize*inter->zsize;
    for (gx=0; gx<inter->xsize; gx++)
      for (gy=0; gy<inter->ysize; gy++)
        for (gz=0; gz<inter->zsize; gz++) {
          float x, y, z;
          inter->voxel_coord(gx, gy, gz, x, y, z);
          if (interp) {
            mapA_mean += mapA->voxel_value_interpolate_from_coord(x,y,z);
            mapB_mean += mapB->voxel_value_interpolate_from_coord(x,y,z);
          } else {
            mapA_mean += mapA->voxel_value_from_coord(x,y,z);
            mapB_mean += mapB->voxel_value_from_coord(x,y,z);
          }
        }
  } else {
    float voxelA, voxelB;
    for (gx=0; gx<inter->xsize; gx++)
      for (gy=0; gy<inter->ysize; gy++)
        for (gz=0; gz<inter->zsize; gz++) {
          float x, y, z;
          inter->voxel_coord(gx, gy, gz, x, y, z);
          if (interp) {
            voxelA = mapA->voxel_value_interpolate_from_coord(x,y,z);
            voxelB = mapB->voxel_value_interpolate_from_coord(x,y,z);
          } else {
            voxelA = mapA->voxel_value_from_coord(x,y,z);
            voxelB = mapB->voxel_value_from_coord(x,y,z);
          }
          if (!ISNAN(voxelA) && !ISNAN(voxelB)) {
            mapA_mean += voxelA;
            mapB_mean += voxelB;
            inter_size++;
          }
        }
  }
  mapA_mean /= inter_size;
  mapB_mean /= inter_size;

  // Calculate mapA and mapB sigmas in the intersection and correlation
  double mapA_sigma = 0.;
  double mapB_sigma = 0.;
  double cc = 0.;
  if (!safe) {
    for (gx=0; gx<inter->xsize; gx++)
      for (gy=0; gy<inter->ysize; gy++)
        for (gz=0; gz<inter->zsize; gz++) {
          float x, y, z;
          float voxelA, voxelB;
          inter->voxel_coord(gx, gy, gz, x, y, z);
          if (interp) {
            voxelA = mapA->voxel_value_interpolate_from_coord(x,y,z);
            voxelB = mapB->voxel_value_interpolate_from_coord(x,y,z);
          } else {
            voxelA = mapA->voxel_value_from_coord(x,y,z);
            voxelB = mapB->voxel_value_from_coord(x,y,z);
          }
          mapA_sigma += (voxelA - mapA_mean)*(voxelA - mapA_mean);
          mapB_sigma += (voxelB - mapB_mean)*(voxelB - mapB_mean);
          cc += (voxelA - mapA_mean)*(voxelB - mapB_mean);
        }
  } else {
    for (gx=0; gx<inter->xsize; gx++)
      for (gy=0; gy<inter->ysize; gy++)
        for (gz=0; gz<inter->zsize; gz++) {
          float x, y, z;
          float voxelA, voxelB;
          inter->voxel_coord(gx, gy, gz, x, y, z);
          if (interp) {
            voxelA = mapA->voxel_value_interpolate_from_coord(x,y,z);
            voxelB = mapB->voxel_value_interpolate_from_coord(x,y,z);
          } else {
            voxelA = mapA->voxel_value_from_coord(x,y,z);
            voxelB = mapB->voxel_value_from_coord(x,y,z);
          }
          if (!ISNAN(voxelA) && !ISNAN(voxelB)) {
            mapA_sigma += (voxelA - mapA_mean)*(voxelA - mapA_mean);
            mapB_sigma += (voxelB - mapB_mean)*(voxelB - mapB_mean);
            cc += (voxelA - mapA_mean)*(voxelB - mapB_mean);
          }
        }
  }
  mapA_sigma = sqrt(mapA_sigma/inter_size);
  mapB_sigma = sqrt(mapB_sigma/inter_size);
  cc /= (inter_size * mapA_sigma * mapB_sigma);


  delete inter;

  printf("Correlation coefficient = %g\n", cc);

}


void VolMap::correlate_map(VolMap *mapA, VolMap *mapB, unsigned int flagsbits, Ops optype) {

  int gx, gy, gz;

  // correlating maps by spatial coords is slower than doing it
  // directly, but allows for precisely correlating unaligned maps,
  // and/or maps of different resolutions

  // Here we calculate the correlation coefficient using
  // \sum _i \frac{(mapA(i) - <mapA>)(mapB(i) - <mapB>)}{N sigma_A sigma_B}
  // where N is the number of voxels of the intersection.

  double radius = 5.0; // XXX - hardcoded for now
  double radius2 = radius*radius;

  clone(mapA);

  // Find out how many grid points does the sphere radius correspond to
  int numgrid = (int) MAX(MAX(radius/vnorm(xdelta), radius/vnorm(ydelta)), radius/vnorm(zdelta));

  printf("DEBUG: numgrid = %d\n", numgrid);

  bool interp = false;
  if (flagsbits & USE_INTERP) interp = true;

  // This condition is already being test in params.C
  if (flagsbits & USE_UNION) {
    fprintf(stderr, "Correlation calculation using union is not supported.\n");
    exit(1);
  }

  //bool safe = false;
  //if (flagsbits & USE_SAFE) safe = true;

  // INTERSECTION VERSION
  
  if (interp) printf("%s <- correlate_map (%s, %s) [using intersection and interpolation]\n", get_refname(), mapA->get_refname(), mapB->get_refname());
  else printf("%s <- correlate_map (%s, %s) [using intersection]\n", get_refname(), mapA->get_refname(), mapB->get_refname());

  init_from_intersection(mapA, mapB);

  // Calculate mapA and mapB means in the intersection
  for (gx=0; gx<xsize; gx++)
    for (gy=0; gy<ysize; gy++)
      for (gz=0; gz<zsize; gz++) {
        float x, y, z, xx, yy, zz;
        int gxx, gyy, gzz;
        voxel_coord(gx, gy, gz, x, y, z); // center of sphere
        
        // Calculate mean
        double mapA_mean = 0.;
        double mapB_mean = 0.;
        int inter_size = 0;
        float voxelA, voxelB;
        for (gxx = gx-numgrid; gxx <= gx+numgrid; gxx++)
          for (gyy = gy-numgrid; gyy <= gy+numgrid; gyy++)
            for (gzz = gz-numgrid; gzz <= gz+numgrid; gzz++) {
              voxel_coord(gxx, gyy, gzz, xx, yy, zz);
              // Only consider this voxel if it is within sphere
              if ((xx-x)*(xx-x) + (yy-y)*(yy-y) + (zz-z)*(zz-z) < radius2) {
                if (interp) {
                  voxelA = mapA->voxel_value_interpolate_from_coord(xx,yy,zz);
                  voxelB = mapB->voxel_value_interpolate_from_coord(xx,yy,zz);
                } else {
                  voxelA = mapA->voxel_value_from_coord(xx,yy,zz);
                  voxelB = mapB->voxel_value_from_coord(xx,yy,zz);
                }
                if (!ISNAN(voxelA) && !ISNAN(voxelB)) {
                  inter_size++;
                  mapA_mean += voxelA;
                  mapB_mean += voxelB;
                }
              }
            }
        mapA_mean /= inter_size;
        mapB_mean /= inter_size;

        // Calculate sigmas and correlation
        double mapA_sigma = 0.;
        double mapB_sigma = 0.;
        double cc = 0.;
        for (gxx = gx-numgrid; gxx <= gx+numgrid; gxx++)
          for (gyy = gy-numgrid; gyy <= gy+numgrid; gyy++)
            for (gzz = gz-numgrid; gzz <= gz+numgrid; gzz++) {
              voxel_coord(gxx, gyy, gzz, xx, yy, zz);
              // Only consider this voxel if it is within sphere
              if ((xx-x)*(xx-x) + (yy-y)*(yy-y) + (zz-z)*(zz-z) < radius2) {
                if (interp) {
                  voxelA = mapA->voxel_value_interpolate_from_coord(xx,yy,zz);
                  voxelB = mapB->voxel_value_interpolate_from_coord(xx,yy,zz);
                } else {
                  voxelA = mapA->voxel_value_from_coord(xx,yy,zz);
                  voxelB = mapB->voxel_value_from_coord(xx,yy,zz);
                }
                if (!ISNAN(voxelA) && !ISNAN(voxelB)) {
                  mapA_sigma += (voxelA - mapA_mean)*(voxelA - mapA_mean);
                  mapB_sigma += (voxelB - mapB_mean)*(voxelB - mapB_mean);
                  cc += (voxelA - mapA_mean)*(voxelB - mapB_mean);
                }
              }
            }

        mapA_sigma = sqrt(mapA_sigma/inter_size);
        mapB_sigma = sqrt(mapB_sigma/inter_size);

        cc /= (inter_size * mapA_sigma * mapB_sigma);
        // XXX - need to deal with NANs (e.g., if sigma = 0)

        data[gz*xsize*ysize + gy*xsize + gx] = cc;

      }

}

//
// Check if two maps are equivalent. We could use the subtract function
// here, but since this function is intended to be used in regression tests,
// it is better (at least for now) to keep an independent implementation.
//
// TODO - need to compare the cells (origin, axes, grid spacing...)
//      - provide more info about difference map, not simply pass/fail
//
void VolMap::compare(VolMap *mapA, VolMap *mapB, unsigned int flagsbits, Ops optype) {

  int gx, gy, gz;

  // subtracting maps by spatial coords is slower than doing it
  // directly, but allows for precisely subtracting unaligned maps,
  // and/or maps of different resolutions
    
  bool interp = false;
  if (flagsbits & USE_INTERP) interp = true;

  if (flagsbits & USE_UNION) {
  
    // UNION VERSION
    
    if (interp) printf("%s <- compare (%s, %s) [using union and interpolation]\n", get_refname(), mapA->get_refname(), mapB->get_refname());
    else printf("%s <- compare (%s, %s) [using union]\n", get_refname(), mapA->get_refname(), mapB->get_refname());

    init_from_union(mapA, mapB);

    // use a 'safe' version of voxel_value_interpolate_from_coord
    // that returns zero if coordinate is outside the map
    for (gx=0; gx<xsize; gx++)
      for (gy=0; gy<ysize; gy++)
        for (gz=0; gz<zsize; gz++) {
          float x, y, z;
          voxel_coord(gx, gy, gz, x, y, z);

          if (interp) data[gz*xsize*ysize + gy*xsize + gx] = \
            mapA->voxel_value_interpolate_from_coord_safe(x,y,z) - \
            mapB->voxel_value_interpolate_from_coord_safe(x,y,z);
          else data[gz*xsize*ysize + gy*xsize + gx] = \
            mapA->voxel_value_from_coord_safe(x,y,z) - \
            mapB->voxel_value_from_coord_safe(x,y,z);
        }

  } else {

    // INTERSECTION VERSION
    
    if (interp) printf("%s <- compare (%s, %s) [using intersection and interpolation]\n", get_refname(), mapA->get_refname(), mapB->get_refname());
    else printf("%s <- compare (%s, %s) [using intersection]\n", get_refname(), mapA->get_refname(), mapB->get_refname());

    init_from_intersection(mapA, mapB);
  
    for (gx=0; gx<xsize; gx++)
      for (gy=0; gy<ysize; gy++)
        for (gz=0; gz<zsize; gz++) {
          float x, y, z;
          voxel_coord(gx, gy, gz, x, y, z);
          
          if (interp) data[gz*xsize*ysize + gy*xsize + gx] = \
            mapA->voxel_value_interpolate_from_coord(x,y,z) - \
            mapB->voxel_value_interpolate_from_coord(x,y,z);
          else data[gz*xsize*ysize + gy*xsize + gx] = \
            mapA->voxel_value_from_coord(x,y,z) - \
            mapB->voxel_value_from_coord(x,y,z);
        }

  }

  // Check if each voxel in difference map is smaller than a certain tolerance
  bool pass=true;
  for (gx=0; gx<xsize; gx++)
    for (gy=0; gy<ysize; gy++)
      for (gz=0; gz<zsize; gz++) 
        if (fabs(voxel_value(gx,gy,gz)) > VOLUTIL_COMPARE_TOL) {
          pass = false;
        }

  if (pass == false) {
    printf("Comparison test FAILED.\n");
  } else {
    printf("Comparison test PASSED.\n");
  }

}


