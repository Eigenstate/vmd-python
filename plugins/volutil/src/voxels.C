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
 *	$RCSfile: voxels.C,v $
 *	$Author: ltrabuco $	$Locker:  $		$State: Exp $
 *	$Revision: 1.1 $	$Date: 2009/08/06 20:58:47 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * volmap.C - This file contains the initialization and manipulation 
 * routines for the VolMap class.
 *
 ***************************************************************************/

#include <math.h>

#include "volmap.h"
#include "vec.h"





/* VOXELS */  

/// return voxel, after safely clamping index to valid range
float VolMap::voxel_value_safe(int x, int y, int z) const {
  int xx, yy, zz; 
  xx = (x > 0) ? ((x < xsize) ? x : xsize-1) : 0;
  yy = (y > 0) ? ((y < ysize) ? y : ysize-1) : 0;
  zz = (z > 0) ? ((z < zsize) ? z : zsize-1) : 0;
  int index = zz*xsize*ysize + yy*xsize + xx;
  return data[index];
}


/// return voxel from user provided data, after safely clamping index to valid range
float VolMap::voxel_value_safe(int x, int y, int z, int myxsize, int myysize, int myzsize, float *mydata) const {
  int xx, yy, zz; 
  xx = (x > 0) ? ((x < myxsize) ? x : myxsize-1) : 0;
  yy = (y > 0) ? ((y < myysize) ? y : myysize-1) : 0;
  zz = (z > 0) ? ((z < myzsize) ? z : myzsize-1) : 0;
  int index = zz*myxsize*myysize + yy*myxsize + xx;
  return mydata[index];
}


/// return interpolated value from 8 nearest neighbor voxels
float VolMap::voxel_value_interpolate(float xv, float yv, float zv) const {
  int x = (int) xv;
  int y = (int) yv;
  int z = (int) zv;
  // fractional offset
  float xf = xv - x;
  float yf = yv - y;
  float zf = zv - z;
  float xlerps[4];
  float ylerps[2];
  float tmp;

  tmp = voxel_value_safe(x, y, z);
  xlerps[0] = tmp + xf*(voxel_value_safe(x+1, y, z) - tmp);

  tmp = voxel_value_safe(x, y+1, z);
  xlerps[1] = tmp + xf*(voxel_value_safe(x+1, y+1, z) - tmp);

  tmp = voxel_value_safe(x, y, z+1);
  xlerps[2] = tmp + xf*(voxel_value_safe(x+1, y, z+1) - tmp);

  tmp = voxel_value_safe(x, y+1, z+1);
  xlerps[3] = tmp + xf*(voxel_value_safe(x+1, y+1, z+1) - tmp);

  ylerps[0] = xlerps[0] + yf*(xlerps[1] - xlerps[0]);
  ylerps[1] = xlerps[2] + yf*(xlerps[3] - xlerps[2]);

  return ylerps[0] + zf*(ylerps[1] - ylerps[0]);
}


/// return interpolated value from 8 nearest neighbor voxels
float VolMap::voxel_value_interpolate_pmf_exp(float xv, float yv, float zv) const {
  int x = (int) xv;
  int y = (int) yv;
  int z = (int) zv;
  float xf = xv - x;
  float yf = yv - y;
  float zf = zv - z;
  float xlerps[4];
  float ylerps[2];
  float tmp;

  tmp = exp(-voxel_value_safe(x, y, z));
  xlerps[0] = tmp + xf*(exp(-voxel_value_safe(x+1, y, z)) - tmp);

  tmp = exp(-voxel_value_safe(x, y+1, z));
  xlerps[1] = tmp + xf*(exp(-voxel_value_safe(x+1, y+1, z)) - tmp);

  tmp = exp(-voxel_value_safe(x, y, z+1));
  xlerps[2] = tmp + xf*(exp(-voxel_value_safe(x+1, y, z+1)) - tmp);

  tmp = exp(-voxel_value_safe(x, y+1, z+1));
  xlerps[3] = tmp + xf*(exp(-voxel_value_safe(x+1, y+1, z+1)) - tmp);

  ylerps[0] = xlerps[0] + yf*(xlerps[1] - xlerps[0]);
  ylerps[1] = xlerps[2] + yf*(xlerps[3] - xlerps[2]);

  return -log(ylerps[0] + zf*(ylerps[1] - ylerps[0]));
}


/// Return voxel index. If coord is outside map dimensions return -1.
int VolMap::coord_to_index(float x, float y, float z) const {
  x -= origin[0];
  y -= origin[1];
  z -= origin[2];
  // XXX Needs to be fixed for non-orthog cells (subtract out projected component every step)
  int gx = int((x*xaxis[0] + y*xaxis[1] + z*xaxis[2])/vnorm(xaxis));
  int gy = int((x*yaxis[0] + y*yaxis[1] + z*yaxis[2])/vnorm(yaxis));
  int gz = int((x*zaxis[0] + y*zaxis[1] + z*zaxis[2])/vnorm(zaxis));
  if (gx>=0 && gy>=0 && gz>=0 && gx<xsize && gy<ysize && gz<zsize)
    return (gx + gy*xsize + gz*ysize*xsize);

  return -1;
}


void VolMap::index_to_coord(int index, float &x, float &y, float &z) const {
  x = origin[0] + xdelta[0]*(index%xsize);
  y = origin[1] + ydelta[1]*((index/xsize)%ysize);
  z = origin[2] + zdelta[2]*(index/(xsize*ysize));
}


  
/// return value of voxel, based on atomic coords.
/// XXX need to account for non-orthog. cells
float VolMap::voxel_value_from_coord(float xpos, float ypos, float zpos) const {
  float min_coord[3];
  for (int i=0; i<3; i++) min_coord[i] = origin[i] - 0.5*(xdelta[i] + ydelta[i] + zdelta[i]);
  xpos -= min_coord[0];
  ypos -= min_coord[1];
  zpos -= min_coord[2];
  int gx = (int) (xpos/xdelta[0]); // XXX this is wrong for non-orthog cells.
  int gy = (int) (ypos/ydelta[1]);
  int gz = (int) (zpos/zdelta[2]);
  if (gx < 0 || gx >= xsize) return kNAN;
  if (gy < 0 || gy >= ysize) return kNAN;
  if (gz < 0 || gz >= zsize) return kNAN;
  return data[gz*xsize*ysize + gy*xsize + gx];
}

/// return value of voxel, based on atomic coords.
/// XXX need to account for non-orthog. cells
/// this version returns zero if the coordinates are outside the map
float VolMap::voxel_value_from_coord_safe(float xpos, float ypos, float zpos) const {
  float min_coord[3];
  for (int i=0; i<3; i++) min_coord[i] = origin[i] - 0.5*(xdelta[i] + ydelta[i] + zdelta[i]);
  xpos -= min_coord[0];
  ypos -= min_coord[1];
  zpos -= min_coord[2];
  int gx = (int) (xpos/xdelta[0]); // XXX this is wrong for non-orthog cells.
  int gy = (int) (ypos/ydelta[1]);
  int gz = (int) (zpos/zdelta[2]);
  if (gx < 0 || gx >= xsize) return 0.;
  if (gy < 0 || gy >= ysize) return 0.;
  if (gz < 0 || gz >= zsize) return 0.;
  return data[gz*xsize*ysize + gy*xsize + gx];
}


/// return interpolated value of voxel, based on atomic coords.
/// XXX need to account for non-orthog. cells
float VolMap::voxel_value_interpolate_from_coord(float xpos, float ypos, float zpos, Ops optype) const {
  xpos = (xpos-origin[0])/xdelta[0];
  ypos = (ypos-origin[1])/ydelta[1];
  zpos = (zpos-origin[2])/zdelta[2];
  int gx = (int) xpos; // XXX this is wrong for non-orthog cells.
  int gy = (int) ypos;
  int gz = (int) zpos;
  if (gx < 0 || gx >= xsize) return kNAN;
  if (gy < 0 || gy >= ysize) return kNAN;
  if (gz < 0 || gz >= zsize) return kNAN;
  
  if (optype==PMF)
    return voxel_value_interpolate_pmf_exp(xpos, ypos, zpos);
  else
    return voxel_value_interpolate(xpos, ypos, zpos);
}

/// return interpolated value of voxel, based on atomic coords.
/// XXX need to account for non-orthog. cells
/// this version returns zero if the coordinates are outside the map
float VolMap::voxel_value_interpolate_from_coord_safe(float xpos, float ypos, float zpos, Ops optype) const {
  xpos = (xpos-origin[0])/xdelta[0];
  ypos = (ypos-origin[1])/ydelta[1];
  zpos = (zpos-origin[2])/zdelta[2];
  int gx = (int) xpos; // XXX this is wrong for non-orthog cells.
  int gy = (int) ypos;
  int gz = (int) zpos;
  if (gx < 0 || gx >= xsize) return 0.;
  if (gy < 0 || gy >= ysize) return 0.;
  if (gz < 0 || gz >= zsize) return 0.;
  
  if (optype==PMF)
    return voxel_value_interpolate_pmf_exp(xpos, ypos, zpos);
  else
    return voxel_value_interpolate(xpos, ypos, zpos);
}


