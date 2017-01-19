/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2016 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile: VolumetricData.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.39 $	$Date: 2016/11/28 03:05:06 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Base class for storing volumetric data and associated gradient data
 *
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "VolumetricData.h"
#include "Matrix4.h"
#include "utilities.h"
  
#ifndef NAN //not a number
  const float NAN = sqrtf(-1.f); //need some kind of portable NAN definition
#endif

/// constructor for single precision origin/axes info
VolumetricData::VolumetricData(const char *dataname, const float *o, 
                 const float *xa, const float *ya, const float *za,
                 int xs, int ys, int zs, float *dataptr) {
  name = stringdup(dataname);
  data = dataptr;
  xsize = xs;
  ysize = ys;
  zsize = zs;

  gradient = NULL;
  datamin = datamax = 0;

  for (int i=0; i<3; i++) {
    origin[i] = (double)o[i];
    xaxis[i] = (double)xa[i];
    yaxis[i] = (double)ya[i];
    zaxis[i] = (double)za[i];
  } 

  const int ndata = xsize*ysize*zsize;
  if (ndata > 0) {
#if 1
    // use fast 16-byte-aligned min/max routine
    minmax_1fv_aligned(data, ndata, &datamin, &datamax);
#else
    float min, max;
    min = max = data[0];
    for (int j=1; j<ndata; j++) {
      if (min > data[j]) min = data[j];
      if (max < data[j]) max = data[j];
    }
    datamin = min;
    datamax = max;
#endif
  }
}


/// constructor for double precision origin/axes info
VolumetricData::VolumetricData(const char *dataname, const double *o, 
                 const double *xa, const double *ya, const double *za,
                 int xs, int ys, int zs, float *dataptr) {
  name = stringdup(dataname);
  data = dataptr;
  xsize = xs;
  ysize = ys;
  zsize = zs;

  gradient = NULL;
  datamin = datamax = 0;

  for (int i=0; i<3; i++) {
    origin[i] = o[i];
    xaxis[i] = xa[i];
    yaxis[i] = ya[i];
    zaxis[i] = za[i];
  } 

  const int ndata = xsize*ysize*zsize;
  if (ndata > 0) {
#if 1
    // use fast 16-byte-aligned min/max routine
    minmax_1fv_aligned(data, ndata, &datamin, &datamax);
#else
    float min, max;
    min = max = data[0];
    for (int j=1; j<ndata; j++) {
      if (min > data[j]) min = data[j];
      if (max < data[j]) max = data[j];
    }
    datamin = min;
    datamax = max;
#endif
  }
}


/// destructor
VolumetricData::~VolumetricData() {
  delete [] name;
  delete [] data;
  delete [] gradient;
}

/// Set the current human readable name of the dataset by the
/// given string.
void VolumetricData::set_name(const char *newname) {
  if (name) delete[] name;
  name = new char[strlen(newname)+1];
  strcpy(name, newname);
}


/// return cell side lengths
void VolumetricData::cell_lengths(float *xl, float *yl, float *zl) const {
  float xsinv, ysinv, zsinv;

  // set scaling factors correctly
  if (xsize > 1)
    xsinv = 1.0f / (xsize - 1.0f);
  else
    xsinv = 1.0f;

  if (ysize > 1)
    ysinv = 1.0f / (ysize - 1.0f);
  else
    ysinv = 1.0f;

  if (zsize > 1)
    zsinv = 1.0f / (zsize - 1.0f);
  else
    zsinv = 1.0f;

  *xl = sqrtf(float(dot_prod(xaxis, xaxis))) * xsinv;
  *yl = sqrtf(float(dot_prod(yaxis, yaxis))) * ysinv;
  *zl = sqrtf(float(dot_prod(zaxis, zaxis))) * zsinv;
}


/// return cell axes
void VolumetricData::cell_axes(float *xax, float *yax, float *zax) const {
  float xsinv, ysinv, zsinv;

  // set scaling factors correctly
  if (xsize > 1)
    xsinv = 1.0f / (xsize - 1.0f);
  else
    xsinv = 1.0f;

  if (ysize > 1)
    ysinv = 1.0f / (ysize - 1.0f);
  else
    ysinv = 1.0f;

  if (zsize > 1)
    zsinv = 1.0f / (zsize - 1.0f);
  else
    zsinv = 1.0f;

  xax[0] = float(xaxis[0] * xsinv);
  xax[1] = float(xaxis[1] * xsinv);
  xax[2] = float(xaxis[2] * xsinv);

  yax[0] = float(yaxis[0] * ysinv);
  yax[1] = float(yaxis[1] * ysinv);
  yax[2] = float(yaxis[2] * ysinv);

  zax[0] = float(zaxis[0] * zsinv);
  zax[1] = float(zaxis[1] * zsinv);
  zax[2] = float(zaxis[2] * zsinv);
}


/// return cell basis directions
void VolumetricData::cell_dirs(float *xad, float *yad, float *zad) const {
  float xl, yl, zl;

  cell_lengths(&xl, &yl, &zl);
  cell_axes(xad, yad, zad);

  xad[0] /= xl;
  xad[1] /= xl;
  xad[2] /= xl;

  yad[0] /= yl;
  yad[1] /= yl;
  yad[2] /= yl;

  zad[0] /= zl;
  zad[1] /= zl;
  zad[2] /= zl;
}


/// return volumetric coordinate from cartesian coordinate
/// XXX need to account for non-orthog. cells
void VolumetricData::voxel_coord_from_cartesian_coord(const float *carcoord, float *voxcoord, int shiftflag) const {
  float min_coord[3];

  if (shiftflag) {
    // shift coordinates by a half-voxel so that integer truncation works
    for (int i=0; i<3; i++) 
      min_coord[i] = (float) origin[i] - 0.5f*float(xaxis[i]/(xsize-1) + yaxis[i]/(ysize-1) + zaxis[i]/(zsize-1));
  } else {
    for (int i=0; i<3; i++)
      min_coord[i] = (float) origin[i];
  }

  //create transformation matrix...
  float matval[16];
  matval[0 ] = (float) xaxis[0]/(xsize-1);
  matval[1 ] = (float) yaxis[0]/(ysize-1);
  matval[2 ] = (float) zaxis[0]/(zsize-1);
  matval[3 ] = 0.f;
  matval[4 ] = (float) xaxis[1]/(xsize-1);
  matval[5 ] = (float) yaxis[1]/(ysize-1);
  matval[6 ] = (float) zaxis[1]/(zsize-1);
  matval[7 ] = 0.f;
  matval[8 ] = (float) xaxis[2]/(xsize-1);
  matval[9 ] = (float) yaxis[2]/(ysize-1);
  matval[10] = (float) zaxis[2]/(zsize-1);
  matval[11] = 0.f;
  matval[12] = (float) min_coord[0];
  matval[13] = (float) min_coord[1];
  matval[14] = (float) min_coord[2];
  matval[15] = 1.f;
  Matrix4 matrix(matval);
  matrix.inverse();

  matrix.multpoint3d(carcoord, voxcoord);
}


/// returns grid index
long VolumetricData::voxel_index_from_coord(float xpos, float ypos, float zpos) const {
  float realcoord[3];
  float gridpoint[3];
  realcoord[0] = xpos;
  realcoord[1] = ypos;
  realcoord[2] = zpos;
 
  // calculate grid coordinate, shifting by a half-voxel 
  voxel_coord_from_cartesian_coord(realcoord, gridpoint, 1);
  
  int gx = (int) gridpoint[0]; 
  int gy = (int) gridpoint[1];
  int gz = (int) gridpoint[2];
  if (gx < 0 || gx >= xsize) return -1;
  if (gy < 0 || gy >= ysize) return -1;
  if (gz < 0 || gz >= zsize) return -1;
  return gz*xsize*ysize + gy*xsize + gx;
}


/// return voxel, after safely clamping index to valid range
float VolumetricData::voxel_value_safe(int x, int y, int z) const {
  int xx, yy, zz; 
  xx = (x > 0) ? ((x < xsize) ? x : xsize-1) : 0;
  yy = (y > 0) ? ((y < ysize) ? y : ysize-1) : 0;
  zz = (z > 0) ? ((z < zsize) ? z : zsize-1) : 0;
  long index = zz*xsize*ysize + yy*xsize + xx;
  return data[index];
}


/// return interpolated value from 8 nearest neighbor voxels
float VolumetricData::voxel_value_interpolate(float xv, float yv, float zv) const {
  int x = (int) xv;
  int y = (int) yv;
  int z = (int) zv;
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


/// return the value of the voxel nearest to the cartesian coordinate
float VolumetricData::voxel_value_from_coord(float xpos, float ypos, float zpos) const {
  long ind = voxel_index_from_coord(xpos, ypos, zpos);
  if (ind > 0)
    return data[ind];
  else
    return NAN;
}


/// return interpolated value of voxel, based on cartesian coords.
float VolumetricData::voxel_value_interpolate_from_coord(float xpos, float ypos, float zpos) const {;
  float realcoord[3];
  float gridpoint[3];
  realcoord[0] = xpos;
  realcoord[1] = ypos;
  realcoord[2] = zpos;
 
  // calculate grid coordinate, shifting by a half-voxel 
  voxel_coord_from_cartesian_coord(realcoord, gridpoint, 0);
  
  int gx = (int) gridpoint[0]; 
  int gy = (int) gridpoint[1];
  int gz = (int) gridpoint[2];
  if (gx < 0 || gx >= xsize) return NAN;
  if (gy < 0 || gy >= ysize) return NAN;
  if (gz < 0 || gz >= zsize) return NAN;
  return voxel_value_interpolate(gridpoint[0], gridpoint[1], gridpoint[2]);
}


/// set the volume gradient
void VolumetricData::set_volume_gradient(float *grad) {
  if (gradient) {
    delete [] gradient; 
    gradient = NULL;
  }

  if (!gradient) {
    long gsz = ((long) xsize) * ((long) ysize) * ((long) zsize) * 3L;
    gradient = new float[gsz];
    memcpy(gradient, grad, gsz*sizeof(float));
  }
}


/// (re)calculate the volume gradient
void VolumetricData::compute_volume_gradient(void) {
  int xi, yi, zi;
  float xs, ys, zs;
  float xl, yl, zl;
  long row;

  if (!gradient) {
    long gsz = ((long) xsize) * ((long) ysize) * ((long) zsize) * 3L;
#if 0
printf("xsz: %d  ysz: %d  zsz: %d\n", xsize, ysize, zsize);
printf("gradient map sz: %ld, MB: %ld\n", gsz, gsz*sizeof(float)/(1024*1024));
#endif
    gradient = new float[gsz];
  }

  // calculate cell side lengths
  cell_lengths(&xl, &yl, &zl);

  // gradient axis scaling values and averaging factors, to correctly
  // calculate the gradient for volumes with irregular cell spacing
  xs = -0.5f / xl;
  ys = -0.5f / yl;
  zs = -0.5f / zl;

  for (zi=0; zi<zsize; zi++) {
    int zm, zp;
    zm = clamp_int(zi - 1, 0, zsize - 1);
    zp = clamp_int(zi + 1, 0, zsize - 1);

    for (yi=0; yi<ysize; yi++) {
      int ym, yp;
      ym = clamp_int(yi - 1, 0, ysize - 1);
      yp = clamp_int(yi + 1, 0, ysize - 1);

      row = (zi * xsize * ysize) + (yi * xsize);
      for (xi=0; xi<xsize; xi++) {
        long index = (row + xi) * 3L;
        int xm, xp;
        xm = clamp_int(xi - 1, 0, xsize - 1);
        xp = clamp_int(xi + 1, 0, xsize - 1);

        // Calculate the volume gradient at each grid cell.
        // Gradients are now stored unnormalized, since we need them in pure
        // form in order to draw field lines etc.  Shading code will now have
        // to do renormalization for itself on-the-fly.

        // XXX this gradient is only correct for orthogonal grids, since
        // we're using the array index offsets rather to calculate the gradient
        // rather than voxel coordinate offsets.  This will have to be
        // re-worked for non-orthogonal datasets.
        gradient[index    ] =
          (voxel_value(xp, yi, zi) - voxel_value(xm, yi, zi)) * xs;
        gradient[index + 1] =
          (voxel_value(xi, yp, zi) - voxel_value(xi, ym, zi)) * ys;
        gradient[index + 2] =
          (voxel_value(xi, yi, zp) - voxel_value(xi, yi, zm)) * zs;
      }
    }
  }
}


/// return gradient, after safely clamping voxel coordinate to valid range
void VolumetricData::voxel_gradient_safe(int x, int y, int z, float *grad) const {
  int xx, yy, zz;
  xx = (x > 0) ? ((x < xsize) ? x : xsize-1) : 0;
  yy = (y > 0) ? ((y < ysize) ? y : ysize-1) : 0;
  zz = (z > 0) ? ((z < zsize) ? z : zsize-1) : 0;
  long index = zz*xsize*ysize + yy*xsize + xx;
  grad[0] = gradient[index*3L    ];
  grad[1] = gradient[index*3L + 1];
  grad[2] = gradient[index*3L + 2];
}


/// return voxel gradient interpolated between 8 nearest neighbor voxels
void VolumetricData::voxel_gradient_interpolate(const float *voxcoord, float *grad) const {
  float gtmpa[3], gtmpb[3];
  int x = (int) voxcoord[0];
  int y = (int) voxcoord[1];
  int z = (int) voxcoord[2];
  float xf = voxcoord[0] - x;
  float yf = voxcoord[1] - y;
  float zf = voxcoord[2] - z;
  float xlerpsa[3];
  float xlerpsb[3];
  float xlerpsc[3];
  float xlerpsd[3];
  float ylerpsa[3];
  float ylerpsb[3];

  voxel_gradient_safe(x,   y,   z,   gtmpa);
  voxel_gradient_safe(x+1, y,   z,   gtmpb);
  vec_lerp(xlerpsa, gtmpa, gtmpb, xf);
 
  voxel_gradient_safe(x,   y+1, z,   gtmpa);
  voxel_gradient_safe(x+1, y+1, z,   gtmpb);
  vec_lerp(xlerpsb, gtmpa, gtmpb, xf);

  voxel_gradient_safe(x,   y,   z+1, gtmpa);
  voxel_gradient_safe(x+1, y,   z+1, gtmpb);
  vec_lerp(xlerpsc, gtmpa, gtmpb, xf);

  voxel_gradient_safe(x,   y+1, z+1, gtmpa);
  voxel_gradient_safe(x+1, y+1, z+1, gtmpb);
  vec_lerp(xlerpsd, gtmpa, gtmpb, xf);

  vec_lerp(ylerpsa, xlerpsa, xlerpsb, yf);
  vec_lerp(ylerpsb, xlerpsc, xlerpsd, yf);

  vec_lerp(grad, ylerpsa, ylerpsb, zf);
}


/// return voxel gradient nearest to the specified cartesian coordinate
void VolumetricData::voxel_gradient_from_coord(const float *coord, float *grad) const {
  int vx, vy, vz;
  float voxcoord[3];
  voxel_coord_from_cartesian_coord(coord, voxcoord, 1);
  vx = (int) voxcoord[0];
  vy = (int) voxcoord[1];
  vz = (int) voxcoord[2];
  voxel_gradient_safe(vx, vy, vz, grad);
}


/// return interpolated voxel gradient for cartesian coordinate
void VolumetricData::voxel_gradient_interpolate_from_coord(const float *coord, float *grad) const {
  float voxcoord[3];
  voxel_coord_from_cartesian_coord(coord, voxcoord, 0);

  if (voxcoord[0] < 0 || voxcoord[0] >= (xsize-1) ||
      voxcoord[1] < 0 || voxcoord[1] >= (ysize-1) ||
      voxcoord[2] < 0 || voxcoord[2] >= (zsize-1)) {
    grad[0] = NAN; 
    grad[1] = NAN; 
    grad[2] = NAN; 
    return; 
  }

  voxel_gradient_interpolate(voxcoord, grad);
}


