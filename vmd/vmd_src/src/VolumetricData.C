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
 *	$RCSfile: VolumetricData.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.60 $	$Date: 2019/01/17 21:21:02 $
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
#include "Segmentation.h"
#include <float.h> // FLT_MAX etc
  
#ifndef NAN //not a number
  const float NAN = sqrtf(-1.f); //need some kind of portable NAN definition
#endif

#define MIN(X,Y) (((X)<(Y))? (X) : (Y))
#define MAX(X,Y) (((X)>(Y))? (X) : (Y))

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
  gradient_isvalid = false;
  invalidate_minmax();   // min/max will be computed on next request
  invalidate_mean();     // mean will be computed on first request
  invalidate_sigma();    // sigma will be computed on first request

  for (int i=0; i<3; i++) {
    origin[i] = (double)o[i];
    xaxis[i] = (double)xa[i];
    yaxis[i] = (double)ya[i];
    zaxis[i] = (double)za[i];
  } 

  compute_minmax();
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
  gradient_isvalid = false;
  invalidate_minmax();   // min/max will be computed on next request
  invalidate_mean();     // mean will be computed on first request
  invalidate_sigma();    // sigma will be computed on first request

  for (int i=0; i<3; i++) {
    origin[i] = o[i];
    xaxis[i] = xa[i];
    yaxis[i] = ya[i];
    zaxis[i] = za[i];
  } 

  compute_minmax();
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
  return gz*long(xsize)*long(ysize) + gy*long(xsize) + gx;
}


/// return voxel, after safely clamping index to valid range
float VolumetricData::voxel_value_safe(int x, int y, int z) const {
  long xx, yy, zz; 
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


/// return the value of the voxel nearest to the cartesian coordinate
/// this safe version returns zero if the coordinates are outside the map
float VolumetricData::voxel_value_from_coord_safe(float xpos, float ypos, float zpos) const {
  long ind = voxel_index_from_coord(xpos, ypos, zpos);
  if (ind > 0)
    return data[ind];
  else
    return 0;
}


/// return interpolated value of voxel, based on cartesian coords.
/// this safe version returns zero if the coordinates are outside the map
float VolumetricData::voxel_value_interpolate_from_coord_safe(float xpos, float ypos, float zpos) const {;
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
  if (gx < 0 || gx >= xsize) return 0;
  if (gy < 0 || gy >= ysize) return 0;
  if (gz < 0 || gz >= zsize) return 0;
  return voxel_value_interpolate(gridpoint[0], gridpoint[1], gridpoint[2]);
}


void VolumetricData::invalidate_gradient() {
  if (gradient) {
    delete [] gradient; 
    gradient = NULL;
    gradient_isvalid = false;
  }
}


const float * VolumetricData::access_volume_gradient() {
  if ((gradient == NULL) || (!gradient_isvalid)) {
    compute_volume_gradient();  
  }
  return gradient;
}


/// set the volume gradient
void VolumetricData::set_volume_gradient(float *grad) {
  if (gradient) {
    delete [] gradient; 
    gradient = NULL;
    gradient_isvalid = false;
  }

  if (!gradient) {
    long gsz = ((long) xsize) * ((long) ysize) * ((long) zsize) * 3L;
    gradient = new float[gsz];
    memcpy(gradient, grad, gsz*sizeof(float));
  }

  gradient_isvalid = true;
}


/// (re)calculate the volume gradient
void VolumetricData::compute_volume_gradient(void) {
  int xi, yi, zi;
  float xs, ys, zs;
  float xl, yl, zl;
  long row;

  if (!gradient) {
    long gsz = this->gridsize() * 3L;
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

      row = (zi * long(xsize) * long(ysize)) + (yi * long(xsize));
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

  gradient_isvalid = true;
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


// Pad each side of the volmap's grid with zeros. 
// Negative padding results in cropping/trimming of the map
void VolumetricData::pad(int padxm, int padxp, int padym, int padyp, int padzm, int padzp) {
  double xdelta[3] = {(xaxis[0] / (xsize - 1)), (xaxis[1] / (xsize - 1)), (xaxis[2] / (xsize - 1))};
  double ydelta[3] = {(yaxis[0] / (ysize - 1)), (yaxis[1] / (ysize - 1)), (yaxis[2] / (ysize - 1))};
  double zdelta[3] = {(zaxis[0] / (zsize - 1)), (zaxis[1] / (zsize - 1)), (zaxis[2] / (zsize - 1))};

  int xsize_new = MAX(1, xsize + padxm + padxp);
  int ysize_new = MAX(1, ysize + padym + padyp);
  int zsize_new = MAX(1, zsize + padzm + padzp);
 
  long newgridsize = long(xsize_new)*long(ysize_new)*long(zsize_new);
  float *data_new = new float[newgridsize];
  memset(data_new, 0, newgridsize*sizeof(float));

  int startx = MAX(0, padxm);
  int starty = MAX(0, padym);
  int startz = MAX(0, padzm);
  int endx = MIN(xsize_new, xsize+padxm);
  int endy = MIN(ysize_new, ysize+padym);
  int endz = MIN(zsize_new, zsize+padzm);

  int gx, gy, gz;
  for (gz=startz; gz<endz; gz++) {
    for (gy=starty; gy<endy; gy++) {
      long oldyzaddrminpadxm = (gy-padym)*long(xsize) + 
                               (gz-padzm)*long(xsize)*long(ysize) - padxm;
      long newyzaddr = gy*long(xsize_new) + gz*long(xsize_new)*long(ysize_new);
      for (gx=startx; gx<endx; gx++) {
        data_new[gx + newyzaddr] = data[gx + oldyzaddrminpadxm];
      }
    }
  }

  delete data;         ///< free the original map
  data = data_new;     ///< replace the original map with the new one

  xsize = xsize_new;
  ysize = ysize_new;
  zsize = zsize_new;

  vec_scaled_add(xaxis, padxm+padxp, xdelta);
  vec_scaled_add(yaxis, padym+padyp, ydelta);
  vec_scaled_add(zaxis, padzm+padzp, zdelta);
  
  vec_scaled_add(origin, -padxm, xdelta);
  vec_scaled_add(origin, -padym, ydelta);
  vec_scaled_add(origin, -padzm, zdelta);

  // Since an arbitrary part of the original map may have been
  // cropped out, we have to recompute the min/max voxel values
  // from scratch.
  //
  // If we know that the map was only padded and not cropped,
  // we could avoid this and instead just update datamin/datamax
  // for the new zero-valued voxels that have been added to the edges.
  invalidate_minmax();   // min/max will be computed on next request

  // Both mean and sigma have to be recomputed from scratch when cropping.
  // They could be scaled in the case of padding since we know how
  // many new zero-valued voxels are added.
  // For now we always recompute them.
  invalidate_mean();     // mean will be computed on next request
  invalidate_sigma();    // sigma will be computed on next request

  // force complete destruction, deallocation, allocation, and 
  // recomputation of the gradient since the actual map dimensions
  // have changed and no longer match the old gradient
  invalidate_gradient();
}


// Crop the map based on minmax values given in coordinate space. If
// the 'cropping box' exceeds the map boundaries, the map is padded
// with zeroes. 
void VolumetricData::crop(double crop_minx, double crop_miny, double crop_minz, double crop_maxx, double crop_maxy, double crop_maxz) {
  double xdelta[3] = {(xaxis[0] / (xsize - 1)), (xaxis[1] / (xsize - 1)), (xaxis[2] / (xsize - 1))};
  double ydelta[3] = {(yaxis[0] / (ysize - 1)), (yaxis[1] / (ysize - 1)), (yaxis[2] / (ysize - 1))};
  double zdelta[3] = {(zaxis[0] / (zsize - 1)), (zaxis[1] / (zsize - 1)), (zaxis[2] / (zsize - 1))};

  // Right now, only works for orthogonal cells.
  int padxm = int((origin[0] - crop_minx)/xdelta[0]);
  int padym = int((origin[1] - crop_miny)/ydelta[1]);
  int padzm = int((origin[2] - crop_minz)/zdelta[2]);

  int padxp = int((crop_maxx - origin[0] - xaxis[0])/xdelta[0]);
  int padyp = int((crop_maxy - origin[1] - yaxis[1])/ydelta[1]);
  int padzp = int((crop_maxz - origin[2] - zaxis[2])/zdelta[2]);

  // the pad() method will update datain/datamax for us
  pad(padxm, padxp, padym, padyp, padzm, padzp);
}


void VolumetricData::clamp(float min_value, float max_value) {
  // clamp the voxel values themselves
  for (long i=0; i<gridsize(); i++) {
    if (data[i] < min_value) data[i] = min_value;
    else if (data[i] > max_value) data[i] = max_value;
  }

  // update datamin/datamax as a result of the value clamping operation
  if (cached_min < min_value)
    cached_min = min_value;

  if (cached_max > max_value)
    cached_max = max_value;

  invalidate_mean();     // mean will be computed on next request
  invalidate_sigma();    // sigma will be computed on next request

  // force regeneration of volume gradient to match clamped voxels
  invalidate_gradient();
}


void VolumetricData::scale_by(float ff) {
  for (long i=0; i<gridsize(); i++) {
    data[i] *= ff; 
  }

  // update min/max as a result of the value scaling operation
  if (ff >= 0.0) {
    cached_min *= ff;
    cached_max *= ff;
  } else {
    // max/min are swapped when negative scale factors are applied
    float tmp = cached_min;
    cached_min = cached_max * ff;
    cached_max = tmp * ff;
  }

  invalidate_mean();     // mean will be computed on next request
  invalidate_sigma();    // sigma will be computed on next request

  // force regeneration of volume gradient to match scaled voxels
  invalidate_gradient();
}


void VolumetricData::scalar_add(float ff) {
  for (long i=0; i<gridsize(); i++){
    data[i] += ff;
  }

  // update datamin/datamax as a result of the scalar addition operation
  cached_min += ff;
  cached_max += ff;

  invalidate_mean();     // mean will be computed on next request
  invalidate_sigma();    // sigma will be computed on next request

  // adding a scalar has no impact on the volume gradient
  // so we leave the existing gradient in place, unmodified 
}


void VolumetricData::rescale_voxel_value_range(float min_val, float max_val) {
  float newvrange = max_val - min_val;
  float oldvrange = cached_max - cached_min;
  float vrangeratio = newvrange / oldvrange;
  for (long i=0; i<gridsize(); i++) {
    data[i] = min_val + vrangeratio*(data[i] - cached_min);
  }

  // update min/max as a result of the rescaling operation
  cached_min = min_val;
  cached_max = max_val;

  invalidate_mean();     // mean will be computed on next request
  invalidate_sigma();    // sigma will be computed on next request

  // force regeneration of volume gradient to match scaled voxels
  invalidate_gradient();
}


// Downsample grid size by factor of 2
// Changed from volutil so it doesn't support PMF; unclear
// whether that is still important.
void VolumetricData::downsample() {
  int xsize_new = xsize/2;
  int ysize_new = ysize/2;
  int zsize_new = zsize/2;
  float *data_new = new float[(long)xsize_new*(long)ysize_new*(long)zsize_new];
  
  int index_shift[8] = {0, 1, xsize, xsize+1, xsize*ysize, xsize*ysize + 1, xsize*ysize + xsize, xsize*ysize + xsize + 1};
  
  int gx, gy, gz, j;
  const double eighth = 1.0/8.0;
  for (gz=0; gz<zsize_new; gz++) {
    for (gy=0; gy<ysize_new; gy++) {
      long oldyzaddr = gy*long(xsize) + gz*long(xsize)*long(ysize);
      long newyzaddr = gy*long(xsize_new) + gz*long(xsize_new)*long(ysize_new);
      for (gx=0; gx<xsize_new; gx++) {
        int n = 2*(gx + oldyzaddr);
        int n_new = gx + newyzaddr;
        double Z=0.0;
        for (j=0; j<8; j++) 
          Z += data[n+index_shift[j]];

        data_new[n_new] = Z * eighth;
      }
    }
  } 

  xsize = xsize_new;
  ysize = ysize_new;
  zsize = zsize_new;

  double xscale = 0.5*(xsize)/(xsize/2);
  double yscale = 0.5*(ysize)/(ysize/2);
  double zscale = 0.5*(zsize)/(zsize/2);
  int i;
  for (i=0; i<3; i++) {
    xaxis[i] *= xscale; 
    yaxis[i] *= yscale; 
    zaxis[i] *= zscale; 
  } 
      
  delete[] data;
  data = data_new;

  invalidate_minmax();   // min/max will be computed on next request
  invalidate_mean();     // mean will be computed on next request
  invalidate_sigma();    // sigma will be computed on next request

  // force regeneration of volume gradient to match resampled voxels
  invalidate_gradient();
}


// Supersample grid size by factor of 2
void VolumetricData::supersample() {
  int gx, gy, gz;
  int xsize_new = xsize*2-1;
  int ysize_new = ysize*2-1;
  int zsize_new = zsize*2-1;
  long xysize = long(xsize)*long(ysize);
  long xysize_new = long(xsize_new)*long(ysize_new);
  float *data_new = new float[long(xsize_new)*long(ysize_new)*long(zsize_new)];
  
  // Copy original voxels to the matching voxels in the new finer grid
  for (gz=0; gz<zsize; gz++) {
    for (gy=0; gy<ysize; gy++) {
      long oldyzplane = long(gy)*long(xsize) + long(gz)*xysize;
      long newyzplane = 2L*long(gy)*long(xsize_new) + 2L*long(gz)*xysize_new;

      // this only copies the matching (even) voxels over, the
      // odd ones must be added through interpolation later
      for (gx=0; gx<xsize; gx++) {
        data_new[2*gx + newyzplane] = data[gx + oldyzplane];
      }
    }
  }

  // Perform cubic interpolation for the rest of the (odd) voxels had not
  // yet been assigned above...

  // x direction
  for (gz=0; gz<zsize; gz++) {
    for (gy=0; gy<ysize; gy++) {
      long newyzplane = 2L*long(gy)*long(xsize_new) + 2L*long(gz)*xysize_new;
      for (gx=1; gx<xsize-2; gx++) {

        // compute odd voxels through cubic interpolation
        data_new[2*gx+1 + newyzplane] = 
          cubic_interp(data_new[(2*gx-2) + newyzplane],
                       data_new[(2*gx)   + newyzplane],
                       data_new[(2*gx+2) + newyzplane],
                       data_new[(2*gx+4) + newyzplane],
                       0.5);
      }
    }
  }

  // borders
  for (gz=0; gz<zsize; gz++) {
    for (gy=0; gy<ysize; gy++) {
      long newyzplane = 2L*long(gy)*long(xsize_new) + 2L*long(gz)*xysize_new;

      // compute odd voxels through cubic interpolation
      // gx = 0
      data_new[1 + newyzplane] = 
        cubic_interp(data_new[0 + newyzplane],
                     data_new[0 + newyzplane],
                     data_new[2 + newyzplane],
                     data_new[4 + newyzplane],
                     0.5);

      // gx = xsize-2
      data_new[2*(xsize-2)+1 + newyzplane] = 
        cubic_interp(data_new[2*(xsize-2)-2 + newyzplane],
                     data_new[2*(xsize-2)   + newyzplane],
                     data_new[2*(xsize-2)+2 + newyzplane],
                     data_new[2*(xsize-2)+2 + newyzplane],
                     0.5);
    }
  }

  // y direction
  for (gz=0; gz<zsize; gz++) {
    for (gy=1; gy<ysize-2; gy++) {
      long newzplane = 2L*long(gz)*xysize_new;
      for (gx=0; gx<xsize_new; gx++) {
        data_new[gx + (2*gy+1)*xsize_new + 2*gz*xysize_new] = 
          cubic_interp(data_new[gx + (2*gy-2)*long(xsize_new) + newzplane],
                       data_new[gx + (2*gy  )*long(xsize_new) + newzplane],
                       data_new[gx + (2*gy+2)*long(xsize_new) + newzplane],
                       data_new[gx + (2*gy+4)*long(xsize_new) + newzplane],
                       0.5);
      }
    }
  }

  // borders
  for (gz=0; gz<zsize; gz++) {
    long newzplane = 2L*long(gz)*xysize_new;
    for (gx=0; gx<xsize_new; gx++) {
      // gy = 0
      data_new[gx + 1*xsize_new + newzplane] = \
        cubic_interp(data_new[gx + 0*xsize_new + newzplane],
                     data_new[gx + 0*xsize_new + newzplane],
                     data_new[gx + 2*xsize_new + newzplane],
                     data_new[gx + 4*xsize_new + newzplane],
                     0.5);

      // gy = ysize-2
      data_new[gx + (2*(ysize-2)+1)*xsize_new + 2*gz*xysize_new] = \
        cubic_interp(data_new[gx + (2*(ysize-2)-2)*xsize_new + newzplane],
                     data_new[gx +  2*(ysize-2)*xsize_new    + newzplane],
                     data_new[gx + (2*(ysize-2)+2)*xsize_new + newzplane],
                     data_new[gx + (2*(ysize-2)+2)*xsize_new + newzplane],
                     0.5);
    }
  }

  // z direction
  for (gy=0; gy<ysize_new; gy++) {
    for (gz=1; gz<zsize-2; gz++) {
      long newyzplane = gy*long(xsize_new) + (2L*gz+1L)*long(xysize_new);
      for (gx=0; gx<xsize_new; gx++) {
        long newxplusyrow = gx + gy*long(xsize_new);
        data_new[gx + newyzplane] = 
          cubic_interp(data_new[newxplusyrow + (2*gz-2)*xysize_new],
                       data_new[newxplusyrow + (2*gz  )*xysize_new],
                       data_new[newxplusyrow + (2*gz+2)*xysize_new],
                       data_new[newxplusyrow + (2*gz+4)*xysize_new],
                       0.5);
      }
    }
  }

  // borders
  for (gy=0; gy<ysize_new; gy++) {
    for (gx=0; gx<xsize_new; gx++) {
      long newxplusyrow = gx + gy*long(xsize_new);

      // gz = 0
      data_new[gx + gy*xsize_new + 1*xysize_new] = \
        cubic_interp(data_new[newxplusyrow + 0*xysize_new],
                     data_new[newxplusyrow + 0*xysize_new],
                     data_new[newxplusyrow + 2*xysize_new],
                     data_new[newxplusyrow + 4*xysize_new],
                     0.5);
      // gz = zsize-2
      data_new[gx + gy*xsize_new + (2*(zsize-2)+1)*xysize_new] = \
        cubic_interp(data_new[newxplusyrow + (2*(zsize-2)-2)*xysize_new],
                     data_new[newxplusyrow +  2*(zsize-2)*xysize_new],
                     data_new[newxplusyrow + (2*(zsize-2)+2)*xysize_new],
                     data_new[newxplusyrow + (2*(zsize-2)+2)*xysize_new],
                     0.5);
    }
  }

  xsize = xsize_new;
  ysize = ysize_new;
  zsize = zsize_new;

  delete[] data;
  data = data_new;

  // XXX in principle we might expect that supersampling a map
  //     would have no impact on the min/max and only a small 
  //     affect on mean/sigma, but for paranoia's sake we will 
  //     recompute them all for the time being...
  invalidate_minmax();   // min/max will be computed on next request
  invalidate_mean();     // mean will be computed on next request
  invalidate_sigma();    // sigma will be computed on next request

  // force regeneration of volume gradient to match resampled voxels
  invalidate_gradient();
}


// Transform map to a sigma scale, so that isovalues in VMD correspond
// to number of sigmas above the mean
void VolumetricData::sigma_scale() {
  float oldmean = mean();
  float oldsigma = sigma();
  float oldsigma_inv = 1.0f / oldsigma;

  for (long i=0; i<gridsize(); i++) {
    data[i] -= oldmean;
    data[i] *= oldsigma_inv;
  }

  invalidate_minmax();   // min/max will be computed on next request
  invalidate_mean();     // mean will be computed on next request
  invalidate_sigma();    // sigma will be computed on next request

  // force regeneration of volume gradient to match rescaled voxels
  invalidate_gradient();
}


// Make a binary mask map for values above a threshold
void VolumetricData::binmask(float threshold) {
  for (long i=0; i<gridsize(); i++) {
    float tmp=data[i];
    data[i] = (tmp > threshold) ? 1 : 0;
  }

  invalidate_minmax();   // min/max will be computed on next request
  invalidate_mean();     // mean will be computed on next request
  invalidate_sigma();    // sigma will be computed on next request

  // force regeneration of volume gradient to match masked voxels
  invalidate_gradient();
}


//
// XXX will this work for an arbitrarily large sigma???
// XXX does it need to?
// XXX needs to be fixed+tested for multi-billion voxel maps
//
void VolumetricData::gaussian_blur(double sigma) {
/*#if defined(VMDCUDA)
  bool cuda = true;
#else 
  bool cuda = false;
#endif
*/
  bool cuda = false;
  GaussianBlur<float>* gaussian_f = new GaussianBlur<float>(data, xsize, ysize, zsize, cuda);
  gaussian_f->blur(sigma);
  
  delete[] data;
  data = gaussian_f->get_image();

  invalidate_minmax();   // min/max will be computed on next request
  invalidate_mean();     // mean will be computed on next request
  invalidate_sigma();    // sigma will be computed on next request

  // force regeneration of volume gradient to match blurred voxels
  invalidate_gradient();
}

void VolumetricData::mdff_potential(double threshold) {
  //calculate new max
  float threshinvert = threshold*-1;
  //calculate new min
  float mininvert = (cached_max*-1);
  //range of the thresholded and inverted (scaleby -1) data
  float oldvrange = threshinvert - mininvert;
  float vrangeratio = 1 / oldvrange;
  for (long i=0; i<gridsize(); i++) {
    // clamp the voxel values themselves
    if (data[i] < threshold) data[i] = threshold;
    //scaleby -1
    data[i] = data[i]*-1;
    //rescale voxel value range between 0 and 1
    data[i] = vrangeratio*(data[i] - mininvert);
  }
  
  // update min/max as a result of the rescaling operation
  cached_min = 0;
  cached_max = 1;

  invalidate_mean();     // mean will be computed on next request
  invalidate_sigma();    // sigma will be computed on next request

  // force regeneration of volume gradient to match scaled voxels
  invalidate_gradient();
}


void VolumetricData::invalidate_minmax() {
  minmax_isvalid = false;
  cached_min = 0;
  cached_max = 0;
}


void VolumetricData::datarange(float &min, float &max) {
  if (!minmax_isvalid && !mean_isvalid) {
    compute_minmaxmean(); // min/max/mean all need updating
  } else if (!minmax_isvalid) {
    compute_minmax(); // only min/max need updating
  }
  
  min = cached_min; 
  max = cached_max; 
}


void VolumetricData::compute_minmaxmean() {
  cached_min = 0;
  cached_max = 0;
  cached_mean = 0;

  long ndata = gridsize();
  if (ndata > 0) {
    // use fast 16-byte-aligned min/max/mean routine
    minmaxmean_1fv_aligned(data, ndata, &cached_min, &cached_max, &cached_mean);
  }

  mean_isvalid = true;
  minmax_isvalid = true;
}


void VolumetricData::compute_minmax() {
  cached_min = 0;
  cached_max = 0;

  long ndata = gridsize();
  if (ndata > 0) {
    // use fast 16-byte-aligned min/max routine
    minmax_1fv_aligned(data, ndata, &cached_min, &cached_max);
  }

  minmax_isvalid = true;
}



float VolumetricData::mean() {
  if (!mean_isvalid && !minmax_isvalid) {
    compute_minmaxmean(); // min/max/mean all need updating
  } else if (!mean_isvalid) {
    compute_mean(); // only mean requires updating
  }

  return cached_mean;
}


void VolumetricData::invalidate_mean() {
  mean_isvalid = false;
  cached_mean = 0;
}


float VolumetricData::sigma() {
  if (!sigma_isvalid)
    compute_sigma();

  return cached_sigma;
}


void VolumetricData::invalidate_sigma() {
  sigma_isvalid = false;
  cached_sigma = 0;
}


void VolumetricData::compute_mean() {
  double mean=0.0;
  long sz = gridsize();

  // XXX This is a slow and lazy implementation that
  //     loses precision if we get large magnitude
  //     values early-on. 
  //     If there is a single NaN value amidst the data
  //     the returned mean will be a NaN.  
  for (long i=0; i<sz; i++) 
    mean += data[i];
  mean /= sz;
  cached_mean = mean;

  mean_isvalid = true;
}


void VolumetricData::compute_sigma() {
  double sigma = 0.0;
  long sz = gridsize();
  float mymean = mean();

  // XXX This is a slow and lazy implementation that
  //     loses precision if we get large magnitude
  //     values early-on. 
  //     If there is a single NaN value amidst the data
  //     the returned mean will be a NaN.  
  for (long i=0; i<sz; i++) {
    float delta = data[i] - mymean;
    sigma += delta*delta;
  }

  sigma /= sz;
  sigma = sqrt(sigma);
  cached_sigma = sigma;

  sigma_isvalid = true;
}


