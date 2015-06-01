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
 *	$RCSfile: unary_ops.C,v $
 *	$Author: ltrabuco $	$Locker:  $		$State: Exp $
 *	$Revision: 1.4 $	$Date: 2009/11/08 20:49:22 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * volmap.C - This file contains the initialization and manipulation 
 * routines for the VolMap class.
 *
 ***************************************************************************/

#include <stdio.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <stdlib.h>

#include "volmap.h"
#include "vec.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Cubic interpolation used by VolMap::supersample
inline float cubic_interp(float y0, float y1, float y2, float y3, float mu) {
  
  float mu2 = mu*mu;
  float a0 = y3 - y2 - y0 + y1;
  float a1 = y0 - y1 - a0;
  float a2 = y2 - y0;
  float a3 = y1;

  return (a0*mu*mu2+a1*mu2+a2*mu+a3);

} 


/* UNARY OPERATIONS */


// Convert from PMF to density (exp(-PMF))
void VolMap::convert_pmf_to_density() {  
  printf("%s :: convert from PMF to density, i.e. exp(-X)\n", get_refname());
  int n;
  int gridsize = xsize*ysize*zsize;
  for (n=0; n<gridsize; n++) data[n] = exp(-data[n]);
}



// Convert from density to PMF (-log(dens))
void VolMap::convert_density_to_pmf() {  
  printf("%s :: convert from density to PMF, i.e. -log(X)\n", get_refname());
  int n;
  int gridsize = xsize*ysize*zsize;
  for (n=0; n<gridsize; n++) {
    if (data[n] < 0.000001)
      data[n] = -log(data[n]);
      if (data[n]>150.f || data[n] != data[n]) data[n] = 150.f;
    else
      data[n] = 150.f;
  }
}



// Pad each side of the volmap's grid with zeros. Negative padding results
// in a trimming of the map
void VolMap::pad(int padxm, int padxp, int padym, int padyp, int padzm, int padzp) {
  printf("%s :: pad by x:%d %d y:%d %d z:%d %d\n", get_refname(), padxm, padxp, padym, padyp, padzm, padzp);

  int xsize_new = MAX(1, xsize + padxm + padxp);
  int ysize_new = MAX(1, ysize + padym + padyp);
  int zsize_new = MAX(1, zsize + padzm + padzp);
  
  int gridsize = xsize_new*ysize_new*zsize_new;
  float *data_new = new float[gridsize];
  memset(data_new, 0, gridsize*sizeof(float));

  int startx = MAX(0, padxm);
  int starty = MAX(0, padym);
  int startz = MAX(0, padzm);
  int endx = MIN(xsize_new, xsize+padxm);
  int endy = MIN(ysize_new, ysize+padym);
  int endz = MIN(zsize_new, zsize+padzm);

  int gx, gy, gz;
  for (gx=startx; gx<endx; gx++)
  for (gy=starty; gy<endy; gy++)
  for (gz=startz; gz<endz; gz++)
    data_new[gx + gy*xsize_new + gz*xsize_new*ysize_new] = data[(gx-padxm) + (gy-padym)*xsize + (gz-padzm)*xsize*ysize];

  delete data;
  data = data_new;

//  double scaling_factor = (double) xsize_new/xsize;
//  vscale(xaxis, scaling_factor);
//  scaling_factor = (double) ysize_new/ysize;
//  vscale(yaxis, scaling_factor);
//  scaling_factor = (double) zsize_new/zsize;
//  vscale(zaxis, scaling_factor);

  xsize = xsize_new;
  ysize = ysize_new;
  zsize = zsize_new;

  vaddscaledto(xaxis, padxm+padxp, xdelta);
  vaddscaledto(yaxis, padym+padyp, ydelta);
  vaddscaledto(zaxis, padzm+padzp, zdelta);
  
  vaddscaledto(origin, -padxm, xdelta);
  vaddscaledto(origin, -padym, ydelta);
  vaddscaledto(origin, -padzm, zdelta);
}

// Crop the map based on minmax values given in coordinate space. If
// the 'cropping box' exceeds the map boundaries, the map is padded
// with zeroes. 
void VolMap::crop(double crop_minx, double crop_miny, double crop_minz, double crop_maxx, double crop_maxy, double crop_maxz) {
  printf("%s :: crop using minmax %g %g %g %g %g %g\n", get_refname(), crop_minx, crop_miny, crop_minz, crop_maxx, crop_maxy, crop_maxz);

  // Right now, only works for orthogonal cells
  require("Crop", REQUIRE_ORTHO);

  int padxm = int((origin[0] - crop_minx)/xdelta[0]);
  int padym = int((origin[1] - crop_miny)/ydelta[1]);
  int padzm = int((origin[2] - crop_minz)/zdelta[2]);

  int padxp = int((crop_maxx - origin[0] - xaxis[0])/xdelta[0]);
  int padyp = int((crop_maxy - origin[1] - yaxis[1])/ydelta[1]);
  int padzp = int((crop_maxz - origin[2] - zaxis[2])/zdelta[2]);

  pad(padxm, padxp, padym, padyp, padzm, padzp);

}


// Downsample grid size by factor of 2
void VolMap::downsample(Ops optype) {
  Operation *ops = GetOps(optype);
  printf("%s :: downsample by x2 (%s)\n", get_refname(), ops->name());
  
  int gx, gy, gz, j;
  int xsize_new = xsize/2;
  int ysize_new = ysize/2;
  int zsize_new = zsize/2;
  float *data_new = new float[xsize_new*ysize_new*zsize_new];
  
  int index_shift[8] = {0, 1, xsize, xsize+1, xsize*ysize, xsize*ysize + 1, xsize*ysize + xsize, xsize*ysize + xsize + 1};
  
  for (gx=0; gx<xsize_new; gx++)
  for (gy=0; gy<ysize_new; gy++)
  for (gz=0; gz<zsize_new; gz++) {
    int n_new = gx + gy*xsize_new + gz*xsize_new*ysize_new;
    int n = 2*(gx + gy*xsize + gz*xsize*ysize);
    double Z=0.;
    for (j=0; j<8; j++) Z += ops->ConvertValue(data[n+index_shift[j]]);
    data_new[n_new] = ops->ConvertAverage(Z/8.);
  }
  
  xsize = xsize_new;
  ysize = ysize_new;
  zsize = zsize_new;
  vscale(xdelta, 2.);
  vscale(ydelta, 2.);
  vscale(zdelta, 2.);
  double scaling_factor = 0.5*(xsize)/(xsize/2);
  vscale(xaxis, scaling_factor);
  scaling_factor = 0.5*(ysize)/(ysize/2);
  vscale(yaxis, scaling_factor);
  scaling_factor = 0.5*(zsize)/(zsize/2);
  vscale(zaxis, scaling_factor);
  
//  vaddscaledto(origin, 0.25, xdelta);
//  vaddscaledto(origin, 0.25, ydelta);
//  vaddscaledto(origin, 0.25, zdelta);
      
  delete[] data;
  data = data_new;
}


// Supersample grid size by factor of 2
void VolMap::supersample(Ops optype) {
  Operation *ops = GetOps(optype);
  printf("%s :: supersample by x2 (%s)\n", get_refname(), ops->name());
  
  int gx, gy, gz;
  int xsize_new = xsize*2-1;
  int ysize_new = ysize*2-1;
  int zsize_new = zsize*2-1;
  int xysize = xsize*ysize;
  int xysize_new = xsize_new*ysize_new;
  float *data_new = new float[xsize_new*ysize_new*zsize_new];
  
  // Copy map to the finer grid
  for (gx=0; gx<xsize; gx++)
    for (gy=0; gy<ysize; gy++)
      for (gz=0; gz<zsize; gz++)
        data_new[2*gx + 2*gy*xsize_new + 2*gz*xysize_new] = \
          data[gx + gy*xsize + gz*xysize];

  // Perform cubic interpolation for the rest of the voxels

  // x direction
  for (gx=1; gx<xsize-2; gx++)
    for (gy=0; gy<ysize; gy++)
      for (gz=0; gz<zsize; gz++)
        data_new[2*gx+1 + 2*gy*xsize_new + 2*gz*xysize_new] = \
          cubic_interp(data_new[(2*gx-2) + 2*gy*xsize_new + 2*gz*xysize_new],
                       data_new[(2*gx)   + 2*gy*xsize_new + 2*gz*xysize_new],
                       data_new[(2*gx+2) + 2*gy*xsize_new + 2*gz*xysize_new],
                       data_new[(2*gx+4) + 2*gy*xsize_new + 2*gz*xysize_new],
                       0.5);
  // borders
  for (gy=0; gy<ysize; gy++)
    for (gz=0; gz<zsize; gz++) {
      // gx = 0
      data_new[1 + 2*gy*xsize_new + 2*gz*xysize_new] = \
        cubic_interp(data_new[0 + 2*gy*xsize_new + 2*gz*xysize_new],
                     data_new[0 + 2*gy*xsize_new + 2*gz*xysize_new],
                     data_new[2 + 2*gy*xsize_new + 2*gz*xysize_new],
                     data_new[4 + 2*gy*xsize_new + 2*gz*xysize_new],
                     0.5);
      // gx = xsize-2
      data_new[2*(xsize-2)+1 + 2*gy*xsize_new + 2*gz*xysize_new] = \
        cubic_interp(data_new[2*(xsize-2)-2 + 2*gy*xsize_new + 2*gz*xysize_new],
                     data_new[2*(xsize-2)   + 2*gy*xsize_new + 2*gz*xysize_new],
                     data_new[2*(xsize-2)+2 + 2*gy*xsize_new + 2*gz*xysize_new],
                     data_new[2*(xsize-2)+2 + 2*gy*xsize_new + 2*gz*xysize_new],
                     0.5);
    }

  // y direction
  for (gx=0; gx<xsize_new; gx++)
    for (gy=1; gy<ysize-2; gy++)
      for (gz=0; gz<zsize; gz++)
        data_new[gx + (2*gy+1)*xsize_new + 2*gz*xysize_new] = \
          cubic_interp(data_new[gx + (2*gy-2)*xsize_new + 2*gz*xysize_new],
                       data_new[gx + (2*gy)*xsize_new   + 2*gz*xysize_new],
                       data_new[gx + (2*gy+2)*xsize_new + 2*gz*xysize_new],
                       data_new[gx + (2*gy+4)*xsize_new + 2*gz*xysize_new],
                       0.5);
  // borders
  for (gx=0; gx<xsize_new; gx++)
    for (gz=0; gz<zsize; gz++) {
      // gy = 0
      data_new[gx + 1*xsize_new + 2*gz*xysize_new] = \
        cubic_interp(data_new[gx + 0*xsize_new + 2*gz*xysize_new],
                     data_new[gx + 0*xsize_new + 2*gz*xysize_new],
                     data_new[gx + 2*xsize_new + 2*gz*xysize_new],
                     data_new[gx + 4*xsize_new + 2*gz*xysize_new],
                     0.5);
      // gy = ysize-2
      data_new[gx + (2*(ysize-2)+1)*xsize_new + 2*gz*xysize_new] = \
        cubic_interp(data_new[gx + (2*(ysize-2)-2)*xsize_new + 2*gz*xysize_new],
                     data_new[gx + 2*(ysize-2)*xsize_new     + 2*gz*xysize_new],
                     data_new[gx + (2*(ysize-2)+2)*xsize_new + 2*gz*xysize_new],
                     data_new[gx + (2*(ysize-2)+2)*xsize_new + 2*gz*xysize_new],
                     0.5);
    }

  // z direction
  for (gx=0; gx<xsize_new; gx++)
    for (gy=0; gy<ysize_new; gy++)
      for (gz=1; gz<zsize-2; gz++)
        data_new[gx + gy*xsize_new + (2*gz+1)*xysize_new] = \
          cubic_interp(data_new[gx + gy*xsize_new + (2*gz-2)*xysize_new],
                       data_new[gx + gy*xsize_new + (2*gz)*xysize_new],
                       data_new[gx + gy*xsize_new + (2*gz+2)*xysize_new],
                       data_new[gx + gy*xsize_new + (2*gz+4)*xysize_new],
                       0.5);
  // borders
  for (gx=0; gx<xsize_new; gx++)
    for (gy=0; gy<ysize_new; gy++) {
      // gz = 0
      data_new[gx + gy*xsize_new + 1*xysize_new] = \
        cubic_interp(data_new[gx + gy*xsize_new + 0*xysize_new],
                     data_new[gx + gy*xsize_new + 0*xysize_new],
                     data_new[gx + gy*xsize_new + 2*xysize_new],
                     data_new[gx + gy*xsize_new + 4*xysize_new],
                     0.5);
      // gz = zsize-2
      data_new[gx + gy*xsize_new + (2*(zsize-2)+1)*xysize_new] = \
        cubic_interp(data_new[gx + gy*xsize_new + (2*(zsize-2)-2)*xysize_new],
                     data_new[gx + gy*xsize_new + 2*(zsize-2)*xysize_new],
                     data_new[gx + gy*xsize_new + (2*(zsize-2)+2)*xysize_new],
                     data_new[gx + gy*xsize_new + (2*(zsize-2)+2)*xysize_new],
                     0.5);
  }


  xsize = xsize_new;
  ysize = ysize_new;
  zsize = zsize_new;
  vscale(xdelta, 0.5);
  vscale(ydelta, 0.5);
  vscale(zdelta, 0.5);

  delete[] data;
  data = data_new;
}



void VolMap::collapse_onto_z(Ops optype) {
  Operation *ops = GetOps(optype);
  printf("%s -> computing z-projection (%s) -> \"collapse.dat\"\n", get_refname(), ops->name());
    
  int gx, gy, gz;
  double projection;
  FILE *fout = fopen("collapse.dat", "w");
      
  for (gz=0; gz<zsize; gz++) {
    projection = 0.;
    
    for (gx=0; gx<xsize; gx++)
    for (gy=0; gy<ysize; gy++) {
      projection += ops->ConvertValue(data[gx + gy*xsize + gz*xsize*ysize]);
    }
    
    projection = projection/(xsize*ysize);
    fprintf(fout, "%g %g\n", origin[2]+gz*zdelta[2], ops->ConvertAverage(projection));
  }
   
  fclose(fout);

}



// Average the map over N rotations of itself
void VolMap::average_over_rotations(Ops optype) {
  Operation *ops = GetOps(optype);
  
  // Hard-code the rotation axis, the rotation center, and the number of rotations:
  const double R_center[3] = {0., 0., 0.};
  const double R_axis[3] = {0., 0., 1.};
  const int num_rot = 4;

  double rot_incr = 2.*M_PI/(double) num_rot;
  printf("%s: averaging the PMF over %d rotations (%s)\n", get_refname(), num_rot, ops->name());

  int gridsize = xsize*ysize*zsize;
  float *data_new = new float[gridsize];
  memset(data_new, 0, gridsize*sizeof(float));
  int *data_count = new int[gridsize];
  memset(data_count, 0, gridsize*sizeof(int));
  
  float x, y, z;
  float xo, yo, zo;
  int rot;
  int n;
  for (rot=0; rot<num_rot; rot++) {
    double angle = rot_incr*rot;  
    double cosA = cos(angle);
    double sinA = sin(angle);
    double t = 1.-cosA;
    for (n=0; n<gridsize; n++) {
      index_to_coord(n, xo, yo, zo);
      xo -= R_center[0];
      yo -= R_center[1];
      zo -= R_center[2];
      x = xo*(t*R_axis[0]*R_axis[0] + cosA) + yo*(t*R_axis[0]*R_axis[1]+sinA*R_axis[2]) + zo*(t*R_axis[0]*R_axis[2]-sinA*R_axis[1]) + R_center[0];
      y = xo*(t*R_axis[0]*R_axis[1]-sinA*R_axis[2]) + yo*(t*R_axis[1]*R_axis[1] + cosA) + zo*(t*R_axis[1]*R_axis[2]+sinA*R_axis[0]) + R_center[1];
      z = xo*(t*R_axis[0]*R_axis[2]+sinA*R_axis[1]) + yo*(t*R_axis[1]*R_axis[2]-sinA*R_axis[0]) + zo*(t*R_axis[2]*R_axis[2] + cosA) + R_center[2];
      
      float val = voxel_value_interpolate_from_coord(x, y, z, optype);
      
      if (val == val) {
        data_new[n] += ops->ConvertValue(val);
        data_count[n]++;
      }
    }
  }
  
  
  for (n=0; n<gridsize; n++) {
    float val = ops->ConvertAverage(data_new[n]/data_count[n]);
    
    if (optype == PMF) {
      if (val == val && val < 150.) data_new[n] = val;
      else data_new[n] = 150.;
    }
    else
      data_new[n] = val; 
  }
  
  delete[] data;
  delete[] data_count;
  data = data_new;
}




void VolMap::total_occupancy() {
    
  int gx, gy, gz;
  double val;
  double occup = 0.;
  int count = 0;
  
  for (gz=0; gz<zsize; gz++)
  for (gx=0; gx<xsize; gx++)
  for (gy=0; gy<ysize; gy++) {
    val = data[gx + gy*xsize + gz*xsize*ysize];
    occup += exp(-val);
    if (val) count++;
  }
  
  double factor = 6.0221e23/22.4e27;  // 1mol/22.4L in particles/A^3 at STP
  occup *= factor;
  
  printf("\nCOUNT: At 1atm, the occupancy of the map is: %g particles\n", occup);
  printf("\nCOUNT: In air, the occupancy of the map is: %g O2\n", occup*0.2);
  printf("\nCOUNT: Non-zero cell count is: %d\n", count);
  printf("\nCOUNT: Occupancy in equiv. vacuum would be %g\n", factor*count);
  printf("\nCOUNT: Occupancy in equiv. air would be %g\n", factor*count*0.2);
}
  


void VolMap::clamp(float min_value, float max_value) {
  printf("%s :: clamp [%g .. %g]\n", get_refname(), min_value, max_value);
  int i;
  for (i=0; i<xsize*ysize*zsize; i++) {
    if (data[i] < min_value) data[i] = min_value;
    else if (data[i] > max_value) data[i] = max_value;
  }
}



void VolMap::scale_by(float ff) {

  printf("%s :: scale by %g\n", get_refname(), ff);
  int i;
  for (i=0; i<xsize*ysize*zsize; i++)
    data[i] *= ff;

}

void VolMap::scalar_add(float ff) {

  printf("%s :: add the scalar %g\n", get_refname(), ff);
  int i;
  for (i=0; i<xsize*ysize*zsize; i++)
    data[i] += ff;

}

void VolMap::fit_to_range(float min_value, float max_value) {
  printf("%s :: fit to range [%g .. %g]\n", get_refname(), min_value, max_value);

  float min = data[0];
  float max = data[0];
  int i;
  for (i=1; i<xsize*ysize*zsize; i++) {
    if (data[i] < min) min = data[i];
    if (data[i] > max) max = data[i];
  }
  
  for (i=0; i<xsize*ysize*zsize; i++)
    data[i] = min_value + (max_value - min_value)*(data[i] - min)/(max - min);

}



void VolMap::dock_grid(float max_value) {
  printf("%s :: creating map for grid-based flexible docking with NAMD with max = %g\n", get_refname(), max_value);

  clamp(0., FLT_MAX);
  scale_by(-1.);
  fit_to_range(0., max_value);

}


// Set every voxel below a threshold to NAN
void VolMap::apply_threshold(float min_value) {
  printf("%s :: setting voxels below %g to NAN\n", get_refname(), min_value);

  clamp(min_value, FLT_MAX);
  int i;
  for (i=0; i<xsize*ysize*zsize; i++) 
    if (data[i] == min_value)
      data[i] = kNAN;

}


// Set every voxel below a threshold to NAN
void VolMap::apply_threshold_sigmas(double sigmas) {
  printf("%s :: setting voxels below %g sigmas to NAN\n", get_refname(), sigmas);

  int size = xsize*ysize*zsize;
  double mean = 0.;
  int i;
  for (i=0; i<size; i++)
    mean += data[i];
  mean /= size;

  double sigma = 0.;
  for (i=0; i<size; i++)
    sigma += (data[i] - mean)*(data[i] - mean);
  sigma /= size;
  sigma = sqrt(sigma);

  double threshold = mean + sigmas*sigma;

  apply_threshold(threshold);

}


// Transform map to a sigma scale, so that isovalues in VMD correspond
// to number of sigmas above the mean
void VolMap::sigma_scale() {
  printf("%s :: transforming to sigma scale\n", get_refname());

  int size = xsize*ysize*zsize;
  double mean = 0.;
  int i;
  for (i=0; i<size; i++)
    mean += data[i];
  mean /= size;

  double sigma = 0.;
  for (i=0; i<size; i++)
    sigma += (data[i] - mean)*(data[i] - mean);
  sigma /= size;
  sigma = sqrt(sigma);

  for (i=0; i<size; i++) {
    data[i] -= mean;
    data[i] /= sigma;
  }

}

// Calculates a density histogram
void VolMap::histogram(int nbins) {
  printf("%s :: calculating histogram with %d bins\n", get_refname(), nbins);

  // Calculate minmax
  double min = data[0];
  double max = data[0];

  int i;
  for (i=1; i<xsize*ysize*zsize; i++) {
    if (data[i] < min) min = data[i];
    if (data[i] > max) max = data[i];
  }

  // Calculate the width of each bin
  double binwidth = (max-min)/nbins;

  // Allocate array that will contain the number of voxels in each bin
  int *bins = (int*) malloc(nbins*sizeof(int));
  memset(bins, 0, nbins*sizeof(int));

  // Calculate histogram
  for (i=0; i<xsize*ysize*zsize; i++) 
    bins[int((data[i]-min)/binwidth)]++;

  printf("Density histogram with min = %g, max = %g, nbins = %d:\n", min, max, nbins);
  for (i=0; i<nbins; i++)
    printf("%d ", bins[i]);
  printf("\n");

  free(bins);

}


// Makes a mask out of a map
// i.e. all values > 0 are set to 1
void VolMap::binmask() {
  clamp(0., FLT_MAX);
  int i;
  for (i=0; i<xsize*ysize*zsize; i++) {
    if (data[i] > 0) data[i] = 1;
  }
}

void VolMap::invmask() {
  binmask();
  scale_by(-1);
  scalar_add(1);
}

// Count how many non-NAN voxels we have and multiply by the voxel 
// volume to calculate a "molecular volume"
void VolMap::calc_volume() {
    
  int i;
  int count = 0;
  double voxel_volume;
  double total_volume;
  double vec_tmp[3];
  
  // Count number of non-NAN voxels
  for (i=0; i<xsize*ysize*zsize; i++)
    if (!ISNAN(data[i])) count++;

  // Calculate the voxel volume
  vcross(vec_tmp, ydelta, zdelta);
  voxel_volume = fabs(vdot(xdelta, vec_tmp));
  
  total_volume = count*voxel_volume;

  printf("VOLUME: %g (%d voxels above threshold)\n", total_volume, count);

}
