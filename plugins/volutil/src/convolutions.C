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
 *	$RCSfile: convolutions.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.3 $	$Date: 2010/01/11 23:17:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * This file contains the initialization and manipulation 
 * routines for the VolMap class.
 *
 ***************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "volmap.h"
#include "vec.h"

/* avoid parameter name collisions with AIX5 "hz" macro */
#undef hz

//Gaussian blurring (as a 3D convolution), but the kernel can easily be changed to something else
void VolMap::convolution_gauss3d(double sigma, unsigned int flagsbits, Ops optype) {
  if (!sigma) return;
  
  //Right now, only works if resolution is the same in all map dimensions
  require("Gaussian blur", REQUIRE_UNIFORM);

  Operation *ops = GetOps(optype);
  printf("%s :: Gaussion blur 3D filter (%s); sigma = %g \n", get_refname(), ops->name(), sigma);
 
  // Pre-divide by sqrt(3) in x/y/z dimensions to get "sigma" in 3D
  sigma /= sqrt(3.);
  
  double delta = xdelta[0];
  int step = (int)(3.*sigma/delta); // size of gaussian convolution
  if (!step) return;
  
  // Build convolution kernel
  int convsize = 2*step+1;
  float *conv = new float[convsize*convsize*convsize];
  memset(conv, 0, convsize*convsize*convsize*sizeof(float));

  // Pad the map if required
  if (flagsbits & USE_PADDING)
    pad(convsize, convsize, convsize, convsize, convsize, convsize);

  int gridsize = xsize*ysize*zsize;
  float *data_new = new float[gridsize];
  memset(data_new, 0, gridsize*sizeof(float));
  
  double r2, norm=0.;
  int cx, cy, cz; 
  for (cz=0; cz<convsize; cz++)
  for (cy=0; cy<convsize; cy++)
  for (cx=0; cx<convsize; cx++) {
    r2 = delta*delta*((cx-step)*(cx-step)+(cy-step)*(cy-step)+(cz-step)*(cz-step));
    conv[cx + cy*convsize + cz*convsize*convsize] = exp(-0.5*r2/(sigma*sigma)); 
    norm += conv[cx + cy*convsize + cz*convsize*convsize];
  }
  
  // Normalize...
  int n;
  for (n=0; n<convsize*convsize*convsize; n++) {
    conv[n] = conv[n]/norm;
  }
 
  // Apply convolution   
  if (!ops->trivial())
    for (n=0; n<gridsize; n++) data[n] = ops->ConvertValue(data[n]);  
  
  int gx, gy, gz, hx, hy, hz; 
  for (gz=0; gz<zsize; gz++)
  for (gy=0; gy<ysize; gy++) 
  for (gx=0; gx<xsize; gx++)
  for (cz=0; cz<convsize; cz++)
  for (cy=0; cy<convsize; cy++)
  for (cx=0; cx<convsize; cx++) {
    hx=gx+cx-step;
    hy=gy+cy-step;
    hz=gz+cz-step;
    if (hx < 0 || hx >= xsize || hy < 0 || hy >= ysize || hz < 0 || hz >= zsize) {
      continue;
    }

    data_new[gx + gy*xsize + gz*xsize*ysize] += data[hx + hy*xsize + hz*xsize*ysize]*conv[cx + cy*convsize + cz*convsize*convsize];  
  }
  
  if (!ops->trivial())
    for (n=0; n<gridsize; n++) data_new[n] = ops->ConvertAverage(data_new[n]);  
  
  delete[] data;
  data = data_new;
}



// Fast Gaussian blur that takes advantage of the fact that the dimensions are separable.
void VolMap::convolution_gauss1d(double sigma, unsigned int flagsbits, Ops optype) {
  if (!sigma) return;

  //Right now, only works if resolution is the same in all map dimensions
  require("Gaussian blur", REQUIRE_UNIFORM);
  
  Operation *ops = GetOps(optype);
  printf("%s :: gaussian blur (sigma = %g, %s)\n", get_refname(), sigma, ops->name());
  
  // Pre-divide by sqrt(3) in x/y/z dimensions to get "sigma" in 3D
  sigma /= sqrt(3.);
  
  double delta = xdelta[0];
  int step = (int)(3.*sigma/delta); // size of gaussian convolution
  if (!step) return;

  // Build convolution kernel
  int convsize = 2*step+1;
  float *conv = new float[convsize];
  memset(conv, 0, convsize*sizeof(float));

  // Pad the map if required
  if (flagsbits & USE_PADDING)
    pad(convsize, convsize, convsize, convsize, convsize, convsize);

  int gridsize = xsize*ysize*zsize;
  float *data_new = new float[gridsize];

  double r2, norm=0.;
  int c;
  for (c=0; c<convsize; c++) {
    r2 = delta*delta*(c-step)*(c-step);
    conv[c] = (float) exp(-0.5*r2/(sigma*sigma)); 
    norm += conv[c];
  }
  
  // Normalize...

  for (c=0; c<convsize; c++) {
    conv[c] = conv[c]/norm;
  }
 
  // Apply convolution   
  int n;
  if (!ops->trivial())
    for (n=0; n<gridsize; n++) data[n] = ops->ConvertValue(data[n]);  

  memset(data_new, 0, gridsize*sizeof(float));
  
  int gx, gy, gz, hx, hy, hz; 
  for (gz=0; gz<zsize; gz++)
  for (gy=0; gy<ysize; gy++)
  for (gx=0; gx<xsize; gx++)
  for (c=0; c<convsize; c++) {
    hx=gx+c-step;
    hy=gy;
    hz=gz;
    if (hx < 0 || hx >= xsize) continue;
    data_new[gx + gy*xsize + gz*xsize*ysize] += data[hx + hy*xsize + hz*xsize*ysize]*conv[c];
  }

  float *dataswap = data;
  data = data_new;
  data_new = dataswap;
  memset(data_new, 0, gridsize*sizeof(float));

  for (gz=0; gz<zsize; gz++)
  for (gy=0; gy<ysize; gy++)
  for (gx=0; gx<xsize; gx++)
  for (c=0; c<convsize; c++) {
    hx=gx;
    hy=gy+c-step;
    hz=gz;
    if (hy < 0 || hy >= ysize) continue;
    data_new[gx + gy*xsize + gz*xsize*ysize] += data[hx + hy*xsize + hz*xsize*ysize]*conv[c];
  }
    
  dataswap = data;
  data = data_new;
  data_new = dataswap;
  memset(data_new, 0, gridsize*sizeof(float));
  
  for (gz=0; gz<zsize; gz++)
  for (gy=0; gy<ysize; gy++)
  for (gx=0; gx<xsize; gx++)
  for (c=0; c<convsize; c++) {
    hx=gx;
    hy=gy;
    hz=gz+c-step;
    if (hz < 0 || hz >= zsize) continue;
    data_new[gx + gy*xsize + gz*xsize*ysize] += data[hx + hy*xsize + hz*xsize*ysize]*conv[c];
  }

  if (!ops->trivial())
    for (n=0; n<gridsize; n++) data_new[n] = ops->ConvertAverage(data_new[n]);
  
  delete[] data;
  data = data_new;
}












