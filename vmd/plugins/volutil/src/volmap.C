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
 *	$RCSfile: volmap.C,v $
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
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "volmap.h"
#include "vec.h"

#define TOLERANCE 1e-3

const float kNAN = sqrt(-1.f);

/* INITIALIZATION */

void VolMap::init() {
  xsize = 0;
  ysize = 0;
  zsize = 0;
  data = NULL;
  refname = NULL;
  dataname = NULL;
  weight = 1.;
  temperature = -1.;
  conditionbits = 0xFFFF;
}


// Constructor
VolMap::VolMap() {
  init();
}



// Clone Constructor
VolMap::VolMap(const VolMap *src) {
  init();    
  clone(src);
}


// Load Constructor
VolMap::VolMap(const char *filename) {
  init();    
  load(filename);
}



/// Destructor
VolMap::~VolMap() {
  if (refname)  delete[] refname;
  if (dataname) delete[] dataname;
  if (data)     delete[] data;
}



// Clone
void VolMap::clone(const VolMap *src) {
  vcopy (origin, src->origin);
  vcopy (xaxis, src->xaxis);
  vcopy (yaxis, src->yaxis);
  vcopy (zaxis, src->zaxis);
  vcopy (xdelta, src->xdelta);
  vcopy (ydelta, src->ydelta);
  vcopy (zdelta, src->zdelta);
  xsize = src->xsize;
  ysize = src->ysize;
  zsize = src->zsize;  
 
  weight = src->weight;
  temperature = src->temperature;

  // XXX need to copy selection info

  int gridsize = xsize*ysize*zsize;
  if (data) delete[] data;
  data = new float[gridsize];
  memcpy(data, src->data, gridsize*sizeof(float));
  
  set_refname(src->refname);
// XXX - There is a bug in set_dataname
//  set_dataname(src->dataname);
}



/// Zero the data array
void VolMap::zero() {
  int gridsize = xsize*ysize*zsize;
  memset(data, 0, gridsize*sizeof(float));
}


/// Make the map "dirty"
/// Right now only applies to cell conditions, but add more later...
void VolMap::dirty() {
  conditionbits = 0xFFFF;
}


void VolMap::set_refname(const char *newname) {
  if (refname) delete[] refname;
  refname = new char[strlen(newname)+1];
  strcpy(refname, newname);
}

void VolMap::set_dataname(const char *newname) {
  char *s, *noquotes;
  noquotes = new char[strlen(newname)+1];
  strcpy(noquotes, newname);

  // replace double quotes by single quotes
  s = noquotes;
  while((s=strchr(s, '"'))) *s = '\'';

  if (dataname) delete[] dataname;
  dataname = new char[strlen(noquotes)+1];
  strcpy(dataname, noquotes);
}



const char *VolMap::get_refname() const {
  if (refname) return refname;
  return "(no name)";
}


bool VolMap::condition(int cond) {
  
  if (conditionbits == 0xFFFF) { // is dirty, so reinitialize
    conditionbits = 0;
    
    // Check that x/y/z axes are not switched
    if (xaxis[0] && yaxis[1] && zaxis[2]) conditionbits |= REQUIRE_ORDERED;
  
    // Check that volmap cell is orthogonal
    if (fabs(xaxis[1]) < TOLERANCE && fabs(xaxis[2]) < TOLERANCE && 
        fabs(yaxis[0]) < TOLERANCE && fabs(yaxis[2]) < TOLERANCE && 
        fabs(zaxis[0]) < TOLERANCE && fabs(zaxis[1]) < TOLERANCE)
      conditionbits |= REQUIRE_ORTHO;

    // Check that resolution is same in all dimensions
    double xres = xdelta[0]*xdelta[0] + xdelta[1]*xdelta[1] + xdelta[2]*xdelta[2];
    double yres = ydelta[0]*ydelta[0] + ydelta[1]*ydelta[1] + ydelta[2]*ydelta[2];
    double zres = zdelta[0]*zdelta[0] + zdelta[1]*zdelta[1] + zdelta[2]*zdelta[2];
    if ( (fabs(xres-yres) < TOLERANCE) && (fabs(xres-zres) < TOLERANCE) ) 
      conditionbits |= REQUIRE_UNIFORM;
  
    // check that volmap cell is non-singular (i.e. that it is really "3D")
    bool singular = false;
    if (!xaxis[0] && !xaxis[1] && !xaxis[2]) singular = true;
    if (!yaxis[0] && !yaxis[1] && !yaxis[2]) singular = true;
    if (!zaxis[0] && !zaxis[1] && !zaxis[2]) singular = true;
    // XXX Also need to check that there is no colinearity!
    if (!singular) conditionbits |= REQUIRE_NONSINGULAR;
  }
  
  if ((cond & conditionbits) == cond) return true;
  return false;
}



void VolMap::require(const char *funcname, int cond) {
  if ((cond & REQUIRE_ORDERED) && !condition(REQUIRE_ORDERED)) {
    printf("Error: \"%s\" operation requires that the volmap cell axes be ordered!\n", funcname);
    exit(1);    
  }
  
  if ((cond & REQUIRE_ORTHO) && !condition(REQUIRE_ORTHO)) {
    printf("Error: \"%s\" operation requires that the volmap cell be orthogonal!\n", funcname);
    exit(1);    
  }

  if ((cond & REQUIRE_UNIFORM) && !condition(REQUIRE_UNIFORM)) {
    printf("Error: \"%s\" operation requires the same resolution in all dimensions!\n", funcname);
    exit(1);    
  }
  
  if ((cond & REQUIRE_NONSINGULAR) && !condition(REQUIRE_NONSINGULAR)) {
    printf("Error: \"%s\" operation requires that the volmap cell be non-singular!\n", funcname);
    exit(1);    
  }
}



void VolMap::print_stats() {  
  int gx, gy, gz;
  double D, E;

  double sum_D = 0.;  
  double sum_E = 0.; 
  double sum_E2 = 0.;  
  double sum_DE = 0.; 
  double sum_DE2 = 0.; 
  double sum_D2 = 0.; 
  
  double min_E = data[0]; 
  double max_E = data[0];  
  
  for (gx=0; gx<xsize; gx++)
  for (gy=0; gy<ysize; gy++) 
  for (gz=0; gz<zsize; gz++) {
    E = data[gx + gy*xsize + gz*xsize*ysize];
    D = exp(-E);
    sum_D  += D;
    sum_D2 += D*D;
    sum_DE += D*E;
    sum_DE2+= D*E*E;
    sum_E  += E;
    sum_E2 += E*E;

    if (E<min_E) min_E = E;
    if (E>max_E) max_E = E;
  }
  
  double N = xsize*ysize*zsize;
  double pmf = -log(sum_D/N);
  
//  double dev_D = sqrt(sum_D2/N - sum_D*sum_D/(N*N));
  double dev_E = sqrt(sum_E2/N - sum_E*sum_E/(N*N));
  
  printf("\nStats for %s:\n", get_refname());
  printf("  WEIGHT:    %g\n", weight);
  printf("  AVERAGE:   %g\n", sum_E/N);
  printf("  STDEV:     %g\n", dev_E);
  printf("  MIN:       %g\n", min_E);
  printf("  MAX:       %g\n", max_E);
  printf("  PMF_AVG:   %g\n", pmf);
  printf("\n");

}


