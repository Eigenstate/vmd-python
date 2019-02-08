/***************************************************************************
 *cr
 *cr		(C) Copyright 1995-2019 The Board of Trustees of the
 *cr			    University of Illinois
 *cr			     All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile: STLDisplayDevice.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.40 $	$Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 *   Render to a STL (Stereo-Lithography file).  
 *   See http://www.sdsc.edu/tmf/ for more information on the file format and
 *   how to make a physical 3-D model from VMD.
 *
 ***************************************************************************/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "STLDisplayDevice.h"
#include "Matrix4.h"
#include "DispCmds.h"
#include "Inform.h"
#include "utilities.h"

// constructor ... call the parent with the right values
STLDisplayDevice::STLDisplayDevice(void) 
: FileRenderer("STL", "STL (triangle mesh only)", "vmdscene.stl", "true") { }

// destructor
STLDisplayDevice::~STLDisplayDevice(void) { }

void STLDisplayDevice::triangle(const float *v1, const float *v2, const float *v3, 
                                const float *n1, const float *n2, const float *n3) {
  float a[3], b[3], c[3];
  float norm1[3], norm2[3], norm3[3];

  // transform the world coordinates
  (transMat.top()).multpoint3d(v1, a);
  (transMat.top()).multpoint3d(v2, b);
  (transMat.top()).multpoint3d(v3, c);

  // and the normals
  (transMat.top()).multnorm3d(n1, norm1);
  (transMat.top()).multnorm3d(n2, norm2);
  (transMat.top()).multnorm3d(n3, norm3);
                                                       
  // draw the triangle 
#if 1
  // do not calculate surface normals, return a 0 vector.
  fprintf(outfile, "  facet normal 0.0 0.0 0.0\n");
#else
  // calculate surface normals for each triangle.
  float nx, ny, nz, n;
  nx = a[1]*(b[2]-c[2])+b[1]*(c[2]-a[2])+c[1]*(a[2]-b[2]);
  ny = a[2]*(b[0]-c[0])+b[2]*(c[0]-a[0])+c[2]*(a[0]-b[0]);
  nz = a[0]*(b[1]-c[1])+b[0]*(c[1]-a[1])+c[0]*(a[1]-b[1]);
  n = nx*nx+ny*ny+nz*nz;
  n = sqrt(n);
  nx /= n; ny /= n; nz /= n;
  fprintf (outfile, "  facet normal %f %f %f\n", nx, ny, nz);
#endif

  fprintf(outfile,"     outer loop\n");
  fprintf(outfile,"       vertex %f %f %f\n", a[0], a[1], a[2]);
  fprintf(outfile,"       vertex %f %f %f\n", b[0], b[1], b[2]);
  fprintf(outfile,"       vertex %f %f %f\n", c[0], c[1], c[2]);
  fprintf(outfile,"     endloop\n");
  fprintf(outfile,"  endfacet\n");
}

void STLDisplayDevice::write_header (void) {
  fprintf (outfile, "solid molecule\n");
}

void STLDisplayDevice::write_trailer (void) {
  fprintf (outfile, "endsolid\n");
  msgWarn << "Only triangles in the present scene have been processed.\n";
  msgWarn << "Materials and colors are not exported to STL files.\n";
}

