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
 *	$RCSfile: VrmlDisplayDevice.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.40 $	$Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 ***************************************************************************/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>  /* this is for the Hash Table */

#include "VrmlDisplayDevice.h"
#include "Matrix4.h"
#include "DispCmds.h"
#include "Inform.h"
#include "utilities.h"

#define DEFAULT_RADIUS 0.002
#define DASH_LENGTH 0.02

///////////////////////// constructor and destructor

// constructor ... initialize some variables
VrmlDisplayDevice::VrmlDisplayDevice(void) : 
  FileRenderer("VRML-1", "VRML 1.0 (VRML94)", "vmdscene.wrl", "true") {
	
  tList = NULL;
}
               
///////////////////////// protected nonvirtual routines
void VrmlDisplayDevice::set_color(int mycolorIndex) {
  write_cindexmaterial(mycolorIndex, materialIndex);
}

// draw a sphere
void VrmlDisplayDevice::sphere(float *xyzr) {
  
  // draw the sphere
  fprintf(outfile, "Separator {\nTranslation { translation %f %f %f }\n",
	  xyzr[0], xyzr[1], xyzr[2]);

  fprintf(outfile, "Sphere { radius %f }\n}\n", xyzr[3]);
}

//// draw a line (cylinder) from a to b
////  Doesn't yet support the dotted line method
void VrmlDisplayDevice::line(float *a, float*b) {
  // ugly and wasteful, but it will work
  // I guess I really should have done the "correct" indexing ...
  // I'll worry about that later.
  fprintf(outfile, "Coordinate3 {point [ %f %f %f, %f %f %f ] }\n",
	  a[0], a[1], a[2], b[0], b[1], b[2]);
  fprintf(outfile, "IndexedLineSet { coordIndex [0, 1, -1] }\n");
}

static Matrix4 convert_endpoints_to_matrix(float *a, float *b) {
  // I need to transform (0,-1,0) - (0,1,0) radius 1
  // into a - b
  float c[3];
  vec_sub(c, b, a);
  
  float len = distance(a,b);
  if (len == 0.0) {
    return Matrix4();
  }
  
  Matrix4 trans;
  trans.translate( (a[0] + b[0]) / 2.0f, (a[1] + b[1]) / 2.0f, 
		   (a[2] + b[2]) / 2.0f);

  // this is a straight copy from transvec, which takes things
  // from the x axis to a given vector
  Matrix4 rot;
  if (c[0] == 0.0 && c[1] == 0.0) {
    // then wants the z axis
    if (c[2] > 0) {
      rot.rot(-90, 'y');
    } else {
      rot.rot(-90, 'y');
    }
  } else {
    float theta, phi;
    theta = (float) atan2(c[1], c[0]);
    phi = (float) atan2(c[2], sqrt(c[0]*c[0] + c[1]*c[1]));
    Matrix4 m1; m1.rot( (float) (-phi  * 180.0f / VMD_PI), 'y');
    rot.rot( (float) (theta * 180.0f / VMD_PI), 'z');
    rot.multmatrix(m1); 
  }

  // Compute everything
  Matrix4 mat(trans);
  mat.multmatrix(rot);

  // now bring to the y axis
  mat.rot(-90, 'z');

  return mat;
}

// draw a cylinder
void VrmlDisplayDevice::cylinder(float *a, float *b, float r, int filled) {
  if (a[0] == b[0] && a[1] == b[1] && a[2] == b[2]) {
    return;  // we don't serve your kind here
  }

  push();
  Matrix4 mat = convert_endpoints_to_matrix(a, b);
  load(mat); // print matrix

  // draw the cylinder
  fprintf(outfile, "Cylinder { \n"
	  "radius %f\n"
	  "height %f\n"
	  "parts %s}\n", 
	  r, distance(a,b),  filled ? "ALL" : "SIDES");

  // pop out of this matrix
  pop();
}


// draw a cone
void VrmlDisplayDevice::cone(float *a, float *b, float r, int /* resolution */) {
  if (a[0] == b[0] && a[1] == b[1] && a[2] == b[2]) {
    return;  // we don't serve your kind here
  }

  push();
  Matrix4 mat = convert_endpoints_to_matrix(a, b);
  load(mat); // print matrix

  // draw the cylinder
  fprintf(outfile, "Cone { \n"
	  "bottomRadius %f\n"
	  "height %f}\n",
	  r, distance(a,b));

  // pop out of this matrix
  pop();
}


// draw a triangle
void VrmlDisplayDevice::triangle(const float *a, const float *b, const float *c, 
				 const float *n1, const float *n2, const float *n3) {
  fprintf(outfile,
	  "Normal { vector [\n"
	  " %f %f %f,\n %f %f %f,\n %f %f %f\n"
	  " ] }\n",
	  n1[0], n1[1], n1[2], n2[0], n2[1], n2[2], 
	  n3[0], n3[1], n3[2]);
  fprintf(outfile, 
	  "Coordinate3 {point [\n"
	  " %f %f %f,\n"
	  " %f %f %f,\n"
	  " %f %f %f\n"
	  " ] }\n",
	  a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
  fprintf(outfile, "IndexedFaceSet { coordIndex [0, 1, 2, -1] }\n");
}

void VrmlDisplayDevice::push(void) {
  fprintf(outfile, "#push matrix\nSeparator {\n");
}

void VrmlDisplayDevice::pop(void) {
  fprintf(outfile, "#pop matrix\n}\n");
}

void VrmlDisplayDevice::multmatrix(const Matrix4 &mat) {
  fprintf(outfile, "# multmatrix\n");
  load(mat);
}

void VrmlDisplayDevice::load(const Matrix4 &mat) {
  fprintf(outfile,
	  "MatrixTransform {\n"
	  "  matrix %g %g %g %g\n"
	  "         %g %g %g %g\n"
	  "         %g %g %g %g\n"
	  "         %g %g %g %g\n"
	  "}\n", 
	  mat.mat[0], mat.mat[1], mat.mat[2], mat.mat[3],
	  mat.mat[4], mat.mat[5], mat.mat[6], mat.mat[7],
	  mat.mat[8], mat.mat[9], mat.mat[10], mat.mat[11],
	  mat.mat[12], mat.mat[13], mat.mat[14], mat.mat[15]
	  );
}

void VrmlDisplayDevice::comment(const char *s) {
  fprintf (outfile, "# %s\n", s);
}

///////////////////// public virtual routines

// initialize the file for output
void VrmlDisplayDevice::write_header(void) {
  fprintf(outfile, "#VRML V1.0 ascii\n");
  fprintf(outfile, "# Created with VMD: "
	  "http://www.ks.uiuc.edu/Research/vmd/\n");
  fprintf(outfile, "Separator {\n");
}

void VrmlDisplayDevice::write_trailer(void) {
  fprintf(outfile, "# That's all, folks\n}\n");
}

void VrmlDisplayDevice::write_cindexmaterial(int cindex, int material) {
  write_colormaterial((float *) &matData[cindex], material);
}

void VrmlDisplayDevice::write_colormaterial(float *rgb, int) {
  // just grab the current material definition
  fprintf(outfile, "Material { \n");
  fprintf(outfile, "diffuseColor %f %f %f\n",
	  mat_diffuse * rgb[0],
	  mat_diffuse * rgb[1],
	  mat_diffuse * rgb[2]);
  fprintf(outfile, "ambientColor %f %f %f\n",
          mat_ambient * rgb[0],
          mat_ambient * rgb[1],
          mat_ambient * rgb[2]);
  fprintf(outfile, "specularColor %f %f %f\n",
          mat_specular * rgb[0],
          mat_specular * rgb[1],
          mat_specular * rgb[2]);
  fprintf(outfile, "shininess %f\n",
	  mat_shininess);
  fprintf(outfile, "transparency %f\n",
	  1.0 - mat_opacity);
  fprintf(outfile, "}\n");
}

