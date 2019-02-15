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
 *	$RCSfile: ArtDisplayDevice.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.37 $	$Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Writes to the ART raytracer.  This is available from gondwana.ecr.mu.oz.au
 *   as part of the vort package.  To see the output I suggest:
 *     art plot.scn 1000 1000
 *     vort2ppm plot.pix > plot.ppm
 *     fromppm plot.ppm plot.rgb
 *     ipaste plot.rgb
 *
 * NOTE: As of 2011, the original VORT package home page has vanished, 
 *       but it can currently be found here:
 *         http://bund.com.au/~dgh/eric/
 *         http://bund.com.au/~dgh/eric/vort.tar.gz
 *         http://bund.com.au/~dgh/eric/artimages/index.html
 *
 ***************************************************************************/
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "ArtDisplayDevice.h"
#include "Matrix4.h"
#include "DispCmds.h"
#include "Inform.h"
#include "utilities.h"

#define DEFAULT_RADIUS 0.002f
#define DASH_LENGTH 0.02f

// Be careful when you modify the coordinates.  To make things view the
// right way, I have to rotate everything around the (x,y,z) = (1,1,1)
// vector so that x->z, y->x, and z->y

#define ORDER(x,y,z) -z, -x, y
//#define ORDER(x,y,z) x,y,z

///////////////////////// constructor and destructor

// constructor ... initialize some variables
ArtDisplayDevice::ArtDisplayDevice() 
: FileRenderer("ART", "ART (VORT ray tracer)", "vmdscene.scn", "art %s 500 650"){ }

//destructor
ArtDisplayDevice::~ArtDisplayDevice(void) { }

///////////////////////// protected nonvirtual routines

// draw a point
void ArtDisplayDevice::point(float * spdata) {
  float vec[3];

  // transform the world coordinates
  (transMat.top()).multpoint3d(spdata, vec);
   
  // draw the sphere
  fprintf(outfile, "sphere {\ncolour %f,%f,%f\n",
          matData[colorIndex][0],
          matData[colorIndex][1],
          matData[colorIndex][2]);

  fprintf(outfile, "radius %f\n", float(lineWidth) * DEFAULT_RADIUS);
  fprintf(outfile, "center (%f,%f,%f)\n}\n", ORDER(vec[0], vec[1], vec[2]));
}

// draw a sphere
void ArtDisplayDevice::sphere(float * spdata) {
  float vec[3];
  float radius;
    
  // transform the world coordinates
  (transMat.top()).multpoint3d(spdata, vec);
  radius = scale_radius(spdata[3]);
   
  // draw the sphere
  fprintf(outfile, "sphere {\ncolour %f,%f,%f\n",
          matData[colorIndex][0],
          matData[colorIndex][1],
          matData[colorIndex][2]);

  fprintf(outfile, "radius %f\n", radius);
  fprintf(outfile, "center (%f,%f,%f)\n}\n", ORDER(vec[0], vec[1], vec[2]));
}


// draw a line (cylinder) from a to b
void ArtDisplayDevice::line(float *a, float *b) {
  int i, j, test;
  float dirvec[3], unitdirvec[3];
  float from[3], to[3], tmp1[3], tmp2[3];
 
  if (lineStyle == ::SOLIDLINE) {
    // transform the world coordinates
    (transMat.top()).multpoint3d(a, from);
    (transMat.top()).multpoint3d(b, to);
    
    // draw the cylinder
    fprintf(outfile, "cylinder {\n");
    fprintf(outfile, "colour %f,%f,%f\n",
           matData[colorIndex][0],
           matData[colorIndex][1],
           matData[colorIndex][2]);
    fprintf(outfile, "center(%f,%f,%f)\n", 
            ORDER(from[0], from[1], from[2])); // first point
    fprintf(outfile, "center(%f,%f,%f)\n", 
            ORDER(to[0], to[1], to[2])); // second point
    fprintf(outfile, "radius %f\n}\n", 
            float(lineWidth)*DEFAULT_RADIUS); // radius
        
  } else if (lineStyle == ::DASHEDLINE) {
     // transform the world coordinates
    (transMat.top()).multpoint3d(a, tmp1);
    (transMat.top()).multpoint3d(b, tmp2);

    // how to create a dashed line
    vec_sub(dirvec, tmp2, tmp1);  // vector from a to b
    vec_copy(unitdirvec, dirvec);
    vec_normalize(unitdirvec);    // unit vector from a to b
    test = 1;
    i = 0;
    while (test == 1) {
      for (j=0; j<3; j++) {
        from[j] = (float) (tmp1[j] + (2*i    )*DASH_LENGTH*unitdirvec[j]);
          to[j] = (float) (tmp1[j] + (2*i + 1)*DASH_LENGTH*unitdirvec[j]);
      }
      if (fabsf(tmp1[0] - to[0]) >= fabsf(dirvec[0])) {
        vec_copy(to, tmp2);
        test = 0;
      }
    
      // draw the cylinder
      fprintf(outfile, "cylinder {\n");
      fprintf(outfile, "colour %f,%f,%f\n",
              matData[colorIndex][0],
              matData[colorIndex][1],
              matData[colorIndex][2]);

      // first point
      fprintf(outfile, "center(%f,%f,%f)\n", ORDER(from[0], from[1], from[2]));
      // second point
      fprintf(outfile, "center(%f,%f,%f)\n", ORDER(to[0], to[1], to[2])); 
      // radius
      fprintf(outfile, "radius %f\n}\n", float(lineWidth)*DEFAULT_RADIUS); 
      i++;
    }
  } else {
    msgErr << "ArtDisplayDevice: Unknown line style " << lineStyle << sendmsg;
  }
}

// draw a cylinder
void ArtDisplayDevice::cylinder(float *a, float *b, float r,int /*filled*/) {

  float vec1[3], vec2[3];
  
  // transform the world coordinates
  (transMat.top()).multpoint3d(a, vec1);
  (transMat.top()).multpoint3d(b, vec2);
    
  // draw the cylinder
  fprintf(outfile, "cylinder {\n");
  fprintf(outfile, "colour %f,%f,%f\n",
          matData[colorIndex][0],
          matData[colorIndex][1],
          matData[colorIndex][2]);
  // first point
  fprintf(outfile, "center(%f,%f,%f)\n", ORDER(vec1[0], vec1[1], vec1[2]));
  // second point
  fprintf(outfile, "center(%f,%f,%f)\n", ORDER(vec2[0], vec2[1], vec2[2])); 
  // radius
  fprintf(outfile, "radius %f\n}\n", scale_radius(r));
}

// draw a cone
void ArtDisplayDevice::cone(float *a, float *b, float r, int /* resolution */) {
  float vec1[3], vec2[3];
  
  // transform the world coordinates
  (transMat.top()).multpoint3d(a, vec1);
  (transMat.top()).multpoint3d(b, vec2);
    
  fprintf(outfile, "cone {\n");
  fprintf(outfile, "colour %f,%f,%f\n",
          matData[colorIndex][0],
          matData[colorIndex][1],
          matData[colorIndex][2]);
  // second point
  fprintf(outfile, "vertex(%f,%f,%f)\n", ORDER(vec2[0], vec2[1], vec2[2])); 
  // first point
  fprintf(outfile, "center(%f,%f,%f)\n", ORDER(vec1[0], vec1[1], vec1[2]));
  // radius
  fprintf(outfile, "radius %f\n}\n", scale_radius(r));
}

// draw a triangle
void ArtDisplayDevice::triangle(const float *a, const float *b, const float *c, const float *n1, 
const float *n2, const float *n3) {

  float vec1[3], vec2[3], vec3[3];
  float nor1[3], nor2[3], nor3[3];
  
  // transform the world coordinates
  (transMat.top()).multpoint3d(a, vec1);
  (transMat.top()).multpoint3d(b, vec2);
  (transMat.top()).multpoint3d(c, vec3);

  // and the normals
  (transMat.top()).multnorm3d(n1, nor1);
  (transMat.top()).multnorm3d(n2, nor2);
  (transMat.top()).multnorm3d(n3, nor3);

  // draw the triangle
  fprintf(outfile, "polygon {\n");
  fprintf(outfile, "colour %f,%f,%f\n",
          matData[colorIndex][0],
          matData[colorIndex][1],
          matData[colorIndex][2]);
  
  fprintf(outfile, "vertex (%f,%f,%f),(%f,%f,%f)\n", 
          ORDER(vec1[0], vec1[1], vec1[2]), // point one
          ORDER(nor1[0], nor1[1], nor1[2]));
  
  fprintf(outfile, "vertex (%f,%f,%f),(%f,%f,%f)\n", 
          ORDER(vec2[0], vec2[1], vec2[2]), // point two
          ORDER(nor2[0], nor2[1], nor2[2]));
  fprintf(outfile, "vertex (%f,%f,%f),(%f,%f,%f)\n", 
          ORDER(vec3[0], vec3[1], vec3[2]), // point three
          ORDER(nor3[0], nor3[1], nor3[2]));
  fprintf(outfile, "}\n");
}

// draw a square
void ArtDisplayDevice::square(float *, float *a, float *b, float *c, float *d) {
  
  float vec1[3], vec2[3], vec3[3], vec4[3];
  
  // transform the world coordinates
  (transMat.top()).multpoint3d(a, vec1);
  (transMat.top()).multpoint3d(b, vec2);
  (transMat.top()).multpoint3d(c, vec3);
  (transMat.top()).multpoint3d(d, vec4);

  // draw the square
  fprintf(outfile, "polygon {\n");
  fprintf(outfile, "colour %f,%f,%f\n", 
          matData[colorIndex][0],
          matData[colorIndex][1],
          matData[colorIndex][2]);

  fprintf(outfile, "vertex (%f,%f,%f)\n", 
          ORDER(vec1[0], vec1[1], vec1[2])); // point one
  fprintf(outfile, "vertex (%f,%f,%f)\n", 
          ORDER(vec2[0], vec2[1], vec2[2])); // point two
  fprintf(outfile, "vertex (%f,%f,%f)\n", 
          ORDER(vec3[0], vec3[1], vec3[2])); // point three
  fprintf(outfile, "vertex (%f,%f,%f)\n", 
          ORDER(vec4[0], vec4[1], vec4[2])); // point four
  fprintf(outfile, "}\n");
  
}

void ArtDisplayDevice::comment(const char *s)
{
  fprintf(outfile, "// %s\n", s);
}

///////////////////// public virtual routines

// initialize the file for output
void ArtDisplayDevice::write_header() {

    Initialized = TRUE;

    fprintf(outfile, "up(0, 0, 1) \n");
    fprintf(outfile, "lookat(-1, 0, 0, 0, 0, 0, 0)\n");
    fprintf(outfile, "fieldofview 45 \n");
    

    // write the light sources
    // The light code doesn't work because at this point I don't have the
    //  correct transformation matrix, so I'll leave only the one light source
    fprintf(outfile, "light {\n\tcolour 1, 1, 1\n"
            "\tlocation (-10, 0, 0)\n}\n");

    // set the background
    fprintf(outfile, "background %f, %f, %f\n", 
            backColor[0], backColor[1], backColor[2]);

    // everything is plastic-like
    fprintf(outfile, "material 0.0, 0.75, 0.25, 20.0\n");
}

    
// clean up after yourself
void ArtDisplayDevice::write_trailer() {
    fprintf(outfile, "//End of tokens \n");
    msgInfo << "Art file generation finished" << sendmsg;
}

