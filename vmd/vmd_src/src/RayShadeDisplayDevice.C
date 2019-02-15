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
 *      $RCSfile: RayShadeDisplayDevice.C,v $
 *      $Author: johns $        $Locker:  $                $State: Exp $
 *      $Revision: 1.47 $      $Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * FileRenderer type for the Rayshade raytracer 
 *
 ***************************************************************************/
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "RayShadeDisplayDevice.h"
#include "Matrix4.h"
#include "DispCmds.h"
#include "Inform.h"
#include "utilities.h"
#define DEFAULT_RADIUS 0.002f
#define DASH_LENGTH 0.02f
#define SCALE_FACTOR 1.3f

///////////////////////// constructor and destructor

// constructor ... initialize some variables
RayShadeDisplayDevice::RayShadeDisplayDevice() 
: FileRenderer("Rayshade", "Rayshade 4.0", "vmdscene.ray", "rayshade < %s > %s.rle") { }
        
// destructor
RayShadeDisplayDevice::~RayShadeDisplayDevice(void) { }

///////////////////////// protected nonvirtual routines

// fix the scaling
float RayShadeDisplayDevice::scale_fix(float x) {
  return ( x / SCALE_FACTOR );  
}

// draw a point
void RayShadeDisplayDevice::point(float * spdata) {
  float vec[3];

  // transform the world coordinates
  (transMat.top()).multpoint3d(spdata, vec);
  
  write_cindexmaterial(colorIndex, materialIndex);
  // draw the sphere
  fprintf(outfile, "sphere %5f %5f %5f %5f\n", 
          scale_fix(float(lineWidth)*DEFAULT_RADIUS),
          scale_fix(vec[0]), scale_fix(vec[1]), scale_fix(vec[2]));
}


// draw a sphere
void RayShadeDisplayDevice::sphere(float * spdata) {
  float vec[3];
  float radius;
    
  // transform the world coordinates
  (transMat.top()).multpoint3d(spdata, vec);
  radius = scale_radius(spdata[3]);
  
  write_cindexmaterial(colorIndex, materialIndex);
  // draw the sphere
  fprintf(outfile, "sphere %5f %5f %5f %5f\n", scale_fix(radius), 
          scale_fix(vec[0]), scale_fix(vec[1]), scale_fix(vec[2]));
}



// draw a line (cylinder) from a to b
void RayShadeDisplayDevice::line(float *a, float*b) {
  int i, j, test;
  float dirvec[3], unitdirvec[3];
  float from[3], to[3], tmp1[3], tmp2[3];
    
  if (lineStyle == ::SOLIDLINE) {
    // transform the world coordinates
   (transMat.top()).multpoint3d(a, from);
   (transMat.top()).multpoint3d(b, to);
   
    write_cindexmaterial(colorIndex, materialIndex);
    fprintf(outfile, "cylinder %5f %5f %5f %5f %5f %5f %5f\n",
            scale_fix(float(lineWidth)*DEFAULT_RADIUS),
            scale_fix(from[0]), scale_fix(from[1]), scale_fix(from[2]),
            scale_fix(to[0]), scale_fix(to[1]), scale_fix(to[2])); 
        
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
      if (fabsf(tmp1[0] - to[0]) >= fabsf(dirvec[0]) ) {
        vec_copy(to, tmp2);
        test = 0;
      }
    
      // draw the cylinder
      write_cindexmaterial(colorIndex, materialIndex);
      fprintf(outfile, "cylinder %5f %5f %5f %5f %5f %5f %5f\n",
              scale_fix(float(lineWidth)*DEFAULT_RADIUS),
              scale_fix(from[0]), scale_fix(from[1]), scale_fix(from[2]),
              scale_fix(to[0]), scale_fix(to[1]), scale_fix(to[2])); 
      i++;
    }
  } else {
    msgErr << "RayShadeDisplayDevice: Unknown line style " << lineStyle << sendmsg;
  }
}


// draw a cylinder
void RayShadeDisplayDevice::cylinder(float *a, float *b, float r,int /*filled*/) {
  float from[3], to[3];
  float radius;
  
  // transform the world coordinates
  (transMat.top()).multpoint3d(a, from);
  (transMat.top()).multpoint3d(b, to);
  radius = scale_radius(r);
    
  write_cindexmaterial(colorIndex, materialIndex);
  fprintf(outfile, "cylinder %5f %5f %5f %5f %5f %5f %5f\n",
          scale_fix(radius),
          scale_fix(from[0]), scale_fix(from[1]), scale_fix(from[2]),
          scale_fix(to[0]), scale_fix(to[1]), scale_fix(to[2])); 

  // put disks on the ends
  fprintf(outfile, "disc %5f %5f %5f %5f %5f %5f %5f\n",
	  scale_fix(radius),
	  scale_fix(from[0]), scale_fix(from[1]), scale_fix(from[2]),
	  scale_fix(from[0]-to[0]), scale_fix(from[1]-to[1]), 
	  scale_fix(from[2]-to[2]));
  fprintf(outfile, "disc %5f %5f %5f %5f %5f %5f %5f\n",
	  scale_fix(radius),
	  scale_fix(to[0]), scale_fix(to[1]), scale_fix(to[2]),
	  scale_fix(to[0]-from[0]), scale_fix(to[1]-from[1]), 
	  scale_fix(to[2]-from[2]));
}

// draw a cone
void RayShadeDisplayDevice::cone(float *a, float *b, float r, int /* resolution */) {
  float from[3], to[3];
  float radius;
  
  // transform the world coordinates
  (transMat.top()).multpoint3d(a, from);
  (transMat.top()).multpoint3d(b, to);
  radius = scale_radius(r);
    
  write_cindexmaterial(colorIndex, materialIndex);
  fprintf(outfile, "cone %5f %5f %5f %5f %5f %5f %5f %5f\n",
          scale_fix(radius), 
          scale_fix(from[0]), scale_fix(from[1]), scale_fix(from[2]), 
          scale_fix(float(lineWidth)*DEFAULT_RADIUS), 
          scale_fix(to[0]), scale_fix(to[1]), scale_fix(to[2]));
}

// draw a triangle
void RayShadeDisplayDevice::triangle(const float *a, const float *b, const float *c, const float *n1, const float *n2, const float *n3) {
  float vec1[3], vec2[3], vec3[3];
  float norm1[3], norm2[3], norm3[3];
  
  
  // transform the world coordinates
  (transMat.top()).multpoint3d(a, vec1);
  (transMat.top()).multpoint3d(b, vec2);
  (transMat.top()).multpoint3d(c, vec3);

  // and the normals
  (transMat.top()).multnorm3d(n1, norm1);
  (transMat.top()).multnorm3d(n2, norm2);
  (transMat.top()).multnorm3d(n3, norm3);

  write_cindexmaterial(colorIndex, materialIndex);
  fprintf(outfile, "triangle %5f %5f %5f %5f %5f %5f ", 
          scale_fix(vec1[0]), scale_fix(vec1[1]), scale_fix(vec1[2]),
          scale_fix(norm1[0]), scale_fix(norm1[1]), scale_fix(norm1[2]));
  fprintf(outfile, "%5f %5f %5f %5f %5f %5f ",
          scale_fix(vec2[0]), scale_fix(vec2[1]), scale_fix(vec2[2]),
          scale_fix(norm2[0]), scale_fix(norm2[1]), scale_fix(norm2[2]));
  fprintf(outfile, "%5f %5f %5f %5f %5f %5f\n",
          scale_fix(vec3[0]), scale_fix(vec3[1]), scale_fix(vec3[2]),
          scale_fix(norm3[0]), scale_fix(norm3[1]), scale_fix(norm3[2]));
}

///////////////////// public virtual routines

// initialize the file for output
void RayShadeDisplayDevice::write_header() {
  time_t t;
    
  // file for RayShade raytracer.
  t = time(NULL);
  fprintf(outfile,"/* Rayshade input script: %s\n", my_filename);
  fprintf(outfile,"   Creation date: %s",asctime(localtime(&t)));
  fprintf(outfile,"  ---------------------------- */\n\n");
        
                
  // The current view 
  fprintf(outfile, "\n/* Define current view */\n");        
  fprintf(outfile, "eyep %5f %5f %5f\n", eyePos[0], eyePos[1], eyePos[2]);
  fprintf(outfile, "lookp %5f %5f %5f\n", eyeDir[0], eyeDir[1], eyeDir[2]);
  fprintf(outfile, "up %5f %5f %5f\n", 0.0, 1.0, 0.0);
  fprintf(outfile, "fov %5f\n", 45.0);
  if (stereo_mode()) {
    fprintf(outfile, "eyesep %5f\n", eyesep() );
  }
  fprintf(outfile, "maxdepth 10\n");
        
  // Size of image in pixels
  fprintf(outfile, "screen %d %d\n", (int) xSize, (int) ySize);
      
  // Lights
  fprintf(outfile, "\n/* Light Definitions */\n");

  for (int i=0;i<DISP_LIGHTS;i++) {
    if (lightState[i].on) {
      fprintf(outfile, "light %3.2f %3.2f %3.2f ", lightState[i].color[0],
              lightState[i].color[1], lightState[i].color[2]);
      fprintf(outfile, "directional %5f %5f %5f\n", lightState[i].pos[0],
              lightState[i].pos[1], lightState[i].pos[2]);
    }
  }
       
  // background color
  fprintf(outfile, "\n/* Set background color */\n");
  fprintf(outfile, "background %3.2f %3.2f %3.2f\n", 
          backColor[0], backColor[1], backColor[2]);
                  
  // start the objects
  fprintf(outfile, "\n/* Start object descriptions */\n");       
  // that's all for the header.
}

void RayShadeDisplayDevice::comment(const char *s) {
  fprintf (outfile, "\n/* %s */\n", s);
}

    
// clean up after yourself
void RayShadeDisplayDevice::write_trailer() {
  fprintf(outfile,"\n/* End of File */\n");
  msgInfo << "Rayshade file generation finished" << sendmsg;
}

void RayShadeDisplayDevice::write_cindexmaterial(int cindex, int material) {
  write_colormaterial((float *) &matData[cindex], material);
}

void RayShadeDisplayDevice::write_colormaterial(float *rgb, int) {
  fprintf(outfile, "applysurf diffuse %3.2f %3.2f %3.2f ",
          mat_diffuse * rgb[0],
          mat_diffuse * rgb[1],
          mat_diffuse * rgb[2]);
  fprintf(outfile, "specular %3.2f %3.2f %3.2f transp %3.2f",
          mat_specular * rgb[0],
          mat_specular * rgb[1],
          mat_specular * rgb[2],
          1.0 - mat_opacity);
  fprintf(outfile, "specpow %3.2f\n", mat_shininess);
}

