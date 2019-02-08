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
*      $RCSfile: GelatoDisplayDevice.C
*      $Author: johns $      $Locker:  $               $State: Exp $
*      $Revision: 1.34 $         $Date: 2019/01/17 21:20:59 $
*
***************************************************************************
* DESCRIPTION:
*
* FileRenderer type for the Gelato interface.
*
***************************************************************************/

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "GelatoDisplayDevice.h"
#include "DispCmds.h"  // needed for line styles
#include "config.h"    // for VMDVERSION string
#include "Hershey.h"   // needed for Hershey font rendering fctns

// The default radius for points and lines (which are displayed
// as small spheres or cylinders, respectively)
#define DEFAULT_RADIUS  0.0025f
#define DASH_LENGTH 0.02f

///
/// Helper functions for calculating the NURBS patch parameters
/// for cylinders and other surfaces of revolution
///
static int intersectlines2D(float *p1, float *t1, float *p2, float *t2, 
                            float *p) {
  float den, nomua, nomub;

  den = t2[1]*t1[0] - t2[0]*t1[1];

  if(fabs(den) < 1.0e-6)
    return 0;

  nomua = (t2[0]*(p1[1]-p2[1]) - t2[1]*(p1[0]-p2[0]));
  nomub = (t1[0]*(p1[1]-p2[1]) - t1[1]*(p1[0]-p2[0]));

  if((fabs(den) < 1.0e-6) &&
     (fabs(nomua) < 1.0e-6) &&
     (fabs(nomub) < 1.0e-6))
    return 0;

  float ua = nomua/den;
  //float ub = nomub/den;

  p[0] = p1[0] + ua*t1[0];
  p[1] = p1[1] + ua*t1[1];

 return 1;
} // intersectlines2D


/// calculate knot vectors for a complete circle
static int fullcirclearc(float r, float *U, float *Pw) {
  float P0[4] = {0};
  float P1[4] = {0};
  float P2[4] = {0};
  float T0[2] = {0};
  float T2[2] = {0};
  float theta = (float) VMD_TWOPI;
  float dtheta = theta/4.0f;
  float w1 = cosf(dtheta/2.0f); // dtheta/2 == base angle


  P0[0] = r;
  P0[1] = 0;
  P0[3] = 1.0;
  T0[0] = 0;
  T0[1] = 1;
  Pw[0] = P0[0];
  Pw[1] = P0[1];
  Pw[3] = 1.0;
  int index = 0; 

  float angle = 0.0;
  int i;
  for(i = 1; i <= 4; i++) {
    angle += dtheta;

    P2[0] = r * cosf(angle);
    P2[1] = r * sinf(angle);
    P2[3] = 1.0;
    T2[0] = -sinf(angle);
    T2[1] = cosf(angle);

    //memset(P1,0,4*sizeof(float));
    intersectlines2D(P0, T0, P2, T2, P1);

    Pw[ (index+1)*4   ] = w1*P1[0];
    Pw[((index+1)*4)+1] = w1*P1[1];
    Pw[((index+1)*4)+3] = w1;

    memcpy(&Pw[(index+2)*4], P2, 4*sizeof(float));

    index += 2;
    if(i < index) {
      memcpy(P0, P2, 4*sizeof(float));
      memcpy(T0, T2, 2*sizeof(float));
    }
  }

  for(i = 0; i < 3; i++) {
    U[i  ] = 0.0;
    U[i+9] = 1.0;
  }

  U[3] = 0.25;
  U[4] = 0.25;
  U[5] = 0.5;
  U[6] = 0.5;
  U[7] = 0.75;
  U[8] = 0.75;

  return 1;
} // fullcirclearc



/// constructor ... initialize some variables

GelatoDisplayDevice::GelatoDisplayDevice() 
: FileRenderer("Gelato", "NVIDIA Gelato 2.1", "vmdscene.pyg", "gelato %s") {
  reset_vars(); // initialize material cache
}
        
/// destructor
GelatoDisplayDevice::~GelatoDisplayDevice(void) { }


/// (re)initialize cached state variables used to track material changes 
void GelatoDisplayDevice::reset_vars(void) {
  old_color[0] = -1;
  old_color[1] = -1;
  old_color[2] = -1;
  old_ambient = -1;
  old_specular = -1;
  old_opacity = -1;
  old_diffuse = -1;
}

void GelatoDisplayDevice::text(float *pos, float size, float thickness,
                               const char *str) {
  float textpos[3];
  float textsize, textthickness;
  hersheyhandle hh;

  // transform the world coordinates
  (transMat.top()).multpoint3d(pos, textpos);
  textsize = size * 1.5f;
  textthickness = thickness*DEFAULT_RADIUS;

  while (*str != '\0') {
    float lm, rm, x, y, ox, oy;
    int draw, odraw;
    ox=oy=x=y=0.0f;
    draw=odraw=0;

    hersheyDrawInitLetter(&hh, *str, &lm, &rm);
    textpos[0] -= lm * textsize;

    while (!hersheyDrawNextLine(&hh, &draw, &x, &y)) {
      float oldpt[3], newpt[3];
      if (draw) {
        newpt[0] = textpos[0] + textsize * x;
        newpt[1] = textpos[1] + textsize * y;
        newpt[2] = textpos[2];

        if (odraw) {
          // if we have both previous and next points, connect them...
          oldpt[0] = textpos[0] + textsize * ox;
          oldpt[1] = textpos[1] + textsize * oy;
          oldpt[2] = textpos[2];

          cylinder_nurb_noxfrm(oldpt, newpt, textthickness, 0); 

          fprintf(outfile, "PushTransform()\n");
          write_materials(1);
          fprintf(outfile, "Translate(%g, %g, %g)\n", 
                  newpt[0], newpt[1], newpt[2]);
          fprintf(outfile, "Sphere(%g, %g, %g, 360)\n", 
                  textthickness, -textthickness, textthickness);
          fprintf(outfile, "PopTransform()\n");
        } else {
          // ...otherwise, just draw the next point
          fprintf(outfile, "PushTransform()\n");
          write_materials(1);
          fprintf(outfile, "Translate(%g, %g, %g)\n", 
                  newpt[0], newpt[1], newpt[2]);
          fprintf(outfile, "Sphere(%g, %g, %g, 360)\n", 
                  textthickness, -textthickness, textthickness);
          fprintf(outfile, "PopTransform()\n");
        }
      }

      ox=x;
      oy=y;
      odraw=draw;
    }
    textpos[0] += rm * textsize;

    str++;
  }
}


/// draw a point
void GelatoDisplayDevice::point(float * spdata) {
  float vec[3];
  // Transform the world coordinates
  (transMat.top()).multpoint3d(spdata, vec);

  fprintf(outfile, "PushTransform()\n");
  write_materials(1);
  fprintf(outfile, "Translate(%g, %g, %g)\n", vec[0], vec[1], vec[2]);
  fprintf(outfile, "Sphere(%g, %g, %g, 360)\n",
    (float)  lineWidth * DEFAULT_RADIUS,
    (float) -lineWidth * DEFAULT_RADIUS,
    (float)  lineWidth * DEFAULT_RADIUS);
  fprintf(outfile, "PopTransform()\n");
}


/// draw a sphere
void GelatoDisplayDevice::sphere(float * spdata) {
  float vec[3];
  float radius;

  // Transform the world coordinates
  (transMat.top()).multpoint3d(spdata, vec);
  radius = scale_radius(spdata[3]);
  if (radius < DEFAULT_RADIUS) {
    radius = (float) DEFAULT_RADIUS;
  }

  // Draw the sphere
  fprintf(outfile, "PushTransform()\n");
  write_materials(1);
  fprintf(outfile, "Translate(%g, %g, %g)\n", vec[0], vec[1], vec[2]);
  fprintf(outfile, "Sphere(%g, %g, %g, 360)\n", radius, -radius, radius);
  fprintf(outfile, "PopTransform()\n");
}


/// draw a line (cylinder) from a to b
void GelatoDisplayDevice::line(float *a, float *b) {
  int i, j, test;
  float dirvec[3], unitdirvec[3];
  float from[3], to[3], tmp1[3], tmp2[3];
   
  if (lineStyle == ::SOLIDLINE) {
    // transform the world coordinates
    (transMat.top()).multpoint3d(a, from);
    (transMat.top()).multpoint3d(b, to);
   
    cylinder_nurb_noxfrm(from, to, (float) (lineWidth * DEFAULT_RADIUS), 0);
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
   
      cylinder_nurb_noxfrm(from, to, (float) (lineWidth * DEFAULT_RADIUS), 0);
      i++;
    }
  } else {
    msgErr << "GelatoDisplayDevice: Unknown line style "
           << lineStyle << sendmsg;
  }
}

/// draw a cylinder
void GelatoDisplayDevice::cylinder(float *a, float *b, float r, int filled) {
  float vec1[3], vec2[3], radius;
   
  if (filled) {
    FileRenderer::cylinder(a, b, r, filled);
    return;
  }

  // Transform the world coordinates
  (transMat.top()).multpoint3d(a, vec1);
  (transMat.top()).multpoint3d(b, vec2);
  radius = scale_radius(r);

  cylinder_nurb_noxfrm(vec1, vec2, radius, filled);
}

/// draw a NURBS cylinder, no transform, must be in world coords already
void GelatoDisplayDevice::cylinder_nurb_noxfrm(float *vec1, float *vec2, 
                                               float radius, int filled) {
  float axis[3];
  float R, phi, rxy, theta;

  // safety check to prevent overly-tiny cylinders
  if (radius < DEFAULT_RADIUS) {
    radius = (float) DEFAULT_RADIUS;
  }

  // Gelato's cylinders always run along the z axis, and must
  // be transformed to the proper position and rotation. This
  // code is taken from OpenGLRenderer.C.
  axis[0] = vec2[0] - vec1[0];
  axis[1] = vec2[1] - vec1[1];
  axis[2] = vec2[2] - vec1[2];

  R = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];
  if (R <= 0) return;

  R = sqrtf(R); // evaluation of sqrt() _after_ early exit

  // determine phi rotation angle, amount to rotate about y
  phi = acosf(axis[2] / R);

  // determine theta rotation, amount to rotate about z
  rxy = sqrtf(axis[0] * axis[0] + axis[1] * axis[1]);
  if (rxy <= 0) {
    theta = 0;
  } else {
    theta = acosf(axis[0] / rxy);
    if (axis[1] < 0) theta = (float) (2.0 * VMD_PI) - theta;
  }

  // Write the cylinder, reorienting and translating it to the correct location
  fprintf(outfile, "PushTransform()\n");
  write_materials(1);
  fprintf(outfile, "Translate(%g,%g,%g)\n", vec1[0], vec1[1], vec1[2]);
  if (theta) 
    fprintf(outfile, "Rotate(%g,0,0,1)\n", (theta / VMD_PI) * 180);
  if (phi) 
    fprintf(outfile, "Rotate(%g,0,1,0)\n", (phi / VMD_PI) * 180);

  // Calculate the NURBS parameters for Cylinder(radius, 0, R, 360) 
  // on the Z axis.  This is mostly hard-coded for speed.
  int A, i;
  float zmin = 0.0;
  float zmax = R;
  float circcv[9*4];
  float uknotv[9+4]; 
  float vknotv[4] = {0.0f,0.0f,1.0f,1.0f};
  float controlv[9*2*4];

  fullcirclearc(radius, uknotv, circcv); // generate full circle arc
  memcpy(controlv, circcv, 9*4*sizeof(float));
  A=0;
  for(i=0; i<9; i++) {
    controlv[A+2] = zmin * controlv[A+3];
    A+=4;
  }
  A=9*4;
  memcpy(&(controlv[A]), circcv, 9*4*sizeof(float));
  for(i = 0; i < 9; i++) {
    controlv[A+2] = zmax * controlv[A+3];
    A += 4;
  }

  // draw the NURBS Cylinder using the Gelato Patch primitive
  fprintf(outfile, "Patch(%d,3,(", 9);
  for (i=0; i<9+3; i++) {
    fprintf(outfile, "%g,", uknotv[i]);
  }
  fprintf(outfile, "),0,1,2,2,(");
  for (i=0; i<4; i++) {
    fprintf(outfile, "%g,", vknotv[i]);
  }
  fprintf(outfile, "),0,1,\"vertex hpoint Pw\", (");
  for (i=0; i<9*2*4; i++) {
    fprintf(outfile, "%g,", controlv[i]);
  }
  fprintf(outfile, "))\n");

  fprintf(outfile, "PopTransform()\n");
}


// draw a triangle
void GelatoDisplayDevice::triangle(const float *a, const float *b, const float *c, const float *n1, const float *n2, const float *n3) {
  float vec1[3], vec2[3], vec3[3];
  float norm1[3], norm2[3], norm3[3];

  // Transform the world coordinates
  (transMat.top()).multpoint3d(a, vec1);
  (transMat.top()).multpoint3d(b, vec2);
  (transMat.top()).multpoint3d(c, vec3);
  (transMat.top()).multnorm3d(n1, norm1);
  (transMat.top()).multnorm3d(n2, norm2);
  (transMat.top()).multnorm3d(n3, norm3);

  // Write the triangle
  write_materials(1);
  fprintf(outfile, "Mesh(\"linear\", (3,), (0, 1, 2), "
          "\"vertex point P\", (%g, %g, %g, %g, %g, %g, %g, %g, %g), "
          "\"vertex normal N\", (%g, %g, %g, %g, %g, %g, %g, %g, %g))\n",
          vec1[0], vec1[1], vec1[2],
          vec2[0], vec2[1], vec2[2],
          vec3[0], vec3[1], vec3[2],
          norm1[0], norm1[1], norm1[2],
          norm2[0], norm2[1], norm2[2],
          norm3[0], norm3[1], norm3[2]);
}


// draw a tricolor
void GelatoDisplayDevice::tricolor(const float *a, const float *b, const float *c,
                      const float *n1, const float *n2, const float *n3,
                      const float *c1, const float *c2, const float *c3) {
  float vec1[3], vec2[3], vec3[3];
  float norm1[3], norm2[3], norm3[3];

  // Transform the world coordinates
  (transMat.top()).multpoint3d(a, vec1);
  (transMat.top()).multpoint3d(b, vec2);
  (transMat.top()).multpoint3d(c, vec3);
  (transMat.top()).multnorm3d(n1, norm1);
  (transMat.top()).multnorm3d(n2, norm2);
  (transMat.top()).multnorm3d(n3, norm3);

  // Write the triangle
  write_materials(0);
  fprintf(outfile, "Mesh(\"linear\", (3,), (0, 1, 2), "
          "\"vertex point P\", (%g, %g, %g, %g, %g, %g, %g, %g, %g), "
          "\"vertex normal N\", (%g, %g, %g, %g, %g, %g, %g, %g, %g), "
          "\"vertex color C\", (%g, %g, %g, %g, %g, %g, %g, %g, %g))\n",
          vec1[0], vec1[1], vec1[2],
          vec2[0], vec2[1], vec2[2],
          vec3[0], vec3[1], vec3[2],
          norm1[0], norm1[1], norm1[2],
          norm2[0], norm2[1], norm2[2],
          norm3[0], norm3[1], norm3[2],
          c1[0], c1[1], c1[2],
          c2[0], c2[1], c2[2],
          c3[0], c3[1], c3[2]);
}

void GelatoDisplayDevice::trimesh_c4n3v3(int numverts, float * cnv,
                                         int numfacets, int * facets) {
  float vec1[3];
  float norm1[3];
  int i;

  write_materials(0);
  fprintf(outfile, "Mesh(\"linear\", (");
 
  for (i=0; i<numfacets; i++) {
    fprintf(outfile, "3,");
  }
  fprintf(outfile, "), (");

  for (i=0; i<numfacets; i++) {
    fprintf(outfile, "%d, %d, %d,", facets[i*3], facets[i*3+1], facets[i*3+2]);
  }
  fprintf(outfile, "), ");

  fprintf(outfile, "\n\"vertex point P\", (");
  for (i=0; i<numverts; i++) {
    (transMat.top()).multpoint3d(cnv + i*10 + 7, vec1);
    fprintf(outfile, "%g, %g, %g,", vec1[0], vec1[1], vec1[2]);
  }
  fprintf(outfile, "), ");
  
  fprintf(outfile,  "\n\"vertex normal N\", (");
  for (i=0; i<numverts; i++) {
    (transMat.top()).multnorm3d(cnv + i*10 + 4, norm1);
    fprintf(outfile, "%g, %g, %g,", norm1[0], norm1[1], norm1[2]);
  }
  fprintf(outfile, "), ");

  fprintf(outfile,  "\n\"vertex color C\", (");
  for (i=0; i<numverts; i++) {
    float *c = cnv + i*10;
    fprintf(outfile, "%g, %g, %g,", c[0], c[1], c[2]);
  }
  fprintf(outfile, "))\n");
}


void GelatoDisplayDevice::tristrip(int numverts, const float * cnv,
                                   int numstrips, const int *vertsperstrip,
                                   const int *facets) {
  float vec1[3];
  float norm1[3];
  int i;
  // render triangle strips one triangle at a time
  // triangle winding order is:
  //   v0, v1, v2, then v2, v1, v3, then v2, v3, v4, etc.
  int strip, v = 0;
  int stripaddr[2][3] = { {0, 1, 2}, {1, 0, 2} };

  write_materials(0);

  fprintf(outfile, "Mesh(\"linear\", (");

  // loop over all of the triangle strips
  for (strip=0; strip < numstrips; strip++) {
    for (i=0; i<(vertsperstrip[strip] - 2); i++) {
      fprintf(outfile, "3,");
    }
  }
  fprintf(outfile, "), (");

  for (strip=0; strip < numstrips; strip++) {
    for (i=0; i<(vertsperstrip[strip] - 2); i++) {
      // render one triangle, using lookup table to fix winding order
      fprintf(outfile, "%d, %d, %d,", 
              facets[v + (stripaddr[i & 0x01][0])],
              facets[v + (stripaddr[i & 0x01][1])],
              facets[v + (stripaddr[i & 0x01][2])]);
      v++; // move on to next vertex
    }
    v+=2; // last two vertices are already used by last triangle
  }
  fprintf(outfile, "), ");

  fprintf(outfile, "\n\"vertex point P\", (");
  for (i=0; i<numverts; i++) {
    (transMat.top()).multpoint3d(cnv + i*10 + 7, vec1);
    fprintf(outfile, "%g, %g, %g,", vec1[0], vec1[1], vec1[2]);
  }
  fprintf(outfile, "), ");
  
  fprintf(outfile,  "\n\"vertex normal N\", (");
  for (i=0; i<numverts; i++) {
    (transMat.top()).multnorm3d(cnv + i*10 + 4, norm1);
    fprintf(outfile, "%g, %g, %g,", norm1[0], norm1[1], norm1[2]);
  }
  fprintf(outfile, "), ");

  fprintf(outfile,  "\n\"vertex color C\", (");
  for (i=0; i<numverts; i++) {
    const float *c = cnv + i*10;
    fprintf(outfile, "%g, %g, %g,", c[0], c[1], c[2]);
  }
  fprintf(outfile, "))\n");
}


// draw a square
void GelatoDisplayDevice::square(float *n, float *a, float *b, float *c, float *d) {
  float vec1[3], vec2[3], vec3[3], vec4[3];
  float norm[3];

  // Transform the world coordinates
  (transMat.top()).multpoint3d(a, vec1);
  (transMat.top()).multpoint3d(b, vec2);
  (transMat.top()).multpoint3d(c, vec3);
  (transMat.top()).multpoint3d(d, vec4);
  (transMat.top()).multnorm3d(n, norm);

  // Write the square
  write_materials(1);
  fprintf(outfile, "Mesh(\"linear\", (4,), (0, 1, 2, 3), "
          "\"vertex point P\", "
          "(%g, %g, %g, %g, %g, %g, %g, %g, %g, %g, %g, %g), "
          "\"vertex normal N\", "
          "(%g, %g, %g, %g, %g, %g, %g, %g, %g, %g, %g, %g))\n",
          vec1[0], vec1[1], vec1[2],
          vec2[0], vec2[1], vec2[2],
          vec3[0], vec3[1], vec3[2],
          vec4[0], vec4[1], vec4[2],
          norm[0], norm[1], norm[2],
          norm[0], norm[1], norm[2],
          norm[0], norm[1], norm[2],
          norm[0], norm[1], norm[2]);
}


// display a comment
void GelatoDisplayDevice::comment(const char *s) {
  fprintf(outfile, "# %s\n", s);
}

///////////////////// public virtual routines

void GelatoDisplayDevice::write_header() {
  int i, n;

  // Initialize the Gelato interface
  fprintf(outfile, "# \n");
  fprintf(outfile, "# Molecular graphics export from VMD %s\n", VMDVERSION);
  fprintf(outfile, "# http://www.ks.uiuc.edu/Research/vmd/\n");
  fprintf(outfile, "# Requires NVIDIA Gelato 2.1, PYG format\n");
  fprintf(outfile, "# \n");

  fprintf(outfile, "Output(\"%s.tif\", \"tiff\", \"rgba\", \"camera\", \"float gain\", 1, \"float gamma\", 1, \"string filter\", \"gaussian\", \"float[2] filterwidth\", (2, 2))\n", my_filename);
  fprintf(outfile, "Attribute(\"int[2] resolution\",  (%ld, %ld))\n", xSize, ySize);
#if 0
  fprintf(outfile, "Attribute(\"float pixelaspect\", %g)\n", 1.0);
  // XXX auto calculated by Gelato, seems to match correctly already
  fprintf(outfile, "FrameAspectRatio %g\n", Aspect);
#endif

  // Make coordinate system right-handed
  fprintf(outfile, "Scale(1, 1, -1)\n");

  if (projection() == PERSPECTIVE) {
    fprintf(outfile, "Attribute(\"string projection\",  \"perspective\")\n");
    fprintf(outfile, "Attribute(\"float fov\", %g)\n",
            360.0*atan2((double) 0.5*vSize, (double) eyePos[2]-zDist)*VMD_1_PI);
  } else {
    fprintf(outfile, "Attribute(\"string projection\",  \"orthographic\")\n");
    // scaling necessary to equalize sizes of vmd screen and image 
    fprintf(outfile, "Attribute(\"float[4] screen\", (%g, %g, %g, %g))\n",
            -Aspect*vSize/4, Aspect*vSize/4, -vSize/4, vSize/4);
  }

  // Set up the camera position
  fprintf(outfile, "Attribute (\"float near\", %g)\n", nearClip);
  fprintf(outfile, "Attribute (\"float far\", %g)\n", farClip); 
  fprintf(outfile, "# translate for camera position\n");
  fprintf(outfile, "Translate(%g, %g, %g)\n", 
          -eyePos[0], -eyePos[1], -eyePos[2]);

#if 0
  // shadows on, comment out for no shadows
  fprintf( outfile, "Declare \"shadows\" \"string\"\n");
  fprintf( outfile, "Attribute \"light\" \"shadows\" \"on\"\n" );
#endif

  // ambient light source
  fprintf(outfile, "Light(\"light0\", \"ambientlight\", \"float intensity\", 1.0, \"color lightcolor\", (1, 1, 1))\n");
  
  n = 1;
  // Write out all the light sources as point lights
  for (i = 0; i < DISP_LIGHTS; i++) {
    if (lightState[i].on) {
//      fprintf(outfile, "Light(\"light%d\", \"pointlight\", \"float intensity\", 1.0, \"color lightcolor\", (%g, %g, %g), \"point from\", (%g, %g, %g))\n",
        fprintf(outfile, "Light(\"light%d\", \"distantlight\", \"float intensity\", 1, \"color lightcolor\", (%g, %g, %g), \"point from\", (%g, %g, %g), \"point to\", (0, 0, 0))\n",
      n++,
      lightState[i].color[0], lightState[i].color[1], lightState[i].color[2],
      lightState[i].pos[0], lightState[i].pos[1], lightState[i].pos[2]);
    }
  }


  fprintf(outfile, "World()\n");

  // Gelato background color shader
  fprintf(outfile, "# Background colors slow down rendering,\n");
  fprintf(outfile, "# but this is what VMD users expect.\n");
  fprintf(outfile, "# Comment these lines for a transparent background.\n");
  fprintf(outfile, "PushAttributes()\n");
  fprintf(outfile, "Shader(\"surface\", \"constant\")\n");
  fprintf(outfile, "Attribute(\"color C\", (%g, %g, %g))\n",
          backColor[0], backColor[1], backColor[2]);
  fprintf(outfile, "Input(\"backplane.pyg\")\n");
  fprintf(outfile, "PopAttributes()\n");

}


void GelatoDisplayDevice::write_trailer(void){
  fprintf(outfile, "Render (\"camera\")\n");
  fprintf(outfile, "# End Input\n");
  reset_vars(); // reinitialize material cache
}


void GelatoDisplayDevice::write_materials(int write_color) {
  // keep track of what the last written material properties
  // are, that way we can avoid writing redundant def's
  if (write_color) {
    // the color has changed since last write, emit an update 
    if ((matData[colorIndex][0] != old_color[0]) ||
        (matData[colorIndex][1] != old_color[1]) ||
        (matData[colorIndex][2] != old_color[2])) {
      fprintf(outfile, "Attribute(\"color C\",  (%g, %g, %g))\n",
              matData[colorIndex][0], 
              matData[colorIndex][1],
              matData[colorIndex][2]);
      // save the last color
      memcpy(old_color, matData[colorIndex], sizeof(float) * 3);
    }
  }

  // now check opacity
  if (mat_opacity != old_opacity) {
    fprintf(outfile, "Attribute(\"color opacity\", (%g, %g, %g))\n", 
            mat_opacity, mat_opacity, mat_opacity);
    old_opacity = mat_opacity;
  }

  // and the lighting and roughness coefficients
  if ((mat_ambient != old_ambient) || 
      (mat_diffuse != old_diffuse) ||
      (mat_specular != old_specular)) {
    float roughness=10000.0f;
    if (mat_shininess > 0.00001f) {
      roughness = 1.0f / mat_shininess;
    }
    fprintf(outfile, "Shader(\"surface\", \"plastic\", " 
            "\"float Ka\", %g, \"float Kd\", %g, "
            "\"float Ks\", %g, \"float roughness\", %g)\n",
            mat_ambient, mat_diffuse, mat_specular, roughness);
    old_ambient = mat_ambient;
    old_specular = mat_specular;
    old_diffuse = mat_diffuse;
  }
}



