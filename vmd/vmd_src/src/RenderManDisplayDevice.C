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
*      $RCSfile: RenderManDisplayDevice.C
*      $Author: johns $      $Locker:  $               $State: Exp $
*      $Revision: 1.60 $         $Date: 2019/01/17 21:21:01 $
*
***************************************************************************
* DESCRIPTION:
*
* FileRenderer type for the RenderMan interface.
*
***************************************************************************/

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "RenderManDisplayDevice.h"
#include "DispCmds.h"  // needed for line styles
#include "config.h"    // for VMDVERSION string
#include "Hershey.h"   // needed for Hershey font rendering fctns


// The default radius for points and lines (which are displayed
// as small spheres or cylinders, respectively)
#define DEFAULT_RADIUS  0.0025f
#define DASH_LENGTH 0.02f

/// constructor ... initialize some variables
RenderManDisplayDevice::RenderManDisplayDevice() 
: FileRenderer("RenderMan", "PIXAR RenderMan", "vmdscene.rib", "prman %s") {
  reset_vars(); // initialize material cache
}
        
/// destructor
RenderManDisplayDevice::~RenderManDisplayDevice(void) { }


/// (re)initialize cached state variables used to track material changes 
void RenderManDisplayDevice::reset_vars(void) {
  old_color[0] = -1;
  old_color[1] = -1;
  old_color[2] = -1;
  old_ambient = -1;
  old_specular = -1;
  old_opacity = -1;
  old_diffuse = -1;
}


void RenderManDisplayDevice::text(float *pos, float size, float thickness,
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

          cylinder_noxfrm(oldpt, newpt, textthickness, 0);

          fprintf(outfile, "TransformBegin\n");
          write_materials(1);
          fprintf(outfile, "  Translate %g %g %g\n",
                  newpt[0], newpt[1], newpt[2]);
          fprintf(outfile, "  Sphere %g %g %g 360\n",
                  textthickness, -textthickness, textthickness);
          fprintf(outfile, "TransformEnd\n");
        } else {
          // ...otherwise, just draw the next point
          fprintf(outfile, "TransformBegin\n");
          write_materials(1);
          fprintf(outfile, "  Translate %g %g %g\n",
                  newpt[0], newpt[1], newpt[2]);
          fprintf(outfile, "  Sphere %g %g %g 360\n",
                  textthickness, -textthickness, textthickness);
          fprintf(outfile, "TransformEnd\n");
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
void RenderManDisplayDevice::point(float * spdata) {
  float vec[3];
  // Transform the world coordinates
  (transMat.top()).multpoint3d(spdata, vec);

  fprintf(outfile, "TransformBegin\n");
  write_materials(1);
  fprintf(outfile, "  Translate %g %g %g\n", vec[0], vec[1], vec[2]);
  fprintf(outfile, "  Sphere %g %g %g 360\n",
    (float)  lineWidth * DEFAULT_RADIUS,
    (float) -lineWidth * DEFAULT_RADIUS,
    (float)  lineWidth * DEFAULT_RADIUS);
  fprintf(outfile, "TransformEnd\n");
}


/// draw a sphere
void RenderManDisplayDevice::sphere(float * spdata) {
  float vec[3];
  float radius;

  // Transform the world coordinates
  (transMat.top()).multpoint3d(spdata, vec);
  radius = scale_radius(spdata[3]);
  if (radius < DEFAULT_RADIUS) {
    radius = (float) DEFAULT_RADIUS;
  }

  // Draw the sphere
  fprintf(outfile, "TransformBegin\n");
  write_materials(1);
  fprintf(outfile, "  Translate %g %g %g\n", vec[0], vec[1], vec[2]);
  fprintf(outfile, "  Sphere %g %g %g 360\n", radius, -radius, radius);
  fprintf(outfile, "TransformEnd\n");
}


/// draw a line (cylinder) from a to b
void RenderManDisplayDevice::line(float *a, float *b) {
  int i, j, test;
  float dirvec[3], unitdirvec[3];
  float from[3], to[3], tmp1[3], tmp2[3];

  if (lineStyle == ::SOLIDLINE) {
    // transform the world coordinates
    (transMat.top()).multpoint3d(a, from);
    (transMat.top()).multpoint3d(b, to);

    cylinder_noxfrm(from, to, (float) (lineWidth * DEFAULT_RADIUS), 0);
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

      cylinder_noxfrm(from, to, (float) (lineWidth * DEFAULT_RADIUS), 0);
      i++;
    }
  } else {
    msgErr << "RenderManDisplayDevice: Unknown line style "
           << lineStyle << sendmsg;
  }
}


/// draw a cylinder
void RenderManDisplayDevice::cylinder(float *a, float *b, float r, int filled) {
  float vec1[3], vec2[3], radius;

  if (filled) {
    FileRenderer::cylinder(a, b, r, filled);
    return;
  }

  // Transform the world coordinates
  (transMat.top()).multpoint3d(a, vec1);
  (transMat.top()).multpoint3d(b, vec2);
  radius = scale_radius(r);

  cylinder_noxfrm(vec1, vec2, radius, filled);
}


/// draw a NURBS cylinder, no transform, must be in world coords already
void RenderManDisplayDevice::cylinder_noxfrm(float *vec1, float *vec2, 
                                             float radius, int filled) {
  float axis[3];
  float R, phi, rxy, theta;

  // safety check to prevent overly-tiny cylinders
  if (radius < DEFAULT_RADIUS) {
    radius = (float) DEFAULT_RADIUS;
  }

  // RenderMan's cylinders always run along the z axis, and must
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

  // Write the cylinder
  fprintf(outfile, "TransformBegin\n");
  write_materials(1);
  fprintf(outfile, "  Translate %g %g %g\n", vec1[0], vec1[1], vec1[2]);
  if (theta) 
    fprintf(outfile, "  Rotate %g 0 0 1\n", (theta / VMD_PI) * 180);
  if (phi) 
    fprintf(outfile, "  Rotate %g 0 1 0\n", (phi / VMD_PI) * 180);
  fprintf(outfile, "  Cylinder %g 0 %g 360\n", radius, R);
  fprintf(outfile, "TransformEnd\n");
}


/// draw a cone
void RenderManDisplayDevice::cone(float *a, float *b, float r, int /* resolution */) {
  float axis[3], vec1[3], vec2[3];
  float R, phi, rxy, theta;
  float radius;

  // Transform the world coordinates
  (transMat.top()).multpoint3d(a, vec1);
  (transMat.top()).multpoint3d(b, vec2);
  radius = scale_radius(r);
  if (radius < DEFAULT_RADIUS) {
    radius = (float) DEFAULT_RADIUS;
  }

  // RenderMan's cylinders always run along the z axis, and must
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

  // Write the cone
  fprintf(outfile, "TransformBegin\n");
  write_materials(1);
  fprintf(outfile, "  Translate %g %g %g\n", vec1[0], vec1[1], vec1[2]);
  if (theta) 
    fprintf(outfile, "  Rotate %g 0 0 1\n", (theta / VMD_PI) * 180);
  if (phi) 
    fprintf(outfile, "  Rotate %g 0 1 0\n", (phi / VMD_PI) * 180);
  fprintf(outfile, "  Cone %g %g 360\n", R, radius);
  fprintf(outfile, "TransformEnd\n");
}


// draw a triangle
void RenderManDisplayDevice::triangle(const float *a, const float *b, const float *c, const float *n1, const float *n2, const float *n3) {
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
  fprintf(outfile, "Polygon \"P\" [ %g %g %g %g %g %g %g %g %g ] ",
          vec1[0], vec1[1], vec1[2],
          vec2[0], vec2[1], vec2[2],
          vec3[0], vec3[1], vec3[2]);
  fprintf(outfile, "\"N\" [ %g %g %g %g %g %g %g %g %g ]\n",
          norm1[0], norm1[1], norm1[2],
          norm2[0], norm2[1], norm2[2],
          norm3[0], norm3[1], norm3[2]);
}


// draw a tricolor
void RenderManDisplayDevice::tricolor(const float *a, const float *b, const float *c,
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
  fprintf(outfile, "Polygon \"P\" [ %g %g %g %g %g %g %g %g %g ] ",
          vec1[0], vec1[1], vec1[2],
          vec2[0], vec2[1], vec2[2],
          vec3[0], vec3[1], vec3[2]);
  fprintf(outfile, "\"N\" [ %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f ] ",
          norm1[0], norm1[1], norm1[2],
          norm2[0], norm2[1], norm2[2],
          norm3[0], norm3[1], norm3[2]);
  fprintf(outfile, "\"Cs\" [ %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f ]\n",
          c1[0], c1[1], c1[2],
          c2[0], c2[1], c2[2],
          c3[0], c3[1], c3[2]);
}


// use an efficient mesh primitve rather than individual triangles
// when possible.
void RenderManDisplayDevice::trimesh_c4n3v3(int numverts, float * cnv,
                                            int numfacets, int * facets) {
  int i;
  float vec1[3];
  float norm1[3];

  write_materials(0);

  fprintf(outfile, "PointsPolygons");

  // emit vertex counts for each face
  fprintf(outfile, "  [ ");
  for (i=0; i<numfacets; i++) {
    fprintf(outfile, "3 ");
  }
  fprintf(outfile, "]\n");

  // emit vertex indices for each facet
  fprintf(outfile, "  [ ");
  for (i=0; i<numfacets*3; i+=3) {
    fprintf(outfile, "%d %d %d ", facets[i], facets[i+1], facets[i+2]);
  }
  fprintf(outfile, "]\n");

  // emit vertex coordinates
  fprintf(outfile, "  \"P\" [ ");
  for (i=0; i<numverts; i++) {
    (transMat.top()).multpoint3d(cnv + i*10 + 7, vec1);
    fprintf(outfile, "%g %g %g ", vec1[0], vec1[1], vec1[2]);
  }
  fprintf(outfile, "]\n");

  // emit surface normals
  fprintf(outfile, "  \"N\" [ ");
  for (i=0; i<numverts; i++) {
    (transMat.top()).multnorm3d(cnv + i*10 + 4, norm1);
    fprintf(outfile, "%.3f %.3f %.3f ", norm1[0], norm1[1], norm1[2]);
  }
  fprintf(outfile, "]\n");

  // don't emit per-vertex colors when volumetric texturing is enabled
  fprintf(outfile, "  \"Cs\" [ ");
  for (i=0; i<numverts; i++) {
    int idx = i * 10;
    fprintf(outfile, "%.3f %.3f %.3f ", cnv[idx], cnv[idx+1], cnv[idx+2]);
  }
  fprintf(outfile, "]\n");

  fprintf(outfile, "\n");
}


// use an efficient mesh primitve rather than individual triangles
// when possible.
void RenderManDisplayDevice::trimesh_c4u_n3b_v3f(unsigned char *c, char *n,
                                                 float *v, int numfacets) {
  int i;
  float vec1[3];
  float norm1[3];
  int numverts = 3*numfacets;

  const float ci2f = 1.0f / 255.0f; // used for uchar2float and normal conv
  const float cn2f = 1.0f / 127.5f;

  write_materials(0);

  fprintf(outfile, "PointsPolygons");

  // emit vertex counts for each face
  fprintf(outfile, "  [ ");
  for (i=0; i<numfacets; i++) {
    fprintf(outfile, "3 ");
  }
  fprintf(outfile, "]\n");

  // emit vertex indices for each facet
  fprintf(outfile, "  [ ");
  for (i=0; i<numverts; i+=3) {
    fprintf(outfile, "%d %d %d ", i, i+1, i+2);
  }
  fprintf(outfile, "]\n");

  // emit vertex coordinates
  fprintf(outfile, "  \"P\" [ ");
  for (i=0; i<numverts; i++) {
    int idx = i * 3;
    (transMat.top()).multpoint3d(v + idx, vec1);
    fprintf(outfile, "%g %g %g ", vec1[0], vec1[1], vec1[2]);
  }
  fprintf(outfile, "]\n");

  // emit surface normals
  fprintf(outfile, "  \"N\" [ ");
  for (i=0; i<numverts; i++) {
    float ntmp[3];
    int idx = i * 3;

    // conversion from GLbyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
    // float = (2c+1)/(2^8-1)
    ntmp[0] = n[idx  ] * cn2f + ci2f;
    ntmp[1] = n[idx+1] * cn2f + ci2f;
    ntmp[2] = n[idx+2] * cn2f + ci2f;

    (transMat.top()).multnorm3d(ntmp, norm1);
    fprintf(outfile, "%.2f %.2f %.2f ", norm1[0], norm1[1], norm1[2]);
  }
  fprintf(outfile, "]\n");

  // don't emit per-vertex colors when volumetric texturing is enabled
  fprintf(outfile, "  \"Cs\" [ ");
  for (i=0; i<numverts; i++) {
    int idx = i * 4;

    // conversion from GLubyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
    // float = c/(2^8-1)
    fprintf(outfile, "%.3f %.3f %.3f ",
            c[idx  ] * ci2f,
            c[idx+1] * ci2f,
            c[idx+2] * ci2f);
  }
  fprintf(outfile, "]\n");

  fprintf(outfile, "\n");
}


void RenderManDisplayDevice::tristrip(int numverts, const float * cnv,
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

  fprintf(outfile, "PointsPolygons");

  // emit vertex counts for each face
  fprintf(outfile, "  [ ");
  // loop over all of the triangle strips
  for (strip=0; strip < numstrips; strip++) {
    for (i=0; i<(vertsperstrip[strip] - 2); i++) {
      fprintf(outfile, "3 ");
    }
  }
  fprintf(outfile, "]\n");

  // emit vertex indices for each facet
  fprintf(outfile, "  [ ");
  for (strip=0; strip < numstrips; strip++) {
    for (i=0; i<(vertsperstrip[strip] - 2); i++) {
      // render one triangle, using lookup table to fix winding order
      fprintf(outfile, "%d %d %d ",
              facets[v + (stripaddr[i & 0x01][0])],
              facets[v + (stripaddr[i & 0x01][1])],
              facets[v + (stripaddr[i & 0x01][2])]);
      v++; // move on to next vertex
    }
    v+=2; // last two vertices are already used by last triangle
  }
  fprintf(outfile, "]\n");

  // emit vertex coordinates
  fprintf(outfile, "  \"P\" [ ");
  for (i=0; i<numverts; i++) {
    (transMat.top()).multpoint3d(cnv + i*10 + 7, vec1);
    fprintf(outfile, "%g %g %g ", vec1[0], vec1[1], vec1[2]);
  }
  fprintf(outfile, "]\n");

  // emit surface normals
  fprintf(outfile, "  \"N\" [ ");
  for (i=0; i<numverts; i++) {
    (transMat.top()).multnorm3d(cnv + i*10 + 4, norm1);
    fprintf(outfile, "%.3f %.3f %.3f ", norm1[0], norm1[1], norm1[2]);
  }
  fprintf(outfile, "]\n");

  // don't emit per-vertex colors when volumetric texturing is enabled
  fprintf(outfile, "  \"Cs\" [ ");
  for (i=0; i<numverts; i++) {
    int idx = i * 10;
    fprintf(outfile, "%.3f %.3f %.3f ", cnv[idx], cnv[idx+1], cnv[idx+2]);
  }
  fprintf(outfile, "]\n");

  fprintf(outfile, "\n");
}


// draw a square
void RenderManDisplayDevice::square(float *n, float *a, float *b, float *c, float *d) {
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
  fprintf(outfile, "Polygon \"P\" [ %g %g %g %g %g %g %g %g %g %g %g %g ] ",
          vec1[0], vec1[1], vec1[2],
          vec2[0], vec2[1], vec2[2],
          vec3[0], vec3[1], vec3[2],
          vec4[0], vec4[1], vec4[2]);
  fprintf(outfile, "\"N\" [ %g %g %g %g %g %g %g %g %g %g %g %g ]\n",
          norm[0], norm[1], norm[2],
          norm[0], norm[1], norm[2],
          norm[0], norm[1], norm[2],
          norm[0], norm[1], norm[2]);
}


// display a comment
void RenderManDisplayDevice::comment(const char *s) {
  fprintf(outfile, "# %s\n", s);
}

///////////////////// public virtual routines

void RenderManDisplayDevice::write_header() {
  int i, n;

  // Initialize the RenderMan interface
  fprintf(outfile, "##RenderMan RIB-Structure 1.1\n");
  fprintf(outfile, "version 3.03\n");
  fprintf(outfile, "##Creator VMD %s\n", VMDVERSION);
  fprintf(outfile, "#\n");
  fprintf(outfile, "# Molecular graphics export from VMD %s\n", VMDVERSION);
  fprintf(outfile, "# http://www.ks.uiuc.edu/Research/vmd/\n");
  fprintf(outfile, "#\n");
  fprintf(outfile, "# Requires PhotoRealistic RenderMan version 13\n");
  fprintf(outfile, "# Older versions may work, but have not been tested...\n");
  fprintf(outfile, "#\n");

  fprintf(outfile, "# VMD output image resolution and aspect ratio\n");
  fprintf(outfile, "Display \"plot.tif\" \"file\" \"rgba\"\n");
  fprintf(outfile, "Format %ld %ld 1\n", xSize, ySize);
  fprintf(outfile, "FrameAspectRatio %g\n", Aspect);

  // background color rendering takes longer, but is expected behavior
  fprintf(outfile, "# VMD background color\n");
  fprintf(outfile, "# Background colors may slow down rendering, \n");
  fprintf(outfile, "# but this is what VMD users expect by default\n");
  fprintf(outfile, "# Comment these lines for a transparent background:\n");
  fprintf(outfile, "Declare \"bgcolor\" \"uniform color\"\n");
  fprintf(outfile, "Imager \"background\" \"bgcolor\" [%g %g %g]\n",
          backColor[0], backColor[1], backColor[2]);

  fprintf(outfile, "# VMD camera definition\n");
  if (projection() == PERSPECTIVE) {
    // XXX after changing the RIB file to move the inversion of the 
    //     Z directions for left-to-right handedness conversion from
    //     the camera parameter block into the world transformations, 
    //     I somehow picked up an extra unit of translation (not sure how)
    //     so I compensate for that here by adjusting the FOV calculation.
    fprintf(outfile, "Projection \"perspective\" \"fov\" %g\n",
            360.0*atan2((double) 0.5*vSize, (double) 1.0+eyePos[2]-zDist)*VMD_1_PI);
  } else {
    // scaling necessary to equalize sizes of vmd screen and image 
    fprintf(outfile, "ScreenWindow %g %g %g %g\n",
            -Aspect*vSize/4, Aspect*vSize/4, -vSize/4, vSize/4);
    fprintf(outfile, "Projection \"orthographic\"\n");
  }

  // Set up the camera position, negate Z for right-handed coordinate system
  fprintf(outfile, "Translate %g %g %g\n", -eyePos[0], -eyePos[1], eyePos[2]);

  // shadows on, comment out for no shadows
  fprintf(outfile, "# Comment out shadow lines below to disable shadows:\n");
  fprintf(outfile, "Declare \"shadows\" \"string\"\n");
  fprintf(outfile, "Attribute \"light\" \"shadows\" \"on\"\n" );

  // ambient light source (for ambient shading values)
  fprintf(outfile, "# VMD ambient light color\n");
  fprintf(outfile, "LightSource \"ambientlight\" 0 \"intensity\" [1.0] \"lightcolor\" [1 1 1]\n" );
  
  fprintf(outfile, "# VMD directional light sources:\n");
  n = 1;
  // Write out all the light sources as point lights
  for (i = 0; i < DISP_LIGHTS; i++) {
    if (lightState[i].on) {
//      fprintf(outfile, "LightSource \"pointlight\" %d \"intensity\" [1.0] \"lightcolor\" [%g %g %g] \"from\" [%g %g %g]\n",
      fprintf(outfile, "LightSource \"distantlight\" %d \"intensity\" [1.0] \"lightcolor\" [%g %g %g] \"from\" [%g %g %g] \"to\" [0 0 0]\n",
      n++,
      lightState[i].color[0], lightState[i].color[1], lightState[i].color[2],
      lightState[i].pos[0], lightState[i].pos[1], -lightState[i].pos[2]);
    }
  }


  fprintf(outfile, "WorldBegin\n");

  // Make coordinate system right-handed
  fprintf(outfile, "# Make coordinate system right handed by applying a top\n");
  fprintf(outfile, "# level transformation to all subsequent geometry...\n");
  fprintf(outfile, "TransformBegin\n");
  fprintf(outfile, "  Scale 1 1 -1\n");
  fprintf(outfile, "# VMD scene begins here...\n");
}


void RenderManDisplayDevice::write_trailer(void){
  // Make coordinate system right-handed
  fprintf(outfile, "# VMD scene ends here...\n");
  fprintf(outfile, "# \n");
  fprintf(outfile, "# End right-handed coordinate system transformation...\n");
  fprintf(outfile, "TransformEnd\n");

  fprintf(outfile, "WorldEnd\n");
  reset_vars(); // reinitialize material cache
}


void RenderManDisplayDevice::write_materials(int write_color) {
  // keep track of what the last written material properties
  // are, that way we can avoid writing redundant def's
  if (write_color) {
    // the color has changed since last write, emit an update 
    if ((matData[colorIndex][0] != old_color[0]) ||
        (matData[colorIndex][1] != old_color[1]) ||
        (matData[colorIndex][2] != old_color[2])) {
      fprintf(outfile, "  Color %g %g %g\n",
              matData[colorIndex][0], 
              matData[colorIndex][1],
              matData[colorIndex][2]);
      // save the last color
      memcpy(old_color, matData[colorIndex], sizeof(float) * 3);
    }
  }

  // now check opacity
  if (mat_opacity != old_opacity) {
    fprintf(outfile, "  Opacity %g %g %g\n", 
            mat_opacity, mat_opacity, mat_opacity);
    old_opacity = mat_opacity;
  }

  // and the lighting and roughness coefficients
  if ((mat_ambient != old_ambient) || 
      (mat_diffuse != old_diffuse) ||
      (mat_specular != old_specular)) {
    float roughness=10000.0;
    if (mat_shininess > 0.00001f) {
      roughness = 1.0f / mat_shininess;
    }
    fprintf(outfile, "  Surface \"plastic\"" 
            "\"Ka\" %g \"Kd\" %g \"Ks\" %g \"roughness\" %g\n",
            mat_ambient, mat_diffuse, mat_specular, roughness);
    old_ambient = mat_ambient;
    old_specular = mat_specular;
    old_diffuse = mat_diffuse;
  }
}



