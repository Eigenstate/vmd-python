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
*      $RCSfile: LibGelatoDisplayDevice.C
*      $Author: johns $      $Locker:  $               $State: Exp $
*      $Revision: 1.17 $         $Date: 2019/01/17 21:20:59 $
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
#include "LibGelatoDisplayDevice.h"
#include "gelatoapi.h"
#include "config.h"     // needed for default image viewer

// The default radius for points and lines (which are displayed
// as small spheres or cylinders, respectively)
#define DEFAULT_RADIUS  0.0025f

/// constructor ... initialize some variables
LibGelatoDisplayDevice::LibGelatoDisplayDevice() 
: FileRenderer("GelatoInternal", "NVIDIA Gelato 2.1 (internal, in-memory rendering)", "vmdscene.tif", DEF_VMDIMAGEVIEWER) {
  reset_vars(); // initialize material cache
  gapi = GelatoAPI::CreateRenderer(); // create gelato rendering context

  // check for valid Gelato handle
  if (gapi == NULL) {
    msgErr << "Failed to initialize Gelato rendering library" << sendmsg;
  }
}
        
/// destructor
LibGelatoDisplayDevice::~LibGelatoDisplayDevice(void) {
  delete gapi;
}


/// (re)initialize cached state variables used to track material changes 
void LibGelatoDisplayDevice::reset_vars(void) {
  old_color[0] = -1;
  old_color[1] = -1;
  old_color[2] = -1;
  old_ambient = -1;
  old_specular = -1;
  old_opacity = -1;
  old_diffuse = -1;
}


/// draw a point
void LibGelatoDisplayDevice::point(float * spdata) {
  float vec[3];
  // Transform the world coordinates
  (transMat.top()).multpoint3d(spdata, vec);

  gapi->PushTransform();
  write_materials(1);
  gapi->Translate(vec[0], vec[1], vec[2]);
  gapi->Sphere((float)  lineWidth * DEFAULT_RADIUS,
               (float) -lineWidth * DEFAULT_RADIUS,
               (float)  lineWidth * DEFAULT_RADIUS,
               360);
  gapi->PopTransform();
}


/// draw a sphere
void LibGelatoDisplayDevice::sphere(float * spdata) {
  float vec[3];
  float radius;

  // Transform the world coordinates
  (transMat.top()).multpoint3d(spdata, vec);
  radius = scale_radius(spdata[3]);
  if (radius < DEFAULT_RADIUS) {
    radius = (float) DEFAULT_RADIUS;
  }

  // Draw the sphere
  gapi->PushTransform();
  write_materials(1);
  gapi->Translate(vec[0], vec[1], vec[2]);
  gapi->Sphere(radius, -radius, radius, 360);
  gapi->PopTransform();
}



/// draw a line (cylinder) from a to b
void LibGelatoDisplayDevice::line(float *a, float *b) {
  cylinder(a, b, (float) (lineWidth * DEFAULT_RADIUS), 0);
}


// draw a triangle
void LibGelatoDisplayDevice::triangle(const float *a, const float *b, const float *c, const float *n1, const float *n2, const float *n3) {
  int nverts = 3;
  int verts[] = {0, 1, 2};
  float points[9];
  float norms[9];

  // Transform the world coordinates
  (transMat.top()).multpoint3d(a, &points[0]);
  (transMat.top()).multpoint3d(b, &points[3]);
  (transMat.top()).multpoint3d(c, &points[6]);
  (transMat.top()).multnorm3d(n1, &norms[0]);
  (transMat.top()).multnorm3d(n2, &norms[3]);
  (transMat.top()).multnorm3d(n3, &norms[6]);

  // Write the triangle
  write_materials(1);

  gapi->Parameter("vertex point P", (float *) &points);
  gapi->Parameter("vertex normal P", (float *) &norms);
  gapi->Mesh("linear", 1, &nverts, verts);
}


// draw a tricolor
void LibGelatoDisplayDevice::tricolor(const float *a, const float *b, const float *c,
                      const float *n1, const float *n2, const float *n3,
                      const float *c1, const float *c2, const float *c3) {
  int nverts = 3;
  int verts[] = {0, 1, 2};
  float points[9];
  float norms[9];
  float colors[9];

  // Transform the world coordinates
  (transMat.top()).multpoint3d(a, &points[0]);
  (transMat.top()).multpoint3d(b, &points[3]);
  (transMat.top()).multpoint3d(c, &points[6]);
  (transMat.top()).multnorm3d(n1, &norms[0]);
  (transMat.top()).multnorm3d(n2, &norms[3]);
  (transMat.top()).multnorm3d(n3, &norms[6]);

  // copy colors
  memcpy(&colors[0], c1, 3*sizeof(float));
  memcpy(&colors[3], c2, 3*sizeof(float));
  memcpy(&colors[6], c3, 3*sizeof(float));
 
  // Write the triangle
  write_materials(0);

  gapi->Parameter("vertex point P", (float *) &points);
  gapi->Parameter("vertex normal P", (float *) &norms);
  gapi->Parameter("vertex color C", (float *) &colors);
  gapi->Mesh("linear", 1, &nverts, verts);
}

void LibGelatoDisplayDevice::trimesh_c4n3v3(int numverts, float * cnv,
                                            int numfacets, int * facets) {
  // XXX replace this with real mesh code asap
  int i;
  for (i=0; i<numfacets; i++) {
    int ind = i * 3;
    int v0 = facets[ind    ] * 10;
    int v1 = facets[ind + 1] * 10;
    int v2 = facets[ind + 2] * 10;
    tricolor(cnv + v0 + 7, // vertices 0, 1, 2
             cnv + v1 + 7,
             cnv + v2 + 7,
             cnv + v0 + 4, // normals 0, 1, 2
             cnv + v1 + 4,
             cnv + v2 + 4,
             cnv + v0,     // colors 0, 1, 2
             cnv + v1,
             cnv + v2);
  }
}


void LibGelatoDisplayDevice::tristrip(int numverts, const float * cnv,
                                   int numstrips, const int *vertsperstrip,
                                   const int *facets) {
  // XXX replace this with real mesh code asap

  // render triangle strips one triangle at a time
  // triangle winding order is:
  //   v0, v1, v2, then v2, v1, v3, then v2, v3, v4, etc.
  int strip, t, v = 0;
  int stripaddr[2][3] = { {0, 1, 2}, {1, 0, 2} };

  // loop over all of the triangle strips
  for (strip=0; strip < numstrips; strip++) {
    // loop over all triangles in this triangle strip
    for (t = 0; t < (vertsperstrip[strip] - 2); t++) {
      // render one triangle, using lookup table to fix winding order
      int v0 = facets[v + (stripaddr[t & 0x01][0])] * 10;
      int v1 = facets[v + (stripaddr[t & 0x01][1])] * 10;
      int v2 = facets[v + (stripaddr[t & 0x01][2])] * 10;

      tricolor(cnv + v0 + 7, // vertices 0, 1, 2
               cnv + v1 + 7,
               cnv + v2 + 7,
               cnv + v0 + 4, // normals 0, 1, 2
               cnv + v1 + 4,
               cnv + v2 + 4,
               cnv + v0,     // colors 0, 1, 2
               cnv + v1,
               cnv + v2);
      v++; // move on to next vertex
    }
    v+=2; // last two vertices are already used by last triangle
  }
}


// draw a square
void LibGelatoDisplayDevice::square(float *n, float *a, float *b, float *c, float *d) {
  int nverts = 4;
  int verts[] = {0, 1, 2, 4};
  float points[12];
  float norms[12];

  // Transform the world coordinates
  (transMat.top()).multpoint3d(a, &points[0]);
  (transMat.top()).multpoint3d(b, &points[3]);
  (transMat.top()).multpoint3d(c, &points[6]);
  (transMat.top()).multpoint3d(d, &points[9]);
  (transMat.top()).multnorm3d(n, &norms[0]);
  memcpy(&norms[3], &norms[0], 3*sizeof(float)); // all verts use same normal
  memcpy(&norms[6], &norms[0], 3*sizeof(float));
  memcpy(&norms[9], &norms[0], 3*sizeof(float));

  // Write the triangle
  write_materials(1);

  gapi->Parameter("vertex point P", (float *) &points);
  gapi->Parameter("vertex normal P", (float *) &norms);
  gapi->Mesh("linear", 1, &nverts, verts);
}


///////////////////// public virtual routines

void LibGelatoDisplayDevice::write_header() {
  int i, n;

  // Initialize the Gelato output, lighting, and camera information
  gapi->Output(my_filename, "tiff", "rgba", "camera");

// XXX these are assumed default for now, may set them later
// "float gain", 1, "float gamma", 1, "string filter", "gaussian", "float[2] filterwidth", (2, 2)));

  int res[2];
  res[0] = xSize;
  res[1] = ySize;
  gapi->Attribute("int[2] resolution",  res);

  // Make coordinate system right-handed
  gapi->Scale(1, 1, -1);

  if (projection() == PERSPECTIVE) {
    gapi->Attribute("string projection",  "perspective");
    float fov=360.0*atan2((double)0.5*vSize, (double)eyePos[2]-zDist)*VMD_1_PI;
    gapi->Attribute("float fov", &fov);
  } else {
    gapi->Attribute("string projection",  "orthographic");
    // scaling necessary to equalize sizes of vmd screen and image 
    float screen[4];
    screen[0] = -Aspect*vSize/4;
    screen[1] =  Aspect*vSize/4;
    screen[2] = -vSize/4;
    screen[3] =  vSize/4;
    gapi->Attribute("float[4] screen", &screen);
  }

  // set near/far clipping planes
  gapi->Attribute("float near", &nearClip);
  gapi->Attribute("float far", &farClip);

  // Set up the camera position
  gapi->Translate(-eyePos[0], -eyePos[1], -eyePos[2]);

#if 0
  // shadows on, comment out for no shadows
  fprintf( outfile, "Declare \"shadows\" \"string\"\n");
  fprintf( outfile, "Attribute \"light\" \"shadows\" \"on\"\n" );
#endif

  // ambient light source
  char lightname[1024];
  sprintf(lightname, "light0");
  float intensity = 1.0;
  float ambcolor[3], lightorigin[3];
  ambcolor[0] = 1.0;
  ambcolor[1] = 1.0;
  ambcolor[2] = 1.0;
  lightorigin[0] = 0.0;
  lightorigin[1] = 0.0;
  lightorigin[2] = 0.0;

  gapi->Parameter("float intensity", &intensity);
  gapi->Parameter("color lightcolor", ambcolor);
  gapi->Light("light0", "ambientlight");

  // Write out all the light sources as point lights
  n = 1;
  for (i = 0; i < DISP_LIGHTS; i++) {
    if (lightState[i].on) {
      gapi->Parameter("float intensity", &intensity);
      gapi->Parameter("color lightcolor", lightState[i].color);
      gapi->Parameter("point from", lightState[i].pos);
      gapi->Parameter("point to", lightorigin);
      sprintf(lightname, "light%d", n);
      n++,
      gapi->Light(lightname, "distantlight");
    }
  }

  gapi->World();

  // Gelato background color shader
  // Background colors slow down rendering,\n");
  // but this is what VMD users expect.\n");
  // Comment these lines for a transparent background.\n");
  gapi->PushAttributes();
  gapi->Shader("surface", "constant");
  gapi->Attribute("color C", backColor);
  gapi->Input("backplane.pyg");
  gapi->PopAttributes();
}


void LibGelatoDisplayDevice::write_trailer(void){
  gapi->Render("camera");
  reset_vars(); // reinitialize material cache
}


void LibGelatoDisplayDevice::write_materials(int write_color) {
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
    float opacity[3];
    opacity[0] = mat_opacity;
    opacity[1] = mat_opacity;
    opacity[2] = mat_opacity;

    gapi->Attribute("color opacity", opacity);
    old_opacity = mat_opacity;
  }

  // and the lighting and roughness coefficients
  if ((mat_ambient != old_ambient) || 
      (mat_diffuse != old_diffuse) ||
      (mat_specular != old_specular)) {
    float roughness=10000.0;
    if (mat_shininess > 0.00001) {
      roughness = 1.0 / mat_shininess;
    }
    gapi->Parameter("float Ka", &mat_ambient);
    gapi->Parameter("float Kd", &mat_diffuse);
    gapi->Parameter("float Ks", &mat_specular);
    gapi->Parameter("float roughness", &roughness);
    gapi->Shader("surface", "plastic");
    old_ambient = mat_ambient;
    old_specular = mat_specular;
    old_diffuse = mat_diffuse;
  }
}



