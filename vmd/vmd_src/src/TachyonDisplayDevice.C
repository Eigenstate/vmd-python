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
*      $RCSfile: TachyonDisplayDevice.C,v $
*      $Author: johns $        $Locker:  $               $State: Exp $
*      $Revision: 1.128 $        $Date: 2019/01/17 21:21:02 $
*
***************************************************************************
* DESCRIPTION:
*
* FileRenderer type for the Tachyon Parallel / Multiprocessor Ray Tracer 
*
***************************************************************************/

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "TachyonDisplayDevice.h"
#include "Matrix4.h"
#include "DispCmds.h"
#include "Inform.h"
#include "utilities.h"
#include "config.h"    // for VMDVERSION string
#include "Hershey.h"   // needed for Hershey font rendering fctns

#define DEFAULT_RADIUS 0.002f
#define DASH_LENGTH 0.02f

#if defined(_MSC_VER) || defined(WIN32)
#define TACHYON_RUN_STRING " -aasamples 12 %s -format BMP -o %s.bmp"
#else
#define TACHYON_RUN_STRING " -aasamples 12 %s -format TARGA -o %s.tga"
#endif

static char tachyon_run_string[2048];

static char * get_tachyon_run_string() {
  char *tbin;
  strcpy(tachyon_run_string, "tachyon");
  
  if ((tbin=getenv("TACHYON_BIN")) != NULL) {
    sprintf(tachyon_run_string, "\"%s\"", tbin);
  }
  strcat(tachyon_run_string, TACHYON_RUN_STRING);
 
  return tachyon_run_string;
}

void TachyonDisplayDevice::update_exec_cmd() {
  const char *tbin;
  if ((tbin = getenv("TACHYON_BIN")) == NULL)
    tbin = "tachyon";

  switch(curformat) {
    case 0:  // BMP
      sprintf(tachyon_run_string, 
        "\"%s\" -aasamples %d %%s -format %s -o %%s.%s", tbin, aasamples, "BMP", "bmp");
      break;

    case 1: // PPM
      sprintf(tachyon_run_string, 
        "\"%s\" -aasamples %d %%s -format %s -o %%s.%s", tbin, aasamples, "PPM", "ppm");
      break;

    case 2: // PPM48
      sprintf(tachyon_run_string, 
        "\"%s\" -aasamples %d %%s -format %s -o %%s.%s", tbin, aasamples, "PPM48", "ppm");
      break;

    case 3: // PSD
      sprintf(tachyon_run_string, 
        "\"%s\" -aasamples %d %%s -format %s -o %%s.%s", tbin, aasamples, "PSD48", "psd");
      break;

    case 4: // SGI RGB
      sprintf(tachyon_run_string, 
        "\"%s\" -aasamples %d %%s -format %s -o %%s.%s", tbin, aasamples, "RGB", "rgb");
      break;

    case 5: // TARGA
    default:
      sprintf(tachyon_run_string, 
        "\"%s\" -aasamples %d %%s -format %s -o %%s.%s", tbin, aasamples, "TARGA", "tga");
      break;
  }
  delete [] execCmd;
  execCmd = stringdup(tachyon_run_string);
}

///////////////////////// constructor and destructor

// constructor ... initialize some variables

TachyonDisplayDevice::TachyonDisplayDevice() : FileRenderer ("Tachyon", "Tachyon", "vmdscene.dat", get_tachyon_run_string()) { 
  // Add supported file formats
  formats.add_name("BMP", 0);
  formats.add_name("PPM", 0);
  formats.add_name("PPM48", 0);
  formats.add_name("PSD48", 0);
  formats.add_name("RGB", 0);
  formats.add_name("TGA", 0);

  // Set default aa level
  has_aa = TRUE;
  aasamples = 12;
  aosamples = 12;

  reset_vars();

  // Default image format depends on platform
#if defined(_MSC_VER) || defined(WIN32)
  curformat = 0; // Windows BMP
#else
  curformat = 5; // Targa
#endif
}
        
// destructor
TachyonDisplayDevice::~TachyonDisplayDevice(void) { }

///////////////////////// protected nonvirtual routines

void TachyonDisplayDevice::reset_vars(void) {
  inclipgroup = 0; // not currently in a clipping group
  involtex = 0;    // volume texturing disabled
  voltexID = -1;   // invalid texture ID
  memset(xplaneeq, 0, sizeof(xplaneeq));
  memset(yplaneeq, 0, sizeof(xplaneeq));
  memset(zplaneeq, 0, sizeof(xplaneeq));
}  


// emit a comment line 
void TachyonDisplayDevice::comment(const char *s) {
  fprintf(outfile, "# %s\n", s);
}


void TachyonDisplayDevice::text(float *pos, float size, float thickness, 
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

          fprintf(outfile, "FCylinder\n"); // flat-ended cylinder
          fprintf(outfile, "  Base %g %g %g\n", oldpt[0], oldpt[1], -oldpt[2]); 
          fprintf(outfile, "  Apex %g %g %g\n", newpt[0], newpt[1], -newpt[2]);
          fprintf(outfile, "  Rad %g \n", textthickness);
          write_cindexmaterial(colorIndex, materialIndex);

          fprintf(outfile, "Sphere \n");  // sphere
          fprintf(outfile, "  Center %g %g %g \n", newpt[0], newpt[1], -newpt[2]);
          fprintf(outfile, "  Rad %g \n", textthickness); 
          write_cindexmaterial(colorIndex, materialIndex);
        } else {
          // ...otherwise, just draw the next point
          fprintf(outfile, "Sphere \n");  // sphere
          fprintf(outfile, "  Center %g %g %g \n", newpt[0], newpt[1], -newpt[2]);
          fprintf(outfile, "  Rad %g \n", textthickness); 
          write_cindexmaterial(colorIndex, materialIndex);
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


// draw a point
void TachyonDisplayDevice::point(float * spdata) {
  float vec[3];
  // transform the world coordinates
  (transMat.top()).multpoint3d(spdata, vec);

  // draw a sphere to represent the point, since we can't really draw points
  fprintf(outfile, "Sphere \n");  // sphere
  fprintf(outfile, "  Center %g %g %g \n ", vec[0], vec[1], -vec[2]);
  fprintf(outfile, "  Rad %g \n",     float(lineWidth)*DEFAULT_RADIUS); 
  write_cindexmaterial(colorIndex, materialIndex);
}


// draw a sphere
void TachyonDisplayDevice::sphere(float * spdata) {
  float vec[3];
  float radius;
    
  // transform the world coordinates
  (transMat.top()).multpoint3d(spdata, vec);
  radius = scale_radius(spdata[3]);
   
  // draw the sphere
  fprintf(outfile, "Sphere \n");  // sphere
  fprintf(outfile, "  Center %g %g %g \n ", vec[0], vec[1], -vec[2]);
  fprintf(outfile, "  Rad %g \n", radius ); 
  write_cindexmaterial(colorIndex, materialIndex);
}


// draw a sphere array
void TachyonDisplayDevice::sphere_array(int spnum, int spres, float *centers, float *radii, float *colors) {
  float vec[3];
  float radius;
  int i, ind;

  ind = 0;
  for (i=0; i<spnum; i++) {
    // transform the world coordinates
    (transMat.top()).multpoint3d(&centers[ind], vec);
    radius = scale_radius(radii[i]);

    // draw the sphere
    fprintf(outfile, "Sphere \n");  // sphere
    fprintf(outfile, "  Center %g %g %g \n ", vec[0], vec[1], -vec[2]);
    fprintf(outfile, "  Rad %g \n", radius );
    write_colormaterial(&colors[ind], materialIndex);
    ind += 3; // next sphere
  }

  // set final color state after array has been drawn
  ind=(spnum-1)*3;
  super_set_color(nearest_index(colors[ind], colors[ind+1], colors[ind+2]));
}


// draw a line (cylinder) from a to b
void TachyonDisplayDevice::line(float *a, float*b) {
  int i, j, test;
  float dirvec[3], unitdirvec[3];
  float from[3], to[3], tmp1[3], tmp2[3];
    
  if (lineStyle == ::SOLIDLINE) {
    // transform the world coordinates
    (transMat.top()).multpoint3d(a, from);
    (transMat.top()).multpoint3d(b, to);
    
    // draw the cylinder
    fprintf(outfile, "FCylinder\n"); // flat-ended cylinder
    fprintf(outfile, "  Base %g %g %g\n", from[0], from[1], -from[2]); 
    fprintf(outfile, "  Apex %g %g %g\n", to[0], to[1], -to[2]);
    fprintf(outfile, "  Rad %g \n", float(lineWidth)*DEFAULT_RADIUS);
    write_cindexmaterial(colorIndex, materialIndex);

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
      fprintf(outfile, "FCylinder\n"); // flat-ended cylinder
      fprintf(outfile, "  Base %g %g %g\n", from[0], from[1], -from[2]); 
      fprintf(outfile, "  Apex %g %g %g\n", to[0], to[1], -to[2]);
      fprintf(outfile, "  Rad %g \n", float(lineWidth)*DEFAULT_RADIUS);
      write_cindexmaterial(colorIndex, materialIndex);
      i++;
    }
  } else {
    msgErr << "TachyonDisplayDevice: Unknown line style " 
           << lineStyle << sendmsg;
  }
}




// draw a cylinder
void TachyonDisplayDevice::cylinder(float *a, float *b, float r, int filled) {
  float from[3], to[3], norm[3];
  float radius;

  // transform the world coordinates
  (transMat.top()).multpoint3d(a, from);
  (transMat.top()).multpoint3d(b, to);
  radius = scale_radius(r);
 
  // draw the cylinder
  fprintf(outfile, "FCylinder\n"); // flat-ended cylinder
  fprintf(outfile, "  Base %g %g %g\n", from[0], from[1], -from[2]); 
  fprintf(outfile, "  Apex %g %g %g\n", to[0], to[1], -to[2]);
  fprintf(outfile, "  Rad %g\n", radius);
  write_cindexmaterial(colorIndex, materialIndex);

  // Cylinder caps?
  if (filled) {
    float div;

    norm[0] = to[0] - from[0];
    norm[1] = to[1] - from[1];
    norm[2] = to[2] - from[2];

    div = 1.0f / sqrtf(norm[0]*norm[0] + norm[1]*norm[1] + norm[2]*norm[2]);
    norm[0] *= div;
    norm[1] *= div;
    norm[2] *= div;

    if (filled & CYLINDER_TRAILINGCAP) {
      fprintf(outfile, "Ring\n");
      fprintf(outfile, "Center %g %g %g \n", from[0], from[1], -from[2]);
      fprintf(outfile, "Normal %g %g %g \n", norm[0], norm[1], -norm[2]); 
      fprintf(outfile, "Inner 0.0  Outer %g \n", radius);
      write_cindexmaterial(colorIndex, materialIndex);
    }
  
    if (filled & CYLINDER_LEADINGCAP) {
      fprintf(outfile, "Ring\n");
      fprintf(outfile, "Center %g %g %g \n", to[0], to[1], -to[2]);
      fprintf(outfile, "Normal %g %g %g \n", -norm[0], -norm[1], norm[2]); 
      fprintf(outfile, "Inner 0.0  Outer %g \n", radius);
      write_cindexmaterial(colorIndex, materialIndex);
    }
  }
}


// draw a triangle
void TachyonDisplayDevice::triangle(const float *a, const float *b, const float *c, const float *n1, const float *n2, const float *n3) {
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

  // draw the triangle
  fprintf(outfile, "STri\n"); // triangle
  fprintf(outfile, "  V0 %g %g %g\n", vec1[0], vec1[1], -vec1[2]);
  fprintf(outfile, "  V1 %g %g %g\n", vec2[0], vec2[1], -vec2[2]); 
  fprintf(outfile, "  V2 %g %g %g\n", vec3[0], vec3[1], -vec3[2]);
  fprintf(outfile, "  N0 %.3f %.3f %.3f\n", -norm1[0], -norm1[1], norm1[2]);
  fprintf(outfile, "  N1 %.3f %.3f %.3f\n", -norm2[0], -norm2[1], norm2[2]); 
  fprintf(outfile, "  N2 %.3f %.3f %.3f\n", -norm3[0], -norm3[1], norm3[2]);
  write_cindexmaterial(colorIndex, materialIndex);
}

// draw triangle with per-vertex colors
void TachyonDisplayDevice::tricolor(const float * xyz1, const float * xyz2, const float * xyz3,
                                    const float * n1,   const float * n2,   const float * n3,
                                    const float *c1, const float *c2, const float *c3) {
  float vec1[3], vec2[3], vec3[3];
  float norm1[3], norm2[3], norm3[3];

  // transform the world coordinates
  (transMat.top()).multpoint3d(xyz1, vec1);
  (transMat.top()).multpoint3d(xyz2, vec2);
  (transMat.top()).multpoint3d(xyz3, vec3);

  // and the normals
  (transMat.top()).multnorm3d(n1, norm1);
  (transMat.top()).multnorm3d(n2, norm2);
  (transMat.top()).multnorm3d(n3, norm3);

  // draw the triangle
  if (!involtex) {
    fprintf(outfile, "VCSTri\n"); // triangle
  } else {
    fprintf(outfile, "STri\n"); // triangle
  }
  fprintf(outfile, "  V0 %g %g %g\n", vec1[0], vec1[1], -vec1[2]);
  fprintf(outfile, "  V1 %g %g %g\n", vec2[0], vec2[1], -vec2[2]);
  fprintf(outfile, "  V2 %g %g %g\n", vec3[0], vec3[1], -vec3[2]);

  fprintf(outfile, "  N0 %.3f %.3f %.3f\n", -norm1[0], -norm1[1], norm1[2]);
  fprintf(outfile, "  N1 %.3f %.3f %.3f\n", -norm2[0], -norm2[1], norm2[2]);
  fprintf(outfile, "  N2 %.3f %.3f %.3f\n", -norm3[0], -norm3[1], norm3[2]);

  if (!involtex) {
    fprintf(outfile, "  C0 %.3f %.3f %.3f\n", c1[0], c1[1], c1[2]);
    fprintf(outfile, "  C1 %.3f %.3f %.3f\n", c2[0], c2[1], c2[2]);
    fprintf(outfile, "  C2 %.3f %.3f %.3f\n", c3[0], c3[1], c3[2]);
  }

  if (materials_on) {
    fprintf(outfile, "  Ambient %g Diffuse %g Specular %g Opacity %g\n",
            mat_ambient, mat_diffuse, mat_mirror, mat_opacity);
  } else {
    fprintf(outfile, "  Ambient %g Diffuse %g Specular %g Opacity %g\n",
            1.0, 0.0, 0.0, mat_opacity);
  }

  if (mat_transmode != 0) {
    fprintf(outfile, "  TransMode R3D ");
  }
  if (mat_outline > 0.0) {
    fprintf(outfile, "  Outline %g Outline_Width %g ", 
            mat_outline, mat_outlinewidth);
  }
  fprintf(outfile, "  Phong Plastic %g Phong_size %g ", mat_specular,
          mat_shininess);
  fprintf(outfile, "VCST\n\n");
}


// use an efficient mesh primitve rather than individual triangles
// when possible.
void TachyonDisplayDevice::trimesh_c4n3v3(int numverts, float * cnv,
                                          int numfacets, int * facets) {
  int i;
  float vec1[3];
  float norm1[3];

  fprintf(outfile, "VertexArray");
  fprintf(outfile, "  Numverts %d\n", numverts);

  fprintf(outfile, "\nCoords\n");
  for (i=0; i<numverts; i++) {
    (transMat.top()).multpoint3d(cnv + i*10 + 7, vec1);
    fprintf(outfile, "%g %g %g\n", vec1[0], vec1[1], -vec1[2]);
  } 

  fprintf(outfile, "\nNormals\n");
  for (i=0; i<numverts; i++) {
    (transMat.top()).multnorm3d(cnv + i*10 + 4, norm1);
    fprintf(outfile, "%.3f %.3f %.3f\n", -norm1[0], -norm1[1], norm1[2]);
  } 

  // don't emit per-vertex colors when volumetric texturing is enabled
  if (!involtex) {
    fprintf(outfile, "\nColors\n");
    for (i=0; i<numverts; i++) {
      int idx = i * 10;
      fprintf(outfile, "%.3f %.3f %.3f\n", cnv[idx], cnv[idx+1], cnv[idx+2]);
    } 
  }

  // emit the texture to be used by the geometry that follows
  write_cindexmaterial(colorIndex, materialIndex);

  // loop over all of the facets in the mesh
  fprintf(outfile, "\nTriMesh %d\n", numfacets);
  for (i=0; i<numfacets*3; i+=3) {
    fprintf(outfile, "%d %d %d\n", facets[i], facets[i+1], facets[i+2]);
  }
 
  // terminate vertex array 
  fprintf(outfile, "\nEnd_VertexArray\n");
}


// use an efficient mesh primitve rather than individual triangles
// when possible.
void TachyonDisplayDevice::trimesh_c4u_n3b_v3f(unsigned char *c, char *n, float *v, int numfacets) {
  int i;
  float vec1[3];
  float norm1[3];
  int numverts = 3*numfacets;

  const float ci2f = 1.0f / 255.0f; // used for uchar2float and normal conv
  const float cn2f = 1.0f / 127.5f;

  fprintf(outfile, "VertexArray");
  fprintf(outfile, "  Numverts %d\n", numverts);

  fprintf(outfile, "\nCoords\n");
  for (i=0; i<numverts; i++) {
    int idx = i * 3;
    (transMat.top()).multpoint3d(v + idx, vec1);
    fprintf(outfile, "%g %g %g\n", vec1[0], vec1[1], -vec1[2]);
  } 

  fprintf(outfile, "\nNormals\n");
  for (i=0; i<numverts; i++) {
    float ntmp[3];
    int idx = i * 3;

    // conversion from GLbyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
    // float = (2c+1)/(2^8-1)
    ntmp[0] = n[idx  ] * cn2f + ci2f;
    ntmp[1] = n[idx+1] * cn2f + ci2f;
    ntmp[2] = n[idx+2] * cn2f + ci2f;
    (transMat.top()).multnorm3d(ntmp, norm1);
    fprintf(outfile, "%.3f %.3f %.3f\n", -norm1[0], -norm1[1], norm1[2]);
  } 

  // don't emit per-vertex colors when volumetric texturing is enabled
  if (!involtex) {
    fprintf(outfile, "\nColors\n");
    for (i=0; i<numverts; i++) {
      int idx = i * 4;

      // conversion from GLubyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
      // float = c/(2^8-1)
      fprintf(outfile, "%.3f %.3f %.3f\n", 
              c[idx  ] * ci2f,
              c[idx+1] * ci2f,
              c[idx+2] * ci2f);
    } 
  }

  // emit the texture to be used by the geometry that follows
  write_cindexmaterial(colorIndex, materialIndex);

  // loop over all of the facets in the mesh
  fprintf(outfile, "\nTriMesh %d\n", numfacets);
  for (i=0; i<numfacets*3; i+=3) {
    fprintf(outfile, "%d %d %d\n", i, i+1, i+2);
  }
 
  // terminate vertex array 
  fprintf(outfile, "\nEnd_VertexArray\n");
}


void TachyonDisplayDevice::tristrip(int numverts, const float * cnv,
                                   int numstrips, const int *vertsperstrip,
                                   const int *facets) {
  int i, strip, v=0;
  float vec1[3];
  float norm1[3];

  fprintf(outfile, "VertexArray");
  fprintf(outfile, "  Numverts %d\n", numverts);

  fprintf(outfile, "\nCoords\n");
  for (i=0; i<numverts; i++) {
    (transMat.top()).multpoint3d(cnv + i*10 + 7, vec1);
    fprintf(outfile, "%g %g %g\n", vec1[0], vec1[1], -vec1[2]);
  } 

  fprintf(outfile, "\nNormals\n");
  for (i=0; i<numverts; i++) {
    (transMat.top()).multnorm3d(cnv + i*10 + 4, norm1);
    fprintf(outfile, "%.3f %.3f %.3f\n", -norm1[0], -norm1[1], norm1[2]);
  } 

  // don't emit per-vertex colors when volumetric texturing is enabled
  if (!involtex) {
    fprintf(outfile, "\nColors\n");
    for (i=0; i<numverts; i++) {
      int idx = i * 10;
      fprintf(outfile, "%.3f %.3f %.3f\n", cnv[idx], cnv[idx+1], cnv[idx+2]);
    } 
  }

  // emit the texture to be used by the geometry that follows
  write_cindexmaterial(colorIndex, materialIndex);

  // loop over all of the triangle strips
  v=0;
  for (strip=0; strip < numstrips; strip++) {
    fprintf(outfile, "\nTriStrip %d\n", vertsperstrip[strip]);

    // loop over all triangles in this triangle strip
    for (i = 0; i < vertsperstrip[strip]; i++) {
      fprintf(outfile, "%d ", facets[v]);
      v++; // move on to the next triangle
    }
  }
 
  // terminate vertex array 
  fprintf(outfile, "\nEnd_VertexArray\n");
}


// define a volumetric texture map
void TachyonDisplayDevice::define_volume_texture(int ID, 
                                                 int xs, int ys, int zs,
                                                 const float *xpq,
                                                 const float *ypq,
                                                 const float *zpq,
                                                 unsigned char *texmap) {
  voltexID = ID; // remember current texture ID

  memcpy(xplaneeq, xpq, sizeof(xplaneeq));
  memcpy(yplaneeq, ypq, sizeof(yplaneeq));
  memcpy(zplaneeq, zpq, sizeof(zplaneeq));

  fprintf(outfile, "# VMD volume texture definition: ID %d\n", ID);
  fprintf(outfile, "#  Res: %d %d %d\n", xs, ys, zs);
  fprintf(outfile, "#  xplaneeq: %g %g %g %g\n",
         xplaneeq[0], xplaneeq[1], xplaneeq[2], xplaneeq[3]);
  fprintf(outfile, "#  yplaneeq: %g %g %g %g\n",
         yplaneeq[0], yplaneeq[1], yplaneeq[2], yplaneeq[3]);
  fprintf(outfile, "#  zplaneeq: %g %g %g %g\n",
         zplaneeq[0], zplaneeq[1], zplaneeq[2], zplaneeq[3]);

  fprintf(outfile, "ImageDef ::VMDVolTex%d\n", ID);
  fprintf(outfile, "  Format RGB24\n");
  fprintf(outfile, "  Resolution %d %d %d\n", xs, ys, zs);
  fprintf(outfile, "  Encoding Hex\n");
 
  int x, y, z;
  for (z=0; z<zs; z++) {
    for (y=0; y<ys; y++) {
      int addr = (z * xs * ys) + (y * xs);
      for (x=0; x<xs; x++) {
        int addr2 = (addr + x) * 3;
        fprintf(outfile, "%02x%02x%02x ", 
                texmap[addr2    ],
                texmap[addr2 + 1],
                texmap[addr2 + 2]);
      }
      fprintf(outfile, "\n");
    }
    fprintf(outfile, "\n");
  }  
  fprintf(outfile, "\n");
  fprintf(outfile, "# End of volume texture ::VMDVolTex%d\n", ID);
  fprintf(outfile, "\n");
  fprintf(outfile, "\n");
}


// enable volumetric texturing, either in "replace" or "modulate" mode
void TachyonDisplayDevice::volume_texture_on(int texmode) {
  involtex = 1;
}


// disable volumetric texturing
void TachyonDisplayDevice::volume_texture_off(void) {
  involtex = 0;
}


void TachyonDisplayDevice::start_clipgroup(void) {
  int i;
  int planesenabled = 0;

  for (i=0; i<VMD_MAX_CLIP_PLANE; i++) {
    if (clip_mode[i] > 0) {
      planesenabled++;  /* count number of active clipping planes */
      if (clip_mode[i] > 1)
        warningflags |= FILERENDERER_NOCLIP; /* emit warnings */
    }
  }

  if (planesenabled > 0) {
    fprintf(outfile, "Start_ClipGroup\n");
    fprintf(outfile, " NumPlanes %d\n", planesenabled);
    for (i=0; i<VMD_MAX_CLIP_PLANE; i++) { 
      if (clip_mode[i] > 0) {
        float tachyon_clip_center[3]; 
        float tachyon_clip_normal[3];
        float tachyon_clip_distance;

        inclipgroup = 1; // we're in a clipping group presently

        // Transform the plane center and the normal
        (transMat.top()).multpoint3d(clip_center[i], tachyon_clip_center);
        (transMat.top()).multnorm3d(clip_normal[i], tachyon_clip_normal);
        vec_negate(tachyon_clip_normal, tachyon_clip_normal);

        // Tachyon uses the distance from the origin to the plane for its
        // representation, instead of the plane center
        tachyon_clip_distance = dot_prod(tachyon_clip_normal, tachyon_clip_center);

        fprintf(outfile, "%g %g %g %g\n", tachyon_clip_normal[0], 
                tachyon_clip_normal[1], -tachyon_clip_normal[2], 
                tachyon_clip_distance);
      }    
    }
    fprintf(outfile, "\n");
  } else {
    inclipgroup = 0; // Not currently in a clipping group
  }
}


///////////////////// public virtual routines

// initialize the file for output
void TachyonDisplayDevice::write_header() {
  fprintf(outfile, "# \n"); 
  fprintf(outfile, "# Molecular graphics exported from VMD %s\n", VMDVERSION);
  fprintf(outfile, "# http://www.ks.uiuc.edu/Research/vmd/\n");
  fprintf(outfile, "# \n"); 
  fprintf(outfile, "# Requires Tachyon version 0.99.0 or newer\n");
  fprintf(outfile, "# \n"); 
  fprintf(outfile, "# Default tachyon rendering command for this scene:\n"); 
  fprintf(outfile, "#   tachyon %s\n", TACHYON_RUN_STRING); 
  fprintf(outfile, "# \n"); 
  // NOTE: the vmd variable "Aspect" has absolutely *nothing* to do
  //       with aspect ratio correction, it is only the ratio of the
  //       width of the graphics window to its height, and so it should
  //       be used only to cause the ray tracer to generate a similarly
  //       proportioned image.

  fprintf(outfile, "Begin_Scene\n");
  fprintf(outfile, "Resolution %d %d\n", (int) xSize, (int) ySize);


  // Emit shading mode information
  fprintf(outfile, "Shader_Mode ");

  // change shading mode depending on whether the user wants shadows
  // or ambient occlusion lighting.
  if (shadows_enabled() || ao_enabled()) {
    fprintf(outfile, "Full\n");
  } else {
    fprintf(outfile, "Medium\n");
  }

  // For VMD we always want to enable flags that preserve a more WYSIWYG  
  // type of output, although in some cases doing things in Tachyon's
  // preferred way might be nicer.  The user can override these with 
  // command line flags still if they want radial fog or other options.
  fprintf(outfile, "  Trans_VMD\n");
  fprintf(outfile, "  Fog_VMD\n");

  // render with ambient occlusion lighting if required
  if (ao_enabled()) {
    fprintf(outfile, "  Ambient_Occlusion\n");
    fprintf(outfile, "    Ambient_Color %g %g %g\n", 
            get_ao_ambient(), get_ao_ambient(), get_ao_ambient());
    fprintf(outfile, "    Rescale_Direct %g\n", get_ao_direct());
    fprintf(outfile, "    Samples %d\n", aosamples);
  }
  fprintf(outfile, "End_Shader_Mode\n");

  write_camera();    // has to be first thing in the file. 
  write_lights();    // could be anywhere.
  write_materials(); // has to be before objects that use them.
}


void TachyonDisplayDevice::end_clipgroup(void) {
  if (inclipgroup) {
    fprintf(outfile, "End_ClipGroup\n");
    inclipgroup = 0; // we're not in a clipping group anymore
  }
}


void TachyonDisplayDevice::write_trailer(void){
  fprintf(outfile, "End_Scene \n");
  if (inclipgroup) {
    msgErr << "TachyonDisplayDevice clipping group still active at end of scene\n" << sendmsg;
  }
  msgInfo << "Tachyon file generation finished" << sendmsg;

  reset_vars();
}



///////////////////// Private routines

void TachyonDisplayDevice::write_camera(void) {
  int raydepth = 50;
 
  // Camera position
  // Tachyon uses a left-handed coordinate system
  // VMD uses right-handed, so z(Tachyon) = -z(VMD).

  switch (projection()) {
    // XXX code for new versions of Tachyon that support orthographic views
    case DisplayDevice::ORTHOGRAPHIC:
      fprintf(outfile, "Camera\n");
      fprintf(outfile, "  Projection Orthographic\n");
      fprintf(outfile, "  Zoom %g\n", 1.0 / (vSize / 2.0));
      fprintf(outfile, "  Aspectratio %g\n", 1.0f);
      fprintf(outfile, "  Antialiasing %d\n", aasamples);
      fprintf(outfile, "  Raydepth %d\n", raydepth);
      fprintf(outfile, "  Center  %g %g %g\n", eyePos[0], eyePos[1], -eyePos[2]);
      fprintf(outfile, "  Viewdir %g %g %g\n", eyeDir[0], eyeDir[1], -eyeDir[2]);
      fprintf(outfile, "  Updir   %g %g %g\n", upDir[0], upDir[1], -upDir[2]);
      fprintf(outfile, "End_Camera\n");
      break;

    case DisplayDevice::PERSPECTIVE:
    default:
      fprintf(outfile, "Camera\n");

      // render with depth of field, but only for perspective projection
      if (dof_enabled() && (projection() == DisplayDevice::PERSPECTIVE)) {
        msgInfo << "DoF focal blur enabled." << sendmsg;
        fprintf(outfile, "  Projection Perspective_DoF\n");
        fprintf(outfile, "  FocalDist %f\n", get_dof_focal_dist());
        fprintf(outfile, "  Aperture %f\n", get_dof_fnumber());
      }

      fprintf(outfile, "  Zoom %g\n", (eyePos[2] - zDist) / vSize);
      fprintf(outfile, "  Aspectratio %g\n", 1.0f);
      fprintf(outfile, "  Antialiasing %d\n", aasamples);
      fprintf(outfile, "  Raydepth %d\n", raydepth);
      fprintf(outfile, "  Center  %g %g %g\n", eyePos[0], eyePos[1], -eyePos[2]);
      fprintf(outfile, "  Viewdir %g %g %g\n", eyeDir[0], eyeDir[1], -eyeDir[2]);
      fprintf(outfile, "  Updir   %g %g %g\n", upDir[0], upDir[1], -upDir[2]);
      fprintf(outfile, "End_Camera\n");
      break;

  }
}

  
void TachyonDisplayDevice::write_lights(void) {  
  // Lights
  int i;  
  int lightcount = 0;
  for (i=0; i<DISP_LIGHTS; i++) {
    if (lightState[i].on) {
      /* give negated light position as the direction vector */
      fprintf(outfile, "Directional_Light Direction %g %g %g ", 
              -lightState[i].pos[0],
              -lightState[i].pos[1],
               lightState[i].pos[2]);
      fprintf(outfile, "Color %g %g %g\n", 
              lightState[i].color[0], lightState[i].color[1], lightState[i].color[2]);
      lightcount++;
    }
  }

  for (i=0; i<DISP_LIGHTS; i++) {
    if (advLightState[i].on) {
      float pos[3];

      // always use world coordinates for now
      vec_copy(pos, advLightState[i].pos);

      if (advLightState[i].spoton) {
        fprintf(outfile, "# SpotLight not implemented yet ...\n");
      } else {
        /* invert handedness of light position vector */
        fprintf(outfile, "Light Center %g %g %g Rad 0.0 ", 
                pos[0], pos[1], -pos[2]);

        /* emit light attentuation parameters if needed */
        if (advLightState[i].constfactor != 1.0f ||
            advLightState[i].linearfactor != 0.0f ||
            advLightState[i].quadfactor != 0.0f) {
          fprintf(outfile, "Attenuation Constant %g Linear %g Quadratic %g\n", 
                  advLightState[i].constfactor, 
                  advLightState[i].linearfactor,
                  advLightState[i].quadfactor);
        }
 
        fprintf(outfile, "Color %g %g %g\n", 
                lightState[i].color[0], lightState[i].color[1], lightState[i].color[2]);
      }

      lightcount++;
    }
  }

  if (lightcount < 1) {
    msgInfo << "Warning: no lights defined in exported scene!!" << sendmsg;
  }
}

void TachyonDisplayDevice::write_materials(void) {
  // background color
  fprintf(outfile, "\nBackground %g %g %g\n", 
          backColor[0], backColor[1], backColor[2]);

  // Specify Tachyon background sky sphere if background gradient
  // mode is enabled.
  if (backgroundmode == 1) {
    float bspheremag = 0.5f;

    // compute positive/negative magnitude of sphere gradient
    switch (projection()) {
      case DisplayDevice::ORTHOGRAPHIC:
        // For orthographic views, Tachyon uses the dot product between
        // the incident ray origin and the sky sphere gradient "up" vector,
        // since all camera rays have the same direction and differ only
        // in their origin.
        bspheremag = vSize / 4.0f;
        break;

      case DisplayDevice::PERSPECTIVE:
      default:
        // For perspective views, Tachyon uses the dot product between
        // the incident ray and the sky sphere gradient "up" vector,
        // so for larger values of vSize, we have to clamp the maximum
        // magnitude to 1.0.
        bspheremag = (vSize / 2.0f) / (eyePos[2] - zDist);
        if (bspheremag > 1.0f)
          bspheremag = 1.0f;
        break;
    }

   
    fprintf(outfile, "Background_Gradient "); 
    if (projection() == DisplayDevice::ORTHOGRAPHIC)
      fprintf(outfile, "Sky_Ortho_Plane\n");
    else
      fprintf(outfile, "Sky_Sphere\n");
    fprintf(outfile, "  UpDir %g %g %g\n", 0.0f, 1.0f, 0.0f);
    fprintf(outfile, "  TopVal %g\n",     bspheremag);
    fprintf(outfile, "  BottomVal %g\n", -bspheremag);
    fprintf(outfile, "  TopColor %g %g %g\n", backgradienttopcolor[0], 
            backgradienttopcolor[1], backgradienttopcolor[2]);
    fprintf(outfile, "  BottomColor %g %g %g\n", backgradientbotcolor[0],
            backgradientbotcolor[1], backgradientbotcolor[2]);
  }

  // set depth cueing parameters
  if (cueingEnabled) {
    switch (cueMode) {
      case CUE_LINEAR:
        fprintf(outfile, 
          "Fog Linear Start %g End %g Density %g Color %g %g %g\n", 
          get_cue_start(), get_cue_end(), 1.0f, 
          backColor[0], backColor[1], backColor[2]);
        break;
 
      case CUE_EXP:
        fprintf(outfile,
          "Fog Exp Start %g End %g Density %g Color %g %g %g\n", 
          0.0, get_cue_end(), get_cue_density(), 
          backColor[0], backColor[1], backColor[2]);
        break;
 
      case CUE_EXP2:
        fprintf(outfile, 
          "Fog Exp2 Start %g End %g Density %g Color %g %g %g\n", 
          0.0, get_cue_end(), get_cue_density(), 
          backColor[0], backColor[1], backColor[2]);
        break;

      case NUM_CUE_MODES:
        // this should never happen
        break;
    }
  } 
}

void TachyonDisplayDevice::write_cindexmaterial(int cindex, int material) {
  write_colormaterial((float *) &matData[cindex], material);
}

// XXX ignores material parameter, may need to improve this..
void TachyonDisplayDevice::write_colormaterial(float *rgb, int /* material */) {
  fprintf(outfile, "Texture\n");
  if (materials_on) {
    fprintf(outfile, "  Ambient %g Diffuse %g Specular %g Opacity %g\n",
            mat_ambient, mat_diffuse, mat_mirror, mat_opacity);
  } else {
    fprintf(outfile, "  Ambient %g Diffuse %g Specular %g Opacity %g\n",
            1.0, 0.0, 0.0, mat_opacity);
  }
  
  if (mat_transmode != 0) {
    fprintf(outfile, "  TransMode R3D ");
  }
  if (mat_outline > 0.0) {
    fprintf(outfile, "  Outline %g Outline_Width %g ", 
            mat_outline, mat_outlinewidth);
  }
  fprintf(outfile, "  Phong Plastic %g Phong_size %g ", mat_specular, 
          mat_shininess);
  fprintf(outfile, "Color %g %g %g ", rgb[0], rgb[1], rgb[2]);

  /// generate volume texture definition, if necessary
  if (!involtex) {
    /// no volume texture, so we use a solid color
    fprintf(outfile, "TexFunc 0\n\n");
  } else {
    /// Perform coordinate system transformations and emit volume texture
    float voluaxs[3];           ///< volume texture coordinate generation
    float volvaxs[3];           ///< parameters in world coordinates
    float volwaxs[3];
    float volcent[3];

    // transform the y/v/w texture coordinate axes from molecule
    // coordinates into world coordinates
    (transMat.top()).multplaneeq3d(xplaneeq, voluaxs);
    (transMat.top()).multplaneeq3d(yplaneeq, volvaxs);
    (transMat.top()).multplaneeq3d(zplaneeq, volwaxs);

    // undo the scaling operation applied by the transformation
    float invscale = 1.0f / scale_radius(1.0f); 
    int i;
    for (i=0; i<3; i++) {
      voluaxs[i] *= invscale;
      volvaxs[i] *= invscale;
      volwaxs[i] *= invscale;
    }

    // compute the volume origin in molecule coordinates by 
    // reverting the scaling factor that was previously applied
    // to the texture plane equation
    float volorgmol[3] = {0,0,0};
    volorgmol[0] = -xplaneeq[3] / norm(xplaneeq);
    volorgmol[1] = -yplaneeq[3] / norm(yplaneeq);
    volorgmol[2] = -zplaneeq[3] / norm(zplaneeq);

    // transform the volume origin into world coordinates
    (transMat.top()).multpoint3d(volorgmol, volcent);

    // emit the texture to the scene file
    fprintf(outfile, "\n  TexFunc  10  ::VMDVolTex%d\n", voltexID);
    fprintf(outfile, "  Center %g %g %g\n", volcent[0], volcent[1], -volcent[2]);
    fprintf(outfile, "  Rotate 0 0 0\n");
    fprintf(outfile, "  Scale  1 1 1\n");
    fprintf(outfile, "  Uaxis %g %g %g\n", voluaxs[0], voluaxs[1], -voluaxs[2]);
    fprintf(outfile, "  Vaxis %g %g %g\n", volvaxs[0], volvaxs[1], -volvaxs[2]);
    fprintf(outfile, "  Waxis %g %g %g\n", volwaxs[0], volwaxs[1], -volwaxs[2]);
    fprintf(outfile, "\n");
  }
}




