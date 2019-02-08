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
 *	$RCSfile: Vrml2DisplayDevice.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.45 $	$Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   VRML2 / VRML97 scene export code
 *
 * VRML2 / VRML97 specification:
 *   http://www.web3d.org/x3d/specifications/vrml/ISO-IEC-14772-VRML97/
 *
 * List of VRML viewers at NIST:
 *   http://cic.nist.gov/vrml/vbdetect.html
 *
 ***************************************************************************/

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "Vrml2DisplayDevice.h"
#include "Matrix4.h"
#include "utilities.h"
#include "DispCmds.h"  // needed for line styles
#include "Hershey.h"   // needed for Hershey font rendering fctns

// The default radius for points and lines (which are displayed
// as small spheres or cylinders, respectively)
#define DEFAULT_RADIUS 0.002f
#define DASH_LENGTH 0.02f

///////////////////////// constructor and destructor

// constructor ... initialize some variables
Vrml2DisplayDevice::Vrml2DisplayDevice(void) : 
  FileRenderer("VRML-2", "VRML 2.0 (VRML97)", "vmdscene.wrl", "true") {
}
               
///////////////////////// protected nonvirtual routines
void Vrml2DisplayDevice::set_color(int mycolorIndex) {
#if 0
  write_cindexmaterial(mycolorIndex, materialIndex);
#endif
}


void Vrml2DisplayDevice::text(float *pos, float size, float thickness,
                              const char *str) {
  float textpos[3];
  hersheyhandle hh;

  // transform the world coordinates
  (transMat.top()).multpoint3d(pos, textpos);
  float textsize = size * 1.5f;
  //  XXX text thickness not usable with VRML2 since we don't have a line
  //      thickness parameter when drawing indexed line sets, apparently...
  //  float textthickness = thickness*DEFAULT_RADIUS;

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

          // ugly and wasteful, but it will work
          fprintf(outfile, "Shape {\n");
          fprintf(outfile, "  ");
          write_cindexmaterial(colorIndex, materialIndex);
          fprintf(outfile, "  geometry IndexedLineSet { \n");
          fprintf(outfile, "    coordIndex [ 0, 1, -1 ]\n");
          fprintf(outfile, "    coord Coordinate { point [ %g %g %g,  %g %g %g ] }\n",
                  oldpt[0], oldpt[1], oldpt[2], newpt[0], newpt[1], newpt[2]);

          float col[3];
          vec_copy(col, matData[colorIndex]);
          fprintf(outfile, "    color Color { color [ %.3f %.3f %.3f, %.3f %.3f %.3f ] }\n",
                  col[0], col[1], col[2], col[0], col[1], col[2]);
          fprintf(outfile, "  }\n");
          fprintf(outfile, "}\n");
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


// draw a sphere
void Vrml2DisplayDevice::sphere(float *xyzr) {
  float cent[3], radius;

  // transform the coordinates
  (transMat.top()).multpoint3d(xyzr, cent);
  radius = scale_radius(xyzr[3]);

  fprintf(outfile, "Transform {\n");
  fprintf(outfile, "  translation %g %g %g\n", cent[0], cent[1], cent[2]);
  fprintf(outfile, "  children [ Shape {\n");
  fprintf(outfile, "    ");
  write_cindexmaterial(colorIndex, materialIndex);
  fprintf(outfile, "    geometry Sphere { radius %g }\n", radius);
  fprintf(outfile, "  }]\n");
  fprintf(outfile, "}\n");
}


// draw a point
void Vrml2DisplayDevice::point(float * xyz) {
  float txyz[3];

  // transform the coordinates
  (transMat.top()).multpoint3d(xyz, txyz);

  // ugly and wasteful, but it will work
  fprintf(outfile, "Shape {\n");
  fprintf(outfile, "  ");
  write_cindexmaterial(colorIndex, materialIndex);

  fprintf(outfile, "  geometry PointSet { \n");
  fprintf(outfile, "    coord Coordinate { point [%g %g %g] }\n",
          txyz[0], txyz[1], txyz[2]);
 
  float col[3];
  vec_copy(col, matData[colorIndex]);
  fprintf(outfile, "    color Color { color [%.3f %.3f %.3f] }\n",
          col[0], col[1], col[2]);
  fprintf(outfile, "  }\n");
  fprintf(outfile, "}\n");
}


//// draw a line from a to b
////  Doesn't yet support the dotted line method
void Vrml2DisplayDevice::line(float *a, float*b) {
  float ta[3], tb[3];

  if (lineStyle == ::SOLIDLINE) {
    // transform the coordinates
    (transMat.top()).multpoint3d(a, ta);
    (transMat.top()).multpoint3d(b, tb);

    // ugly and wasteful, but it will work
    fprintf(outfile, "Shape {\n");
    fprintf(outfile, "  ");
    write_cindexmaterial(colorIndex, materialIndex);
    fprintf(outfile, "  geometry IndexedLineSet { \n"); 
    fprintf(outfile, "    coordIndex [ 0, 1, -1 ]\n");
    fprintf(outfile, "    coord Coordinate { point [ %g %g %g,  %g %g %g ] }\n",
            ta[0], ta[1], ta[2], tb[0], tb[1], tb[2]);

    float col[3];
    vec_copy(col, matData[colorIndex]);
    fprintf(outfile, "    color Color { color [ %.3f %.3f %.3f, %.3f %.3f %.3f ] }\n", 
            col[0], col[1], col[2], col[0], col[1], col[2]);

    fprintf(outfile, "  }\n");
    fprintf(outfile, "}\n");
  } else if (lineStyle == ::DASHEDLINE) {
    float dirvec[3], unitdirvec[3], tmp1[3], tmp2[3];
    int i, j, test;

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
        ta[j] = (float) (tmp1[j] + (2*i    )*DASH_LENGTH*unitdirvec[j]);
        tb[j] = (float) (tmp1[j] + (2*i + 1)*DASH_LENGTH*unitdirvec[j]);
      }
      if (fabsf(tmp1[0] - tb[0]) >= fabsf(dirvec[0])) {
        vec_copy(tb, tmp2);
        test = 0;
      }

      // ugly and wasteful, but it will work
      fprintf(outfile, "Shape {\n");
      fprintf(outfile, "  ");
      write_cindexmaterial(colorIndex, materialIndex);
      fprintf(outfile, "  geometry IndexedLineSet { \n"); 
      fprintf(outfile, "    coordIndex [ 0, 1, -1 ]\n");
      fprintf(outfile, "    coord Coordinate { point [ %g %g %g,  %g %g %g ] }\n",
              ta[0], ta[1], ta[2], tb[0], tb[1], tb[2]);

      float col[3];
      vec_copy(col, matData[colorIndex]);
      fprintf(outfile, "    color Color { color [ %.3f %.3f %.3f, %.3f %.3f %.3f ] }\n", 
              col[0], col[1], col[2], col[0], col[1], col[2]);

      fprintf(outfile, "  }\n");
      fprintf(outfile, "}\n");
      i++;
    }
  } else {
    msgErr << "Vrml2DisplayDevice: Unknown line style "
           << lineStyle << sendmsg;
  }
}


// draw a cylinder
void Vrml2DisplayDevice::cylinder(float *a, float *b, float r, int filled) {
  float ta[3], tb[3], radius;

  // transform the coordinates
  (transMat.top()).multpoint3d(a, ta);
  (transMat.top()).multpoint3d(b, tb);
  radius = scale_radius(r);

  cylinder_noxfrm(ta, tb, radius, filled);
}


void Vrml2DisplayDevice::cylinder_noxfrm(float *ta, float *tb, float radius, int filled) {
  if (ta[0] == tb[0] && ta[1] == tb[1] && ta[2] == tb[2]) {
    return;  // we don't serve your kind here
  }

  float height = distance(ta, tb);

  fprintf(outfile, "Transform {\n");
  fprintf(outfile, "  translation %g %g %g\n", 
          ta[0], ta[1] + (height / 2.0), ta[2]);

  float rotaxis[3];
  float cylaxdir[3];
  float yaxis[3] = {0.0, 1.0, 0.0};

  vec_sub(cylaxdir, tb, ta);
  vec_normalize(cylaxdir);
  float dp = dot_prod(yaxis, cylaxdir);

  cross_prod(rotaxis, cylaxdir, yaxis);
  vec_normalize(rotaxis);

  // if we have decent rotation vector, use it
  if ((rotaxis[0]*rotaxis[0] + 
      rotaxis[1]*rotaxis[1] + 
      rotaxis[2]*rotaxis[2]) > 0.5) { 
    fprintf(outfile, "  center 0.0 %g 0.0\n", -(height / 2.0));
    fprintf(outfile, "  rotation %g %g %g  %g\n", 
            rotaxis[0], rotaxis[1], rotaxis[2], -acosf(dp));
  } else if (dp < -0.98) {
    // if we have denormalized rotation vector, we can assume it is
    // caused by a cylinder axis that is nearly coaxial with the Y axis.
    // If this is the case, we either perform no rotation in the case of a
    // angle cosine near 1.0, or a 180 degree rotation for a cosine near -1.
    fprintf(outfile, "  center 0.0 %g 0.0\n", -(height / 2.0));
    fprintf(outfile, "  rotation 0 0 -1  -3.14159\n");
  }
          
  fprintf(outfile, "  children [ Shape {\n");
  fprintf(outfile, "    ");
  write_cindexmaterial(colorIndex, materialIndex);

#if 0
  // draw the cylinder
  fprintf(outfile, "    geometry Cylinder { "
          "bottom %s height %g radius %g side %s top %s }\n", 
          filled ? "TRUE" : "FALSE",
          height,  
          radius, 
          "TRUE",
          filled ? "TRUE" : "FALSE");
#else
  if (filled) {
    fprintf(outfile, "    geometry Cylinder { "
            "height %g radius %g }\n", height,  radius);
  } else {
    fprintf(outfile, "    geometry VMDCyl { "
            "h %g r %g }\n", height,  radius);
  }
#endif

  fprintf(outfile, "  }]\n");
  fprintf(outfile, "}\n");
}


void Vrml2DisplayDevice::cone(float *a, float *b, float r, int /* resolution */) {
  float ta[3], tb[3], radius;

  if (a[0] == b[0] && a[1] == b[1] && a[2] == b[2]) {
    return;  // we don't serve your kind here
  }

  // transform the coordinates
  (transMat.top()).multpoint3d(a, ta);
  (transMat.top()).multpoint3d(b, tb);
  radius = scale_radius(r);

  float height = distance(ta, tb);

  fprintf(outfile, "Transform {\n");
  fprintf(outfile, "  translation %g %g %g\n", 
          ta[0], ta[1] + (height / 2.0), ta[2]);

  float rotaxis[3];
  float cylaxdir[3];
  float yaxis[3] = {0.0, 1.0, 0.0};

  vec_sub(cylaxdir, tb, ta);
  vec_normalize(cylaxdir);
  float dp = dot_prod(yaxis, cylaxdir);

  cross_prod(rotaxis, cylaxdir, yaxis);
  vec_normalize(rotaxis);

  if ((rotaxis[0]*rotaxis[0] + 
      rotaxis[1]*rotaxis[1] + 
      rotaxis[2]*rotaxis[2]) > 0.5) { 
    fprintf(outfile, "  center 0.0 %g 0.0\n", -(height / 2.0));
    fprintf(outfile, "  rotation %g %g %g  %g\n", 
            rotaxis[0], rotaxis[1], rotaxis[2], -acosf(dp));
  }
          
  fprintf(outfile, "  children [ Shape {\n");
  fprintf(outfile, "    ");
  write_cindexmaterial(colorIndex, materialIndex);

  // draw the cone
  fprintf(outfile, "    geometry Cone { bottomRadius %g height %g }\n", 
          radius, height);

  fprintf(outfile, "  }]\n");
  fprintf(outfile, "}\n");
}


// draw a triangle
void Vrml2DisplayDevice::triangle(const float *a, const float *b, const float *c, 
                                  const float *n1, const float *n2, const float *n3) {
  float ta[3], tb[3], tc[3], tn1[3], tn2[3], tn3[3];

  // transform the world coordinates
  (transMat.top()).multpoint3d(a, ta);
  (transMat.top()).multpoint3d(b, tb);
  (transMat.top()).multpoint3d(c, tc);

  // and the normals
  (transMat.top()).multnorm3d(n1, tn1);
  (transMat.top()).multnorm3d(n2, tn2);
  (transMat.top()).multnorm3d(n3, tn3);

  // ugly and wasteful, but it will work
  fprintf(outfile, "Shape {\n");
  fprintf(outfile, "  ");
  write_cindexmaterial(colorIndex, materialIndex);
  fprintf(outfile, "  geometry IndexedFaceSet { \n"); 
  fprintf(outfile, "    solid FALSE coordIndex [ 0, 1, 2, -1 ]\n");
  fprintf(outfile, "    coord Coordinate { point [ %g %g %g,  %g %g %g,  %g %g %g ] }\n",
          ta[0], ta[1], ta[2], tb[0], tb[1], tb[2], tc[0], tc[1], tc[2]);
   
  fprintf(outfile, "    normal Normal { vector [ %g %g %g, %g %g %g, %g %g %g ] }\n",
          tn1[0], tn1[1], tn1[2], tn2[0], tn2[1], tn2[2], tn3[0], tn3[1], tn3[2]);

  fprintf(outfile, "  }\n");
  fprintf(outfile, "}\n");
}


// draw a color-per-vertex triangle
void Vrml2DisplayDevice::tricolor(const float * a, const float * b, const float * c, 
                        const float * n1, const float * n2, const float * n3,
                        const float *c1, const float *c2, const float *c3) {
  float ta[3], tb[3], tc[3], tn1[3], tn2[3], tn3[3];

  // transform the world coordinates
  (transMat.top()).multpoint3d(a, ta);
  (transMat.top()).multpoint3d(b, tb);
  (transMat.top()).multpoint3d(c, tc);

  // and the normals
  (transMat.top()).multnorm3d(n1, tn1);
  (transMat.top()).multnorm3d(n2, tn2);
  (transMat.top()).multnorm3d(n3, tn3);

  // ugly and wasteful, but it will work
  fprintf(outfile, "Shape {\n");
  fprintf(outfile, "  ");
  write_cindexmaterial(colorIndex, materialIndex);
  fprintf(outfile, "  geometry IndexedFaceSet { \n"); 
  fprintf(outfile, "    solid FALSE coordIndex [ 0, 1, 2, -1 ]\n");
  fprintf(outfile, "    coord Coordinate { point [ %g %g %g,  %g %g %g,  %g %g %g ] }\n",
          ta[0], ta[1], ta[2], tb[0], tb[1], tb[2], tc[0], tc[1], tc[2]);

  fprintf(outfile, "    color Color { color [ %.3f %.3f %.3f, %.3f %.3f %.3f, %.3f %.3f %.3f ] }\n", 
          c1[0], c1[1], c1[2], c2[0], c2[1], c2[2], c3[0], c3[1], c3[2]);
   
  fprintf(outfile, "    normal Normal { vector [ %.3f %.3f %.3f, %.3f %.3f %.3f, %.3f %.3f %.3f ] }\n",
          tn1[0], tn1[1], tn1[2], tn2[0], tn2[1], tn2[2], tn3[0], tn3[1], tn3[2]);

  fprintf(outfile, "  }\n");
  fprintf(outfile, "}\n");
}


// use an efficient mesh primitve rather than individual triangles
// when possible.
void Vrml2DisplayDevice::trimesh_c4n3v3(int numverts, float * cnv,
                                        int numfacets, int * facets) {
  int i;

  fprintf(outfile, "Shape {\n");
  fprintf(outfile, "  ");
  write_cindexmaterial(colorIndex, materialIndex);
  fprintf(outfile, "  geometry IndexedFaceSet { \n"); 

  // loop over all of the facets in the mesh
  fprintf(outfile, "    coordIndex [ ");
  for (i=0; i<numfacets*3; i+=3) {
    fprintf(outfile, "%c %d, %d, %d, -1", (i==0) ? ' ' : ',',
            facets[i], facets[i+1], facets[i+2]);
  }
  fprintf(outfile, " ]\n");

  // loop over all of the vertices
  fprintf(outfile, "    coord Coordinate { point [ ");
  for (i=0; i<numverts; i++) {
    const float *v = cnv + i*10 + 7;
    float tv[3];
    (transMat.top()).multpoint3d(v, tv);
    fprintf(outfile, "%c %g %g %g", (i==0) ? ' ' : ',', tv[0], tv[1], tv[2]);
  }
  fprintf(outfile, " ] }\n");

  // loop over all of the colors
  fprintf(outfile, "    color Color { color [ ");
  for (i=0; i<numverts; i++) {
    const float *c = cnv + i*10;
    fprintf(outfile, "%c %.3f %.3f %.3f", (i==0) ? ' ' : ',', c[0], c[1], c[2]);
  }
  fprintf(outfile, " ] }\n");
   
  // loop over all of the normals
  fprintf(outfile, "    normal Normal { vector [ ");
  for (i=0; i<numverts; i++) {
    const float *n = cnv + i*10 + 4;
    float tn[3];
    (transMat.top()).multnorm3d(n, tn);
    fprintf(outfile, "%c %.3f %.3f %.3f", (i==0) ? ' ' : ',', tn[0], tn[1], tn[2]);
  }
  fprintf(outfile, " ] }\n");

  // close the IndexedFaceSet node
  fprintf(outfile, "  }\n");

  // close the shape node
  fprintf(outfile, "}\n");
}


// use an efficient mesh primitve rather than individual triangles
// when possible.
void Vrml2DisplayDevice::trimesh_c4u_n3b_v3f(unsigned char *c, char *n, 
                                             float *v, int numfacets) {
  int i;
  int numverts = 3*numfacets;

  const float ci2f = 1.0f / 255.0f; // used for uchar2float and normal conv
  const float cn2f = 1.0f / 127.5f;

  fprintf(outfile, "Shape {\n");
  fprintf(outfile, "  ");
  write_cindexmaterial(colorIndex, materialIndex);
  fprintf(outfile, "  geometry IndexedFaceSet { \n"); 

  // loop over all of the facets in the mesh
  fprintf(outfile, "    coordIndex [ ");
  for (i=0; i<numfacets*3; i+=3) {
    fprintf(outfile, "%c %d, %d, %d, -1", (i==0) ? ' ' : ',', i, i+1, i+2);
  }
  fprintf(outfile, " ]\n");

  // loop over all of the vertices
  fprintf(outfile, "    coord Coordinate { point [ ");
  for (i=0; i<numverts; i++) {
    float tv[3];
    int idx = i * 3;
    (transMat.top()).multpoint3d(&v[idx], tv);
    fprintf(outfile, "%c %g %g %g", (i==0) ? ' ' : ',', tv[0], tv[1], tv[2]);
  }
  fprintf(outfile, " ] }\n");

  // loop over all of the colors
  fprintf(outfile, "    color Color { color [ ");
  for (i=0; i<numverts; i++) {
    int idx = i * 4;

    // conversion from GLubyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
    // float = c/(2^8-1)
    fprintf(outfile, "%c %.3f %.3f %.3f",
            (i==0) ? ' ' : ',',
            c[idx  ] * ci2f,
            c[idx+1] * ci2f,
            c[idx+2] * ci2f);
  }
  fprintf(outfile, " ] }\n");
   
  // loop over all of the normals
  fprintf(outfile, "    normal Normal { vector [ ");
  for (i=0; i<numverts; i++) {
    float tn[3], ntmp[3];
    int idx = i * 3;

    // conversion from GLbyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
    // float = (2c+1)/(2^8-1)
    ntmp[0] = n[idx  ] * cn2f + ci2f;
    ntmp[1] = n[idx+1] * cn2f + ci2f;
    ntmp[2] = n[idx+2] * cn2f + ci2f;

    (transMat.top()).multnorm3d(ntmp, tn);
    fprintf(outfile, "%c %.2f %.2f %.2f", (i==0) ? ' ' : ',', tn[0], tn[1], tn[2]);
  }
  fprintf(outfile, " ] }\n");

  // close the IndexedFaceSet node
  fprintf(outfile, "  }\n");

  // close the shape node
  fprintf(outfile, "}\n");
}


// use an efficient mesh primitve rather than individual triangles
// when possible.
void Vrml2DisplayDevice::tristrip(int numverts, const float * cnv,
                                  int numstrips, const int *vertsperstrip,
                                  const int *facets) {
  int i;
  // render triangle strips one triangle at a time
  // triangle winding order is:
  //   v0, v1, v2, then v2, v1, v3, then v2, v3, v4, etc.
  int strip, v = 0;
  int stripaddr[2][3] = { {0, 1, 2}, {1, 0, 2} };

  fprintf(outfile, "Shape {\n");
  fprintf(outfile, "  ");
  write_cindexmaterial(colorIndex, materialIndex);
  fprintf(outfile, "  geometry IndexedFaceSet { \n"); 

  // loop over all of the facets in the mesh
  // emit vertex indices for each facet
  fprintf(outfile, "    coordIndex [ ");
  for (strip=0; strip < numstrips; strip++) {
    for (i=0; i<(vertsperstrip[strip] - 2); i++) {
      // render one triangle, using lookup table to fix winding order
      fprintf(outfile, "%c %d, %d, %d, -1", (i==0) ? ' ' : ',',
              facets[v + (stripaddr[i & 0x01][0])],
              facets[v + (stripaddr[i & 0x01][1])],
              facets[v + (stripaddr[i & 0x01][2])]);
      v++; // move on to next vertex
    }
    v+=2; // last two vertices are already used by last triangle
  }
  fprintf(outfile, " ]\n");

  // loop over all of the vertices
  fprintf(outfile, "    coord Coordinate { point [ ");
  for (i=0; i<numverts; i++) {
    const float *v = cnv + i*10 + 7;
    float tv[3];
    (transMat.top()).multpoint3d(v, tv);
    fprintf(outfile, "%c %g %g %g", (i==0) ? ' ' : ',', tv[0], tv[1], tv[2]);
  }
  fprintf(outfile, " ] }\n");

  // loop over all of the colors
  fprintf(outfile, "    color Color { color [ ");
  for (i=0; i<numverts; i++) {
    const float *c = cnv + i*10;
    fprintf(outfile, "%c %.3f %.3f %.3f", (i==0) ? ' ' : ',', c[0], c[1], c[2]);
  }
  fprintf(outfile, " ] }\n");
   
  // loop over all of the normals
  fprintf(outfile, "    normal Normal { vector [ ");
  for (i=0; i<numverts; i++) {
    const float *n = cnv + i*10 + 4;
    float tn[3];
    (transMat.top()).multnorm3d(n, tn);
    fprintf(outfile, "%c %.3f %.3f %.3f", (i==0) ? ' ' : ',', tn[0], tn[1], tn[2]);
  }
  fprintf(outfile, " ] }\n");

  // close the IndexedFaceSet node
  fprintf(outfile, "  }\n");

  // close the shape node
  fprintf(outfile, "}\n");
}


void Vrml2DisplayDevice::multmatrix(const Matrix4 &mat) {
}


void Vrml2DisplayDevice::load(const Matrix4 &mat) {
}


void Vrml2DisplayDevice::comment(const char *s) {
  fprintf (outfile, "# %s\n", s);
}

///////////////////// public virtual routines

// initialize the file for output
void Vrml2DisplayDevice::write_header(void) {
  fprintf(outfile, "#VRML V2.0 utf8\n");
  fprintf(outfile, "# Created with VMD: "
          "http://www.ks.uiuc.edu/Research/vmd/\n");

  // define our special node types
  fprintf(outfile, "# Define some custom nodes VMD to decrease file size\n");
  fprintf(outfile, "# custom VMD cylinder node\n");
  fprintf(outfile, "PROTO VMDCyl [\n");
  fprintf(outfile, "  field SFBool  bottom FALSE\n");
  fprintf(outfile, "  field SFFloat h      2    \n");
  fprintf(outfile, "  field SFFloat r      1    \n");
  fprintf(outfile, "  field SFBool  side   TRUE \n");
  fprintf(outfile, "  field SFBool  top    FALSE\n");
  fprintf(outfile, "  ] {\n");
  fprintf(outfile, "  Cylinder {\n"); 
  fprintf(outfile, "    bottom IS bottom\n");
  fprintf(outfile, "    height IS h     \n");
  fprintf(outfile, "    radius IS r     \n");
  fprintf(outfile, "    top    IS top   \n");
  fprintf(outfile, "  }\n");
  fprintf(outfile, "}\n\n");

  fprintf(outfile, "# custom VMD materials node\n");
  fprintf(outfile, "PROTO VMDMat [\n");
  fprintf(outfile, "  field SFFloat Ka               0.0\n"); 
  fprintf(outfile, "  field SFColor Kd               0.8 0.8 0.8\n");
  fprintf(outfile, "  field SFColor emissiveColor    0.0 0.0 0.0\n");
  fprintf(outfile, "  field SFFloat Ksx              0.0\n"); 
  fprintf(outfile, "  field SFColor Ks               0.0 0.0 0.0\n");
  fprintf(outfile, "  field SFFloat Kt               0.0\n"); 
  fprintf(outfile, "  ] {\n");
  fprintf(outfile, "  Appearance {\n");
  fprintf(outfile, "    material Material {\n");
  fprintf(outfile, "      ambientIntensity IS Ka           \n");
  fprintf(outfile, "      diffuseColor     IS Kd           \n");
  fprintf(outfile, "      emissiveColor    IS emissiveColor\n");
  fprintf(outfile, "      shininess        IS Ksx          \n");
  fprintf(outfile, "      specularColor    IS Ks           \n");
  fprintf(outfile, "      transparency     IS Kt           \n");
  fprintf(outfile, "    }\n");
  fprintf(outfile, "  }\n");
  fprintf(outfile, "}\n\n");

  fprintf(outfile, "\n");
  fprintf(outfile, "# begin the actual scene\n");
  fprintf(outfile, "Group {\n");
  fprintf(outfile, "  children [\n");

  if (backgroundmode == 1) {
    // emit background sky color gradient
    fprintf(outfile, "Background { skyColor [%g %g %g, %g %g %g, %g %g %g] ",
            backgradienttopcolor[0], // top pole
            backgradienttopcolor[1],
            backgradienttopcolor[2],
            (backgradienttopcolor[0]+backgradientbotcolor[0])/2.0f, // horizon
            (backgradientbotcolor[1]+backgradienttopcolor[1])/2.0f,
            (backgradienttopcolor[2]+backgradientbotcolor[2])/2.0f,
            backgradientbotcolor[0], // bottom pole
            backgradientbotcolor[1],
            backgradientbotcolor[2]);
    fprintf(outfile, "skyAngle [ 1.5, 3.0] }");
  } else {
    // otherwise emit constant color background sky
    fprintf(outfile, "Background { skyColor [ %g %g %g ] }",
            backColor[0], backColor[1], backColor[2]);
  }
  fprintf(outfile, "\n");
}

void Vrml2DisplayDevice::write_trailer(void) {
  fprintf(outfile, "  ]\n");
  fprintf(outfile, "}\n");
}

void Vrml2DisplayDevice::write_cindexmaterial(int cindex, int material) {
  write_colormaterial((float *) &matData[cindex], material);
}

void Vrml2DisplayDevice::write_colormaterial(float *rgb, int) {

#if 0
  // use the current material definition
  fprintf(outfile, "        appearance Appearance {\n");
  fprintf(outfile, "          material Material {\n"); 
  fprintf(outfile, "            ambientIntensity %g\n", mat_ambient);
  fprintf(outfile, "            diffuseColor %g %g %g\n",
          mat_diffuse * rgb[0],
          mat_diffuse * rgb[1],
          mat_diffuse * rgb[2]);
  fprintf(outfile, "            shininess %g\n", mat_shininess);
  fprintf(outfile, "            specularColor %g %g %g\n",
          mat_specular,
          mat_specular,
          mat_specular);
  fprintf(outfile, "            transparency %g\n", 1.0 - mat_opacity);
  fprintf(outfile, "          }\n");
  fprintf(outfile, "        }\n");
#else
  // use the current material definition
  fprintf(outfile, "appearance VMDMat { ");
  if (mat_ambient > 0.0) {
    fprintf(outfile, "Ka %g ", mat_ambient);
  } 

  fprintf(outfile, "Kd %g %g %g ",
          mat_diffuse * rgb[0],
          mat_diffuse * rgb[1],
          mat_diffuse * rgb[2]);

  if (mat_specular > 0.0) {
    fprintf(outfile, "Ksx %g ", mat_shininess);
    fprintf(outfile, "Ks %g %g %g ", mat_specular, mat_specular, mat_specular);
  }

  if (mat_opacity < 1.0) {
    fprintf(outfile, "Kt %g ", 1.0 - mat_opacity);
  }
  fprintf(outfile, " }\n");
#endif
}



