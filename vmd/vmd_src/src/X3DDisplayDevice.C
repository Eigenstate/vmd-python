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
 *	$RCSfile: X3DDisplayDevice.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.47 $	$Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   X3D XML scene export code
 *
 * X3D XML encoding specification:
 *   http://www.web3d.org/x3d/specifications/ISO-IEC-19776-X3DEncodings-All-Edition-1/
 *
 * Feature compatibility matrix for X3D specification versions 3.0 up to 3.2
 *   http://web3d.org/x3d/specifications/ISO-IEC-19775-1.2-X3D-AbstractSpecification/Part01/versionContent.html
 *
 * General X3D information page:
 *   http://www.web3d.org/x3d/content/help.html
 *
 * List of X3D viewers at NIST:
 *   http://cic.nist.gov/vrml/vbdetect.html
 *
 ***************************************************************************/

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "X3DDisplayDevice.h"
#include "Matrix4.h"
#include "utilities.h"
#include "DispCmds.h"  // needed for line styles
#include "Hershey.h"   // needed for Hershey font rendering fctns

// The default radius for points and lines (which are displayed
// as small spheres or cylinders, respectively)
#define DEFAULT_RADIUS 0.002f
#define DASH_LENGTH 0.02f

//
// The full X3D export subclass currently uses the following node types:
//   Scene, Background, OrthoViewpoint, Viewpoint,
//   Shape, Appearance, Material, Color, Coordinate, Normal, Transform, 
//   LineProperties, IndexedLineSet, LineSet,
//   IndexedFaceSet, IndexedTriangleSet, IndexedTriangleStripSet,
//   Cone, Cylinder, PointSet, Sphere
//   ClipPlane (not quite yet)
//

///////////////////////// constructor and destructor

/// create the renderer; set the 'visible' name for the renderer list
X3DDisplayDevice::X3DDisplayDevice(
               const char *public_name,
               const char *public_pretty_name,
               const char *default_file_name,
               const char *default_command_line) :
                 FileRenderer(public_name, public_pretty_name,
                 default_file_name, default_command_line) {
}

// constructor ... initialize some variables
X3DDisplayDevice::X3DDisplayDevice(void) : 
  FileRenderer("X3D", "X3D (XML) full specification", "vmdscene.x3d", "true") {
}
               
///////////////////////// protected nonvirtual routines
void X3DDisplayDevice::set_color(int mycolorIndex) {
#if 0
  write_cindexmaterial(mycolorIndex, materialIndex);
#endif
}


void X3DDisplayDevice::text(float *pos, float size, float thickness,
                            const char *str) {
  float textpos[3];
  float textsize;
  hersheyhandle hh;

  // transform the world coordinates
  (transMat.top()).multpoint3d(pos, textpos);
  textsize = size * 1.5f;

  ResizeArray<int>   idxs; 
  ResizeArray<float> pnts;
  idxs.clear();
  pnts.clear();

  int idx=0;
  while (*str != '\0') {
    float lm, rm, x, y;
    int draw;
    x=y=0.0f;
    draw=0;

    hersheyDrawInitLetter(&hh, *str, &lm, &rm);
    textpos[0] -= lm * textsize;

    while (!hersheyDrawNextLine(&hh, &draw, &x, &y)) {
      float pt[3];

      if (draw) {
        // add another connected point to the line strip
        idxs.append(idx);

        pt[0] = textpos[0] + textsize * x;
        pt[1] = textpos[1] + textsize * y;
        pt[2] = textpos[2];

        pnts.append3(&pt[0]);

        idx++;
      } else {
        idxs.append(-1); // pen-up, end of the line strip
      }
    }
    idxs.append(-1); // pen-up, end of the line strip
    textpos[0] += rm * textsize;
    str++;
  }

  fprintf(outfile, "<Shape>\n");
  fprintf(outfile, "  ");

  // 
  // Emit the line properties
  // 
  fprintf(outfile, "<Appearance><Material ");
  fprintf(outfile, "ambientIntensity='%g' ", mat_ambient);
  fprintf(outfile, "diffuseColor='0 0 0' "); 

  const float *rgb = matData[colorIndex];
  fprintf(outfile, "emissiveColor='%.3f %.3f %.3f' ",
          mat_diffuse * rgb[0], mat_diffuse * rgb[1], mat_diffuse * rgb[2]);
  fprintf(outfile, "/>");

  // emit a line thickness directive, if needed
  if (thickness < 0.99f || thickness > 1.01f) {
    fprintf(outfile, "  <LineProperties linewidthScaleFactor=\"%g\" "
            "containerField=\"lineProperties\"/>\n",
            (double) thickness);
  }
  fprintf(outfile, "</Appearance>\n");

  //
  // Emit the line set
  // 
  fprintf(outfile, "  <IndexedLineSet coordIndex='");
  int i, cnt;
  cnt = idxs.num();
  for (i=0; i<cnt; i++) {
    fprintf(outfile, "%d ", idxs[i]);
  }
  fprintf(outfile, "'>\n");

  fprintf(outfile, "    <Coordinate point='");
  cnt = pnts.num();
  for (i=0; i<cnt; i+=3) {
    fprintf(outfile, "%c%g %g %g", 
            (i==0) ? ' ' : ',',
            pnts[i], pnts[i+1], pnts[i+2]);
  }
  fprintf(outfile, "'/>\n");
  fprintf(outfile, "  </IndexedLineSet>\n");
  fprintf(outfile, "</Shape>\n");
}


// draw a sphere
void X3DDisplayDevice::sphere(float *xyzr) {
  float cent[3], radius;

  // transform the coordinates
  (transMat.top()).multpoint3d(xyzr, cent);
  radius = scale_radius(xyzr[3]);

  fprintf(outfile, "<Transform translation='%g %g %g'>\n",
          cent[0], cent[1], cent[2]);
  fprintf(outfile, "  <Shape>\n");
  fprintf(outfile, "  ");
  write_cindexmaterial(colorIndex, materialIndex);
  fprintf(outfile, "  <Sphere radius='%g'/>\n", radius);
  fprintf(outfile, "  </Shape>\n");
  fprintf(outfile, "</Transform>\n");
}


// draw a point
void X3DDisplayDevice::point(float * xyz) {
  float txyz[3];

  // transform the coordinates
  (transMat.top()).multpoint3d(xyz, txyz);

  // ugly and wasteful, but it will work
  fprintf(outfile, "<Shape>\n");
  fprintf(outfile, "  ");

  // Emit the point material properties
  fprintf(outfile, "<Appearance><Material ");
  fprintf(outfile, "ambientIntensity='%g' ", mat_ambient);
  fprintf(outfile, "diffuseColor='0 0 0' ");

  const float *rgb = matData[colorIndex];
  fprintf(outfile, "emissiveColor='%.3f %.3f %.3f' ",
          mat_diffuse * rgb[0], mat_diffuse * rgb[1], mat_diffuse * rgb[2]);
  fprintf(outfile, "/>");
  fprintf(outfile, "</Appearance>\n");

  fprintf(outfile, "  <PointSet>\n");
  fprintf(outfile, "    <Coordinate point='%g %g %g'/>\n",
          txyz[0], txyz[1], txyz[2]);
  
  float col[3];
  vec_copy(col, matData[colorIndex]);
  fprintf(outfile, "    <Color color='%.3f %.3f %.3f'/>\n", 
          col[0], col[1], col[2]);
  fprintf(outfile, "  </PointSet>\n");
  fprintf(outfile, "</Shape>\n");
}


// draw an array of points of the same size
void X3DDisplayDevice::point_array(int num, float size,
                                   float *xyz, float *colors) {
  float txyz[3];

  // ugly and wasteful, but it will work
  fprintf(outfile, "<Shape>\n");
  fprintf(outfile, "  ");

  // Emit the point material properties
  fprintf(outfile, "<Appearance><Material ");
  fprintf(outfile, "ambientIntensity='%g' ", mat_ambient);
  fprintf(outfile, "diffuseColor='0 0 0' ");
  fprintf(outfile, "emissiveColor='1 1 1' ");
  fprintf(outfile, "/>");
  fprintf(outfile, "</Appearance>\n");

  fprintf(outfile, "  <PointSet>\n");

  int i;
  fprintf(outfile, "    <Coordinate point='");
  for (i=0; i<num; i++) {
    // transform the coordinates
    (transMat.top()).multpoint3d(&xyz[i*3], txyz);
    fprintf(outfile, "%c %g %g %g",
            (i==0) ? ' ' : ',',
            txyz[0], txyz[1], txyz[2]);
  } 
  fprintf(outfile, "'/>\n");

  fprintf(outfile, "    <Color color='");
  for (i=0; i<num; i++) {
    int cind = i*3;
    fprintf(outfile, "%c %.3f %.3f %.3f", 
            (i==0) ? ' ' : ',',
            colors[cind], colors[cind+1], colors[cind+2]);
  } 
  fprintf(outfile, "'/>\n");

  fprintf(outfile, "  </PointSet>\n");
  fprintf(outfile, "</Shape>\n");
}


/// draw a line from a to b
void X3DDisplayDevice::line(float *a, float*b) {
  float ta[3], tb[3];

  if (lineStyle == ::SOLIDLINE) {
    // transform the coordinates
    (transMat.top()).multpoint3d(a, ta);
    (transMat.top()).multpoint3d(b, tb);

    // ugly and wasteful, but it will work
    fprintf(outfile, "<Shape>\n");
    fprintf(outfile, "  ");
    write_cindexmaterial(colorIndex, materialIndex);

    fprintf(outfile, "  <IndexedLineSet coordIndex='0 1 -1'>\n");
    fprintf(outfile, "    <Coordinate point='%g %g %g, %g %g %g'/>\n",
            ta[0], ta[1], ta[2], tb[0], tb[1], tb[2]);
  
    float col[3];
    vec_copy(col, matData[colorIndex]);
    fprintf(outfile, "    <Color color='%.3f %.3f %.3f, %.3f %.3f %.3f'/>\n", 
            col[0], col[1], col[2], col[0], col[1], col[2]);
    fprintf(outfile, "  </IndexedLineSet>\n");
    fprintf(outfile, "</Shape>\n");
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
      fprintf(outfile, "<Shape>\n");
      fprintf(outfile, "  ");
      write_cindexmaterial(colorIndex, materialIndex);

      fprintf(outfile, "  <IndexedLineSet coordIndex='0 1 -1'>\n");
      fprintf(outfile, "    <Coordinate point='%g %g %g, %g %g %g'/>\n",
              ta[0], ta[1], ta[2], tb[0], tb[1], tb[2]);
  
      float col[3];
      vec_copy(col, matData[colorIndex]);
      fprintf(outfile, "    <Color color='%.3f %.3f %.3f, %.3f %.3f %.3f'/>\n", 
              col[0], col[1], col[2], col[0], col[1], col[2]);
      fprintf(outfile, "  </IndexedLineSet>\n");
      fprintf(outfile, "</Shape>\n");
      i++;
    }
  } else {
    msgErr << "X3DDisplayDevice: Unknown line style "
           << lineStyle << sendmsg;
  }
}


void X3DDisplayDevice::line_array(int num, float thickness, float *points) {
  float *v = points;
  float txyz[3];
  int i;

  fprintf(outfile, "<Shape>\n");
  fprintf(outfile, "  ");

  // Emit the line properties
  fprintf(outfile, "<Appearance><Material ");
  fprintf(outfile, "ambientIntensity='%g' ", mat_ambient);
  fprintf(outfile, "diffuseColor='0 0 0' ");

  const float *rgb = matData[colorIndex];
  fprintf(outfile, "emissiveColor='%g %g %g' ",
          mat_diffuse * rgb[0], mat_diffuse * rgb[1], mat_diffuse * rgb[2]);
  fprintf(outfile, "/>");

  // emit a line thickness directive, if needed
  if (thickness < 0.99f || thickness > 1.01f) {
    fprintf(outfile, "  <LineProperties linewidthScaleFactor=\"%g\" "
            "containerField=\"lineProperties\"/>\n",
            (double) thickness);
  }
  fprintf(outfile, "</Appearance>\n");

  // Emit the line set
  fprintf(outfile, "  <IndexedLineSet coordIndex='");
  for (i=0; i<num; i++) {
    fprintf(outfile, "%d %d -1 ", i*2, i*2+1);
  }
  fprintf(outfile, "'>\n");

  fprintf(outfile, "    <Coordinate point='");
  // write two vertices for each line
  for (i=0; i<(num*2); i++) {
    // transform the coordinates
    (transMat.top()).multpoint3d(v, txyz);
    fprintf(outfile, "%c%g %g %g",
            (i==0) ? ' ' : ',',
            txyz[0], txyz[1], txyz[2]);
    v += 3;
  }
  fprintf(outfile, "'/>\n");

  fprintf(outfile, "  </IndexedLineSet>\n");
  fprintf(outfile, "</Shape>\n");
}


void X3DDisplayDevice::polyline_array(int num, float thickness, float *points) {
  float *v = points;
  float txyz[3];

  fprintf(outfile, "<Shape>\n");
  fprintf(outfile, "  ");

  // Emit the line properties
  fprintf(outfile, "<Appearance><Material ");
  fprintf(outfile, "ambientIntensity='%g' ", mat_ambient);
  fprintf(outfile, "diffuseColor='0 0 0' ");

  const float *rgb = matData[colorIndex];
  fprintf(outfile, "emissiveColor='%g %g %g' ",
          mat_diffuse * rgb[0], mat_diffuse * rgb[1], mat_diffuse * rgb[2]);
  fprintf(outfile, "/>");

  // emit a line thickness directive, if needed
  if (thickness < 0.99f || thickness > 1.01f) {
    fprintf(outfile, "  <LineProperties linewidthScaleFactor=\"%g\" "
            "containerField=\"lineProperties\"/>\n",
            (double) thickness);
  }
  fprintf(outfile, "</Appearance>\n");

  // Emit the line set
  fprintf(outfile, "  <LineSet vertexCount='%d'>", num);

  fprintf(outfile, "    <Coordinate point='");
  for (int i=0; i<num; i++) {
    // transform the coordinates
    (transMat.top()).multpoint3d(v, txyz);
    fprintf(outfile, "%c%g %g %g",
            (i==0) ? ' ' : ',',
            txyz[0], txyz[1], txyz[2]);
    v += 3;
  }
  fprintf(outfile, "'/>\n");

  fprintf(outfile, "  </LineSet>\n");
  fprintf(outfile, "</Shape>\n");
}


// draw a cylinder
void X3DDisplayDevice::cylinder(float *a, float *b, float r, int filled) {
  float ta[3], tb[3], radius;

  // transform the coordinates
  (transMat.top()).multpoint3d(a, ta);
  (transMat.top()).multpoint3d(b, tb);
  radius = scale_radius(r);

  cylinder_noxfrm(ta, tb, radius, filled);
}


// draw a cylinder
void X3DDisplayDevice::cylinder_noxfrm(float *ta, float *tb, float radius, int filled) {
  if (ta[0] == tb[0] && ta[1] == tb[1] && ta[2] == tb[2]) {
    return;  // we don't serve your kind here
  }

  float height = distance(ta, tb);

  fprintf(outfile, "<Transform translation='%g %g %g' ",
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
    fprintf(outfile, "center='0.0 %g 0.0' ", -(height / 2.0));
    fprintf(outfile, "rotation='%g %g %g  %g'", 
            rotaxis[0], rotaxis[1], rotaxis[2], -acosf(dp));
  } else if (dp < -0.98) {
    // if we have denormalized rotation vector, we can assume it is
    // caused by a cylinder axis that is nearly coaxial with the Y axis.
    // If this is the case, we either perform no rotation in the case of a
    // angle cosine near 1.0, or a 180 degree rotation for a cosine near -1.
    fprintf(outfile, "center='0.0 %g 0.0' ", -(height / 2.0));
    fprintf(outfile, "rotation='0 0 -1 -3.14159'");
  }
  fprintf(outfile, ">\n");  
        
  fprintf(outfile, "  <Shape>\n");
  fprintf(outfile, "  ");
  write_cindexmaterial(colorIndex, materialIndex);

  // draw the cylinder
  fprintf(outfile, "    <Cylinder "
          "bottom='%s' height='%g' radius='%g' side='%s' top='%s' />\n", 
          filled ? "true" : "false",
          height,  
          radius, 
          "true",
          filled ? "true" : "false");

  fprintf(outfile, "  </Shape>\n");
  fprintf(outfile, "</Transform>\n");
}


void X3DDisplayDevice::cone(float *a, float *b, float r, int /* resolution */) {
  float ta[3], tb[3], radius;

  if (a[0] == b[0] && a[1] == b[1] && a[2] == b[2]) {
    return;  // we don't serve your kind here
  }

  // transform the coordinates
  (transMat.top()).multpoint3d(a, ta);
  (transMat.top()).multpoint3d(b, tb);
  radius = scale_radius(r);

  float height = distance(a, b);

  fprintf(outfile, "<Transform translation='%g %g %g' ", 
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
    fprintf(outfile, "center='0.0 %g 0.0' ", -(height / 2.0));
    fprintf(outfile, "rotation='%g %g %g  %g'", 
            rotaxis[0], rotaxis[1], rotaxis[2], -acosf(dp));
  }
  fprintf(outfile, ">\n");  
          
  fprintf(outfile, "  <Shape>\n");
  fprintf(outfile, "  ");
  write_cindexmaterial(colorIndex, materialIndex);

  // draw the cone
  fprintf(outfile, "  <Cone bottomRadius='%g' height='%g'/>\n", radius, height);

  fprintf(outfile, "  </Shape>\n");
  fprintf(outfile, "</Transform>\n");
}


// draw a triangle
void X3DDisplayDevice::triangle(const float *a, const float *b, const float *c, 
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

  fprintf(outfile, "<Shape>\n");
  fprintf(outfile, "  ");
  write_cindexmaterial(colorIndex, materialIndex);
  fprintf(outfile, "  <IndexedFaceSet solid='false' coordIndex='0 1 2 -1'>\n");
  fprintf(outfile, "    <Coordinate point='%g %g %g, %g %g %g, %g %g %g'/>\n",
          ta[0], ta[1], ta[2], tb[0], tb[1], tb[2], tc[0], tc[1], tc[2]);
   
  fprintf(outfile, "    <Normal vector='%g %g %g, %g %g %g, %g %g %g'/>\n",
          tn1[0], tn1[1], tn1[2], tn2[0], tn2[1], tn2[2], tn3[0], tn3[1], tn3[2]);
  fprintf(outfile, "  </IndexedFaceSet>\n");
  fprintf(outfile, "</Shape>\n");
}


// draw a color-per-vertex triangle
void X3DDisplayDevice::tricolor(const float * a, const float * b, const float * c, 
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
  fprintf(outfile, "<Shape>\n");
  fprintf(outfile, "  ");
  write_cindexmaterial(colorIndex, materialIndex);
  fprintf(outfile, "  <IndexedFaceSet solid='false' coordIndex='0 1 2 -1'>\n");
  fprintf(outfile, "    <Coordinate point='%g %g %g, %g %g %g, %g %g %g'/>\n",
          ta[0], ta[1], ta[2], tb[0], tb[1], tb[2], tc[0], tc[1], tc[2]);
   
  fprintf(outfile, "    <Normal vector='%g %g %g, %g %g %g, %g %g %g'/>\n",
          tn1[0], tn1[1], tn1[2], tn2[0], tn2[1], tn2[2], tn3[0], tn3[1], tn3[2]);
  fprintf(outfile, "    <Color color='%.3f %.3f %.3f, %.3f %.3f %.3f, %.3f %.3f %.3f'/>\n", 
          c1[0], c1[1], c1[2], c2[0], c2[1], c2[2], c3[0], c3[1], c3[2]);
  fprintf(outfile, "  </IndexedFaceSet>\n");
  fprintf(outfile, "</Shape>\n");
}


// use an efficient mesh primitve rather than individual triangles
// when possible.
void X3DDisplayDevice::trimesh_c4n3v3(int numverts, float * cnv,
                                      int numfacets, int * facets) {
  int i;

  fprintf(outfile, "<Shape>\n");
  fprintf(outfile, "  ");
  write_cindexmaterial(colorIndex, materialIndex);

  // loop over all of the facets in the mesh
  fprintf(outfile, "  <IndexedTriangleSet solid='false' index='");
  for (i=0; i<numfacets*3; i+=3) {
    fprintf(outfile, "%d %d %d ", facets[i], facets[i+1], facets[i+2]);
  }
  fprintf(outfile, "'>\n");

  // loop over all of the vertices
  fprintf(outfile, "    <Coordinate point='");
  for (i=0; i<numverts; i++) {
    const float *v = cnv + i*10 + 7;
    float tv[3];
    (transMat.top()).multpoint3d(v, tv);
    fprintf(outfile, "%c %g %g %g", (i==0) ? ' ' : ',', tv[0], tv[1], tv[2]);
  }
  fprintf(outfile, "'/>\n");

  // loop over all of the colors
  fprintf(outfile, "    <Color color='");
  for (i=0; i<numverts; i++) {
    const float *c = cnv + i*10;
    fprintf(outfile, "%c %.3f %.3f %.3f", (i==0) ? ' ' : ',', c[0], c[1], c[2]);
  }
  fprintf(outfile, "'/>\n");
   
  // loop over all of the normals
  fprintf(outfile, "    <Normal vector='");
  for (i=0; i<numverts; i++) {
    const float *n = cnv + i*10 + 4;
    float tn[3];
    (transMat.top()).multnorm3d(n, tn);

    // reduce precision of surface normals to reduce X3D file size
    fprintf(outfile, "%c %.2f %.2f %.2f", (i==0) ? ' ' : ',', tn[0], tn[1], tn[2]);
  }
  fprintf(outfile, "'/>\n");

  fprintf(outfile, "  </IndexedTriangleSet>\n");
  fprintf(outfile, "</Shape>\n");
}


// use an efficient mesh primitve rather than individual triangles
// when possible.
void X3DDisplayDevice::trimesh_c4u_n3b_v3f(unsigned char *c, char *n, float *v, int numfacets) {
  int i;
  int numverts = 3*numfacets;

  const float ci2f = 1.0f / 255.0f; // used for uchar2float and normal conv
  const float cn2f = 1.0f / 127.5f;

  fprintf(outfile, "<Shape>\n");
  fprintf(outfile, "  ");
  write_cindexmaterial(colorIndex, materialIndex);

#if 1
  fprintf(outfile, "  <TriangleSet solid='false'>\n ");
#else
  // loop over all of the facets in the mesh
  fprintf(outfile, "  <IndexedTriangleSet solid='false' index='");
  for (i=0; i<numfacets*3; i+=3) {
    fprintf(outfile, "%d %d %d ", i, i+1, i+2);
  }
  fprintf(outfile, "'>\n");
#endif

  // loop over all of the vertices
  fprintf(outfile, "    <Coordinate point='");
  for (i=0; i<numverts; i++) {
    float tv[3];
    int idx = i * 3;
    (transMat.top()).multpoint3d(&v[idx], tv);
    fprintf(outfile, "%c %g %g %g", (i==0) ? ' ' : ',', tv[0], tv[1], tv[2]);
  }
  fprintf(outfile, "'/>\n");

  // loop over all of the colors
  fprintf(outfile, "    <Color color='");
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
  fprintf(outfile, "'/>\n");
   
  // loop over all of the normals
  fprintf(outfile, "    <Normal vector='");
  for (i=0; i<numverts; i++) {
    float tn[3], ntmp[3];
    int idx = i * 3;

    // conversion from GLbyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
    // float = (2c+1)/(2^8-1)
    ntmp[0] = n[idx  ] * cn2f + ci2f;
    ntmp[1] = n[idx+1] * cn2f + ci2f;
    ntmp[2] = n[idx+2] * cn2f + ci2f;

    (transMat.top()).multnorm3d(ntmp, tn);

    // reduce precision of surface normals to reduce X3D file size
    fprintf(outfile, "%c %.2f %.2f %.2f", (i==0) ? ' ' : ',', tn[0], tn[1], tn[2]);
  }
  fprintf(outfile, "'/>\n");

#if 1
  fprintf(outfile, "  </TriangleSet>\n");
#else
  fprintf(outfile, "  </IndexedTriangleSet>\n");
#endif
  fprintf(outfile, "</Shape>\n");
}



// use an efficient mesh primitve rather than individual triangles
// when possible.
void X3DDisplayDevice::tristrip(int numverts, const float * cnv,
                                int numstrips, const int *vertsperstrip,
                                const int *facets) {
  // render directly using the IndexedTriangleStripSet primitive 
  int i, strip, v = 0;
  fprintf(outfile, "<Shape>\n");
  fprintf(outfile, "  ");
  write_cindexmaterial(colorIndex, materialIndex);

  // loop over all of the facets in the mesh
  // emit vertex indices for each facet
  fprintf(outfile, "  <IndexedTriangleStripSet solid='false' index='");
  for (strip=0; strip < numstrips; strip++) {
    for (i=0; i<vertsperstrip[strip]; i++) {
      fprintf(outfile, "%d ", facets[v]);
      v++; // move on to next vertex
    }
    fprintf(outfile, "-1 "); // mark end of strip with a -1 index
  }
  fprintf(outfile, "'>\n");

  // loop over all of the vertices
  fprintf(outfile, "    <Coordinate point='");
  for (i=0; i<numverts; i++) {
    const float *v = cnv + i*10 + 7;
    float tv[3];
    (transMat.top()).multpoint3d(v, tv);
    fprintf(outfile, "%c %g %g %g", (i==0) ? ' ' : ',', tv[0], tv[1], tv[2]);
  }
  fprintf(outfile, "'/>\n");

  // loop over all of the colors
  fprintf(outfile, "    <Color color='");
  for (i=0; i<numverts; i++) {
    const float *c = cnv + i*10;
    fprintf(outfile, "%c %.3f %.3f %.3f", (i==0) ? ' ' : ',', c[0], c[1], c[2]);
  }
  fprintf(outfile, "'/>\n");
  
  // loop over all of the normals
  fprintf(outfile, "    <Normal vector='");
  for (i=0; i<numverts; i++) {
    const float *n = cnv + i*10 + 4;
    float tn[3];
    (transMat.top()).multnorm3d(n, tn);

    // reduce precision of surface normals to reduce X3D file size
    fprintf(outfile, "%c %.2f %.2f %.2f", (i==0) ? ' ' : ',', tn[0], tn[1], tn[2]);
  }
  fprintf(outfile, "'/>\n");

  fprintf(outfile, "  </IndexedTriangleStripSet>\n");
  fprintf(outfile, "</Shape>\n");
}


void X3DDisplayDevice::multmatrix(const Matrix4 &mat) {
}


void X3DDisplayDevice::load(const Matrix4 &mat) {
}


void X3DDisplayDevice::comment(const char *s) {
  fprintf (outfile, "<!-- %s -->\n", s);
}

///////////////////// public virtual routines

// initialize the file for output
void X3DDisplayDevice::write_header(void) {
  fprintf(outfile, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
  fprintf(outfile, "<!DOCTYPE X3D PUBLIC \"ISO//Web3D//DTD X3D 3.0//EN\"\n");
  fprintf(outfile, "  \"http://www.web3d.org/specifications/x3d-3.0.dtd\">\n");
  fprintf(outfile, "\n");

  // check for special features that require newer X3D versions
  // At present the features that require the latest X3D spec include:
  //  - orthographic camera
  //  - clipping planes
  //  - various special shaders 
  if (projection() == PERSPECTIVE) {
    // if we use a perspective camera, we are compatible with X3D 3.1 spec
    fprintf(outfile, "<X3D version='3.1' profile='Interchange'>\n");
  } else {
    // if we use an orthographic camera, need the newest X3D 3.2 spec
    fprintf(outfile, "<X3D version='3.2' profile='Interchange'>\n");
  }

  fprintf(outfile, "<head>\n");
  fprintf(outfile, "  <meta name='description' content='VMD Molecular Graphics'/>\n");
  fprintf(outfile, "</head>\n");
  fprintf(outfile, "<Scene>\n");
  fprintf(outfile, "<!-- Created with VMD: -->\n");
  fprintf(outfile, "<!-- http://www.ks.uiuc.edu/Research/vmd/ -->\n");

  // export camera definition
  if (projection() == PERSPECTIVE) {
    float vfov = float(2.0*atan2((double) 0.5*vSize, (double) eyePos[2]-zDist));
    if (vfov > VMD_PI)
      vfov=float(VMD_PI); // X3D spec disallows FOV over 180 degrees

    fprintf(outfile, "<Viewpoint description=\"VMD Perspective View\" fieldOfView=\"%g\" orientation=\"0 0 -1 0\" position=\"%g %g %g\" centerOfRotation=\"0 0 0\" />\n",
            vfov, eyePos[0], eyePos[1], eyePos[2]);
  } else {
    fprintf(outfile, "<OrthoViewpoint description=\"VMD Orthographic View\" fieldOfView=\"%g %g %g %g\" orientation=\"0 0 -1 0\" position=\"%g %g %g\" centerOfRotation=\"0 0 0\" />\n",
            -Aspect*vSize/4, -vSize/4, Aspect*vSize/4, vSize/4,
            eyePos[0], eyePos[1], eyePos[2]);
  }

  if (backgroundmode == 1) {
    // emit background sky color gradient
    fprintf(outfile, "<Background skyColor='%g %g %g, %g %g %g, %g %g %g' ", 
            backgradienttopcolor[0], // top pole
            backgradienttopcolor[1],
            backgradienttopcolor[2], 
            (backgradienttopcolor[0]+backgradientbotcolor[0])/2.0f, // horizon
            (backgradientbotcolor[1]+backgradienttopcolor[1])/2.0f, 
            (backgradienttopcolor[2]+backgradientbotcolor[2])/2.0f,
            backgradientbotcolor[0], // bottom pole
            backgradientbotcolor[1],
            backgradientbotcolor[2]);
    fprintf(outfile, "skyAngle='1.5, 3.0' />");
  } else {
    // otherwise emit constant color background sky
    fprintf(outfile, "<Background skyColor='%g %g %g'/>",
            backColor[0], backColor[1], backColor[2]);
  }
  fprintf(outfile, "\n");
}

void X3DDisplayDevice::write_trailer(void) {
  fprintf(outfile, "</Scene>\n");
  fprintf(outfile, "</X3D>\n");
}

void X3DDisplayDevice::write_cindexmaterial(int cindex, int material) {
  write_colormaterial((float *) &matData[cindex], material);
}

void X3DDisplayDevice::write_colormaterial(float *rgb, int) {
  // use the current material definition
  fprintf(outfile, "<Appearance><Material "); 
  fprintf(outfile, "ambientIntensity='%g' ", mat_ambient);
  fprintf(outfile, "diffuseColor='%g %g %g' ",
          mat_diffuse * rgb[0], mat_diffuse * rgb[1], mat_diffuse * rgb[2]);
  fprintf(outfile, "shininess='%g' ", mat_shininess);
  fprintf(outfile, "specularColor='%g %g %g' ",
          mat_specular, mat_specular, mat_specular);
  fprintf(outfile, "transparency='%g' ", 1.0 - mat_opacity);
  fprintf(outfile, "/></Appearance>\n");
}



//
// Export an X3D subset that is compatible with X3DOM v1.1
//
// The X3DOM-compatible X3D subset cannot use a few of the 
// nodes that may be used in the full-feature X3D export subclass:
//   LineSet, LineProperties, IndexedTriangleStripSet, OrthoViewpoint,
//   ClipPlane
//
// The list of nodes implemented in X3DOM v1.1 is available here:
//   http://x3dom.org/x3dom/release/dumpNodeTypeTree-v1.1.html
//

///////////////////////// constructor and destructor

// constructor ... initialize some variables
X3DOMDisplayDevice::X3DOMDisplayDevice(void) :
  X3DDisplayDevice("X3DOM", "X3D (XML) limited subset for X3DOM v1.1", "vmdscene.x3d", "true") {
}


// To write an X3DOM-compatible scene file, we cannot include
// LineProperties nodes.
void X3DOMDisplayDevice::line_array(int num, float thickness, float *points) {
  float *v = points;
  float txyz[3];
  int i;

  fprintf(outfile, "<Shape>\n");
  fprintf(outfile, "  ");

  // Emit the line properties
  fprintf(outfile, "<Appearance><Material ");
  fprintf(outfile, "ambientIntensity='%g' ", mat_ambient);
  fprintf(outfile, "diffuseColor='0 0 0' ");

  const float *rgb = matData[colorIndex];
  fprintf(outfile, "emissiveColor='%g %g %g' ",
          mat_diffuse * rgb[0], mat_diffuse * rgb[1], mat_diffuse * rgb[2]);
  fprintf(outfile, "/>");

#if 0
  // XXX X3DOM v1.1 doesn't handle LineProperties nodes
  // emit a line thickness directive, if needed
  if (thickness < 0.99f || thickness > 1.01f) {
    fprintf(outfile, "  <LineProperties linewidthScaleFactor=\"%g\" "
            "containerField=\"lineProperties\"/>\n",
            (double) thickness);
  }
#endif
  fprintf(outfile, "</Appearance>\n");

  // Emit the line set
  fprintf(outfile, "  <IndexedLineSet coordIndex='");
  for (i=0; i<num; i++) {
    fprintf(outfile, "%d %d -1 ", i*2, i*2+1);
  }
  fprintf(outfile, "'>\n");

  fprintf(outfile, "    <Coordinate point='");
  // write two vertices for each line
  for (i=0; i<(num*2); i++) {
    // transform the coordinates
    (transMat.top()).multpoint3d(v, txyz);
    fprintf(outfile, "%c%g %g %g",
            (i==0) ? ' ' : ',',
            txyz[0], txyz[1], txyz[2]);
    v += 3;
  }
  fprintf(outfile, "'/>\n");

  fprintf(outfile, "  </IndexedLineSet>\n");
  fprintf(outfile, "</Shape>\n");
}


// To write an X3DOM-compatible scene file, we cannot include
// LineProperties or LineSet nodes, so we use an IndexedLineSet instead.
void X3DOMDisplayDevice::polyline_array(int num, float thickness, float *points) {
  float *v = points;
  float txyz[3];
  int i;

  fprintf(outfile, "<Shape>\n");
  fprintf(outfile, "  ");

  // Emit the line properties
  fprintf(outfile, "<Appearance><Material ");
  fprintf(outfile, "ambientIntensity='%g' ", mat_ambient);
  fprintf(outfile, "diffuseColor='0 0 0' ");

  const float *rgb = matData[colorIndex];
  fprintf(outfile, "emissiveColor='%g %g %g' ",
          mat_diffuse * rgb[0], mat_diffuse * rgb[1], mat_diffuse * rgb[2]);
  fprintf(outfile, "/>");

#if 0
  // XXX X3DOM v1.1 doesn't handle LineProperties nodes
  // emit a line thickness directive, if needed
  if (thickness < 0.99f || thickness > 1.01f) {
    fprintf(outfile, "  <LineProperties linewidthScaleFactor=\"%g\" "
            "containerField=\"lineProperties\"/>\n",
            (double) thickness);
  }
#endif
  fprintf(outfile, "</Appearance>\n");

  // Emit the line set
  // XXX X3DOM v1.1 doesn't handle LineSet nodes, 
  // so we have to use IndexedLineSet instead
  fprintf(outfile, "  <IndexedLineSet coordIndex='");
  for (i=0; i<num; i++) {
    fprintf(outfile, "%d ", i);
  }
  fprintf(outfile, "'>\n");

  fprintf(outfile, "    <Coordinate point='");
  for (i=0; i<num; i++) {
    // transform the coordinates
    (transMat.top()).multpoint3d(v, txyz);
    fprintf(outfile, "%c%g %g %g",
            (i==0) ? ' ' : ',',
            txyz[0], txyz[1], txyz[2]);
    v += 3;
  }
  fprintf(outfile, "'/>\n");

  fprintf(outfile, "  </IndexedLineSet>\n");
  fprintf(outfile, "</Shape>\n");
}


// To write an X3DOM-compatible scene file, we cannot include
// LineProperties nodes.
void X3DOMDisplayDevice::text(float *pos, float size, float thickness,
                               const char *str) {
  float textpos[3];
  float textsize;
  hersheyhandle hh;

  // transform the world coordinates
  (transMat.top()).multpoint3d(pos, textpos);
  textsize = size * 1.5f;

  ResizeArray<int>   idxs; 
  ResizeArray<float> pnts;
  idxs.clear();
  pnts.clear();

  int idx=0;
  while (*str != '\0') {
    float lm, rm, x, y;
    int draw;
    x=y=0.0f;
    draw=0;

    hersheyDrawInitLetter(&hh, *str, &lm, &rm);
    textpos[0] -= lm * textsize;

    while (!hersheyDrawNextLine(&hh, &draw, &x, &y)) {
      float pt[3];

      if (draw) {
        // add another connected point to the line strip
        idxs.append(idx);

        pt[0] = textpos[0] + textsize * x;
        pt[1] = textpos[1] + textsize * y;
        pt[2] = textpos[2];

        pnts.append3(&pt[0]);

        idx++;
      } else {
        idxs.append(-1); // pen-up, end of the line strip
      }
    }
    idxs.append(-1); // pen-up, end of the line strip
    textpos[0] += rm * textsize;
    str++;
  }

  fprintf(outfile, "<Shape>\n");
  fprintf(outfile, "  ");

  // 
  // Emit the line properties
  // 
  fprintf(outfile, "<Appearance><Material ");
  fprintf(outfile, "ambientIntensity='%g' ", mat_ambient);
  fprintf(outfile, "diffuseColor='0 0 0' "); 

  const float *rgb = matData[colorIndex];
  fprintf(outfile, "emissiveColor='%g %g %g' ",
          mat_diffuse * rgb[0], mat_diffuse * rgb[1], mat_diffuse * rgb[2]);
  fprintf(outfile, "/>");

#if 0
  // XXX X3DOM v1.1 doesn't handle LineProperties nodes
  // emit a line thickness directive, if needed
  if (thickness < 0.99f || thickness > 1.01f) {
    fprintf(outfile, "  <LineProperties linewidthScaleFactor=\"%g\" "
            "containerField=\"lineProperties\"/>\n",
            (double) thickness);
  }
#endif
  fprintf(outfile, "</Appearance>\n");

  //
  // Emit the line set
  // 
  fprintf(outfile, "  <IndexedLineSet coordIndex='");
  int i, cnt;
  cnt = idxs.num();
  for (i=0; i<cnt; i++) {
    fprintf(outfile, "%d ", idxs[i]);
  }
  fprintf(outfile, "'>\n");

  fprintf(outfile, "    <Coordinate point='");
  cnt = pnts.num();
  for (i=0; i<cnt; i+=3) {
    fprintf(outfile, "%c%g %g %g", 
            (i==0) ? ' ' : ',',
            pnts[i], pnts[i+1], pnts[i+2]);
  }
  fprintf(outfile, "'/>\n");
  fprintf(outfile, "  </IndexedLineSet>\n");
  fprintf(outfile, "</Shape>\n");
}


// Use a less-efficient, but X3DOM-compatible 
// IndexedTriangleSet primitve rather than triangle strips.
void X3DOMDisplayDevice::tristrip(int numverts, const float * cnv,
                                  int numstrips, const int *vertsperstrip,
                                  const int *facets) {
  // render triangle strips one triangle at a time
  // triangle winding order is:
  //   v0, v1, v2, then v2, v1, v3, then v2, v3, v4, etc.
  int i, strip, v = 0;
  int stripaddr[2][3] = { {0, 1, 2}, {1, 0, 2} };

  fprintf(outfile, "<Shape>\n");
  fprintf(outfile, "  ");
  write_cindexmaterial(colorIndex, materialIndex);

  // loop over all of the facets in the mesh
  // emit vertex indices for each facet
  fprintf(outfile, "  <IndexedTriangleSet solid='false' index='");
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
  fprintf(outfile, "'>\n");

  // loop over all of the vertices
  fprintf(outfile, "    <Coordinate point='");
  for (i=0; i<numverts; i++) {
    const float *v = cnv + i*10 + 7;
    float tv[3];
    (transMat.top()).multpoint3d(v, tv);
    fprintf(outfile, "%c %g %g %g", (i==0) ? ' ' : ',', tv[0], tv[1], tv[2]);
  }
  fprintf(outfile, "'/>\n");

  // loop over all of the colors
  fprintf(outfile, "    <Color color='");
  for (i=0; i<numverts; i++) {
    const float *c = cnv + i*10;
    fprintf(outfile, "%c %g %g %g", (i==0) ? ' ' : ',', c[0], c[1], c[2]);
  }
  fprintf(outfile, "'/>\n");

  // loop over all of the normals
  fprintf(outfile, "    <Normal vector='");
  for (i=0; i<numverts; i++) {
    const float *n = cnv + i*10 + 4;
    float tn[3];
    (transMat.top()).multnorm3d(n, tn);

    // reduce precision of surface normals to reduce X3D file size
    fprintf(outfile, "%c %.2f %.2f %.2f", (i==0) ? ' ' : ',', tn[0], tn[1], tn[2]);
  }
  fprintf(outfile, "'/>\n");

  fprintf(outfile, "  </IndexedTriangleSet>\n");
  fprintf(outfile, "</Shape>\n");
}

///////////////////// public virtual routines



