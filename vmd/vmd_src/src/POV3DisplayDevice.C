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
*      $RCSfile: POV3DisplayDevice.C,v $
*      $Author: johns $        $Locker:  $               $State: Exp $
*      $Revision: 1.127 $        $Date: 2019/01/17 21:21:00 $
*
***************************************************************************
* DESCRIPTION:
*
* FileRenderer type for the Persistence of Vision raytracer, v 3.5+
*
***************************************************************************/

#include <string.h>
#include <stdio.h>
#include <math.h>
#include "POV3DisplayDevice.h"
#include "Matrix4.h"
#include "Inform.h"
#include "utilities.h"
#include "DispCmds.h"  // need for line styles
#include "config.h"    // for VMDVERSION string
#include "Hershey.h"   // needed for Hershey font rendering fctns

#define DEFAULT_RADIUS  0.002f
#define DASH_LENGTH     0.02f
#define PHONG_DIVISOR  64.0f

// Enable triangle coordinate scaling hacks to prevent POV-Ray 3.x
// from emitting millions of "all determinants too small" warnings
// when rendering finely tessellated geometry. #$@!#@$@ POV-Ray....
// If/when POV-Ray gets fixed, this hack should gladly be removed.
#define POVRAY_BRAIN_DAMAGE_WORKAROUND   1 
#define POVRAY_SCALEHACK                 1000.0f

///////////////////////// constructor and destructor

// constructor ... initialize some variables

POV3DisplayDevice::POV3DisplayDevice() : FileRenderer("POV3", "POV-Ray 3.6", "vmdscene.pov", "povray +W%w +H%h -I%s -O%s.tga +D +X +A +FT") {
  reset_vars(); // initialize state variables
}
        
// destructor
POV3DisplayDevice::~POV3DisplayDevice(void) { }


void POV3DisplayDevice::reset_vars(void) {
  degenerate_triangles = 0;
  degenerate_cylinders = 0;
  degenerate_cones = 0;
  memset(&clip_on, 0, sizeof(clip_on));
  old_materialIndex = -1;
}


///////////////////////// protected routines

void POV3DisplayDevice::text(float *pos, float size, float thickness,
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

          fprintf(outfile, "VMD_cylinder(<%.8f,%.8f,%.8f>,<%.8f,%.8f,%.8f>",
                  oldpt[0], oldpt[1], -oldpt[2], newpt[0], newpt[1], -newpt[2]);
          fprintf(outfile, "%.4f,rgbt<%.3f,%.3f,%.3f,%.3f>,%d)\n",
                  textthickness, matData[colorIndex][0], matData[colorIndex][1],
                  matData[colorIndex][2], 1 - mat_opacity, 1);

          fprintf(outfile, "VMD_sphere(<%.4f,%.4f,%.4f>,%.4f,",
                  newpt[0], newpt[1], -newpt[2], textthickness);
          fprintf(outfile, "rgbt<%.3f,%.3f,%.3f,%.3f>)\n",
                  matData[colorIndex][0], matData[colorIndex][1], 
                  matData[colorIndex][2], 1 - mat_opacity);
        } else {
          // ...otherwise, just draw the next point
          fprintf(outfile, "VMD_sphere(<%.4f,%.4f,%.4f>,%.4f,",
                  newpt[0], newpt[1], -newpt[2], textthickness);
          fprintf(outfile, "rgbt<%.3f,%.3f,%.3f,%.3f>)\n",
                  matData[colorIndex][0], matData[colorIndex][1], 
                  matData[colorIndex][2], 1 - mat_opacity);
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
void POV3DisplayDevice::point(float * spdata) {
  float vec[3];
  // transform the world coordinates
  (transMat.top()).multpoint3d(spdata, vec);

//  write_materials();

  // Draw the point
  fprintf(outfile, "VMD_point(<%.4f,%.4f,%.4f>,%.4f,rgbt<%.3f,%.3f,%.3f,%.3f>)\n",
          vec[0], vec[1], -vec[2], ((float)pointSize)*DEFAULT_RADIUS, 
          matData[colorIndex][0], matData[colorIndex][1], 
          matData[colorIndex][2], 1 - mat_opacity);
}

// draw a sphere
void POV3DisplayDevice::sphere(float * spdata) {
  float vec[3];
  float radius;
    
  // transform the world coordinates
  (transMat.top()).multpoint3d(spdata, vec);
  radius = scale_radius(spdata[3]);

//  write_materials();

  // Draw the sphere
  fprintf(outfile, "VMD_sphere(<%.4f,%.4f,%.4f>,%.4f,",
    vec[0], vec[1], -vec[2], radius);
  fprintf(outfile, "rgbt<%.3f,%.3f,%.3f,%.3f>)\n",
    matData[colorIndex][0], matData[colorIndex][1], matData[colorIndex][2],
    1 - mat_opacity);
}

// draw a line from a to b
void POV3DisplayDevice::line(float *a, float*b) {
  int i, j, test;
  float dirvec[3], unitdirvec[3];
  float from[3], to[3], tmp1[3], tmp2[3];

  if (lineStyle == ::SOLIDLINE) {
    // transform the world coordinates
    (transMat.top()).multpoint3d(a, from);
    (transMat.top()).multpoint3d(b, to);

//    write_materials();

    // Draw the line
    fprintf(outfile, "VMD_line(<%.4f,%.4f,%.4f>,<%.4f,%.4f,%.4f>,",
            from[0], from[1], -from[2], to[0], to[1], -to[2]);
    fprintf(outfile, "rgbt<%.3f,%.3f,%.3f,%.3f>)\n",
      matData[colorIndex][0], matData[colorIndex][1], matData[colorIndex][2],
      1 - mat_opacity);

  } 
  else if (lineStyle == ::DASHEDLINE) {
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
        from[j] = (float) (tmp1[j] + (2*i)*DASH_LENGTH*unitdirvec[j]);
        to[j] =   (float) (tmp1[j] + (2*i + 1)*DASH_LENGTH*unitdirvec[j]);
      }
      if (fabsf(tmp1[0] - to[0]) >= fabsf(dirvec[0])) {
        vec_copy(to, tmp2);
        test = 0;
      }

//      write_materials();

      // Draw the line
      fprintf(outfile, "VMD_line(<%.4f,%.4f,%.4f>,<%.4f,%.4f,%.4f>,",
              from[0], from[1], -from[2], to[0], to[1], -to[2]);
      fprintf(outfile, "rgbt<%.3f,%.3f,%.3f,%.3f>)\n",
        matData[colorIndex][0], matData[colorIndex][1], matData[colorIndex][2],
        1 - mat_opacity);

      i++;
    }
  } 
  else {
    msgErr << "POV3DisplayDevice: Unknown line style " << lineStyle << sendmsg;
  }
}


// draw a cylinder
void POV3DisplayDevice::cylinder(float *a, float *b, float r, int filled) {
  float from[3], to[3];
  float radius;

  // transform the world coordinates
  (transMat.top()).multpoint3d(a, from);
  (transMat.top()).multpoint3d(b, to);
  radius = scale_radius(r);

  // check for degenerate cylinders
  if ( ((from[0]-to[0])*(from[0]-to[0]) +
        (from[1]-to[1])*(from[1]-to[1]) +
        (from[2]-to[2])*(from[2]-to[2])) < 1e-20 ) {
    degenerate_cylinders++;
    return;
  }

//  write_materials();
   
  fprintf(outfile, "VMD_cylinder(<%g,%g,%g>,<%g,%g,%g>",
          from[0], from[1], -from[2], to[0], to[1], -to[2]);
  fprintf(outfile, "%.4f,rgbt<%.3f,%.3f,%.3f,%.3f>,%d)\n",
          radius, matData[colorIndex][0], matData[colorIndex][1],
          matData[colorIndex][2], 1 - mat_opacity, !filled);
}

// draw a cone
void POV3DisplayDevice::cone(float *a, float *b, float r, int /* resolution */) {
  float from[3], to[3];
  float radius;
  
  // transform the world coordinates
  (transMat.top()).multpoint3d(a, from);
  (transMat.top()).multpoint3d(b, to);
  radius = scale_radius(r);

  // check for degenerate cylinders
  if ( ((from[0]-to[0])*(from[0]-to[0]) +
        (from[1]-to[1])*(from[1]-to[1]) +
        (from[2]-to[2])*(from[2]-to[2])) < 1e-20 ) {
    degenerate_cones++;
    return;
  }

//  write_materials();

  // Draw the cone
  fprintf(outfile, "VMD_cone (<%g,%g,%g>,<%g,%g,%g>,%.4f,",
          from[0], from[1], -from[2], to[0], to[1], -to[2], radius);
  fprintf(outfile, "rgbt<%.3f,%.3f,%.3f,%.3f>)\n",
          matData[colorIndex][0], matData[colorIndex][1], matData[colorIndex][2],
          1 - mat_opacity);
}

// draw a triangle using the current color
// XXX - POV-Ray doesn't support indexed color for triangles -- we need to
// use an RGB triple. Here we just use the same RGB triple for each vertex
// and call tricolor.
void POV3DisplayDevice::triangle(const float *a, const float *b, const float *c, 
                                 const float *n1, const float *n2, const float *n3) {
  float c1[3], c2[3], c3[3];

  memcpy(c1, matData[colorIndex], 3 * sizeof(float));
  memcpy(c2, matData[colorIndex], 3 * sizeof(float));
  memcpy(c3, matData[colorIndex], 3 * sizeof(float));

  tricolor(a, b, c, n1, n2, n3, c1, c2, c3);
  return;
}

// draw triangle with per-vertex colors
void POV3DisplayDevice::tricolor(const float * xyz1, const float * xyz2, const float * xyz3,
                                 const float * n1,   const float * n2,   const float * n3,
                                 const float *c1, const float *c2, const float *c3) {
  float vec1[3], vec2[3], vec3[3], norm1[3], norm2[3], norm3[3];
  float leg1[3], leg2[3], trinorm[3], ang1, ang2, ang3;

  // transform the world coordinates
  (transMat.top()).multpoint3d(xyz1, vec1);
  (transMat.top()).multpoint3d(xyz2, vec2);
  (transMat.top()).multpoint3d(xyz3, vec3);

  // and the normals
  (transMat.top()).multnorm3d(n1, norm1);
  (transMat.top()).multnorm3d(n2, norm2);
  (transMat.top()).multnorm3d(n3, norm3);

//  write_materials();

  // Don't write degenerate triangles -- those with all normals more than 90
  // degrees from triangle normal or its inverse.
  vec_sub(leg1, vec2, vec1);
  vec_sub(leg2, vec3, vec1);
  cross_prod(trinorm, leg1, leg2);
  ang1 = dot_prod(trinorm, norm1);
  ang2 = dot_prod(trinorm, norm2);
  ang3 = dot_prod(trinorm, norm3);
  if ( ((ang1 >= 0.0) || (ang2 >= 0.0) || (ang3 >= 0.0)) &&
       ((ang1 <= 0.0) || (ang2 <= 0.0) || (ang3 <= 0.0)) ) {
    degenerate_triangles++;
    return;
  }

  // If all verticies have the same color, don't bother with per-vertex
  // coloring
  if ( (c1[0] == c2[0]) && (c1[0] == c3[0]) &&
       (c1[1] == c2[1]) && (c1[1] == c3[1]) &&
       (c1[2] == c2[2]) && (c1[2] == c3[2]) ) {
    fprintf(outfile, "VMD_triangle(");
    fprintf(outfile, "<%.8g,%.8g,%.8g>,<%.8g,%.8g,%.8g>,<%.8g,%.8g,%.8g>,",
        vec1[0], vec1[1], -vec1[2], vec2[0], vec2[1], -vec2[2], 
        vec3[0], vec3[1], -vec3[2]);
    fprintf(outfile, "<%.8g,%.8g,%.8g>,<%.8g,%.8g,%.8g>,<%.8g,%.8g,%.8g>,",
        norm1[0], norm1[1], -norm1[2], norm2[0], norm2[1], -norm2[2], 
        norm3[0], norm3[1], -norm3[2]);
    fprintf(outfile, "rgbt<%.3f,%.3f,%.3f,%.3f>)\n",
        c1[0], c1[1], c1[2], 1 - mat_opacity);
  }
  else {
    fprintf(outfile, "VMD_tricolor(");
    fprintf(outfile, "<%.8g,%.8g,%.8g>,<%.8g,%.8g,%.8g>,<%.8g,%.8g,%.8g>,",
        vec1[0], vec1[1], -vec1[2], vec2[0], vec2[1], -vec2[2], 
        vec3[0], vec3[1], -vec3[2]);
    fprintf(outfile, "<%.8g,%.8g,%.8g>,<%.8g,%.8g,%.8g>,<%.8g,%.8g,%.8g>,",
        norm1[0], norm1[1], -norm1[2], norm2[0], norm2[1], -norm2[2], 
        norm3[0], norm3[1], -norm3[2]);
    fprintf(outfile, "rgbt<%.3f,%.3f,%.3f,%.3f>,rgbt<%.3f,%.3f,%.3f,%.3f>,rgbt<%.3f,%.3f,%.3f,%.3f>)\n",
        c1[0], c1[1], c1[2], 1 - mat_opacity, c2[0], c2[1], c2[2], 
        1 - mat_opacity, c3[0], c3[1], c3[2], 1 - mat_opacity);
  }
}

#if 1
// Draw a triangle mesh as a mesh2 POV-Ray object
void POV3DisplayDevice::trimesh_c4n3v3(int numverts, float *cnv, 
                                       int numfacets, int *facets) {
  int i;
 
//  write_materials();

  if (clip_on[2]) {
    fprintf(outfile, "intersection {\n");
  }
  fprintf(outfile, "mesh2 {\n");

  // Print the Vertex Vectors
  fprintf(outfile, "  vertex_vectors {\n");
  fprintf(outfile, "  %d,\n", numverts);
  for (i=0; i<numverts; i++) {
    int ind = i * 10;
    float vtmp[3];
    transMat.top().multpoint3d(cnv + ind + 7, vtmp);
#if defined(POVRAY_BRAIN_DAMAGE_WORKAROUND)
    vtmp[0] *= POVRAY_SCALEHACK;
    vtmp[1] *= POVRAY_SCALEHACK;
    vtmp[2] *= POVRAY_SCALEHACK;
#endif
    fprintf(outfile, "  <%.4f,%.4f,%.4f>,\n", vtmp[0], vtmp[1], -vtmp[2]);
  }
  fprintf(outfile, "  }\n");

  // Print the Normal Vectors
  fprintf(outfile, "  normal_vectors {\n");
  fprintf(outfile, "  %d,\n", numverts);
  for (i=0; i<numverts; i++) {
    int ind = i * 10;
    float ntmp[3];
    transMat.top().multnorm3d(cnv + ind + 4, ntmp);
    fprintf(outfile, "  <%.4f,%.4f,%.4f>,\n", ntmp[0], ntmp[1], -ntmp[2]);
  }
  fprintf(outfile, "  }\n");

  // Print the Texture List
  fprintf(outfile, "  texture_list {\n");
  fprintf(outfile, "  %d,\n", numverts);
  for (i=0; i<numverts; i++) {
    int ind = i * 10;
    float *rgb = cnv + ind;
    fprintf(outfile, "  VMDC(<%.3f,%.3f,%.3f,%.3f>)\n", 
            rgb[0], rgb[1], rgb[2], 1 - mat_opacity);
  }
  fprintf(outfile, "  }\n");

  // Face Indices
  fprintf(outfile, "  face_indices {\n");
  fprintf(outfile, "  %d\n", numfacets);
  for (i = 0; i < numfacets; i++) {
    int ind = i * 3;

    fprintf(outfile, "  <%d,%d,%d>,%d,%d,%d\n",
            facets[ind], facets[ind + 1], facets[ind + 2], 
            facets[ind], facets[ind + 1], facets[ind + 2]);
  }
  fprintf(outfile, "  }\n");

  // Object Modifiers
  fprintf(outfile, "  inside_vector <0, 0, 1>\n");
  if (clip_on[1]) {
#if defined(POVRAY_BRAIN_DAMAGE_WORKAROUND)
    fprintf(outfile, "  clipped_by { VMD_scaledclip[1] }\n");
#else
    fprintf(outfile, "  clipped_by { VMD_clip[1] }\n");
#endif
  }
  if (!shadows_enabled())
    fprintf(outfile, "  no_shadow\n");

#if defined(POVRAY_BRAIN_DAMAGE_WORKAROUND)
  Matrix4 hackmatrix;
  hackmatrix.identity();
  hackmatrix.scale(1.0f / POVRAY_SCALEHACK);
  const float *trans = hackmatrix.mat;
  fprintf(outfile, "matrix < \n");
  fprintf(outfile, "  %f, %f, %f,\n", trans[ 0], trans[ 1], trans[ 2]);
  fprintf(outfile, "  %f, %f, %f,\n", trans[ 4], trans[ 5], trans[ 6]);
  fprintf(outfile, "  %f, %f, %f,\n", trans[ 8], trans[ 9], trans[10]);
  fprintf(outfile, "  %f, %f, %f \n", trans[12], trans[13], trans[14]);
  fprintf(outfile, "> ");
#endif

  fprintf(outfile, "}\n");

  if (clip_on[2]) {
    fprintf(outfile, "  VMD_clip[2]\n");
    if (!shadows_enabled())
      fprintf(outfile, "  no_shadow\n");
    fprintf(outfile, "}\n");
  }
}

#else

// Draw a triangle mesh as a mesh2 POV-Ray object
void POV3DisplayDevice::trimesh_c4n3v3(int numverts, float *cnv, 
                                       int numfacets, int *facets) {
  float (*vert)[3], (*norm)[3], (*color)[3];
  int i, ind, v0, v1, v2, *c_index, curr_index;
 
//  write_materials();

  if (clip_on[2]) {
    fprintf(outfile, "intersection {\n");
  }
  fprintf(outfile, "mesh2 {\n");

  // Read the mesh, storing vertex coordinates, normals, and (unique) colors
  // XXX - this can use a *lot* of memory, but not as much as POV will when
  // parsing the resulting scene file.
  vert = new float[numfacets * 3][3];
  norm = new float[numfacets * 3][3];
  color = new float[numfacets * 3][3];
  c_index = new int[numfacets * 3];
  curr_index = -1;

  float prev_color[3] = { -1, -1, -1 };
  for (i = 0; i < numfacets; i++) {
    ind = i * 3;
    v0 = facets[ind    ] * 10;
    v1 = facets[ind + 1] * 10;
    v2 = facets[ind + 2] * 10;

    // transform the verticies and store them in the array
    transMat.top().multpoint3d(cnv + v0 + 7, vert[ind    ]);
    transMat.top().multpoint3d(cnv + v1 + 7, vert[ind + 1]);
    transMat.top().multpoint3d(cnv + v2 + 7, vert[ind + 2]);

    // transform the normals and store them in the array
    transMat.top().multnorm3d(cnv + v0 + 4, norm[ind    ]);
    transMat.top().multnorm3d(cnv + v1 + 4, norm[ind + 1]);
    transMat.top().multnorm3d(cnv + v2 + 4, norm[ind + 2]);

#if defined(POVRAY_BRAIN_DAMAGE_WORKAROUND)
    vert[ind    ][0] *= POVRAY_SCALEHACK;
    vert[ind    ][1] *= POVRAY_SCALEHACK;
    vert[ind    ][2] *= POVRAY_SCALEHACK;
    vert[ind + 1][0] *= POVRAY_SCALEHACK;
    vert[ind + 1][1] *= POVRAY_SCALEHACK;
    vert[ind + 1][2] *= POVRAY_SCALEHACK;
    vert[ind + 2][0] *= POVRAY_SCALEHACK;
    vert[ind + 2][1] *= POVRAY_SCALEHACK;
    vert[ind + 2][2] *= POVRAY_SCALEHACK;
#endif

    // Only store a color if it's different than the previous color,
    // this saves a lot of space for large triangle meshes.
    if (memcmp(prev_color, (cnv + v0), 3*sizeof(float)) != 0) {
      curr_index++;
      memcpy(color[curr_index], (cnv + v0), 3*sizeof(float));
      memcpy(prev_color, (cnv + v0), 3*sizeof(float));
    }
    c_index[ind] = curr_index;

    if (memcmp(prev_color, (cnv + v1), 3*sizeof(float)) != 0) {
      curr_index++;
      memcpy(color[curr_index], (cnv + v1), 3*sizeof(float));
      memcpy(prev_color, (cnv + v1), 3*sizeof(float));
    }
    c_index[ind+1] = curr_index;

    if (memcmp(prev_color, (cnv + v2), 3*sizeof(float)) != 0) {
      curr_index++;
      memcpy(color[curr_index], (cnv + v2), 3*sizeof(float));
      memcpy(prev_color, (cnv + v2), 3*sizeof(float));
    }
    c_index[ind+2] = curr_index;
  }

  // Print the Vertex Vectors
  fprintf(outfile, "  vertex_vectors {\n");
  fprintf(outfile, "  %d,\n", numfacets * 3);
  for (i = 0; i < (numfacets * 3); i++) {
    fprintf(outfile, "  <%.4f,%.4f,%.4f>,\n", 
            vert[i][0], vert[i][1], -vert[i][2]);
  }
  fprintf(outfile, "  }\n");

  // Print the Normal Vectors
  fprintf(outfile, "  normal_vectors {\n");
  fprintf(outfile, "  %d,\n", numfacets * 3);
  for (i = 0; i < (numfacets * 3); i++) {
    fprintf(outfile, "  <%.4f,%.4f,%.4f>,\n", 
            norm[i][0], norm[i][1], -norm[i][2]);
  }
  fprintf(outfile, "  }\n");

  // Print the Texture List
  fprintf(outfile, "  texture_list {\n");
  fprintf(outfile, "  %d,\n", curr_index+1);
  for (i = 0; i <= curr_index; i++) {
    fprintf(outfile, "  VMDC(<%.3f,%.3f,%.3f,%.3f>)\n", 
            color[i][0], color[i][1], color[i][2], 1 - mat_opacity);
  }
  fprintf(outfile, "  }\n");

  // Face Indices
  fprintf(outfile, "  face_indices {\n");
  fprintf(outfile, "  %d\n", numfacets);
  for (i = 0; i < numfacets; i++) {
    ind = i * 3;

    // Print three vertex/normal and color indicies.
    if ((c_index[ind] == c_index[ind+1]) && (c_index[ind] == c_index[ind+2])) {
      // Only one color index is required if the triangle doesn't use
      // per-vertex shading
      fprintf(outfile, "  <%d,%d,%d>,%d\n",
              ind, ind + 1, ind + 2, c_index[ind]);
    }
    else {
      fprintf(outfile, "  <%d,%d,%d>,%d,%d,%d\n",
              ind, ind + 1, ind + 2, 
              c_index[ind], c_index[ind+1], c_index[ind+2]);
    }
  }
  fprintf(outfile, "  }\n");

  // Object Modifiers
  fprintf(outfile, "  inside_vector <0, 0, 1>\n");
  if (clip_on[1]) {
#if defined(POVRAY_BRAIN_DAMAGE_WORKAROUND)
    fprintf(outfile, "  clipped_by { VMD_scaledclip[1] }\n");
#else
    fprintf(outfile, "  clipped_by { VMD_clip[1] }\n");
#endif
  }
  if (!shadows_enabled())
    fprintf(outfile, "  no_shadow\n");

#if defined(POVRAY_BRAIN_DAMAGE_WORKAROUND)
  Matrix4 hackmatrix;
  hackmatrix.identity();
  hackmatrix.scale(1.0f / POVRAY_SCALEHACK);
  const float *trans = hackmatrix.mat;
  fprintf(outfile, "matrix < \n");
  fprintf(outfile, "  %f, %f, %f,\n", trans[ 0], trans[ 1], trans[ 2]);
  fprintf(outfile, "  %f, %f, %f,\n", trans[ 4], trans[ 5], trans[ 6]);
  fprintf(outfile, "  %f, %f, %f,\n", trans[ 8], trans[ 9], trans[10]);
  fprintf(outfile, "  %f, %f, %f \n", trans[12], trans[13], trans[14]);
  fprintf(outfile, "> ");
#endif

  fprintf(outfile, "}\n");

  if (clip_on[2]) {
    fprintf(outfile, "  VMD_clip[2]\n");
    if (!shadows_enabled())
      fprintf(outfile, "  no_shadow\n");
    fprintf(outfile, "}\n");
  }

  delete [] vert;
  delete [] norm;
  delete [] color;
  delete [] c_index;
}
#endif


// Draw a triangle mesh as a mesh2 POV-Ray object
void POV3DisplayDevice::trimesh_c4u_n3b_v3f(unsigned char *c, char *n,
                                            float *v, int numfacets) {
  int i;
  int numverts = 3*numfacets;

  const float ci2f = 1.0f / 255.0f; // used for uchar2float and normal conv
  const float cn2f = 1.0f / 127.5f;
 
//  write_materials();

  if (clip_on[2]) {
    fprintf(outfile, "intersection {\n");
  }
  fprintf(outfile, "mesh2 {\n");

  // Read the mesh storing a list of unique colors
  float (*color)[3] = new float[numverts][3];
  int *c_index = new int[numverts];
  int curr_index = -1;
  float prev_color[3] = { -1, -1, -1 };
  for (i = 0; i < numverts; i++) {
    // Only store a color if it's different than the previous color,
    // this saves a lot of space for large triangle meshes.
    float ctmp[3];
    int ind = i * 4;
    ctmp[0] = c[ind  ] * ci2f; 
    ctmp[1] = c[ind+1] * ci2f; 
    ctmp[2] = c[ind+2] * ci2f; 
    if (memcmp(prev_color, ctmp, 3*sizeof(float)) != 0) {
      curr_index++;
      memcpy(color[curr_index], ctmp, 3*sizeof(float));
      memcpy(prev_color, ctmp, 3*sizeof(float));
    }
    c_index[i] = curr_index;
  }

  // Print the Vertex Vectors
  fprintf(outfile, "  vertex_vectors {\n");
  fprintf(outfile, "  %d,\n", numverts);
  for (i = 0; i < numverts; i++) {
    int ind = i * 3;
    float vtmp[3];
    transMat.top().multpoint3d(v+ind, vtmp);

#if defined(POVRAY_BRAIN_DAMAGE_WORKAROUND)
    vtmp[0] *= POVRAY_SCALEHACK;
    vtmp[1] *= POVRAY_SCALEHACK;
    vtmp[2] *= POVRAY_SCALEHACK;
#endif

    fprintf(outfile, "  <%.4f,%.4f,%.4f>,\n", vtmp[0], vtmp[1], -vtmp[2]);
  }
  fprintf(outfile, "  }\n");

  // Print the Normal Vectors
  fprintf(outfile, "  normal_vectors {\n");
  fprintf(outfile, "  %d,\n", numverts);
  for (i = 0; i < numverts; i++) {
    int ind = i * 3;
    float ntmp[3], ntmp2[3];

    // conversion from GLbyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
    // float = (2c+1)/(2^8-1)
    ntmp[0] = n[ind  ] * cn2f + ci2f;
    ntmp[1] = n[ind+1] * cn2f + ci2f;
    ntmp[2] = n[ind+2] * cn2f + ci2f;

    // transform the normals and store them in the array
    transMat.top().multnorm3d(ntmp, ntmp2);
    fprintf(outfile, "  <%.3f,%.3f,%.3f>,\n", ntmp2[0], ntmp2[1], -ntmp[2]);
  }
  fprintf(outfile, "  }\n");

  // Print the Texture List
  fprintf(outfile, "  texture_list {\n");
  fprintf(outfile, "  %d,\n", curr_index+1);
  for (i = 0; i <= curr_index; i++) {
    fprintf(outfile, "  VMDC(<%.3f,%.3f,%.3f,%.3f>)\n", 
            color[i][0], color[i][1], color[i][2], 1 - mat_opacity);
  }
  fprintf(outfile, "  }\n");

  // Face Indices
  fprintf(outfile, "  face_indices {\n");
  fprintf(outfile, "  %d\n", numfacets);
  for (i = 0; i < numfacets; i++) {
    int ind = i * 3;

    // Print three vertex/normal and color indicies.
    if ((c_index[ind] == c_index[ind+1]) && (c_index[ind] == c_index[ind+2])) {
      // Only one color index is required if the triangle doesn't use
      // per-vertex shading
      fprintf(outfile, "  <%d,%d,%d>,%d\n",
              ind, ind + 1, ind + 2, c_index[ind]);
    }
    else {
      fprintf(outfile, "  <%d,%d,%d>,%d,%d,%d\n",
              ind, ind + 1, ind + 2, 
              c_index[ind], c_index[ind+1], c_index[ind+2]);
    }
  }
  fprintf(outfile, "  }\n");

  // Object Modifiers
  fprintf(outfile, "  inside_vector <0, 0, 1>\n");
  if (clip_on[1]) {
#if defined(POVRAY_BRAIN_DAMAGE_WORKAROUND)
    fprintf(outfile, "  clipped_by { VMD_scaledclip[1] }\n");
#else
    fprintf(outfile, "  clipped_by { VMD_clip[1] }\n");
#endif
  }
  if (!shadows_enabled())
    fprintf(outfile, "  no_shadow\n");

#if defined(POVRAY_BRAIN_DAMAGE_WORKAROUND)
  Matrix4 hackmatrix;
  hackmatrix.identity();
  hackmatrix.scale(1.0f / POVRAY_SCALEHACK);
  const float *trans = hackmatrix.mat;
  fprintf(outfile, "matrix < \n");
  fprintf(outfile, "  %f, %f, %f,\n", trans[ 0], trans[ 1], trans[ 2]);
  fprintf(outfile, "  %f, %f, %f,\n", trans[ 4], trans[ 5], trans[ 6]);
  fprintf(outfile, "  %f, %f, %f,\n", trans[ 8], trans[ 9], trans[10]);
  fprintf(outfile, "  %f, %f, %f \n", trans[12], trans[13], trans[14]);
  fprintf(outfile, "> ");
#endif

  fprintf(outfile, "}\n");

  if (clip_on[2]) {
    fprintf(outfile, "  VMD_clip[2]\n");
    if (!shadows_enabled())
      fprintf(outfile, "  no_shadow\n");
    fprintf(outfile, "}\n");
  }

  delete [] color;
  delete [] c_index;
}


// Draw a collection of triangle strips as a mesh2 POV-Ray object
void POV3DisplayDevice::tristrip(int numverts, const float *cnv, 
                        int numstrips, const int *vertsperstrip, 
                        const int *facets) {
  int strip, v, i, numfacets;
  float (*vert)[3], (*norm)[3], (*color)[3];

  // POV-Ray does use triangle winding-order to determine the orientation of
  // a triangle. Although the default triangle macro doesn't make use of
  // this, the interior_texture property can be specified to give
  // back-facing triangles a different texture.
  int stripaddr[2][3] = { {0, 1, 2}, {1, 0, 2} };

//  write_materials();

  if (clip_on[2]) {
    fprintf(outfile, "intersection {\n");
  }
  fprintf(outfile, "mesh2 {\n");

  // Read the mesh, storing vertex coordinates, normals, and colors
  // XXX - this can use a *lot* of memory, but not as much as POV will when
  // parsing the resulting scene file.
  vert = new float[numverts][3];
  norm = new float[numverts][3];
  color = new float[numverts][3];

  for (i = 0; i < numverts; i++) {
    transMat.top().multpoint3d(cnv + i*10 + 7, vert[i]);
    transMat.top().multnorm3d(cnv + i*10 + 4, norm[i]);

#if defined(POVRAY_BRAIN_DAMAGE_WORKAROUND)
    vert[i][0] *= POVRAY_SCALEHACK;
    vert[i][1] *= POVRAY_SCALEHACK;
    vert[i][2] *= POVRAY_SCALEHACK;
#endif

    memcpy(color[i], cnv + i*10, 3*sizeof(float));
  }

  // Print the Vertex Vectors
  fprintf(outfile, "  vertex_vectors {\n");
  fprintf(outfile, "  %d,\n", numverts);
  for (i = 0; i < numverts; i++) {
    fprintf(outfile, "  <%.4f,%.4f,%.4f>,\n", 
            vert[i][0], vert[i][1], -vert[i][2]);
  }
  fprintf(outfile, "  }\n");

  // Print the Normal Vectors
  fprintf(outfile, "  normal_vectors {\n");
  fprintf(outfile, "  %d,\n", numverts);
  for (i = 0; i < numverts; i++) {
    fprintf(outfile, "  <%.4f,%.4f,%.4f>,\n", 
            norm[i][0], norm[i][1], -norm[i][2]);
  }
  fprintf(outfile, "  }\n");

  // Print the Texture List
  fprintf(outfile, "  texture_list {\n");
  fprintf(outfile, "  %d,\n", numverts);
  for (i = 0; i < numverts; i++) {
    fprintf(outfile, "  VMDC(<%.3f,%.3f,%.3f,%.3f>)\n", 
            color[i][0], color[i][1], color[i][2], 1 - mat_opacity);
  }
  fprintf(outfile, "  }\n");

  // Find the number of facets
  numfacets = 0;
  for (strip = 0; strip < numstrips; strip++) {
    numfacets += (vertsperstrip[strip] - 2);
  }

  // Print the Face Indices
  v = 0;
  fprintf(outfile, "  face_indices {\n");
  fprintf(outfile, "  %d\n", numfacets);
  for (strip = 0; strip < numstrips; strip++) {
    for (i = 0; i < (vertsperstrip[strip] - 2); i++) {
      fprintf(outfile, "  <%d,%d,%d>,%d,%d,%d\n",
              facets[v + (stripaddr[i & 0x01][0])],
              facets[v + (stripaddr[i & 0x01][1])],
              facets[v + (stripaddr[i & 0x01][2])],
              facets[v + (stripaddr[i & 0x01][0])],
              facets[v + (stripaddr[i & 0x01][1])],
              facets[v + (stripaddr[i & 0x01][2])] );
      v++;
    }
    v += 2;
  }
  fprintf(outfile, "  }\n");

  // Object Modifiers
  fprintf(outfile, "  inside_vector <0, 0, 1>\n");
  if (clip_on[1]) {
#if defined(POVRAY_BRAIN_DAMAGE_WORKAROUND)
    fprintf(outfile, "  clipped_by { VMD_scaledclip[1] }\n");
#else
    fprintf(outfile, "  clipped_by { VMD_clip[1] }\n");
#endif
  }

#if defined(POVRAY_BRAIN_DAMAGE_WORKAROUND)
  Matrix4 hackmatrix;
  hackmatrix.identity();
  hackmatrix.scale(1.0f / POVRAY_SCALEHACK);
  const float *trans = hackmatrix.mat;
  fprintf(outfile, "matrix < \n");
  fprintf(outfile, "  %f, %f, %f,\n", trans[ 0], trans[ 1], trans[ 2]);
  fprintf(outfile, "  %f, %f, %f,\n", trans[ 4], trans[ 5], trans[ 6]);
  fprintf(outfile, "  %f, %f, %f,\n", trans[ 8], trans[ 9], trans[10]);
  fprintf(outfile, "  %f, %f, %f \n", trans[12], trans[13], trans[14]);
  fprintf(outfile, "> ");
#endif

  if (!shadows_enabled())
    fprintf(outfile, "  no_shadow\n");
  fprintf(outfile, "}\n");

  if (clip_on[2]) {
    fprintf(outfile, "  VMD_clip[2]\n");
    if (!shadows_enabled())
      fprintf(outfile, "  no_shadow\n");
    fprintf(outfile, "}\n");
  }

  delete [] vert;
  delete [] norm;
  delete [] color;
}

// display a comment
void POV3DisplayDevice::comment(const char *s) {
  fprintf (outfile, "// %s\n", s);
}

///////////////////// public virtual routines

void POV3DisplayDevice::write_header() {
  long myXsize;
  float zDirection;

  // cross-eyes and side-by-side stereo split the screen; so we need
  // to cut xSize in half in this case
  myXsize = xSize;
  //if (inStereo == OPENGL_STEREO_SIDE)
  //   myXsize /= 2;
  // if (inStereo == OPENGL_STEREO_ABOVEBELOW) 
  //   myXsize *= 2;
  fprintf(outfile, "// \n");
  fprintf(outfile, "// Molecular graphics export from VMD %s\n", VMDVERSION);
  fprintf(outfile, "// http://www.ks.uiuc.edu/Research/vmd/\n");
  fprintf(outfile, "// Requires POV-Ray 3.5 or later\n");
  fprintf(outfile, "// \n");

  fprintf(outfile, "// POV 3.x input script : %s \n", my_filename);
  fprintf(outfile, "// try povray +W%ld +H%ld -I%s ", myXsize, ySize, my_filename);
  fprintf(outfile, "-O%s.tga +P +X +A +FT +C", my_filename);

  // need to disable the vista buffer when stereo rendering
  if (whichEye != DisplayDevice::NOSTEREO) fprintf(outfile, " -UV");
  fprintf(outfile, "\n");

#if 0
  msgInfo << "Default povray command line should be:" << sendmsg;

  msgInfo << "  povray +W" << myXsize << " +H" << ySize << " -I" << my_filename
          << " -O" << my_filename << ".tga +P +X +A +FT +C";
  if (whichEye != DisplayDevice::NOSTEREO) msgInfo << " -UV";
  msgInfo << sendmsg;
#endif

  // Warn the user if the plugin was compiled for a different version of POV
  // than they're using
  fprintf(outfile, "#if (version < 3.5) \n");
  fprintf(outfile, "#error \"VMD POV3DisplayDevice has been compiled for POV-Ray 3.5 or above.\\nPlease upgrade POV-Ray or recompile VMD.\"\n");
  fprintf(outfile, "#end \n");

  // Initialize POV-Ray state variables
  fprintf(outfile, "#declare VMD_clip_on=array[3] {0, 0, 0};\n");
  fprintf(outfile, "#declare VMD_clip=array[3];\n");
  fprintf(outfile, "#declare VMD_scaledclip=array[3];\n");
  fprintf(outfile, "#declare VMD_line_width=%.4f;\n", 
          ((float)lineWidth)*DEFAULT_RADIUS);

  //
  // Macros for VMD-like graphic primitives in POV.
  //
 
  // Color/Texture: save space when emitting texture lines for mesh2 primitives
  fprintf(outfile, "#macro VMDC ( C1 )\n");
  fprintf(outfile, "  texture { pigment { rgbt C1 }}\n");
  fprintf(outfile, "#end\n");

  // Point: can be quickly approximated as spheres with no shading.
  fprintf(outfile, "#macro VMD_point (P1, R1, C1)\n");
  fprintf(outfile, "  #local T = texture { finish { ambient 1.0 diffuse 0.0 phong 0.0 specular 0.0 } pigment { C1 } }\n");
  fprintf(outfile, "  #if(VMD_clip_on[2])\n");
  fprintf(outfile, "  intersection {\n");
  fprintf(outfile, "    sphere {P1, R1 texture {T} #if(VMD_clip_on[1]) clipped_by {VMD_clip[1]} #end %s}\n", (shadows_enabled()) ? "" : "no_shadow");
  fprintf(outfile, "    VMD_clip[2]\n");
  fprintf(outfile, "  }\n  #else\n");
  fprintf(outfile, "  sphere {P1, R1 texture {T} #if(VMD_clip_on[1]) clipped_by {VMD_clip[1]} #end %s}\n", (shadows_enabled()) ? "" : "no_shadow");
  fprintf(outfile, "  #end\n");
  fprintf(outfile, "#end\n");

  // Line: can be quickly approximated as cylinders with no shading
  fprintf(outfile, "#macro VMD_line (P1, P2, C1)\n");
  fprintf(outfile, "  #local T = texture { finish { ambient 1.0 diffuse 0.0 phong 0.0 specular 0.0 } pigment { C1 } }\n");
  fprintf(outfile, "  #if(VMD_clip_on[2])\n");
  fprintf(outfile, "  intersection {\n");
  fprintf(outfile, "    cylinder {P1, P2, VMD_line_width texture {T} #if(VMD_clip_on[1]) clipped_by {VMD_clip[1]} #end %s}\n", (shadows_enabled()) ? "" : "no_shadow");
  fprintf(outfile, "    VMD_clip[2]\n");
  fprintf(outfile, "  }\n  #else\n");
  fprintf(outfile, "  cylinder {P1, P2, VMD_line_width texture {T} #if(VMD_clip_on[1]) clipped_by {VMD_clip[1]} #end %s}\n", (shadows_enabled()) ? "" : "no_shadow");
  fprintf(outfile, "  #end\n");
  fprintf(outfile, "#end\n");

  // Sphere
  fprintf(outfile, "#macro VMD_sphere (P1, R1, C1)\n");
  fprintf(outfile, "  #local T = texture { pigment { C1 } }\n");
  fprintf(outfile, "  #if(VMD_clip_on[2])\n");
  fprintf(outfile, "  intersection {\n");
  fprintf(outfile, "    sphere {P1, R1 texture {T} #if(VMD_clip_on[1]) clipped_by {VMD_clip[1]} #end %s}\n", (shadows_enabled()) ? "" : "no_shadow");
  fprintf(outfile, "    VMD_clip[2]\n");
  fprintf(outfile, "  }\n  #else\n");
  fprintf(outfile, "  sphere {P1, R1 texture {T} #if(VMD_clip_on[1]) clipped_by {VMD_clip[1]} #end %s}\n", (shadows_enabled()) ? "" : "no_shadow");
  fprintf(outfile, "  #end\n");
  fprintf(outfile, "#end\n");

  // Cylinder: open iff O1 == 1
  fprintf(outfile, "#macro VMD_cylinder (P1, P2, R1, C1, O1)\n");
  fprintf(outfile, "  #local T = texture { pigment { C1 } }\n");
  fprintf(outfile, "  #if(VMD_clip_on[2])\n");
  fprintf(outfile, "  intersection {\n");
  fprintf(outfile, "    cylinder {P1, P2, R1 #if(O1) open #end texture {T} #if(VMD_clip_on[1]) clipped_by {VMD_clip[1]} #end %s}\n", (shadows_enabled()) ? "" : "no_shadow");
  fprintf(outfile, "    VMD_clip[2]\n");
  fprintf(outfile, "  }\n  #else\n");
  fprintf(outfile, "  cylinder {P1, P2, R1 #if(O1) open #end texture {T} #if(VMD_clip_on[1]) clipped_by {VMD_clip[1]} #end %s}\n", (shadows_enabled()) ? "" : "no_shadow");
  fprintf(outfile, "  #end\n");
  fprintf(outfile, "#end\n");

  // Cone: use the current lineWidth for the cap radius
  fprintf(outfile, "#macro VMD_cone (P1, P2, R1, C1)\n");
  fprintf(outfile, "  #local T = texture { pigment { C1 } }\n");
  fprintf(outfile, "  #if(VMD_clip_on[2])\n");
  fprintf(outfile, "  intersection {\n");
  fprintf(outfile, "    cone {P1, R1, P2, VMD_line_width texture {T} #if(VMD_clip_on[1]) clipped_by {VMD_clip[1]} #end %s}\n", (shadows_enabled()) ? "" : "no_shadow");
  fprintf(outfile, "    VMD_clip[2]\n");
  fprintf(outfile, "  }\n  #else\n");
  fprintf(outfile, "  cone {P1, R1, P2, VMD_line_width texture {T} #if(VMD_clip_on[1]) clipped_by {VMD_clip[1]} #end %s}\n", (shadows_enabled()) ? "" : "no_shadow");
  fprintf(outfile, "  #end\n");
  fprintf(outfile, "#end\n");

  // Triangle: single color, vertex normals
  // XXX - don't CSG clip triangles, behavior is undefined
  fprintf(outfile, "#macro VMD_triangle (P1, P2, P3, N1, N2, N3, C1)\n");
  fprintf(outfile, "  #local T = texture { pigment { C1 } }\n");
  fprintf(outfile, "  smooth_triangle {P1, N1, P2, N2, P3, N3 texture {T} #if(VMD_clip_on[1]) clipped_by {VMD_clip[1]} #end %s}\n", (shadows_enabled()) ? "" : "no_shadow");
  fprintf(outfile, "#end\n");

  // Tricolor: vertex colors and normals
  // XXX - don't CSG clip triangles, behavior is undefined
  fprintf(outfile, "#macro VMD_tricolor (P1, P2, P3, N1, N2, N3, C1, C2, C3)\n");
  fprintf(outfile, "  #local NX = P2-P1;\n");
  fprintf(outfile, "  #local NY = P3-P1;\n");
  fprintf(outfile, "  #local NZ = vcross(NX, NY);\n");
  fprintf(outfile, "  #local T = texture { pigment {\n");

  // Create a color cube with the vertex colors at three corners
  fprintf(outfile, "    average pigment_map {\n");
  fprintf(outfile, "      [1 gradient x color_map {[0 rgb 0] [1 C2*3]}]\n");
  fprintf(outfile, "      [1 gradient y color_map {[0 rgb 0] [1 C3*3]}]\n");
  fprintf(outfile, "      [1 gradient z color_map {[0 rgb 0] [1 C1*3]}]\n");
  fprintf(outfile, "    }\n");

  // Transform the cube so those corners match the triangle vertices
  fprintf(outfile, "    matrix <1.01,0,1,0,1.01,1,0,0,1,-.002,-.002,-1>\n");
  fprintf(outfile, "    matrix <NX.x,NX.y,NX.z,NY.x,NY.y,NY.z,NZ.x,NZ.y,NZ.z,P1.x,P1.y,P1.z>\n");
  fprintf(outfile, "  } }\n");

  fprintf(outfile, "  smooth_triangle {P1, N1, P2, N2, P3, N3 texture {T} #if(VMD_clip_on[1]) clipped_by {VMD_clip[1]} #end %s}\n", (shadows_enabled()) ? "" : "no_shadow");
  fprintf(outfile, "#end\n");


  // Camera position
  // POV uses a left-handed coordinate system 
  // VMD uses right-handed, so z(pov) = -z(vmd).

  switch (projection()) {

    case DisplayDevice::ORTHOGRAPHIC:

      fprintf(outfile, "camera {\n");
      fprintf(outfile, "  orthographic\n");
      fprintf(outfile, "  location <%.4f, %.4f, %.4f>\n",
              eyePos[0], eyePos[1], -eyePos[2]);
      fprintf(outfile, "  look_at <%.4f, %.4f, %.4f>\n",
              eyeDir[0], eyeDir[1], -eyeDir[2]);
      fprintf(outfile, "  up <0.0000, %.4f, 0.0000>\n", vSize / 2.0f);
      fprintf(outfile, "  right <%.4f, 0.0000, 0.0000>\n", Aspect * vSize / 2.0f);
      fprintf(outfile, "}\n");

      break;

    case DisplayDevice::PERSPECTIVE:
    default:

      if (whichEye != DisplayDevice::NOSTEREO) {
         if (whichEye == DisplayDevice::LEFTEYE)
            fprintf(outfile, "// Stereo rendering enabled. Now rendering left eye.\n");
         else
            fprintf(outfile, "// Stereo rendering enabled. Now rendering right eye.\n");

         fprintf(outfile, "// POV-Ray may give you a warning about non-perpendicular\n");
         fprintf(outfile, "// camera vectors; this is a result of the stereo rendering.\n");
         fprintf(outfile, "#warning \"You may ignore the following warning about "
                          "nonperpendicular camera vectors.\"\n");
      }

      fprintf(outfile, "camera {\n");
      fprintf(outfile, "  up <0, %.4f, 0>\n", vSize);
      fprintf(outfile, "  right <%.4f, 0, 0>\n", Aspect * vSize);
      fprintf(outfile, "  location <%.4f, %.4f, %.4f>\n",
              eyePos[0], eyePos[1], -eyePos[2]);
      fprintf(outfile, "  look_at <%.4f, %.4f, %.4f>\n",
              eyePos[0] + eyeDir[0],
              eyePos[1] + eyeDir[1],
              -(eyePos[2] + eyeDir[2]));


      // POV-Ray doesn't handle negative directions (i.e. when the image
      // plane is behind the viewpoint) well: the image should be mirrored
      // about both the x and y axes. We simulate this case by using a sky
      // vector.
      zDirection = eyePos[2] - zDist;
      if (zDirection < 0) {
        fprintf(outfile, "  direction <%.4f, %.4f, %.4f>\n",
                -eyePos[0], -eyePos[1], -zDirection);
        fprintf(outfile, "  sky <0, -1, 0>\n");
      }
      else {
        fprintf(outfile, "  direction <%.4f, %.4f, %.4f>\n",
                -eyePos[0], -eyePos[1], zDirection);
      }


      // render with depth of field, but only for perspective projection
      if (dof_enabled() && (projection() == DisplayDevice::PERSPECTIVE)) {
        msgInfo << "DoF focal blur enabled." << sendmsg;
        fprintf(outfile, "  focal_point <%g, %g, %g>\n",
                eyePos[0] + eyeDir[0]*get_dof_focal_dist(),
                eyePos[1] + eyeDir[1]*get_dof_focal_dist(),
                -(eyePos[2] + eyeDir[2]*get_dof_focal_dist()));
        fprintf(outfile, "  aperture %f\n", 
                vSize * 4.0f * get_dof_focal_dist() / get_dof_fnumber());
        fprintf(outfile, "  blur_samples 100\n");
        fprintf(outfile, "  confidence 0.9\n");
        fprintf(outfile, "  variance 1/128\n");
      }

      fprintf(outfile, "}\n");

      break;

  } // switch (projection())
        
  // Lights
  int i;
  for (i=0;i<DISP_LIGHTS;i++) {
    if (lightState[i].on) {
      // directional light source, as implemented in povray 3.5
      fprintf(outfile, "light_source { \n  <%.4f, %.4f, %.4f> \n",
              lightState[i].pos[0], lightState[i].pos[1],
              -lightState[i].pos[2]);
      fprintf(outfile, "  color rgb<%.3f, %.3f, %.3f> \n",
              lightState[i].color[0], lightState[i].color[1],
              lightState[i].color[2]);
      fprintf(outfile, "  parallel \n  point_at <0.0, 0.0, 0.0> \n}\n");
    }
  }
       

  // background color
  fprintf(outfile, "background {\n  color rgb<%.3f, %.3f, %.3f>\n}\n", 
          backColor[0], backColor[1], backColor[2]);

  // Specify background sky sphere if background gradient mode is enabled.
  if (backgroundmode == 1) {
    fprintf(outfile, "\n");
    fprintf(outfile, "sky_sphere {\n");
    fprintf(outfile, "  pigment {\n");
    fprintf(outfile, "    gradient y\n");
    fprintf(outfile, "    color_map {\n");
    fprintf(outfile, "      [ 0.0  color rgb<%.3f, %.3f, %.3f> ]\n",
            backgradientbotcolor[0], backgradientbotcolor[1], backgradientbotcolor[2]);
    fprintf(outfile, "      [ 1.0  color rgb<%.3f, %.3f, %.3f> ]\n",
            backgradienttopcolor[0], backgradienttopcolor[1], backgradienttopcolor[2]);
    fprintf(outfile, "    }\n");
    fprintf(outfile, "    scale 2\n");
    fprintf(outfile, "    translate -1\n");
    fprintf(outfile, "  }\n");
    fprintf(outfile, "}\n");
    fprintf(outfile, "\n");
  }

  // depth-cueing (fog)
  if (cueingEnabled && (get_cue_density() >= 1e-4)) {
    fprintf(outfile, "fog {\n");

    switch (cueMode) {
      case CUE_EXP2:
      case CUE_LINEAR:
      case CUE_EXP:
        // XXX We use povray's exponential fog for all cases 
        // since it doesn't currently support any other fog types yet.
        fprintf(outfile, "  distance %.4f \n", 
                (get_cue_density() >= 1e4) ? 1e-4 : 1.0/get_cue_density() );
        fprintf(outfile, "  fog_type 1 \n");
      break;

      case NUM_CUE_MODES:
        // this should never happen
        break;
    }

    // for depth-cueing, the fog color is the background color
    fprintf(outfile, "  color rgb<%.3f, %.3f, %.3f> \n",
            backColor[0], backColor[1], backColor[2] );
    fprintf(outfile, "} \n");
  }
}

void POV3DisplayDevice::write_trailer(void){
  fprintf(outfile, "// End of POV-Ray 3.x generation \n");

  if (degenerate_cones != 0) {
    msgWarn << "Skipped " << degenerate_cones 
            << " degenerate cones" << sendmsg;
  }
  if (degenerate_cylinders != 0) {
    msgWarn << "Skipped " << degenerate_cylinders 
            << " degenerate cylinders" << sendmsg;
  }
  if (degenerate_triangles != 0) {
    msgWarn << "Skipped " << degenerate_triangles 
            << " degenerate triangles" << sendmsg;
  }

  reset_vars(); // Reset variables before the next rendering.
}
    

void POV3DisplayDevice::write_materials(void) {
  if (old_materialIndex != materialIndex) {

    old_materialIndex = materialIndex;

    fprintf(outfile, "#default { texture {\n");
    fprintf(outfile, " finish { ambient %.3f diffuse %.3f",
      mat_ambient, mat_diffuse);
    fprintf(outfile, " phong 0.1 phong_size %.3f specular %.3f }\n",
      mat_shininess, mat_specular);
    fprintf(outfile, "} }\n");
  }
}


void POV3DisplayDevice::start_clipgroup(void) {
  int i, num_clipplanes[3], mode;
  float pov_clip_center[3], pov_clip_distance[VMD_MAX_CLIP_PLANE];
  float pov_clip_normal[VMD_MAX_CLIP_PLANE][3];

  write_materials();

  memset(num_clipplanes, 0, 3*sizeof(int));
  for (i = 0; i < VMD_MAX_CLIP_PLANE; i++) {
    if (clip_mode[i] != 0) {
      // Count the number of clipping planes for each clip mode
      num_clipplanes[clip_mode[i]]++;

      // Translate the plane center
      (transMat.top()).multpoint3d(clip_center[i], pov_clip_center);

      // and the normal
      (transMat.top()).multnorm3d(clip_normal[i], pov_clip_normal[i]);
      vec_negate(pov_clip_normal[i], pov_clip_normal[i]);

      // POV-Ray uses the distance from the origin to the plane for its
      // representation, instead of the plane center
      pov_clip_distance[i] = dot_prod(pov_clip_normal[i], pov_clip_center);
    }
  }

  // Define the clip object for each clip mode
  for (mode = 1; mode < 3; mode++) {
    if (num_clipplanes[mode] > 0) {
      // This flag is used within VMD to determine if clipping information
      // should be written to the scene file
      clip_on[mode] = 1;

      // This flag is used within POV to determine if clipping should be done
      // within macros
      fprintf(outfile, "#declare VMD_clip_on[%d]=1;\n", mode);

      if (num_clipplanes[mode] == 1) {
        for (i = 0; i < VMD_MAX_CLIP_PLANE; i++) {
          if (clip_mode[i] == mode) {
            if (mode == 2) {
              // Textured plane for CSG clipping
              fprintf(outfile, "#declare VMD_clip[%d] = plane { <%.4f, %.4f, %.4f>, %.4f texture { pigment { rgbt<%.3f, %.3f, %.3f, %.3f> } } }\n",
                      mode, pov_clip_normal[i][0], pov_clip_normal[i][1], 
                      -pov_clip_normal[i][2], pov_clip_distance[i],
                      clip_color[i][0], clip_color[i][1], clip_color[i][2],
                      1 - mat_opacity);
            } else {
              // Non-textured plane for non-CSG clipping
              fprintf(outfile, "#declare VMD_clip[%d] = plane { <%.4f, %.4f, %.4f>, %.4f }\n",
                      mode, pov_clip_normal[i][0], pov_clip_normal[i][1], 
                      -pov_clip_normal[i][2], pov_clip_distance[i]);

#if defined(POVRAY_BRAIN_DAMAGE_WORKAROUND)
              // Non-textured plane for non-CSG clipping, but scaled for use
              // when emitting meshes with the scaling hack.
              fprintf(outfile, "#declare VMD_scaledclip[%d] = plane { <%.4f, %.4f, %.4f>, %.4f }\n",
                      mode, pov_clip_normal[i][0], pov_clip_normal[i][1], -pov_clip_normal[i][2], 
                      pov_clip_distance[i] * POVRAY_SCALEHACK);
#endif
            }
          }
        }
      }

      // Declare the clipping object to be an intersection of planes
      else {
        fprintf(outfile, "#declare VMD_clip[%d] = intersection {\n", mode);
        for (i = 0; i < VMD_MAX_CLIP_PLANE; i++) {
          if (clip_mode[i] == mode) {
            if (mode == 2) {
              // Textured plane for CSG clipping
              fprintf(outfile, "  plane { <%.4f, %.4f, %.4f>, %.4f texture { pigment { rgbt<%.3f, %.3f, %.3f, %.3f> } } }\n",
                      pov_clip_normal[i][0], pov_clip_normal[i][1], 
                      -pov_clip_normal[i][2], pov_clip_distance[i],
                      clip_color[i][0], clip_color[i][1], clip_color[i][2],
                      1 - mat_opacity);
            } else {
              // Non-textured plane for non-CSG clipping
              fprintf(outfile, "  plane { <%.4f, %.4f, %.4f>, %.4f }\n",
                    pov_clip_normal[i][0], pov_clip_normal[i][1], 
                    -pov_clip_normal[i][2], pov_clip_distance[i]);
            }
          }
        }
        fprintf(outfile, "}\n");

#if defined(POVRAY_BRAIN_DAMAGE_WORKAROUND)
        fprintf(outfile, "#declare VMD_scaledclip[%d] = intersection {\n", mode);
        for (i = 0; i < VMD_MAX_CLIP_PLANE; i++) {
          if (clip_mode[i] == mode) {
            if (mode == 2) {
              // Textured plane for CSG clipping
              fprintf(outfile, "  plane { <%.4f, %.4f, %.4f>, %.4f texture { pigment { rgbt<%.3f, %.3f, %.3f, %.3f> } } }\n",
                      pov_clip_normal[i][0], pov_clip_normal[i][1], 
                      -pov_clip_normal[i][2], pov_clip_distance[i] * POVRAY_SCALEHACK,
                      clip_color[i][0], clip_color[i][1], clip_color[i][2],
                      1 - mat_opacity);
            } else {
              // Non-textured plane for non-CSG clipping
              fprintf(outfile, "  plane { <%.4f, %.4f, %.4f>, %.4f }\n",
                    pov_clip_normal[i][0], pov_clip_normal[i][1], 
                    -pov_clip_normal[i][2], pov_clip_distance[i] * POVRAY_SCALEHACK);
            }
          }
        }
        fprintf(outfile, "}\n");
#endif


      }

    }
  }
}

void POV3DisplayDevice::end_clipgroup(void) {
  int i;
  for (i = 0; i < 3; i++) {
    if (clip_on[i]) {
      fprintf(outfile, "#declare VMD_clip_on[%d]=0;\n", i);
      clip_on[i] = 0;
    }
  }
}

void POV3DisplayDevice::set_line_width(int new_width) {
  // XXX - find out why lineWidth is getting set outside this function!
//  if (lineWidth != new_width) {
  {
    lineWidth = new_width;
    fprintf(outfile, "#declare VMD_line_width=%.4f;\n", 
            ((float)new_width)*DEFAULT_RADIUS);
  }
}

