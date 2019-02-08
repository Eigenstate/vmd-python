/***************************************************************************
 *cr
 *cr		(C) Copyright 1995-2019 The Board of Trustees of the
 *cr			    University of Illinois
 *cr			     All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile: WavefrontDisplayDevice.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.30 $	$Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 *   Render to a Wavefront Object or "OBJ" file.
 *   This file format is one of the most universally supported 3-D model
 *   file formats, particular for animation software such as Maya, 
 *   3-D Studio, etc.  The file format is simple, but good enough for getting
 *   a lot of things done.  The Wavefront format only supports per-facet 
 *   colors and materials.  This code currently generates a fixed size set of 
 *   color/material entries and indexes into this set based on the active VMD
 *   color index.
 *
 ***************************************************************************/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "WavefrontDisplayDevice.h"
#include "Matrix4.h"
#include "DispCmds.h"
#include "Inform.h"
#include "utilities.h"
#include "config.h"    // for VMDVERSION string

#define DASH_LENGTH 0.02f

#define VMDGENMTLFILE 1

static int replacefileextension(char * s, 
                                const char * oldextension, 
                                const char * newextension) {
  int sz, extsz;
  sz = strlen(s);
  extsz = strlen(oldextension);

  if (strlen(newextension) != strlen(oldextension))
   return -1;

  if (extsz > sz)
    return -1;

  if (strupncmp(s + (sz - extsz), oldextension, extsz)) {
    return -1;
  }

  strcpy(s + (sz - extsz), newextension);

  return 0;
}

static char * stripleadingfilepath(const char *fullpath) {
  int i, j;
  char *s = NULL;
  int len=strlen(fullpath);
  s = (char *) calloc(1, len+1);

  // find last '/' or '\' path separator character 
  // and copy remaining string
  for (i=0,j=0; i<len; i++) {
    if (fullpath[i] == '/' || fullpath[i] == '\\')
      j=i;
  }

  // char after the last path separator begins shortened path
  if (j != 0)
    strcpy(s, fullpath+j+1); // strip leading path separators
  else
    strcpy(s, fullpath);     // nothing to strip off...

  return s;
}


// constructor ... call the parent with the right values
WavefrontDisplayDevice::WavefrontDisplayDevice(void) 
: FileRenderer("Wavefront", "Wavefront (OBJ and MTL)", "vmdscene.obj", "true") { }

// destructor
WavefrontDisplayDevice::~WavefrontDisplayDevice(void) { }

// emit a representation geometry group line
void WavefrontDisplayDevice::beginrepgeomgroup(const char *s) {
  fprintf(outfile, "g %s\n", s);
}

// emit a comment line
void WavefrontDisplayDevice::comment(const char *s) {
  fprintf(outfile, "# %s\n", s);
}

// draw a point
void WavefrontDisplayDevice::point(float * spdata) {
  float vec[3];
  // transform the world coordinates
  (transMat.top()).multpoint3d(spdata, vec);

  // draw the sphere
  fprintf(outfile, "v %5f %5f %5f\n", vec[0], vec[1], -vec[2]);
  fprintf(outfile, "p -1\n");
}

// draw a line from a to b
void WavefrontDisplayDevice::line(float *a, float*b) {
  int i, j, test;
  float dirvec[3], unitdirvec[3];
  float from[3], to[3], tmp1[3], tmp2[3];
  float len;
   
  if (lineStyle == ::SOLIDLINE) {
    // transform the world coordinates
    (transMat.top()).multpoint3d(a, from);
    (transMat.top()).multpoint3d(b, to);

    // draw the solid line
    fprintf(outfile, "v %5f %5f %5f\n", from[0], from[1], -from[2]);
    fprintf(outfile, "v %5f %5f %5f\n", to[0], to[1], -to[2]);
    fprintf(outfile, "l -1 -2\n");
  } else if (lineStyle == ::DASHEDLINE) {
     // transform the world coordinates
    (transMat.top()).multpoint3d(a, tmp1);
    (transMat.top()).multpoint3d(b, tmp2);

    // how to create a dashed line
    for(i=0; i<3; i++) {
      dirvec[i] = tmp2[i] - tmp1[i];  // vector from a to b
    }
    len = sqrtf(dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2])
;
    for(i=0;i<3;i++) {
      unitdirvec[i] = dirvec[i] / sqrtf(len); // unit vector from a to b
    }
          
    test = 1;
    i = 0;
    while(test == 1) {
      for(j=0;j<3;j++) {
        from[j] = (float) (tmp1[j] + (2*i    )*DASH_LENGTH*unitdirvec[j]);
          to[j] = (float) (tmp1[j] + (2*i + 1)*DASH_LENGTH*unitdirvec[j]);
      }

      if (fabsf(tmp1[0] - to[0]) >= fabsf(dirvec[0])) {
        for(j=0;j<3;j++)
          to[j] = tmp2[j];
        test = 0;
      }

      // draw the solid line dash
      fprintf(outfile, "v %5f %5f %5f\n", from[0], from[1], -from[2]);
      fprintf(outfile, "v %5f %5f %5f\n", to[0], to[1], -to[2]);
      fprintf(outfile, "l -1 -2\n");
      i++;
    }
  } else {
    msgErr << "WavefrontDisplayDevice: Unknown line style "
           << lineStyle << sendmsg;
  }
}




void WavefrontDisplayDevice::triangle(const float *v1, const float *v2, const float *v3, 
                                      const float *n1, const float *n2, const float *n3) {
  float a[3], b[3], c[3];
  float norm1[3], norm2[3], norm3[3];

  // transform the world coordinates
  (transMat.top()).multpoint3d(v1, a);
  (transMat.top()).multpoint3d(v2, b);
  (transMat.top()).multpoint3d(v3, c);

  // and the normals
  (transMat.top()).multnorm3d(n1, norm1);
  (transMat.top()).multnorm3d(n2, norm2);
  (transMat.top()).multnorm3d(n3, norm3);

#ifdef VMDGENMTLFILE
  // set colors
  write_cindexmaterial(colorIndex, materialIndex);
#endif
                                                       
  // draw the triangle 
  fprintf(outfile,"v %f %f %f\n", a[0], a[1], a[2]);
  fprintf(outfile,"v %f %f %f\n", b[0], b[1], b[2]);
  fprintf(outfile,"v %f %f %f\n", c[0], c[1], c[2]);
  fprintf(outfile,"vn %.4f %.4f %.4f\n", norm1[0], norm1[1], norm1[2]);
  fprintf(outfile,"vn %.4f %.4f %.4f\n", norm2[0], norm2[1], norm2[2]);
  fprintf(outfile,"vn %.4f %.4f %.4f\n", norm3[0], norm3[1], norm3[2]);
  fprintf(outfile,"f -3//-3 -2//-2 -1//-1\n");
}


// use an efficient mesh primitve rather than individual triangles
// when possible.
void WavefrontDisplayDevice::trimesh_c4n3v3(int numverts, float * cnv,
                                            int numfacets, int * facets) {
  int i;
  float vec1[3];
  float norm1[3];
  const float onethird = (1.0f / 3.0f);

  // write out list of vertices and normals
  for (i=0; i<numverts; i++) {
    int idx = i*10;

    (transMat.top()).multpoint3d(cnv + idx + 7, vec1);
    fprintf(outfile, "v %f %f %f\n", vec1[0], vec1[1], vec1[2]);

    (transMat.top()).multnorm3d(cnv + idx + 4, norm1);
    fprintf(outfile, "vn %.4f %.4f %.4f\n", norm1[0], norm1[1], norm1[2]);
  }

  // loop over all of the facets in the mesh
  for (i=0; i<numfacets*3; i+=3) {
    int v0 = facets[i    ];
    int v1 = facets[i + 1];
    int v2 = facets[i + 2];

#ifdef VMDGENMTLFILE
    // The Wavefront format does not allow per-vertex colors/materials,
    // so we use per-facet coloring, averaging the three vertex colors and
    // selecting the closest color from the VMD color table.
    const float *c1 = cnv + v0 * 10;
    const float *c2 = cnv + v1 * 10;
    const float *c3 = cnv + v2 * 10;
    float r, g, b;
    r = (c1[0] + c2[0] + c3[0]) * onethird; // average three vertex colors
    g = (c1[1] + c2[1] + c3[1]) * onethird;
    b = (c1[2] + c2[2] + c3[2]) * onethird;

    int cindex = nearest_index(r, g, b);
    write_cindexmaterial(cindex, materialIndex);
#endif

    // use negative relative indices required for wavefront obj format
    v0 -= numverts;
    v1 -= numverts;
    v2 -= numverts;
    fprintf(outfile, "f %d//%d %d//%d %d//%d\n", v0, v0, v1, v1, v2, v2);
  }
}


// use an efficient mesh primitve rather than individual triangles
// when possible.
void WavefrontDisplayDevice::trimesh_c4u_n3b_v3f(unsigned char *c, char *n,
                                                 float *v, int numfacets) {
  int i;
  float vec1[3];
  float norm1[3];
  int numverts = 3*numfacets;

  const float onethird = (1.0f / 3.0f); // used for color averaging
  const float ci2f = 1.0f / 255.0f; // used for uchar2float and normal conv
  const float cn2f = 1.0f / 127.5f;

  // write out list of vertices and normals
  for (i=0; i<numverts; i++) {
    float ntmp[3];
    int idx = i * 3;

    (transMat.top()).multpoint3d(v + idx, vec1);
    fprintf(outfile, "v %f %f %f\n", vec1[0], vec1[1], vec1[2]);

    // conversion from GLbyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
    // float = (2c+1)/(2^8-1)
    ntmp[0] = n[idx  ] * cn2f + ci2f;
    ntmp[1] = n[idx+1] * cn2f + ci2f;
    ntmp[2] = n[idx+2] * cn2f + ci2f;
    (transMat.top()).multnorm3d(ntmp, norm1);
    fprintf(outfile, "vn %.3f %.3f %.3f\n", norm1[0], norm1[1], norm1[2]);
  }

  // loop over all of the facets in the mesh
  for (i=0; i<numfacets*3; i+=3) {
    int idx;

    int v0 = i;
    int v1 = i+1;
    int v2 = i+2;

#ifdef VMDGENMTLFILE
    // The Wavefront format does not allow per-vertex colors/materials,
    // so we use per-facet coloring, averaging the three vertex colors and
    // selecting the closest color from the VMD color table.

    // conversion from GLubyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
    // float = c/(2^8-1)
    float c0[3], c1[3], c2[3];
    idx = v0 * 4;
    c0[0] = c[idx    ] * ci2f;
    c0[1] = c[idx + 1] * ci2f;
    c0[2] = c[idx + 2] * ci2f;

    idx = v1 * 4;
    c1[0] = c[idx    ] * ci2f;
    c1[1] = c[idx + 1] * ci2f;
    c1[2] = c[idx + 2] * ci2f;

    idx = v2 * 4;
    c2[0] = c[idx    ] * ci2f;
    c2[1] = c[idx + 1] * ci2f;
    c2[2] = c[idx + 2] * ci2f;

    float r, g, b;
    r = (c0[0] + c1[0] + c2[0]) * onethird; // average three vertex colors
    g = (c0[1] + c1[1] + c2[1]) * onethird;
    b = (c0[2] + c1[2] + c2[2]) * onethird;

    int cindex = nearest_index(r, g, b);
    write_cindexmaterial(cindex, materialIndex);
#endif

    // use negative relative indices required for wavefront obj format
    v0 -= numverts;
    v1 -= numverts;
    v2 -= numverts;
    fprintf(outfile, "f %d//%d %d//%d %d//%d\n", v0, v0, v1, v1, v2, v2);
  }
}


void WavefrontDisplayDevice::tristrip(int numverts, const float * cnv,
                                      int numstrips, const int *vertsperstrip,
                                      const int *facets) {
  int i, strip, t, v = 0;
  int stripaddr[2][3] = { {0, 1, 2}, {1, 0, 2} };
  float vec1[3];
  float norm1[3];
  const float onethird = (1.0f / 3.0f);

  // write out list of vertices and normals
  for (i=0; i<numverts; i++) {
    int idx = i*10;

    (transMat.top()).multpoint3d(cnv + idx + 7, vec1);
    fprintf(outfile, "v %f %f %f\n", vec1[0], vec1[1], vec1[2]);

    (transMat.top()).multnorm3d(cnv + idx + 4, norm1);
    fprintf(outfile, "vn %f %f %f\n", norm1[0], norm1[1], norm1[2]);
  }

  // render triangle strips one triangle at a time
  // triangle winding order is:
  //   v0, v1, v2, then v2, v1, v3, then v2, v3, v4, etc.
  // loop over all of the triangle strips
  for (strip=0; strip < numstrips; strip++) {
    // loop over all triangles in this triangle strip
    for (t = 0; t < (vertsperstrip[strip] - 2); t++) {
      // render one triangle, using lookup table to fix winding order
      int v0 = facets[v + (stripaddr[t & 0x01][0])];
      int v1 = facets[v + (stripaddr[t & 0x01][1])];
      int v2 = facets[v + (stripaddr[t & 0x01][2])];

#ifdef VMDGENMTLFILE
      // The Wavefront format does not allow per-vertex colors/materials,
      // so we use per-facet coloring, averaging the three vertex colors and
      // selecting the closest color from the VMD color table.
      const float *c1 = cnv + v0 * 10;
      const float *c2 = cnv + v1 * 10;
      const float *c3 = cnv + v2 * 10;
      float r, g, b;
      r = (c1[0] + c2[0] + c3[0]) * onethird; // average three vertex colors
      g = (c1[1] + c2[1] + c3[1]) * onethird;
      b = (c1[2] + c2[2] + c3[2]) * onethird;

      int cindex = nearest_index(r, g, b);
      write_cindexmaterial(cindex, materialIndex);
#endif

      // use negative relative indices required for wavefront obj format
      v0 -= numverts;
      v1 -= numverts;
      v2 -= numverts;
      fprintf(outfile, "f %d//%d %d//%d %d//%d\n", v0, v0, v1, v1, v2, v2); 
      v++; // move on to next vertex
    }
    v+=2; // last two vertices are already used by last triangle
  }
}


int WavefrontDisplayDevice::open_file(const char *filename) {
  if (isOpened) {
    close_file();
  }
  if ((outfile = fopen(filename, "w")) == NULL) {
    msgErr << "Could not open file " << filename
           << " in current directory for writing!" << sendmsg;
    return FALSE;
  }
  my_filename = stringdup(filename);

#ifdef VMDGENMTLFILE
  mtlfilename = stringdup(filename);
  if (replacefileextension(mtlfilename, ".obj", ".mtl")) {
    msgErr << "Could not generate material filename" << sendmsg;
    return FALSE;
  }
  if ((mtlfile = fopen(mtlfilename, "w")) == NULL) {
    msgErr << "Could not open file " << mtlfilename
           << " in current directory for writing!" << sendmsg;
    return FALSE;
  }
#endif

  isOpened = TRUE;
  reset_state();
  oldColorIndex = -1; 
  oldMaterialIndex = -1; 
  oldMaterialState = -1; 
  return TRUE;
}

void WavefrontDisplayDevice::close_file(void) {
  if (outfile) {
    fclose(outfile);
    outfile = NULL;
  }
  delete [] my_filename;
  my_filename = NULL;

#ifdef VMDGENMTLFILE
  if (mtlfile) {
    fclose(mtlfile);
    mtlfile = NULL;
  }
  delete [] mtlfilename;
  mtlfilename = NULL;
#endif

  isOpened = FALSE;
}

void WavefrontDisplayDevice::write_header(void) {
  fprintf(outfile, "# Wavefront OBJ file export by VMD\n");
  fprintf(outfile, "# \n");
  fprintf(outfile, "# Molecular graphics exported from VMD %s\n", VMDVERSION);
  fprintf(outfile, "# http://www.ks.uiuc.edu/Research/vmd/\n");
  fprintf(outfile, "# \n");

#ifdef VMDGENMTLFILE
  fprintf(mtlfile, "# Wavefront OBJ MTL file export by VMD\n");
  fprintf(mtlfile, "# \n");
  fprintf(mtlfile, "# Molecular graphics exported from VMD %s\n", VMDVERSION);
  fprintf(mtlfile, "# http://www.ks.uiuc.edu/Research/vmd/\n");
  fprintf(mtlfile, "# \n");

  if (mtlfilename) {
    char *shortmtlfilename = NULL;
    shortmtlfilename = stripleadingfilepath(mtlfilename);
    fprintf(outfile, "# Load Material Library paired with this scene:\n");
    fprintf(outfile, "mtllib %s\n", shortmtlfilename);
    free(shortmtlfilename);
  }

  write_material_block();
#endif
}

void WavefrontDisplayDevice::write_material_block(void) {
#ifdef VMDGENMTLFILE
   int n;

   // materials for normal lighting modes
   for (n=BEGREGCLRS; n < (BEGREGCLRS + REGCLRS + MAPCLRS); n++) {
     float rgb[3];
     fprintf(mtlfile, "newmtl vmd_mat_cindex_%d\n", n); 
     vec_scale(rgb, 0.0f, matData[n]);
     fprintf(mtlfile, "Ka %.2f %.2f %.2f\n", rgb[0], rgb[1], rgb[2]); 
     vec_scale(rgb, 0.65f, matData[n]);
     fprintf(mtlfile, "Kd %.2f %.2f %.2f\n", rgb[0], rgb[1], rgb[2]); 
     vec_scale(rgb, 0.50f, matData[n]);
     fprintf(mtlfile, "Ks %.2f %.2f %.2f\n", rgb[0], rgb[1], rgb[2]); 
     vec_scale(rgb, 0.0f, matData[n]);
     fprintf(mtlfile, "Tf %.2f %.2f %.2f\n", rgb[0], rgb[1], rgb[2]); 
     fprintf(mtlfile, "d 1.0\n");
     fprintf(mtlfile, "Ns 40.0\n");
     fprintf(mtlfile, "illum_4\n");
     fprintf(mtlfile, "\n");
   }

   // materials for non-lighted modes
   for (n=BEGREGCLRS; n < (BEGREGCLRS + REGCLRS + MAPCLRS); n++) {
     float rgb[3];
     fprintf(mtlfile, "newmtl vmd_nomat_cindex_%d\n", n); 
     vec_scale(rgb, 0.0f, matData[n]);
     fprintf(mtlfile, "Ka %.2f %.2f %.2f\n", rgb[0], rgb[1], rgb[2]); 
     vec_scale(rgb, 0.65f, matData[n]);
     fprintf(mtlfile, "Kd %.2f %.2f %.2f\n", rgb[0], rgb[1], rgb[2]); 
     fprintf(mtlfile, "illum_0\n");
     fprintf(mtlfile, "\n");
   }
#endif
}

void WavefrontDisplayDevice::write_cindexmaterial(int cindex, int material) {
#ifdef VMDGENMTLFILE
  if ((oldColorIndex != cindex) ||
      (oldMaterialIndex != material) ||
      (oldMaterialState != materials_on)) {
    if (materials_on) {
      fprintf(outfile, "usemtl vmd_mat_cindex_%d\n", cindex);
    } else {
      fprintf(outfile, "usemtl vmd_nomat_cindex_%d\n", cindex);
    }
  }
#endif
  oldMaterialIndex = material;
  oldColorIndex = cindex;
  oldMaterialState = materials_on;
}

void WavefrontDisplayDevice::write_colormaterial(float *rgb, int material) {
#ifdef VMDGENMTLFILE
  int cindex = nearest_index(rgb[0], rgb[1], rgb[2]);
  write_cindexmaterial(cindex, material);
#endif
}

void WavefrontDisplayDevice::write_trailer (void) {
  msgWarn << "Materials are not exported to Wavefront files.\n";
}

