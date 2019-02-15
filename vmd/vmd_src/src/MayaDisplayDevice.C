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
 *	$RCSfile: MayaDisplayDevice.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.9 $	$Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 *   Render to a Maya ASCII file.  Tested with Maya 2010 and Maya 2011.
 *
 ***************************************************************************/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "MayaDisplayDevice.h"
#include "Matrix4.h"
#include "DispCmds.h"
#include "Inform.h"
#include "utilities.h"
#include "config.h"    // for VMDVERSION string

#define DASH_LENGTH 0.02

// constructor ... call the parent with the right values
MayaDisplayDevice::MayaDisplayDevice(void) 
: FileRenderer("Maya", "Maya (ASCII)", "vmdscene.ma", "true") { }

// destructor
MayaDisplayDevice::~MayaDisplayDevice(void) { }

// emit a representation geometry group line
void MayaDisplayDevice::beginrepgeomgroup(const char *s) {
  fprintf(outfile, "// g %s\n", s);
}

// emit a comment line
void MayaDisplayDevice::comment(const char *s) {
  fprintf(outfile, "// %s\n", s);
}

// draw a point
void MayaDisplayDevice::point(float * spdata) {
  float vec[3];
  // transform the world coordinates
  (transMat.top()).multpoint3d(spdata, vec);

  // draw the sphere
  fprintf(outfile, "// v %5f %5f %5f\n", vec[0], vec[1], -vec[2]);
  fprintf(outfile, "// p -1\n");
}


// draw a sphere
void MayaDisplayDevice::sphere(float * spdata) {
  float vec[3];
  float radius;
   
  // transform the world coordinates
  (transMat.top()).multpoint3d(spdata, vec);
  radius = scale_radius(spdata[3]);
  
  // draw the sphere
  fprintf(outfile, "createNode transform -n \"vmd%d\";\n", objnameindex);
  fprintf(outfile, "  setAttr \".s\" -type \"double3\" %f %f %f;\n", radius, radius, radius);
  fprintf(outfile, "  setAttr \".t\" -type \"double3\" %f %f %f;\n", vec[0], vec[1], vec[2]);
  fprintf(outfile, "parent -s -nc -r -add \"|VMDNurbSphere|nurbsSphereShape1\" \"vmd%d\";\n", objnameindex);

  char strbuf[1024];
  sprintf(strbuf, "|vmd%d|nurbsSphereShape1", objnameindex);
  write_cindexmaterial(strbuf, colorIndex, materialIndex);

  // increment object name counter
  objnameindex++;
}


// draw a line from a to b
void MayaDisplayDevice::line(float *a, float*b) {
  int i, j, test;
  float dirvec[3], unitdirvec[3];
  float from[3], to[3], tmp1[3], tmp2[3];
   
  if (lineStyle == ::SOLIDLINE) {
    // transform the world coordinates
    (transMat.top()).multpoint3d(a, from);
    (transMat.top()).multpoint3d(b, to);

    // draw the solid line
    fprintf(outfile, "// XXX lines not supported yet\n");
    fprintf(outfile, "// v %5f %5f %5f\n", from[0], from[1], -from[2]);
    fprintf(outfile, "// v %5f %5f %5f\n", to[0], to[1], -to[2]);
    fprintf(outfile, "// l -1 -2\n");
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

      // draw the solid line dash
      fprintf(outfile, "// XXX lines not supported yet\n");
      fprintf(outfile, "// v %5f %5f %5f\n", from[0], from[1], -from[2]);
      fprintf(outfile, "// v %5f %5f %5f\n", to[0], to[1], -to[2]);
      fprintf(outfile, "// l -1 -2\n");
      i++;
    }
  } else {
    msgErr << "MayaDisplayDevice: Unknown line style "
           << lineStyle << sendmsg;
  }
}



void MayaDisplayDevice::triangle(const float *v1, const float *v2, const float *v3, 
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

  // draw the triangle
  fprintf(outfile, "createNode transform -n \"vmd%d\";\n", objnameindex);
  fprintf(outfile, "createNode mesh -n \"vmd%dShape\" -p \"vmd%d\";\n", objnameindex, objnameindex);
  fprintf(outfile, "  setAttr -k off \".v\";\n");
  fprintf(outfile, "  setAttr -s 1 \".iog[0].og\";\n");
  fprintf(outfile, "  setAttr \".iog[0].og[0].gcl\" -type \"componentList\" 1 \"f[0]\";\n");
  fprintf(outfile, "  setAttr \".uvst[0].uvsn\" -type \"string\" \"map1\";\n");
  fprintf(outfile, "  setAttr \".cuvs\" -type \"string\" \"map1\";\n");
  fprintf(outfile, "  setAttr \".dcc\" -type \"string\" \"Ambient+Diffuse\";\n");
  fprintf(outfile, "  setAttr -s 3 \".vt[0:2]\" \n"); 
  fprintf(outfile, "    %f %f %f\n", a[0], a[1], a[2]);
  fprintf(outfile, "    %f %f %f\n", b[0], b[1], b[2]);
  fprintf(outfile, "    %f %f %f ;\n", c[0], c[1], c[2]);
  fprintf(outfile, "  setAttr -s 3 \".ed[0:2]\"  \n");
  fprintf(outfile, "    0 1 0\n");
  fprintf(outfile, "    1 2 0\n");
  fprintf(outfile, "    2 0 0 ;\n");
  fprintf(outfile, "  setAttr -s 3 \".n[0:2]\" -type \"float3\"  \n");
  fprintf(outfile, "    %f %f %f\n", norm1[0], norm1[1], norm1[2]);
  fprintf(outfile, "    %f %f %f\n", norm2[0], norm2[1], norm2[2]);
  fprintf(outfile, "    %f %f %f ;\n", norm3[0], norm3[1], norm3[2]);
  fprintf(outfile, "  setAttr -s 1 \".fc[0]\" -type \"polyFaces\"\n"); 
  fprintf(outfile, "    f 3 0 1 2 ;\n");
  fprintf(outfile, "  setAttr \".cd\" -type \"dataPolyComponent\" Index_Data Edge 0 ;\n");
  fprintf(outfile, "  setAttr \".cvd\" -type \"dataPolyComponent\" Index_Data Vertex 0 ;\n");

  char strbuf[1024];
  sprintf(strbuf, "|vmd%d|vmd%dShape", objnameindex, objnameindex);
  write_cindexmaterial(strbuf, colorIndex, materialIndex);

  // increment object name counter
  objnameindex++;
}

#if 0
void MayaDisplayDevice::tricolor(const float *v1, const float *v2, const float *v3, 
                                 const float *n1, const float *n2, const float *n3,
                                 const float *c1, const float *c2, const float *c3) {
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

  // draw the triangle
  fprintf(outfile, "createNode transform -n \"vmd%d\";\n", objnameindex);
  fprintf(outfile, "createNode mesh -n \"vmd%dShape\" -p \"vmd%d\";\n", objnameindex, objnameindex);
  fprintf(outfile, "  setAttr -k off \".v\";\n");
  fprintf(outfile, "  setAttr -s 1 \".iog[0].og\";\n");
  fprintf(outfile, "  setAttr \".iog[0].og[0].gcl\" -type \"componentList\" 1 \"f[0]\";\n");
  fprintf(outfile, "  setAttr \".uvst[0].uvsn\" -type \"string\" \"map1\";\n");
  fprintf(outfile, "  setAttr \".cuvs\" -type \"string\" \"map1\";\n");

  fprintf(outfile, "  setAttr \".dcol\" yes;\n");
  fprintf(outfile, "  setAttr \".dcc\" -type \"string\" \"Ambient+Diffuse\";\n");
// ????
  fprintf(outfile, "  setAttr \".ccls\" -type \"string\" \"colorSet1\";\n");
  fprintf(outfile, "  setAttr \".clst[0].clsn\" -type \"string\" \"colorSet1\";\n");

#if 0
  fprintf(outfile, "  setAttr -s 3 \".clst[0].clsp[0:2]\" -type \"colorSet1\" \n");
  fprintf(outfile, "    %f %f %f 1.0\n", c1[0], c1[1], c1[2]);
  fprintf(outfile, "    %f %f %f 1.0\n", c2[0], c2[1], c2[2]);
  fprintf(outfile, "    %f %f %f 1.0 ;\n", c3[0], c3[1], c3[2]);
#elif 0
  fprintf(outfile, "  setAttr -s 3 \".vclr[0].vfcl\";\n");
  fprintf(outfile, "  setAttr \".vclr[0].vfcl[0].frgb\" -type \"float3\" %f %f %f ;\n", c1[0], c1[1], c1[2]);
  fprintf(outfile, "  setAttr \".vclr[0].vfcl[1].frgb\" -type \"float3\" %f %f %f ;\n", c2[0], c2[1], c2[2]);
  fprintf(outfile, "  setAttr \".vclr[0].vfcl[2].frgb\" -type \"float3\" %f %f %f ;\n", c3[0], c3[1], c3[2]);
#elif 0
  fprintf(outfile, "  setAttr \".vclr[0].vrgb\" -type \"float3\" ");
  fprintf(outfile, "    %f %f %f ;\n", c1[0], c1[1], c1[2]);
  fprintf(outfile, "  setAttr \".vclr[1].vrgb\" -type \"float3\" ");
  fprintf(outfile, "    %f %f %f ;\n", c2[0], c2[1], c2[2]);
  fprintf(outfile, "  setAttr \".vclr[2].vrgb\" -type \"float3\" "); 
  fprintf(outfile, "    %f %f %f ;\n", c3[0], c3[1], c3[2]);
#endif

  fprintf(outfile, "  setAttr -s 3 \".vt[0:2]\" \n"); 
  fprintf(outfile, "    %f %f %f\n", a[0], a[1], a[2]);
  fprintf(outfile, "    %f %f %f\n", b[0], b[1], b[2]);
  fprintf(outfile, "    %f %f %f ;\n", c[0], c[1], c[2]);

#if 1
  fprintf(outfile, "  setAttr -s 3 \".clr[0:2]\" \n");
  fprintf(outfile, "    %f %f %f 1\n", c1[0], c1[1], c1[2]);
  fprintf(outfile, "    %f %f %f 1\n", c2[0], c2[1], c2[2]);
  fprintf(outfile, "    %f %f %f 1 ;\n", c3[0], c3[1], c3[2]);
#endif

  fprintf(outfile, "  setAttr -s 3 \".ed[0:2]\"  \n");
  fprintf(outfile, "    0 1 0\n");
  fprintf(outfile, "    1 2 0\n");
  fprintf(outfile, "    2 0 0 ;\n");
  fprintf(outfile, "  setAttr -s 3 \".n[0:2]\" -type \"float3\"  \n");
  fprintf(outfile, "    %f %f %f\n", norm1[0], norm1[1], norm1[2]);
  fprintf(outfile, "    %f %f %f\n", norm2[0], norm2[1], norm2[2]);
  fprintf(outfile, "    %f %f %f ;\n", norm3[0], norm3[1], norm3[2]);
  fprintf(outfile, "  setAttr -s 1 \".fc[0]\" -type \"polyFaces\"\n"); 
  fprintf(outfile, "    f 3 0 1 2 ;\n");
  fprintf(outfile, "  setAttr \".cd\" -type \"dataPolyComponent\" Index_Data Edge 0 ;\n");
  fprintf(outfile, "  setAttr \".cvd\" -type \"dataPolyComponent\" Index_Data Vertex 0 ;\n");

#if 0
  fprintf(outfile, "createNode polyColorPerVertex -n \"polyColorPerVertex%d\";\n", objnameindex);
  fprintf(outfile, "  setAttr \".uopa\" yes;\n");
  fprintf(outfile, "  setAttr -s 3 \".vclr\";\n");
  fprintf(outfile, "  setAttr -s 1 \".vclr[0].vfcl\";\n");
  fprintf(outfile, "  setAttr \".vclr[0].vfcl[0].frgb\" -type \"float3\" %f %f %f ;\n", c1[0], c1[1], c1[2]);
  fprintf(outfile, "  setAttr \".vclr[1].vfcl[0].frgb\" -type \"float3\" %f %f %f ;\n", c2[0], c2[1], c2[2]);
  fprintf(outfile, "  setAttr \".vclr[1].vfcl[0].frgb\" -type \"float3\" %f %f %f ;\n", c3[0], c3[1], c3[2]);
  fprintf(outfile, "  setAttr \".cn\" -type \"string\" \"colorSet1\";\n");
  fprintf(outfile, "  setAttr \".clam\" no;\n");
//  fprintf(outfile, "  connectAttr \"polyColorPerVertex%d.out\" \"vmd%dShape.i\";\n", objnameindex, objnameindex);
//  fprintf(outfile, "  connectAttr \"vmd%d.out\" \"polyColorPerVertex%d.ip\";\n", objnameindex, objnameindex);
#endif


  char strbuf[1024];
  sprintf(strbuf, "|vmd%d|vmd%dShape", objnameindex, objnameindex);
  write_cindexmaterial(strbuf, colorIndex, materialIndex);

  // increment object name counter
  objnameindex++;
}
#endif

int MayaDisplayDevice::open_file(const char *filename) {
  if (isOpened) {
    close_file();
  }
  if ((outfile = fopen(filename, "w")) == NULL) {
    msgErr << "Could not open file " << filename
           << " in current directory for writing!" << sendmsg;
    return FALSE;
  }
  my_filename = stringdup(filename);
  isOpened = TRUE;
  reset_state();
  objnameindex = 0;
  oldColorIndex = -1; 
  oldMaterialIndex = -1; 
  oldMaterialState = -1; 
  return TRUE;
}

void MayaDisplayDevice::close_file(void) {
  if (outfile) {
    fclose(outfile);
    outfile = NULL;
  }
  delete [] my_filename;
  my_filename = NULL;
  isOpened = FALSE;
}

void MayaDisplayDevice::write_header(void) {
  fprintf(outfile, "//Maya ASCII 2010 scene\n");
  fprintf(outfile, "//Codeset: UTF-8\n");
  fprintf(outfile, "requires maya \"2010\";\n");
  fprintf(outfile, "currentUnit -l centimeter -a degree -t film;\n");
  fprintf(outfile, "fileInfo \"application\" \"vmd\";\n");
  fprintf(outfile, "fileInfo \"product\" \"VMD %s\";\n", VMDVERSION);
  fprintf(outfile, "fileInfo \"version\" \"%s\";\n", VMD_ARCH);
 
  write_material_block();

  fprintf(outfile, "// VMD template objects for instancing/copying geometry\n");
  fprintf(outfile, "createNode transform -n \"VMDNurbSphere\";\n");
  // hide the template NURBS Sphere by default
  fprintf(outfile, "  setAttr \".v\" no;\n");

  fprintf(outfile, "createNode nurbsSurface -n \"nurbsSphereShape1\" -p \"VMDNurbSphere\";\n");
  fprintf(outfile, "  setAttr -k off \".v\";\n");
  fprintf(outfile, "  setAttr \".vir\" yes;\n");
  fprintf(outfile, "  setAttr \".vif\" yes;\n");
  fprintf(outfile, "  setAttr \".tw\" yes;\n");
  fprintf(outfile, "  setAttr \".covm[0]\"  0 1 1;\n");
  fprintf(outfile, "  setAttr \".cdvm[0]\"  0 1 1;\n");
  fprintf(outfile, "  setAttr \".dvu\" 0;\n");
  fprintf(outfile, "  setAttr \".dvv\" 0;\n");
  fprintf(outfile, "  setAttr \".cpr\" 3;\n");
  fprintf(outfile, "  setAttr \".cps\" 3;\n");
  fprintf(outfile, "  setAttr \".nufa\" 4.5;\n");
  fprintf(outfile, "  setAttr \".nvfa\" 4.5;\n");

  fprintf(outfile, "createNode makeNurbSphere -n \"makeNurbSphere1\";\n");
  fprintf(outfile, "  setAttr \".ax\" -type \"double3\" 0 1 0;\n");
  fprintf(outfile, "  setAttr \".r\" 1.0;\n");

  fprintf(outfile, "connectAttr \"makeNurbSphere1.os\" \"nurbsSphereShape1.cr\";\n");
  fprintf(outfile, "connectAttr \"|VMDNurbSphere|nurbsSphereShape1.iog\" \":initialShadingGroup.dsm\" -na;\n");

  fprintf(outfile, "// End of template objects\n");
}

void MayaDisplayDevice::write_material_block(void) {
   int n;

   // materials for normal lighting modes
   for (n=BEGREGCLRS; n < (BEGREGCLRS + REGCLRS + MAPCLRS); n++) {
     fprintf(outfile, "createNode blinn -n \"vmd_mat_cindex_%d\";\n", n);
     fprintf(outfile, "  setAttr \".c\" -type \"float3\" %f %f %f;\n", 
             matData[n][0], matData[n][1], matData[n][2]);
     fprintf(outfile, "createNode shadingEngine -n \"vmd_mat_cindex_%d_SG\";\n", n);
     fprintf(outfile, "  setAttr \".ihi\" 0;\n");
     fprintf(outfile, "  setAttr \".ro\" yes;\n");
     fprintf(outfile, "createNode materialInfo -n \"vmd_mat_info_cindex_%d\";\n", n);
     fprintf(outfile, "connectAttr \"vmd_mat_cindex_%d.oc\" \"vmd_mat_cindex_%d_SG.ss\";\n", n, n);
     fprintf(outfile, "connectAttr \"vmd_mat_cindex_%d_SG.msg\" \"vmd_mat_info_cindex_%d.sg\";\n", n, n);
     fprintf(outfile, "connectAttr \"vmd_mat_cindex_%d.msg\" \"vmd_mat_info_cindex_%d.m\";\n", n, n);
     fprintf(outfile, "\n");
   }

   // materials for non-lighted modes
   for (n=BEGREGCLRS; n < (BEGREGCLRS + REGCLRS + MAPCLRS); n++) {
     fprintf(outfile, "createNode lambert -n \"vmd_nomat_cindex_%d\";\n", n);
     fprintf(outfile, "  setAttr \".c\" -type \"float3\" %f %f %f;\n", 
             matData[n][0], matData[n][1], matData[n][2]);
     fprintf(outfile, "createNode shadingEngine -n \"vmd_nomat_cindex_%d_SG\";\n", n);
     fprintf(outfile, "  setAttr \".ihi\" 0;\n");
     fprintf(outfile, "  setAttr \".ro\" yes;\n");
     fprintf(outfile, "createNode materialInfo -n \"vmd_nomat_info_cindex_%d\";\n", n);
     fprintf(outfile, "connectAttr \"vmd_nomat_cindex_%d.oc\" \"vmd_nomat_cindex_%d_SG.ss\";\n", n, n);
     fprintf(outfile, "connectAttr \"vmd_nomat_cindex_%d_SG.msg\" \"vmd_nomat_info_cindex_%d.sg\";\n", n, n);
     fprintf(outfile, "connectAttr \"vmd_nomat_cindex_%d.msg\" \"vmd_nomat_info_cindex_%d.m\";\n", n, n);
     fprintf(outfile, "\n");
   }
}

void MayaDisplayDevice::write_cindexmaterial(const char *mayaobjstr, int cindex, int material) {
//  fprintf(outfile, "connectAttr \"%s.iog\" \":initialShadingGroup.dsm\" -na;\n", mayaobjstr, cindex);
  if (materials_on) {
    fprintf(outfile, "connectAttr \"%s.iog\" \"vmd_mat_cindex_%d_SG.dsm\" -na;\n", mayaobjstr, cindex);
  } else {
    fprintf(outfile, "connectAttr \"%s.iog\" \"vmd_nomat_cindex_%d_SG.dsm\" -na;\n", mayaobjstr, cindex);
  }
  oldMaterialIndex = material;
  oldColorIndex = cindex;
  oldMaterialState = materials_on;
}

void MayaDisplayDevice::write_colormaterial(float *rgb, int material) {
//  int cindex = nearest_index(rgb[0], rgb[1], rgb[2]);
//  write_cindexmaterial(cindex, material);
}

void MayaDisplayDevice::write_trailer (void) {
  msgWarn << "Materials are not exported to Maya files.\n";
}

