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
 *	$RCSfile: DispCmds.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.113 $	$Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * DispCmds - different display commands which take data and put it in
 *	a storage space provided by a given VMDDisplayList object.
 *
 * Notes:
 *	1. All coordinates are stored as 3 points (x,y,z), even if meant
 * for a 2D object.  The 3rd coord for 2D objects will be ignored.
 ***************************************************************************/

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifdef VMDACTC
extern "C" {
// XXX 
// The regular ACTC distribution compiles as plain C, need to send 
// a header file fix to Brad Grantham so C++ codes don't need this.
#include <tc.h>
}
#endif

#include "Scene.h"
#include "DispCmds.h"
#include "utilities.h"
#include "Matrix4.h"
#include "VMDDisplayList.h"
#include "Inform.h"
#include "VMDApp.h" // needed for texture serial numbers

//*************************************************************
// Mark the beginning of the geometry associated with a representation
void DispCmdBeginRepGeomGroup::putdata(const char *newtxt, VMDDisplayList *dobj) {
  char *buf = (char *) dobj->append(DBEGINREPGEOMGROUP, strlen(newtxt)+1);
  if (buf == NULL)
    return;
  memcpy(buf, newtxt, strlen(newtxt)+1);
}


//*************************************************************
// include comments in the display list, useful for Token Rendering
void DispCmdComment::putdata(const char *newtxt, VMDDisplayList *dobj) {
  char *buf = (char *) dobj->append(DCOMMENT, strlen(newtxt)+1);
  if (buf == NULL)
    return;
  memcpy(buf, newtxt, strlen(newtxt)+1);
}


//*************************************************************
// plot a point at the given position
void DispCmdPoint::putdata(const float *newpos, VMDDisplayList *dobj) {
  DispCmdPoint *ptr = (DispCmdPoint *)(dobj->append(DPOINT, 
                                       sizeof(DispCmdPoint)));
  if (ptr == NULL)
    return;
  ptr->pos[0]=newpos[0];
  ptr->pos[1]=newpos[1];
  ptr->pos[2]=newpos[2];
}

//*************************************************************
// plot a sphere of specified radius at the given position
void DispCmdSphere::putdata(float *newpos, float radius, VMDDisplayList *dobj) {
  DispCmdSphere *ptr = (DispCmdSphere *)(dobj->append(DSPHERE, 
                                         sizeof(DispCmdSphere)));
  if (ptr == NULL)
    return;
  ptr->pos_r[0]=newpos[0];
  ptr->pos_r[1]=newpos[1];
  ptr->pos_r[2]=newpos[2];
  ptr->pos_r[3]=radius; 
} 


void DispCmdSphereArray::putdata(const float * spcenters,
                                const float * spradii,
                                const float * spcolors,
                                int num_spheres,
                                int sphere_res,
                                VMDDisplayList * dobj) {

  DispCmdSphereArray *ptr = (DispCmdSphereArray *) dobj->append(DSPHEREARRAY, 
                           sizeof(DispCmdSphereArray) +
                           sizeof(float) * num_spheres * 3L +
                           sizeof(float) * num_spheres + 
                           sizeof(float) * num_spheres * 3L +
                           sizeof(int) * 2L);
  if (ptr == NULL)
    return;
  ptr->numspheres = num_spheres;
  ptr->sphereres = sphere_res;

  float *centers;
  float *radii;
  float *colors;
  ptr->getpointers(centers, radii, colors);

  memcpy(centers, spcenters, sizeof(float) * num_spheres * 3L);
  memcpy(radii, spradii, sizeof(float) * num_spheres);
  memcpy(colors, spcolors, sizeof(float) * num_spheres * 3L);
}

//*************************************************************
// plot a lattice cube with side length equal to 2x radius at the given position
void DispCmdLatticeCubeArray::putdata(const float * cbcenters,
                                      const float * cbradii,
                                      const float * cbcolors,
                                      int num_cubes,
                                      VMDDisplayList * dobj) {

  DispCmdLatticeCubeArray *ptr = (DispCmdLatticeCubeArray *) dobj->append(DCUBEARRAY, 
                           sizeof(DispCmdLatticeCubeArray) +
                           sizeof(float) * num_cubes * 3L +
                           sizeof(float) * num_cubes + 
                           sizeof(float) * num_cubes * 3L +
                           sizeof(int) * 1L);
  if (ptr == NULL)
    return;
  ptr->numcubes = num_cubes;

  float *centers;
  float *radii;
  float *colors;
  ptr->getpointers(centers, radii, colors);

  memcpy(centers, cbcenters, sizeof(float) * num_cubes * 3L);
  memcpy(radii, cbradii, sizeof(float) * num_cubes);
  memcpy(colors, cbcolors, sizeof(float) * num_cubes * 3L);
}


//*************************************************************

void DispCmdPointArray::putdata(const float * pcenters,
                                const float * pcolors,
                                float psize,
                                int num_points,
                                VMDDisplayList * dobj) {

  DispCmdPointArray *ptr = (DispCmdPointArray *) dobj->append(DPOINTARRAY, 
                           sizeof(DispCmdPointArray) +
                           sizeof(float) * num_points * 3L +
                           sizeof(float) * num_points * 3L +
                           sizeof(float) +
                           sizeof(int));
  if (ptr == NULL)
    return;
  ptr->size = psize;
  ptr->numpoints = num_points;

  float *centers;
  float *colors;
  ptr->getpointers(centers, colors);

  memcpy(centers, pcenters, sizeof(float) * num_points * 3L);
  memcpy(colors, pcolors, sizeof(float) * num_points * 3L);
}

void DispCmdPointArray::putdata(const float * pcenters,
                                const int * pcolors,
                                Scene * scene,
                                float psize,
                                int num_atoms,
                                const int *on,
                                int num_selected,
                                VMDDisplayList * dobj) {

  // If we have a reasonable size atom selection and therefore 
  // vertex buffer size, we use a very fast/simple path for populating the 
  // display command buffer.
  // If we have too many vertices, we have to break up the vertex buffers
  // and emit several smaller buffers to prevent integer wraparound in 
  // vertex indexing in back-end renderers.  
  int totalpoints=0;
  int i=0;
  while (totalpoints < num_selected) {
    int chunksize = num_selected - totalpoints;
    if (chunksize > VMDMAXVERTEXBUFSZ)
      chunksize = VMDMAXVERTEXBUFSZ;

    DispCmdPointArray *ptr = (DispCmdPointArray *) dobj->append(DPOINTARRAY, 
                             sizeof(DispCmdPointArray) +
                             sizeof(float) * chunksize * 3L +
                             sizeof(float) * chunksize * 3L +
                             sizeof(float) +
                             sizeof(int));
    if (ptr == NULL)
      return;
    ptr->size = psize;
    ptr->numpoints = chunksize;

    float *centers, *colors;
    ptr->getpointers(centers, colors);

    const float *fp = pcenters + 3L*i;
    long ind;
    int cnt;
    for (ind=0,cnt=0; ((cnt < VMDMAXVERTEXBUFSZ) && (i < num_atoms)); i++) {
      // draw a point for each selected atom
      if (on[i]) {
        cnt++;
        centers[ind    ] = fp[0];
        centers[ind + 1] = fp[1];
        centers[ind + 2] = fp[2];

        const float *cp = scene->color_value(pcolors[i]);
        colors[ind    ] = cp[0];
        colors[ind + 1] = cp[1];
        colors[ind + 2] = cp[2];
        ind += 3L;
      }
      fp += 3L;
    }
    totalpoints+=cnt;
  }
}



//*************************************************************

void DispCmdLitPointArray::putdata(const float * pcenters,
                                   const float * pnormals,
                                   const float * pcolors,
                                   float psize,
                                   int num_points,
                                   VMDDisplayList * dobj) {

  DispCmdLitPointArray *ptr = (DispCmdLitPointArray *) dobj->append(DLITPOINTARRAY, 
                           sizeof(DispCmdLitPointArray) +
                           sizeof(float) * num_points * 3L +
                           sizeof(float) * num_points * 3L +
                           sizeof(float) * num_points * 3L +
                           sizeof(float) +
                           sizeof(int));
  if (ptr == NULL)
    return;
  ptr->size = psize;
  ptr->numpoints = num_points;

  float *centers;
  float *normals;
  float *colors;
  ptr->getpointers(centers, normals, colors);

  memcpy(centers, pcenters, sizeof(float) * num_points * 3L);
  memcpy(normals, pnormals, sizeof(float) * num_points * 3L);
  memcpy(colors, pcolors, sizeof(float) * num_points * 3L);
}

//*************************************************************

// plot a line at the given position
void DispCmdLine::putdata(float *newpos1, float *newpos2, VMDDisplayList *dobj) {
  DispCmdLine *ptr = (DispCmdLine *)(dobj->append(DLINE, 
                                         sizeof(DispCmdLine)));
  if (ptr == NULL)
    return;
  memcpy(ptr->pos1, newpos1, 3L*sizeof(float));
  memcpy(ptr->pos2, newpos2, 3L*sizeof(float));
}

// draw a series of independent lines, (v0 v1), (v2 v3), (v4 v5)
void DispCmdLineArray::putdata(float *v, int n, VMDDisplayList *dobj) {
  void *ptr = dobj->append(DLINEARRAY, (1+6L*n)*sizeof(float));
  if (ptr == NULL)
    return;
  float *fltptr = (float *)ptr;
  *fltptr = (float)n;
  memcpy(fltptr+1, v, 6L*n*sizeof(float));
}

// draw a series of connected polylines, (v0 v1 v2 v3 v4 v5)
void DispCmdPolyLineArray::putdata(float *v, int n, VMDDisplayList *dobj) {
  void *ptr = dobj->append(DPOLYLINEARRAY, (1+3L*n)*sizeof(float));
  if (ptr == NULL)
    return;
  float *fltptr = (float *)ptr;
  *fltptr = (float)n;
  memcpy(fltptr+1, v, 3L*n*sizeof(float));
}

//*************************************************************
// draw a triangle

// set up the data for the DTRIANGLE drawing command
void DispCmdTriangle::set_array(const float *p1,const float *p2,const float *p3,
  const float *n1, const float *n2, const float *n3, VMDDisplayList *dobj) {
  DispCmdTriangle *ptr = (DispCmdTriangle *)(dobj->append(DTRIANGLE, 
                                         sizeof(DispCmdTriangle)));
  if (ptr == NULL)
    return;
  memcpy(ptr->pos1, p1, 3L*sizeof(float)); 
  memcpy(ptr->pos2, p2, 3L*sizeof(float)); 
  memcpy(ptr->pos3, p3, 3L*sizeof(float)); 
  memcpy(ptr->norm1, n1, 3L*sizeof(float)); 
  memcpy(ptr->norm2, n2, 3L*sizeof(float)); 
  memcpy(ptr->norm3, n3, 3L*sizeof(float)); 
}

// put in new data, and put the command
void DispCmdTriangle::putdata(const float *p1, const float *p2, 
                              const float *p3, VMDDisplayList *dobj) {
  int i;
  float tmp1[3], tmp2[3], tmp3[3];  // precompute the normal for
  for (i=0; i<3; i++) {             //   faster drawings later
     tmp1[i] = p2[i] - p1[i];
     tmp2[i] = p3[i] - p2[i];
  }
  cross_prod( tmp3, tmp1, tmp2);  
  vec_normalize(tmp3);
  set_array(p1, p2, p3, tmp3, tmp3, tmp3, dobj);
}
void DispCmdTriangle::putdata(const float *p1, const float *p2,const float *p3,
			      const float *n1, const float *n2,const float *n3,
                              VMDDisplayList *dobj) {
  set_array(p1,p2,p3,n1,n2,n3,dobj);
}

//*************************************************************

// draw a square, given 3 of four points
void DispCmdSquare::putdata(float *p1, float *p2,float *p3,VMDDisplayList *dobj) {
  DispCmdSquare *ptr = (DispCmdSquare *)(dobj->append(DSQUARE, 
                                         sizeof(DispCmdSquare)));
  if (ptr == NULL)
    return;
  int i;
  float tmp1[3], tmp2[3];           // precompute the normal for
  for (i=0; i<3; i++) {             //   faster drawings later
    tmp1[i] = p2[i] - p1[i];
    tmp2[i] = p3[i] - p2[i];
  }
  cross_prod(ptr->norml, tmp1, tmp2);  
  vec_normalize(ptr->norml);

  memcpy(ptr->pos1, p1, 3L*sizeof(float));
  memcpy(ptr->pos2, p2, 3L*sizeof(float));
  memcpy(ptr->pos3, p3, 3L*sizeof(float));
  for (i=0; i<3; i++)
    ptr->pos4[i] = p1[i] + tmp2[i];  // compute the fourth point
}


//*************************************************************
// draw a mesh consisting of vertices, facets, colors, normals etc.
void DispCmdTriMesh::putdata(const float * vertices,
                             const float * normals,
                             const float * colors,
                             int num_facets,
                             VMDDisplayList * dobj) {
  // make a triangle mesh (no strips)
  DispCmdTriMesh *ptr;
  if (colors == NULL) {
    ptr = (DispCmdTriMesh *) 
                (dobj->append(DTRIMESH_C3F_N3F_V3F, sizeof(DispCmdTriMesh) +
                              sizeof(float) * num_facets * 3L * 6L));
  } else {
    ptr = (DispCmdTriMesh *) 
                (dobj->append(DTRIMESH_C3F_N3F_V3F, sizeof(DispCmdTriMesh) +
                              sizeof(float) * num_facets * 3L * 9L));
  }

  if (ptr == NULL)
    return;

  ptr->numverts=num_facets * 3L;
  ptr->numfacets=num_facets;

  float *c=NULL, *n=NULL, *v=NULL;
  if (colors == NULL) {
    ptr->pervertexcolors=0;
    ptr->getpointers(n, v);
  } else {
    ptr->pervertexcolors=1;
    ptr->getpointers(c, n, v);
    memcpy(c, colors,   ptr->numverts * 3L * sizeof(float));
  }

  if (normals == NULL) {
    ptr->pervertexnormals=0;
    long i;
    for (i=0; i<(num_facets * 9L); i+=9) {
      float tmp1[3], tmp2[3], tmpnorm[3];
      const float *v0 = &vertices[i  ];
      const float *v1 = &vertices[i+3];
      const float *v2 = &vertices[i+6];

      vec_sub(tmp1, v1, v0);
      vec_sub(tmp2, v2, v1);
      cross_prod(tmpnorm, tmp1, tmp2);
      vec_normalize(tmpnorm);

      n[i  ] = tmpnorm[0];
      n[i+1] = tmpnorm[1];
      n[i+2] = tmpnorm[2];

      n[i+3] = tmpnorm[0];
      n[i+4] = tmpnorm[1];
      n[i+5] = tmpnorm[2];

      n[i+6] = tmpnorm[0];
      n[i+7] = tmpnorm[1];
      n[i+8] = tmpnorm[2];
    }  
  } else {
    ptr->pervertexnormals=1;
    memcpy(n, normals,  ptr->numverts * 3L * sizeof(float));
  }

  memcpy(v, vertices, ptr->numverts * 3L * sizeof(float));
}


//*************************************************************
// draw a mesh consisting of vertices, facets, colors, normals etc.
void DispCmdTriMesh::putdata(const float * vertices,
                             const float * normals,
                             const unsigned char * colors,
                             int num_facets,
                             VMDDisplayList * dobj) {
  // make a triangle mesh (no strips)
  DispCmdTriMesh *ptr;
  if (colors == NULL) {
    ptr = (DispCmdTriMesh *) 
                (dobj->append(DTRIMESH_C4U_N3F_V3F, sizeof(DispCmdTriMesh) +
                              sizeof(float) * num_facets * 3L * 6L));
  } else {
    ptr = (DispCmdTriMesh *) 
                (dobj->append(DTRIMESH_C4U_N3F_V3F, sizeof(DispCmdTriMesh) +
                              4L * sizeof(unsigned char) * num_facets * 3L +
                              sizeof(float) * num_facets * 3L * 6L));
  }

  if (ptr == NULL)
    return;

  ptr->numverts=num_facets * 3L;
  ptr->numfacets=num_facets;

  unsigned char *c=NULL;
  float *n=NULL, *v=NULL;
  if (colors == NULL) {
    ptr->pervertexcolors=0;
    ptr->getpointers(n, v);
  } else {
    ptr->pervertexcolors=1;
    ptr->getpointers(c, n, v);
    memcpy(c, colors,   ptr->numverts * 4L * sizeof(unsigned char));
  }

  ptr->pervertexnormals=1;
  memcpy(n, normals,  ptr->numverts * 3L * sizeof(float));
  memcpy(v, vertices, ptr->numverts * 3L * sizeof(float));
}


//*************************************************************
// draw a mesh consisting of vertices, facets, colors, normals etc.
void DispCmdTriMesh::putdata(const float * vertices,
                             const char * normals,
                             const unsigned char * colors,
                             int num_facets,
                             VMDDisplayList * dobj) {
  // make a triangle mesh (no strips)
  DispCmdTriMesh *ptr;
  if (colors == NULL) {
    ptr = (DispCmdTriMesh *) 
                (dobj->append(DTRIMESH_C4U_N3B_V3F, sizeof(DispCmdTriMesh) +
                              sizeof(char) * num_facets * 3L * 3L +
                              sizeof(float) * num_facets * 3L * 3L));
  } else {
    ptr = (DispCmdTriMesh *) 
                (dobj->append(DTRIMESH_C4U_N3B_V3F, sizeof(DispCmdTriMesh) +
                              4L * sizeof(unsigned char) * num_facets * 3L +
                              sizeof(char) * num_facets * 3L * 3L +
                              sizeof(float) * num_facets * 3L * 3L));
  }

  if (ptr == NULL)
    return;

  ptr->numverts=num_facets * 3L;
  ptr->numfacets=num_facets;

  unsigned char *c=NULL;
  char *n=NULL;
  float *v=NULL;
  if (colors == NULL) {
    ptr->pervertexcolors=0;
    ptr->getpointers(n, v);
  } else {
    ptr->pervertexcolors=1;
    ptr->getpointers(c, n, v);
    memcpy(c, colors,   ptr->numverts * 4L * sizeof(unsigned char));
  }

  ptr->pervertexnormals=1;
  memcpy(n, normals,  ptr->numverts * 3L * sizeof(char));
  memcpy(v, vertices, ptr->numverts * 3L * sizeof(float));
}


// draw a mesh consisting of vertices, facets, colors, normals etc.
void DispCmdTriMesh::putdata(const float * vertices,
                             const float * normals,
                             const float * colors,
                             int num_verts,
                             const int * facets,
                             int num_facets, 
                             int enablestrips,
                             VMDDisplayList * dobj) {
  int builtstrips = 0; 

#if defined(VMDACTC) 
  if (enablestrips)  {
    // Rearrange face data into triangle strips
    ACTCData *tc = actcNew();  // intialize ACTC stripification library
    long fsize = num_facets * 3L;
    long i, ind, ii;
    long iPrimCount = 0;
    long iCurrPrimSize;

    // XXX over-allocate the vertex and facet buffers to prevent an
    //     apparent bug in ACTC 1.1 from crashing VMD.  This was causing
    //     Surf surfaces to crash ACTC at times.
    int *p_iPrimSize = new int[fsize + 6];  // num vertices in a primitive 
    unsigned int *f2 = new uint[fsize + 6];
    
    if (tc == NULL) {
      msgErr << "ACTC initialization failed, using triangle mesh." << sendmsg;
    } else {
      msgInfo << "Performing ACTC Triangle Consolidation..." << sendmsg;
 
      // only produce strips, not fans, give a ridiculously high min value.
      actcParami(tc, ACTC_OUT_MIN_FAN_VERTS, 2147483647);

      // disabling honoring vertex winding order might allow ACTC to
      // consolidate more triangles into strips, but this is only useful
      // if VMD has two-sided lighting enabled.
      // actcParami(tc, ACTC_OUT_HONOR_WINDING, ACTC_TRUE);
        
      // send triangle data over to ACTC library
      actcBeginInput(tc);
      for (ii=0; ii < num_facets; ii++) {
        ind = ii * 3L;
        if ((actcAddTriangle(tc, facets[ind], facets[ind + 1], facets[ind + 2])) != ACTC_NO_ERROR) {
          msgInfo << "ACTC Add Triangle Error." << sendmsg;
        }
      }
      actcEndInput(tc);
        
      // get triangle strips back from ACTC, loop through once to get sizes
      actcBeginOutput(tc);
      i = 0;
      while ((actcStartNextPrim(tc, &f2[i], &f2[i+1]) != ACTC_DATABASE_EMPTY)) {
        iCurrPrimSize = 2;  // if we're here, we got 2 vertices
        i+=2;               // increment array position
        while (actcGetNextVert(tc, &f2[i]) != ACTC_PRIM_COMPLETE) {
          iCurrPrimSize++;  // increment number of vertices for this primitive
          i++;              // increment array position
        }

        p_iPrimSize[iPrimCount] = iCurrPrimSize;  // save vertex count
        iPrimCount++;       // increment primitive counter
      }
      actcEndOutput(tc);
      msgInfo << "ACTC: Created " << iPrimCount << " triangle strips" << sendmsg;
      msgInfo << "ACTC: Average vertices per strip = " << i / iPrimCount  << sendmsg;

      // Draw triangle strips, uses double-sided lighting until we change
      // things to allow the callers to specify the desired lighting 
      // explicitly.
      DispCmdTriStrips::putdata(vertices, normals, colors, num_verts, p_iPrimSize, iPrimCount, f2, i, 1, dobj);
          
      // delete temporary memory
      delete [] f2;
      delete [] p_iPrimSize;

      // delete ACTC handle
      actcDelete(tc);

      builtstrips = 1; // don't generate a regular triangle mesh
    }  
  } 
#endif

  if (!builtstrips) {
    // make a triangle mesh (no strips)
    DispCmdTriMesh *ptr = (DispCmdTriMesh *) 
                  (dobj->append(DTRIMESH_C4F_N3F_V3F, sizeof(DispCmdTriMesh) +
                                          sizeof(float) * num_verts * 10L +
                                          sizeof(int) * num_facets * 3L));
    if (ptr == NULL)
      return;
    ptr->pervertexcolors=1;
    ptr->pervertexnormals=1;
    ptr->numverts=num_verts;
    ptr->numfacets=num_facets;
    float *cnv;
    int *f;
    ptr->getpointers(cnv, f);

#if 1
    long ind10, ind3;
    for (ind10=0,ind3=0; ind10<num_verts*10L; ind10+=10,ind3+=3) {
      cnv[ind10    ] =   colors[ind3    ];
      cnv[ind10 + 1] =   colors[ind3 + 1]; 
      cnv[ind10 + 2] =   colors[ind3 + 2]; 
      cnv[ind10 + 3] =   1.0; 
      cnv[ind10 + 4] =  normals[ind3    ];
      cnv[ind10 + 5] =  normals[ind3 + 1];
      cnv[ind10 + 6] =  normals[ind3 + 2];
      cnv[ind10 + 7] = vertices[ind3    ];
      cnv[ind10 + 8] = vertices[ind3 + 1];
      cnv[ind10 + 9] = vertices[ind3 + 2];
    }
#else
    long i, ind, ind2;
    for (i=0; i<num_verts; i++) {
      ind = i * 10L;
      ind2 = i * 3L;
      cnv[ind    ] =   colors[ind2    ];
      cnv[ind + 1] =   colors[ind2 + 1]; 
      cnv[ind + 2] =   colors[ind2 + 2]; 
      cnv[ind + 3] =   1.0; 
      cnv[ind + 4] =  normals[ind2    ];
      cnv[ind + 5] =  normals[ind2 + 1];
      cnv[ind + 6] =  normals[ind2 + 2];
      cnv[ind + 7] = vertices[ind2    ];
      cnv[ind + 8] = vertices[ind2 + 1];
      cnv[ind + 9] = vertices[ind2 + 2];
    } 
#endif

    memcpy(f, facets, ptr->numfacets * 3L * sizeof(int));
  }
}

//*************************************************************

// draw a set of triangle strips
void DispCmdTriStrips::putdata(const float * vertices,
                               const float * normals,
                               const float * colors,
                               int num_verts,
                               const int * verts_per_strip,
                               int num_strips,
                               const unsigned int * strip_data,
                               const int num_strip_verts,
                               int double_sided_lighting,
                               VMDDisplayList * dobj) {

  DispCmdTriStrips *ptr = (DispCmdTriStrips *) (dobj->append(DTRISTRIP, 
                                         sizeof(DispCmdTriStrips) +
                                         sizeof(int *) * num_strips +
                                         sizeof(float) * num_verts * 10L +
                                         sizeof(int) * num_strip_verts +
                                         sizeof(int) * num_strips));
  if (ptr == NULL) 
    return;
  ptr->numverts=num_verts;
  ptr->numstrips=num_strips;
  ptr->numstripverts=num_strip_verts;
  ptr->doublesided=double_sided_lighting;

  float *cnv;
  int *f;
  int *vertsperstrip;
  ptr->getpointers(cnv, f, vertsperstrip);

  // copy vertex,color,normal data
  long i, ind, ind2;
  for (i=0; i<num_verts; i++) {
    ind = i * 10L;
    ind2 = i * 3L;
    cnv[ind    ] =   colors[ind2    ];
    cnv[ind + 1] =   colors[ind2 + 1];
    cnv[ind + 2] =   colors[ind2 + 2];
    cnv[ind + 3] =   1.0;
    cnv[ind + 4] =  normals[ind2    ];
    cnv[ind + 5] =  normals[ind2 + 1];
    cnv[ind + 6] =  normals[ind2 + 2];
    cnv[ind + 7] = vertices[ind2    ];
    cnv[ind + 8] = vertices[ind2 + 1];
    cnv[ind + 9] = vertices[ind2 + 2];
  }

  // copy vertices per strip data
  for (i=0; i<num_strips; i++) {
    vertsperstrip[i] = verts_per_strip[i];
  }

  // copy face (triangle) data
  for (i=0; i<num_strip_verts; i++) {
    f[i] = strip_data[i];
  }
}


//*************************************************************

void DispCmdWireMesh::putdata(const float * vertices,
                             const float * normals,
                             const float * colors,
                             int num_verts,
                             const int * lines,
                             int num_lines, VMDDisplayList * dobj) {
 
  DispCmdWireMesh *ptr = (DispCmdWireMesh *) (dobj->append(DWIREMESH, 
                                         sizeof(DispCmdWireMesh) +
                                         sizeof(float) * num_verts * 10L +
                                         sizeof(int) * num_lines * 3L));
  if (ptr == NULL) 
    return;
  ptr->numverts=num_verts;
  ptr->numlines=num_lines;

  float *cnv;
  int *l;
  ptr->getpointers(cnv, l);

  long i, ind, ind2;
  for (i=0; i<num_verts; i++) {
    ind = i * 10L;
    ind2 = i * 3L;
    cnv[ind    ] =   colors[ind2    ];
    cnv[ind + 1] =   colors[ind2 + 1]; 
    cnv[ind + 2] =   colors[ind2 + 2]; 
    cnv[ind + 3] =   1.0; 
    cnv[ind + 4] =  normals[ind2    ];
    cnv[ind + 5] =  normals[ind2 + 1];
    cnv[ind + 6] =  normals[ind2 + 2];
    cnv[ind + 7] = vertices[ind2    ];
    cnv[ind + 8] = vertices[ind2 + 1];
    cnv[ind + 9] = vertices[ind2 + 2];
  } 

  memcpy(l, lines, ptr->numlines * 2L * sizeof(int));
}

//*************************************************************
// plot a cylinder at the given position
// this is used to precalculate the cylinder data for speedup
// in renderers without hardware cylinders.  For example, the GL
// library.  There are res number of edges (with a norm, and two points)

DispCmdCylinder::DispCmdCylinder(void) {
  lastres = 0;
}

void DispCmdCylinder::putdata(const float *pos1, const float *pos2, float rad, 
                      int res, int filled, VMDDisplayList *dobj) {

  float lenaxis[3];
  vec_sub(lenaxis, pos1, pos2);  // check that it's valid
  if (dot_prod(lenaxis,lenaxis) == 0.0 || res <= 0) return;

  if (lastres != res ) {
    rot[0] = cosf( (float) VMD_TWOPI / (float) res);
    rot[1] = sinf( (float) VMD_TWOPI / (float) res);
  }
  lastres = res;
  size_t size = (9L + res*3L*3L)*sizeof(float);

  float *pos = (float *)(dobj->append(DCYLINDER, size));
  if (pos == NULL) 
    return;

  memcpy(pos,   pos1, 3L*sizeof(float));
  memcpy(pos+3, pos2, 3L*sizeof(float));
  pos[6] = rad;
  pos[7] = (float)res;
  pos[8] = (float)filled;

  float axis[3];
  vec_sub(axis, pos1, pos2);
  vec_normalize(axis);
  int i;  // find an axis not aligned with the cylinder
  if (fabs(axis[0]) < fabs(axis[1]) &&
      fabs(axis[0]) < fabs(axis[2])) {
     i = 0;
  } else if (fabs(axis[1]) < fabs(axis[2])) {
     i = 1;
  } else {
     i = 2;
  }
  float perp[3];
  perp[i] = 0;                    // this is not aligned with the cylinder
  perp[(i+1)%3] = axis[(i+2)%3];
  perp[(i+2)%3] = -axis[(i+1)%3];
  vec_normalize(perp);
  float perp2[3];
  cross_prod(perp2, axis, perp); // find a normal to the cylinder

  float *posptr = pos+9;
  float m = rot[0], n = rot[1];
  for (int h=0; h<res; h++) {
    float tmp0, tmp1, tmp2;
    
    tmp0 = m*perp[0] + n*perp2[0]; // add the normal
    tmp1 = m*perp[1] + n*perp2[1];
    tmp2 = m*perp[2] + n*perp2[2];

    posptr[0] = tmp0; // add the normal
    posptr[1] = tmp1;
    posptr[2] = tmp2;

    posptr[3] = pos2[0] + rad * tmp0; // start
    posptr[4] = pos2[1] + rad * tmp1;
    posptr[5] = pos2[2] + rad * tmp2;

    posptr[6] = posptr[3] + lenaxis[0];  // and end of the edge
    posptr[7] = posptr[4] + lenaxis[1];
    posptr[8] = posptr[5] + lenaxis[2];
    posptr += 9;
    // use angle addition formulae:
    // cos(A+B) = cos A cos B - sin A sin B
    // sin(A+B) = cos A sin B + sin A cos B
    float mtmp = rot[0]*m - rot[1]*n;
    float ntmp = rot[0]*n + rot[1]*m; 
    m = mtmp;
    n = ntmp;
  }
}
 
//*************************************************************

void DispCmdCone::putdata(float *p1,float *p2,float newrad,float newrad2,int newres,
                          VMDDisplayList *dobj) {
  DispCmdCone *ptr = (DispCmdCone *)(dobj->append(DCONE, 
                                         sizeof(DispCmdCone)));
  if (ptr == NULL) 
    return;
  memcpy(ptr->pos1, p1, 3L*sizeof(float));
  memcpy(ptr->pos2, p2, 3L*sizeof(float));
  ptr->radius=newrad;
  ptr->radius2=newrad2;
  ptr->res=newres;
}

// put in new data, and put the command
void DispCmdColorIndex::putdata(int newcol, VMDDisplayList *dobj) {
  DispCmdColorIndex *ptr = (DispCmdColorIndex *)(dobj->append(DCOLORINDEX, 
                                         sizeof(DispCmdColorIndex)));
  if (ptr == NULL) 
    return;
  ptr->color = newcol;
}

//*************************************************************

// display text at the given text coordinates
void DispCmdText::putdata(const float *c, const char *s, 
                          float thickness, VMDDisplayList *dobj) {
  if (s != NULL) {
    size_t len = strlen(s)+1;
    char *buf = (char *)(dobj->append(DTEXT, len+4L*sizeof(float)));
    if (buf == NULL) 
      return;
    ((float *)buf)[0] = c[0];          // X
    ((float *)buf)[1] = c[1];          // Y
    ((float *)buf)[2] = c[2];          // Z
    ((float *)buf)[3] = thickness;     // thickness
    memcpy(buf+4L*sizeof(float),s,len); // text string
  }
}

void DispCmdTextOffset::putdata(float ox, float oy, VMDDisplayList *dobj) {
  DispCmdTextOffset *cmd = (DispCmdTextOffset *)(dobj->append(DTEXTOFFSET,
        sizeof(DispCmdTextOffset)));
  cmd->x = ox;
  cmd->y = oy;
}

//*************************************************************

void DispCmdTextSize::putdata(float size1, VMDDisplayList *dobj) {
  DispCmdTextSize *ptr  = (DispCmdTextSize *)dobj->append(DTEXTSIZE,
                           sizeof(DispCmdTextSize));
  if (ptr == NULL)
    return;
  ptr->size = size1;
}

//*************************************************************

void DispCmdVolSlice::putdata(int mode, const float *pnormal, const float *verts, 
    const float *texs, VMDDisplayList *dobj) {

  DispCmdVolSlice *cmd = (DispCmdVolSlice *) dobj->append(DVOLSLICE, 
                                       sizeof(DispCmdVolSlice));
  if (cmd == NULL)
    return;

  cmd->texmode = mode;
  memcpy(cmd->normal, pnormal, 3L*sizeof(float));
  memcpy(cmd->v, verts, 12L*sizeof(float));
  memcpy(cmd->t, texs,  12L*sizeof(float));
}

//*************************************************************


void DispCmdVolumeTexture::putdata(unsigned long texID, 
    const int size[3], unsigned char *texptr, const float pv0[3], 
    const float pv1[3], const float pv2[3], const float pv3[3], 
    VMDDisplayList *dobj) {

  DispCmdVolumeTexture *cmd = (DispCmdVolumeTexture *) dobj->append(DVOLUMETEXTURE,
      sizeof(DispCmdVolumeTexture));

  if (cmd == NULL) return;

  cmd->ID = texID;
  cmd->xsize = size[0];
  cmd->ysize = size[1];
  cmd->zsize = size[2];
  cmd->texmap = texptr;
  memcpy(cmd->v0, pv0, 3L*sizeof(float));
  memcpy(cmd->v1, pv1, 3L*sizeof(float));
  memcpy(cmd->v2, pv2, 3L*sizeof(float));
  memcpy(cmd->v3, pv3, 3L*sizeof(float));
}

//*************************************************************
// put in new data, and put the command
void DispCmdSphereRes::putdata(int newres, VMDDisplayList *dobj) {
  DispCmdSphereRes *ptr  = (DispCmdSphereRes *)dobj->append(DSPHERERES,
                           sizeof(DispCmdSphereRes));
  if (ptr == NULL)
    return;
  ptr->res = newres;
}

//*************************************************************

// put in new data, and put the command
void DispCmdSphereType::putdata(int newtype, VMDDisplayList *dobj) {
  DispCmdSphereType *ptr  = (DispCmdSphereType *)dobj->append(DSPHERETYPE,
                           sizeof(DispCmdSphereType));
  if (ptr == NULL)
    return;
  ptr->type = newtype;
}

//*************************************************************

// put in new data, and put the command
void DispCmdLineType::putdata(int newtype, VMDDisplayList *dobj) {
  DispCmdLineType* ptr  = (DispCmdLineType *)dobj->append(DLINESTYLE,
                           sizeof(DispCmdLineType));
  if (ptr == NULL)
    return;
  ptr->type = newtype;
}

//*************************************************************

void DispCmdLineWidth::putdata(int newwidth, VMDDisplayList *dobj) {
  DispCmdLineWidth * ptr  = (DispCmdLineWidth *)dobj->append(DLINEWIDTH,
                           sizeof(DispCmdLineWidth));
  if (ptr == NULL)
    return;
  ptr->width = newwidth;
}

//*************************************************************

void DispCmdPickPoint::putdata(float *pos, int newtag, VMDDisplayList *dobj) {
  DispCmdPickPoint *ptr = (DispCmdPickPoint *)(dobj->append(DPICKPOINT, 
                                               sizeof(DispCmdPickPoint)));
  if (ptr == NULL)
    return;
  memcpy(ptr->postag, pos, 3L*sizeof(float));
  ptr->tag=newtag;
}

//*************************************************************

// put in new data, and put the command
void DispCmdPickPointArray::putdata(int num, int numsel, int firstsel, int *on, 
                                    float *coords, VMDDisplayList *dobj) {
  if (numsel < 1)
    return;

  DispCmdPickPointArray *ptr;
  if (num == numsel) {
    // if all indices in a contiguous block are enabled (e.g. "all" selection)
    // then there's no need to actually store the pick point indices
    ptr = (DispCmdPickPointArray *) (dobj->append(DPICKPOINT_ARRAY, 
                                     sizeof(DispCmdPickPointArray) +
                                     3L * sizeof(float) * numsel));
  } else {
    // if only some of the indices are selected, then we allocate storage
    // for the list of indices to be copied in.
    ptr = (DispCmdPickPointArray *) (dobj->append(DPICKPOINT_ARRAY, 
                                     sizeof(DispCmdPickPointArray) + 
                                     3L * sizeof(float) * numsel +
                                     sizeof(int) * numsel));
  }

  if (ptr == NULL)
    return;

  ptr->numpicks = numsel;
  ptr->firstindex = firstsel;

  float *crds;
  int *tags;
  if (num == numsel) {
    // if all indices are selected note it, copy in coords, and we're done.
    ptr->allselected = 1;
    ptr->getpointers(crds, tags);
    memcpy(crds, coords, 3L * sizeof(float) * numsel);
  } else {
    // if only some indices are selected, copy in the selected ones
    ptr->allselected = 0;
    ptr->getpointers(crds, tags);

    // copy tags for selected/enabled indices
    long cnt=numsel; // early-exit as soon as we found the last selected atom
    long i,cp;
    for (cp=0,i=0; cnt > 0; i++) {
      if (on[i]) {
        cnt--;

        long idx = i*3L;
        long idx2 = cp*3L;
        crds[idx2    ] = coords[idx    ];
        crds[idx2 + 1] = coords[idx + 1];
        crds[idx2 + 2] = coords[idx + 2];

        tags[cp] = i + firstsel;

        cp++;
      }
    }
  }
}

//*************************************************************

// put in new data, and put the command
void DispCmdPickPointArray::putdata(int num, int *indices,
                                    float *coords, VMDDisplayList *dobj) {
  DispCmdPickPointArray *ptr;

  ptr = (DispCmdPickPointArray *) (dobj->append(DPICKPOINT_ARRAY, 
                                   sizeof(DispCmdPickPointArray) + 
                                   3L * sizeof(float) * num +
                                   sizeof(int) * num));

  if (ptr == NULL)
    return;

  ptr->numpicks = num;
  ptr->allselected = 0; // use the index array entries
  ptr->firstindex = indices[0];

  float *crds;
  int *tags;
  ptr->getpointers(crds, tags);
  memcpy(crds, coords, num * 3L * sizeof(float));
  memcpy(tags, indices, num * sizeof(int));
}

