/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2019 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "Inform.h"
#include "utilities.h"

#define ISOCONTOUR_INTERNAL 1
#include "Isocontour.h"
#include "VolumetricData.h"

IsoContour::IsoContour(void) {}

void IsoContour::clear(void) {
  numtriangles=0;
  v.clear();
  f.clear();
}


int IsoContour::compute(const VolumetricData *data, float isovalue, int step) {
  int x, y, z; 
  int tricount=0;

  vol=data;

  xax[0] = (float) (vol->xaxis[0] / ((float) (vol->xsize - 1)));
  xax[1] = (float) (vol->xaxis[1] / ((float) (vol->xsize - 1)));
  xax[2] = (float) (vol->xaxis[2] / ((float) (vol->xsize - 1)));

  yax[0] = (float) (vol->yaxis[0] / ((float) (vol->ysize - 1)));
  yax[1] = (float) (vol->yaxis[1] / ((float) (vol->ysize - 1)));
  yax[2] = (float) (vol->yaxis[2] / ((float) (vol->ysize - 1)));

  zax[0] = (float) (vol->zaxis[0] / ((float) (vol->zsize - 1)));
  zax[1] = (float) (vol->zaxis[1] / ((float) (vol->zsize - 1)));
  zax[2] = (float) (vol->zaxis[2] / ((float) (vol->zsize - 1)));

  for (z=0; z<(vol->zsize - step); z+=step) {
    for (y=0; y<(vol->ysize - step); y+=step) {
      for (x=0; x<(vol->xsize - step); x+=step) {
        tricount += DoCell(x, y, z, isovalue, step);
      }
    }
  }

  return 1;
}



int IsoContour::DoCell(int x, int y, int z, float isovalue, int step) {
  SQUARECELL gc;
  int addr, row, plane, rowstep, planestep, tricount;
  LINE tris[5];

  row = vol->xsize; 
  plane = vol->xsize * vol->ysize;
  addr = z*plane + y*row + x;
  rowstep = row*step;
  planestep = plane*step;
  
  gc.val[0] = vol->data[addr                             ];
  gc.val[1] = vol->data[addr + step                      ];
  gc.val[3] = vol->data[addr +        rowstep            ];
  gc.val[2] = vol->data[addr + step + rowstep            ];
  gc.val[4] = vol->data[addr +                  planestep];
  gc.val[5] = vol->data[addr + step +           planestep];
  gc.val[7] = vol->data[addr +        rowstep + planestep];
  gc.val[6] = vol->data[addr + step + rowstep + planestep];

  /*
     Determine the index into the edge table which
     tells us which vertices are inside of the surface
  */
  int cubeindex = 0;
  if (gc.val[0] < isovalue) cubeindex |= 1;
  if (gc.val[1] < isovalue) cubeindex |= 2;
  if (gc.val[2] < isovalue) cubeindex |= 4;
  if (gc.val[3] < isovalue) cubeindex |= 8;
  if (gc.val[4] < isovalue) cubeindex |= 16;
  if (gc.val[5] < isovalue) cubeindex |= 32;
  if (gc.val[6] < isovalue) cubeindex |= 64;
  if (gc.val[7] < isovalue) cubeindex |= 128;

  /* Cube is entirely in/out of the surface */
  if (edgeTable[cubeindex] == 0)
    return(0);
  gc.cubeindex = cubeindex;

  gc.p[0].x = (float) x;
  gc.p[0].y = (float) y;

  gc.p[1].x = (float) x + step;
  gc.p[1].y = (float) y;

  gc.p[2].x = (float) x + step;
  gc.p[2].y = (float) y + step;
  
  gc.p[3].x = (float) x;
  gc.p[3].y = (float) y + step;

  // calculate vertices and facets for this cube,
  // calculate normals by interpolating between the negated 
  // normalized volume gradients for the 8 reference voxels
  tricount = Polygonise(gc, isovalue, (LINE *) &tris);

  if (tricount > 0) {
    int i;

    for (i=0; i<tricount; i++) {
      float xx, yy, zz;

      int tritmp = numtriangles * 3;
      f.append3(tritmp, tritmp+1, tritmp+2);
      numtriangles++;

      // add new vertices and normals into vertex and normal lists
      xx = tris[i].p[0].x;
      yy = tris[i].p[0].y;
      v.append3((float) vol->origin[0] + xx * xax[0] + yy * yax[0] + zz * zax[0],
                (float) vol->origin[1] + xx * xax[1] + yy * yax[1] + zz * zax[1],
                (float) vol->origin[2] + xx * xax[2] + yy * yax[2] + zz * zax[2]);
      xx = tris[i].p[1].x;
      yy = tris[i].p[1].y;
      v.append3((float) vol->origin[0] + xx * xax[0] + yy * yax[0] + zz * zax[0],
                (float) vol->origin[1] + xx * xax[1] + yy * yax[1] + zz * zax[1],
                (float) vol->origin[2] + xx * xax[2] + yy * yax[2] + zz * zax[2]);
      xx = tris[i].p[2].x;
      yy = tris[i].p[2].y;
      v.append3((float) vol->origin[0] + xx * xax[0] + yy * yax[0] + zz * zax[0],
                (float) vol->origin[1] + xx * xax[1] + yy * yax[1] + zz * zax[1],
                (float) vol->origin[2] + xx * xax[2] + yy * yax[2] + zz * zax[2]);
    }
  }

  return tricount;
}


/*
   Given a grid cell and an isolevel, calculate the triangular
   facets required to represent the isocontour through the cell.
   Return the number of triangular facets, the array "triangles"
   will be loaded up with the vertices at most 5 triangular facets.
        0 will be returned if the grid cell is either totally above
   of totally below the isolevel.
   This code calculates vertex normals by interpolating the volume gradients.
*/
int IsoContour::Polygonise(const SQUARECELL grid, const float isolevel, LINE *triangles) {
   int i,ntriang;
   int cubeindex = grid.cubeindex;
   XY vertlist[12];

   /* Find the vertices where the surface intersects the cube */
   if (edgeTable[cubeindex] & 1)
      VertexInterp(isolevel, grid, 0, 1, &vertlist[0]);
   if (edgeTable[cubeindex] & 2)
      VertexInterp(isolevel, grid, 1, 2, &vertlist[1]);
   if (edgeTable[cubeindex] & 4)
      VertexInterp(isolevel, grid, 2, 3, &vertlist[2]);
   if (edgeTable[cubeindex] & 8)
      VertexInterp(isolevel, grid, 3, 0, &vertlist[3]);
   if (edgeTable[cubeindex] & 16)
      VertexInterp(isolevel, grid, 4, 5, &vertlist[4]);
   if (edgeTable[cubeindex] & 32)
      VertexInterp(isolevel, grid, 5, 6, &vertlist[5]);
   if (edgeTable[cubeindex] & 64)
      VertexInterp(isolevel, grid, 6, 7, &vertlist[6]);
   if (edgeTable[cubeindex] & 128)
      VertexInterp(isolevel, grid, 7, 4, &vertlist[7]);
   if (edgeTable[cubeindex] & 256)
      VertexInterp(isolevel, grid, 0, 4, &vertlist[8]);
   if (edgeTable[cubeindex] & 512)
      VertexInterp(isolevel, grid, 1, 5, &vertlist[9]);
   if (edgeTable[cubeindex] & 1024)
      VertexInterp(isolevel, grid, 2, 6, &vertlist[10]);
   if (edgeTable[cubeindex] & 2048)
      VertexInterp(isolevel, grid, 3, 7, &vertlist[11]);

   /* Create the triangle */
   ntriang = 0;
   for (i=0; lineTable[cubeindex][i]!=-1;i+=3) {
     triangles[ntriang].p[0] = vertlist[lineTable[cubeindex][i  ]];
     triangles[ntriang].p[1] = vertlist[lineTable[cubeindex][i+1]];
     triangles[ntriang].p[2] = vertlist[lineTable[cubeindex][i+2]];
     ntriang++;
   }

   return ntriang;
}


/*
   Linearly interpolate the position where an isocontour cuts
   an edge between two vertices, each with their own scalar value,
   interpolating vertex position and vertex normal based on the 
   isovalue.
*/
void IsoContour::VertexInterp(float isolevel, const SQUARECELL grid, int ind1, int ind2, XY * vert) {
  float mu;

  XY p1 = grid.p[ind1];
  XY p2 = grid.p[ind2];
  float valp1 = grid.val[ind1];
  float valp2 = grid.val[ind2];

  if (fabs(isolevel-valp1) < 0.00001) {
    *vert = grid.p[ind1];
    return;
  }

  if (fabs(isolevel-valp2) < 0.00001) {
    *vert = grid.p[ind2];
    return;
  }

  if (fabs(valp1-valp2) < 0.00001) {
    *vert = grid.p[ind1];
    return;
  }

  mu = (isolevel - valp1) / (valp2 - valp1);

  vert->x = p1.x + mu * (p2.x - p1.x);
  vert->y = p1.y + mu * (p2.y - p1.y);
}

