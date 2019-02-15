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

#define ISOSURFACE_INTERNAL 1
#include "Isosurface.h"
#include "VolumetricData.h"

#define MIN(X,Y) (((X)<(Y))? (X) : (Y))
#define MAX(X,Y) (((X)>(Y))? (X) : (Y))

IsoSurface::IsoSurface(void) {}

void IsoSurface::clear(void) {
  numtriangles=0;
  v.clear();
  n.clear();
  c.clear();
  f.clear();
}


int IsoSurface::compute(VolumetricData *data, float isovalue, long step) {
  long x, y, z; 
  long tricount=0;

  vol=data;

  // calculate cell axes
  vol->cell_axes(xax, yax, zax);
  vol->cell_dirs(xad, yad, zad);

  // flip normals if coordinate system is in the wrong handedness
  float vtmp[3];
  cross_prod(vtmp, xad, yad);
  if (dot_prod(vtmp, zad) < 0) {
    xad[0] *= -1;
    xad[1] *= -1;
    xad[2] *= -1;
    yad[0] *= -1;
    yad[1] *= -1;
    yad[2] *= -1;
    zad[0] *= -1;
    zad[1] *= -1;
    zad[2] *= -1;
  }

  int i;
  int axisposnorms = 1;

  // check that the grid is in the all-positive octant of the coordinate system
  for (i=0; i<3; i++) {
    if (xad[i] < 0 || yad[i] < 0 || zad[i] < 0)  
      axisposnorms = 0;
  }

  // check that the grid is axis-aligned
  if (xax[1] != 0.0f || xax[2] != 0.0f ||
      yax[0] != 0.0f || xax[2] != 0.0f ||
      yax[0] != 0.0f || xax[1] != 0.0f)
    axisposnorms = 0;

  if (axisposnorms) {
    tricount = DoGridPosNorms(isovalue, step);
  } else {
    // general case, any handedness, non-rectangular grids
    for (z=0; z<(vol->zsize - step); z+=step) {
      for (y=0; y<(vol->ysize - step); y+=step) {
        for (x=0; x<(vol->xsize - step); x+=step) {
          tricount += DoCellGeneral(x, y, z, isovalue, step);
        }
      }
    }
  }

  return 1;
}


//
// Special case for axis-aligned grid with positive unit vector normals
//
long IsoSurface::DoGridPosNorms(float isovalue, long step) {
  GRIDCELL gc;
  TRIANGLE tris[5];
  long tricount=0;
  long globtricount=0;
  long row = vol->xsize; 
  long plane = vol->xsize * vol->ysize;
  long rowstep = row*step;
  long planestep = plane*step;

  // get direct access to gradient data for speed, and force 
  // generation of the gradients if they don't already exist
  const float *volgradient = vol->access_volume_gradient();

  // precompute plane and row sizes to eliminate indirection
  long psz = long(vol->xsize)*long(vol->ysize);
  long rsz = long(vol->xsize);
  long x, y, z;
  for (z=0; z<(vol->zsize - step); z+=step) {
    for (y=0; y<(vol->ysize - step); y+=step) {
      for (x=0; x<(vol->xsize - step); x+=step) {
        long addr = z*plane + y*row + x;
        gc.val[0] = vol->data[addr                             ];
        gc.val[1] = vol->data[addr + step                      ];
        gc.val[3] = vol->data[addr +        rowstep            ];
        gc.val[2] = vol->data[addr + step + rowstep            ];
        gc.val[4] = vol->data[addr +                  planestep];
        gc.val[5] = vol->data[addr + step +           planestep];
        gc.val[7] = vol->data[addr +        rowstep + planestep];
        gc.val[6] = vol->data[addr + step + rowstep + planestep];

        // Determine the index into the edge table which
        // tells us which vertices are inside of the surface
        int cubeindex = 0;
        if (gc.val[0] < isovalue) cubeindex |= 1;
        if (gc.val[1] < isovalue) cubeindex |= 2;
        if (gc.val[2] < isovalue) cubeindex |= 4;
        if (gc.val[3] < isovalue) cubeindex |= 8;
        if (gc.val[4] < isovalue) cubeindex |= 16;
        if (gc.val[5] < isovalue) cubeindex |= 32;
        if (gc.val[6] < isovalue) cubeindex |= 64;
        if (gc.val[7] < isovalue) cubeindex |= 128;

        // Cube is entirely in/out of the surface
        if (edgeTable[cubeindex] == 0)
          continue;

        gc.cubeindex = cubeindex;

        gc.p[0].x = (float) x;
        gc.p[0].y = (float) y;
        gc.p[0].z = (float) z;
        VOXEL_GRADIENT_FAST(volgradient, psz, rsz, x, y, z, &gc.g[0].x)

        gc.p[1].x = (float) x + step;
        gc.p[1].y = (float) y;
        gc.p[1].z = (float) z;
        VOXEL_GRADIENT_FAST(volgradient, psz, rsz, x + step, y, z, &gc.g[1].x)

        gc.p[3].x = (float) x;
        gc.p[3].y = (float) y + step;
        gc.p[3].z = (float) z;
        VOXEL_GRADIENT_FAST(volgradient, psz, rsz, x, y + step, z, &gc.g[3].x)

        gc.p[2].x = (float) x + step;
        gc.p[2].y = (float) y + step;
        gc.p[2].z = (float) z;
        VOXEL_GRADIENT_FAST(volgradient, psz, rsz, x + step, y + step, z, &gc.g[2].x)

        gc.p[4].x = (float) x;
        gc.p[4].y = (float) y;
        gc.p[4].z = (float) z + step;
        VOXEL_GRADIENT_FAST(volgradient, psz, rsz, x, y, z + step, &gc.g[4].x)

        gc.p[5].x = (float) x + step;
        gc.p[5].y = (float) y;
        gc.p[5].z = (float) z + step;
        VOXEL_GRADIENT_FAST(volgradient, psz, rsz, x + step, y, z + step, &gc.g[5].x)

        gc.p[7].x = (float) x;
        gc.p[7].y = (float) y + step;
        gc.p[7].z = (float) z + step;
        VOXEL_GRADIENT_FAST(volgradient, psz, rsz, x, y + step, z + step, &gc.g[7].x)

        gc.p[6].x = (float) x + step;
        gc.p[6].y = (float) y + step;
        gc.p[6].z = (float) z + step;
        VOXEL_GRADIENT_FAST(volgradient, psz, rsz, x + step, y + step, z + step, &gc.g[6].x)

        // calculate vertices and facets for this cube,
        // calculate normals by interpolating between the negated 
        // normalized volume gradients for the 8 reference voxels
        tricount = Polygonise(gc, isovalue, (TRIANGLE *) &tris);
        globtricount += tricount;

        long i;
        for (i=0; i<tricount; i++) {
          int tritmp = numtriangles * 3;
          f.append3(tritmp, tritmp+1, tritmp+2);
          numtriangles++;

          // add new vertices and normals into vertex and normal lists
          v.append3((float) vol->origin[0] + tris[i].p[0].x * xax[0],
                    (float) vol->origin[1] + tris[i].p[0].y * yax[1],
                    (float) vol->origin[2] + tris[i].p[0].z * zax[2]);
          n.append3(tris[i].n[0].x, tris[i].n[0].y, tris[i].n[0].z);

          v.append3((float) vol->origin[0] + tris[i].p[1].x * xax[0],
                    (float) vol->origin[1] + tris[i].p[1].y * yax[1],
                    (float) vol->origin[2] + tris[i].p[1].z * zax[2]);
          n.append3(tris[i].n[1].x, tris[i].n[1].y, tris[i].n[1].z);

          v.append3((float) vol->origin[0] + tris[i].p[2].x * xax[0],
                    (float) vol->origin[1] + tris[i].p[2].y * yax[1],
                    (float) vol->origin[2] + tris[i].p[2].z * zax[2]);
          n.append3(tris[i].n[2].x, tris[i].n[2].y, tris[i].n[2].z);
        }
      }
    }
  }

  return globtricount;
}


long IsoSurface::DoCellGeneral(long x, long y, long z, float isovalue, long step) {
  GRIDCELL gc;
  long addr, row, plane, rowstep, planestep, tricount;
  TRIANGLE tris[5];

  // get direct access to gradient data for speed, and force 
  // generation of the gradients if they don't already exist
  const float *volgradient = vol->access_volume_gradient();

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

  // Determine the index into the edge table which
  // tells us which vertices are inside of the surface
  int cubeindex = 0;
  if (gc.val[0] < isovalue) cubeindex |= 1;
  if (gc.val[1] < isovalue) cubeindex |= 2;
  if (gc.val[2] < isovalue) cubeindex |= 4;
  if (gc.val[3] < isovalue) cubeindex |= 8;
  if (gc.val[4] < isovalue) cubeindex |= 16;
  if (gc.val[5] < isovalue) cubeindex |= 32;
  if (gc.val[6] < isovalue) cubeindex |= 64;
  if (gc.val[7] < isovalue) cubeindex |= 128;

  // Cube is entirely in/out of the surface
  if (edgeTable[cubeindex] == 0)
    return 0;

  // precompute plane and row sizes to eliminate indirection
  long psz = long(vol->xsize)*long(vol->ysize);
  long rsz = long(vol->xsize);

  gc.cubeindex = cubeindex;

  gc.p[0].x = (float) x;
  gc.p[0].y = (float) y;
  gc.p[0].z = (float) z;
  VOXEL_GRADIENT_FAST(volgradient, psz, rsz, x, y, z, &gc.g[0].x)

  gc.p[1].x = (float) x + step;
  gc.p[1].y = (float) y;
  gc.p[1].z = (float) z;
  VOXEL_GRADIENT_FAST(volgradient, psz, rsz, x + step, y, z, &gc.g[1].x)

  gc.p[3].x = (float) x;
  gc.p[3].y = (float) y + step;
  gc.p[3].z = (float) z;
  VOXEL_GRADIENT_FAST(volgradient, psz, rsz, x, y + step, z, &gc.g[3].x)

  gc.p[2].x = (float) x + step;
  gc.p[2].y = (float) y + step;
  gc.p[2].z = (float) z;
  VOXEL_GRADIENT_FAST(volgradient, psz, rsz, x + step, y + step, z, &gc.g[2].x)

  gc.p[4].x = (float) x;
  gc.p[4].y = (float) y;
  gc.p[4].z = (float) z + step;
  VOXEL_GRADIENT_FAST(volgradient, psz, rsz, x, y, z + step, &gc.g[4].x)

  gc.p[5].x = (float) x + step;
  gc.p[5].y = (float) y;
  gc.p[5].z = (float) z + step;
  VOXEL_GRADIENT_FAST(volgradient, psz, rsz, x + step, y, z + step, &gc.g[5].x)

  gc.p[7].x = (float) x;
  gc.p[7].y = (float) y + step;
  gc.p[7].z = (float) z + step;
  VOXEL_GRADIENT_FAST(volgradient, psz, rsz, x, y + step, z + step, &gc.g[7].x)

  gc.p[6].x = (float) x + step;
  gc.p[6].y = (float) y + step;
  gc.p[6].z = (float) z + step;
  VOXEL_GRADIENT_FAST(volgradient, psz, rsz, x + step, y + step, z + step, &gc.g[6].x)

  // calculate vertices and facets for this cube,
  // calculate normals by interpolating between the negated 
  // normalized volume gradients for the 8 reference voxels
  tricount = Polygonise(gc, isovalue, (TRIANGLE *) &tris);

  long i;
  for (i=0; i<tricount; i++) {
    float xx, yy, zz;
    float xn, yn, zn;

    int tritmp = numtriangles * 3;
    f.append3(tritmp, tritmp+1, tritmp+2);
    numtriangles++;

    // add new vertices and normals into vertex and normal lists
    xx = tris[i].p[0].x;
    yy = tris[i].p[0].y;
    zz = tris[i].p[0].z;

    v.append3((float) vol->origin[0] + xx * xax[0] + yy * yax[0] + zz * zax[0],
              (float) vol->origin[1] + xx * xax[1] + yy * yax[1] + zz * zax[1],
              (float) vol->origin[2] + xx * xax[2] + yy * yax[2] + zz * zax[2]);

    xn = tris[i].n[0].x;
    yn = tris[i].n[0].y;
    zn = tris[i].n[0].z;
    n.append3((float) xn * xad[0] + yn * yad[0] + zn * zad[0],
              (float) xn * xad[1] + yn * yad[1] + zn * zad[1],
              (float) xn * xad[2] + yn * yad[2] + zn * zad[2]);

    xx = tris[i].p[1].x;
    yy = tris[i].p[1].y;
    zz = tris[i].p[1].z;
    v.append3((float) vol->origin[0] + xx * xax[0] + yy * yax[0] + zz * zax[0],
              (float) vol->origin[1] + xx * xax[1] + yy * yax[1] + zz * zax[1],
              (float) vol->origin[2] + xx * xax[2] + yy * yax[2] + zz * zax[2]);

    xn = tris[i].n[1].x;
    yn = tris[i].n[1].y;
    zn = tris[i].n[1].z;
    n.append3((float) xn * xad[0] + yn * yad[0] + zn * zad[0],
              (float) xn * xad[1] + yn * yad[1] + zn * zad[1],
              (float) xn * xad[2] + yn * yad[2] + zn * zad[2]);

    xx = tris[i].p[2].x;
    yy = tris[i].p[2].y;
    zz = tris[i].p[2].z;
    v.append3((float) vol->origin[0] + xx * xax[0] + yy * yax[0] + zz * zax[0],
              (float) vol->origin[1] + xx * xax[1] + yy * yax[1] + zz * zax[1],
              (float) vol->origin[2] + xx * xax[2] + yy * yax[2] + zz * zax[2]);

    xn = tris[i].n[2].x;
    yn = tris[i].n[2].y;
    zn = tris[i].n[2].z;
    n.append3((float) xn * xad[0] + yn * yad[0] + zn * zad[0],
              (float) xn * xad[1] + yn * yad[1] + zn * zad[1],
              (float) xn * xad[2] + yn * yad[2] + zn * zad[2]);
  }

  return tricount;
}



// normalize surface normals resulting from interpolation between 
// unnormalized volume gradients
void IsoSurface::normalize() {
  int i;
  for (i=0; i<n.num(); i+=3) {
    vec_normalize(&n[i]); 
  }  
}

// merge duplicated vertices detected by a simple windowed search
int IsoSurface::vertexfusion(int offset, int len) {
  int i, j, newverts, oldverts, faceverts, matchcount;

  faceverts = f.num();
  oldverts = v.num() / 3; 

  // abort if we get an empty list
  if (!faceverts || !oldverts)
    return 0;

  int * vmap = new int[oldverts];

  vmap[0] = 0;
  newverts = 1;
  matchcount = 0;

  for (i=1; i<oldverts; i++) {
    int matchindex = -1;
    int start = ((newverts - offset) < 0)  ? 0        : (newverts - offset);
    int end   = ((start + len) > newverts) ? newverts : (start + len);
    int matched = 0;
    int vi = i * 3;
    for (j=start; j<end; j++) {
      int vj = j * 3;
      if (v[vi  ] == v[vj  ] && 
          v[vi+1] == v[vj+1] &&
          v[vi+2] == v[vj+2]) {
        matched = 1;
        matchindex = j;
        matchcount++;
        break;
      } 
    }

    if (matched) {
      vmap[i] = matchindex;
    } else {
      int vn = newverts * 3;
      v[vn    ] = v[vi    ];
      v[vn + 1] = v[vi + 1];
      v[vn + 2] = v[vi + 2];
      n[vn    ] = n[vi    ];
      n[vn + 1] = n[vi + 1];
      n[vn + 2] = n[vi + 2];
      vmap[i] = newverts;
      newverts++;
    } 
  }

//  printf("Info) Vertex fusion: found %d shared vertices of %d, %d unique\n",
//         matchcount, oldverts, newverts);

  // zap the old face, vertex, and normal arrays and replace with the new ones
  for (i=0; i<faceverts; i++) {
    f[i] = vmap[f[i]];
  }
  delete [] vmap;

  v.truncatelastn((oldverts - newverts) * 3);
  n.truncatelastn((oldverts - newverts) * 3);

  return 0;
}


/*
   Given a grid cell and an isolevel, calculate the triangular
   facets required to represent the isosurface through the cell.
   Return the number of triangular facets, the array "triangles"
   will be loaded up with the vertices at most 5 triangular facets.
        0 will be returned if the grid cell is either totally above
   of totally below the isolevel.
   This code calculates vertex normals by interpolating the volume gradients.
*/
int IsoSurface::Polygonise(const GRIDCELL grid, const float isolevel, TRIANGLE *triangles) {
   int i,ntriang;
   int cubeindex = grid.cubeindex;
   XYZ vertlist[12];
   XYZ normlist[12];

   /* Find the vertices where the surface intersects the cube */
   if (edgeTable[cubeindex] & 1)
      VertexInterp(isolevel, grid, 0, 1, &vertlist[0], &normlist[0]);
   if (edgeTable[cubeindex] & 2)
      VertexInterp(isolevel, grid, 1, 2, &vertlist[1], &normlist[1]);
   if (edgeTable[cubeindex] & 4)
      VertexInterp(isolevel, grid, 2, 3, &vertlist[2], &normlist[2]);
   if (edgeTable[cubeindex] & 8)
      VertexInterp(isolevel, grid, 3, 0, &vertlist[3], &normlist[3]);
   if (edgeTable[cubeindex] & 16)
      VertexInterp(isolevel, grid, 4, 5, &vertlist[4], &normlist[4]);
   if (edgeTable[cubeindex] & 32)
      VertexInterp(isolevel, grid, 5, 6, &vertlist[5], &normlist[5]);
   if (edgeTable[cubeindex] & 64)
      VertexInterp(isolevel, grid, 6, 7, &vertlist[6], &normlist[6]);
   if (edgeTable[cubeindex] & 128)
      VertexInterp(isolevel, grid, 7, 4, &vertlist[7], &normlist[7]);
   if (edgeTable[cubeindex] & 256)
      VertexInterp(isolevel, grid, 0, 4, &vertlist[8], &normlist[8]);
   if (edgeTable[cubeindex] & 512)
      VertexInterp(isolevel, grid, 1, 5, &vertlist[9], &normlist[9]);
   if (edgeTable[cubeindex] & 1024)
      VertexInterp(isolevel, grid, 2, 6, &vertlist[10], &normlist[10]);
   if (edgeTable[cubeindex] & 2048)
      VertexInterp(isolevel, grid, 3, 7, &vertlist[11], &normlist[11]);

   /* Create the triangle */
   ntriang = 0;
   for (i=0;triTable[cubeindex][i]!=-1;i+=3) {
     triangles[ntriang].p[0] = vertlist[triTable[cubeindex][i  ]];
     triangles[ntriang].p[1] = vertlist[triTable[cubeindex][i+1]];
     triangles[ntriang].p[2] = vertlist[triTable[cubeindex][i+2]];
     triangles[ntriang].n[0] = normlist[triTable[cubeindex][i  ]];
     triangles[ntriang].n[1] = normlist[triTable[cubeindex][i+1]];
     triangles[ntriang].n[2] = normlist[triTable[cubeindex][i+2]];
     ntriang++;
   }

   return ntriang;
}


/*
   Linearly interpolate the position where an isosurface cuts
   an edge between two vertices, each with their own scalar value,
   interpolating vertex position and vertex normal based on the 
   isovalue.
*/
void IsoSurface::VertexInterp(float isolevel, const GRIDCELL grid, int ind1, int ind2, XYZ * vert, XYZ * norm) {
  XYZ p1 = grid.p[ind1];
  XYZ p2 = grid.p[ind2];
  XYZ n1 = grid.g[ind1];
  XYZ n2 = grid.g[ind2];
  float valp1 = grid.val[ind1];
  float valp2 = grid.val[ind2];
  float isodiffp1   = isolevel - valp1;
  float diffvalp2p1 = valp2 - valp1;
  float mu = 0.0f;

  // if the difference between vertex values is zero or nearly
  // zero, we can get an IEEE NAN for mu.  We must either avoid this
  // by testing the denominator beforehand, by coping with the resulting
  // NAN value after the fact.  The only important thing is that mu be
  // assigned a value between zero and one.
  
#if 0
  if (fabsf(isodiffp1) < 0.00001) {
    *vert = p1;
    *norm = n1;
    return;
  }

  if (fabsf(isolevel-valp2) < 0.00001) {
    *vert = p2;
    *norm = n2;
    return;
  }

  if (fabsf(diffvalp2p1) < 0.00001) {
    *vert = p1;
    *norm = n1;
    return;
  }
#endif

  if (fabsf(diffvalp2p1) > 0.0f) 
    mu = isodiffp1 / diffvalp2p1;

#if 0
  if (mu > 1.0f)
    mu=1.0f;

  if (mu < 0.0f)
    mu=0.0f;
#endif

  vert->x = p1.x + mu * (p2.x - p1.x);
  vert->y = p1.y + mu * (p2.y - p1.y);
  vert->z = p1.z + mu * (p2.z - p1.z);

  norm->x = n1.x + mu * (n2.x - n1.x);
  norm->y = n1.y + mu * (n2.y - n1.y);
  norm->z = n1.z + mu * (n2.z - n1.z);
}


/// assign a single color for the entire mesh
int IsoSurface::set_color_rgb3fv(const float *rgb) {
  int i;
  int numverts = v.num() / 3; 
  for (i=0; i<numverts; i++) {
    c.append3(&rgb[0]); // red, green, blue color components
  }

  return 0;
}

/// assign per-vertex colors from a volumetric texture map with the
/// same dimensions as the original volumetric data
/// XXX only handles orthogonal volumes currently 
int IsoSurface::set_color_voltex_rgb3fv(const float *voltex) {
  int i;
  int numverts = v.num() / 3;
  int row = vol->xsize; 
  int plane = vol->xsize * vol->ysize;
 
  float xinv = 1.0f / xax[0];
  float yinv = 1.0f / yax[1];
  float zinv = 1.0f / zax[2];
  int xs = vol->xsize - 1;
  int ys = vol->ysize - 1;
  int zs = vol->zsize - 1;

  for (i=0; i<numverts; i++) {
    int ind = i*3;
    float vx = float((v[ind    ] - vol->origin[0]) * xinv);
    float vy = float((v[ind + 1] - vol->origin[1]) * yinv);
    float vz = float((v[ind + 2] - vol->origin[2]) * zinv);

    int x = MIN(MAX(((int) vx), 0), xs);
    int y = MIN(MAX(((int) vy), 0), ys);
    int z = MIN(MAX(((int) vz), 0), zs);

    int caddr = (z*plane + y*row + x)*3;

#if 0
    // non-interpolated texture lookup
    float rgb[3];
    vec_copy(rgb, &voltex[caddr]);

    // normalize colors to eliminate gridding artifacts for the time being
    vec_normalize(rgb);
#else
    float mux = (vx - x);
    float muy = (vy - y);
    float muz = (vz - z);

    float c0[3], c1[3], c2[3], c3[3]; 

    int caddrp1 = (x < xs) ? caddr+3 : caddr;
    c0[0] = (1-mux)*voltex[caddr    ] + mux*voltex[caddrp1    ];
    c0[1] = (1-mux)*voltex[caddr + 1] + mux*voltex[caddrp1 + 1];
    c0[2] = (1-mux)*voltex[caddr + 2] + mux*voltex[caddrp1 + 2];

    int yinc = (y < ys) ? row*3 : 0; 
    caddr += yinc;
    caddrp1 += yinc;
    c1[0] = (1-mux)*voltex[caddr    ] + mux*voltex[caddrp1    ];
    c1[1] = (1-mux)*voltex[caddr + 1] + mux*voltex[caddrp1 + 1];
    c1[2] = (1-mux)*voltex[caddr + 2] + mux*voltex[caddrp1 + 2];
    
    int zinc = (z < zs) ? plane*3 : 0; 
    caddr += zinc;
    caddrp1 += zinc;
    c3[0] = (1-mux)*voltex[caddr    ] + mux*voltex[caddrp1    ];
    c3[1] = (1-mux)*voltex[caddr + 1] + mux*voltex[caddrp1 + 1];
    c3[2] = (1-mux)*voltex[caddr + 2] + mux*voltex[caddrp1 + 2];

    caddr -= yinc;
    caddrp1 -= yinc;
    c2[0] = (1-mux)*voltex[caddr    ] + mux*voltex[caddrp1    ];
    c2[1] = (1-mux)*voltex[caddr + 1] + mux*voltex[caddrp1 + 1];
    c2[2] = (1-mux)*voltex[caddr + 2] + mux*voltex[caddrp1 + 2];

    float cz0[3];
    cz0[0] = (1-muy)*c0[0] + muy*c1[0];
    cz0[1] = (1-muy)*c0[1] + muy*c1[1];
    cz0[2] = (1-muy)*c0[2] + muy*c1[2];

    float cz1[3];
    cz1[0] = (1-muy)*c2[0] + muy*c3[0];
    cz1[1] = (1-muy)*c2[1] + muy*c3[1];
    cz1[2] = (1-muy)*c2[2] + muy*c3[2];
   
    float rgb[3];
    rgb[0] = (1-muz)*cz0[0] + muz*cz1[0];
    rgb[1] = (1-muz)*cz0[1] + muz*cz1[1];
    rgb[2] = (1-muz)*cz0[2] + muz*cz1[2];
#endif

    c.append3(&rgb[0]);
  }

  return 0;
}




