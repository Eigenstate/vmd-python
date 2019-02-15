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
 *      $RCSfile: CUDAMarchingCubes.cu,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.35 $       $Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   CUDA Marching Cubes Implementation
 *
 * LICENSE:
 *   UIUC Open Source License
 *   http://www.ks.uiuc.edu/Research/vmd/plugins/pluginlicense.html
 *
 ***************************************************************************/

//
// Description: This class computes an isosurface for a given density grid
//              using a CUDA Marching Cubes (MC) alorithm. 
//
//              The implementation is loosely based on the MC demo from 
//              the Nvidia GPU Computing SDK, but the design has been 
//              improved and extended in several ways.  
//
//              This implementation achieves higher performance
//              by reducing the number of temporary memory
//              buffers, reduces the number of scan calls by using 
//              vector integer types, and allows extraction of 
//              per-vertex normals and optionally computes 
//              per-vertex colors if a volumetric texture map is provided.
//
// Author: Michael Krone <michael.krone@visus.uni-stuttgart.de>
//         John Stone <johns@ks.uiuc.edu>
//
// Copyright 2011
//

#include "CUDAKernels.h"
#define CUDAMARCHINGCUBES_INTERNAL 1
#include "CUDAMarchingCubes.h"
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/functional.h>

#include "CUDAParPrefixOps.h"

//
// Restrict macro to make it easy to do perf tuning tests
//
#if 0
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif

// The number of threads to use for triangle generation 
// (limited by shared memory size)
#define NTHREADS 48

// The number of threads to use for all of the other kernels
#define KERNTHREADS 256

//
// Various math operators for vector types not already 
// provided by the regular CUDA headers
//
// "+" operator
inline __host__ __device__ float3 operator+(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ uint3 operator+(uint3 a, uint3 b) {
  return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ uint2 operator+(uint2 a, uint2 b) {
  return make_uint2(a.x + b.x, a.y + b.y);
}

// "-" operator
inline __host__ __device__ float3 operator-(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ uint3 operator-(uint3 a, uint3 b) {
  return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z);
}

// "*" operator
inline __host__ __device__ float3 operator*(float b, float3 a) {
  return make_float3(b * a.x, b * a.y, b * a.z);
}

// dot()
inline __host__ __device__ float dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

// fma() for float and float3
inline __host__ __device__ float3 fmaf3(float x, float3 y, float3 z) {
  return make_float3(fmaf(x, y.x, z.x), 
                     fmaf(x, y.y, z.y), 
                     fmaf(x, y.z, z.z));
}


// lerp()
inline __device__ __host__ float3 lerp(float3 a, float3 b, float t) {
#if 0
  return fmaf3(t, b, fmaf3(-t, a, a));
#elif 1
  return a + t*(b-a);
#else
  return (1-t)*a + t*b;
#endif
}

// length()
inline __host__ __device__ float length(float3 v) {
  return sqrtf(dot(v, v));
}

//
// color format conversion routines
//
inline __device__ void convert_color(float3 & cf, float3 cf2) {
  cf = cf2;
}

inline __device__ void convert_color(uchar4 & cu, float3 cf) {
  // conversion to GLubyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
  // clamp color values to prevent integer wraparound
  cu = make_uchar4(fminf(cf.x * 255.0f, 255.0f),
                   fminf(cf.y * 255.0f, 255.0f),
                   fminf(cf.z * 255.0f, 255.0f),
                   255);
}

inline __device__ void convert_color(float3 & cf, uchar4 cu) {
  const float i2f = 1.0f / 255.0f;
  cf.x = cu.x * i2f;
  cf.y = cu.y * i2f;
  cf.z = cu.z * i2f;
}

//
// normal format conversion routines
//
inline __device__ void convert_normal(float3 & nf, float3 nf2) {
  nf = nf2;
}

inline __device__ void convert_normal(char3 & cc, float3 cf) {
  // normalize input values before conversion to fixed point representation
  float invlen = rsqrtf(cf.x*cf.x + cf.y*cf.y + cf.z*cf.z);

  // conversion to GLbyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
  cc = make_char3(cf.x * invlen * 127.5f - 0.5f,
                  cf.y * invlen * 127.5f - 0.5f,
                  cf.z * invlen * 127.5f - 0.5f);
}


//
// CUDA textures containing marching cubes look-up tables
// Note: SIMD marching cubes implementations have no need for the edge table
//
texture<int, 1, cudaReadModeElementType> triTex;
texture<unsigned int, 1, cudaReadModeElementType> numVertsTex;
// 3D 24-bit RGB texture
texture<float, 3, cudaReadModeElementType> volumeTex;

// sample volume data set at a point p, p CAN NEVER BE OUT OF BOUNDS
// XXX The sampleVolume() call underperforms vs. peak memory bandwidth
//     because we don't strictly enforce coalescing requirements in the
//     layout of the input volume presently.  If we forced X/Y dims to be
//     warp-multiple it would become possible to use wider fetches and
//     a few other tricks to improve global memory bandwidth 
__device__ float sampleVolume(const float * RESTRICT data, 
                              uint3 p, uint3 gridSize) {
    return data[(p.z*gridSize.x*gridSize.y) + (p.y*gridSize.x) + p.x];
}


// sample volume texture at a point p, p CAN NEVER BE OUT OF BOUNDS
__device__ float3 sampleColors(const float3 * RESTRICT data,
                               uint3 p, uint3 gridSize) {
    return data[(p.z*gridSize.x*gridSize.y) + (p.y*gridSize.x) + p.x];
}

// sample volume texture at a point p, p CAN NEVER BE OUT OF BOUNDS
__device__ float3 sampleColors(const uchar4 * RESTRICT data,
                               uint3 p, uint3 gridSize) {
    uchar4 cu = data[(p.z*gridSize.x*gridSize.y) + (p.y*gridSize.x) + p.x];
    float3 cf;
    convert_color(cf, cu);
    return cf;
}


// compute position in 3d grid from 1d index
__device__ uint3 calcGridPos(unsigned int i, uint3 gridSize) {
    uint3 gridPos;
    unsigned int gridsizexy = gridSize.x * gridSize.y;
    gridPos.z = i / gridsizexy;
    unsigned int tmp1 = i - (gridsizexy * gridPos.z);
    gridPos.y = tmp1 / gridSize.x;
    gridPos.x = tmp1 - (gridSize.x * gridPos.y);
    return gridPos;
}


// compute interpolated vertex along an edge
__device__ float3 vertexInterp(float isolevel, float3 p0, float3 p1, float f0, float f1) {
    float t = (isolevel - f0) / (f1 - f0);
    return lerp(p0, p1, t);
}


// classify voxel based on number of vertices it will generate one thread per two voxels
template <int gridis3d, int subgrid>
__global__ void 
// __launch_bounds__ ( KERNTHREADS, 1 )
classifyVoxel(uint2 * RESTRICT voxelVerts, 
              const float * RESTRICT volume,
              uint3 gridSize, unsigned int numVoxels, float3 voxelSize,
              uint3 subGridStart, uint3 subGridEnd,
              float isoValue) {
    uint3 gridPos;
    unsigned int i;

    // Compute voxel indices and address using either 2-D or 3-D 
    // thread indexing depending on the caller-provided gridis3d parameter
    if (gridis3d) {
      // Compute voxel index using 3-D thread indexing
      // compute 3D grid position
      gridPos.x = blockIdx.x * blockDim.x + threadIdx.x;
      gridPos.y = blockIdx.y * blockDim.y + threadIdx.y;
      gridPos.z = blockIdx.z * blockDim.z + threadIdx.z;

      // safety check
      if (gridPos.x >= gridSize.x || 
          gridPos.y >= gridSize.y || 
          gridPos.z >= gridSize.z)
        return;

      // compute 1D grid index
      i = gridPos.z*gridSize.x*gridSize.y + gridPos.y*gridSize.x + gridPos.x;
    } else {
      // Compute voxel index using 2-D thread indexing
      unsigned int blockId = (blockIdx.y * gridDim.x) + blockIdx.x;
      i = (blockId * blockDim.x) + threadIdx.x;

      // safety check
      if (i >= numVoxels)
        return;

      // compute current grid position
      gridPos = calcGridPos(i, gridSize);
    }

    // If we are told to compute the isosurface for only a sub-region
    // of the volume, we use a more complex boundary test, otherwise we
    // use just the maximum voxel dimension
    uint2 numVerts = make_uint2(0, 0); // initialize vertex output to zero
    if (subgrid) {
      if (gridPos.x < subGridStart.x || 
          gridPos.y < subGridStart.y || 
          gridPos.z < subGridStart.z ||
          gridPos.x >= subGridEnd.x || 
          gridPos.y >= subGridEnd.y || 
          gridPos.z >= subGridEnd.z) {
        voxelVerts[i] = numVerts; // no vertices returned
        return;
      }
    } else {
      if (gridPos.x > (gridSize.x - 2) || gridPos.y > (gridSize.y - 2) || gridPos.z > (gridSize.z - 2)) {
        voxelVerts[i] = numVerts; // no vertices returned
        return;
      }
    }

    // read field values at neighbouring grid vertices
    float field[8];
    field[0] = sampleVolume(volume, gridPos, gridSize);
    // TODO early exit test for
    //if (field[0] < 0.000001f)  {
    //    voxelVerts[i] = numVerts;
    //    return;
    //}
    field[1] = sampleVolume(volume, gridPos + make_uint3(1, 0, 0), gridSize);
    field[2] = sampleVolume(volume, gridPos + make_uint3(1, 1, 0), gridSize);
    field[3] = sampleVolume(volume, gridPos + make_uint3(0, 1, 0), gridSize);
    field[4] = sampleVolume(volume, gridPos + make_uint3(0, 0, 1), gridSize);
    field[5] = sampleVolume(volume, gridPos + make_uint3(1, 0, 1), gridSize);
    field[6] = sampleVolume(volume, gridPos + make_uint3(1, 1, 1), gridSize);
    field[7] = sampleVolume(volume, gridPos + make_uint3(0, 1, 1), gridSize);

    // calculate flag indicating if each vertex is inside or outside isosurface
    unsigned int cubeindex;
    cubeindex =  ((unsigned int) (field[0] < isoValue));
    cubeindex += ((unsigned int) (field[1] < isoValue))*2;
    cubeindex += ((unsigned int) (field[2] < isoValue))*4;
    cubeindex += ((unsigned int) (field[3] < isoValue))*8;
    cubeindex += ((unsigned int) (field[4] < isoValue))*16;
    cubeindex += ((unsigned int) (field[5] < isoValue))*32;
    cubeindex += ((unsigned int) (field[6] < isoValue))*64;
    cubeindex += ((unsigned int) (field[7] < isoValue))*128;

    // read number of vertices from texture
    numVerts.x = tex1Dfetch(numVertsTex, cubeindex);
    numVerts.y = (numVerts.x > 0);

    voxelVerts[i] = numVerts;
}


// compact voxel array
__global__ void 
// __launch_bounds__ ( KERNTHREADS, 1 )
compactVoxels(unsigned int * RESTRICT compactedVoxelArray, 
              const uint2 * RESTRICT voxelOccupied, 
              unsigned int lastVoxel, unsigned int numVoxels, 
              unsigned int numVoxelsm1) {
  unsigned int blockId = (blockIdx.y * gridDim.x) + blockIdx.x;
  unsigned int i = (blockId * blockDim.x) + threadIdx.x;

  if ((i < numVoxels) && ((i < numVoxelsm1) ? voxelOccupied[i].y < voxelOccupied[i+1].y : lastVoxel)) {
    compactedVoxelArray[ voxelOccupied[i].y ] = i;
  }
}


// version that calculates no surface normal or color,  only triangle vertices
__global__ void 
// __launch_bounds__ ( NTHREADS, 1 )
generateTriangleVerticesSMEM(float3 * RESTRICT pos, 
                             const unsigned int * RESTRICT compactedVoxelArray, 
                             const uint2 * RESTRICT numVertsScanned, 
                             const float * RESTRICT volume,
                             uint3 gridSize, float3 voxelSize, 
                             float isoValue, unsigned int activeVoxels, 
                             unsigned int maxVertsM3) {
    unsigned int zOffset = blockIdx.z * gridDim.x * gridDim.y;
    unsigned int blockId = (blockIdx.y * gridDim.x) + blockIdx.x;
    unsigned int i = zOffset * (blockDim.x * blockDim.y) + (blockId * blockDim.x) + threadIdx.x;

    if (i >= activeVoxels)
        return;

    unsigned int voxel = compactedVoxelArray[i];

    // compute position in 3d grid
    uint3 gridPos = calcGridPos(voxel, gridSize);

    float3 p;
    p.x = gridPos.x * voxelSize.x;
    p.y = gridPos.y * voxelSize.y;
    p.z = gridPos.z * voxelSize.z;

    // calculate cell vertex positions
    float3 v[8];
    v[0] = p;
    v[1] = p + make_float3(voxelSize.x, 0, 0);
    v[2] = p + make_float3(voxelSize.x, voxelSize.y, 0);
    v[3] = p + make_float3(0, voxelSize.y, 0);
    v[4] = p + make_float3(0, 0, voxelSize.z);
    v[5] = p + make_float3(voxelSize.x, 0, voxelSize.z);
    v[6] = p + make_float3(voxelSize.x, voxelSize.y, voxelSize.z);
    v[7] = p + make_float3(0, voxelSize.y, voxelSize.z);

    float field[8];
    field[0] = sampleVolume(volume, gridPos, gridSize);
    field[1] = sampleVolume(volume, gridPos + make_uint3(1, 0, 0), gridSize);
    field[2] = sampleVolume(volume, gridPos + make_uint3(1, 1, 0), gridSize);
    field[3] = sampleVolume(volume, gridPos + make_uint3(0, 1, 0), gridSize);
    field[4] = sampleVolume(volume, gridPos + make_uint3(0, 0, 1), gridSize);
    field[5] = sampleVolume(volume, gridPos + make_uint3(1, 0, 1), gridSize);
    field[6] = sampleVolume(volume, gridPos + make_uint3(1, 1, 1), gridSize);
    field[7] = sampleVolume(volume, gridPos + make_uint3(0, 1, 1), gridSize);

    // recalculate flag
    unsigned int cubeindex;
    cubeindex =  ((unsigned int)(field[0] < isoValue)); 
    cubeindex += ((unsigned int)(field[1] < isoValue))*2; 
    cubeindex += ((unsigned int)(field[2] < isoValue))*4; 
    cubeindex += ((unsigned int)(field[3] < isoValue))*8; 
    cubeindex += ((unsigned int)(field[4] < isoValue))*16; 
    cubeindex += ((unsigned int)(field[5] < isoValue))*32; 
    cubeindex += ((unsigned int)(field[6] < isoValue))*64; 
    cubeindex += ((unsigned int)(field[7] < isoValue))*128;

    // find the vertices where the surface intersects the cube 
    // Note: SIMD marching cubes implementations have no need
    //       for an edge table, because branch divergence eliminates any
    //       potential performance gain from only computing the per-edge
    //       vertices when indicated by the edgeTable.

    // Use shared memory to keep register pressure under control.
    // No need to call __syncthreads() since each thread uses its own
    // private shared memory buffer.
    __shared__ float3 vertlist[12*NTHREADS];

    vertlist[threadIdx.x] = vertexInterp(isoValue, v[0], v[1], field[0], field[1]);
    vertlist[NTHREADS+threadIdx.x] = vertexInterp(isoValue, v[1], v[2], field[1], field[2]);
    vertlist[(NTHREADS*2)+threadIdx.x] = vertexInterp(isoValue, v[2], v[3], field[2], field[3]);
    vertlist[(NTHREADS*3)+threadIdx.x] = vertexInterp(isoValue, v[3], v[0], field[3], field[0]);
    vertlist[(NTHREADS*4)+threadIdx.x] = vertexInterp(isoValue, v[4], v[5], field[4], field[5]);
    vertlist[(NTHREADS*5)+threadIdx.x] = vertexInterp(isoValue, v[5], v[6], field[5], field[6]);
    vertlist[(NTHREADS*6)+threadIdx.x] = vertexInterp(isoValue, v[6], v[7], field[6], field[7]);
    vertlist[(NTHREADS*7)+threadIdx.x] = vertexInterp(isoValue, v[7], v[4], field[7], field[4]);
    vertlist[(NTHREADS*8)+threadIdx.x] = vertexInterp(isoValue, v[0], v[4], field[0], field[4]);
    vertlist[(NTHREADS*9)+threadIdx.x] = vertexInterp(isoValue, v[1], v[5], field[1], field[5]);
    vertlist[(NTHREADS*10)+threadIdx.x] = vertexInterp(isoValue, v[2], v[6], field[2], field[6]);
    vertlist[(NTHREADS*11)+threadIdx.x] = vertexInterp(isoValue, v[3], v[7], field[3], field[7]);

    // output triangle vertices
    unsigned int numVerts = tex1Dfetch(numVertsTex, cubeindex);
    for(int i=0; i<numVerts; i+=3) {
        unsigned int index = numVertsScanned[voxel].x + i;

        float3 *vert[3];
        int edge;
        edge = tex1Dfetch(triTex, (cubeindex*16) + i);
        vert[0] = &vertlist[(edge*NTHREADS)+threadIdx.x];

        edge = tex1Dfetch(triTex, (cubeindex*16) + i + 1);
        vert[1] = &vertlist[(edge*NTHREADS)+threadIdx.x];

        edge = tex1Dfetch(triTex, (cubeindex*16) + i + 2);
        vert[2] = &vertlist[(edge*NTHREADS)+threadIdx.x];

        if (index < maxVertsM3) {
            pos[index  ] = *vert[0];
            pos[index+1] = *vert[1];
            pos[index+2] = *vert[2];
        }
    }
}


__global__ void 
// __launch_bounds__ ( KERNTHREADS, 1 )
offsetTriangleVertices(float3 * RESTRICT pos,
                       float3 origin, unsigned int numVertsM1) {
  unsigned int zOffset = blockIdx.z * gridDim.x * gridDim.y;
  unsigned int blockId = (blockIdx.y * gridDim.x) + blockIdx.x;
  unsigned int i = zOffset + (blockId * blockDim.x) + threadIdx.x;

  if (i > numVertsM1)
    return;

  float3 p = pos[i];
  p.x += origin.x;
  p.y += origin.y;
  p.z += origin.z;
  pos[i] = p;
}


// version that calculates the surface normal for each triangle vertex
template <class NORMAL>
__global__ void 
// __launch_bounds__ ( KERNTHREADS, 1 )
generateTriangleNormals(const float3 * RESTRICT pos,
                        NORMAL *norm, float3 gridSizeInv, 
                        float3 bBoxInv, unsigned int numVerts) {
  unsigned int zOffset = blockIdx.z * gridDim.x * gridDim.y;
  unsigned int blockId = (blockIdx.y * gridDim.x) + blockIdx.x;
  unsigned int i = zOffset + (blockId * blockDim.x) + threadIdx.x;

  if (i > numVerts - 1)
    return;

  float3 n;
  float3 p, p1, p2;
  // normal calculation using central differences
  // TODO
  //p = (pos[i] + make_float3(1.0f)) * 0.5f;
  p = pos[i];
  p.x *= bBoxInv.x;
  p.y *= bBoxInv.y;
  p.z *= bBoxInv.z;
  p1 = p + make_float3(gridSizeInv.x, 0.0f, 0.0f);
  p2 = p - make_float3(gridSizeInv.x, 0.0f, 0.0f);
  n.x = tex3D(volumeTex, p2.x, p2.y, p2.z) - tex3D(volumeTex, p1.x, p1.y, p1.z);
  p1 = p + make_float3(0.0f, gridSizeInv.y, 0.0f);
  p2 = p - make_float3(0.0f, gridSizeInv.y, 0.0f);
  n.y = tex3D(volumeTex, p2.x, p2.y, p2.z) - tex3D(volumeTex, p1.x, p1.y, p1.z);
  p1 = p + make_float3(0.0f, 0.0f, gridSizeInv.z);
  p2 = p - make_float3(0.0f, 0.0f, gridSizeInv.z);
  n.z = tex3D(volumeTex, p2.x, p2.y, p2.z) - tex3D(volumeTex, p1.x, p1.y, p1.z);

  NORMAL no;
  convert_normal(no, n);
  norm[i] = no;
}


// version that calculates the surface normal and color for each triangle vertex
template <class VERTEXCOL, class VOLTEX, class NORMAL>
__global__ void
// __launch_bounds__ ( KERNTHREADS, 1 )
generateTriangleColorNormal(const float3 * RESTRICT pos, 
                            VERTEXCOL * RESTRICT col,
                            NORMAL * RESTRICT norm,
                            const VOLTEX * RESTRICT colors,
                            uint3 gridSize, float3 gridSizeInv, 
                            float3 bBoxInv, unsigned int numVerts) {
  unsigned int zOffset = blockIdx.z * gridDim.x * gridDim.y;
  unsigned int blockId = (blockIdx.y * gridDim.x) + blockIdx.x;
  unsigned int i = zOffset + (blockId * blockDim.x) + threadIdx.x;

  if (i > numVerts - 1)
    return;

  float3 p = pos[i];
  p.x *= bBoxInv.x;
  p.y *= bBoxInv.y;
  p.z *= bBoxInv.z;
  // color computation
  float3 gridPosF = p;
  gridPosF.x *= float(gridSize.x);
  gridPosF.y *= float(gridSize.y);
  gridPosF.z *= float(gridSize.z);
  float3 gridPosFloor;
  // Without the offset, rounding errors can occur
  // TODO why do we need the offset??
  gridPosFloor.x = floorf(gridPosF.x + 0.0001f);
  gridPosFloor.y = floorf(gridPosF.y + 0.0001f);
  gridPosFloor.z = floorf(gridPosF.z + 0.0001f);
  float3 gridPosCeil;
  // Without the offset, rounding errors can occur
  // TODO why do we need the offset??
  gridPosCeil.x = ceilf(gridPosF.x - 0.0001f);
  gridPosCeil.y = ceilf(gridPosF.y - 0.0001f);
  gridPosCeil.z = ceilf(gridPosF.z - 0.0001f);
  uint3 gridPos0;
  gridPos0.x = gridPosFloor.x;
  gridPos0.y = gridPosFloor.y;
  gridPos0.z = gridPosFloor.z;
  uint3 gridPos1;
  gridPos1.x = gridPosCeil.x;
  gridPos1.y = gridPosCeil.y;
  gridPos1.z = gridPosCeil.z;

  // compute interpolated color
  float3 field[2];
  field[0] = sampleColors(colors, gridPos0, gridSize);
  field[1] = sampleColors(colors, gridPos1, gridSize);
  float3 tmp = gridPosF - gridPosFloor;
  float a = fmaxf(fmaxf(tmp.x, tmp.y), tmp.z);
  VERTEXCOL c;
  convert_color(c, lerp(field[0], field[1], a));
  col[i] = c;

  // normal calculation using central differences
  float3 p1, p2, n;
  p1 = p + make_float3(gridSizeInv.x, 0.0f, 0.0f);
  p2 = p - make_float3(gridSizeInv.x, 0.0f, 0.0f);
  n.x = tex3D(volumeTex, p2.x, p2.y, p2.z) - tex3D(volumeTex, p1.x, p1.y, p1.z);
  p1 = p + make_float3(0.0f, gridSizeInv.y, 0.0f);
  p2 = p - make_float3(0.0f, gridSizeInv.y, 0.0f);
  n.y = tex3D(volumeTex, p2.x, p2.y, p2.z) - tex3D(volumeTex, p1.x, p1.y, p1.z);
  p1 = p + make_float3(0.0f, 0.0f, gridSizeInv.z);
  p2 = p - make_float3(0.0f, 0.0f, gridSizeInv.z);
  n.z = tex3D(volumeTex, p2.x, p2.y, p2.z) - tex3D(volumeTex, p1.x, p1.y, p1.z);

  NORMAL no;
  convert_normal(no, n);
  norm[i] = no;
}


void allocateTextures(int **d_triTable, unsigned int **d_numVertsTable) {
    cudaChannelFormatDesc channelDescSigned = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);
    cudaMalloc((void**) d_triTable, 256*16*sizeof(int));
    cudaMemcpy((void *)*d_triTable, (void *)triTable, 256*16*sizeof(int), cudaMemcpyHostToDevice);
    cudaBindTexture(0, triTex, *d_triTable, channelDescSigned);

    cudaChannelFormatDesc channelDescUnsigned = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
    cudaMalloc((void**) d_numVertsTable, 256*sizeof(unsigned int));
    cudaMemcpy((void *)*d_numVertsTable, (void *)numVertsTable, 256*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaBindTexture(0, numVertsTex, *d_numVertsTable, channelDescUnsigned);
}


void bindVolumeTexture(cudaArray *d_vol, cudaChannelFormatDesc desc) {
    // set texture parameters
    volumeTex.normalized = 1;
    volumeTex.filterMode = cudaFilterModeLinear;
    //volumeTex.filterMode = cudaFilterModePoint;
    volumeTex.addressMode[0] = cudaAddressModeClamp;
    volumeTex.addressMode[1] = cudaAddressModeClamp;
    volumeTex.addressMode[2] = cudaAddressModeClamp;
    // bind array to 3D texture
    cudaBindTextureToArray(volumeTex, d_vol, desc);
}


#if 0

void ThrustScanWrapperUint2(uint2* output, uint2* input, unsigned int numElements) {
  const uint2 zero = make_uint2(0, 0);
  long scanwork_sz = 0;
  scanwork_sz = dev_excl_scan_sum_tmpsz(input, ((long) numElements), output, zero);
  void *scanwork_d = NULL;
  cudaMalloc(&scanwork_d, scanwork_sz);
  dev_excl_scan_sum(input, ((long) numElements), output, scanwork_d, scanwork_sz, zero);
  cudaFree(scanwork_d);
}

#elif CUDART_VERSION >= 9000
//
// XXX CUDA 9.0RC breaks the usability of Thrust scan() prefix sums when
//     used with the built-in uint2 vector integer types.  To workaround
//     the problem we have to define our own type and associated conversion
//     routines etc.
//

// XXX workaround for uint2 breakage in all CUDA revs 9.0 through 10.0
struct myuint2 : uint2 {
  __host__ __device__ myuint2() : uint2(make_uint2(0, 0)) {}
  __host__ __device__ myuint2(int val) : uint2(make_uint2(val, val)) {}
  __host__ __device__ myuint2(uint2 val) : uint2(make_uint2(val.x, val.y)) {}
};

void ThrustScanWrapperUint2(uint2* output, uint2* input, unsigned int numElements) {
    const uint2 zero = make_uint2(0, 0);
    thrust::exclusive_scan(thrust::device_ptr<myuint2>((myuint2*)input),
                           thrust::device_ptr<myuint2>((myuint2*)input + numElements),
                           thrust::device_ptr<myuint2>((myuint2*)output),
                           (myuint2) zero);
}

#else

void ThrustScanWrapperUint2(uint2* output, uint2* input, unsigned int numElements) {
    const uint2 zero = make_uint2(0, 0);
    thrust::exclusive_scan(thrust::device_ptr<uint2>(input),
                           thrust::device_ptr<uint2>(input + numElements),
                           thrust::device_ptr<uint2>(output),
                           zero);
}

#endif

void ThrustScanWrapperArea(float* output, float* input, unsigned int numElements) {
    thrust::inclusive_scan(thrust::device_ptr<float>(input), 
                           thrust::device_ptr<float>(input + numElements),
                           thrust::device_ptr<float>(output));
}


__global__ void 
// __launch_bounds__ ( KERNTHREADS, 1 )
computeTriangleAreas(const float3 * RESTRICT pos,
                     float * RESTRICT area, unsigned int maxTria) {
    unsigned int zOffset = blockIdx.z * gridDim.x * gridDim.y;
    unsigned int blockId = (blockIdx.y * gridDim.x) + blockIdx.x;
    unsigned int i = zOffset + (blockId * blockDim.x) + threadIdx.x;

    // prevent overrunning of array boundary
    if (i >= maxTria)
      return;

    // get all three triangle vertices
    float3 v0 = pos[3*i];
    float3 v1 = pos[3*i+1];
    float3 v2 = pos[3*i+2];

    // compute edge lengths
    float a = length(v0 - v1);
    float b = length(v0 - v2);
    float c = length(v1 - v2);

    // compute area (Heron's formula)
    float rad = (a + b + c) * (a + b - c) * (b + c - a) * (c + a - b);
    // make sure radicand is not negative
    rad = rad > 0.0f ? rad : 0.0f;
    area[i] = 0.25f * sqrtf(rad);
}


///////////////////////////////////////////////////////////////////////////////
//
// class CUDAMarchingCubes
//
///////////////////////////////////////////////////////////////////////////////

CUDAMarchingCubes::CUDAMarchingCubes() {
    // initialize values
    isoValue = 0.5f;
    maxNumVoxels = 0;
    numVoxels    = 0;
    activeVoxels = 0;
    totalVerts   = 0;
    d_volume = 0;
    d_colors = 0;
    texformat = RGB3F;
    d_volumeArray = 0;
    d_voxelVerts = 0;
    d_numVertsTable = 0;
    d_triTable = 0;
    useColor = false;
    useSubGrid = false;
    initialized = false;
    setdata = false;
    cudadevice = 0;
    cudacomputemajor = 0;

    // Query GPU device attributes so we can launch the best kernel type
    cudaDeviceProp deviceProp;
    memset(&deviceProp, 0, sizeof(cudaDeviceProp));

    if (cudaGetDevice(&cudadevice) != cudaSuccess) {
      // XXX do something more useful here...
    }

    if (cudaGetDeviceProperties(&deviceProp, cudadevice) != cudaSuccess) {
      // XXX do something more useful here...
    }

    cudacomputemajor = deviceProp.major;
}


CUDAMarchingCubes::~CUDAMarchingCubes() {
    Cleanup();
}


void CUDAMarchingCubes::Cleanup() {
    if (d_triTable) cudaFree(d_triTable);
    if (d_numVertsTable) cudaFree(d_numVertsTable);
    if (d_voxelVerts) cudaFree(d_voxelVerts);
    if (d_compVoxelArray) cudaFree(d_compVoxelArray);
    if (ownCudaDataArrays) {
        if(d_volume) cudaFree(d_volume);
        if(d_colors) cudaFree(d_colors);
    }
    if (d_volumeArray) cudaFreeArray(d_volumeArray);

    maxNumVoxels = 0;
    numVoxels    = 0;
    d_triTable = 0;
    d_numVertsTable = 0;
    d_voxelVerts = 0;
    d_compVoxelArray = 0;
    d_volume = 0;
    d_colors = 0;
    ownCudaDataArrays = false;
    d_volumeArray = 0;
    initialized = false;
    setdata = false;
}


////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void CUDAMarchingCubes::computeIsosurfaceVerts(float3* vertOut, unsigned int maxverts, dim3 & grid3) {
    // check if data is available
    if (!this->setdata)
      return;

    int threads = 256;
    dim3 grid((unsigned int) (ceil(float(numVoxels) / float(threads))), 1, 1);
    // get around maximum grid size of 65535 in each dimension
    if (grid.x > 65535) {
        grid.y = (unsigned int) (ceil(float(grid.x) / 32768.0f));
        grid.x = 32768;
    }

    dim3 threads3D(256, 1, 1);
    dim3 grid3D((unsigned int) (ceil(float(gridSize.x) / float(threads3D.x))), 
                (unsigned int) (ceil(float(gridSize.y) / float(threads3D.y))), 
                (unsigned int) (ceil(float(gridSize.z) / float(threads3D.z))));

    // calculate number of vertices need per voxel
    if (cudacomputemajor >= 2) {
      // launch a 3-D grid if we have a new enough device (Fermi or later...)
      if (useSubGrid) {
        classifyVoxel<1,1><<<grid3D, threads3D>>>(d_voxelVerts, d_volume, 
                             gridSize, numVoxels, voxelSize, 
                             subGridStart, subGridEnd, isoValue);
      } else {
        classifyVoxel<1,0><<<grid3D, threads3D>>>(d_voxelVerts, d_volume, 
                             gridSize, numVoxels, voxelSize, 
                             subGridStart, subGridEnd, isoValue);
      }
    } else {
      // launch a 2-D grid for older devices
      if (useSubGrid) {
        classifyVoxel<0,1><<<grid, threads>>>(d_voxelVerts, d_volume, 
                             gridSize, numVoxels, voxelSize,
                             subGridStart, subGridEnd, isoValue);
      } else {
        classifyVoxel<0,0><<<grid, threads>>>(d_voxelVerts, d_volume, 
                             gridSize, numVoxels, voxelSize,
                             subGridStart, subGridEnd, isoValue);
      }
    }

    // scan voxel vertex/occupation array (use in-place prefix sum for lower memory consumption)
    uint2 lastElement, lastScanElement;
    cudaMemcpy((void *) &lastElement, (void *)(d_voxelVerts + numVoxels-1), sizeof(uint2), cudaMemcpyDeviceToHost);

    ThrustScanWrapperUint2(d_voxelVerts, d_voxelVerts, numVoxels);

    // read back values to calculate total number of non-empty voxels
    // since we are using an exclusive scan, the total is the last value of
    // the scan result plus the last value in the input array
    cudaMemcpy((void *) &lastScanElement, (void *) (d_voxelVerts + numVoxels-1), sizeof(uint2), cudaMemcpyDeviceToHost);
    activeVoxels = lastElement.y + lastScanElement.y;
    // add up total number of vertices
    totalVerts = lastElement.x + lastScanElement.x;
    totalVerts = totalVerts < maxverts ? totalVerts : maxverts; // min

    if (activeVoxels==0) {
      // return if there are no full voxels
      totalVerts = 0;
      return;
    }

    grid.x = (unsigned int) (ceil(float(numVoxels) / float(threads)));
    grid.y = grid.z = 1;
    // get around maximum grid size of 65535 in each dimension
    if (grid.x > 65535) {
        grid.y = (unsigned int) (ceil(float(grid.x) / 32768.0f));
        grid.x = 32768;
    }

    // compact voxel index array
    compactVoxels<<<grid, threads>>>(d_compVoxelArray, d_voxelVerts, lastElement.y, numVoxels, numVoxels - 1);

    dim3 grid2((unsigned int) (ceil(float(activeVoxels) / (float) NTHREADS)), 1, 1);
    while(grid2.x > 65535) {
        grid2.x = (unsigned int) (ceil(float(grid2.x) / 2.0f));
        grid2.y *= 2;
    }

    grid3 = dim3((unsigned int) (ceil(float(totalVerts) / (float) threads)), 1, 1);
    while(grid3.x > 65535) {
        grid3.x = (unsigned int) (ceil(float(grid3.x) / 2.0f));
        grid3.y *= 2;
    }
    while(grid3.y > 65535) {
        grid3.y = (unsigned int) (ceil(float(grid3.y) / 2.0f));
        grid3.z *= 2;
    }

    // separate computation of vertices and vertex color/normal for higher occupancy and speed
    generateTriangleVerticesSMEM<<<grid2, NTHREADS>>>(vertOut, 
        d_compVoxelArray, d_voxelVerts, d_volume, gridSize, voxelSize, 
        isoValue, activeVoxels, maxverts - 3);
}


//
// Generate colors using float3 vertex array format
//
void CUDAMarchingCubes::computeIsosurface(float3* vertOut, float3* normOut, float3* colOut, unsigned int maxverts) {
    // check if data is available
    if (!this->setdata)
      return;

    int threads = 256;
    dim3 grid3;
    computeIsosurfaceVerts(vertOut, maxverts, grid3);

    float3 gridSizeInv = make_float3(1.0f / float(gridSize.x), 1.0f / float(gridSize.y), 1.0f / float(gridSize.z));
    float3 bBoxInv = make_float3(1.0f / bBox.x, 1.0f / bBox.y, 1.0f / bBox.z);
    if (this->useColor) {
      switch (texformat) {
        case RGB3F:
          generateTriangleColorNormal<float3, float3, float3><<<grid3, threads>>>(vertOut, colOut, normOut, (float3 *) d_colors, this->gridSize, gridSizeInv, bBoxInv, totalVerts);
          break;

        case RGB4U:
          generateTriangleColorNormal<float3, uchar4, float3><<<grid3, threads>>>(vertOut, colOut, normOut, (uchar4 *) d_colors, this->gridSize, gridSizeInv, bBoxInv, totalVerts);
          break;
      }
    } else {
      generateTriangleNormals<float3><<<grid3, threads>>>(vertOut, normOut, gridSizeInv, bBoxInv, totalVerts);
    }

    offsetTriangleVertices<<<grid3, threads>>>(vertOut, this->origin, totalVerts - 1);
}


//
// Generate colors using uchar4 vertex array format
//
void CUDAMarchingCubes::computeIsosurface(float3* vertOut, float3* normOut, uchar4* colOut, unsigned int maxverts) {
    // check if data is available
    if (!this->setdata)
      return;

    int threads = 256;
    dim3 grid3;
    computeIsosurfaceVerts(vertOut, maxverts, grid3);

    float3 gridSizeInv = make_float3(1.0f / float(gridSize.x), 1.0f / float(gridSize.y), 1.0f / float(gridSize.z));
    float3 bBoxInv = make_float3(1.0f / bBox.x, 1.0f / bBox.y, 1.0f / bBox.z);
    if (this->useColor) {
      switch (texformat) {
        case RGB3F:
          generateTriangleColorNormal<uchar4, float3, float3><<<grid3, threads>>>(vertOut, colOut, normOut, (float3 *) d_colors, this->gridSize, gridSizeInv, bBoxInv, totalVerts);
          break;

        case RGB4U:
          generateTriangleColorNormal<uchar4, uchar4, float3><<<grid3, threads>>>(vertOut, colOut, normOut, (uchar4 *) d_colors, this->gridSize, gridSizeInv, bBoxInv, totalVerts);
          break;
      }
    } else {
      generateTriangleNormals<float3><<<grid3, threads>>>(vertOut, normOut, gridSizeInv, bBoxInv, totalVerts);
    }

    offsetTriangleVertices<<<grid3, threads>>>(vertOut, this->origin, totalVerts - 1);
}


//
// Generate normals w/ char format, colors w/ uchar4 vertex array format
//
void CUDAMarchingCubes::computeIsosurface(float3* vertOut, char3* normOut, uchar4* colOut, unsigned int maxverts) {
    // check if data is available
    if (!this->setdata)
      return;

    int threads = 256;
    dim3 grid3;
    computeIsosurfaceVerts(vertOut, maxverts, grid3);

    float3 gridSizeInv = make_float3(1.0f / float(gridSize.x), 1.0f / float(gridSize.y), 1.0f / float(gridSize.z));
    float3 bBoxInv = make_float3(1.0f / bBox.x, 1.0f / bBox.y, 1.0f / bBox.z);
    if (this->useColor) {
      switch (texformat) {
        case RGB3F:
          generateTriangleColorNormal<uchar4, float3, char3><<<grid3, threads>>>(vertOut, colOut, normOut, (float3 *) d_colors, this->gridSize, gridSizeInv, bBoxInv, totalVerts);
          break;

        case RGB4U:
          generateTriangleColorNormal<uchar4, uchar4, char3><<<grid3, threads>>>(vertOut, colOut, normOut, (uchar4 *) d_colors, this->gridSize, gridSizeInv, bBoxInv, totalVerts);
          break;
      }
    } else {
      generateTriangleNormals<char3><<<grid3, threads>>>(vertOut, normOut, gridSizeInv, bBoxInv, totalVerts);
    }

    offsetTriangleVertices<<<grid3, threads>>>(vertOut, this->origin, totalVerts - 1);
}


bool CUDAMarchingCubes::Initialize(uint3 maxgridsize) {
    // check if already initialized
    if (initialized) return false;

    // use max grid size initially
    maxGridSize = maxgridsize;
    gridSize = maxGridSize;
    maxNumVoxels = gridSize.x*gridSize.y*gridSize.z;
    numVoxels = maxNumVoxels;
    
    // initialize subgrid dimensions to the entire volume by default
    subGridStart = make_uint3(0, 0, 0);
    subGridEnd = gridSize - make_uint3(1, 1, 1);

    // allocate textures
    allocateTextures(&d_triTable, &d_numVertsTable);

    // allocate device memory
    if (cudaMalloc((void**) &d_voxelVerts, sizeof(uint2) * numVoxels) != cudaSuccess) {
        Cleanup();
        return false;
    }
    if (cudaMalloc((void**) &d_compVoxelArray, sizeof(unsigned int) * numVoxels) != cudaSuccess) {
        Cleanup();
        return false;
    }

    // success
    initialized = true;
    return true;
}


bool CUDAMarchingCubes::SetVolumeData(float *volume, void *colors, 
                                      VolTexFormat vtexformat, uint3 gridsize, 
                                      float3 gridOrigin, float3 boundingBox, 
                                      bool cudaArray) {
  bool newgridsize = false;

  // check if initialize was successful
  if (!initialized) return false;

  // check if the grid size matches
  if (gridsize.x != gridSize.x ||
      gridsize.y != gridSize.y ||
      gridsize.z != gridSize.z) {
    newgridsize = true;
    int nv = gridsize.x*gridsize.y*gridsize.z;
    if (nv > maxNumVoxels)
      return false;

    gridSize = gridsize;
    numVoxels = nv;

    // initialize subgrid dimensions to the entire volume by default
    subGridStart = make_uint3(0, 0, 0);
    subGridEnd = gridSize - make_uint3(1, 1, 1);
  }

  // copy the grid origin, bounding box dimensions, 
  // and update dependent variables
  origin = gridOrigin;
  bBox = boundingBox;
  voxelSize = make_float3(bBox.x / gridSize.x,
                          bBox.y / gridSize.y,
                          bBox.z / gridSize.z);

  // check colors
  useColor = colors ? true : false;

  // copy cuda array pointers or create cuda arrays and copy data
  if (cudaArray) {
    // check ownership flag and free if necessary
    if (ownCudaDataArrays) {
      if (d_volume) cudaFree(d_volume);
      d_volume = NULL;

      if (d_colors) cudaFree(d_colors);
      d_colors = NULL;
    }

    // copy data pointers
    d_volume = volume;
    d_colors = colors;
    texformat = vtexformat;

    // set ownership flag
    ownCudaDataArrays = false;
  } else {
    // create the volume array (using max size) and copy data
    unsigned int size = numVoxels * sizeof(float);
    unsigned int maxsize = maxNumVoxels * sizeof(float);

    // check data array allocation
    if (!d_volume) {
      if (cudaMalloc((void**) &d_volume, maxsize) != cudaSuccess) {
        Cleanup();
        return false;
      }
      if (cudaMemcpy(d_volume, volume, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        Cleanup();
        return false;
      }
    }

    if (colors) {
      if (!d_colors) {
        // create the color array and copy colors
        unsigned int maxtxsize = 0; 
        unsigned int txsize = 0;
        texformat = vtexformat;
        switch (texformat) {
          case RGB3F: 
            txsize = numVoxels * sizeof(float3);
            maxtxsize = maxNumVoxels * sizeof(float3);
            break;

          case RGB4U: 
            txsize = numVoxels * sizeof(uchar4);
            maxtxsize = maxNumVoxels * sizeof(uchar4);
            break;
        } 

        if (cudaMalloc((void**) &d_colors, maxtxsize) != cudaSuccess) {
          Cleanup();
          return false;
        }
        if (cudaMemcpy(d_colors, colors, txsize, cudaMemcpyHostToDevice) != cudaSuccess) {
          Cleanup();
          return false;
        }
      }
    }

    // set ownership flag
    ownCudaDataArrays = true;
  }

  // Compute grid extents and channel description for the 3-D array
  cudaExtent gridExtent = make_cudaExtent(gridSize.x, gridSize.y, gridSize.z);
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

  // Check to see if existing 3D array allocation matches current grid size,
  // deallocate it if not so that we regenerate it with the correct size.
  if (d_volumeArray && newgridsize) { 
    cudaFreeArray(d_volumeArray);
    d_volumeArray = NULL;
  }

  // allocate the 3D array if needed
  if (!d_volumeArray) { 
    // create 3D array
    if (cudaMalloc3DArray(&d_volumeArray, &channelDesc, gridExtent) != cudaSuccess) {
      Cleanup();
      return false;
    }
  }

  // copy data to 3D array
  cudaMemcpy3DParms copyParams = {0};
  copyParams.srcPtr   = make_cudaPitchedPtr((void*)d_volume, 
                                            gridExtent.width*sizeof(float),
                                            gridExtent.width,
                                            gridExtent.height);
  copyParams.dstArray = d_volumeArray;
  copyParams.extent   = gridExtent;
  copyParams.kind     = cudaMemcpyDeviceToDevice;
  if (cudaMemcpy3D(&copyParams) != cudaSuccess) {
    Cleanup();
    return false;
  }

  // bind the array to a volume texture
  bindVolumeTexture(d_volumeArray, channelDesc);

  // success
  setdata = true;

  return true;
}


bool CUDAMarchingCubes::SetVolumeData(float *volume, float3 *colors, uint3 gridsize, float3 gridOrigin, float3 boundingBox, bool cudaArray) {
  return SetVolumeData(volume, colors, RGB3F, gridsize, gridOrigin, 
                       boundingBox, cudaArray);
}


bool CUDAMarchingCubes::SetVolumeData(float *volume, uchar4 *colors, uint3 gridsize, float3 gridOrigin, float3 boundingBox, bool cudaArray) {
  return SetVolumeData(volume, colors, RGB4U, gridsize, gridOrigin, 
                       boundingBox, cudaArray);
}



void CUDAMarchingCubes::SetSubVolume(uint3 start, uint3 end) {
  subGridStart = start;
  subGridEnd = end;
  useSubGrid = true;

  if (subGridEnd.x >= gridSize.x)
    subGridEnd.x = gridSize.x - 1;
  if (subGridEnd.y >= gridSize.y)
    subGridEnd.y = gridSize.y - 1;
  if (subGridEnd.z >= gridSize.z)
    subGridEnd.z = gridSize.z - 1;
}


bool CUDAMarchingCubes::computeIsosurface(float *volume, void *colors, 
                                          VolTexFormat vtexformat,
                                          uint3 gridsize, float3 gridOrigin, 
                                          float3 boundingBox, bool cudaArray, 
                                          float3* vertOut, float3* normOut, 
                                          float3* colOut, unsigned int maxverts) {
    // Setup
    if (!Initialize(gridsize))
        return false;

    if (!SetVolumeData(volume, colors, vtexformat, gridsize, gridOrigin, boundingBox, cudaArray))
        return false;

    // Compute and Render Isosurface
    computeIsosurface(vertOut, normOut, colOut, maxverts);

    // Tear down and free resources
    Cleanup();

    return true;
}


float CUDAMarchingCubes::computeSurfaceArea(float3 *verts, unsigned int triaCount) {
    // do nothing for zero triangles
    if(triaCount <= 0) return 0.0f;

    // initialize and allocate device arrays
    float *d_areas;
    unsigned long memSize = sizeof(float) * triaCount;
    if (cudaMalloc((void**) &d_areas, memSize) != cudaSuccess) {
        return -1.0f;
    }

    // compute area for each triangle
    const int threads = 256;
    dim3 grid(int(ceil(float(triaCount) / float(threads))), 1, 1);

    // get around maximum grid size of 65535 in each dimension
    while(grid.x > 65535) {
        grid.x = (unsigned int) (ceil(float(grid.x) / 2.0f));
        grid.y *= 2;
    }
    while(grid.y > 65535) {
        grid.y = (unsigned int) (ceil(float(grid.y) / 2.0f));
        grid.z *= 2;
    }

    computeTriangleAreas<<<grid, threads>>>(verts, d_areas, triaCount);

    // use prefix sum to compute total surface area
    ThrustScanWrapperArea(d_areas, d_areas, triaCount);

    // readback total surface area
    float totalArea;
    cudaMemcpy((void *)&totalArea, (void *)(d_areas + triaCount - 1), sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_areas); // clean up
    return totalArea;  // return result
}



