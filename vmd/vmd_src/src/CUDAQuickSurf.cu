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
 *      $RCSfile: CUDAQuickSurf.cu,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.86 $      $Date: 2019/01/17 21:38:55 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   CUDA accelerated gaussian density calculation
 *
 ***************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#if CUDART_VERSION >= 9000
#include <cuda_fp16.h>  // need to explicitly include for CUDA 9.0
#endif

#if CUDART_VERSION < 4000
#error The VMD QuickSurf feature requires CUDA 4.0 or later
#endif

#include "Inform.h"
#include "utilities.h"
#include "WKFThreads.h"
#include "WKFUtils.h"
#include "CUDAKernels.h" 
#include "CUDASpatialSearch.h"
#include "CUDAMarchingCubes.h"
#include "CUDAQuickSurf.h" 

#include "DispCmds.h"
#include "VMDDisplayList.h"

#if 1
#define CUERR { cudaError_t err; \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
  printf("CUDA error: %s, %s line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
  printf("Thread aborting...\n"); \
  return NULL; }}
#else
#define CUERR
#endif


//
// density format conversion routines
//

// no-op conversion for float to float
inline __device__ void convert_density(float & df, float df2) {
  df = df2;
}

// Convert float (32-bit) to half-precision (16-bit floating point) stored
// into an unsigned short (16-bit integer type). 
inline __device__ void convert_density(unsigned short & dh, float df2) {
  dh = __float2half_rn(df2);
}



//
// color format conversion routines
//

// No-op conversion for float3 to float3
inline __device__ void convert_color(float3 & cf, float3 cf2) {
  cf = cf2;
}

// Convert float3 colors to uchar4, performing the necessary bias, scaling, 
// and range clamping so we don't encounter integer wraparound, etc.
inline __device__ void convert_color(uchar4 & cu, float3 cf) {
  // conversion to GLubyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
  // c = f * (2^8-1)

  // scale color values to prevent overflow, 
  // and convert to fixed-point representation all at once
  float invmaxcolscale = __frcp_rn(fmaxf(fmaxf(fmaxf(cf.x, cf.y), cf.z), 1.0f)) * 255.0f;

  // clamp color values to prevent integer wraparound
  cu = make_uchar4(cf.x * invmaxcolscale,
                   cf.y * invmaxcolscale,
                   cf.z * invmaxcolscale,
                   255);
}

// convert uchar4 colors to float3
inline __device__ void convert_color(float3 & cf, uchar4 cu) {
  const float i2f = 1.0f / 255.0f;
  cf.x = cu.x * i2f;
  cf.y = cu.y * i2f;
  cf.z = cu.z * i2f;
}


//
// Restrict macro to make it easy to do perf tuning tests
//
#if 0
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif

// 
// Parameters for linear-time range-limited gaussian density kernels
//
#define GGRIDSZ   8.0f
#define GBLOCKSZX 8
#define GBLOCKSZY 8

#if 1
#define GTEXBLOCKSZZ 2
#define GTEXUNROLL   4
#define GBLOCKSZZ    2
#define GUNROLL      4
#else
#define GTEXBLOCKSZZ 8
#define GTEXUNROLL   1
#define GBLOCKSZZ    8
#define GUNROLL      1
#endif

#define MAXTHRDENS  ( GBLOCKSZX * GBLOCKSZY * GBLOCKSZZ )
#if __CUDA_ARCH__ >= 600
#define MINBLOCKDENS 16
#elif __CUDA_ARCH__ >= 300
#define MINBLOCKDENS 16
#elif __CUDA_ARCH__ >= 200
#define MINBLOCKDENS 1
#else
#define MINBLOCKDENS 1
#endif


//
// Templated version of the density map kernel to handle multiple 
// data formats for the output density volume and volumetric texture.
// This variant of the density map algorithm normalizes densities so
// that the target isovalue is a density of 1.0.
//
template<class DENSITY, class VOLTEX>
__global__ static void 
__launch_bounds__ ( MAXTHRDENS, MINBLOCKDENS )
gaussdensity_fast_tex_norm(int natoms,
                      const float4 * RESTRICT sorted_xyzr, 
                      const float4 * RESTRICT sorted_color, 
                      int3 volsz,
                      int3 acncells,
                      float acgridspacing,
                      float invacgridspacing,
                      const uint2 * RESTRICT cellStartEnd,
                      float gridspacing, unsigned int z, 
                      DENSITY * RESTRICT densitygrid,
                      VOLTEX * RESTRICT voltexmap,
                      float invisovalue) {
  unsigned int xindex  = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned int yindex  = (blockIdx.y * blockDim.y) + threadIdx.y;
  unsigned int zindex  = ((blockIdx.z * blockDim.z) + threadIdx.z) * GTEXUNROLL;

  // shave register use slightly
  unsigned int outaddr = zindex * volsz.x * volsz.y + 
                         yindex * volsz.x + xindex;

  // early exit if this thread is outside of the grid bounds
  if (xindex >= volsz.x || yindex >= volsz.y || zindex >= volsz.z)
    return;

  zindex += z;

  // compute ac grid index of lower corner minus gaussian radius
  int xabmin = ((blockIdx.x * blockDim.x) * gridspacing - acgridspacing) * invacgridspacing;
  int yabmin = ((blockIdx.y * blockDim.y) * gridspacing - acgridspacing) * invacgridspacing;
  int zabmin = ((z + blockIdx.z * blockDim.z * GTEXUNROLL) * gridspacing - acgridspacing) * invacgridspacing;

  // compute ac grid index of upper corner plus gaussian radius
  int xabmax = (((blockIdx.x+1) * blockDim.x) * gridspacing + acgridspacing) * invacgridspacing;
  int yabmax = (((blockIdx.y+1) * blockDim.y) * gridspacing + acgridspacing) * invacgridspacing;
  int zabmax = ((z + (blockIdx.z+1) * blockDim.z * GTEXUNROLL) * gridspacing + acgridspacing) * invacgridspacing;

  xabmin = (xabmin < 0) ? 0 : xabmin;
  yabmin = (yabmin < 0) ? 0 : yabmin;
  zabmin = (zabmin < 0) ? 0 : zabmin;
  xabmax = (xabmax >= acncells.x-1) ? acncells.x-1 : xabmax;
  yabmax = (yabmax >= acncells.y-1) ? acncells.y-1 : yabmax;
  zabmax = (zabmax >= acncells.z-1) ? acncells.z-1 : zabmax;

  float coorx = gridspacing * xindex;
  float coory = gridspacing * yindex;
  float coorz = gridspacing * zindex;

  float densityval1=0.0f;
  float3 densitycol1=make_float3(0.0f, 0.0f, 0.0f);
#if GTEXUNROLL >= 2
  float densityval2=0.0f;
  float3 densitycol2=densitycol1;
#endif
#if GTEXUNROLL >= 4
  float densityval3=0.0f;
  float3 densitycol3=densitycol1;
  float densityval4=0.0f;
  float3 densitycol4=densitycol1;
#endif

  int acplanesz = acncells.x * acncells.y;
  int xab, yab, zab;
  for (zab=zabmin; zab<=zabmax; zab++) {
    for (yab=yabmin; yab<=yabmax; yab++) {
      for (xab=xabmin; xab<=xabmax; xab++) {
        int abcellidx = zab * acplanesz + yab * acncells.x + xab;
        // this biggest latency hotspot in the kernel, if we could improve
        // packing of the grid cell map, we'd likely improve performance 
        uint2 atomstartend = cellStartEnd[abcellidx];
        if (atomstartend.x != GRID_CELL_EMPTY) {
          unsigned int atomid;
          for (atomid=atomstartend.x; atomid<atomstartend.y; atomid++) {
            float4 atom  = sorted_xyzr[atomid];
            float4 color = sorted_color[atomid];
            float dx = coorx - atom.x;
            float dy = coory - atom.y;
            float dxy2 = dx*dx + dy*dy;

            float dz = coorz - atom.z;
            float r21 = (dxy2 + dz*dz) * atom.w;
            float tmp1 = invisovalue * exp2f(r21); // normalized density
            densityval1 += tmp1;
            densitycol1.x += tmp1 * color.x;
            densitycol1.y += tmp1 * color.y;
            densitycol1.z += tmp1 * color.z;

#if GTEXUNROLL >= 2
            float dz2 = dz + gridspacing;
            float r22 = (dxy2 + dz2*dz2) * atom.w;
            float tmp2 = invisovalue * exp2f(r22); // normalized density
            densityval2 += tmp2;
            densitycol2.x += tmp2 * color.x;
            densitycol2.y += tmp2 * color.y;
            densitycol2.z += tmp2 * color.z;
#endif
#if GTEXUNROLL >= 4
            float dz3 = dz2 + gridspacing;
            float r23 = (dxy2 + dz3*dz3) * atom.w;
            float tmp3 = invisovalue * exp2f(r23); // normalized density
            densityval3 += tmp3;
            densitycol3.x += tmp3 * color.x;
            densitycol3.y += tmp3 * color.y;
            densitycol3.z += tmp3 * color.z;

            float dz4 = dz3 + gridspacing;
            float r24 = (dxy2 + dz4*dz4) * atom.w;
            float tmp4 = invisovalue * exp2f(r24); // normalized density
            densityval4 += tmp4;
            densitycol4.x += tmp4 * color.x;
            densitycol4.y += tmp4 * color.y;
            densitycol4.z += tmp4 * color.z;
#endif
          }
        }
      }
    }
  }

  DENSITY densityout;
  VOLTEX texout;
  convert_density(densityout, densityval1);
  densitygrid[outaddr          ] = densityout;
  convert_color(texout, densitycol1);
  voltexmap[outaddr          ] = texout;

#if GTEXUNROLL >= 2
  int planesz = volsz.x * volsz.y;
  convert_density(densityout, densityval2);
  densitygrid[outaddr + planesz] = densityout;
  convert_color(texout, densitycol2);
  voltexmap[outaddr + planesz] = texout;
#endif
#if GTEXUNROLL >= 4
  convert_density(densityout, densityval3);
  densitygrid[outaddr + 2*planesz] = densityout;
  convert_color(texout, densitycol3);
  voltexmap[outaddr + 2*planesz] = texout;

  convert_density(densityout, densityval4);
  densitygrid[outaddr + 3*planesz] = densityout;
  convert_color(texout, densitycol4);
  voltexmap[outaddr + 3*planesz] = texout;
#endif
}


__global__ static void 
__launch_bounds__ ( MAXTHRDENS, MINBLOCKDENS )
gaussdensity_fast_tex3f(int natoms,
                        const float4 * RESTRICT sorted_xyzr, 
                        const float4 * RESTRICT sorted_color, 
                        int3 volsz,
                        int3 acncells,
                        float acgridspacing,
                        float invacgridspacing,
                        const uint2 * RESTRICT cellStartEnd,
                        float gridspacing, unsigned int z, 
                        float * RESTRICT densitygrid,
                        float3 * RESTRICT voltexmap,
                        float invisovalue) {
  unsigned int xindex  = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned int yindex  = (blockIdx.y * blockDim.y) + threadIdx.y;
  unsigned int zindex  = ((blockIdx.z * blockDim.z) + threadIdx.z) * GTEXUNROLL;

  // shave register use slightly
  unsigned int outaddr = zindex * volsz.x * volsz.y + 
                         yindex * volsz.x + xindex;

  // early exit if this thread is outside of the grid bounds
  if (xindex >= volsz.x || yindex >= volsz.y || zindex >= volsz.z)
    return;

  zindex += z;

  // compute ac grid index of lower corner minus gaussian radius
  int xabmin = ((blockIdx.x * blockDim.x) * gridspacing - acgridspacing) * invacgridspacing;
  int yabmin = ((blockIdx.y * blockDim.y) * gridspacing - acgridspacing) * invacgridspacing;
  int zabmin = ((z + blockIdx.z * blockDim.z * GTEXUNROLL) * gridspacing - acgridspacing) * invacgridspacing;

  // compute ac grid index of upper corner plus gaussian radius
  int xabmax = (((blockIdx.x+1) * blockDim.x) * gridspacing + acgridspacing) * invacgridspacing;
  int yabmax = (((blockIdx.y+1) * blockDim.y) * gridspacing + acgridspacing) * invacgridspacing;
  int zabmax = ((z + (blockIdx.z+1) * blockDim.z * GTEXUNROLL) * gridspacing + acgridspacing) * invacgridspacing;

  xabmin = (xabmin < 0) ? 0 : xabmin;
  yabmin = (yabmin < 0) ? 0 : yabmin;
  zabmin = (zabmin < 0) ? 0 : zabmin;
  xabmax = (xabmax >= acncells.x-1) ? acncells.x-1 : xabmax;
  yabmax = (yabmax >= acncells.y-1) ? acncells.y-1 : yabmax;
  zabmax = (zabmax >= acncells.z-1) ? acncells.z-1 : zabmax;

  float coorx = gridspacing * xindex;
  float coory = gridspacing * yindex;
  float coorz = gridspacing * zindex;

  float densityval1=0.0f;
  float3 densitycol1=make_float3(0.0f, 0.0f, 0.0f);
#if GTEXUNROLL >= 2
  float densityval2=0.0f;
  float3 densitycol2=densitycol1;
#endif
#if GTEXUNROLL >= 4
  float densityval3=0.0f;
  float3 densitycol3=densitycol1;
  float densityval4=0.0f;
  float3 densitycol4=densitycol1;
#endif

  int acplanesz = acncells.x * acncells.y;
  int xab, yab, zab;
  for (zab=zabmin; zab<=zabmax; zab++) {
    for (yab=yabmin; yab<=yabmax; yab++) {
      for (xab=xabmin; xab<=xabmax; xab++) {
        int abcellidx = zab * acplanesz + yab * acncells.x + xab;
        // this biggest latency hotspot in the kernel, if we could improve
        // packing of the grid cell map, we'd likely improve performance 
        uint2 atomstartend = cellStartEnd[abcellidx];
        if (atomstartend.x != GRID_CELL_EMPTY) {
          unsigned int atomid;
          for (atomid=atomstartend.x; atomid<atomstartend.y; atomid++) {
            float4 atom  = sorted_xyzr[atomid];
            float4 color = sorted_color[atomid];
            float dx = coorx - atom.x;
            float dy = coory - atom.y;
            float dxy2 = dx*dx + dy*dy;

            float dz = coorz - atom.z;
            float r21 = (dxy2 + dz*dz) * atom.w;
            float tmp1 = exp2f(r21);
            densityval1 += tmp1;
            tmp1 *= invisovalue;
            densitycol1.x += tmp1 * color.x;
            densitycol1.y += tmp1 * color.y;
            densitycol1.z += tmp1 * color.z;

#if GTEXUNROLL >= 2
            float dz2 = dz + gridspacing;
            float r22 = (dxy2 + dz2*dz2) * atom.w;
            float tmp2 = exp2f(r22);
            densityval2 += tmp2;
            tmp2 *= invisovalue;
            densitycol2.x += tmp2 * color.x;
            densitycol2.y += tmp2 * color.y;
            densitycol2.z += tmp2 * color.z;
#endif
#if GTEXUNROLL >= 4
            float dz3 = dz2 + gridspacing;
            float r23 = (dxy2 + dz3*dz3) * atom.w;
            float tmp3 = exp2f(r23);
            densityval3 += tmp3;
            tmp3 *= invisovalue;
            densitycol3.x += tmp3 * color.x;
            densitycol3.y += tmp3 * color.y;
            densitycol3.z += tmp3 * color.z;

            float dz4 = dz3 + gridspacing;
            float r24 = (dxy2 + dz4*dz4) * atom.w;
            float tmp4 = exp2f(r24);
            densityval4 += tmp4;
            tmp4 *= invisovalue;
            densitycol4.x += tmp4 * color.x;
            densitycol4.y += tmp4 * color.y;
            densitycol4.z += tmp4 * color.z;
#endif
          }
        }
      }
    }
  }

  densitygrid[outaddr          ] = densityval1;
  voltexmap[outaddr          ].x = densitycol1.x;
  voltexmap[outaddr          ].y = densitycol1.y;
  voltexmap[outaddr          ].z = densitycol1.z;

#if GTEXUNROLL >= 2
  int planesz = volsz.x * volsz.y;
  densitygrid[outaddr + planesz] = densityval2;
  voltexmap[outaddr + planesz].x = densitycol2.x;
  voltexmap[outaddr + planesz].y = densitycol2.y;
  voltexmap[outaddr + planesz].z = densitycol2.z;
#endif
#if GTEXUNROLL >= 4
  densitygrid[outaddr + 2*planesz] = densityval3;
  voltexmap[outaddr + 2*planesz].x = densitycol3.x;
  voltexmap[outaddr + 2*planesz].y = densitycol3.y;
  voltexmap[outaddr + 2*planesz].z = densitycol3.z;

  densitygrid[outaddr + 3*planesz] = densityval4;
  voltexmap[outaddr + 3*planesz].x = densitycol4.x;
  voltexmap[outaddr + 3*planesz].y = densitycol4.y;
  voltexmap[outaddr + 3*planesz].z = densitycol4.z;
#endif
}


__global__ static void 
// __launch_bounds__ ( MAXTHRDENS, MINBLOCKDENS )
gaussdensity_fast(int natoms,
                  const float4 * RESTRICT sorted_xyzr, 
                  int3 volsz,
                  int3 acncells,
                  float acgridspacing,
                  float invacgridspacing,
                  const uint2 * RESTRICT cellStartEnd,
                  float gridspacing, unsigned int z, 
                  float * RESTRICT densitygrid) {
  unsigned int xindex  = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned int yindex  = (blockIdx.y * blockDim.y) + threadIdx.y;
  unsigned int zindex  = ((blockIdx.z * blockDim.z) + threadIdx.z) * GUNROLL;
  unsigned int outaddr = zindex * volsz.x * volsz.y + 
                         yindex * volsz.x + 
                         xindex;

  // early exit if this thread is outside of the grid bounds
  if (xindex >= volsz.x || yindex >= volsz.y || zindex >= volsz.z)
    return;

  zindex += z;

  // compute ac grid index of lower corner minus gaussian radius
  int xabmin = ((blockIdx.x * blockDim.x) * gridspacing - acgridspacing) * invacgridspacing;
  int yabmin = ((blockIdx.y * blockDim.y) * gridspacing - acgridspacing) * invacgridspacing;
  int zabmin = ((z + blockIdx.z * blockDim.z * GUNROLL) * gridspacing - acgridspacing) * invacgridspacing;

  // compute ac grid index of upper corner plus gaussian radius
  int xabmax = (((blockIdx.x+1) * blockDim.x) * gridspacing + acgridspacing) * invacgridspacing;
  int yabmax = (((blockIdx.y+1) * blockDim.y) * gridspacing + acgridspacing) * invacgridspacing;
  int zabmax = ((z + (blockIdx.z+1) * blockDim.z * GUNROLL) * gridspacing + acgridspacing) * invacgridspacing;

  xabmin = (xabmin < 0) ? 0 : xabmin;
  yabmin = (yabmin < 0) ? 0 : yabmin;
  zabmin = (zabmin < 0) ? 0 : zabmin;
  xabmax = (xabmax >= acncells.x-1) ? acncells.x-1 : xabmax;
  yabmax = (yabmax >= acncells.y-1) ? acncells.y-1 : yabmax;
  zabmax = (zabmax >= acncells.z-1) ? acncells.z-1 : zabmax;

  float coorx = gridspacing * xindex;
  float coory = gridspacing * yindex;
  float coorz = gridspacing * zindex;

  float densityval1=0.0f;
#if GUNROLL >= 2
  float densityval2=0.0f;
#endif
#if GUNROLL >= 4
  float densityval3=0.0f;
  float densityval4=0.0f;
#endif

  int acplanesz = acncells.x * acncells.y;
  int xab, yab, zab;
  for (zab=zabmin; zab<=zabmax; zab++) {
    for (yab=yabmin; yab<=yabmax; yab++) {
      for (xab=xabmin; xab<=xabmax; xab++) {
        int abcellidx = zab * acplanesz + yab * acncells.x + xab;
        uint2 atomstartend = cellStartEnd[abcellidx];
        if (atomstartend.x != GRID_CELL_EMPTY) {
          unsigned int atomid;
          for (atomid=atomstartend.x; atomid<atomstartend.y; atomid++) {
            float4 atom = sorted_xyzr[atomid];
            float dx = coorx - atom.x;
            float dy = coory - atom.y;
            float dxy2 = dx*dx + dy*dy;
  
            float dz = coorz - atom.z;
            float r21 = (dxy2 + dz*dz) * atom.w;
            densityval1 += exp2f(r21);

#if GUNROLL >= 2
            float dz2 = dz + gridspacing;
            float r22 = (dxy2 + dz2*dz2) * atom.w;
            densityval2 += exp2f(r22);
#endif
#if GUNROLL >= 4
            float dz3 = dz2 + gridspacing;
            float r23 = (dxy2 + dz3*dz3) * atom.w;
            densityval3 += exp2f(r23);

            float dz4 = dz3 + gridspacing;
            float r24 = (dxy2 + dz4*dz4) * atom.w;
            densityval4 += exp2f(r24);
#endif
          }
        }
      }
    }
  }

  densitygrid[outaddr            ] = densityval1;
#if GUNROLL >= 2
  int planesz = volsz.x * volsz.y;
  densitygrid[outaddr +   planesz] = densityval2;
#endif
#if GUNROLL >= 4
  densitygrid[outaddr + 2*planesz] = densityval3;
  densitygrid[outaddr + 3*planesz] = densityval4;
#endif
}


// per-GPU handle with various memory buffer pointers, etc.
typedef struct {
  cudaDeviceProp deviceProp; ///< GPU hw properties
  int dev;                   ///< CUDA device ID
  int verbose;               ///< emit verbose debugging/timing info

  /// max grid sizes and attributes the current allocations will support
  long int natoms;           ///< total atom count
  int colorperatom;          ///< flag indicating color-per-atom array
  int acx;                   ///< accel grid X dimension
  int acy;                   ///< accel grid Y dimension
  int acz;                   ///< accel grid Z dimension
  int gx;                    ///< density grid X dimension
  int gy;                    ///< density grid Y dimension
  int gz;                    ///< density grid Z dimension

  CUDAMarchingCubes *mc;     ///< Marching cubes class used to extract surface

  float *devdensity;         ///< density map stored in GPU memory
  void *devvoltexmap;        ///< volumetric texture map
  float4 *xyzr_d;            ///< atom coords and radii
  float4 *sorted_xyzr_d;     ///< cell-sorted coords and radii
  float4 *color_d;           ///< colors
  float4 *sorted_color_d;    ///< cell-sorted colors

  unsigned int *atomIndex_d; ///< cell index for each atom
  unsigned int *atomHash_d;  ///<  
  uint2 *cellStartEnd_d;     ///< cell start/end indices 

  void *safety;              ///< Thrust/CUB sort/scan workspace allocation

  float3 *v3f_d;             ///< device vertex array allocation
  float3 *n3f_d;             ///< device fp32 normal array allocation
  float3 *c3f_d;             ///< device fp32 color array allocation
  char3 *n3b_d;              ///< device 8-bit char normal array allocation
  uchar4 *c4u_d;             ///< device 8-bit uchar color array allocation
} qsurf_gpuhandle;


CUDAQuickSurf::CUDAQuickSurf() {
  voidgpu = calloc(1, sizeof(qsurf_gpuhandle));
  qsurf_gpuhandle *gpuh = (qsurf_gpuhandle *) voidgpu;

  if (getenv("VMDQUICKSURFVERBOSE") != NULL) {
    gpuh->verbose = 1;
    int tmp = atoi(getenv("VMDQUICKSURFVERBOSE"));
    if (tmp > 0)
      gpuh->verbose = tmp;
  }

  if (cudaGetDevice(&gpuh->dev) != cudaSuccess) {
    gpuh->dev = -1; // flag GPU as unusable
  }

  if (cudaGetDeviceProperties(&gpuh->deviceProp, gpuh->dev) != cudaSuccess) {
    cudaError_t err = cudaGetLastError(); // eat error so next CUDA op succeeds
    gpuh->dev = -1; // flag GPU as unusable
  }
}


CUDAQuickSurf::~CUDAQuickSurf() {
  qsurf_gpuhandle *gpuh = (qsurf_gpuhandle *) voidgpu;

  // free all working buffers if not done already
  free_bufs();

  // delete marching cubes object
  delete gpuh->mc;

  free(voidgpu);
}


int CUDAQuickSurf::free_bufs() {
  qsurf_gpuhandle *gpuh = (qsurf_gpuhandle *) voidgpu;

  // zero out max buffer capacities
  gpuh->natoms = 0;
  gpuh->colorperatom = 0;
  gpuh->acx = 0;
  gpuh->acy = 0;
  gpuh->acz = 0;
  gpuh->gx = 0;
  gpuh->gy = 0;
  gpuh->gz = 0;

  if (gpuh->safety != NULL)
    cudaFree(gpuh->safety);
  gpuh->safety=NULL;

  if (gpuh->devdensity != NULL)
    cudaFree(gpuh->devdensity);
  gpuh->devdensity=NULL;

  if (gpuh->devvoltexmap != NULL)
    cudaFree(gpuh->devvoltexmap);
  gpuh->devvoltexmap=NULL;

  if (gpuh->xyzr_d != NULL)
    cudaFree(gpuh->xyzr_d);
  gpuh->xyzr_d=NULL;

  if (gpuh->sorted_xyzr_d != NULL)
    cudaFree(gpuh->sorted_xyzr_d);  
  gpuh->sorted_xyzr_d=NULL;

  if (gpuh->color_d != NULL)
    cudaFree(gpuh->color_d);
  gpuh->color_d=NULL;

  if (gpuh->sorted_color_d != NULL)
    cudaFree(gpuh->sorted_color_d);
  gpuh->sorted_color_d=NULL;

  if (gpuh->atomIndex_d != NULL)
    cudaFree(gpuh->atomIndex_d);
  gpuh->atomIndex_d=NULL;

  if (gpuh->atomHash_d != NULL)
    cudaFree(gpuh->atomHash_d);
  gpuh->atomHash_d=NULL;

  if (gpuh->cellStartEnd_d != NULL)
    cudaFree(gpuh->cellStartEnd_d);
  gpuh->cellStartEnd_d=NULL;

  if (gpuh->v3f_d != NULL)
    cudaFree(gpuh->v3f_d);
  gpuh->v3f_d=NULL;

  if (gpuh->n3f_d != NULL)
    cudaFree(gpuh->n3f_d);
  gpuh->n3f_d=NULL;

  if (gpuh->c3f_d != NULL)
    cudaFree(gpuh->c3f_d);
  gpuh->c3f_d=NULL;

  if (gpuh->n3b_d != NULL)
    cudaFree(gpuh->n3b_d);
  gpuh->n3b_d=NULL;

  if (gpuh->c4u_d != NULL)
    cudaFree(gpuh->c4u_d);
  gpuh->c4u_d=NULL;


  return 0;
}


int CUDAQuickSurf::check_bufs(long int natoms, int colorperatom, 
                              int acx, int acy, int acz,
                              int gx, int gy, int gz) {
  qsurf_gpuhandle *gpuh = (qsurf_gpuhandle *) voidgpu;

  // If the current atom count, texturing mode, and total voxel count
  // use the same or less storage than the size of the existing buffers,
  // we can reuse the same buffers without having to go through the 
  // complex allocation and validation loops.  This is a big performance
  // benefit during trajectory animation.
  if (natoms <= gpuh->natoms &&
      colorperatom <= gpuh->colorperatom &&
      (acx*acy*acz) <= (gpuh->acx * gpuh->acy * gpuh->acz) && 
      (gx*gy*gz) <= (gpuh->gx * gpuh->gy * gpuh->gz))
    return 0;
 
  return -1; // no existing bufs, or too small to be used 
}


int CUDAQuickSurf::alloc_bufs(long int natoms, int colorperatom, 
                              VolTexFormat vtexformat, 
                              int acx, int acy, int acz,
                              int gx, int gy, int gz) {
  qsurf_gpuhandle *gpuh = (qsurf_gpuhandle *) voidgpu;

  // early exit from allocation call if we've already got existing
  // buffers that are large enough to support the request
  if (check_bufs(natoms, colorperatom, acx, acy, acz, gx, gy, gz) == 0)
    return 0;

  // If we have any existing allocations, trash them as they weren't
  // usable for this new request and we need to reallocate them from scratch
  free_bufs();

  long int acncells = ((long) acx) * ((long) acy) * ((long) acz);
  long int ncells = ((long) gx) * ((long) gy) * ((long) gz);
  long int volmemsz = ncells * sizeof(float);
  long int chunkmaxverts = 3L * ncells; // assume worst case 50% triangle occupancy
  long int MCsz = CUDAMarchingCubes::MemUsageMC(gx, gy, gz);

  // Allocate all of the memory buffers our algorithms will need up-front,
  // so we can retry and gracefully reduce the sizes of various buffers
  // to attempt to fit within available GPU memory 
  long int totalmemsz = 
    volmemsz +                                       // volume
    (2L * natoms * sizeof(unsigned int)) +           // bin sort
    (acncells * sizeof(uint2)) +                     // bin sort
    (3L * chunkmaxverts * sizeof(float3)) +          // MC vertex bufs 
    natoms*sizeof(float4) +                          // thrust
    8L * gx * gy * sizeof(float) +                   // thrust
    MCsz;                                            // mcubes

  cudaMalloc((void**)&gpuh->devdensity, volmemsz);
  if (colorperatom) {
    int voltexsz = 0;
    switch (vtexformat) {
      case RGB3F:
        voltexsz = ncells * sizeof(float3);
        break;

      case RGB4U:
        voltexsz = ncells * sizeof(uchar4);
        break;
    }
    cudaMalloc((void**)&gpuh->devvoltexmap, voltexsz);
    cudaMalloc((void**)&gpuh->color_d, natoms * sizeof(float4));
    cudaMalloc((void**)&gpuh->sorted_color_d, natoms * sizeof(float4));
    totalmemsz += 2 * natoms * sizeof(float4);
  }
  cudaMalloc((void**)&gpuh->xyzr_d, natoms * sizeof(float4));
  cudaMalloc((void**)&gpuh->sorted_xyzr_d, natoms * sizeof(float4));
  cudaMalloc((void**)&gpuh->atomIndex_d, natoms * sizeof(unsigned int));
  cudaMalloc((void**)&gpuh->atomHash_d, natoms * sizeof(unsigned int));
  cudaMalloc((void**)&gpuh->cellStartEnd_d, acncells * sizeof(uint2));

  // allocate marching cubes output buffers
  cudaMalloc((void**)&gpuh->v3f_d, 3 * chunkmaxverts * sizeof(float3));
#if 1
  cudaMalloc((void**)&gpuh->n3b_d, 3 * chunkmaxverts * sizeof(char3));
  totalmemsz += 3 * chunkmaxverts * sizeof(char3);   // MC normal bufs 
#else
  cudaMalloc((void**)&gpuh->n3f_d, 3 * chunkmaxverts * sizeof(float3));
  totalmemsz += 3 * chunkmaxverts * sizeof(float3);  // MC normal bufs 
#endif
#if 1
  cudaMalloc((void**)&gpuh->c4u_d, 3 * chunkmaxverts * sizeof(uchar4));
  totalmemsz += 3 * chunkmaxverts * sizeof(uchar4);  // MC vertex color bufs 
#else
  cudaMalloc((void**)&gpuh->c3f_d, 3 * chunkmaxverts * sizeof(float3));
  totalmemsz += 3 * chunkmaxverts * sizeof(float3);  // MC vertex color bufs 
#endif

  // Allocate an extra phantom array to act as a safety net to
  // ensure that subsequent allocations performed internally by 
  // the NVIDIA thrust template library or by our 
  // marching cubes implementation don't fail, since we can't 
  // currently pre-allocate all of them.
  cudaMalloc(&gpuh->safety, 
             natoms*sizeof(float4) +                          // thrust
             8 * gx * gy * sizeof(float) +                    // thrust
             MCsz);                                           // mcubes

  if (gpuh->verbose > 1)
    printf("Total QuickSurf mem size: %d MB\n", totalmemsz / (1024*1024));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return -1;

  // once the allocation has succeeded, we update the GPU handle info 
  // so that the next test/allocation pass knows the latest state.
  gpuh->natoms = natoms;
  gpuh->colorperatom = colorperatom;

  gpuh->acx = acx;
  gpuh->acy = acy;
  gpuh->acz = acz;

  gpuh->gx = gx;
  gpuh->gy = gy;
  gpuh->gz = gz;

  return 0;
}


int CUDAQuickSurf::get_chunk_bufs(int testexisting,
                                  long int natoms, int colorperatom, 
                                  VolTexFormat vtexformat,
                                  int acx, int acy, int acz,
                                  int gx, int gy, int gz,
                                  int &cx, int &cy, int &cz,
                                  int &sx, int &sy, int &sz) {
  dim3 Bsz(GBLOCKSZX, GBLOCKSZY, GBLOCKSZZ);
  if (colorperatom)
    Bsz.z = GTEXBLOCKSZZ;

  cudaError_t err = cudaGetLastError(); // eat error so next CUDA op succeeds

  // enter loop to attempt a single-pass computation, but if the
  // allocation fails, cut the chunk size Z dimension by half
  // repeatedly until we either run chunks of 8 planes at a time,
  // otherwise we assume it is hopeless.
  cz <<= 1; // premultiply by two to simplify loop body
  int chunkiters = 0;
  int chunkallocated = 0;
  while (!chunkallocated) {
    // Cut the Z chunk size in half
    chunkiters++;
    cz >>= 1;

    // if we've already dropped to a subvolume size, subtract off the
    // four extra Z planes from last time before we do the modulo padding
    // calculation so we don't hit an infinite loop trying to go below 
    // 16 planes due the padding math below.
    if (cz != gz)
      cz-=4;

    // Pad the chunk to a multiple of the computational tile size since
    // each thread computes multiple elements (unrolled in the Z direction)
    cz += (8 - (cz % 8));

    // The density map "slab" size is the chunk size but without the extra
    // plane used to copy the last plane of the previous chunk's density
    // into the start, for use by the marching cubes.
    sx = cx;
    sy = cy;
    sz = cz;

    // Add four extra Z-planes for copying the previous end planes into 
    // the start of the next chunk.
    cz+=4;

#if 0
    printf("  Trying slab size: %d (test: %d)\n", sz, testexisting);
#endif

#if 1
    // test to see if total number of thread blocks exceeds maximum
    // number we can reasonably run prior to a kernel timeout error
    dim3 tGsz((sx+Bsz.x-1) / Bsz.x, 
              (sy+Bsz.y-1) / Bsz.y,
              (sz+(Bsz.z*GUNROLL)-1) / (Bsz.z * GUNROLL));
    if (colorperatom) {
      tGsz.z = (sz+(Bsz.z*GTEXUNROLL)-1) / (Bsz.z * GTEXUNROLL);
    }
    if (tGsz.x * tGsz.y * tGsz.z > 65535)
      continue; 
#endif

    // Bail out if we can't get enough memory to run at least
    // 8 slices in a single pass (making sure we've freed any allocations
    // beforehand, so they aren't leaked).
    if (sz <= 8) {
      return -1;
    }
 
    if (testexisting) {
      if (check_bufs(natoms, colorperatom, acx, acy, acz, cx, cy, cz) != 0)
        continue;
    } else {
      if (alloc_bufs(natoms, colorperatom, vtexformat, acx, acy, acz, cx, cy, cz) != 0)
        continue;
    }

    chunkallocated=1;
  }

  return 0;
}


int CUDAQuickSurf::calc_surf(long int natoms, const float *xyzr_f, 
                             const float *colors_f,
                             int colorperatom,
                             float *origin, int *numvoxels, float maxrad,
                             float radscale, float gridspacing, 
                             float isovalue, float gausslim,
                             VMDDisplayList *cmdList) {
  qsurf_gpuhandle *gpuh = (qsurf_gpuhandle *) voidgpu;

  // if there were any problems when the constructor tried to get
  // GPU hardware capabilities, we consider it to be unusable for all time.
  if (gpuh->dev < 0)
    return -1;

  // This code currently requires compute capability 2.x or greater.
  // We absolutely depend on hardware broadcasts for 
  // global memory reads by multiple threads reading the same element,
  // and the code more generally assumes the Fermi L1 cache and prefers
  // to launch 3-D grids where possible. 
  if (gpuh->deviceProp.major < 2) {
    return -1;
  }

  // start timing...
  wkf_timerhandle globaltimer = wkf_timer_create();
  wkf_timer_start(globaltimer);

  int vtexsize = 0;
  const VolTexFormat voltexformat = RGB4U; // XXX caller may want to set this
  switch (voltexformat) {
    case RGB3F: 
      vtexsize = sizeof(float3);
      break;

    case RGB4U: 
      vtexsize = sizeof(uchar4);
      break;
  }

  float4 *colors = (float4 *) colors_f;
  int chunkmaxverts=0;
  int chunknumverts=0; 
  int numverts=0;
  int numfacets=0;

  // compute grid spacing for the acceleration grid
  float acgridspacing = gausslim * radscale * maxrad;

  // ensure acceleration grid spacing >= density grid spacing
  if (acgridspacing < gridspacing)
    acgridspacing = gridspacing;

  // Allocate output arrays for the gaussian density map and 3-D texture map
  // We test for errors carefully here since this is the most likely place
  // for a memory allocation failure due to the size of the grid.
  int3 volsz = make_int3(numvoxels[0], numvoxels[1], numvoxels[2]);
  int3 chunksz = volsz;
  int3 slabsz = volsz;

  // An alternative scheme to minimize the QuickSurf GPU memory footprint
  if (getenv("VMDQUICKSURFMINMEM")) {
    if (volsz.z > 32) {
      slabsz.z = chunksz.z = 16;
    }
  }

  int3 accelcells;
  accelcells.x = max(int((volsz.x*gridspacing) / acgridspacing), 1);
  accelcells.y = max(int((volsz.y*gridspacing) / acgridspacing), 1);
  accelcells.z = max(int((volsz.z*gridspacing) / acgridspacing), 1);

  dim3 Bsz(GBLOCKSZX, GBLOCKSZY, GBLOCKSZZ);
  if (colorperatom)
    Bsz.z = GTEXBLOCKSZZ;

  // check to see if it's possible to use an existing allocation,
  // if so, just leave things as they are, and do the computation 
  // using the existing buffers
  if (gpuh->natoms == 0 ||
      get_chunk_bufs(1, natoms, colorperatom, voltexformat,
                     accelcells.x, accelcells.y, accelcells.z,
                     volsz.x, volsz.y, volsz.z,
                     chunksz.x, chunksz.y, chunksz.z,
                     slabsz.x, slabsz.y, slabsz.z) == -1) {
    // reset the chunksz and slabsz after failing to try and
    // fit them into the existing allocations...
    chunksz = volsz;
    slabsz = volsz;

    // reallocate the chunk buffers from scratch since we weren't
    // able to reuse them
    if (get_chunk_bufs(0, natoms, colorperatom, voltexformat,
                       accelcells.x, accelcells.y, accelcells.z,
                       volsz.x, volsz.y, volsz.z,
                       chunksz.x, chunksz.y, chunksz.z,
                       slabsz.x, slabsz.y, slabsz.z) == -1) {
      wkf_timer_destroy(globaltimer);
      return -1;
    }
  }
  chunkmaxverts = 3 * chunksz.x * chunksz.y * chunksz.z;

  // Free the "safety padding" memory we allocate to ensure we dont
  // have trouble with thrust calls that allocate their own memory later
  if (gpuh->safety != NULL)
    cudaFree(gpuh->safety);
  gpuh->safety = NULL;

  if (gpuh->verbose > 1) {
    printf("  Using GPU chunk size: %d\n", chunksz.z);
    printf("  Accel grid(%d, %d, %d) spacing %f\n",
           accelcells.x, accelcells.y, accelcells.z, acgridspacing);
  }

  // pre-process the atom coordinates and radii as needed
  // short-term fix until a new CUDA kernel takes care of this
  int i, i4;
  float4 *xyzr = (float4 *) malloc(natoms * sizeof(float4));
  float log2e = log2(2.718281828);
  for (i=0,i4=0; i<natoms; i++,i4+=4) {
    xyzr[i].x = xyzr_f[i4    ];
    xyzr[i].y = xyzr_f[i4 + 1];
    xyzr[i].z = xyzr_f[i4 + 2];

    float scaledrad = xyzr_f[i4 + 3] * radscale;
    float arinv = -1.0f * log2e / (2.0f*scaledrad*scaledrad);

    xyzr[i].w = arinv;
  }
  cudaMemcpy(gpuh->xyzr_d, xyzr, natoms * sizeof(float4), cudaMemcpyHostToDevice);
  free(xyzr);

  if (colorperatom)
    cudaMemcpy(gpuh->color_d, colors, natoms * sizeof(float4), cudaMemcpyHostToDevice);
 
  // build uniform grid acceleration structure
  if (vmd_cuda_build_density_atom_grid(natoms, gpuh->xyzr_d, gpuh->color_d,
                                       gpuh->sorted_xyzr_d,
                                       gpuh->sorted_color_d,
                                       gpuh->atomIndex_d, gpuh->atomHash_d,
                                       gpuh->cellStartEnd_d, 
                                       accelcells, 1.0f / acgridspacing) != 0) {
    wkf_timer_destroy(globaltimer);
    free_bufs();
    return -1;
  }

  double sorttime = wkf_timer_timenow(globaltimer);
  double lastlooptime = sorttime;

  double densitykerneltime = 0.0f;
  double densitytime = 0.0f;
  double mckerneltime = 0.0f;
  double mctime = 0.0f; 
  double copycalltime = 0.0f;
  double copytime = 0.0f;

  float *volslab_d = NULL;
  void *texslab_d = NULL;

  int lzplane = GBLOCKSZZ * GUNROLL;
  if (colorperatom)
    lzplane = GTEXBLOCKSZZ * GTEXUNROLL;

  // initialize CUDA marching cubes class instance or rebuild it if needed
  uint3 mgsz = make_uint3(chunksz.x, chunksz.y, chunksz.z);
  if (gpuh->mc == NULL) {
    gpuh->mc = new CUDAMarchingCubes(); 
    if (!gpuh->mc->Initialize(mgsz)) {
      printf("QuickSurf call to MC Initialize() failed\n");
    }
  } else {
    uint3 mcmaxgridsize = gpuh->mc->GetMaxGridSize();
    if (slabsz.x > mcmaxgridsize.x ||
        slabsz.y > mcmaxgridsize.y ||
        slabsz.z > mcmaxgridsize.z) {
      if (gpuh->verbose)
        printf("*** QuickSurf Allocating new MC object...\n");
 
      // delete marching cubes object
      delete gpuh->mc;

      // create and initialize CUDA marching cubes class instance
      gpuh->mc = new CUDAMarchingCubes(); 

      if (!gpuh->mc->Initialize(mgsz)) {
        printf("QuickSurf MC Initialize() call failed to recreate MC object\n");
      }
    } 
  }

  int z;
  int chunkcount=0;
  float invacgridspacing = 1.0f / acgridspacing;
  float invisovalue = 1.0f / isovalue;
  for (z=0; z<volsz.z; z+=slabsz.z) {
    int3 curslab = slabsz;
    if (z+curslab.z > volsz.z)
      curslab.z = volsz.z - z; 
  
    int slabplanesz = curslab.x * curslab.y;

    dim3 Gsz((curslab.x+Bsz.x-1) / Bsz.x, 
             (curslab.y+Bsz.y-1) / Bsz.y,
             (curslab.z+(Bsz.z*GUNROLL)-1) / (Bsz.z * GUNROLL));
    if (colorperatom)
      Gsz.z = (curslab.z+(Bsz.z*GTEXUNROLL)-1) / (Bsz.z * GTEXUNROLL);

    // For SM >= 2.x, we can run the entire slab in one pass by launching
    // a 3-D grid of thread blocks.
    dim3 Gszslice = Gsz;

    if (gpuh->verbose > 1) {
      printf("CUDA device %d, grid size %dx%dx%d\n", 
             0, Gsz.x, Gsz.y, Gsz.z);
      printf("CUDA: vol(%d,%d,%d) accel(%d,%d,%d)\n",
             curslab.x, curslab.y, curslab.z,
             accelcells.x, accelcells.y, accelcells.z);
      printf("Z=%d, curslab.z=%d\n", z, curslab.z);
    }

    // For all but the first density slab, we copy the last four 
    // planes of the previous run into the start of the next run so
    // that we can extract the isosurface with no discontinuities
    if (z == 0) {
      volslab_d = gpuh->devdensity;
      if (colorperatom)
        texslab_d = gpuh->devvoltexmap;
    } else {
      int fourplanes = 4 * slabplanesz;
      cudaMemcpy(gpuh->devdensity,
                 volslab_d + (slabsz.z-4) * slabplanesz, 
                 fourplanes * sizeof(float), cudaMemcpyDeviceToDevice);
      volslab_d = gpuh->devdensity + fourplanes;

      if (colorperatom) {
        cudaMemcpy(gpuh->devvoltexmap,
                   ((unsigned char *) texslab_d) + (slabsz.z-4) * slabplanesz * vtexsize, 
                   fourplanes * vtexsize, cudaMemcpyDeviceToDevice);
        texslab_d = ((unsigned char *) gpuh->devvoltexmap) + fourplanes * vtexsize;
      }
    }

    // loop over the planes/slices in a slab and compute density and texture
    for (int lz=0; lz<Gsz.z; lz+=Gszslice.z) {
      int lzinc = lz * lzplane;
      float *volslice_d = volslab_d + lzinc * slabplanesz;

      if (colorperatom) {
        void *texslice_d = ((unsigned char *) texslab_d) + lzinc * slabplanesz * vtexsize;
        switch (voltexformat) {
          case RGB3F:
            gaussdensity_fast_tex3f<<<Gszslice, Bsz, 0>>>(natoms, 
                gpuh->sorted_xyzr_d, gpuh->sorted_color_d, 
                curslab, accelcells, acgridspacing, invacgridspacing, 
                gpuh->cellStartEnd_d, gridspacing, z+lzinc,
                volslice_d, (float3 *) texslice_d, invisovalue);
            break;

          case RGB4U:
            gaussdensity_fast_tex_norm<float, uchar4><<<Gszslice, Bsz, 0>>>(natoms, 
                gpuh->sorted_xyzr_d, gpuh->sorted_color_d, 
                curslab, accelcells, acgridspacing, invacgridspacing, 
                gpuh->cellStartEnd_d, gridspacing, z+lzinc,
                volslice_d, (uchar4 *) texslice_d, invisovalue);
            break;
        }
      } else {
        gaussdensity_fast<<<Gszslice, Bsz, 0>>>(natoms, gpuh->sorted_xyzr_d, 
            curslab, accelcells, acgridspacing, invacgridspacing, 
            gpuh->cellStartEnd_d, gridspacing, z+lzinc, volslice_d);
      }
    }
    cudaDeviceSynchronize(); 
    densitykerneltime = wkf_timer_timenow(globaltimer);

#if 0
    printf("  CUDA mcubes..."); fflush(stdout);
#endif

    uint3 gvsz = make_uint3(curslab.x, curslab.y, curslab.z);

    // For all but the first density slab, we copy the last four
    // planes of the previous run into the start of the next run so
    // that we can extract the isosurface with no discontinuities
    if (z != 0)
      gvsz.z=curslab.z + 4;

    float3 bbox = make_float3(gvsz.x * gridspacing, gvsz.y * gridspacing,
                              gvsz.z * gridspacing);

    float3 gorigin = make_float3(origin[0], origin[1], 
                                 origin[2] + (z * gridspacing));
    if (z != 0)
      gorigin.z = origin[2] + ((z-4) * gridspacing);

#if 0
printf("\n  ... vsz: %d %d %d\n", gvsz.x, gvsz.y, gvsz.z);
printf("  ... org: %.2f %.2f %.2f\n", gorigin.x, gorigin.y, gorigin.z);
printf("  ... bxs: %.2f %.2f %.2f\n", bbox.x, bbox.y, bbox.y);
printf("  ... bbe: %.2f %.2f %.2f\n", gorigin.x+bbox.x, gorigin.y+bbox.y, gorigin.z+bbox.z);
#endif

    // If we are computing the volume using multiple passes, we have to 
    // overlap the marching cubes grids and compute a sub-volume to exclude
    // the end planes, except for the first and last sub-volume, in order to
    // get correct per-vertex normals at the edges of each sub-volume 
    int skipstartplane=0;
    int skipendplane=0;
    if (chunksz.z < volsz.z) {
      // on any but the first pass, we skip the first Z plane
      if (z != 0)
        skipstartplane=1;

      // on any but the last pass, we skip the last Z plane
      if (z+curslab.z < volsz.z)
        skipendplane=1;
    }

    //
    // Extract density map isosurface using marching cubes
    //

    // Choose the isovalue dependingon whether the desnity map 
    // contains normalized or un-normalized density values
    if (voltexformat == RGB4U) {
      // incoming densities are pre-normalized so that the target isovalue
      // is represented as a density of 1.0f
      gpuh->mc->SetIsovalue(1.0f);
    } else {
      gpuh->mc->SetIsovalue(isovalue);
    }

    int mcstat = 0;
    switch (voltexformat) {
      case RGB3F:
        mcstat = gpuh->mc->SetVolumeData(gpuh->devdensity, 
                                         (float3 *) gpuh->devvoltexmap,
                                         gvsz, gorigin, bbox, true);
        break;

      case RGB4U:
        mcstat = gpuh->mc->SetVolumeData(gpuh->devdensity, 
                                         (uchar4 *) gpuh->devvoltexmap,
                                         gvsz, gorigin, bbox, true);
        break;
    }
    if (!mcstat) {
      printf("QuickSurf call to MC SetVolumeData() failed\n");
    }

    // set the sub-volume starting/ending indices if needed
    if (skipstartplane || skipendplane) {
      uint3 volstart = make_uint3(0, 0, 0);
      uint3 volend = make_uint3(gvsz.x, gvsz.y, gvsz.z);

      if (skipstartplane)
        volstart.z = 2;

      if (skipendplane)
        volend.z = gvsz.z - 2;

      gpuh->mc->SetSubVolume(volstart, volend);
    }
    if (gpuh->n3b_d) {
      gpuh->mc->computeIsosurface(gpuh->v3f_d, gpuh->n3b_d, 
                                  gpuh->c4u_d, chunkmaxverts);
    } else if (gpuh->c4u_d) {
      gpuh->mc->computeIsosurface(gpuh->v3f_d, gpuh->n3f_d, 
                                  gpuh->c4u_d, chunkmaxverts);
    } else {
      gpuh->mc->computeIsosurface(gpuh->v3f_d, gpuh->n3f_d, 
                                  gpuh->c3f_d, chunkmaxverts);
    }
    chunknumverts = gpuh->mc->GetVertexCount();

#if 0
    printf("generated %d vertices, max vert limit: %d\n", chunknumverts, chunkmaxverts);
#endif
    if (chunknumverts == chunkmaxverts)
      printf("  *** QuickSurf exceeded marching cubes vertex limit (%d verts)\n", chunknumverts);

    cudaDeviceSynchronize(); 
    mckerneltime = wkf_timer_timenow(globaltimer);

    // Create a triangle mesh
    if (chunknumverts > 0) {
      DispCmdTriMesh cmdTriMesh;
      if (colorperatom) {
        // emit triangle mesh with per-vertex colors
        if (gpuh->n3b_d) {
          cmdTriMesh.cuda_putdata((const float *) gpuh->v3f_d, 
                                  (const char *) gpuh->n3b_d, 
                                  (const unsigned char *) gpuh->c4u_d,
                                  chunknumverts/3, cmdList);
        } else if (gpuh->c4u_d) {
          cmdTriMesh.cuda_putdata((const float *) gpuh->v3f_d, 
                                  (const float *) gpuh->n3f_d, 
                                  (const unsigned char *) gpuh->c4u_d,
                                  chunknumverts/3, cmdList);
        } else {
          cmdTriMesh.cuda_putdata((const float *) gpuh->v3f_d, 
                                  (const float *) gpuh->n3f_d, 
                                  (const float *) gpuh->c3f_d,
                                  chunknumverts/3, cmdList);
        }
      } else {
        // emit triangle mesh with no colors, uses current rendering state
        if (gpuh->n3b_d) {
          cmdTriMesh.cuda_putdata((const float *) gpuh->v3f_d, 
                                  (const char *) gpuh->n3b_d, 
                                  (const unsigned char *) NULL,
                                  chunknumverts/3, cmdList);
        } else {
          cmdTriMesh.cuda_putdata((const float *) gpuh->v3f_d,
                                  (const float *) gpuh->n3f_d, 
                                  (const float *) NULL,
                                  chunknumverts/3, cmdList);
        }
      }
    }
    numverts+=chunknumverts;
    numfacets+=chunknumverts/3;

#if 0
   // XXX we'll hold onto this as we'll want to rescue this approach
   //     for electrostatics coloring where we have to have the 
   //     entire triangle mesh in order to do the calculation
    int l;
    int vertstart = 3 * numverts;
    int vertbufsz = 3 * (numverts + chunknumverts) * sizeof(float);
    int facebufsz = (numverts + chunknumverts) * sizeof(int);
    int chunkvertsz = 3 * chunknumverts * sizeof(float);

    v = (float*) realloc(v, vertbufsz);
    n = (float*) realloc(n, vertbufsz);
    c = (float*) realloc(c, vertbufsz);
    f = (int*)   realloc(f, facebufsz);
    cudaMemcpy(v+vertstart, gpuh->v3f_d, chunkvertsz, cudaMemcpyDeviceToHost);
    cudaMemcpy(n+vertstart, gpuh->n3f_d, chunkvertsz, cudaMemcpyDeviceToHost);
    if (colorperatom) {
      cudaMemcpy(c+vertstart, gpuh->c3f_d, chunkvertsz, cudaMemcpyDeviceToHost);
    } else {
      float *color = c+vertstart;
      for (l=0; l<chunknumverts*3; l+=3) {
        color[l + 0] = colors[0].x;
        color[l + 1] = colors[0].y;
        color[l + 2] = colors[0].z;
      }
    }
    for (l=numverts; l<numverts+chunknumverts; l++) {
      f[l]=l;
    }
    numverts+=chunknumverts;
    numfacets+=chunknumverts/3;
#endif

    copycalltime = wkf_timer_timenow(globaltimer);

    densitytime += densitykerneltime - lastlooptime;
    mctime += mckerneltime - densitykerneltime;
    copytime += copycalltime - mckerneltime;

    lastlooptime = wkf_timer_timenow(globaltimer);

    chunkcount++; // increment number of chunks processed
  }

  // catch any errors that may have occured so that at the very least,
  // all of the subsequent resource deallocation calls will succeed
  cudaError_t err = cudaGetLastError();

  wkf_timer_stop(globaltimer);
  double totalruntime = wkf_timer_time(globaltimer);
  wkf_timer_destroy(globaltimer);

  // If an error occured, we print it and return an error code, once
  // all of the memory deallocations have completed.
  if (err != cudaSuccess) { 
    printf("CUDA error: %s, %s line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    return -1;
  }

  if (gpuh->verbose) {
    printf("  GPU generated %d vertices, %d facets, in %d passes\n", 
           numverts, numfacets, chunkcount);
    printf("  GPU time (%s): %.3f [sort: %.3f density %.3f mcubes: %.3f copy: %.3f]\n", 
           "SM >= 2.x", totalruntime, sorttime, densitytime, mctime, copytime);
  }

  return 0;
}





