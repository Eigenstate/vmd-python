/***************************************************************************
 *cr
 *cr            (C) Copyright 2013-2014 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
* RCS INFORMATION:
*
*      $RCSfile: OptiXRenderer.C
*      $Author: johns $      $Locker:  $               $State: Exp $
*      $Revision: 1.22 $         $Date: 2015/05/24 03:15:38 $
*
***************************************************************************
* DESCRIPTION:
*   OptiX PTX shader code
*
*  "GPU-Accelerated Molecular Visualization on
*   Petascale Supercomputing Platforms"
*   John E. Stone, Kirby L. Vandivort, and Klaus Schulten.
*   UltraVis'13: Proceedings of the 8th International Workshop on
*   Ultrascale Visualization, pp. 6:1-6:8, 2013.
*   http://dx.doi.org/10.1145/2535571.2535595
*
***************************************************************************/

#ifndef OPTIXSHADERS
#define OPTIXSHADERS
#include <optixu/optixu_vector_types.h>

// When compiling with OptiX 3.8 or grater, we use the new
// progressive rendering APIs rather than our previous hand-coded
// progressive renderer.
#if (defined(VMDOPTIX_VCA) || (OPTIX_VERSION >= 3080))
#define VMDOPTIX_PROGRESSIVEAPI 1
#endif

#if 1 || defined(VMDOPTIX_PROGRESSIVEAPI)
#define VMDOPTIX_LIGHTUSEROBJS 1
#endif

#if defined(VMDOPTIX_LIGHTUSEROBJS)
#include "Scene.h" // for DISP_LIGHTS macro
#endif

#if defined(__cplusplus)
  typedef optix::float3 float3;
#endif

// Enable reversed traversal of any-hit rays for shadows/AO.
// This optimization yields a 20% performance gain in many cases.
// #define USE_REVERSE_SHADOW_RAYS 1
 
// Use reverse rays by default rather than only when enabled interactively
// #define USE_REVERSE_SHADOW_RAYS_DEFAULT 1
enum RtShadowMode { RT_SHADOWS_OFF=0,        ///< shadows disabled
                    RT_SHADOWS_ON=1,         ///< shadows on, std. impl.
                    RT_SHADOWS_ON_REVERSE=2  ///< any-hit traversal reversal 
                  };

#if defined(VMDOPTIX_LIGHTUSEROBJS)
typedef struct {
  int num_lights;
  float3 dirs[DISP_LIGHTS+1];
//  float3 dirs[5]; ///< VMD DISP_LIGHTS directional light count macro is 4 
} DirectionalLightList;
#endif

typedef struct {
  float3 dir;
  int    padding; // pad to next power of two
} DirectionalLight;

//
// Cylinders
//

// XXX memory layout is likely suboptimal
typedef struct {
  float3 start;
  float radius;
  float3 axis;
  float pad;
} vmd_cylinder;

// XXX memory layout is likely suboptimal
typedef struct {
  float3 start;
  float radius;
  float3 axis;
  float3 color;
} vmd_cylinder_color;

//
// Rings (annular or otherwise)
//

// XXX memory layout is likely suboptimal, but is a multiple float4
typedef struct {
  float3 center;
  float3 norm;
  float inrad;
  float outrad;
  float3 color;
  float pad;
} vmd_ring_color;


//
// Spheres
//

typedef struct {
  float3 center;
  float  radius;
} vmd_sphere;

// XXX memory layout is likely suboptimal
typedef struct {
  float3 center;
  float  radius;
  float3 color;
  float  pad;
} vmd_sphere_color;


//
// Triangle meshes of various kinds
//

// XXX memory layout is definitely suboptimal
typedef struct {
  float3 v0;
  float3 v1;
  float3 v2;
  float3 n0;
  float3 n1;
  float3 n2;
  float3 c0;
  float3 c1;
  float3 c2;
} vmd_tricolor;

typedef struct {
  uchar4 c0;
  uchar4 c1;
  uchar4 c2;
  char4  n0;
  char4  n1;
  char4  n2;
  float3 v0;
  float3 v1;
  float3 v2;
} vmd_trimesh_c4u_n3b_v3f;

typedef struct {
  float3 n0;
  float3 n1;
  float3 n2;
  float3 v0;
  float3 v1;
  float3 v2;
} vmd_trimesh_n3f_v3f;

typedef struct {
  char4  n0;
  char4  n1;
  char4  n2;
  float3 v0;
  float3 v1;
  float3 v2;
} vmd_trimesh_n3b_v3f;

typedef struct {
  float3 v0;
  float3 v1;
  float3 v2;
} vmd_trimesh_v3f;

#endif

