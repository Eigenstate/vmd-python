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
*      $RCSfile: OptiXRenderer.C
*      $Author: johns $      $Locker:  $               $State: Exp $
*      $Revision: 1.36 $         $Date: 2019/01/17 21:38:55 $
*
***************************************************************************
* DESCRIPTION:
*   OptiX PTX shader code
*
* This work is described in:
*  "GPU-Accelerated Molecular Visualization on
*   Petascale Supercomputing Platforms"
*   John E. Stone, Kirby L. Vandivort, and Klaus Schulten.
*   UltraVis'13: Proceedings of the 8th International Workshop on
*   Ultrascale Visualization, pp. 6:1-6:8, 2013.
*   http://dx.doi.org/10.1145/2535571.2535595
*
*  "Atomic Detail Visualization of Photosynthetic Membranes with
*   GPU-Accelerated Ray Tracing"
*   John E. Stone, Melih Sener, Kirby L. Vandivort, Angela Barragan,
*   Abhishek Singharoy, Ivan Teo, João V. Ribeiro, Barry Isralewitz,
*   Bo Liu, Boon Chong Goh, James C. Phillips, Craig MacGregor-Chatwin,
*   Matthew P. Johnson, Lena F. Kourkoutis, C. Neil Hunter, and Klaus Schulten
*   J. Parallel Computing, 55:17-27, 2016.
*   http://dx.doi.org/10.1016/j.parco.2015.10.015
*
*  "Immersive Molecular Visualization with Omnidirectional
*   Stereoscopic Ray Tracing and Remote Rendering"
*   John E. Stone, William R. Sherman, and Klaus Schulten.
*   High Performance Data Analysis and Visualization Workshop,
*   2016 IEEE International Parallel and Distributed Processing
*   Symposium Workshops (IPDPSW), pp. 1048-1057, 2016.
*   http://dx.doi.org/10.1109/IPDPSW.2016.121
*
*  "Omnidirectional Stereoscopic Projections for VR"
*   John E. Stone.
*   In, William R. Sherman, editor,
*   VR Developer Gems, Taylor and Francis / CRC Press, Chapter 24, 2019.
*
*  "Interactive Ray Tracing Techniques for
*   High-Fidelity Scientific Visualization"
*   John E. Stone.
*   In, Eric Haines and Tomas Akenine-Möller, editors,
*   Ray Tracing Gems, Apress, 2019.
*
*  "A Planetarium Dome Master Camera"
*   John E. Stone.
*   In, Eric Haines and Tomas Akenine-Möller, editors,
*   Ray Tracing Gems, Apress, 2019.
*
* Portions of this code are derived from Tachyon:
*   "An Efficient Library for Parallel Ray Tracing and Animation"
*   John E. Stone.  Master's Thesis, University of Missouri-Rolla,
*   Department of Computer Science, April 1998
*
*   "Rendering of Numerical Flow Simulations Using MPI"
*   John Stone and Mark Underwood.
*   Second MPI Developers Conference, pages 138-141, 1996.
*   http://dx.doi.org/10.1109/MPIDC.1996.534105
*
***************************************************************************/

#ifndef OPTIXSHADERS
#define OPTIXSHADERS
#include <optixu/optixu_vector_types.h>

// Compile-time flag for collection and reporting of ray statistics
#if 0
#define ORT_RAYSTATS 1
#endif

// Compile-time flag to enable the use of RTX hardware ray tracing 
// acceleration APIs in OptiX
#if OPTIX_VERSION >= 50200
#define ORT_USERTXAPIS 1
#endif

// When compiling with OptiX 3.8 or grater, we use the new
// progressive rendering APIs rather than our previous hand-coded
// progressive renderer.
#if (defined(VMDOPTIX_VCA) || (OPTIX_VERSION >= 3080)) // && !defined(VMDUSEOPENHMD)
#define VMDOPTIX_PROGRESSIVEAPI 1
#endif

#if 1 || defined(VMDOPTIX_PROGRESSIVEAPI)
#define VMDOPTIX_LIGHTUSEROBJS 1
#endif

#if defined(VMDOPTIX_LIGHTUSEROBJS)
#include "Scene.h" // for DISP_LIGHTS macro
#endif

// "*" operator
inline __host__ __device__ float3 operator*(char4 a, float b) {
  return make_float3(b * a.x, b * a.y, b * a.z);
}

inline __host__ __device__ float3 operator*(uchar4 a, float b) {
  return make_float3(b * a.x, b * a.y, b * a.z);
}

#if defined(__cplusplus)
  typedef optix::float3 float3;
#endif


// XXX OptiX 4.0 and later versions have a significant performance impact
//     on VMD startup if we use 256-way combinatorial shader specialization.
//     Shader template specialization had very little impact on
//     OptiX versions 3.[789].x previously.  The new LLVM based compiler 
//     back-end used in recent versions of OptiX has much more overhead
//     when processing large numbers of shaders single PTX files.  
//     If we want to retain the template specialization approach, 
//     we will have to generate shader code and store it in many separate 
//     PTX files to mitigate overheads in back-end compiler infrastructure.
#if OPTIX_VERSION < 40000
// this macro enables or disables the use of an array of
// template-specialized shaders for every combination of
// scene-wide and material-specific shader features.
#define ORT_USE_TEMPLATE_SHADERS 1
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


//
// Lighting data structures
//
#if defined(VMDOPTIX_LIGHTUSEROBJS)
typedef struct {
  int num_lights;
  float3 dirs[DISP_LIGHTS+1];  ///< VMD directional light count macro is 4 
} DirectionalLightList;

typedef struct {
  int num_lights;
  float3 posns[DISP_LIGHTS+1]; ///< VMD light count macro is 4 
} PositionalLightList;
#endif

typedef struct {
  float3 dir;
  int    padding; // pad to next power of two
} DirectionalLight;

typedef struct {
  float3 pos;
  int    padding; // pad to next power of two
} PositionalLight;


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



//
// Methods for packing normals into a 4-byte quantity, such as a 
// [u]int or [u]char4, and similar.  See JCGT article by Cigolle et al.,
// "A Survey of Efficient Representations for Independent Unit Vectors",
// J. Computer Graphics Techniques 3(2), 2014.
// http://jcgt.org/published/0003/02/01/
//
#if defined(ORT_USERTXAPIS)
#include <optixu/optixu_math_namespace.h> // for make_xxx() fctns

#if 1

//
// oct32: 32-bit octahedral normal encoding using [su]norm16x2 quantization
// Meyer et al., "On Floating Point Normal Vectors", In Proc. 21st
// Eurographics Conference on Rendering.
//   http://dx.doi.org/10.1111/j.1467-8659.2010.01737.x
// Others:
// https://twitter.com/Stubbesaurus/status/937994790553227264
// https://knarkowicz.wordpress.com/2014/04/16/octahedron-normal-vector-encoding
//
static __host__ __device__ __inline__ float3 OctDecode(float2 projected) {
  float3 n = make_float3(projected.x, 
                         projected.y, 
                         1.0f - (fabsf(projected.x) + fabsf(projected.y)));
  if (n.z < 0.0f) {
    float oldX = n.x;
    n.x = copysignf(1.0f - fabsf(n.y), oldX);
    n.y = copysignf(1.0f - fabsf(oldX), n.y);
  }

  return n;
}

//
// XXX TODO: implement a high-precision OctPEncode() variant, based on 
//           floored snorms and an error minimization scheme using a 
//           comparison of internally decoded values for least error
//

static __host__ __device__ __inline__ float2 OctEncode(float3 n) {
  const float invL1Norm = 1.0f / (fabsf(n.x) + fabsf(n.y) + fabsf(n.z));
  float2 projected;
  if (n.z < 0.0f) {
    projected = 1.0f - make_float2(fabsf(n.y), fabsf(n.x)) * invL1Norm;
    projected.x = copysignf(projected.x, n.x);
    projected.y = copysignf(projected.y, n.y);
  } else {
    projected = make_float2(n.x, n.y) * invL1Norm;
  }

  return projected;
}
 

static __host__ __device__ __inline__ uint convfloat2uint32(float2 f2) {
  f2 = f2 * 0.5f + 0.5f;
  uint packed;
  packed = ((uint) (f2.x * 65535)) | ((uint) (f2.y * 65535) << 16);
  return packed;
}

static __host__ __device__ __inline__ float2 convuint32float2(uint packed) {
  float2 f2;
  f2.x = (float)((packed      ) & 0x0000ffff) / 65535;
  f2.y = (float)((packed >> 16) & 0x0000ffff) / 65535;
  return f2 * 2.0f - 1.0f;
}


static __host__ __device__ __inline__ uint packNormal(const float3& normal) {
  float2 octf2 = OctEncode(normal);
  return convfloat2uint32(octf2);
}

static __host__ __device__ __inline__ float3 unpackNormal(uint packed) {
  float2 octf2 = convuint32float2(packed);
  return OctDecode(octf2);
}

#elif 1

// 
// unorm10x3: unsigned 10-bit-per-component scalar unit real representation
// Not quite as good as 'snorm' representations
// This is largely equivalent to OpenGL's UNSIGNED_INT_2_10_10_10_REV 
// Described in the GLSL 4.20 specification, J. Kessenich 2011
//   i=round(clamp(r,0,1) * (2^b - 1))
//   r=i/(2^b - 1)
//
static __host__ __device__ __inline__ uint packNormal(const float3& normal) {
  const float3 N = normal * 0.5f + 0.5f;
  const uint packed = ((uint) (N.x * 1023)) |
                      ((uint) (N.y * 1023) << 10) |
                      ((uint) (N.z * 1023) << 20);
  return packed;
}

static __host__ __device__ __inline__ float3 unpackNormal(uint packed) {
  float3 N;
  N.x = (float)(packed & 0x000003ff) / 1023;
  N.y = (float)(((packed >> 10) & 0x000003ff)) / 1023;
  N.z = (float)(((packed >> 20) & 0x000003ff)) / 1023;
  return N * 2.0f - 1.0f;
}

#elif 0

// 
// snorm10x3: signed 10-bit-per-component scalar unit real representation
// Better representation than unorm.  
// Supported by most fixed-function graphics hardware.
// https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_texture_snorm.txt
//   i=round(clamp(r,-1,1) * (2^(b-1) - 1)
//   r=clamp(i/(2^(b-1) - 1), -1, 1)
//

#elif 1

// OpenGL GLbyte signed quantization scheme
//   i = r * (2^b - 1) - 0.5;
//   r = (2i + 1)/(2^b - 1)
static __host__ __device__ __inline__ uint packNormal(const float3& normal) {
  // conversion to GLbyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
  const float3 N = normal * 127.5f - 0.5f;
  const char4 packed = make_char4(N.x, N.y, N.z, 0);
  return *((uint *) &packed);
}

static __host__ __device__ __inline__ float3 unpackNormal(uint packed) {
  char4 c4norm = *((char4 *) &packed);

  // conversion from GLbyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
  // float = (2c+1)/(2^8-1)
  const float ci2f = 1.0f / 255.0f;
  const float cn2f = 1.0f / 127.5f;
  float3 N = c4norm * cn2f + ci2f;

  return N;
}

#endif
#endif


#endif

