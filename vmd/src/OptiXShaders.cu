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
*      $Revision: 1.101 $         $Date: 2015/05/28 23:07:22 $
*
***************************************************************************
* DESCRIPTION:
*   OptiX PTX shader code.
*
* Major parts of this code are directly taken from the Tachyon parallel 
* ray tracer, and were contributed to VMD by the Tachyon author 
* and lead VMD developer John E. Stone.
*
* This work is described in:
*  "GPU-Accelerated Molecular Visualization on
*   Petascale Supercomputing Platforms"
*   John E. Stone, Kirby L. Vandivort, and Klaus Schulten.
*   UltraVis'13: Proceedings of the 8th International Workshop on
*   Ultrascale Visualization, pp. 6:1-6:8, 2013.
*   http://dx.doi.org/10.1145/2535571.2535595
*
* Major parts of this code are derived from Tachyon:
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
#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include "OptiXShaders.h"

// Runtime-based RT output coloring
//#define ORT_TIME_COLORING 1
// normalize runtime color by max acceptable pixel runtime 
#define ORT_TIME_NORMALIZATION (1e-9 / 2.5f)
// the default time coloring method averages contributions,
// but we often want to know what the worst time was rather than
// the average. 
#define ORT_TIME_COMBINE_MAX 1

// Macros related to ray origin epsilon stepping to prevent 
// self-intersections with the surface we're leaving
// This is a cheesy way of avoiding self-intersection 
// but it ameliorates the problem.
// At present without the ray origin stepping, even a simple
// STMV test case will exhibit self-intersection artifacts that
// are particularly obvious in shadowing for direct lighting.
// Since changing the scene epsilon even to large values does not 
// always cure the problem, this workaround is still required.
#define ORT_USE_RAY_STEP 1
#define ORT_RAY_STEP     N*scene_epsilon*4.0f

// reverse traversal of any-hit rays for shadows/AO
//
// XXX The macro definition of USE_REVERSE_SHADOW_RAYS is now located in 
//     OptiXShaders.h, since both the shader and the host code need to
//     know if it is defined...
//
// XXX We currently hard-code REVERSE_RAY_LENGTH in such a way that
//     it works well for scenes that fall within the VMD view volume,
//     given the relationship between the model and camera coordinate
//     systems, but this would be best computed by the diagonal of the 
//     AABB for the full scene, and then scaled into camera coordinates.
//     The REVERSE_RAY_STEP size is computed to avoid self intersection 
//     with the surface we're shading.
#define REVERSE_RAY_STEP       (scene_epsilon*10.0f)
#define REVERSE_RAY_LENGTH     3.0f

using namespace optix;

// "*" operator
inline __host__ __device__ float3 operator*(char4 a, float b) {
  return make_float3(b * a.x, b * a.y, b * a.z);
}

inline __host__ __device__ float3 operator*(uchar4 a, float b) {
  return make_float3(b * a.x, b * a.y, b * a.z);
}


//
// TEA, a tiny encryption algorithm.
// D. Wheeler and R. Needham, 2nd Intl. Workshop Fast Software Encryption, 
// LNCS, pp. 363-366, 1994.
//
// GPU Random Numbers via the Tiny Encryption Algorithm
// F. Zafar, M. Olano, and A. Curtis.
// HPG '10 Proceedings of the Conference on High Performance Graphics,
// pp. 133-141, 2010.  
//
template<unsigned int N> static __host__ __device__ __inline__ 
unsigned int tea(unsigned int val0, unsigned int val1) {
  unsigned int v0 = val0;
  unsigned int v1 = val1;
  unsigned int s0 = 0;

  for( unsigned int n = 0; n < N; n++ ) {
    s0 += 0x9e3779b9;
    v0 += ((v1<<4)+0xa341316c)^(v1+s0)^((v1>>5)+0xc8013ea4);
    v1 += ((v0<<4)+0xad90777d)^(v0+s0)^((v0>>5)+0x7e95761e);
  }

  return v0;
}


// output image framebuffer, accumulation buffer, and ray launch parameters
rtBuffer<uchar4, 2> framebuffer;
rtBuffer<float4, 2> accumulation_buffer;
rtDeclareVariable(float, accumulation_normalization_factor, , );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim, rtLaunchDim, );
#if defined(VMDOPTIX_PROGRESSIVEAPI)
rtDeclareVariable(unsigned int, progressiveSubframeIndex, rtSubframeIndex, );
#endif
rtDeclareVariable(unsigned int, accumCount, , );
rtDeclareVariable(int, progressive_enabled, , );

// epsilon value to use to avoid self-intersection 
rtDeclareVariable(float, scene_epsilon, , );

// max ray recursion depth...
rtDeclareVariable(int, max_depth, , );

// shadow rendering mode
rtDeclareVariable(int, shadows_enabled, , );
rtDeclareVariable(int, aa_samples, , );

// ambient occlusion sample counts and scaling factors
rtDeclareVariable(int, ao_samples, , );
rtDeclareVariable(float, ao_ambient, , );
rtDeclareVariable(float, ao_direct, , );

// background color and/or background gradient
rtDeclareVariable(float3, scene_bg_color, , );
rtDeclareVariable(float3, scene_bg_color_grad_top, , );
rtDeclareVariable(float3, scene_bg_color_grad_bot, , );
rtDeclareVariable(float3, scene_gradient, , );
rtDeclareVariable(float, scene_gradient_topval, , );
rtDeclareVariable(float, scene_gradient_botval, , );
rtDeclareVariable(float, scene_gradient_invrange, , );

// fog state
rtDeclareVariable(int, fog_mode, , );
rtDeclareVariable(float, fog_start, , );
rtDeclareVariable(float, fog_end, , );
rtDeclareVariable(float, fog_density, , );

// camera parameters
rtDeclareVariable(float,  cam_zoom, , );
rtDeclareVariable(float3, cam_pos, , );
rtDeclareVariable(float3, cam_U, , );
rtDeclareVariable(float3, cam_V, , );
rtDeclareVariable(float3, cam_W, , );

// stereoscopic camera parameters
rtDeclareVariable(float,  cam_stereo_eyesep, , );
rtDeclareVariable(float,  cam_stereo_convergence_dist, , );

// camera depth-of-field parameters
rtDeclareVariable(float,  cam_dof_aperture_rad, , );
rtDeclareVariable(float,  cam_dof_focal_dist, , );

// various shading related per-ray state
rtDeclareVariable(unsigned int, radiance_ray_type, , );
rtDeclareVariable(unsigned int, shadow_ray_type, , );
rtDeclareVariable(rtObject, root_object, , );
rtDeclareVariable(rtObject, root_shadower, , );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
//rtDeclareVariable(float3, texcoord, attribute texcoord, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );

// object color assigned at intersection time
rtDeclareVariable(float3, obj_color, attribute obj_color, );

struct PerRayData_radiance {
  float3 result;
  float importance;
  int depth;
};

struct PerRayData_shadow {
  float3 attenuation;
};

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_radiance, prd, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );

// list of directional lights
#if defined(VMDOPTIX_LIGHTUSEROBJS)
rtDeclareVariable(DirectionalLightList, light_list, , );
#else
rtBuffer<DirectionalLight> lights;
#endif


//
// convert float3 rgb data to uchar4 with alpha channel set to 255.
//
static __device__ __inline__ uchar4 make_color_rgb4u(const float3& c) {
  return make_uchar4(static_cast<unsigned char>(__saturatef(c.x)*255.99f),  
                     static_cast<unsigned char>(__saturatef(c.y)*255.99f),  
                     static_cast<unsigned char>(__saturatef(c.z)*255.99f),  255u);
}

//
// convert float4 rgba data to uchar4
//
static __device__ __inline__ uchar4 make_color_rgb4u(const float4& c) {
  return make_uchar4(static_cast<unsigned char>(__saturatef(c.x)*255.99f),  
                     static_cast<unsigned char>(__saturatef(c.y)*255.99f),  
                     static_cast<unsigned char>(__saturatef(c.z)*255.99f),  
                     static_cast<unsigned char>(__saturatef(c.w)*255.99f));
}


//
// OptiX miss programs for drawing the background color or 
// background color gradient when no objects are hit
//

// Miss program for solid background
RT_PROGRAM void miss_solid_bg() {
  // Fog overrides the background color if we're using
  // Tachyon radial fog, but not for OpenGL style fog.
  prd_radiance.result = scene_bg_color;
}


// Miss program for gradient background with perspective projection,
// adapted from Tachyon
// Fog overrides the background color if we're using
// Tachyon radial fog, but not for OpenGL style fog.
RT_PROGRAM void miss_gradient_bg_sky_sphere() {
  float IdotG = dot(ray.direction, scene_gradient);
  float val = (IdotG - scene_gradient_botval) * scene_gradient_invrange;
  val = clamp(val, 0.0f, 1.0f);
  float3 col = val * scene_bg_color_grad_top + 
               (1.0f - val) * scene_bg_color_grad_bot; 
  prd_radiance.result = col;
}


// Miss program for gradient background with orthographic projection,
// adapted from Tachyon
// Fog overrides the background color if we're using
// Tachyon radial fog, but not for OpenGL style fog.
RT_PROGRAM void miss_gradient_bg_sky_plane() {
  float IdotG = dot(ray.origin, scene_gradient);
  float val = (IdotG - scene_gradient_botval) * scene_gradient_invrange;
  val = clamp(val, 0.0f, 1.0f);
  float3 col = val * scene_bg_color_grad_top + 
               (1.0f - val) * scene_bg_color_grad_bot; 
  prd_radiance.result = col;
}


//
// Various random number routines adapted from Tachyon
//
#define MYRT_RAND_MAX 4294967296.0f       /* Max random value from rt_rand  */
#define MYRT_RAND_MAX_INV .00000000023283064365f   /* normalize rt_rand  */

//
// Quick and dirty 32-bit LCG random number generator [Fishman 1990]:
//   A=1099087573 B=0 M=2^32
//   Period: 10^9
// Fastest gun in the west, but fails many tests after 10^6 samples,
// and fails all statistics tests after 10^7 samples.
// It fares better than the Numerical Recipes LCG.  This is the fastest
// power of two rand, and has the best multiplier for 2^32, found by
// brute force[Fishman 1990].  Test results:
//   http://www.iro.umontreal.ca/~lecuyer/myftp/papers/testu01.pdf
//   http://www.shadlen.org/ichbin/random/
//
static __device__ __inline__ 
unsigned int myrt_rand(unsigned int &idum) {
  // on machines where int is 32-bits, no need to mask
  idum *= 1099087573;
  return idum;
}


// Generate an offset to jitter AA samples in the image plane, adapted
// from the code in Tachyon
static __device__ __inline__ 
void jitter_offset2f(unsigned int &pval, float2 &xy) {
  xy.x = (myrt_rand(pval) * MYRT_RAND_MAX_INV) - 0.5f;
  xy.y = (myrt_rand(pval) * MYRT_RAND_MAX_INV) - 0.5f;
}


// Generate an offset to jitter DoF samples in the Circle of Confusion,
// adapted from the code in Tachyon
static __device__ __inline__ 
void jitter_disc2f(unsigned int &pval, float2 &xy, float radius) {
#if 1
  // Since the GPU RT currently uses super cheap/sleazy LCG RNGs,
  // it is best to avoid using sample picking, which can fail if
  // we use a multiply-only RNG and we hit a zero in the PRN sequence.
  // The special functions are slow, but have bounded runtime and 
  // minimal branch divergence.
  float   r=(myrt_rand(pval) * MYRT_RAND_MAX_INV);
  float phi=(myrt_rand(pval) * MYRT_RAND_MAX_INV) * 2.0f * M_PIf;
  __sincosf(phi, &xy.x, &xy.y); // fast approximation
  xy *= sqrtf(r) * radius;
#else
  // Pick uniform samples that fall within the disc --
  // this scheme can hang in an endless loop if a poor quality
  // RNG is used and it gets stuck in a short PRN sub-sequence
  do { 
    xy.x = 2.0f * (myrt_rand(pval) * MYRT_RAND_MAX_INV) - 1.0f;
    xy.y = 2.0f * (myrt_rand(pval) * MYRT_RAND_MAX_INV) - 1.0f;
  } while ((xy.x*xy.x + xy.y*xy.y) > 1.0f);
  xy *= radius; 
#endif
}


// Generate a randomly oriented ray, based on Tachyon implementation
static __device__ __inline__ 
void jitter_sphere3f(unsigned int &pval, float3 &dir) {
#if 1
  //
  // Use GPU fast/approximate math routines
  //
  /* Archimedes' cylindrical projection scheme       */
  /* generate a point on a unit cylinder and project */
  /* back onto the sphere.  This approach is likely  */
  /* faster for SIMD hardware, despite the use of    */
  /* transcendental functions.                       */
  float u1 = myrt_rand(pval) * MYRT_RAND_MAX_INV;
  dir.z = 2.0f * u1 - 1.0f;
  float R = __fsqrt_rn(1.0f - dir.z*dir.z);  // fast approximation
  float u2 = myrt_rand(pval) * MYRT_RAND_MAX_INV;
  float phi = 2.0f * M_PIf * u2;
  float sinphi, cosphi;
  __sincosf(phi, &sinphi, &cosphi); // fast approximation
  dir.x = R * cosphi;
  dir.y = R * sinphi;
#elif 1
  /* Archimedes' cylindrical projection scheme       */
  /* generate a point on a unit cylinder and project */
  /* back onto the sphere.  This approach is likely  */
  /* faster for SIMD hardware, despite the use of    */
  /* transcendental functions.                       */
  float u1 = myrt_rand(pval) * MYRT_RAND_MAX_INV;
  dir.z = 2.0f * u1 - 1.0f;
  float R = sqrtf(1.0f - dir.z*dir.z); 

  float u2 = myrt_rand(pval) * MYRT_RAND_MAX_INV;
  float phi = 2.0f * M_PIf * u2;
  float sinphi, cosphi;
  sincosf(phi, &sinphi, &cosphi);
  dir.x = R * cosphi;
  dir.y = R * sinphi;
#else
  /* Marsaglia's uniform sphere sampling scheme           */
  /* In order to correctly sample a sphere, using rays    */
  /* generated randomly within a cube we must throw out   */
  /* direction vectors longer than 1.0, otherwise we'll   */
  /* oversample the corners of the cube relative to       */
  /* a true sphere.                                       */
  float len;
  float3 d;
  do {
    d.x = (myrt_rand(pval) * MYRT_RAND_MAX_INV) - 0.5f;
    d.y = (myrt_rand(pval) * MYRT_RAND_MAX_INV) - 0.5f;
    d.z = (myrt_rand(pval) * MYRT_RAND_MAX_INV) - 0.5f;
    len = dot(d, d);
  } while (len > 0.250f);
  float invlen = rsqrtf(len);

  /* finish normalizing the direction vector */
  dir = d * invlen;
#endif
}


//
// Clear the accumulation buffer to zeros
//
RT_PROGRAM void clear_accumulation_buffer() {
  accumulation_buffer[launch_index] = make_float4(0.0f,0.0f,0.0f,0.0f);
}


//
// Copy the contents of the accumulation buffer to the destination 
// framebuffer, while converting the representation of the data from
// floating point down to unsigned chars, performing clamping and any
// other postprocessing at the same time.
//
RT_PROGRAM void draw_accumulation_buffer() {
#if defined(ORT_TIME_COLORING) && defined(ORT_TIME_COMBINE_MAX)
  float4 curcol = accumulation_buffer[launch_index];

  // divide time value by normalization factor (to scale up the max value)
  curcol.x /= accumulation_normalization_factor;

  // multiply the remaining color components (to average them)
  curcol.y *= accumulation_normalization_factor;
  curcol.z *= accumulation_normalization_factor;
  curcol.w *= accumulation_normalization_factor;

  framebuffer[launch_index] = make_color_rgb4u(curcol);
#else
  framebuffer[launch_index] = make_color_rgb4u(accumulation_buffer[launch_index] * accumulation_normalization_factor);
#endif
}

// no-op placeholder used when running with progressive rendering
RT_PROGRAM void draw_accumulation_buffer_stub() {
}


//
// OptiX programs that implement the camera models and ray generation 
// code for both perspective and orthographic projection modes of VMD
//

//
// Ray gen accumulation buffer helper routines
//
static void __inline__ __device__ accumulate_color(float3 &col) {
#if defined(VMDOPTIX_PROGRESSIVEAPI)
  if (progressive_enabled) {
    col *= accumulation_normalization_factor;

#if OPTIX_VERSION < 3080
    // XXX prior to OptiX 3.8, a hard-coded gamma correction was required
    // VCA gamma correction workaround, changes gamma 2.2 back to gamma 1.0
    float invgamma = 1.0f / 0.4545f;
    col.x = powf(col.x, invgamma);
    col.y = powf(col.y, invgamma);
    col.z = powf(col.z, invgamma);
#endif

    // for optix-vca progressive mode accumulation is handled in server code
    accumulation_buffer[launch_index]  = make_float4(col, 1.0f);
  } else {
    // For batch mode we accumulate ourselves
    accumulation_buffer[launch_index] += make_float4(col, 1.0f);
  }
#else
  // For batch mode we accumulate ourselves
  accumulation_buffer[launch_index] += make_float4(col, 1.0f);
#endif
}


#if defined(ORT_TIME_COLORING)
// special accumulation helper routine for time-based coloring
static void __inline__ __device__ 
accumulate_time_coloring(float3 &col, clock_t t0) {
  clock_t t1 = clock(); // stop per-pixel RT timer (in usec)
  float4 curcol = accumulation_buffer[launch_index];

  // overwrite the red channel with fraction of the max allowable runtime,
  // clamped to the range 0-1, in the case of excessively long traces
  float pixel_time = (t1 - t0) * ORT_TIME_NORMALIZATION;
  col.x = clamp(pixel_time, 0.0f, 1.0f);

#if defined(ORT_TIME_COMBINE_MAX)
  // return the slowest (max) time, but average the colors
  curcol.x = fmaxf(curcol.x, col.x);
  curcol.y += col.y;
  curcol.z += col.z;
  curcol.w += 1.0f;
#else
  // average both time and colors
  curcol += make_float4(col, 1.0f);
#endif
  
  accumulation_buffer[launch_index] = curcol;
}
#endif


static int __inline__ __device__ subframe_count() {
#if defined(VMDOPTIX_PROGRESSIVEAPI)
  return (accumCount + progressiveSubframeIndex);
#else
  return accumCount;
#endif
}


//
// Camera ray generation code for planetarium dome display
// Generates a fisheye style frame with ~180 degree FoV
// 
// template<int STEREO_ON>
// static __device__ __inline__
// void vmd_camera_dome_general() {
RT_PROGRAM void vmd_camera_dome_master() {
#if defined(ORT_TIME_COLORING)
  clock_t t0 = clock(); // start per-pixel RT timer
#endif

  float fov = 180.0f * cam_zoom;             // dome FoV in degrees 
  float rmax = 0.5 * fov * (M_PIf / 180.0f); // half FoV in radians
  float2 viewport_sz = make_float2(launch_dim.x, launch_dim.y);
  float2 radperpix = (M_PIf / 180.0f) * fov / viewport_sz;
  float2 viewport_mid = viewport_sz * 0.5f;

  unsigned int randseed = tea<4>(launch_dim.x*(launch_index.y)+launch_index.x, subframe_count());

  float3 col = make_float3(0.0f);

  for (int s=0; s<aa_samples; s++) {
    float2 jxy;  
    jitter_offset2f(randseed, jxy);

    float2 viewport_idx = make_float2(launch_index.x, launch_index.y) + jxy;
    float2 rd = (viewport_idx - viewport_mid) * radperpix;
    float rangle = hypotf(rd.x, rd.y); 

    // pixels outside the dome are set to black
    if (rangle < rmax) {
      float3 ray_direction;

      // handle center of dome where azimuth is undefined
      if (rangle == 0) {
        ray_direction = normalize(cam_W);
      } else {
        float rsin = sinf(rangle) / rangle;
        ray_direction = normalize(cam_U*rsin*rd.x + cam_V*rsin*rd.y + cam_W*cos(rangle));
      }

      // trace the new ray...
      PerRayData_radiance prd;
      prd.importance = 1.f;
      prd.depth = 0;
      optix::Ray ray = optix::make_Ray(cam_pos, ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
      rtTrace(root_object, ray, prd);
      col += prd.result;
    }
  }

#if defined(ORT_TIME_COLORING)
  accumulate_time_coloring(col, t0);
#else
  accumulate_color(col);
#endif
}

#if 0
RT_PROGRAM void vmd_camera_dome_master() {
  vmd_camera_dome_general<0>();
}

RT_PROGRAM void vmd_camera_dome_master_stereo() {
  vmd_camera_dome_general<1>();
}
#endif


//
// Camera ray generation code for 360 degre FoV 
// equirectangular (lat/long) projection suitable
// for use a texture map for a sphere, e.g. for 
// immersive VR HMDs, other spheremap-based projections.
// 
// template<int STEREO_ON>
// static __device__ __inline__
// void vmd_camera_equirectangular_general() {
RT_PROGRAM void vmd_camera_equirectangular() {
#if defined(ORT_TIME_COLORING)
  clock_t t0 = clock(); // start per-pixel RT timer
#endif

  float2 viewport_sz = make_float2(launch_dim.x, launch_dim.y);
  float2 radperpix = M_PIf / viewport_sz * make_float2(2.0f, 1.0f);
  float2 viewport_mid = viewport_sz * 0.5f;

  unsigned int randseed = tea<4>(launch_dim.x*(launch_index.y)+launch_index.x, subframe_count());

  float3 col = make_float3(0.0f);

  for (int s=0; s<aa_samples; s++) {
    float2 jxy;  
    jitter_offset2f(randseed, jxy);

    float2 viewport_idx = make_float2(launch_index.x, launch_index.y) + jxy;
    float2 rangle = (viewport_idx - viewport_mid) * radperpix;

    float sin_ax, cos_ax, sin_ay, cos_ay;
    sincosf(rangle.x, &sin_ax, &cos_ax);
    sincosf(rangle.y, &sin_ay, &cos_ay);

    float3 ray_direction = normalize(cos_ay * (cos_ax * cam_W + sin_ax * cam_U) + sin_ay * cam_V);

    // trace the new ray...
    PerRayData_radiance prd;
    prd.importance = 1.f;
    prd.depth = 0;
    optix::Ray ray = optix::make_Ray(cam_pos, ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
    rtTrace(root_object, ray, prd);
    col += prd.result;
  }

#if defined(ORT_TIME_COLORING)
  accumulate_time_coloring(col, t0);
#else
  accumulate_color(col);
#endif
}

#if 0
RT_PROGRAM void vmd_camera_equirectangular() {
  vmd_camera_equirectangular_general<0>();
}

RT_PROGRAM void vmd_camera_equirectangular_stereo() {
  vmd_camera_equirectangular_general<1>();
}
#endif


//
// Templated perspective camera ray generation code
//
template<int STEREO_ON, int DOF_ON> 
static __device__ __inline__ 
void vmd_camera_perspective_general() {
#if defined(ORT_TIME_COLORING)
  clock_t t0 = clock(); // start per-pixel RT timer
#endif

  // Stereoscopic rendering is provided by rendering in an over/under
  // format with the left eye image into the top half of a double-high 
  // framebuffer, and the right eye into the lower half.  The subsequent 
  // OpenGL drawing code can trivially unpack and draw the two images 
  // with simple pointer offset arithmetic.
  float3 eyepos;
  uint viewport_sz_y, viewport_idx_y;
  if (STEREO_ON) {
    // render into a double-high framebuffer when stereo is enabled
    viewport_sz_y = launch_dim.y >> 1;
    if (launch_index.y >= viewport_sz_y) {
      // right image
      viewport_idx_y = launch_index.y - viewport_sz_y;
      eyepos = cam_pos + cam_U * cam_stereo_eyesep * 0.5f;
    } else {
      // left image
      viewport_idx_y = launch_index.y;
      eyepos = cam_pos - cam_U * cam_stereo_eyesep * 0.5f;
    }
  } else {
    // render into a normal size framebuffer if stereo is not enabled
    viewport_sz_y = launch_dim.y;
    viewport_idx_y = launch_index.y;
    eyepos = cam_pos;
  }

  // 
  // general primary ray calculations
  //
  float2 aspect = make_float2(float(launch_dim.x) / float(viewport_sz_y), 1.0f) * cam_zoom;
  float2 viewportscale = 1.0f / make_float2(launch_dim.x, viewport_sz_y);
  float2 d = make_float2(launch_index.x, viewport_idx_y) * viewportscale * aspect * 2.f - aspect; // center of pixel in image plane

  unsigned int randseed = tea<4>(launch_dim.x*(viewport_idx_y)+launch_index.x, subframe_count());

  float3 col = make_float3(0.0f);
  float3 newcampos = eyepos;
  for (int s=0; s<aa_samples; s++) {
    float2 jxy;  
    jitter_offset2f(randseed, jxy);
    jxy = jxy * viewportscale * aspect * 2.f + d;
    float3 ray_direction = normalize(jxy.x*cam_U + jxy.y*cam_V + cam_W);

    // compute new ray origin and ray direction
    if (DOF_ON) {
      float3 focuspoint = eyepos + ray_direction * cam_dof_focal_dist;
      float2 dofjxy;
      jitter_disc2f(randseed, dofjxy, cam_dof_aperture_rad);
      newcampos = eyepos + dofjxy.x*cam_U + dofjxy.y*cam_V;
      ray_direction = normalize(focuspoint - newcampos);
    }

    // trace the new ray...
    PerRayData_radiance prd;
    prd.importance = 1.f;
    prd.depth = 0;
    optix::Ray ray = optix::make_Ray(newcampos, ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
    rtTrace(root_object, ray, prd);
    col += prd.result; 
  }

#if defined(ORT_TIME_COLORING)
  accumulate_time_coloring(col, t0);
#else
  accumulate_color(col);
#endif
}


RT_PROGRAM void vmd_camera_perspective() {
  vmd_camera_perspective_general<0, 0>();
}

RT_PROGRAM void vmd_camera_perspective_dof() {
  vmd_camera_perspective_general<0, 1>();
}

RT_PROGRAM void vmd_camera_perspective_stereo() {
  vmd_camera_perspective_general<1, 0>();
}

RT_PROGRAM void vmd_camera_perspective_stereo_dof() {
  vmd_camera_perspective_general<1, 1>();
}


//
// Templated orthographic camera ray generation code
//
template<int STEREO_ON> 
static __device__ __inline__ 
void vmd_camera_orthographic_general() {
#if defined(ORT_TIME_COLORING)
  clock_t t0 = clock(); // start per-pixel RT timer
#endif

  // Stereoscopic rendering is provided by rendering in an over/under
  // format with the left eye image into the top half of a double-high 
  // framebuffer, and the right eye into the lower half.  The subsequent 
  // OpenGL drawing code can trivially unpack and draw the two images 
  // with simple pointer offset arithmetic.
  float3 eyepos;
  uint viewport_sz_y, viewport_idx_y;
  float3 ray_direction;
  if (STEREO_ON) {
    // render into a double-high framebuffer when stereo is enabled
    viewport_sz_y = launch_dim.y >> 1;
    if (launch_index.y >= viewport_sz_y) {
      // right image
      viewport_idx_y = launch_index.y - viewport_sz_y;
      eyepos = cam_pos + cam_U * cam_stereo_eyesep * 0.5f;
    } else {
      // left image
      viewport_idx_y = launch_index.y;
      eyepos = cam_pos - cam_U * cam_stereo_eyesep * 0.5f;
    }
    ray_direction = normalize(cam_pos-eyepos + normalize(cam_W) * cam_stereo_convergence_dist);
  } else {
    // render into a normal size framebuffer if stereo is not enabled
    viewport_sz_y = launch_dim.y;
    viewport_idx_y = launch_index.y;
    eyepos = cam_pos;
    ray_direction = normalize(cam_W);
  }

  // 
  // general primary ray calculations
  //
  float2 aspect = make_float2(float(launch_dim.x) / float(viewport_sz_y), 1.0f) * cam_zoom;
  float2 viewportscale = 1.0f / make_float2(launch_dim.x, viewport_sz_y);

  float2 d = make_float2(launch_index.x, viewport_idx_y) * viewportscale * aspect * 2.f - aspect; // center of pixel in image plane

  unsigned int randseed = tea<4>(launch_dim.x*(viewport_idx_y)+launch_index.x, subframe_count());

  float3 col = make_float3(0.0f);
  for (int s=0; s<aa_samples; s++) {
    float2 jxy;  
    jitter_offset2f(randseed, jxy);
    jxy = jxy * viewportscale * aspect * 2.f + d;
    float3 ray_origin = eyepos + jxy.x*cam_U + jxy.y*cam_V;

    // trace the new ray...
    PerRayData_radiance prd;
    prd.importance = 1.f;
    prd.depth = 0;
    optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
    rtTrace(root_object, ray, prd);
    col += prd.result; 
  }

#if defined(ORT_TIME_COLORING)
  accumulate_time_coloring(col, t0);
#else
  accumulate_color(col);
#endif
}

RT_PROGRAM void vmd_camera_orthographic() {
  vmd_camera_orthographic_general<0>();
}

RT_PROGRAM void vmd_camera_orthographic_stereo() {
  vmd_camera_orthographic_general<1>();
}



//
// Default exception handling behavior
//
RT_PROGRAM void exception() {
  const unsigned int code = rtGetExceptionCode();
  switch (code) {
    case RT_EXCEPTION_STACK_OVERFLOW:
      rtPrintf("Stack overflow at launch index (%d,%d):\n",
               launch_index.x, launch_index.y );
      break;

#if OPTIX_VERSION >= 3050
    case RT_EXCEPTION_TEXTURE_ID_INVALID:
    case RT_EXCEPTION_BUFFER_ID_INVALID:
#endif
    case RT_EXCEPTION_INDEX_OUT_OF_BOUNDS:
    case RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS:
    case RT_EXCEPTION_INVALID_RAY:
    case RT_EXCEPTION_INTERNAL_ERROR:
    case RT_EXCEPTION_USER:
    default: 
      printf("Caught exception 0x%X (%d) at launch index (%d,%d)\n", 
             code, code, launch_index.x, launch_index.y );
      break;
  }

#ifndef VMDOPTIX_PROGRESSIVEAPI
  framebuffer[launch_index] = make_color_rgb4u(make_float3(0.f, 0.f, 0.f));
#endif
}


//
// Fog helper function based on Tachyon
//
static __device__ float fog_coord(float3 hit_point) {
  // Compute planar fog (e.g. to match OpenGL) by projecting t value onto 
  // the camera view direction vector to yield a planar a depth value.
  float r = dot(ray.direction, cam_W) * t_hit;
  float f=1.0f;
  float v;

  switch (fog_mode) { 
    case 1: // RT_FOG_LINEAR
      f = (fog_end - r) / (fog_end - fog_start);
      break;

    case 2: // RT_FOG_EXP
      // XXX Tachyon allows fog_start to be non-zero for exponential fog,
      //     but OpenGL and VMD do not...
      // float v = fog_density * (r - fog_start);
      v = fog_density * r;
      f = expf(-v);
      break;

    case 3: // RT_FOG_EXP2
      // XXX Tachyon allows fog_start to be non-zero for exponential fog,
      //     but OpenGL and VMD do not...
      // float v = fog_density * (r - fog_start);
      v = fog_density * r;
      f = expf(-v*v);
      break;

    case 0: // RT_FOG_NONE
    default:
      break;
  }

  return clamp(f, 0.0f, 1.0f);
}


static __device__ float3 fog_color(float fogmod, float3 hit_col) {
  float3 col = (fogmod * hit_col) + ((1.0f - fogmod) * scene_bg_color);
  return col;
}



//
// trivial ambient occlusion implementation based on Tachyon
//
static __device__ float3 shade_ambient_occlusion(float3 hit, float3 N, float aoimportance) {
  // unweighted non-importance-sampled scaling factor
  float lightscale = 2.0f / ao_samples;
  float3 inten = make_float3(0.0f);

  unsigned int randseed = tea<2>(subframe_count(), subframe_count()); 

  PerRayData_shadow shadow_prd;
#if 1
  // do all the samples requested, with no observance of importance
  for (int s=0; s<ao_samples; s++) {
#else
  // dynamically scale the number of AO samples depending on the 
  // importance assigned to the incident ray and the opacity of the 
  // surface we are lighting.
  // XXX this scheme can create crawlies when animating since the 
  //     AO sample rays are no longer identical between neighboring 
  //     pixels and there's no guarantee that the samples we're skipping
  //     were low-importance in terms of their contribution.
  //     This kind of scheme would need much more development to be usable.
  int nsamples = ao_samples * prd.importance * aoimportance;
  if (nsamples < 1)
    nsamples=1;
  lightscale = 2.0f / nsamples;
  for (int s=0; s<nsamples; s++) {
#endif
    float3 dir;
    jitter_sphere3f(randseed, dir);
    float ndotambl = dot(N, dir);

    // flip the ray so it's in the same hemisphere as the surface normal
    if (ndotambl < 0.0f) {
      ndotambl = -ndotambl;
      dir = -dir;
    }

    Ray ambray;
#ifdef USE_REVERSE_SHADOW_RAYS 
    if (shadows_enabled == RT_SHADOWS_ON_REVERSE) {
      // reverse any-hit ray traversal direction for increased perf
      // XXX We currently hard-code REVERSE_RAY_LENGTH in such a way that
      //     it works well for scenes that fall within the VMD view volume,
      //     given the relationship between the model and camera coordinate
      //     systems, but this would be best computed by the diagonal of the 
      //     AABB for the full scene, and then scaled into camera coordinates.
      //     The REVERSE_RAY_STEP size is computed to avoid self intersection 
      //     with the surface we're shading.
      float tmax = REVERSE_RAY_LENGTH - REVERSE_RAY_STEP;
      ambray = make_Ray(hit + dir * REVERSE_RAY_LENGTH, -dir, shadow_ray_type, 0, tmax);
    } else
#endif
#if defined(ORT_USE_RAY_STEP) 
    ambray = make_Ray(hit + ORT_RAY_STEP, dir, shadow_ray_type, scene_epsilon, RT_DEFAULT_MAX);
#else
    ambray = make_Ray(hit, dir, shadow_ray_type, scene_epsilon, RT_DEFAULT_MAX);
#endif

    shadow_prd.attenuation = make_float3(1.0f);
    rtTrace(root_shadower, ambray, shadow_prd);
    inten += ndotambl * shadow_prd.attenuation;
  } 

  return inten * lightscale;
}


//
// Minimalistic VMD-centric re-implementation of the key portions 
// of Tachyon's main shader.  
//
// This shader has been written to be expanded into a large set of
// fully specialized shaders generated through combinatorial expansion
// of each of the major shader features associated with scene-wide or
// material-specific shading properties.
// At present, there are three scene-wide properties (fog, shadows, AO),
// and three material-specific properties (outline, reflection, transmission).
// Tere can be a performance cost for OptiX work scheduling of disparate 
// materials if too many unique materials are used in a scene. 
// Although there are 8 combinations of scene-wide parameters and 
// 8 combinations of material-specific parameters (64 in total),
// the scene-wide parameters are uniform for the whole scene. 
// We will therefore only have at most 8 different shader variants 
// in use in a given scene, due to the 8 possible combinations
// of material-specific (outline, reflection, transmission) properties.
// 
// The macros that generate the full set of 64 possible shader variants
// are at the very end of this source file.
//
template<int FOG_ON,           /// scene-wide shading property
         int SHADOWS_ON,       /// scene-wide shading property
         int AO_ON,            /// scene-wide shading property
         int OUTLINE_ON,       /// material-specific shading property
         int REFLECTION_ON,    /// material-specific shading property
         int TRANSMISSION_ON>  /// material-specific shading property
static __device__ void shader_template(float3 p_obj_color, float3 N, 
                                       float p_Ka, float p_Kd, float p_Ks,
                                       float p_phong_exp, float p_reflectivity,
                                       float p_opacity,
                                       float p_outline, float p_outlinewidth,
                                       int p_transmode) {
  float3 hit_point = ray.origin + t_hit * ray.direction;
  float3 result = make_float3(0.0f);
  float3 phongcol = make_float3(0.0f);

  // add depth cueing / fog if enabled
  // use fog coordinate to modulate importance for AO rays, etc.
  float fogmod = 1.0f;
  if (FOG_ON && fog_mode != 0) {
    fogmod = fog_coord(hit_point);
  }

  // execute the object's texture function
  float3 col = p_obj_color; // XXX no texturing implemented yet

  // compute lighting from directional lights
#if defined(VMDOPTIX_LIGHTUSEROBJS)
  unsigned int num_lights = light_list.num_lights;
  for (int i = 0; i < num_lights; ++i) {
    float3 L = light_list.dirs[i];
#else
  unsigned int num_lights = lights.size();
  for (int i = 0; i < num_lights; ++i) {
    DirectionalLight light = lights[i];
    float3 L = light.dir;
#endif
    float inten = dot(N, L);

    // cast shadow ray
    float3 light_attenuation = make_float3(static_cast<float>(inten > 0.0f));
    if (SHADOWS_ON && shadows_enabled && inten > 0.0f) {
      PerRayData_shadow shadow_prd;
      shadow_prd.attenuation = make_float3(1.0f);

      Ray shadow_ray;
#ifdef USE_REVERSE_SHADOW_RAYS
      if (shadows_enabled == RT_SHADOWS_ON_REVERSE) {
        // reverse any-hit ray traversal direction for increased perf
        // XXX We currently hard-code REVERSE_RAY_LENGTH in such a way that
        //     it works well for scenes that fall within the VMD view volume,
        //     given the relationship between the model and camera coordinate
        //     systems, but this would be best computed by the diagonal of the 
        //     AABB for the full scene, and then scaled into camera coordinates.
        //     The REVERSE_RAY_STEP size is computed to avoid self intersection 
        //     with the surface we're shading.
        float tmax = REVERSE_RAY_LENGTH - REVERSE_RAY_STEP;
        shadow_ray = make_Ray(hit_point + L * REVERSE_RAY_LENGTH, -L, shadow_ray_type, 0, tmax); 
      } 
      else
#endif
      shadow_ray = make_Ray(hit_point + ORT_RAY_STEP, L, shadow_ray_type, scene_epsilon, RT_DEFAULT_MAX);

      rtTrace(root_shadower, shadow_ray, shadow_prd);
      light_attenuation = shadow_prd.attenuation;
    }

    // If not completely shadowed, light the hit point.
    // When shadows are disabled, the light can't possibly be attenuated.
    if (!SHADOWS_ON || fmaxf(light_attenuation) > 0.0f) {
      result += col * p_Kd * inten * light_attenuation;

      // add specular hightlight using Blinn's halfway vector approach
      float3 H = normalize(L - ray.direction);
      float nDh = dot(N, H);
      if (nDh > 0) {
        float power = powf(nDh, p_phong_exp);
        phongcol += make_float3(p_Ks) * power * light_attenuation;
      }
    }
  }

  // add ambient occlusion diffuse lighting, if enabled
  if (AO_ON && ao_samples > 0) {
    result *= ao_direct;
    result += ao_ambient * col * p_Kd * shade_ambient_occlusion(hit_point, N, fogmod * p_opacity);
  }

  // add edge shading if applicable
  if (OUTLINE_ON && p_outline > 0.0f) {
    float edgefactor = dot(N, ray.direction);
    edgefactor *= edgefactor;
    edgefactor = 1.0f - edgefactor;
    edgefactor = 1.0f - powf(edgefactor, (1.0f - p_outlinewidth) * 32.0f);
    float outlinefactor = (1.0f - p_outline) + (edgefactor * p_outline);
    result *= outlinefactor;
  }

  result += make_float3(p_Ka); // white ambient contribution
  result += phongcol;          // add phong highlights

  // spawn reflection rays if necessary
  if (REFLECTION_ON && p_reflectivity > 0.0f) {
    // ray tree attenuation
    PerRayData_radiance new_prd;
    new_prd.importance = prd.importance * p_reflectivity;
    new_prd.depth = prd.depth + 1;

    // shoot a reflection ray
    if (new_prd.importance >= 0.001f && new_prd.depth <= max_depth) {
      float3 refl_dir = reflect(ray.direction, N);
#if defined(ORT_USE_RAY_STEP)
      Ray refl_ray = make_Ray(hit_point + ORT_RAY_STEP, refl_dir, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
#else
      Ray refl_ray = make_Ray(hit_point, refl_dir, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
#endif
      rtTrace(root_object, refl_ray, new_prd);
      result += p_reflectivity * new_prd.result;
    }
  }

  // spawn transmission rays if necessary
  float alpha = p_opacity;
  if (TRANSMISSION_ON && alpha < 0.999f ) {
    // Emulate Tachyon/Raster3D's angle-dependent surface opacity if enabled
    if (p_transmode) {
      alpha = 1.0f + cosf(3.1415926f * (1.0f-alpha) * dot(N, ray.direction));
      alpha = alpha*alpha * 0.25f;
    }

    result *= alpha; // scale down current lighting by opacity

    // shoot a transmission ray
    PerRayData_radiance new_prd;
    new_prd.importance = prd.importance * (1.0f - alpha);
    new_prd.result = scene_bg_color;
    new_prd.depth = prd.depth + 1;
    if (new_prd.importance >= 0.001f && new_prd.depth <= max_depth) {
#if defined(ORT_USE_RAY_STEP)
      // step the ray in the direction opposite the surface normal (going in)
      // rather than out, for transmission rays...
      Ray trans_ray = make_Ray(hit_point - ORT_RAY_STEP, ray.direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
#else
      Ray trans_ray = make_Ray(hit_point, ray.direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
#endif
      rtTrace(root_object, trans_ray, new_prd);
    }
    result += (1.0f - alpha) * new_prd.result;
  }

  // add depth cueing / fog if enabled
  if (FOG_ON && fogmod < 1.0f) {
    result = fog_color(fogmod, result);
  }

  prd.result = result; // pass the color back up the tree
}


// color state associated with meshes or other primitives that 
// don't provide per-vertex, or per-facet color data
rtDeclareVariable(float3, uniform_color, , );

// VMD material shading coefficients
rtDeclareVariable(float, Ka, , );
rtDeclareVariable(float, Kd, , );
rtDeclareVariable(float, Ks, , );
rtDeclareVariable(float, phong_exp, , );
rtDeclareVariable(float, Krefl, , );
rtDeclareVariable(float, opacity, , );
rtDeclareVariable(float, outline, , );
rtDeclareVariable(float, outlinewidth, , );
rtDeclareVariable(int, transmode, , );


// Any hit program for opaque objects
RT_PROGRAM void any_hit_opaque() {
  // this material is opaque, so it fully attenuates all shadow rays
  prd_shadow.attenuation = optix::make_float3(0.0f);
}


// Any hit program required for shadow filtering through transparent materials
RT_PROGRAM void any_hit_transmission() {
  // use a VERY simple shadow filtering scheme based on opacity
  prd_shadow.attenuation = make_float3(1.0f - opacity);

  // check to see if we've hit 100% shadow or not
  if (fmaxf(prd_shadow.attenuation) < 0.001f )
    rtTerminateRay();
  else
    rtIgnoreIntersection();
}


// normal calc routine needed only to simplify the macro to produce the
// complete combinatorial expansion of template-specialized 
// closest hit radiance functions 
static __inline__ __device__ float3 calc_ffworld_normal() {
  float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
  float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
  return faceforward(world_shading_normal, -ray.direction, world_geometric_normal);
}


// general-purpose any-hit program, with all template options enabled, 
// intended for shader debugging and comparison with the original
// Tachyon full_shade() code.
RT_PROGRAM void closest_hit_radiance_general() {
  shader_template<1, 1, 1, 1, 1, 1>(obj_color, calc_ffworld_normal(), 
                                    Ka, Kd, Ks, phong_exp, Krefl, opacity,
                                    outline, outlinewidth, transmode);
}


//
// Object and/or vertex/color/normal buffers...
//

// cylinder array buffers
rtBuffer<vmd_cylinder> cylinder_buffer;
rtBuffer<vmd_cylinder_color> cylinder_color_buffer;

// ring array buffer
rtBuffer<vmd_ring_color> ring_color_buffer;

// sphere array buffers
rtBuffer<vmd_sphere> sphere_buffer;
rtBuffer<vmd_sphere_color> sphere_color_buffer;

// triangle mesh buffers
rtBuffer<vmd_tricolor> tricolor_buffer;
rtBuffer<vmd_trimesh_c4u_n3b_v3f> trimesh_c4u_n3b_v3f_buffer;
rtBuffer<vmd_trimesh_n3f_v3f> trimesh_n3f_v3f_buffer;
rtBuffer<vmd_trimesh_n3b_v3f> trimesh_n3b_v3f_buffer;
rtBuffer<vmd_trimesh_v3f> trimesh_v3f_buffer;


//
// Cylinder array primitive
//
RT_PROGRAM void cylinder_array_intersect(int primIdx) {
  float3 start = cylinder_buffer[primIdx].start;
  float radius = cylinder_buffer[primIdx].radius;
  float3 axis = cylinder_buffer[primIdx].axis;

  float3 rc = ray.origin - start;
  float3 n = cross(ray.direction, axis);
  float ln = length(n);

  // check if ray is parallel to cylinder
  if (ln == 0.0f) {
    return; // ray is parallel, we missed or went through the "hole"
  } 
  n /= ln;
  float d = fabsf(dot(rc, n));

  // check for cylinder intersection
  if (d <= radius) {
    float3 O = cross(rc, axis);
    float t = -dot(O, n) / ln;
    O = cross(n, axis);
    O = normalize(O);
    float s = dot(ray.direction, O); 
    s = fabs(sqrtf(radius*radius - d*d) / s);
    float axlen = length(axis);
    float3 axis_u = normalize(axis);

    // test hit point against cylinder ends
    float tin = t - s;
    float3 hit = ray.origin + ray.direction * tin;
    float3 tmp2 = hit - start;
    float tmp = dot(tmp2, axis_u);
    if ((tmp > 0.0f) && (tmp < axlen)) {
      if (rtPotentialIntersection(tin)) {
        shading_normal = geometric_normal = normalize(hit - (tmp * axis_u + start));

        // uniform color for the entire object
        obj_color = uniform_color;
        rtReportIntersection(0);
      }
    }
    
    // continue with second test...
    float tout = t + s;
    hit = ray.origin + ray.direction * tout;
    tmp2 = hit - start;
    tmp = dot(tmp2, axis_u);
    if ((tmp > 0.0f) && (tmp < axlen)) {
      if (rtPotentialIntersection(tout)) {
        shading_normal = geometric_normal = normalize(hit - (tmp * axis_u + start));

        // uniform color for the entire object
        obj_color = uniform_color;
        rtReportIntersection(0);
      }
    }
  }
}


RT_PROGRAM void cylinder_array_bounds(int primIdx, float result[6]) {
  const float3 start = cylinder_buffer[primIdx].start;
  const float3 end = start + cylinder_buffer[primIdx].axis;
  const float3 rad = make_float3(cylinder_buffer[primIdx].radius);
  optix::Aabb* aabb = (optix::Aabb*)result;

  if (rad.x > 0.0f && !isinf(rad.x)) {
    aabb->m_min = fminf(start - rad, end - rad);
    aabb->m_max = fmaxf(start + rad, end + rad);
  } else {
    aabb->invalidate();
  }
}


//
// Color-per-cylinder array primitive
//
RT_PROGRAM void cylinder_array_color_intersect(int primIdx) {
  float3 start = cylinder_color_buffer[primIdx].start;
  float radius = cylinder_color_buffer[primIdx].radius;
  float3 axis = cylinder_color_buffer[primIdx].axis;

  float3 rc = ray.origin - start;
  float3 n = cross(ray.direction, axis);
  float ln = length(n);

  // check if ray is parallel to cylinder
  if (ln == 0.0f) {
    return; // ray is parallel, we missed or went through the "hole"
  } 
  n /= ln;
  float d = fabsf(dot(rc, n));

  // check for cylinder intersection
  if (d <= radius) {
    float3 O = cross(rc, axis);
    float t = -dot(O, n) / ln;
    O = cross(n, axis);
    O = normalize(O);
    float s = dot(ray.direction, O); 
    s = fabs(sqrtf(radius*radius - d*d) / s);
    float axlen = length(axis);
    float3 axis_u = normalize(axis);

    // test hit point against cylinder ends
    float tin = t - s;
    float3 hit = ray.origin + ray.direction * tin;
    float3 tmp2 = hit - start;
    float tmp = dot(tmp2, axis_u);
    if ((tmp > 0.0f) && (tmp < axlen)) {
      if (rtPotentialIntersection(tin)) {
        shading_normal = geometric_normal = normalize(hit - (tmp * axis_u + start));
        obj_color = cylinder_color_buffer[primIdx].color;
        rtReportIntersection(0);
      }
    }
    
    // continue with second test...
    float tout = t + s;
    hit = ray.origin + ray.direction * tout;
    tmp2 = hit - start;
    tmp = dot(tmp2, axis_u);
    if ((tmp > 0.0f) && (tmp < axlen)) {
      if (rtPotentialIntersection(tout)) {
        shading_normal = geometric_normal = normalize(hit - (tmp * axis_u + start));
        obj_color = cylinder_color_buffer[primIdx].color;
        rtReportIntersection(0);
      }
    }
  }
}


RT_PROGRAM void cylinder_array_color_bounds(int primIdx, float result[6]) {
  const float3 start = cylinder_color_buffer[primIdx].start;
  const float3 end = start + cylinder_color_buffer[primIdx].axis;
  const float3 rad = make_float3(cylinder_color_buffer[primIdx].radius);
  optix::Aabb* aabb = (optix::Aabb*)result;

  if (rad.x > 0.0f && !isinf(rad.x)) {
    aabb->m_min = fminf(start - rad, end - rad);
    aabb->m_max = fmaxf(start + rad, end + rad);
  } else {
    aabb->invalidate();
  }
}


//
// Ring array primitive
//
RT_PROGRAM void ring_array_color_intersect(int primIdx) {
  const float3 center = ring_color_buffer[primIdx].center;
  const float3 norm = ring_color_buffer[primIdx].norm;
  const float inrad = ring_color_buffer[primIdx].inrad;
  const float outrad = ring_color_buffer[primIdx].outrad;
  const float3 color = ring_color_buffer[primIdx].color;

  float d = -dot(center, norm); 
  float t = -(d + dot(norm, ray.origin));
  float td = dot(norm, ray.direction);
  if (td != 0.0f) {
    t /= td;
    if (t >= 0.0f) {
      float3 hit = ray.origin + t * ray.direction;
      float rd = length(hit - center);
      if ((rd > inrad) && (rd < outrad)) {
        if (rtPotentialIntersection(t)) {
          shading_normal = geometric_normal = norm;
          obj_color = color;
          rtReportIntersection(0);
        }
      }
    }
  }
}


RT_PROGRAM void ring_array_color_bounds(int primIdx, float result[6]) {
  const float3 center = ring_color_buffer[primIdx].center;
  const float3 rad = make_float3(ring_color_buffer[primIdx].outrad);
  optix::Aabb* aabb = (optix::Aabb*)result;

  if (rad.x > 0.0f && !isinf(rad.x)) {
    aabb->m_min = center - rad;
    aabb->m_max = center + rad;
  } else {
    aabb->invalidate();
  }
}



//
// Sphere array primitive
//
RT_PROGRAM void sphere_array_intersect(int primIdx) {
  float3 center = sphere_buffer[primIdx].center;
  float radius = sphere_buffer[primIdx].radius;

  float3 V = center - ray.origin;
  float b = dot(V, ray.direction);
  float disc = b*b + radius*radius - dot(V, V);
  if (disc > 0.0f) {
    disc = sqrtf(disc);

//#define FASTONESIDEDSPHERES 1
#if defined(FASTONESIDEDSPHERES)
    // only calculate the nearest intersection, for speed
    float t1 = b - disc;
    if (rtPotentialIntersection(t1)) {
      shading_normal = geometric_normal = (t1*ray.direction - V) / radius;
      obj_color = uniform_color; // uniform color for the entire object
      rtReportIntersection(0);
    }
#else
    float t2 = b + disc;
    if (rtPotentialIntersection(t2)) {
      shading_normal = geometric_normal = (t2*ray.direction - V) / radius;
      float3 offset = shading_normal * scene_epsilon;
      obj_color = uniform_color; // uniform color for the entire object
      rtReportIntersection(0);
    }

    float t1 = b - disc;
    if (rtPotentialIntersection(t1)) {
      shading_normal = geometric_normal = (t1*ray.direction - V) / radius;
      obj_color = uniform_color; // uniform color for the entire object
      rtReportIntersection(0);
    }
#endif
  }
}

RT_PROGRAM void sphere_array_bounds(int primIdx, float result[6]) {
  const float3 cen = sphere_buffer[primIdx].center;
  const float3 rad = make_float3(sphere_buffer[primIdx].radius);
  optix::Aabb* aabb = (optix::Aabb*)result;

  if (rad.x > 0.0f && !isinf(rad.x)) {
    aabb->m_min = cen - rad;
    aabb->m_max = cen + rad;
  } else {
    aabb->invalidate();
  }
}


//
// Color-per-sphere sphere array 
//
RT_PROGRAM void sphere_array_color_intersect(int primIdx) {
  float3 center = sphere_color_buffer[primIdx].center;
  float radius = sphere_color_buffer[primIdx].radius;

  float3 V = center - ray.origin;
  float b = dot(V, ray.direction);
  float disc = b*b + radius*radius - dot(V, V);
  if (disc > 0.0f) {
    disc = sqrtf(disc);

//#define FASTONESIDEDSPHERES 1
#if defined(FASTONESIDEDSPHERES)
    // only calculate the nearest intersection, for speed
    float t1 = b - disc;
    if (rtPotentialIntersection(t1)) {
      shading_normal = geometric_normal = (t1*ray.direction - V) / radius;
      obj_color = sphere_color_buffer[primIdx].color;
      rtReportIntersection(0);
    }
#else
    float t2 = b + disc;
    if (rtPotentialIntersection(t2)) {
      shading_normal = geometric_normal = (t2*ray.direction - V) / radius;
      obj_color = sphere_color_buffer[primIdx].color;
      rtReportIntersection(0);
    }

    float t1 = b - disc;
    if (rtPotentialIntersection(t1)) {
      shading_normal = geometric_normal = (t1*ray.direction - V) / radius;
      obj_color = sphere_color_buffer[primIdx].color;
      rtReportIntersection(0);
    }
#endif
  }
}

RT_PROGRAM void sphere_array_color_bounds(int primIdx, float result[6]) {
  const float3 cen = sphere_color_buffer[primIdx].center;
  const float3 rad = make_float3(sphere_color_buffer[primIdx].radius);
  optix::Aabb* aabb = (optix::Aabb*)result;

  if (rad.x > 0.0f && !isinf(rad.x)) {
    aabb->m_min = cen - rad;
    aabb->m_max = cen + rad;
  } else {
    aabb->invalidate();
  }
}


//
// Triangle list primitive - unstructured triangle soup
//


// inline device function for computing triangle bounding boxes
__device__ __inline__ void generic_tri_bounds(optix::Aabb *aabb,
                                              float3 v0, float3 v1, float3 v2) {
#if 1
  // conventional paranoid implementation that culls degenerate triangles
  float area = length(cross(v1-v0, v2-v0));
  if (area > 0.0f && !isinf(area)) {
    aabb->m_min = fminf(fminf(v0, v1), v2);
    aabb->m_max = fmaxf(fmaxf(v0, v1), v2);
  } else {
    aabb->invalidate();
  }
#else
  // don't cull any triangles, even if they might be degenerate
  aabb->m_min = fminf(fminf(v0, v1), v2);
  aabb->m_max = fmaxf(fmaxf(v0, v1), v2);
#endif
}



RT_PROGRAM void tricolor_intersect(int primIdx) {
  float3 v0 = tricolor_buffer[primIdx].v0;
  float3 v1 = tricolor_buffer[primIdx].v1;
  float3 v2 = tricolor_buffer[primIdx].v2;

  // Intersect ray with triangle
  float3 n;
  float t, beta, gamma;
  if (intersect_triangle(ray, v0, v1, v2, n, t, beta, gamma)) {
    if (rtPotentialIntersection(t)) {
      float3 n0 = tricolor_buffer[primIdx].n0;
      float3 n1 = tricolor_buffer[primIdx].n1;
      float3 n2 = tricolor_buffer[primIdx].n2;
      shading_normal = normalize(n1*beta + n2*gamma + n0*(1.0f-beta-gamma));
      geometric_normal = normalize(n);

      float3 c0 = tricolor_buffer[primIdx].c0;
      float3 c1 = tricolor_buffer[primIdx].c1;
      float3 c2 = tricolor_buffer[primIdx].c2;
      obj_color = c1*beta + c2*gamma + c0*(1.0f-beta-gamma);
      rtReportIntersection(0);
    }
  }
}

RT_PROGRAM void tricolor_bounds(int primIdx, float result[6]) {
  float3 v0 = tricolor_buffer[primIdx].v0;
  float3 v1 = tricolor_buffer[primIdx].v1;
  float3 v2 = tricolor_buffer[primIdx].v2;

  optix::Aabb *aabb = (optix::Aabb*)result;
  generic_tri_bounds(aabb, v0, v1, v2);
}



RT_PROGRAM void trimesh_c4u_n3b_v3f_intersect(int primIdx) {
  float3 v0 = trimesh_c4u_n3b_v3f_buffer[primIdx].v0;
  float3 v1 = trimesh_c4u_n3b_v3f_buffer[primIdx].v1;
  float3 v2 = trimesh_c4u_n3b_v3f_buffer[primIdx].v2;

  // Intersect ray with triangle
  float3 n;
  float t, beta, gamma;
  if (intersect_triangle(ray, v0, v1, v2, n, t, beta, gamma)) {
    if (rtPotentialIntersection(t)) {
      // conversion from GLbyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
      // float = (2c+1)/(2^8-1)
      const float ci2f = 1.0f / 255.0f;
      const float cn2f = 1.0f / 127.5f;

      float3 n0 = trimesh_c4u_n3b_v3f_buffer[primIdx].n0 * cn2f + ci2f;
      float3 n1 = trimesh_c4u_n3b_v3f_buffer[primIdx].n1 * cn2f + ci2f;
      float3 n2 = trimesh_c4u_n3b_v3f_buffer[primIdx].n2 * cn2f + ci2f;
      shading_normal = normalize(n1*beta + n2*gamma + n0*(1.0f-beta-gamma));
      geometric_normal = normalize(n);

      // conversion from GLubyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
      // float = c/(2^8-1)
      float3 c0 = trimesh_c4u_n3b_v3f_buffer[primIdx].c0 * ci2f;
      float3 c1 = trimesh_c4u_n3b_v3f_buffer[primIdx].c1 * ci2f;
      float3 c2 = trimesh_c4u_n3b_v3f_buffer[primIdx].c2 * ci2f;
      obj_color = c1*beta + c2*gamma + c0*(1.0f-beta-gamma);
      rtReportIntersection(0);
    }
  }
}

RT_PROGRAM void trimesh_c4u_n3b_v3f_bounds(int primIdx, float result[6]) {
  float3 v0 = trimesh_c4u_n3b_v3f_buffer[primIdx].v0;
  float3 v1 = trimesh_c4u_n3b_v3f_buffer[primIdx].v1;
  float3 v2 = trimesh_c4u_n3b_v3f_buffer[primIdx].v2;

  optix::Aabb *aabb = (optix::Aabb*)result;
  generic_tri_bounds(aabb, v0, v1, v2);
}



RT_PROGRAM void trimesh_n3f_v3f_intersect(int primIdx) {
  float3 v0 = trimesh_n3f_v3f_buffer[primIdx].v0;
  float3 v1 = trimesh_n3f_v3f_buffer[primIdx].v1;
  float3 v2 = trimesh_n3f_v3f_buffer[primIdx].v2;

  // Intersect ray with triangle
  float3 n;
  float t, beta, gamma;
  if (intersect_triangle(ray, v0, v1, v2, n, t, beta, gamma)) {
    if (rtPotentialIntersection(t)) {
      float3 n0 = trimesh_n3f_v3f_buffer[primIdx].n0;
      float3 n1 = trimesh_n3f_v3f_buffer[primIdx].n1;
      float3 n2 = trimesh_n3f_v3f_buffer[primIdx].n2;

      shading_normal = normalize(n1*beta + n2*gamma + n0*(1.0f-beta-gamma) );
      geometric_normal = normalize(n);

      // uniform color for the entire object
      obj_color = uniform_color;
      rtReportIntersection(0);
    }
  }
}

RT_PROGRAM void trimesh_n3f_v3f_bounds(int primIdx, float result[6]) {
  float3 v0 = trimesh_n3f_v3f_buffer[primIdx].v0;
  float3 v1 = trimesh_n3f_v3f_buffer[primIdx].v1;
  float3 v2 = trimesh_n3f_v3f_buffer[primIdx].v2;

  optix::Aabb *aabb = (optix::Aabb*)result;
  generic_tri_bounds(aabb, v0, v1, v2);
}



RT_PROGRAM void trimesh_n3b_v3f_intersect(int primIdx) {
  float3 v0 = trimesh_n3b_v3f_buffer[primIdx].v0;
  float3 v1 = trimesh_n3b_v3f_buffer[primIdx].v1;
  float3 v2 = trimesh_n3b_v3f_buffer[primIdx].v2;

  // Intersect ray with triangle
  float3 n;
  float t, beta, gamma;
  if (intersect_triangle(ray, v0, v1, v2, n, t, beta, gamma)) {
    if (rtPotentialIntersection(t)) {
      // conversion from GLbyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
      // float = (2c+1)/(2^8-1)
      const float ci2f = 1.0f / 255.0f;
      const float cn2f = 1.0f / 127.5f;

      float3 n0 = trimesh_n3b_v3f_buffer[primIdx].n0 * cn2f + ci2f;
      float3 n1 = trimesh_n3b_v3f_buffer[primIdx].n1 * cn2f + ci2f;
      float3 n2 = trimesh_n3b_v3f_buffer[primIdx].n2 * cn2f + ci2f;
      shading_normal = normalize(n1*beta + n2*gamma + n0*(1.0f-beta-gamma) );
      geometric_normal = normalize(n);

      // uniform color for the entire object
      obj_color = uniform_color;
      rtReportIntersection(0);
    }
  }
}

RT_PROGRAM void trimesh_n3b_v3f_bounds(int primIdx, float result[6]) {
  float3 v0 = trimesh_n3b_v3f_buffer[primIdx].v0;
  float3 v1 = trimesh_n3b_v3f_buffer[primIdx].v1;
  float3 v2 = trimesh_n3b_v3f_buffer[primIdx].v2;

  optix::Aabb *aabb = (optix::Aabb*)result;
  generic_tri_bounds(aabb, v0, v1, v2);
}


RT_PROGRAM void trimesh_v3f_intersect(int primIdx) {
  float3 v0 = trimesh_v3f_buffer[primIdx].v0;
  float3 v1 = trimesh_v3f_buffer[primIdx].v1;
  float3 v2 = trimesh_v3f_buffer[primIdx].v2;

  // Intersect ray with triangle
  float3 n;
  float t, beta, gamma;
  if (intersect_triangle(ray, v0, v1, v2, n, t, beta, gamma)) {
    if (rtPotentialIntersection(t)) {
      shading_normal = geometric_normal = normalize(n);

      // uniform color for the entire object
      obj_color = uniform_color;
      rtReportIntersection(0);
    }
  }
}

RT_PROGRAM void trimesh_v3f_bounds(int primIdx, float result[6]) {
  float3 v0 = trimesh_v3f_buffer[primIdx].v0;
  float3 v1 = trimesh_v3f_buffer[primIdx].v1;
  float3 v2 = trimesh_v3f_buffer[primIdx].v2;

  optix::Aabb *aabb = (optix::Aabb*)result;
  generic_tri_bounds(aabb, v0, v1, v2);
}


//
// The code below this point generates the combinatorial expansion of
// the series of shading feature on/off flags.  This produces a complete
// set of template-specialized shader variants that handle every 
// performance-beneficial combination of scene-wide and material-specific 
// shading features, which are then linked to the OptiX scene graph 
// material nodes.
//

#define FOG_on            1
#define FOG_off           0
#define SHADOWS_on        1
#define SHADOWS_off       0
#define AO_on             1
#define AO_off            0
#define OUTLINE_on        1
#define OUTLINE_off       0
#define REFL_on           1
#define REFL_off          0
#define TRANS_on          1
#define TRANS_off         0

#define DEFINE_CLOSEST_HIT( mfog, mshad, mao, moutl, mrefl, mtrans )     \
  RT_PROGRAM void                                                        \
  closest_hit_radiance_FOG_##mfog##_SHADOWS_##mshad##_AO_##mao##_OUTLINE_##moutl##_REFL_##mrefl##_TRANS_##mtrans() { \
                                                                         \
    shader_template<FOG_##mfog,                                          \
                    SHADOWS_##mshad,                                     \
                    AO_##mao,                                            \
                    OUTLINE_##moutl,                                     \
                    REFL_##mrefl,                                        \
                    TRANS_##mtrans >                                     \
                    (obj_color, calc_ffworld_normal(),                   \
                     Ka, Kd, Ks, phong_exp, Krefl,                       \
                     opacity, outline, outlinewidth, transmode);         \
  }


//
// Generate all of the 2^6 parameter combinations as 
// template-specialized shaders...
//
DEFINE_CLOSEST_HIT(  on,  on,  on,  on,  on,  on )
DEFINE_CLOSEST_HIT(  on,  on,  on,  on,  on, off )

DEFINE_CLOSEST_HIT(  on,  on,  on,  on, off,  on )
DEFINE_CLOSEST_HIT(  on,  on,  on,  on, off, off )

DEFINE_CLOSEST_HIT(  on,  on,  on, off,  on,  on )
DEFINE_CLOSEST_HIT(  on,  on,  on, off,  on, off )

DEFINE_CLOSEST_HIT(  on,  on,  on, off, off,  on )
DEFINE_CLOSEST_HIT(  on,  on,  on, off, off, off )

DEFINE_CLOSEST_HIT(  on,  on, off,  on,  on,  on )
DEFINE_CLOSEST_HIT(  on,  on, off,  on,  on, off )

DEFINE_CLOSEST_HIT(  on,  on, off,  on, off,  on )
DEFINE_CLOSEST_HIT(  on,  on, off,  on, off, off )

DEFINE_CLOSEST_HIT(  on,  on, off, off,  on,  on )
DEFINE_CLOSEST_HIT(  on,  on, off, off,  on, off )

DEFINE_CLOSEST_HIT(  on,  on, off, off, off,  on )
DEFINE_CLOSEST_HIT(  on,  on, off, off, off, off )


DEFINE_CLOSEST_HIT(  on, off,  on,  on,  on,  on )
DEFINE_CLOSEST_HIT(  on, off,  on,  on,  on, off )

DEFINE_CLOSEST_HIT(  on, off,  on,  on, off,  on )
DEFINE_CLOSEST_HIT(  on, off,  on,  on, off, off )

DEFINE_CLOSEST_HIT(  on, off,  on, off,  on,  on )
DEFINE_CLOSEST_HIT(  on, off,  on, off,  on, off )

DEFINE_CLOSEST_HIT(  on, off,  on, off, off,  on )
DEFINE_CLOSEST_HIT(  on, off,  on, off, off, off )

DEFINE_CLOSEST_HIT(  on, off, off,  on,  on,  on )
DEFINE_CLOSEST_HIT(  on, off, off,  on,  on, off )

DEFINE_CLOSEST_HIT(  on, off, off,  on, off,  on )
DEFINE_CLOSEST_HIT(  on, off, off,  on, off, off )

DEFINE_CLOSEST_HIT(  on, off, off, off,  on,  on )
DEFINE_CLOSEST_HIT(  on, off, off, off,  on, off )

DEFINE_CLOSEST_HIT(  on, off, off, off, off,  on )
DEFINE_CLOSEST_HIT(  on, off, off, off, off, off )



DEFINE_CLOSEST_HIT( off,  on,  on,  on,  on,  on )
DEFINE_CLOSEST_HIT( off,  on,  on,  on,  on, off )

DEFINE_CLOSEST_HIT( off,  on,  on,  on, off,  on )
DEFINE_CLOSEST_HIT( off,  on,  on,  on, off, off )

DEFINE_CLOSEST_HIT( off,  on,  on, off,  on,  on )
DEFINE_CLOSEST_HIT( off,  on,  on, off,  on, off )

DEFINE_CLOSEST_HIT( off,  on,  on, off, off,  on )
DEFINE_CLOSEST_HIT( off,  on,  on, off, off, off )

DEFINE_CLOSEST_HIT( off,  on, off,  on,  on,  on )
DEFINE_CLOSEST_HIT( off,  on, off,  on,  on, off )

DEFINE_CLOSEST_HIT( off,  on, off,  on, off,  on )
DEFINE_CLOSEST_HIT( off,  on, off,  on, off, off )

DEFINE_CLOSEST_HIT( off,  on, off, off,  on,  on )
DEFINE_CLOSEST_HIT( off,  on, off, off,  on, off )

DEFINE_CLOSEST_HIT( off,  on, off, off, off,  on )
DEFINE_CLOSEST_HIT( off,  on, off, off, off, off )


DEFINE_CLOSEST_HIT( off, off,  on,  on,  on,  on )
DEFINE_CLOSEST_HIT( off, off,  on,  on,  on, off )

DEFINE_CLOSEST_HIT( off, off,  on,  on, off,  on )
DEFINE_CLOSEST_HIT( off, off,  on,  on, off, off )

DEFINE_CLOSEST_HIT( off, off,  on, off,  on,  on )
DEFINE_CLOSEST_HIT( off, off,  on, off,  on, off )

DEFINE_CLOSEST_HIT( off, off,  on, off, off,  on )
DEFINE_CLOSEST_HIT( off, off,  on, off, off, off )

DEFINE_CLOSEST_HIT( off, off, off,  on,  on,  on )
DEFINE_CLOSEST_HIT( off, off, off,  on,  on, off )

DEFINE_CLOSEST_HIT( off, off, off,  on, off,  on )
DEFINE_CLOSEST_HIT( off, off, off,  on, off, off )

DEFINE_CLOSEST_HIT( off, off, off, off,  on,  on )
DEFINE_CLOSEST_HIT( off, off, off, off,  on, off )

DEFINE_CLOSEST_HIT( off, off, off, off, off,  on )
DEFINE_CLOSEST_HIT( off, off, off, off, off, off )

#undef FOG_on
#undef FOG_off
#undef SHADOWS_on
#undef SHADOWS_off
#undef AO_on
#undef AO_off
#undef OUTLINE_on
#undef OUTLINE_off
#undef REFL_on       
#undef REFL_off           
#undef TRANS_on         
#undef TRANS_off          
#undef DEFINE_CLOSEST_HIT




