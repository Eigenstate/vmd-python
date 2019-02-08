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
*      $RCSfile: OptiXShaders.cu,v $
*      $Author: johns $      $Locker:  $               $State: Exp $
*      $Revision: 1.163 $         $Date: 2019/01/17 21:38:55 $
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
#define ORT_USE_RAY_STEP       1
#define ORT_TRANS_USE_INCIDENT 1
#define ORT_RAY_STEP           N*scene_epsilon*4.0f
#define ORT_RAY_STEP2          ray.direction*scene_epsilon*4.0f

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


// Macros to enable particular ray-geometry intersection variants that
// optimize for speed, or some combination of speed and accuracy
#define ORT_USE_SPHERES_HEARNBAKER 1


using namespace optix;

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

#if defined(ORT_RAYSTATS)
// input/output ray statistics buffer
rtBuffer<uint4, 2> raystats1_buffer; // x=prim, y=shad-dir, z=shad-ao, w=miss
rtBuffer<uint4, 2> raystats2_buffer; // x=trans, y=trans-skip, z=?, w=refl
#endif

// epsilon value to use to avoid self-intersection 
rtDeclareVariable(float, scene_epsilon, , );

// max ray recursion depth
rtDeclareVariable(int, max_depth, , );

// max number of transparent surfaces (max_trans <= max_depth)
rtDeclareVariable(int, max_trans, , );

// XXX global interpolation coordinate for experimental  animated 
// representations that loop over some fixed sequence of motion 
// over the domain [0:1].
rtDeclareVariable(float, anim_interp, , );

// shadow rendering mode
rtDeclareVariable(int, shadows_enabled, , );
rtDeclareVariable(int, aa_samples, , );

// ambient occlusion sample counts and scaling factors
rtDeclareVariable(int, ao_samples, , );
rtDeclareVariable(float, ao_ambient, , );
rtDeclareVariable(float, ao_direct, , );
rtDeclareVariable(float, ao_maxdist, , ); ///< max AO occluder distance...

// background color and/or background gradient
rtDeclareVariable(float3, scene_bg_color, , );
rtDeclareVariable(float3, scene_bg_color_grad_top, , );
rtDeclareVariable(float3, scene_bg_color_grad_bot, , );
rtDeclareVariable(float3, scene_gradient, , );
rtDeclareVariable(float, scene_gradient_topval, , );
rtDeclareVariable(float, scene_gradient_botval, , );
rtDeclareVariable(float, scene_gradient_invrange, , );

// VR HMD fade+clipping plane/sphere
rtDeclareVariable(int, clipview_mode, , );
rtDeclareVariable(float, clipview_start, , );
rtDeclareVariable(float, clipview_end, , );

// VR HMD headlight
rtDeclareVariable(int, headlight_mode, , );

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

//
// Classic API
//
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
// object color assigned at intersection time
rtDeclareVariable(float3, obj_color, attribute obj_color, );

#if defined(ORT_USERTXAPIS)
//
// OptiX RTX hardware triangle API
//
rtDeclareVariable(float2, barycentrics, attribute rtTriangleBarycentrics, );
rtBuffer<uint4> normalBuffer; // packed normals: ng [n0 n1 n2]
rtBuffer<uchar4> colorBuffer; // unsigned char color representation
rtDeclareVariable(int, has_vertex_normals, , );
rtDeclareVariable(int, has_vertex_colors, , );
#endif


struct PerRayData_radiance {
  float3 result;     // final shaded surface color
  float alpha;       // alpha value to back-propagate to framebuffer
  float importance;  // importance of recursive ray tree
  int depth;         // current recursion depth
  int transcnt;      // transmission ray surface count/depth
};

struct PerRayData_shadow {
  float3 attenuation;
};

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_radiance, prd, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );

// list of directional and positional lights
#if defined(VMDOPTIX_LIGHTUSEROBJS)
rtDeclareVariable(DirectionalLightList, dir_light_list, , );
rtDeclareVariable(PositionalLightList, pos_light_list, , );
#else
rtBuffer<DirectionalLight> dir_lights;
rtBuffer<DirectionalLight> pos_lights;
#endif

//
// convert float3 rgb data to uchar4 with alpha channel set to 255.
//
static __device__ __inline__ uchar4 make_color_rgb4u(const float3& c) {
  return make_uchar4(static_cast<unsigned char>(__saturatef(c.x)*255.99f),  
                     static_cast<unsigned char>(__saturatef(c.y)*255.99f),  
                     static_cast<unsigned char>(__saturatef(c.z)*255.99f),  
                     255u);
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
  prd_radiance.alpha = 0.0f; // alpha of background is 0.0f;
#if defined(ORT_RAYSTATS)
  raystats1_buffer[launch_index].w++; // increment miss counter
#endif
}


// Miss program for gradient background with perspective projection,
// adapted from Tachyon
// Fog overrides the background color if we're using
// Tachyon radial fog, but not for OpenGL style fog.
RT_PROGRAM void miss_gradient_bg_sky_sphere() {
  float IdotG = dot(ray.direction, scene_gradient);
  float val = (IdotG - scene_gradient_botval) * scene_gradient_invrange;
  val = __saturatef(val);
  float3 col = val * scene_bg_color_grad_top + 
               (1.0f - val) * scene_bg_color_grad_bot; 
  prd_radiance.result = col;
  prd_radiance.alpha = 0.0f; // alpha of background is 0.0f;
#if defined(ORT_RAYSTATS)
  raystats1_buffer[launch_index].w++; // increment miss counter
#endif
}


// Miss program for gradient background with orthographic projection,
// adapted from Tachyon
// Fog overrides the background color if we're using
// Tachyon radial fog, but not for OpenGL style fog.
RT_PROGRAM void miss_gradient_bg_sky_plane() {
  float IdotG = dot(ray.origin, scene_gradient);
  float val = (IdotG - scene_gradient_botval) * scene_gradient_invrange;
  val = __saturatef(val);
  float3 col = val * scene_bg_color_grad_top + 
               (1.0f - val) * scene_bg_color_grad_bot; 
  prd_radiance.result = col;
  prd_radiance.alpha = 0.0f; // alpha of background is 0.0f;
#if defined(ORT_RAYSTATS)
  raystats1_buffer[launch_index].w++; // increment miss counter
#endif
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
// Device functions for clipping rays by geometric primitives
// 

// fade_start: onset of fading 
//   fade_end: fully transparent, begin clipping of geometry
__device__ void sphere_fade_and_clip(const float3 &hit_point, 
                                     const float3 &cam_pos,
                                     float fade_start, float fade_end,
                                     float &alpha) {
  float camdist = length(hit_point - cam_pos);

  // we can omit the distance test since alpha modulation value is clamped
  // if (1 || camdist < fade_start) {
    float fade_len = fade_start - fade_end;
    alpha *= __saturatef((camdist - fade_start) / fade_len);
  // }
}


__device__ void ray_sphere_clip_interval(const optix::Ray &ray, float3 center,
                                         float rad, float2 &tinterval) {
  float3 V = center - ray.origin;
  float b = dot(V, ray.direction);
  float disc = b*b + rad*rad - dot(V, V);

  // if the discriminant is positive, the ray hits...
  if (disc > 0.0f) {
    disc = sqrtf(disc);
    tinterval.x = b-disc;
    tinterval.y = b+disc;
  } else {
    tinterval.x = -RT_DEFAULT_MAX; 
    tinterval.y =  RT_DEFAULT_MAX; 
  }
}


__device__ void clip_ray_by_plane(optix::Ray &ray, const float4 plane) {
  float3 n = make_float3(plane);                                              
  float dt = dot(ray.direction, n);                                            
  float t = (-plane.w - dot(n, ray.origin))/dt;                                 
  if(t > ray.tmin && t < ray.tmax) {                                          
    if (dt <= 0) {                                                              
      ray.tmax = t;                                                             
    } else {                                                                    
      ray.tmin = t;                                                             
    }                                                                           
  } else {                                                                      
    // ray interval lies completely on one side of the plane.  Test one point.
    float3 p = ray.origin + ray.tmin * ray.direction;                         
    if (dot(make_float4(p, 1.0f), plane) < 0) {
      ray.tmin = ray.tmax = RT_DEFAULT_MAX; // cull geometry
    }                                                                         
  }                                                                             
}               


//
// Clear the raystats buffers to zeros
//
#if defined(ORT_RAYSTATS)
RT_PROGRAM void clear_raystats_buffers() {
  raystats1_buffer[launch_index] = make_uint4(0, 0, 0, 0); // clear ray counters to zero
  raystats2_buffer[launch_index] = make_uint4(0, 0, 0, 0); // clear ray counters to zero
}
#endif


//
// Clear the accumulation buffer to zeros
//
RT_PROGRAM void clear_accumulation_buffer() {
  accumulation_buffer[launch_index] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
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
static void __inline__ __device__ accumulate_color(float3 &col, 
                                                   float alpha = 1.0f) {
#if defined(VMDOPTIX_PROGRESSIVEAPI)
  if (progressive_enabled) {
    col *= accumulation_normalization_factor;
    alpha *= accumulation_normalization_factor;

#if OPTIX_VERSION < 3080
    // XXX prior to OptiX 3.8, a hard-coded gamma correction was required
    // VCA gamma correction workaround, changes gamma 2.2 back to gamma 1.0
    float invgamma = 1.0f / 0.4545f;
    col.x = powf(col.x, invgamma);
    col.y = powf(col.y, invgamma);
    col.z = powf(col.z, invgamma);
#endif

    // for optix-vca progressive mode accumulation is handled in server code
    accumulation_buffer[launch_index]  = make_float4(col, alpha);
  } else {
    // For batch mode we accumulate ourselves
    accumulation_buffer[launch_index] += make_float4(col, alpha);
  }
#else
  // For batch mode we accumulate ourselves
  accumulation_buffer[launch_index] += make_float4(col, alpha);
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
  col.x = __saturatef(pixel_time);

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
// CUDA device function for computing the new ray origin
// and ray direction, given the radius of the circle of confusion disc,
// and an orthonormal basis for each ray.
//
static __device__ __inline__
void dof_ray(const float3 &ray_origin_orig, float3 &ray_origin, 
             const float3 &ray_direction_orig, float3 &ray_direction,
             unsigned int &randseed, const float3 &up, const float3 &right) {
  float3 focuspoint = ray_origin_orig + ray_direction_orig * cam_dof_focal_dist;
  float2 dofjxy;
  jitter_disc2f(randseed, dofjxy, cam_dof_aperture_rad);
  ray_origin = ray_origin_orig + dofjxy.x*right + dofjxy.y*up;
  ray_direction = normalize(focuspoint - ray_origin);
}


//
// 360-degree stereoscopic cubemap image format for use with
// Oculus, Google Cardboard, and similar VR headsets
//
template<int STEREO_ON, int DOF_ON> 
static __device__ __inline__ 
void vmd_camera_cubemap_general() {
#if defined(ORT_TIME_COLORING)
  clock_t t0 = clock(); // start per-pixel RT timer
#endif

  // compute which cubemap face we're drawing by the X index.
  uint facesz = launch_dim.y; // square cube faces, equal to image height
  uint face = (launch_index.x / facesz) % 6;
  uint2 face_idx = make_uint2(launch_index.x % facesz, launch_index.y);

  // For the OTOY ORBX viewer, Oculus VR software, and some of the 
  // related apps, the cubemap image is stored with the X axis oriented
  // such that when viewed as a 2-D image, they are all mirror images.
  // The mirrored left-right orientation used here corresponds to what is
  // seen standing outside the cube, whereas the ray tracer shoots
  // rays from the inside, so we flip the X-axis pixel storage order.
  // The top face of the cubemap has both the left-right and top-bottom
  // orientation flipped also.
  // Set per-face orthonormal basis for camera
  float3 face_U, face_V, face_W;
  switch (face) {
    case 0: // back face
      face_U =  cam_U;
      face_V =  cam_V;
      face_W = -cam_W;
      break;

    case 1: // front face
      face_U =  -cam_U;
      face_V =  cam_V;
      face_W =  cam_W;
      break;

    case 2: // top face
      face_U = -cam_W;
      face_V =  cam_U;
      face_W =  cam_V;
      break;

    case 3: // bottom face
      face_U = -cam_W;
      face_V = -cam_U;
      face_W = -cam_V;
      break;

    case 4: // left face
      face_U = -cam_W;
      face_V =  cam_V;
      face_W = -cam_U;
      break;

    case 5: // right face
      face_U =  cam_W;
      face_V =  cam_V;
      face_W =  cam_U;
      break;
  }

  // Stereoscopic rendering is provided by rendering in a side-by-side
  // format with the left eye image into the left half of a double-wide
  // framebuffer, and the right eye into the right half.  The subsequent 
  // OpenGL drawing code can trivially unpack and draw the two images 
  // into an efficient cubemap texture.
  uint viewport_sz_x, viewport_idx_x;
  float eyeshift;
  if (STEREO_ON) {
    // render into a double-wide framebuffer when stereo is enabled
    viewport_sz_x = launch_dim.x >> 1;
    if (launch_index.x >= viewport_sz_x) {
      // right image
      viewport_idx_x = launch_index.x - viewport_sz_x;
      eyeshift =  0.5f * cam_stereo_eyesep;
    } else {
      // left image
      viewport_idx_x = launch_index.x;
      eyeshift = -0.5f * cam_stereo_eyesep;
    }
  } else {
    // render into a normal size framebuffer if stereo is not enabled
    viewport_sz_x = launch_dim.x;
    viewport_idx_x = launch_index.x;
    eyeshift = 0.0f;
  }

  // 
  // general primary ray calculations, locked to 90-degree FoV per face...
  //
  float facescale = 1.0f / facesz;
  float2 d = make_float2(face_idx.x, face_idx.y) * facescale * 2.f - 1.0f; // center of pixel in image plane

  unsigned int randseed = tea<4>(launch_dim.x*(launch_index.y)+viewport_idx_x, subframe_count());

  float3 col = make_float3(0.0f);
  for (int s=0; s<aa_samples; s++) {
    float2 jxy;  
    jitter_offset2f(randseed, jxy);
    jxy = jxy * facescale * 2.f + d;
    float3 ray_direction = normalize(jxy.x*face_U + jxy.y*face_V + face_W);

    float3 ray_origin = cam_pos;
    if (STEREO_ON) {
      ray_origin += eyeshift * cross(ray_direction, cam_V);
    }

    // compute new ray origin and ray direction
    if (DOF_ON) {
      dof_ray(ray_origin, ray_origin, ray_direction, ray_direction,
              randseed, face_V, face_U);
    }

    // trace the new ray...
    PerRayData_radiance prd;
    prd.importance = 1.f;
    prd.depth = 0;
    prd.transcnt = max_trans;
    optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
    rtTrace(root_object, ray, prd);
#if defined(ORT_RAYSTATS)
    raystats1_buffer[launch_index].x++; // increment primary ray counter
#endif
    col += prd.result; 
  }

#if defined(ORT_TIME_COLORING)
  accumulate_time_coloring(col, t0);
#else
  accumulate_color(col);
#endif
}

RT_PROGRAM void vmd_camera_cubemap() {
  vmd_camera_cubemap_general<0, 0>();
}

RT_PROGRAM void vmd_camera_cubemap_dof() {
  vmd_camera_cubemap_general<0, 1>();
}

RT_PROGRAM void vmd_camera_cubemap_stereo() {
  vmd_camera_cubemap_general<1, 0>();
}

RT_PROGRAM void vmd_camera_cubemap_stereo_dof() {
  vmd_camera_cubemap_general<1, 1>();
}



static __device__ __inline__
void dof_ray2(const float3 &ray_org_orig, float3 &ray_org,
             const float3 &ray_dir_orig, float3 &ray_dir,
             const float3 &up, const float3 &right,
             unsigned int &randseed) {
  float3 focuspoint = ray_org_orig +
                      (ray_dir_orig * cam_dof_focal_dist);
  float2 dofjxy;
  jitter_disc2f(randseed, dofjxy, cam_dof_aperture_rad);
  ray_org = ray_org_orig + dofjxy.x*right + dofjxy.y*up;
  ray_dir = normalize(focuspoint - ray_org);
}



static __host__ __device__ __inline__
float3 eyeshift2(float3 ray_origin,  // original non-stereo eye origin
                float eyesep,       // interocular dist, world coords
                int whicheye,       // left/right eye flag
                float3 DcrossQ) {   // ray dir x audience "up" dir
  float shift = 0.0;
  switch (whicheye) {
    case -1: 
      shift = -0.5f * eyesep; // shift ray origin left
      break;

    case 1:
      shift = 0.5f * eyesep; // shift ray origin right
      break; 
             
    case 0:
   default:  
      shift = 0.0; // monoscopic projection
      break;
  }
  
  return ray_origin + shift * DcrossQ;
} 

static __device__ __inline__
int dome_ray2(float fov,            // FoV in radians
             float2 vp_sz,         // viewport size
             float2 i,             // pixel/point in image plane
             float3 &raydir,       // returned ray direction
             float3 &updir,        // up, aligned w/ longitude line
             float3 &rightdir) {   // right, aligned w/ latitude line
  float thetamax = 0.5f * fov;     // half-FoV in radians
  float2 radperpix = fov / vp_sz;  // calc radians/pixel in X/Y
  float2 m = vp_sz * 0.5f;         // calc viewport center/midpoint
  float2 p = (i - m) * radperpix;  // calc azimuth, theta components
  float theta = hypotf(p.x, p.y);  // hypotf() ensures best accuracy
  if (theta < thetamax) {
    if (theta == 0) {
      // At the dome center, azimuth is undefined and we must avoid
      // division by zero, so we set the ray direction to the zenith
      raydir = make_float3(0, 0, 1);
      updir = make_float3(0, 1, 0);
      rightdir = make_float3(1, 0, 0);
    } else {
      // Normal case, calc+combine azimuth and elevation components
      float sintheta, costheta;
      sincosf(theta, &sintheta, &costheta);
      raydir    = make_float3(sintheta * p.x / theta,
                              sintheta * p.y / theta,
                              costheta);
      updir     = make_float3(-costheta * p.x / theta,
                              -costheta * p.y / theta,
                              sintheta);
      rightdir  = make_float3(p.y / theta, -p.x / theta, 0);
    }

    return 1; // point in image plane is within FoV
  }

  raydir = make_float3(0, 0, 0); // outside of FoV
  updir = rightdir = raydir;
  return 0; // point in image plane is outside FoV
}



//
// Camera ray generation code for planetarium dome display
// Generates a fisheye style frame with ~180 degree FoV
// 
static __device__ __inline__
int dome_ray_dir(float fov,          // FoV in radians
                 float2 vp_sz,       // viewport size
                 float2 i,           // pixel/point in image plane
                 float3 &raydir,     // returned ray direction
                 float3 &updir,      // up, aligned w/ vertical longitude line
                 float3 &rightdir) { // right, aligned w/ horiz latitude line
  float thetamax = 0.5f * fov;       // half-FoV in radians, beyond is black
  float2 radperpix = fov / vp_sz;    // calc radians/pixel factors in X/Y
  float2 m = vp_sz * 0.5f;           // calc viewport center/midpoint
  float2 p = (i - m) * radperpix;    // calc azimuth and theta components
  float theta = hypotf(p.x, p.y);    // hypotf() ensures best accuracy
  if (theta < thetamax) {
    if (theta == 0) {
      // At the center of the dome, azimuth is undefined and we must avoid
      // division by zero, so we set the ray direction to the zenith
      raydir = make_float3(0, 0, 1);
      updir = make_float3(0, 1, 0);
      rightdir = make_float3(1, 0, 0);
    } else {
      // Normal case, calc+combine azimuth and elevation components
      float sintheta, costheta;
      sincosf(theta, &sintheta, &costheta);
      raydir    = make_float3(sintheta * p.x / theta,
                              sintheta * p.y / theta,
                              costheta);
      updir     = make_float3(-costheta * p.x / theta,
                              -costheta * p.y / theta,
                              sintheta);
      rightdir  = make_float3(p.y / theta, -p.x / theta, 0);
    }

    return 1; // point in image plane is within FoV
  }

  raydir = make_float3(0, 0, 0); // outside of FoV
  updir = rightdir = raydir;
  return 0; // point in image plane is outside FoV
}


template<int STEREO_ON, int DOF_ON>
static __device__ __inline__
void vmd_camera_dome_general() {
  uint viewport_sz_y, viewport_idx_y;
  float eyeshift;
  if (STEREO_ON) {
    // render into a double-high framebuffer when stereo is enabled
    viewport_sz_y = launch_dim.y >> 1;
    if (launch_index.y >= viewport_sz_y) {
      // left image
      viewport_idx_y = launch_index.y - viewport_sz_y;
      eyeshift = -0.5f * cam_stereo_eyesep;
    } else {
      // right image
      viewport_idx_y = launch_index.y;
      eyeshift =  0.5f * cam_stereo_eyesep;
    }
  } else {
    // render into a normal size framebuffer if stereo is not enabled
    viewport_sz_y = launch_dim.y;
    viewport_idx_y = launch_index.y;
    eyeshift = 0.0f;
  }

  float3 col = make_float3(0.0f);
  float alpha = 0.0f;

  float2 viewport_idx = make_float2(launch_index.x, viewport_idx_y);
  float fov = 3.1415926f; // 180 degrees
  float3 ray_direction, up_direction, right_direction;
  float3 ray_origin = cam_pos;

  unsigned int randseed = tea<4>(launch_dim.x*(launch_index.y)+launch_index.x, subframe_count());

#if 1
  if (!dome_ray2(fov, 
                    make_float2(launch_dim.x, viewport_sz_y), viewport_idx, 
                    ray_direction, up_direction, right_direction)) {
#else
  if (!dome_ray_dir(fov, 
                    make_float2(launch_dim.x, viewport_sz_y), viewport_idx, 
                    ray_direction, up_direction, right_direction)) {
#endif
    col = make_float3(0, 0, 0);
    accumulate_color(col, 1.0f);
    return;
  }

  ray_direction = cam_U*ray_direction.x + 
                  cam_V*ray_direction.y + 
                  cam_W*ray_direction.z;

  up_direction = cam_U*up_direction.x + 
                  cam_V*up_direction.y + 
                  cam_W*up_direction.z;

  right_direction = cam_U*right_direction.x + 
                    cam_V*right_direction.y + 
                    cam_W*right_direction.z;
  
  if (STEREO_ON) {
#if 1
    // normalize, or not, if we want to avoid backward stereo at the pole
    ray_origin = eyeshift2(ray_origin, cam_stereo_eyesep, 0, 
                          cross(ray_direction, cam_W));
#else
    // normalize, or not, if we want to avoid backward stereo at the pole
    ray_origin += eyeshift * cross(ray_direction, cam_W);
#endif
  }

  if (DOF_ON) {
#if 1
    dof_ray2(ray_origin, ray_origin, ray_direction, ray_direction,
             up_direction, right_direction, randseed);
#else
    dof_ray(ray_origin, ray_origin, ray_direction, ray_direction,
            randseed, up_direction, right_direction);
#endif
  }

  // trace the new ray...
  PerRayData_radiance prd;
  prd.importance = 1.f;
  prd.alpha = 1.f;
  prd.depth = 0;
  prd.transcnt = max_trans;
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
  rtTrace(root_object, ray, prd);
#if defined(ORT_RAYSTATS)
  raystats1_buffer[launch_index].x++; // increment primary ray counter
#endif
  col += prd.result;
  alpha += prd.alpha;

  accumulate_color(col, alpha);
}


template<int STEREO_ON, int DOF_ON>
static __device__ __inline__
void vmd_camera_dome_general_orig() {
#if defined(ORT_TIME_COLORING)
  clock_t t0 = clock(); // start per-pixel RT timer
#endif

  // Stereoscopic rendering is provided by rendering in an over/under
  // format with the left eye image into the top half of a double-high 
  // framebuffer, and the right eye into the lower half.  The subsequent 
  // OpenGL drawing code can trivially unpack and draw the two images 
  // with simple pointer offset arithmetic.
  uint viewport_sz_y, viewport_idx_y;
  float eyeshift;
  if (STEREO_ON) {
    // render into a double-high framebuffer when stereo is enabled
    viewport_sz_y = launch_dim.y >> 1;
    if (launch_index.y >= viewport_sz_y) {
      // left image
      viewport_idx_y = launch_index.y - viewport_sz_y;
      eyeshift = -0.5f * cam_stereo_eyesep;
    } else {
      // right image
      viewport_idx_y = launch_index.y;
      eyeshift =  0.5f * cam_stereo_eyesep;
    }
  } else {
    // render into a normal size framebuffer if stereo is not enabled
    viewport_sz_y = launch_dim.y;
    viewport_idx_y = launch_index.y;
    eyeshift = 0.0f;
  }

// XXX eliminate fov scaling related to viewport height factors,
//     we should add an explicit fov control for this camera
//  float fov = 180.0f * cam_zoom;             // dome FoV in degrees 
  float fov = 180.0f;                          // dome FoV in degrees 

  // half FoV in radians, pixels beyond this distance are outside
  // of the field of view of the projection, and are set black
  float rmax = 0.5 * fov * (M_PIf / 180.0f);

  // The dome angle from center of the projection is proportional 
  // to the image-space distance from the center of the viewport.
  // viewport_sz contains the viewport size, radperpix contains the
  // radians/pixel scaling factors in X/Y, and viewport_mid contains
  // the midpoint coordinate of the viewpoint used to compute the 
  // distance from center.
  float2 viewport_sz = make_float2(launch_dim.x, viewport_sz_y);
  float2 radperpix = (M_PIf / 180.0f) * fov / viewport_sz;
  float2 viewport_mid = viewport_sz * 0.5f;

  unsigned int randseed = tea<4>(launch_dim.x*(launch_index.y)+launch_index.x, subframe_count());

  float3 col = make_float3(0.0f);
  float alpha = 0.0f;
  for (int s=0; s<aa_samples; s++) {
    // compute the jittered image plane sample coordinate
    float2 jxy;  
    jitter_offset2f(randseed, jxy);
    float2 viewport_idx = make_float2(launch_index.x, viewport_idx_y) + jxy;

    // compute the ray angles in X/Y and total angular distance from center
    float2 rd = (viewport_idx - viewport_mid) * radperpix;
    float rangle = hypotf(rd.x, rd.y); 

    // pixels outside the dome FoV are treated as black by not
    // contributing to the color accumulator
    if (rangle < rmax) {
      float3 ray_direction;
      float3 ray_origin = cam_pos;

      if (rangle == 0) {
        // handle center of dome where azimuth is undefined by 
        // setting the ray direction to the zenith
        ray_direction = normalize(cam_W);
      } else {
        float rasin, racos;
        sincosf(rangle, &rasin, &racos);
        float rsin = rasin / rangle;
        ray_direction = normalize(cam_U*rsin*rd.x + cam_V*rsin*rd.y + cam_W*racos);

        if (STEREO_ON) {
#if 1

#else
          ray_origin += eyeshift * cross(ray_direction, cam_V);
#endif
        }

        if (DOF_ON) {
          float rcos = racos / rangle;
          float3 ray_right = normalize(cam_U*rcos*rd.x + cam_V*rcos*rd.y + cam_W* rasin);
          float3 ray_up = cross(ray_direction, ray_right);
          dof_ray(ray_origin, ray_origin, ray_direction, ray_direction,
                  randseed, ray_up, ray_right);
        }
      }

      // trace the new ray...
      PerRayData_radiance prd;
      prd.importance = 1.f;
      prd.alpha = 1.f;
      prd.depth = 0;
      prd.transcnt = max_trans;
      optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
      rtTrace(root_object, ray, prd);
#if defined(ORT_RAYSTATS)
      raystats1_buffer[launch_index].x++; // increment primary ray counter
#endif
      col += prd.result;
      alpha += prd.alpha;
    }
  }

#if defined(ORT_TIME_COLORING)
  accumulate_time_coloring(col, t0);
#else
  accumulate_color(col, alpha);
#endif
}

RT_PROGRAM void vmd_camera_dome_master() {
  vmd_camera_dome_general<0, 0>();
}

RT_PROGRAM void vmd_camera_dome_master_dof() {
  vmd_camera_dome_general<0, 1>();
}

RT_PROGRAM void vmd_camera_dome_master_stereo() {
  vmd_camera_dome_general<1, 0>();
}

RT_PROGRAM void vmd_camera_dome_master_stereo_dof() {
  vmd_camera_dome_general<1, 1>();
}


//
// Camera ray generation code for 360 degre FoV 
// equirectangular (lat/long) projection suitable
// for use a texture map for a sphere, e.g. for 
// immersive VR HMDs, other spheremap-based projections.
// 
template<int STEREO_ON, int DOF_ON>
static __device__ __inline__
void vmd_camera_equirectangular_general() {
#if defined(ORT_TIME_COLORING)
  clock_t t0 = clock(); // start per-pixel RT timer
#endif

  // The Samsung GearVR OTOY ORBX players have the left eye image on top, 
  // and the right eye image on the bottom.
  // Stereoscopic rendering is provided by rendering in an over/under
  // format with the left eye image into the top half of a double-high 
  // framebuffer, and the right eye into the lower half.  The subsequent 
  // OpenGL drawing code can trivially unpack and draw the two images 
  // with simple pointer offset arithmetic.
  uint viewport_sz_y, viewport_idx_y;
  float eyeshift;
  if (STEREO_ON) {
    // render into a double-high framebuffer when stereo is enabled
    viewport_sz_y = launch_dim.y >> 1;
    if (launch_index.y >= viewport_sz_y) {
      // left image
      viewport_idx_y = launch_index.y - viewport_sz_y;
      eyeshift = -0.5f * cam_stereo_eyesep;
    } else {
      // right image
      viewport_idx_y = launch_index.y;
      eyeshift =  0.5f * cam_stereo_eyesep;
    }
  } else {
    // render into a normal size framebuffer if stereo is not enabled
    viewport_sz_y = launch_dim.y;
    viewport_idx_y = launch_index.y;
    eyeshift = 0.0f;
  }

  float2 viewport_sz = make_float2(launch_dim.x, viewport_sz_y);
  float2 radperpix = M_PIf / viewport_sz * make_float2(2.0f, 1.0f);
  float2 viewport_mid = viewport_sz * 0.5f;

  unsigned int randseed = tea<4>(launch_dim.x*(launch_index.y)+launch_index.x, subframe_count());

  float3 col = make_float3(0.0f);
  for (int s=0; s<aa_samples; s++) {
    float2 jxy;  
    jitter_offset2f(randseed, jxy);

    float2 viewport_idx = make_float2(launch_index.x, viewport_idx_y) + jxy;
    float2 rangle = (viewport_idx - viewport_mid) * radperpix;

    float sin_ax, cos_ax, sin_ay, cos_ay;
    sincosf(rangle.x, &sin_ax, &cos_ax);
    sincosf(rangle.y, &sin_ay, &cos_ay);

    float3 ray_direction = normalize(cos_ay * (cos_ax * cam_W + sin_ax * cam_U) + sin_ay * cam_V);

    float3 ray_origin = cam_pos;
    if (STEREO_ON) {
      ray_origin += eyeshift * cross(ray_direction, cam_V);
    }

    // compute new ray origin and ray direction
    if (DOF_ON) {
      float3 ray_right = normalize(cos_ay * (-sin_ax * cam_W - cos_ax * cam_U) + sin_ay * cam_V);
      float3 ray_up = cross(ray_direction, ray_right);
      dof_ray(ray_origin, ray_origin, ray_direction, ray_direction,
              randseed, ray_up, ray_right);
    }

    // trace the new ray...
    PerRayData_radiance prd;
    prd.importance = 1.f;
    prd.depth = 0;
    prd.transcnt = max_trans;
    optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
    rtTrace(root_object, ray, prd);
#if defined(ORT_RAYSTATS)
    raystats1_buffer[launch_index].x++; // increment primary ray counter
#endif
    col += prd.result;
  }

#if defined(ORT_TIME_COLORING)
  accumulate_time_coloring(col, t0);
#else
  accumulate_color(col);
#endif
}

RT_PROGRAM void vmd_camera_equirectangular() {
  vmd_camera_equirectangular_general<0, 0>();
}

RT_PROGRAM void vmd_camera_equirectangular_dof() {
  vmd_camera_equirectangular_general<0, 1>();
}

RT_PROGRAM void vmd_camera_equirectangular_stereo() {
  vmd_camera_equirectangular_general<1, 0>();
}

RT_PROGRAM void vmd_camera_equirectangular_stereo_dof() {
  vmd_camera_equirectangular_general<1, 1>();
}


//
// Templated Oculus Rift perspective camera ray generation code
//
template<int STEREO_ON, int DOF_ON> 
static __device__ __inline__ 
void vmd_camera_oculus_rift_general() {
#if defined(ORT_TIME_COLORING)
  clock_t t0 = clock(); // start per-pixel RT timer
#endif

  // Stereoscopic rendering is provided by rendering in a side-by-side
  // format with the left eye image in the left half of a double-wide
  // framebuffer, and the right eye in the right half.  The subsequent 
  // OpenGL drawing code can trivially unpack and draw the two images 
  // with simple pointer offset arithmetic.
  uint viewport_sz_x, viewport_idx_x;
  float eyeshift;
  if (STEREO_ON) {
    // render into a double-wide framebuffer when stereo is enabled
    viewport_sz_x = launch_dim.x >> 1;
    if (launch_index.x >= viewport_sz_x) {
      // right image
      viewport_idx_x = launch_index.x - viewport_sz_x;
      eyeshift =  0.5f * cam_stereo_eyesep;
    } else {
      // left image
      viewport_idx_x = launch_index.x;
      eyeshift = -0.5f * cam_stereo_eyesep;
    }
  } else {
    // render into a normal size framebuffer if stereo is not enabled
    viewport_sz_x = launch_dim.x;
    viewport_idx_x = launch_index.x;
    eyeshift = 0.0f;
  }

  // 
  // general primary ray calculations
  //
  float2 aspect = make_float2(float(viewport_sz_x) / float(launch_dim.y), 1.0f) * cam_zoom;
  float2 viewportscale = 1.0f / make_float2(viewport_sz_x, launch_dim.y);
  float2 d = make_float2(viewport_idx_x, launch_index.y) * viewportscale * aspect * 2.f - aspect; // center of pixel in image plane


  // Compute barrel distortion required to correct for the pincushion inherent
  // in the plano-convex optics in the Oculus Rift, Google Cardboard, etc.
  // Barrel distortion involves computing distance of the pixel from the 
  // center of the eye viewport, and then scaling this distance by a factor
  // based on the original distance: 
  //   rnew = 0.24 * r^4 + 0.22 * r^2 + 1.0
  // Since we are only using even powers of r, we can use efficient 
  // squared distances everywhere.
  // The current implementation doesn't discard rays that would have fallen
  // outside of the original viewport FoV like most OpenGL implementations do.
  // The current implementation computes the distortion for the initial ray 
  // but doesn't apply these same corrections to antialiasing jitter, to
  // depth-of-field jitter, etc, so this leaves something to be desired if
  // we want best quality, but this raygen code is really intended for 
  // interactive display on an Oculus Rift or Google Cardboard type viewer,
  // so I err on the side of simplicity/speed for now. 
  float2 cp = make_float2(viewport_sz_x >> 1, launch_dim.y >> 1) * viewportscale * aspect * 2.f - aspect;;
  float2 dr = d - cp;
  float r2 = dr.x*dr.x + dr.y*dr.y;
  float r = 0.24f*r2*r2 + 0.22f*r2 + 1.0f;
  d = r * dr; 

  int subframecount = subframe_count();
  unsigned int randseed = tea<4>(launch_dim.x*(launch_index.y)+viewport_idx_x, subframecount);

  float3 eyepos = cam_pos;
  if (STEREO_ON) {
    eyepos += eyeshift * cam_U;
  } 

  float3 ray_origin = eyepos;
  float3 col = make_float3(0.0f);
  for (int s=0; s<aa_samples; s++) {
    float2 jxy;  
    jitter_offset2f(randseed, jxy);

    // don't jitter the first sample, since when using an HMD we often run
    // with only one sample per pixel unless the user wants higher fidelity
    jxy *= (subframecount > 0 || s > 0);

    jxy = jxy * viewportscale * aspect * 2.f + d;
    float3 ray_direction = normalize(jxy.x*cam_U + jxy.y*cam_V + cam_W);
 
    // compute new ray origin and ray direction
    if (DOF_ON) {
      dof_ray(eyepos, ray_origin, ray_direction, ray_direction,
              randseed, cam_V, cam_U);
    }

    // trace the new ray...
    PerRayData_radiance prd;
    prd.importance = 1.f;
    prd.depth = 0;
    prd.transcnt = max_trans;
    optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
    rtTrace(root_object, ray, prd);
#if defined(ORT_RAYSTATS)
    raystats1_buffer[launch_index].x++; // increment primary ray counter
#endif
    col += prd.result; 
  }

#if defined(ORT_TIME_COLORING)
  accumulate_time_coloring(col, t0);
#else
  accumulate_color(col);
#endif
}

RT_PROGRAM void vmd_camera_oculus_rift() {
  vmd_camera_oculus_rift_general<0, 0>();
}

RT_PROGRAM void vmd_camera_oculus_rift_dof() {
  vmd_camera_oculus_rift_general<0, 1>();
}

RT_PROGRAM void vmd_camera_oculus_rift_stereo() {
  vmd_camera_oculus_rift_general<1, 0>();
}

RT_PROGRAM void vmd_camera_oculus_rift_stereo_dof() {
  vmd_camera_oculus_rift_general<1, 1>();
}



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
  float alpha = 0.0f;
  float3 ray_origin = eyepos;
  for (int s=0; s<aa_samples; s++) {
    float2 jxy;  
    jitter_offset2f(randseed, jxy);

    jxy = jxy * viewportscale * aspect * 2.f + d;
    float3 ray_direction = normalize(jxy.x*cam_U + jxy.y*cam_V + cam_W);

    // compute new ray origin and ray direction
    if (DOF_ON) {
      dof_ray(eyepos, ray_origin, ray_direction, ray_direction,
              randseed, cam_V, cam_U);
    }

    // trace the new ray...
    PerRayData_radiance prd;
    prd.importance = 1.f;
    prd.alpha = 1.f;
    prd.depth = 0;
    prd.transcnt = max_trans;
    optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
    rtTrace(root_object, ray, prd);
#if defined(ORT_RAYSTATS)
    raystats1_buffer[launch_index].x++; // increment primary ray counter
#endif
    col += prd.result; 
    alpha += prd.alpha;
  }

#if defined(ORT_TIME_COLORING)
  accumulate_time_coloring(col, t0);
#else
  accumulate_color(col, alpha);
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
template<int STEREO_ON, int DOF_ON> 
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
  float3 view_direction;
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
    view_direction = normalize(cam_pos-eyepos + normalize(cam_W) * cam_stereo_convergence_dist);
  } else {
    // render into a normal size framebuffer if stereo is not enabled
    viewport_sz_y = launch_dim.y;
    viewport_idx_y = launch_index.y;
    eyepos = cam_pos;
    view_direction = normalize(cam_W);
  }

  // 
  // general primary ray calculations
  //
  float2 aspect = make_float2(float(launch_dim.x) / float(viewport_sz_y), 1.0f) * cam_zoom;
  float2 viewportscale = 1.0f / make_float2(launch_dim.x, viewport_sz_y);

  float2 d = make_float2(launch_index.x, viewport_idx_y) * viewportscale * aspect * 2.f - aspect; // center of pixel in image plane

  unsigned int randseed = tea<4>(launch_dim.x*(viewport_idx_y)+launch_index.x, subframe_count());

  float3 col = make_float3(0.0f);
  float alpha = 0.0f;
  float3 ray_direction = view_direction;
  for (int s=0; s<aa_samples; s++) {
    float2 jxy;  
    jitter_offset2f(randseed, jxy);
    jxy = jxy * viewportscale * aspect * 2.f + d;
    float3 ray_origin = eyepos + jxy.x*cam_U + jxy.y*cam_V;

    // compute new ray origin and ray direction
    if (DOF_ON) {
      dof_ray(ray_origin, ray_origin, view_direction, ray_direction,
              randseed, cam_V, cam_U);
    }

    // trace the new ray...
    PerRayData_radiance prd;
    prd.importance = 1.f;
    prd.alpha = 1.f;
    prd.depth = 0;
    prd.transcnt = max_trans;
    optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
    rtTrace(root_object, ray, prd);
#if defined(ORT_RAYSTATS)
    raystats1_buffer[launch_index].x++; // increment primary ray counter
#endif
    col += prd.result; 
    alpha += prd.alpha;
  }

#if defined(ORT_TIME_COLORING)
  accumulate_time_coloring(col, t0);
#else
  accumulate_color(col, alpha);
#endif
}

RT_PROGRAM void vmd_camera_orthographic() {
  vmd_camera_orthographic_general<0, 0>();
}

RT_PROGRAM void vmd_camera_orthographic_dof() {
  vmd_camera_orthographic_general<0, 1>();
}

RT_PROGRAM void vmd_camera_orthographic_stereo() {
  vmd_camera_orthographic_general<1, 0>();
}

RT_PROGRAM void vmd_camera_orthographic_stereo_dof() {
  vmd_camera_orthographic_general<1, 1>();
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

  return __saturatef(f);
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
    ambray = make_Ray(hit + ORT_RAY_STEP, dir, shadow_ray_type, scene_epsilon, ao_maxdist);
#else
    ambray = make_Ray(hit, dir, shadow_ray_type, scene_epsilon, ao_maxdist);
#endif

    shadow_prd.attenuation = make_float3(1.0f);
    rtTrace(root_shadower, ambray, shadow_prd);
#if defined(ORT_RAYSTATS)
    raystats1_buffer[launch_index].z++; // increment AO shadow ray counter
#endif
    inten += ndotambl * shadow_prd.attenuation;
  } 

  return inten * lightscale;
}


template<int SHADOWS_ON>       /// scene-wide shading property
static __device__ __inline__ void shade_light(float3 &result,
                                              float3 &hit_point, 
                                              float3 &N, float3 &L, 
                                              float p_Kd, 
                                              float p_Ks,
                                              float p_phong_exp,
                                              float3 &col, 
                                              float3 &phongcol,
                                              float shadow_tmax) {
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
      shadow_ray = make_Ray(hit_point + L * REVERSE_RAY_LENGTH, -L, shadow_ray_type, 0, fminf(tmax, shadow_tmax)); 
    } 
    else
#endif
    shadow_ray = make_Ray(hit_point + ORT_RAY_STEP, L, shadow_ray_type, scene_epsilon, shadow_tmax);

    rtTrace(root_shadower, shadow_ray, shadow_prd);
#if defined(ORT_RAYSTATS)
    raystats1_buffer[launch_index].y++; // increment shadow ray counter
#endif
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
template<int CLIP_VIEW_ON,     /// scene-wide shading property
         int HEADLIGHT_ON,     /// scene-wide shading property
         int FOG_ON,           /// scene-wide shading property
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

#if 1
  // XXX we really shouldn't have to do this, but it fixes shading 
  //     on really bad geometry that can arise from marching cubes 
  //     extraction on very noisy cryo-EM and cryo-ET maps
  float Ntest = N.x + N.y + N.z;
  if (isnan(Ntest) || isinf(Ntest)) {
    // add depth cueing / fog if enabled
    if (FOG_ON && fogmod < 1.0f) {
      result = fog_color(fogmod, result);
    }
    return;
  }
#endif

#if 1
  // don't render transparent surfaces if we've reached the max count
  // this implements the same logic as the -trans_max_surfaces argument
  // in the CPU version of Tachyon.
  if ((p_opacity < 1.0f) && (prd.transcnt < 1)) {
    // shoot a transmission ray
    PerRayData_radiance new_prd;
    new_prd.importance = prd.importance * (1.0f - p_opacity);
    new_prd.alpha = 1.0f;
    new_prd.result = scene_bg_color;
    new_prd.depth = prd.depth; // don't increment recursion depth
    new_prd.transcnt = prd.transcnt - 1;
    if (new_prd.importance >= 0.001f && 
        new_prd.depth <= max_depth) {
#if defined(ORT_USE_RAY_STEP)
#if defined(ORT_TRANS_USE_INCIDENT)
      // step the ray in the incident ray direction
      Ray trans_ray = make_Ray(hit_point + ORT_RAY_STEP2, ray.direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
#else
      // step the ray in the direction opposite the surface normal (going in)
      // rather than out, for transmission rays...
      Ray trans_ray = make_Ray(hit_point - ORT_RAY_STEP, ray.direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
#endif
#else
      Ray trans_ray = make_Ray(hit_point, ray.direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
#endif
      rtTrace(root_object, trans_ray, new_prd);
#if defined(ORT_RAYSTATS)
      raystats2_buffer[launch_index].x++; // increment trans ray counter
#endif
    }
    prd.result = new_prd.result;
    return; // early-exit
  }
#endif

  // execute the object's texture function
  float3 col = p_obj_color; // XXX no texturing implemented yet

  // compute lighting from directional lights
#if defined(VMDOPTIX_LIGHTUSEROBJS)
  unsigned int num_dir_lights = dir_light_list.num_lights;
  for (int i = 0; i < num_dir_lights; ++i) {
    float3 L = dir_light_list.dirs[i];
#else
  unsigned int num_dir_lights = dir_lights.size();
  for (int i = 0; i < num_dir_lights; ++i) {
    DirectionalLight light = dir_lights[i];
    float3 L = light.dir;
#endif
    shade_light<SHADOWS_ON>(result, hit_point, N, L, p_Kd, p_Ks, p_phong_exp, 
                            col, phongcol, RT_DEFAULT_MAX);
  }

#if 0
  // compute lighting from positional lights
#if defined(VMDOPTIX_LIGHTUSEROBJS)
  unsigned int num_pos_lights = pos_light_list.num_lights;
  for (int i = 0; i < num_pos_lights; ++i) {
    float3 L = pos_light_list.posns[i];
#else
  unsigned int num_pos_lights = pos_lights.size();
  for (int i = 0; i < num_pos_lights; ++i) {
    PositionalLight light = pos_lights[i];
    float3 L = light.pos - hit_point;
#endif
    float shadow_tmax = length(L); // compute positional light shadow tmax
    L = normalize(L);
    shade_light<SHADOWS_ON>(result, hit_point, N, L, p_Kd, p_Ks, p_phong_exp, 
                            col, phongcol, shadow_tmax);
  }
#endif

  // add point light for camera headlight need for Oculus Rift HMDs,
  // equirectangular panorama images, and planetarium dome master images
  if (HEADLIGHT_ON && (headlight_mode != 0)) {
    float3 L = cam_pos - hit_point;
    float shadow_tmax = length(L); // compute positional light shadow tmax
    L = normalize(L);
    shade_light<SHADOWS_ON>(result, hit_point, N, L, p_Kd, p_Ks, p_phong_exp, 
                            col, phongcol, shadow_tmax);
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
    float outlinefactor = __saturatef((1.0f - p_outline) + (edgefactor * p_outline));
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
    new_prd.transcnt = prd.transcnt;

    // shoot a reflection ray
    if (new_prd.importance >= 0.001f && new_prd.depth <= max_depth) {
      float3 refl_dir = reflect(ray.direction, N);
#if defined(ORT_USE_RAY_STEP)
      Ray refl_ray = make_Ray(hit_point + ORT_RAY_STEP, refl_dir, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
#else
      Ray refl_ray = make_Ray(hit_point, refl_dir, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
#endif
      rtTrace(root_object, refl_ray, new_prd);
#if defined(ORT_RAYSTATS)
      raystats2_buffer[launch_index].w++; // increment refl ray counter
#endif
      result += p_reflectivity * new_prd.result;
    }
  }

  // spawn transmission rays if necessary
  float alpha = p_opacity;

#if 1
  if (CLIP_VIEW_ON && (clipview_mode == 2))
    sphere_fade_and_clip(hit_point, cam_pos, 
                         clipview_start, clipview_end, alpha);
#else
  if (CLIP_VIEW_ON && (clipview_mode == 2)) {
    // draft implementation of a smooth "fade-out-and-clip sphere"  
    float fade_start = 1.00f; // onset of fading 
    float fade_end   = 0.20f; // fully transparent
    float camdist = length(hit_point - cam_pos);

    // XXX we can omit the distance test since alpha modulation value is clamped
    // if (1 || camdist < fade_start) {
      float fade_len = fade_start - fade_end;
      alpha *= __saturatef((camdist - fade_start) / fade_len);
    // }
  }
#endif

  // TRANSMISSION_ON: handles transparent surface shading, test is only
  // performed when the VMD geometry has a known-transparent material
  // CLIP_VIEW_ON: forces check of alpha value for all geom as per transparent 
  // material, since all geometry may become tranparent with the 
  // fade+clip sphere active
  if ((TRANSMISSION_ON || CLIP_VIEW_ON) && alpha < 0.999f ) {
    // Emulate Tachyon/Raster3D's angle-dependent surface opacity if enabled
    if (p_transmode) {
      alpha = 1.0f + cosf(3.1415926f * (1.0f-alpha) * dot(N, ray.direction));
      alpha = alpha*alpha * 0.25f;
    }

    result *= alpha; // scale down current lighting by opacity

    // shoot a transmission ray
    PerRayData_radiance new_prd;
    new_prd.importance = prd.importance * (1.0f - alpha);
    new_prd.alpha = 1.0f;
    new_prd.result = scene_bg_color;
    new_prd.depth = prd.depth + 1;
    new_prd.transcnt = prd.transcnt - 1;
    if (new_prd.importance >= 0.001f && 
        new_prd.depth <= max_depth) {
#if defined(ORT_USE_RAY_STEP)
#if defined(ORT_TRANS_USE_INCIDENT)
      // step the ray in the incident ray direction
      Ray trans_ray = make_Ray(hit_point + ORT_RAY_STEP2, ray.direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
#else
      // step the ray in the direction opposite the surface normal (going in)
      // rather than out, for transmission rays...
      Ray trans_ray = make_Ray(hit_point - ORT_RAY_STEP, ray.direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
#endif
#else
      Ray trans_ray = make_Ray(hit_point, ray.direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
#endif
      rtTrace(root_object, trans_ray, new_prd);
#if defined(ORT_RAYSTATS)
      raystats2_buffer[launch_index].x++; // increment trans ray counter
#endif
    }
    result += (1.0f - alpha) * new_prd.result;
    prd.alpha = alpha + (1.0f - alpha) * new_prd.alpha; 
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
  prd_shadow.attenuation *= make_float3(1.0f - opacity);

  // check to see if we've hit 100% shadow or not
  if (fmaxf(prd_shadow.attenuation) < 0.001f) {
    rtTerminateRay();
  } else {
#if defined(ORT_RAYSTATS)
    raystats2_buffer[launch_index].y++; // increment trans ray skip counter
#endif
    rtIgnoreIntersection();
  }
}


// Any hit program required for shadow filtering when an 
// HMD/camera fade-and-clip is active, through both 
// solid and transparent materials
RT_PROGRAM void any_hit_clip_sphere() {

  // compute hit point for use in evaluating fade/clip effect
  float3 hit_point = ray.origin + t_hit * ray.direction;

  // compute additional attenuation from clipping sphere if enabled
  float clipalpha = 1.0f;
  if (clipview_mode == 2) {
    sphere_fade_and_clip(hit_point, cam_pos, clipview_start, clipview_end, 
                         clipalpha);
  }

  // use a VERY simple shadow filtering scheme based on opacity
  prd_shadow.attenuation = make_float3(1.0f - (clipalpha * opacity));

  // check to see if we've hit 100% shadow or not
  if (fmaxf(prd_shadow.attenuation) < 0.001f) {
    rtTerminateRay();
  } else {
#if defined(ORT_RAYSTATS)
    raystats2_buffer[launch_index].y++; // increment trans ray skip counter
#endif
    rtIgnoreIntersection();
  }
}



// normal calc routine needed only to simplify the macro to produce the
// complete combinatorial expansion of template-specialized 
// closest hit radiance functions 
static __inline__ __device__ float3 calc_ffworld_normal(const float3 &Nshading, const float3 &Ngeometric) {
  float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, Nshading));
  float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, Ngeometric));
  return faceforward(world_shading_normal, -ray.direction, world_geometric_normal);
}


template<int HWTRIANGLES> static __device__ __inline__ 
float3 get_intersection_normal() {
#if defined(ORT_USERTXAPIS)
  // OptiX RTX hardware triangle API
  if (HWTRIANGLES) {
    const unsigned int primIdx = rtGetPrimitiveIndex();
    const float3 Ng = unpackNormal(normalBuffer[primIdx].x);
    float3 Ns;
    if (has_vertex_normals) {
      const float3& n0 = unpackNormal(normalBuffer[primIdx].y);
      const float3& n1 = unpackNormal(normalBuffer[primIdx].z);
      const float3& n2 = unpackNormal(normalBuffer[primIdx].w);
      Ns = optix::normalize(n0 * (1.0f - barycentrics.x - barycentrics.y) +
                            n1 * barycentrics.x + 
                            n2 * barycentrics.y);
    } else {
      Ns = Ng;
    }
    return calc_ffworld_normal(Ns, Ng);
  } else {
    // classic OptiX APIs and non-triangle geometry
    return calc_ffworld_normal(shading_normal, geometric_normal);
  } 
#else
  // classic OptiX APIs and non-triangle geometry
  return calc_ffworld_normal(shading_normal, geometric_normal);
#endif
}


template<int HWTRIANGLES> static __device__ __inline__ 
float3 get_intersection_color() {
#if defined(ORT_USERTXAPIS)
  // OptiX RTX hardware triangle API
  if (HWTRIANGLES) {
    if (has_vertex_colors) {
      const float ci2f = 1.0f / 255.0f;
      const unsigned int primIdx = rtGetPrimitiveIndex();
      const float3 c0 = colorBuffer[3 * primIdx + 0] * ci2f;
      const float3 c1 = colorBuffer[3 * primIdx + 1] * ci2f;
      const float3 c2 = colorBuffer[3 * primIdx + 2] * ci2f;

      // interpolate triangle color from barycentrics
      return (c0 * (1.0f - barycentrics.x - barycentrics.y) +
              c1 * barycentrics.x + 
              c2 * barycentrics.y);
    } else {
      return uniform_color; // return uniform mesh color
    }
  } else {
    // classic OptiX APIs and non-triangle geometry
    return obj_color; // return object color determined during intersection
  }
#else
  // classic OptiX APIs and non-triangle geometry
  return obj_color; // return object color determined during intersection
#endif
}


// general-purpose any-hit program, with all template options enabled, 
// intended for shader debugging and comparison with the original
// Tachyon full_shade() code.
RT_PROGRAM void closest_hit_radiance_general() {
  shader_template<1, 1, 1, 1, 1, 1, 1, 1>(get_intersection_color<0>(),
                                          get_intersection_normal<0>(), 
                                          Ka, Kd, Ks, phong_exp, Krefl, opacity,
                                          outline, outlinewidth, transmode);
}

#if defined(ORT_USERTXAPIS)
// OptiX RTX hardware triangle API
RT_PROGRAM void closest_hit_radiance_general_hwtri() {
  shader_template<1, 1, 1, 1, 1, 1, 1, 1>(get_intersection_color<1>(),
                                          get_intersection_normal<1>(), 
                                          Ka, Kd, Ks, phong_exp, Krefl, opacity,
                                          outline, outlinewidth, transmode);
}
#endif


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



#if defined(ORT_USE_SPHERES_HEARNBAKER)

// Ray-sphere intersection method with improved floating point precision
// for cases where the sphere size is small relative to the distance
// from the camera to the sphere.  This implementation is based on
// Eq. 10-72, p.603 of "Computer Graphics with OpenGL", 3rd Ed.,
// by Donald Hearn and Pauline Baker, 2004.  Shown in Eq. 10, p.639
// in the 4th edition of the book (Hearn, Baker, Carithers).
static __host__ __device__ __inline__
void sphere_intersect_hearn_baker2(float3 center, float radius,
                                   const float3 &spcolor) {
  float3 deltap = center - ray.origin;
  float ddp = dot(ray.direction, deltap);
  float3 remedyTerm = deltap - ddp * ray.direction;
  float disc = radius*radius - dot(remedyTerm, remedyTerm);
  if (disc >= 0.0f) {
    float disc_root = sqrtf(disc);
    float t1 = ddp - disc_root;
    float t2 = ddp + disc_root;

    if (rtPotentialIntersection(t1)) {
      shading_normal = geometric_normal = (t1*ray.direction - deltap) / radius;
      obj_color = spcolor;
      rtReportIntersection(0);
    }

    if (rtPotentialIntersection(t2)) {
      shading_normal = geometric_normal = (t2*ray.direction - deltap) / radius;
      obj_color = spcolor;
      rtReportIntersection(0);
    }
  }
}

#else

//
// Ray-sphere intersection using standard geometric solution approach
//
static __host__ __device__ __inline__
void sphere_intersect_classic(float3 center, float radius,
                              const float3 &spcolor) {
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
      obj_color = spcolor;
      rtReportIntersection(0);
    }
#else
    float t2 = b + disc;
    if (rtPotentialIntersection(t2)) {
      shading_normal = geometric_normal = (t2*ray.direction - V) / radius;
      obj_color = spcolor;
      rtReportIntersection(0);
    }

    float t1 = b - disc;
    if (rtPotentialIntersection(t1)) {
      shading_normal = geometric_normal = (t1*ray.direction - V) / radius;
      obj_color = spcolor;
      rtReportIntersection(0);
    }
#endif
  }
}

#endif


//
// Sphere array primitive
//
RT_PROGRAM void sphere_array_intersect(int primIdx) {
  float3 center = sphere_buffer[primIdx].center;
  float radius = sphere_buffer[primIdx].radius;

  // uniform color for the entire object
#if defined(ORT_USE_SPHERES_HEARNBAKER)
  sphere_intersect_hearn_baker2(center, radius, uniform_color);
#else
  sphere_intersect_classic(center, radius, uniform_color);
#endif
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

#if defined(ORT_USE_SPHERES_HEARNBAKER)
  sphere_intersect_hearn_baker2(center, radius, sphere_color_buffer[primIdx].color);
#else
  sphere_intersect_classic(center, radius, sphere_color_buffer[primIdx].color);
#endif
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
#if defined(ORT_USE_TEMPLATE_SHADERS)

#define CLIP_VIEW_on      1
#define CLIP_VIEW_off     0
#define HEADLIGHT_on      1
#define HEADLIGHT_off     0
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

#define DEFINE_CLOSEST_HIT( mclipview, mhlight, mfog, mshad, mao, moutl, mrefl, mtrans ) \
  RT_PROGRAM void                                                        \
  closest_hit_radiance_CLIP_VIEW_##mclipview##_HEADLIGHT_##mhlight##_FOG_##mfog##_SHADOWS_##mshad##_AO_##mao##_OUTLINE_##moutl##_REFL_##mrefl##_TRANS_##mtrans() { \
                                                                         \
    shader_template<CLIP_VIEW_##mclipview,                               \
                    HEADLIGHT_##mhlight,                                 \
                    FOG_##mfog,                                          \
                    SHADOWS_##mshad,                                     \
                    AO_##mao,                                            \
                    OUTLINE_##moutl,                                     \
                    REFL_##mrefl,                                        \
                    TRANS_##mtrans >                                     \
                    (get_intersection_color<0>(),                        \
                     get_intersection_normal<0>(),                       \
                     Ka, Kd, Ks, phong_exp, Krefl,                       \
                     opacity, outline, outlinewidth, transmode);         \
  }


//
// Generate all of the 2^8 parameter combinations as 
// template-specialized shaders...
//

DEFINE_CLOSEST_HIT(  on,  on,  on,  on,  on,  on,  on,  on )
DEFINE_CLOSEST_HIT(  on,  on,  on,  on,  on,  on,  on, off )

DEFINE_CLOSEST_HIT(  on,  on,  on,  on,  on,  on, off,  on )
DEFINE_CLOSEST_HIT(  on,  on,  on,  on,  on,  on, off, off )

DEFINE_CLOSEST_HIT(  on,  on,  on,  on,  on, off,  on,  on )
DEFINE_CLOSEST_HIT(  on,  on,  on,  on,  on, off,  on, off )

DEFINE_CLOSEST_HIT(  on,  on,  on,  on,  on, off, off,  on )
DEFINE_CLOSEST_HIT(  on,  on,  on,  on,  on, off, off, off )

DEFINE_CLOSEST_HIT(  on,  on,  on,  on, off,  on,  on,  on )
DEFINE_CLOSEST_HIT(  on,  on,  on,  on, off,  on,  on, off )

DEFINE_CLOSEST_HIT(  on,  on,  on,  on, off,  on, off,  on )
DEFINE_CLOSEST_HIT(  on,  on,  on,  on, off,  on, off, off )

DEFINE_CLOSEST_HIT(  on,  on,  on,  on, off, off,  on,  on )
DEFINE_CLOSEST_HIT(  on,  on,  on,  on, off, off,  on, off )

DEFINE_CLOSEST_HIT(  on,  on,  on,  on, off, off, off,  on )
DEFINE_CLOSEST_HIT(  on,  on,  on,  on, off, off, off, off )


DEFINE_CLOSEST_HIT(  on,  on,  on, off,  on,  on,  on,  on )
DEFINE_CLOSEST_HIT(  on,  on,  on, off,  on,  on,  on, off )

DEFINE_CLOSEST_HIT(  on,  on,  on, off,  on,  on, off,  on )
DEFINE_CLOSEST_HIT(  on,  on,  on, off,  on,  on, off, off )

DEFINE_CLOSEST_HIT(  on,  on,  on, off,  on, off,  on,  on )
DEFINE_CLOSEST_HIT(  on,  on,  on, off,  on, off,  on, off )

DEFINE_CLOSEST_HIT(  on,  on,  on, off,  on, off, off,  on )
DEFINE_CLOSEST_HIT(  on,  on,  on, off,  on, off, off, off )

DEFINE_CLOSEST_HIT(  on,  on,  on, off, off,  on,  on,  on )
DEFINE_CLOSEST_HIT(  on,  on,  on, off, off,  on,  on, off )

DEFINE_CLOSEST_HIT(  on,  on,  on, off, off,  on, off,  on )
DEFINE_CLOSEST_HIT(  on,  on,  on, off, off,  on, off, off )

DEFINE_CLOSEST_HIT(  on,  on,  on, off, off, off,  on,  on )
DEFINE_CLOSEST_HIT(  on,  on,  on, off, off, off,  on, off )

DEFINE_CLOSEST_HIT(  on,  on,  on, off, off, off, off,  on )
DEFINE_CLOSEST_HIT(  on,  on,  on, off, off, off, off, off )



DEFINE_CLOSEST_HIT(  on,  on, off,  on,  on,  on,  on,  on )
DEFINE_CLOSEST_HIT(  on,  on, off,  on,  on,  on,  on, off )

DEFINE_CLOSEST_HIT(  on,  on, off,  on,  on,  on, off,  on )
DEFINE_CLOSEST_HIT(  on,  on, off,  on,  on,  on, off, off )

DEFINE_CLOSEST_HIT(  on,  on, off,  on,  on, off,  on,  on )
DEFINE_CLOSEST_HIT(  on,  on, off,  on,  on, off,  on, off )

DEFINE_CLOSEST_HIT(  on,  on, off,  on,  on, off, off,  on )
DEFINE_CLOSEST_HIT(  on,  on, off,  on,  on, off, off, off )

DEFINE_CLOSEST_HIT(  on,  on, off,  on, off,  on,  on,  on )
DEFINE_CLOSEST_HIT(  on,  on, off,  on, off,  on,  on, off )

DEFINE_CLOSEST_HIT(  on,  on, off,  on, off,  on, off,  on )
DEFINE_CLOSEST_HIT(  on,  on, off,  on, off,  on, off, off )

DEFINE_CLOSEST_HIT(  on,  on, off,  on, off, off,  on,  on )
DEFINE_CLOSEST_HIT(  on,  on, off,  on, off, off,  on, off )

DEFINE_CLOSEST_HIT(  on,  on, off,  on, off, off, off,  on )
DEFINE_CLOSEST_HIT(  on,  on, off,  on, off, off, off, off )


DEFINE_CLOSEST_HIT(  on,  on, off, off,  on,  on,  on,  on )
DEFINE_CLOSEST_HIT(  on,  on, off, off,  on,  on,  on, off )

DEFINE_CLOSEST_HIT(  on,  on, off, off,  on,  on, off,  on )
DEFINE_CLOSEST_HIT(  on,  on, off, off,  on,  on, off, off )

DEFINE_CLOSEST_HIT(  on,  on, off, off,  on, off,  on,  on )
DEFINE_CLOSEST_HIT(  on,  on, off, off,  on, off,  on, off )

DEFINE_CLOSEST_HIT(  on,  on, off, off,  on, off, off,  on )
DEFINE_CLOSEST_HIT(  on,  on, off, off,  on, off, off, off )

DEFINE_CLOSEST_HIT(  on,  on, off, off, off,  on,  on,  on )
DEFINE_CLOSEST_HIT(  on,  on, off, off, off,  on,  on, off )

DEFINE_CLOSEST_HIT(  on,  on, off, off, off,  on, off,  on )
DEFINE_CLOSEST_HIT(  on,  on, off, off, off,  on, off, off )

DEFINE_CLOSEST_HIT(  on,  on, off, off, off, off,  on,  on )
DEFINE_CLOSEST_HIT(  on,  on, off, off, off, off,  on, off )

DEFINE_CLOSEST_HIT(  on,  on, off, off, off, off, off,  on )
DEFINE_CLOSEST_HIT(  on,  on, off, off, off, off, off, off )

//
// block of 64 pgms
//

DEFINE_CLOSEST_HIT(  on, off,  on,  on,  on,  on,  on,  on )
DEFINE_CLOSEST_HIT(  on, off,  on,  on,  on,  on,  on, off )

DEFINE_CLOSEST_HIT(  on, off,  on,  on,  on,  on, off,  on )
DEFINE_CLOSEST_HIT(  on, off,  on,  on,  on,  on, off, off )

DEFINE_CLOSEST_HIT(  on, off,  on,  on,  on, off,  on,  on )
DEFINE_CLOSEST_HIT(  on, off,  on,  on,  on, off,  on, off )

DEFINE_CLOSEST_HIT(  on, off,  on,  on,  on, off, off,  on )
DEFINE_CLOSEST_HIT(  on, off,  on,  on,  on, off, off, off )

DEFINE_CLOSEST_HIT(  on, off,  on,  on, off,  on,  on,  on )
DEFINE_CLOSEST_HIT(  on, off,  on,  on, off,  on,  on, off )

DEFINE_CLOSEST_HIT(  on, off,  on,  on, off,  on, off,  on )
DEFINE_CLOSEST_HIT(  on, off,  on,  on, off,  on, off, off )

DEFINE_CLOSEST_HIT(  on, off,  on,  on, off, off,  on,  on )
DEFINE_CLOSEST_HIT(  on, off,  on,  on, off, off,  on, off )

DEFINE_CLOSEST_HIT(  on, off,  on,  on, off, off, off,  on )
DEFINE_CLOSEST_HIT(  on, off,  on,  on, off, off, off, off )


DEFINE_CLOSEST_HIT(  on, off,  on, off,  on,  on,  on,  on )
DEFINE_CLOSEST_HIT(  on, off,  on, off,  on,  on,  on, off )

DEFINE_CLOSEST_HIT(  on, off,  on, off,  on,  on, off,  on )
DEFINE_CLOSEST_HIT(  on, off,  on, off,  on,  on, off, off )

DEFINE_CLOSEST_HIT(  on, off,  on, off,  on, off,  on,  on )
DEFINE_CLOSEST_HIT(  on, off,  on, off,  on, off,  on, off )

DEFINE_CLOSEST_HIT(  on, off,  on, off,  on, off, off,  on )
DEFINE_CLOSEST_HIT(  on, off,  on, off,  on, off, off, off )

DEFINE_CLOSEST_HIT(  on, off,  on, off, off,  on,  on,  on )
DEFINE_CLOSEST_HIT(  on, off,  on, off, off,  on,  on, off )

DEFINE_CLOSEST_HIT(  on, off,  on, off, off,  on, off,  on )
DEFINE_CLOSEST_HIT(  on, off,  on, off, off,  on, off, off )

DEFINE_CLOSEST_HIT(  on, off,  on, off, off, off,  on,  on )
DEFINE_CLOSEST_HIT(  on, off,  on, off, off, off,  on, off )

DEFINE_CLOSEST_HIT(  on, off,  on, off, off, off, off,  on )
DEFINE_CLOSEST_HIT(  on, off,  on, off, off, off, off, off )



DEFINE_CLOSEST_HIT(  on, off, off,  on,  on,  on,  on,  on )
DEFINE_CLOSEST_HIT(  on, off, off,  on,  on,  on,  on, off )

DEFINE_CLOSEST_HIT(  on, off, off,  on,  on,  on, off,  on )
DEFINE_CLOSEST_HIT(  on, off, off,  on,  on,  on, off, off )

DEFINE_CLOSEST_HIT(  on, off, off,  on,  on, off,  on,  on )
DEFINE_CLOSEST_HIT(  on, off, off,  on,  on, off,  on, off )

DEFINE_CLOSEST_HIT(  on, off, off,  on,  on, off, off,  on )
DEFINE_CLOSEST_HIT(  on, off, off,  on,  on, off, off, off )

DEFINE_CLOSEST_HIT(  on, off, off,  on, off,  on,  on,  on )
DEFINE_CLOSEST_HIT(  on, off, off,  on, off,  on,  on, off )

DEFINE_CLOSEST_HIT(  on, off, off,  on, off,  on, off,  on )
DEFINE_CLOSEST_HIT(  on, off, off,  on, off,  on, off, off )

DEFINE_CLOSEST_HIT(  on, off, off,  on, off, off,  on,  on )
DEFINE_CLOSEST_HIT(  on, off, off,  on, off, off,  on, off )

DEFINE_CLOSEST_HIT(  on, off, off,  on, off, off, off,  on )
DEFINE_CLOSEST_HIT(  on, off, off,  on, off, off, off, off )


DEFINE_CLOSEST_HIT(  on, off, off, off,  on,  on,  on,  on )
DEFINE_CLOSEST_HIT(  on, off, off, off,  on,  on,  on, off )

DEFINE_CLOSEST_HIT(  on, off, off, off,  on,  on, off,  on )
DEFINE_CLOSEST_HIT(  on, off, off, off,  on,  on, off, off )

DEFINE_CLOSEST_HIT(  on, off, off, off,  on, off,  on,  on )
DEFINE_CLOSEST_HIT(  on, off, off, off,  on, off,  on, off )

DEFINE_CLOSEST_HIT(  on, off, off, off,  on, off, off,  on )
DEFINE_CLOSEST_HIT(  on, off, off, off,  on, off, off, off )

DEFINE_CLOSEST_HIT(  on, off, off, off, off,  on,  on,  on )
DEFINE_CLOSEST_HIT(  on, off, off, off, off,  on,  on, off )

DEFINE_CLOSEST_HIT(  on, off, off, off, off,  on, off,  on )
DEFINE_CLOSEST_HIT(  on, off, off, off, off,  on, off, off )

DEFINE_CLOSEST_HIT(  on, off, off, off, off, off,  on,  on )
DEFINE_CLOSEST_HIT(  on, off, off, off, off, off,  on, off )

DEFINE_CLOSEST_HIT(  on, off, off, off, off, off, off,  on )
DEFINE_CLOSEST_HIT(  on, off, off, off, off, off, off, off )

///
/// Mid-way point (128 pgms)
///

DEFINE_CLOSEST_HIT( off,  on,  on,  on,  on,  on,  on,  on )
DEFINE_CLOSEST_HIT( off,  on,  on,  on,  on,  on,  on, off )

DEFINE_CLOSEST_HIT( off,  on,  on,  on,  on,  on, off,  on )
DEFINE_CLOSEST_HIT( off,  on,  on,  on,  on,  on, off, off )

DEFINE_CLOSEST_HIT( off,  on,  on,  on,  on, off,  on,  on )
DEFINE_CLOSEST_HIT( off,  on,  on,  on,  on, off,  on, off )

DEFINE_CLOSEST_HIT( off,  on,  on,  on,  on, off, off,  on )
DEFINE_CLOSEST_HIT( off,  on,  on,  on,  on, off, off, off )

DEFINE_CLOSEST_HIT( off,  on,  on,  on, off,  on,  on,  on )
DEFINE_CLOSEST_HIT( off,  on,  on,  on, off,  on,  on, off )

DEFINE_CLOSEST_HIT( off,  on,  on,  on, off,  on, off,  on )
DEFINE_CLOSEST_HIT( off,  on,  on,  on, off,  on, off, off )

DEFINE_CLOSEST_HIT( off,  on,  on,  on, off, off,  on,  on )
DEFINE_CLOSEST_HIT( off,  on,  on,  on, off, off,  on, off )

DEFINE_CLOSEST_HIT( off,  on,  on,  on, off, off, off,  on )
DEFINE_CLOSEST_HIT( off,  on,  on,  on, off, off, off, off )


DEFINE_CLOSEST_HIT( off,  on,  on, off,  on,  on,  on,  on )
DEFINE_CLOSEST_HIT( off,  on,  on, off,  on,  on,  on, off )

DEFINE_CLOSEST_HIT( off,  on,  on, off,  on,  on, off,  on )
DEFINE_CLOSEST_HIT( off,  on,  on, off,  on,  on, off, off )

DEFINE_CLOSEST_HIT( off,  on,  on, off,  on, off,  on,  on )
DEFINE_CLOSEST_HIT( off,  on,  on, off,  on, off,  on, off )

DEFINE_CLOSEST_HIT( off,  on,  on, off,  on, off, off,  on )
DEFINE_CLOSEST_HIT( off,  on,  on, off,  on, off, off, off )

DEFINE_CLOSEST_HIT( off,  on,  on, off, off,  on,  on,  on )
DEFINE_CLOSEST_HIT( off,  on,  on, off, off,  on,  on, off )

DEFINE_CLOSEST_HIT( off,  on,  on, off, off,  on, off,  on )
DEFINE_CLOSEST_HIT( off,  on,  on, off, off,  on, off, off )

DEFINE_CLOSEST_HIT( off,  on,  on, off, off, off,  on,  on )
DEFINE_CLOSEST_HIT( off,  on,  on, off, off, off,  on, off )

DEFINE_CLOSEST_HIT( off,  on,  on, off, off, off, off,  on )
DEFINE_CLOSEST_HIT( off,  on,  on, off, off, off, off, off )



DEFINE_CLOSEST_HIT( off,  on, off,  on,  on,  on,  on,  on )
DEFINE_CLOSEST_HIT( off,  on, off,  on,  on,  on,  on, off )

DEFINE_CLOSEST_HIT( off,  on, off,  on,  on,  on, off,  on )
DEFINE_CLOSEST_HIT( off,  on, off,  on,  on,  on, off, off )

DEFINE_CLOSEST_HIT( off,  on, off,  on,  on, off,  on,  on )
DEFINE_CLOSEST_HIT( off,  on, off,  on,  on, off,  on, off )

DEFINE_CLOSEST_HIT( off,  on, off,  on,  on, off, off,  on )
DEFINE_CLOSEST_HIT( off,  on, off,  on,  on, off, off, off )

DEFINE_CLOSEST_HIT( off,  on, off,  on, off,  on,  on,  on )
DEFINE_CLOSEST_HIT( off,  on, off,  on, off,  on,  on, off )

DEFINE_CLOSEST_HIT( off,  on, off,  on, off,  on, off,  on )
DEFINE_CLOSEST_HIT( off,  on, off,  on, off,  on, off, off )

DEFINE_CLOSEST_HIT( off,  on, off,  on, off, off,  on,  on )
DEFINE_CLOSEST_HIT( off,  on, off,  on, off, off,  on, off )

DEFINE_CLOSEST_HIT( off,  on, off,  on, off, off, off,  on )
DEFINE_CLOSEST_HIT( off,  on, off,  on, off, off, off, off )


DEFINE_CLOSEST_HIT( off,  on, off, off,  on,  on,  on,  on )
DEFINE_CLOSEST_HIT( off,  on, off, off,  on,  on,  on, off )

DEFINE_CLOSEST_HIT( off,  on, off, off,  on,  on, off,  on )
DEFINE_CLOSEST_HIT( off,  on, off, off,  on,  on, off, off )

DEFINE_CLOSEST_HIT( off,  on, off, off,  on, off,  on,  on )
DEFINE_CLOSEST_HIT( off,  on, off, off,  on, off,  on, off )

DEFINE_CLOSEST_HIT( off,  on, off, off,  on, off, off,  on )
DEFINE_CLOSEST_HIT( off,  on, off, off,  on, off, off, off )

DEFINE_CLOSEST_HIT( off,  on, off, off, off,  on,  on,  on )
DEFINE_CLOSEST_HIT( off,  on, off, off, off,  on,  on, off )

DEFINE_CLOSEST_HIT( off,  on, off, off, off,  on, off,  on )
DEFINE_CLOSEST_HIT( off,  on, off, off, off,  on, off, off )

DEFINE_CLOSEST_HIT( off,  on, off, off, off, off,  on,  on )
DEFINE_CLOSEST_HIT( off,  on, off, off, off, off,  on, off )

DEFINE_CLOSEST_HIT( off,  on, off, off, off, off, off,  on )
DEFINE_CLOSEST_HIT( off,  on, off, off, off, off, off, off )

//
// block of 64 pgms
//

DEFINE_CLOSEST_HIT( off, off,  on,  on,  on,  on,  on,  on )
DEFINE_CLOSEST_HIT( off, off,  on,  on,  on,  on,  on, off )

DEFINE_CLOSEST_HIT( off, off,  on,  on,  on,  on, off,  on )
DEFINE_CLOSEST_HIT( off, off,  on,  on,  on,  on, off, off )

DEFINE_CLOSEST_HIT( off, off,  on,  on,  on, off,  on,  on )
DEFINE_CLOSEST_HIT( off, off,  on,  on,  on, off,  on, off )

DEFINE_CLOSEST_HIT( off, off,  on,  on,  on, off, off,  on )
DEFINE_CLOSEST_HIT( off, off,  on,  on,  on, off, off, off )

DEFINE_CLOSEST_HIT( off, off,  on,  on, off,  on,  on,  on )
DEFINE_CLOSEST_HIT( off, off,  on,  on, off,  on,  on, off )

DEFINE_CLOSEST_HIT( off, off,  on,  on, off,  on, off,  on )
DEFINE_CLOSEST_HIT( off, off,  on,  on, off,  on, off, off )

DEFINE_CLOSEST_HIT( off, off,  on,  on, off, off,  on,  on )
DEFINE_CLOSEST_HIT( off, off,  on,  on, off, off,  on, off )

DEFINE_CLOSEST_HIT( off, off,  on,  on, off, off, off,  on )
DEFINE_CLOSEST_HIT( off, off,  on,  on, off, off, off, off )


DEFINE_CLOSEST_HIT( off, off,  on, off,  on,  on,  on,  on )
DEFINE_CLOSEST_HIT( off, off,  on, off,  on,  on,  on, off )

DEFINE_CLOSEST_HIT( off, off,  on, off,  on,  on, off,  on )
DEFINE_CLOSEST_HIT( off, off,  on, off,  on,  on, off, off )

DEFINE_CLOSEST_HIT( off, off,  on, off,  on, off,  on,  on )
DEFINE_CLOSEST_HIT( off, off,  on, off,  on, off,  on, off )

DEFINE_CLOSEST_HIT( off, off,  on, off,  on, off, off,  on )
DEFINE_CLOSEST_HIT( off, off,  on, off,  on, off, off, off )

DEFINE_CLOSEST_HIT( off, off,  on, off, off,  on,  on,  on )
DEFINE_CLOSEST_HIT( off, off,  on, off, off,  on,  on, off )

DEFINE_CLOSEST_HIT( off, off,  on, off, off,  on, off,  on )
DEFINE_CLOSEST_HIT( off, off,  on, off, off,  on, off, off )

DEFINE_CLOSEST_HIT( off, off,  on, off, off, off,  on,  on )
DEFINE_CLOSEST_HIT( off, off,  on, off, off, off,  on, off )

DEFINE_CLOSEST_HIT( off, off,  on, off, off, off, off,  on )
DEFINE_CLOSEST_HIT( off, off,  on, off, off, off, off, off )



DEFINE_CLOSEST_HIT( off, off, off,  on,  on,  on,  on,  on )
DEFINE_CLOSEST_HIT( off, off, off,  on,  on,  on,  on, off )

DEFINE_CLOSEST_HIT( off, off, off,  on,  on,  on, off,  on )
DEFINE_CLOSEST_HIT( off, off, off,  on,  on,  on, off, off )

DEFINE_CLOSEST_HIT( off, off, off,  on,  on, off,  on,  on )
DEFINE_CLOSEST_HIT( off, off, off,  on,  on, off,  on, off )

DEFINE_CLOSEST_HIT( off, off, off,  on,  on, off, off,  on )
DEFINE_CLOSEST_HIT( off, off, off,  on,  on, off, off, off )

DEFINE_CLOSEST_HIT( off, off, off,  on, off,  on,  on,  on )
DEFINE_CLOSEST_HIT( off, off, off,  on, off,  on,  on, off )

DEFINE_CLOSEST_HIT( off, off, off,  on, off,  on, off,  on )
DEFINE_CLOSEST_HIT( off, off, off,  on, off,  on, off, off )

DEFINE_CLOSEST_HIT( off, off, off,  on, off, off,  on,  on )
DEFINE_CLOSEST_HIT( off, off, off,  on, off, off,  on, off )

DEFINE_CLOSEST_HIT( off, off, off,  on, off, off, off,  on )
DEFINE_CLOSEST_HIT( off, off, off,  on, off, off, off, off )


DEFINE_CLOSEST_HIT( off, off, off, off,  on,  on,  on,  on )
DEFINE_CLOSEST_HIT( off, off, off, off,  on,  on,  on, off )

DEFINE_CLOSEST_HIT( off, off, off, off,  on,  on, off,  on )
DEFINE_CLOSEST_HIT( off, off, off, off,  on,  on, off, off )

DEFINE_CLOSEST_HIT( off, off, off, off,  on, off,  on,  on )
DEFINE_CLOSEST_HIT( off, off, off, off,  on, off,  on, off )

DEFINE_CLOSEST_HIT( off, off, off, off,  on, off, off,  on )
DEFINE_CLOSEST_HIT( off, off, off, off,  on, off, off, off )

DEFINE_CLOSEST_HIT( off, off, off, off, off,  on,  on,  on )
DEFINE_CLOSEST_HIT( off, off, off, off, off,  on,  on, off )

DEFINE_CLOSEST_HIT( off, off, off, off, off,  on, off,  on )
DEFINE_CLOSEST_HIT( off, off, off, off, off,  on, off, off )

DEFINE_CLOSEST_HIT( off, off, off, off, off, off,  on,  on )
DEFINE_CLOSEST_HIT( off, off, off, off, off, off,  on, off )

DEFINE_CLOSEST_HIT( off, off, off, off, off, off, off,  on )
DEFINE_CLOSEST_HIT( off, off, off, off, off, off, off, off )


#undef CLIP_VIEW_on
#undef CLIP_VIEW_off
#undef HEADLIGHT_on
#undef HEADLIGHT_off
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

#endif // ORT_USE_TEMPLATE_SHADERS


