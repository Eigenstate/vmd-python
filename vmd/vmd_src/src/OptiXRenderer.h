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
*      $RCSfile: OptiXDisplayDevice.h
*      $Author: johns $      $Locker:  $               $State: Exp $
*      $Revision: 1.115 $         $Date: 2019/01/17 21:21:00 $
*
***************************************************************************
* DESCRIPTION:
*   VMD built-in Tachyon/OptiX renderer implementation.
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

#ifndef LIBOPTIXRENDERER
#define LIBOPTIXRENDERER

#include <stdio.h>
#include <stdlib.h>
#include <optix.h>
#include <optix_math.h>
#include "Matrix4.h"
#include "ResizeArray.h"
#include "WKFUtils.h"

class VMDApp;

// Compile-time flag for collection and reporting of ray statistics
#if 0
#define ORT_RAYSTATS 1
#endif

// Compile-time flag to enable the use of RTX hardware ray tracing
// acceleration APIs in OptiX
#if OPTIX_VERSION >= 50200
#define ORT_USERTXAPIS 1
#endif

// #define VMDOPTIX_VCA_TABSZHACK 1
#ifdef VMDOPTIX_VCA_TABSZHACK
#define ORTMTABSZ 64
#else
#define ORTMTABSZ 256
#endif

// When compiling with OptiX 3.8 or greater, we use the new
// progressive rendering APIs rather than our previous hand-coded
// progressive renderer.
#if (defined(VMDOPTIX_VCA) || (OPTIX_VERSION >= 3080)) // && !defined(VMDUSEOPENHMD)
#define VMDOPTIX_PROGRESSIVEAPI 1
#endif

#if 1 || defined(VMDOPTIX_PROGRESSIVEAPI)
#define VMDOPTIX_LIGHTUSEROBJS 1
#endif

// Prevent interactive RT window code from being compiled when
// VMD isn't compiled with an interactive GUI
#if defined(VMDOPTIX_INTERACTIVE_OPENGL) && !defined(VMDOPENGL)
#undef VMDOPTIX_INTERACTIVE_OPENGL
#endif

#if defined(VMDOPTIX_INTERACTIVE_OPENGL)
#include "glwin.h"
#endif

/// remote device access 
typedef void * RTRDev; 

/// structure containing material properties used to shade a Displayable
typedef struct {
  RTmaterial mat; 
  int isvalid;
  float ambient;
  float diffuse;
  float specular;
  float shininess;
  float reflectivity;
  float opacity;
  float outline;
  float outlinewidth;
  int transmode;
  int ind;
} ort_material;

typedef struct {
  float dir[3];
  float color[3]; // XXX ignored for now
} ort_directional_light;

typedef struct {
  float pos[3];
  float color[3]; // XXX ignored for now
} ort_positional_light;


class OptiXRenderer {
public: 
  enum ClipMode { RT_CLIP_NONE=0, RT_CLIP_PLANE=1, RT_CLIP_SPHERE=2 };
  enum HeadlightMode { RT_HEADLIGHT_OFF=0, RT_HEADLIGHT_ON=1 };
  enum FogMode  { RT_FOG_NONE=0, RT_FOG_LINEAR=1, RT_FOG_EXP=2, RT_FOG_EXP2=3 };
  enum CameraProjection { RT_PERSPECTIVE=0, 
                          RT_ORTHOGRAPHIC=1,
                          RT_CUBEMAP=2,
                          RT_DOME_MASTER=3,
                          RT_EQUIRECTANGULAR=4,
                          RT_OCULUS_RIFT
                        };
  enum Verbosity { RT_VERB_MIN=0, RT_VERB_TIMING=1, RT_VERB_DEBUG=2 };
  enum BGMode { RT_BACKGROUND_TEXTURE_SOLID=0,
                RT_BACKGROUND_TEXTURE_SKY_SPHERE=1,
                RT_BACKGROUND_TEXTURE_SKY_ORTHO_PLANE=2 };
  enum RayType { RT_RAY_TYPE_RADIANCE=0,      ///< normal radiance rays
                 RT_RAY_TYPE_SHADOW=1,        ///< shadow probe/AO rays
                 RT_RAY_TYPE_COUNT=2 };       ///< total count of ray types
  enum RayGen  { RT_RAY_GEN_CLEAR_ACCUMULATION_BUFFER=0,
                 RT_RAY_GEN_ACCUMULATE=1,     ///< a render pass to accum buf
                 RT_RAY_GEN_COPY_FINISH=2,    ///< copy accum buf to framebuffer
#if defined(ORT_RAYSTATS)
                 RT_RAY_GEN_CLEAR_RAYSTATS=3, ///< clear raystats buffers
                 RT_RAY_GEN_COUNT=4 };        ///< total count of ray gen pgms
#else
                 RT_RAY_GEN_COUNT=3 };        ///< total count of ray gen pgms
#endif

private:
  VMDApp *app;                            ///< VMDApp ptr for video streaming
  void *remote_device;                    ///< remote rendering device/cluster
  Verbosity verbose;                      ///< console perf/debugging output
  int width;                              ///< image width in pixels
  int height;                             ///< image height in pixels
  char shaderpath[8192];                  ///< path to OptiX shader PTX file

  wkf_timerhandle ort_timer;              ///< general purpose timer
  double time_ctx_create;                 ///< time taken to create ctx
  double time_ctx_setup;                  ///< time taken to setup/init ctx
  double time_ctx_validate;               ///< time for ctx compile+validate
  double time_ctx_AS_build;               ///< time for AS build
  double time_ctx_destroy_scene;          ///< time to destroy existing scene
  double time_ray_tracing;                ///< time to trace the rays...
  double time_image_io;                   ///< time to write image(s) to disk

  // OptiX objects managed by VMD
  int context_created;                    ///< flag when context is valid
  RTcontext ctx;                          ///< OptiX main context
  RTresult lasterror;                     ///< Last OptiX error code if any

#if defined(ORT_USERTXAPIS)
  int hwtri_enabled;                      ///< OptiX HW triangle API enabled
#endif
  int buffers_allocated;                  ///< flag for buffer state
  int buffers_progressive;                ///< progressive API flag
  RTbuffer framebuffer;                   ///< output image buffer
  RTvariable framebuffer_v;               ///< output image buffer variable
  RTbuffer accumulation_buffer;           ///< intermediate GPU-local accum buf
  RTvariable accumulation_buffer_v;       ///< accum buffer variable
  RTvariable accum_count_v;               ///< accumulation subframe count 
#if defined(ORT_RAYSTATS)
  // the ray stats buffers get cleared when clearing the accumulation buffer
  RTbuffer raystats1_buffer;              ///< intermediate GPU-local stat buf
  RTvariable raystats1_buffer_v;          ///< ray stats buffer variable
  RTbuffer raystats2_buffer;              ///< intermediate GPU-local stat buf
  RTvariable raystats2_buffer_v;          ///< ray stats buffer variable
#endif

  int clipview_mode;                      ///< VR fade+clipping sphere/plane
  RTvariable clipview_mode_v;             ///< VR fade+clipping sphere/plane
  float clipview_start;                   ///< VR fade+clipping sphere/plane
  RTvariable clipview_start_v;            ///< VR fade+clipping sphere/plane
  float clipview_end;                     ///< VR fade+clipping sphere/plane
  RTvariable clipview_end_v;              ///< VR fade+clipping sphere/plane

  int headlight_mode;                     ///< VR HMD headlight
  RTvariable headlight_mode_v;            ///< VR HMD headlight

#if defined(VMDOPTIX_LIGHTUSEROBJS)
  RTvariable dir_light_list_v;            ///< list of directional lights
  RTvariable pos_light_list_v;            ///< list of positional lights
#else
  RTvariable dir_lightbuffer_v;           ///< list of directional lights
  RTbuffer dir_lightbuffer;               ///< list of directional lights
  RTvariable pos_lightbuffer_v;           ///< list of positional lights
  RTbuffer pos_lightbuffer;               ///< list of positional lights
#endif

  RTvariable ao_ambient_v;                ///< AO ambient lighting scalefactor
  float ao_ambient;                       ///< AO ambient lighting scalefactor
  RTvariable ao_direct_v;                 ///< AO direct lighting scalefactor
  float ao_direct;                        ///< AO direct lighting scalefactor
  RTvariable ao_maxdist_v;                ///< AO maximum occlusion distance 
  float ao_maxdist;                       ///< AO maximum occlusion distance 

  RTprogram exception_pgm;                ///< exception handling program

  int set_accum_raygen_pgm(CameraProjection &proj, int stereo_on, int dof_on);

#if defined(ORT_RAYSTATS)
  RTprogram clear_raystats_buffers_pgm;        ///< clear raystats buffers
#endif

  RTprogram clear_accumulation_buffer_pgm;     ///< clear accum buf
  RTprogram draw_accumulation_buffer_pgm;      ///< copy accum to framebuffer
  RTprogram draw_accumulation_buffer_stub_pgm; ///< progressive mode no-op stub

  RTprogram ray_gen_pgm_cubemap;                    ///< VR cubemap for Oculus
  RTprogram ray_gen_pgm_cubemap_dof;                ///< VR cubemap for Oculus
  RTprogram ray_gen_pgm_cubemap_stereo;             ///< VR cubemap for Oculus
  RTprogram ray_gen_pgm_cubemap_stereo_dof;         ///< VR cubemap for Oculus

  RTprogram ray_gen_pgm_dome_master;                ///< planetarium dome master
  RTprogram ray_gen_pgm_dome_master_dof;            ///< planetarium dome master
  RTprogram ray_gen_pgm_dome_master_stereo;         ///< planetarium dome master
  RTprogram ray_gen_pgm_dome_master_stereo_dof;     ///< planetarium dome master

  RTprogram ray_gen_pgm_equirectangular;            ///< 360 FoV for Oculus
  RTprogram ray_gen_pgm_equirectangular_dof;        ///< 360 FoV for Oculus
  RTprogram ray_gen_pgm_equirectangular_stereo;     ///< 360 FoV for Oculus
  RTprogram ray_gen_pgm_equirectangular_stereo_dof; ///< 360 FoV for Oculus

  RTprogram ray_gen_pgm_oculus_rift;                ///< Oculus Rift barrel
  RTprogram ray_gen_pgm_oculus_rift_dof;            ///< Oculus Rift barrel
  RTprogram ray_gen_pgm_oculus_rift_stereo;         ///< Oculus Rift barrel
  RTprogram ray_gen_pgm_oculus_rift_stereo_dof;     ///< Oculus Rift barrel

  RTprogram ray_gen_pgm_perspective;           ///< perspective cam (non-stereo)
  RTprogram ray_gen_pgm_perspective_dof;       ///< perspective cam (non-stereo)
  RTprogram ray_gen_pgm_perspective_stereo;    ///< perspective cam (stereo)
  RTprogram ray_gen_pgm_perspective_stereo_dof; //< perspective cam (stereo)

  RTprogram ray_gen_pgm_orthographic;            ///< ortho cam (non-stereo)
  RTprogram ray_gen_pgm_orthographic_dof;        ///< ortho cam (non-stereo)
  RTprogram ray_gen_pgm_orthographic_stereo;     ///< ortho cam (stereo)
  RTprogram ray_gen_pgm_orthographic_stereo_dof; ///< ortho cam (stereo)

  RTprogram closest_hit_pgm_general;             ///< fully general shader
#if defined(ORT_USE_TEMPLATE_SHADERS)
  RTprogram closest_hit_pgm_special[ORTMTABSZ];  ///< template-specialized fctns
#endif
#if defined(ORT_USERTXAPIS)
  RTprogram closest_hit_pgm_general_hwtri;       ///< general RTX tri shader
#if defined(ORT_USE_TEMPLATE_SHADERS)
  RTprogram closest_hit_pgm_special_hwtri[ORTMTABSZ];  ///< template-specialized fctns
#endif
#endif

  RTprogram any_hit_pgm_opaque;            ///< shadows for opaque objects
  RTprogram any_hit_pgm_transmission;      ///< shadows w/ filtering
  RTprogram any_hit_pgm_clip_sphere;       ///< shadows w/ clipping+filtering

  RTprogram miss_pgm_solid;                ///< miss/background shader
  RTprogram miss_pgm_sky_sphere;           ///< miss/background shader
  RTprogram miss_pgm_sky_ortho_plane;      ///< miss/background shader


  RTmaterial material_general;             ///< fully-general material
#if defined(ORT_USE_TEMPLATE_SHADERS)
  RTmaterial material_special[ORTMTABSZ];  ///< 2^8 specialized materials
#endif
#if defined(ORT_USERTXAPIS)
  RTmaterial material_general_hwtri;       ///< RTX tri material
#if defined(ORT_USE_TEMPLATE_SHADERS)
  RTmaterial material_special_hwtri[ORTMTABSZ];  ///< 2^8 specialized materials
#endif
#endif
  int material_special_counts[ORTMTABSZ];  ///< usage count in current scene


  // cylinder array primitive
  RTprogram cylinder_array_isct_pgm;       ///< cylinder array intersection code
  RTprogram cylinder_array_bbox_pgm;       ///< cylinder array bounding box code
  long cylinder_array_cnt;                 ///< number of cylinder in scene

  // color-per-cylinder array primitive
  RTprogram cylinder_array_color_isct_pgm; ///< cylinder array intersection code
  RTprogram cylinder_array_color_bbox_pgm; ///< cylinder array bounding box code
  long cylinder_array_color_cnt;           ///< number of cylinders in scene


  // color-per-ring array primitive
  RTprogram ring_array_color_isct_pgm;    ///< ring array intersection code
  RTprogram ring_array_color_bbox_pgm;    ///< ring array bounding box code
  long ring_array_color_cnt;              ///< number of rings in scene


  // sphere array primitive
  RTprogram sphere_array_isct_pgm;        ///< sphere array intersection code
  RTprogram sphere_array_bbox_pgm;        ///< sphere array bounding box code
  long sphere_array_cnt;                  ///< number of spheres in scene

  // color-per-sphere array primitive
  RTprogram sphere_array_color_isct_pgm;  ///< sphere array intersection code
  RTprogram sphere_array_color_bbox_pgm;  ///< sphere array bounding box code
  long sphere_array_color_cnt;            ///< number of spheres in scene


  // triangle mesh primitives of various types
  RTprogram tricolor_isct_pgm;            ///< triangle mesh intersection code
  RTprogram tricolor_bbox_pgm;            ///< triangle mesh bounding box code
  long tricolor_cnt;                      ///< number of triangles scene

  RTprogram trimesh_c4u_n3b_v3f_isct_pgm; ///< triangle mesh intersection code
  RTprogram trimesh_c4u_n3b_v3f_bbox_pgm; ///< triangle mesh bounding box code
  long trimesh_c4u_n3b_v3f_cnt;           ///< number of triangles scene

  RTprogram trimesh_n3f_v3f_isct_pgm;     ///< triangle mesh intersection code
  RTprogram trimesh_n3f_v3f_bbox_pgm;     ///< triangle mesh bounding box code
  long trimesh_n3b_v3f_cnt;               ///< number of triangles scene

  RTprogram trimesh_n3b_v3f_isct_pgm;     ///< triangle mesh intersection code
  RTprogram trimesh_n3b_v3f_bbox_pgm;     ///< triangle mesh bounding box code
  long trimesh_n3f_v3f_cnt;               ///< number of triangles scene

  RTprogram trimesh_v3f_isct_pgm;         ///< triangle mesh intersection code
  RTprogram trimesh_v3f_bbox_pgm;         ///< triangle mesh bounding box code
  long trimesh_v3f_cnt;                   ///< number of triangles scene

  // state variables to hold scene geometry
  int scene_created;
  RTgeometrygroup geometrygroup;          ///< node containing geom instances
  RTacceleration  acceleration;           ///< AS for scene geomgroup
#if defined(ORT_USERTXAPIS)
  // OptiX RTX hardware-accelerated triangles API
  RTgeometrygroup geometrytrianglesgroup; ///< hardware triangles group
  RTacceleration  trianglesacceleration;  ///< AS for hardware triangles
#endif

  RTgroup         root_group;             ///< root node for entire scene
  RTacceleration  root_acceleration;      ///< AS for root node 
  RTvariable      root_object_v;          ///< top level of scene graph 
  RTvariable      root_shadower_v;        ///< top level of scene graph

  //
  // OptiX shader state variables and the like
  //

  // progressive rendering mode (vs. batch)
  RTvariable progressive_enabled_v;     ///< progressive rendering flag

  RTvariable max_depth_v;               ///< max ray recursion depth
  RTvariable max_trans_v;               ///< max transmission ray depth

  RTvariable radiance_ray_type_v;       ///< index of active radiance ray type
  RTvariable shadow_ray_type_v;         ///< index of active shadow ray type
  RTvariable scene_epsilon_v;           ///< scene-wide epsilon value

  // shadow rendering mode 
  RTvariable shadows_enabled_v;         ///< shadow enable/disable flag  
  int shadows_enabled;                  ///< shadow enable/disable flag  

  RTvariable cam_zoom_v;                ///< camera zoom factor
  float cam_zoom;                       ///< camera zoom factor
  
  RTvariable cam_pos_v;                 ///< camera location
  RTvariable cam_U_v;                   ///< camera basis vector
  RTvariable cam_V_v;                   ///< camera basis vector
  RTvariable cam_W_v;                   ///< camera basis vector

  RTvariable cam_stereo_eyesep_v;           ///< stereo eye separation
  float cam_stereo_eyesep;                  ///< stereo eye separation
  RTvariable cam_stereo_convergence_dist_v; ///< stereo convergence distance
  float cam_stereo_convergence_dist;        ///< stereo convergence distance

  int dof_enabled;                      ///< DoF enable/disable flag  
  RTvariable cam_dof_focal_dist_v;      ///< DoF focal distance
  RTvariable cam_dof_aperture_rad_v;    ///< DoF aperture radius for CoC
  float cam_dof_focal_dist;             ///< DoF focal distance
  float cam_dof_fnumber;                ///< DoF f/stop number

#if 0
  RTvariable camera_projection_v;       ///< camera projection mode
#endif
  CameraProjection camera_projection;   ///< camera projection mode

  int ext_aa_loops;                     ///< Multi-pass AA iterations

  RTvariable accum_norm_v;              ///< Accum. buf normalization factor

  RTvariable aa_samples_v;              ///< AA samples per pixel
  int aa_samples;                       ///< AA samples per pixel

  RTvariable ao_samples_v;              ///< AO samples per pixel
  int ao_samples;                       ///< AO samples per pixel

  // background color and/or gradient parameters
  BGMode scene_background_mode;         ///< which miss program to use...
  RTvariable scene_bg_color_v;          ///< background color
  float scene_bg_color[3];              ///< background color
  RTvariable scene_bg_grad_top_v;       ///< background gradient top color
  float scene_bg_grad_top[3];           ///< background gradient top color
  RTvariable scene_bg_grad_bot_v;       ///< background gradient bottom color
  float scene_bg_grad_bot[3];           ///< background gradient bottom color
  RTvariable scene_gradient_v;          ///< background gradient vector
  float scene_gradient[3];              ///< background gradient vector
  RTvariable scene_gradient_topval_v;   ///< background gradient top value
  float scene_gradient_topval;          ///< background gradient top value
  RTvariable scene_gradient_botval_v;   ///< background gradient bot value
  float scene_gradient_botval;          ///< background gradient bot value
  RTvariable scene_gradient_invrange_v; ///< background gradient rcp range
  float scene_gradient_invrange;        ///< background gradient rcp range

  // clipping plane/sphere parameters
  int clip_mode;                        ///< clip mode
  float clip_start;                     ///< clip start (Z or radial dist)
  float clip_end;                       ///< clip end (Z or radial dist)

  // fog / depth cueing parameters
  RTvariable fog_mode_v;                ///< fog mode
  int fog_mode;                         ///< fog mode
  RTvariable fog_start_v;               ///< fog start
  float fog_start;                      ///< fog start
  RTvariable fog_end_v;                 ///< fog end
  float fog_end;                        ///< fog end
  RTvariable fog_density_v;             ///< fog density
  float fog_density;                    ///< fog density

  ResizeArray<ort_material> materialcache; ///< cache of VMD material values

  ResizeArray<ort_directional_light> directional_lights; ///< list of directional lights
  ResizeArray<ort_positional_light> positional_lights;   ///< list of positional lights

  // keep track of all of the OptiX objects we create on-the-fly...
  ResizeArray<RTbuffer> bufferlist;                  ///< list of all buffers
  ResizeArray<RTgeometry> geomlist;                  ///< list of all geom bufs
  ResizeArray<RTgeometryinstance> geominstancelist;  ///< list of all instances
#if defined(ORT_USERTXAPIS)
  // OptiX RTX hardware-accelerated triangles API
  ResizeArray<RTgeometrytriangles> geomtriangleslist;
  ResizeArray<RTgeometryinstance> geomtrianglesinstancelist;
#endif

  void append_objects(RTbuffer buf, RTgeometry geom, 
                      RTgeometryinstance instance) {
    bufferlist.append(buf);
    geomlist.append(geom);
    geominstancelist.append(instance);
  }

#if defined(ORT_USERTXAPIS)
  // OptiX RTX hardware-accelerated triangles API
  void append_objects(RTbuffer nbuf, RTbuffer cbuf,
                      RTgeometrytriangles geom_hwtri, 
                      RTgeometryinstance instance_hwtri) {
    bufferlist.append(nbuf);
    bufferlist.append(cbuf);
    geomtriangleslist.append(geom_hwtri);
    geomtrianglesinstancelist.append(instance_hwtri);
  }

  void append_buffer(RTbuffer buf) {
    bufferlist.append(buf);
  }
#endif


public:
  OptiXRenderer(VMDApp *vmdapp, void *remote_cluster_dev);
  ~OptiXRenderer(void);

#if OPTIX_VERSION >= 3080
  /// methods for managing remote VCA rendering (OptiX >= 3.8 only)
  static RTRDev remote_connect(const char *cluster, const char *user, const char *pw);
  static void remote_detach(RTRDev rdev);
#endif

  /// static methods for querying OptiX-supported GPU hardware independent
  /// of whether we actually have an active context.
  static unsigned int device_list(int **, char ***);
  static unsigned int device_count(void);
  static unsigned int optix_version(void);

  static int material_shader_table_size(void);

  /// check environment variables that modify verbose output
  void check_verbose_env();

  /// initialize the OptiX context 
  void create_context(void);
  void setup_context(int width, int height);
  
  /// report various context statistics for memory leak debugging, etc.
  void report_context_stats(void);

  /// shadows
  void shadows_on(int onoff) { shadows_enabled = (onoff != 0); }

  /// antialiasing (samples > 1 == on)
  void set_aa_samples(int cnt) { aa_samples = cnt; }

  /// set the camera projection mode
  void set_camera_projection(CameraProjection m) { camera_projection = m; }

  /// set camera zoom factor
  void set_camera_zoom(float zoomfactor) { cam_zoom = zoomfactor; }

  /// set stereo eye separation
  void set_camera_stereo_eyesep(float eyesep) { cam_stereo_eyesep = eyesep; }
  
  /// set stereo convergence distance
  void set_camera_stereo_convergence_dist(float dist) {
    cam_stereo_convergence_dist = dist;
  }

  /// depth of field on/off
  void dof_on(int onoff) { dof_enabled = (onoff != 0); }

  /// set depth of field focal plane distance
  void set_camera_dof_focal_dist(float d) { cam_dof_focal_dist = d; }

  /// set depth of field f/stop number
  void set_camera_dof_fnumber(float n) { cam_dof_fnumber = n; }

  /// ambient occlusion (samples > 1 == on)
  void set_ao_samples(int cnt) { ao_samples = cnt; }

  /// set AO ambient lighting factor
  void set_ao_ambient(float aoa) { ao_ambient = aoa; }

  /// set AO direct lighting factor
  void set_ao_direct(float aod) { ao_direct = aod; }

  void set_bg_mode(BGMode m) { scene_background_mode = m; }
  void set_bg_color(float *rgb) { memcpy(scene_bg_color, rgb, sizeof(scene_bg_color)); }
  void set_bg_color_grad_top(float *rgb) { memcpy(scene_bg_grad_top, rgb, sizeof(scene_bg_grad_top)); }
  void set_bg_color_grad_bot(float *rgb) { memcpy(scene_bg_grad_bot, rgb, sizeof(scene_bg_grad_bot)); }
  void set_bg_gradient(float *vec) { memcpy(scene_gradient, vec, sizeof(scene_gradient)); }
  void set_bg_gradient_topval(float v) { scene_gradient_topval = v; }
  void set_bg_gradient_botval(float v) { scene_gradient_botval = v; }

  /// set camera clipping plane/sphere mode and parameters
  void set_clip_sphere(ClipMode mode, float start, float end) {
    clip_mode = mode;
    clip_start = start;
    clip_end = end;
  }

  /// set depth cueing mode and parameters
  void set_cue_mode(FogMode mode, float start, float end, float density) {
    fog_mode = mode;
    fog_start = start;
    fog_end = end;
    fog_density = density;
  }

  void init_materials();
  void add_material(int matindex, float ambient, float diffuse,
                    float specular, float shininess, float reflectivity,
                    float opacity, float outline, float outlinewidth, 
                    int transmode);
  void set_material(RTgeometryinstance instance, int matindex, float *uniform_color, int hwtri=0);

  void clear_all_lights() { 
    directional_lights.clear(); 
    positional_lights.clear(); 
  }
  void set_clipview_mode(int mode) { 
    clipview_mode = mode;
  };
  void set_headlight_onoff(int onoff) { 
    headlight_mode = (onoff==1) ? RT_HEADLIGHT_ON : RT_HEADLIGHT_OFF; 
  };
  void add_directional_light(const float *dir, const float *color);
  void add_positional_light(const float *pos, const float *color);

  void update_rendering_state(int interactive);

  void config_framebuffer(int fbwidth, int fbheight, int interactive);
  void resize_framebuffer(int fbwidth, int fbheight);
  void destroy_framebuffer(void);

  void render_compile_and_validate(void);
  void render_to_file(const char *filename, int writealpha); 
#if defined(VMDOPTIX_INTERACTIVE_OPENGL)
  void render_to_glwin(const char *filename, int writealpha);
#endif
  void render_to_videostream(const char *filename, int writealpha);

  void destroy_scene(void);
  void destroy_context(void);

  void cylinder_array(Matrix4 *wtrans, float rscale, float *uniform_color,
                      int cylnum, float *points, int matindex);

  void cylinder_array_color(Matrix4 *wtrans, float rscale, int cylnum, 
                            float *points, float *radii, float *colors,
                            int matindex);

  void ring_array_color(Matrix4 & wtrans, float rscale, int rnum, 
                        float *centers, float *norms, float *radii, 
                        float *colors, int matindex);

  void sphere_array(Matrix4 *wtrans, float rscale, float *uniform_color,
                    int spnum, float *centers, float *radii, int matindex);

  void sphere_array_color(Matrix4 & wtrans, float rscale, int spnum, 
                          float *centers, float *radii, float *colors, 
                          int matindex);


  // 
  // Triangle mesh geometry
  //   Beginning with OptiX 5.2 and Turing-class GPUs, VMD uses the new
  //   NVIDIA RTX hardware-accelerated triangle APIs.  In combination with
  //   general OptiX software improvements, the Turing GPUs implement 
  //   hardware-accelerated BVH traversal, BVH-embedded storage of
  //   triangle meshes, and hardware-accelerated triangle intersection.
  //   In combination, these advances can provide a factor of ~8x
  //   performance gain compared with the classic OptiX APIs running on 
  //   Volta-class hardware.
  // 
#if defined(ORT_USERTXAPIS)
  void tricolor_list_hwtri(Matrix4 & wtrans, int numtris, float *vnc, int matindex);
#endif
  void tricolor_list(Matrix4 & wtrans, int numtris, float *vnc, int matindex);


#if defined(ORT_USERTXAPIS)
  void trimesh_c4n3v3_hwtri(Matrix4 & wtrans, int numverts,
                            float *cnv, int numfacets, int * facets, 
                            int matindex);
#endif
  void trimesh_c4n3v3(Matrix4 & wtrans, int numverts,
                      float *cnv, int numfacets, int * facets, int matindex);


#if defined(ORT_USERTXAPIS)
  void trimesh_c4u_n3b_v3f_hwtri(Matrix4 & wtrans, unsigned char *c, char *n, 
                                 float *v, int numfacets, int matindex);
#endif
  void trimesh_c4u_n3b_v3f(Matrix4 & wtrans, unsigned char *c, char *n, 
                           float *v, int numfacets, int matindex);


#if defined(ORT_USERTXAPIS)
  void trimesh_c4u_n3f_v3f_hwtri(Matrix4 & wtrans, unsigned char *c, 
                                 float *n, float *v, int numfacets, 
                                 int matindex);
#endif
  void trimesh_c4u_n3f_v3f(Matrix4 & wtrans, unsigned char *c, 
                           float *n, float *v, int numfacets, int matindex);


#if defined(ORT_USERTXAPIS)
  void trimesh_n3b_v3f_hwtri(Matrix4 & wtrans, float *uniform_color, 
                             char *n, float *v, int numfacets, int matindex);
#endif
  void trimesh_n3b_v3f(Matrix4 & wtrans, float *uniform_color, 
                       char *n, float *v, int numfacets, int matindex);


#if defined(ORT_USERTXAPIS)
  void trimesh_n3f_v3f_hwtri(Matrix4 & wtrans, float *uniform_color, 
                             float *n, float *v, int numfacets, int matindex);
#endif
  void trimesh_n3f_v3f(Matrix4 & wtrans, float *uniform_color, 
                       float *n, float *v, int numfacets, int matindex);


#if defined(ORT_USERTXAPIS)
  void trimesh_v3f_hwtri(Matrix4 & wtrans, float *uniform_color, 
                         float *v, int numfacets, int matindex);
#endif
  void trimesh_v3f(Matrix4 & wtrans, float *uniform_color, 
                   float *v, int numfacets, int matindex);


#if defined(ORT_USERTXAPIS)
  void tristrip_hwtri(Matrix4 & wtrans, int numverts, const float * cnv,
                      int numstrips, const int *vertsperstrip,
                      const int *facets, int matindex);
#endif
  void tristrip(Matrix4 & wtrans, int numverts, const float * cnv,
                int numstrips, const int *vertsperstrip,
                const int *facets, int matindex);

}; 

#endif

