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
*      $RCSfile: OSPRayDisplayDevice.h
*      $Author: johns $      $Locker:  $               $State: Exp $
*      $Revision: 1.24 $         $Date: 2019/01/17 21:21:00 $
*
***************************************************************************
* DESCRIPTION:
*   VMD built-in Tachyon/OSPRay renderer implementation.
*
* This work is briefly outlined in:
#  "OSPRay - A CPU Ray Tracing Framework for Scientific Visualization"
*   Ingo Wald, Gregory Johnson, Jefferson Amstutz, Carson Brownlee,
*   Aaron Knoll, Jim Jeffers, Johannes Guenther, Paul Navratil
*   IEEE Vis, 2016 (in-press)
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

#ifndef LIBOSPRAYRENDERER
#define LIBOSPRAYRENDERER

#include <stdio.h>
#include <stdlib.h>
#include <ospray/ospray.h>
#include "Matrix4.h"
#include "ResizeArray.h"
#include "WKFUtils.h"
#if !defined(OSPRAY_VERSION_MAJOR)
#include <ospray/version.h>
#endif
#if (OSPRAY_VERSION_MAJOR >= 1) 
#include <vector>
#endif

// Prevent interactive RT window code from being compiled when
// VMD isn't compiled with an interactive GUI
#if defined(VMDOPTIX_INTERACTIVE_OPENGL) && !defined(VMDOPENGL)
#undef VMDOPTIX_INTERACTIVE_OPENGL
#endif

#if defined(VMDOSPRAY_INTERACTIVE_OPENGL)
#include "glwin.h"
#endif

/// structure containing material properties used to shade a Displayable
typedef struct {
  OSPMaterial mat;
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
  float color[3];
} ort_directional_light;

typedef struct {
  float pos[3];
  float color[3];
} ort_positional_light;


typedef struct {
  int num;  
  float *v;
  float *n;
  float *c;
  int   *f;
  OSPGeometry geom;
  OSPData verts;
  OSPData norms;
  OSPData cols;
  OSPData ind;
  int matindex;
} ort_trimesh_v3f_n3f_c3f;


typedef struct {
  int num;  
  float *xyzr; // xyzr vec4
  float *colors;  // RGBA vec4
  unsigned int *indices;
  OSPGeometry geom;
  OSPData cents;
  OSPData cols;
  int matindex;
} ort_sphere_array_color;


typedef struct {
  int num; // number of cylinders
  float *cylinders; // point a, point b, radius, colorID
  float *colors;    // rgba
  int offset_v0;
  int offset_v1;
  int offset_radius;
  int offset_materialID;
  int radius; // default for all
  OSPGeometry geom;
  OSPData cyls;
  OSPData cols;
  int matindex;
} ort_cylinder_array_color;


class OSPRayRenderer {
public: 
  // Use reverse rays by default rather than only when enabled interactively
  enum RtShadowMode { RT_SHADOWS_OFF=0,        ///< shadows disabled
                      RT_SHADOWS_ON=1,         ///< shadows on, std. impl.
                    };
  enum FogMode { RT_FOG_NONE=0, RT_FOG_LINEAR=1, RT_FOG_EXP=2, RT_FOG_EXP2=3 };
  enum CameraProjection { RT_PERSPECTIVE=0, 
                          RT_ORTHOGRAPHIC=1 
// ,
//                          RT_CUBEMAP=2,
//                          RT_DOME_MASTER=3,
//                          RT_EQUIRECTANGULAR=4,
//                          RT_OCULUS_RIFT
                        };
  enum Verbosity { RT_VERB_MIN=0, RT_VERB_TIMING=1, RT_VERB_DEBUG=2 };
  enum BGMode { RT_BACKGROUND_TEXTURE_SOLID=0,
                RT_BACKGROUND_TEXTURE_SKY_SPHERE=1,
                RT_BACKGROUND_TEXTURE_SKY_ORTHO_PLANE=2 };

private:
  struct vec3 { float x, y, z; };
  struct rgba { float r, g, b, a; };

  struct Sphere {
    vec3  center;
    float radius;
    rgba  color;
    int   type;        //Material ID
  };

  Verbosity verbose;                      ///< console perf/debugging output
  int width;                              ///< image width in pixels
  int height;                             ///< image height in pixels
  char shaderpath[8192];                  ///< path to OSPRay renderer .so file

  wkf_timerhandle ort_timer;              ///< general purpose timer
  double time_ctx_create;                 ///< time taken to create ctx
  double time_ctx_setup;                  ///< time taken to setup/init ctx
  double time_ctx_validate;               ///< time for ctx compile+validate
  double time_ctx_AS_build;               ///< time for AS build
  double time_ctx_destroy_scene;          ///< time to destroy existing scene
  double time_ray_tracing;                ///< time to trace the rays...
  double time_image_io;                   ///< time to write image(s) to disk

  // OSPRay objects managed by VMD
  int context_created;                    ///< flag when context is valid
  OSPRenderer ospRenderer;                ///< OSPRay renderer
  OSPModel ospModel;                      ///< scene graph
  std::vector<OSPLight> ospLights;        ///< light objects
  OSPData ospLightData;                   ///< light object buffer shared with osp
  OSPCamera ospCamera;                    ///< camera/raygen code
  OSPFrameBuffer ospFrameBuffer;          ///< output and/or accum buffers

  std::vector<Sphere> spheres;            ///< accumulated sphere data

  int interactive_renderer;               ///< interactive RT flag

  int lasterror;                          ///< Last OSPRay error code if any
  int buffers_allocated;                  ///< flag for buffer state
  int headlight_enabled;                  ///< VR HMD headlight

  float ao_ambient;                       ///< AO ambient lighting scalefactor
  float ao_direct;                        ///< AO direct lighting scalefactor

  // cylinder array primitive
  long cylinder_array_cnt;                 ///< number of cylinder in scene

  // color-per-cylinder array primitive
  long cylinder_array_color_cnt;           ///< number of cylinders in scene

  // color-per-ring array primitive
  long ring_array_color_cnt;              ///< number of rings in scene

  // sphere array primitive
  long sphere_array_cnt;                  ///< number of spheres in scene

  // color-per-sphere array primitive
  long sphere_array_color_cnt;            ///< number of spheres in scene


  // triangle mesh primitives of various types
  long tricolor_cnt;                      ///< number of triangles scene
  long trimesh_c4u_n3b_v3f_cnt;           ///< number of triangles scene
  long trimesh_n3b_v3f_cnt;               ///< number of triangles scene
  long trimesh_n3f_v3f_cnt;               ///< number of triangles scene
  long trimesh_v3f_cnt;                   ///< number of triangles scene

  // state variables to hold scene geometry
  int scene_created;

  //
  // OSPRay shader state variables and the like
  //

  // shadow rendering mode 
  int shadows_enabled;                  ///< shadow enable/disable flag  
  float cam_zoom;                       ///< camera zoom factor
  vec3 cam_pos_v;                       ///< camera location
  vec3 cam_U_v;                         ///< camera basis vector (right)
  vec3 cam_V_v;                         ///< camera basis vector (up)
  vec3 cam_W_v;                         ///< camera basis vector (view dir)


  float cam_stereo_eyesep;              ///< stereo eye separation
  float cam_stereo_convergence_dist;    ///< stereo convergence distance

  int dof_enabled;                      ///< DoF enable/disable flag  
  float cam_dof_focal_dist;             ///< DoF focal distance
  float cam_dof_fnumber;                ///< DoF f/stop number

  CameraProjection camera_projection;   ///< camera projection mode

  int ext_aa_loops;                     ///< Multi-pass AA iterations
  int aa_samples;                       ///< AA samples per pixel
  int ao_samples;                       ///< AO samples per pixel

  // background color and/or gradient parameters
  BGMode scene_background_mode;         ///< which miss program to use...
  float scene_bg_color[3];              ///< background color
  float scene_bg_grad_top[3];           ///< background gradient top color
  float scene_bg_grad_bot[3];           ///< background gradient bottom color
  float scene_gradient[3];              ///< background gradient vector
  float scene_gradient_topval;          ///< background gradient top value
  float scene_gradient_botval;          ///< background gradient bot value
  float scene_gradient_invrange;        ///< background gradient rcp range

  // fog / depth cueing parameters
  int fog_mode;                         ///< fog mode
  float fog_start;                      ///< fog start
  float fog_end;                        ///< fog end
  float fog_density;                    ///< fog density

  ResizeArray<ort_material> materialcache; ///< cache of VMD material values

  ResizeArray<ort_directional_light> directional_lights; ///< list of directional lights
  ResizeArray<ort_positional_light> positional_lights;   ///< list of positional lights

  // keep track of all of the OSPRay objects we create on-the-fly...
  ResizeArray<ort_trimesh_v3f_n3f_c3f> trimesh_v3f_n3f_c3f;
  ResizeArray<ort_sphere_array_color> spheres_color;
  ResizeArray<ort_cylinder_array_color> cylinders_color;

public:
  static void OSPRay_Global_Init(void); ///< global init routine, call ONCE

  /// normal constructors and destructors
  OSPRayRenderer(void);
  ~OSPRayRenderer(void);

  /// check environment variables that modify verbose output
  void check_verbose_env();

  /// initialize the OSPRay context 
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
  void set_material(OSPGeometry &geom, int matindex, float *uniform_color);

  void clear_all_lights() { 
    headlight_enabled = 0;
    directional_lights.clear(); 
    positional_lights.clear(); 
  }
  void headlight_onoff(int onoff) { headlight_enabled = (onoff==1); };
  void add_directional_light(const float *dir, const float *color);
  void add_positional_light(const float *pos, const float *color);

  void update_rendering_state(int interactive);

  void config_framebuffer(int fbwidth, int fbheight);
  void resize_framebuffer(int fbwidth, int fbheight);
  void destroy_framebuffer(void);

  void render_compile_and_validate(void);
  void render_to_file(const char *filename); 
#if defined(VMDOSPRAY_INTERACTIVE_OPENGL)
  void render_to_glwin(const char *filename);
#endif

  void destroy_scene(void);
  void destroy_context(void);

#if 1
  void cylinder_array(Matrix4 *wtrans, float rscale, float *uniform_color,
                      int cylnum, float *points, int matindex);
#endif

  void cylinder_array_color(Matrix4 & wtrans, float rscale, int cylnum, 
                            float *points, float *radii, float *colors,
                            int matindex);

#if 0
  void ring_array_color(Matrix4 & wtrans, float rscale, int rnum, 
                        float *centers, float *norms, float *radii, 
                        float *colors, int matindex);
#endif

  void sphere_array(Matrix4 *wtrans, float rscale, float *uniform_color,
                    int spnum, float *centers, float *radii, int matindex);

  void sphere_array_color(Matrix4 & wtrans, float rscale, int spnum, 
                          float *centers, float *radii, float *colors, 
                          int matindex);

  void tricolor_list(Matrix4 & wtrans, int numtris, float *vnc, int matindex);

  void trimesh_c4n3v3(Matrix4 & wtrans, int numverts,
                      float *cnv, int numfacets, int * facets, int matindex);

  void trimesh_c4u_n3b_v3f(Matrix4 & wtrans, unsigned char *c, char *n, 
                           float *v, int numfacets, int matindex);

  void trimesh_c4u_n3f_v3f(Matrix4 & wtrans, unsigned char *c, 
                           float *n, float *v, int numfacets, int matindex);

  void trimesh_n3b_v3f(Matrix4 & wtrans, float *uniform_color, 
                       char *n, float *v, int numfacets, int matindex);

  void trimesh_n3f_v3f(Matrix4 & wtrans, float *uniform_color, 
                       float *n, float *v, int numfacets, int matindex);

#if 0
  void trimesh_v3f(Matrix4 & wtrans, float *uniform_color, 
                   float *v, int numfacets, int matindex);
#endif

  void tristrip(Matrix4 & wtrans, int numverts, const float * cnv,
                int numstrips, const int *vertsperstrip,
                const int *facets, int matindex);

}; 

#endif

