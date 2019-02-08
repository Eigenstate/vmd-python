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
*      $RCSfile: OSPRayRenderer.C
*      $Author: johns $      $Locker:  $               $State: Exp $
*      $Revision: 1.71 $         $Date: 2019/01/17 21:38:55 $
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

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#if defined(__linux)
#include <unistd.h>   // needed for symlink() in movie recorder
#endif

#include "Inform.h"
#include "ImageIO.h"
#include "OSPRayRenderer.h"
// #include "OSPRayShaders.ih" /// ISPC code at some point?
#include "Matrix4.h"
#include "utilities.h"
#include "WKFUtils.h"

#if !(OSPRAY_VERSION_MAJOR >= 1 && OSPRAY_VERSION_MINOR >= 2)
#error VMD requires OSPRay >= 1.2.0 for correct transparent AO shading
// VMD requires OSPRay >= 1.1.2 for correct rendering of cylinders
#endif

// enable the interactive ray tracing capability
#if defined(VMDOSPRAY_INTERACTIVE_OPENGL)
#include <GL/gl.h>
#endif

#if 1
#define DBG() 
#else
#define DBG() printf("OSPRayRenderer) %s\n", __func__);
#endif

// Global OSPRay initialization routine -- call it only ONCE...
void OSPRayRenderer::OSPRay_Global_Init(void) {
  DBG();

  // initialize OSPRay itself
  const char *ospraynormalargs[] = {"vmd", "--osp:mpi"};
  const char *ospraydebugargs[] = {"vmd", "--osp:debug", "--osp:mpi"};
  const char **osprayargs = ospraynormalargs; 
  int argcount = 1;

  if (getenv("VMDOSPRAYDEBUG") != NULL) {
    osprayargs = ospraydebugargs;
    argcount=2;    
  }

  // only pass in the second "--osp:mpi" flag if the user has
  // requested that the MPI renderer back-end be enabled through
  // environment variable flags
  if (getenv("VMDOSPRAYMPI") || getenv("VMD_OSPRAY_MPI")) {
    msgInfo << "OSPRayRenderer) Initializing OSPRay in MPI mode" << sendmsg;
    argcount++;
  }
 
  ospInit(&argcount, osprayargs);
}


/// constructor ... initialize some variables
OSPRayRenderer::OSPRayRenderer(void) {
  DBG();

  ort_timer = wkf_timer_create(); // create and initialize timer
  wkf_timer_start(ort_timer);

  // setup path to pre-compiled shader shared library .so 
  const char *vmddir = getenv("VMDDIR");
  if (vmddir == NULL)
    vmddir = ".";
  sprintf(shaderpath, "%s/shaders/%s", vmddir, "OSPRayRenderer.so");

  // allow runtime override of the default shader path for testing
  if (getenv("VMDOSPRAYSHADERPATH"))
    strcpy(shaderpath, getenv("VMDOSPRAYSHADERPATH"));

  // set OSPRay state handles/variables to NULL
  ospRenderer = NULL;
  ospFrameBuffer = NULL;
  ospCamera = NULL;
  ospModel = NULL;
  ospLightData = NULL;

  lasterror = 0;               // begin with no error state set
  context_created = 0;         // no context yet
  buffers_allocated = 0;       // flag no buffer allocated yet
  scene_created = 0;           // scene has been created

  destroy_scene();             // clear/init geometry vectors

  // clear timers
  time_ctx_setup = 0.0;
  time_ctx_validate = 0.0;
  time_ctx_AS_build = 0.0;
  time_ray_tracing = 0.0;
  time_image_io = 0.0;

  // set default scene background state
  scene_background_mode = RT_BACKGROUND_TEXTURE_SOLID;
  memset(scene_bg_color, 0, sizeof(scene_bg_color));
  memset(scene_bg_grad_top, 0, sizeof(scene_bg_grad_top));
  memset(scene_bg_grad_bot, 0, sizeof(scene_bg_grad_bot));
  memset(scene_gradient, 0, sizeof(scene_gradient));
  scene_gradient_topval = 1.0f;
  scene_gradient_botval = 0.0f;
  // XXX this has to be recomputed prior to rendering..
  scene_gradient_invrange = 1.0f / (scene_gradient_topval - scene_gradient_botval);

  cam_zoom = 1.0f;
  cam_stereo_eyesep = 0.06f;
  cam_stereo_convergence_dist = 2.0f;

  headlight_enabled = 0;     // VR HMD headlight disabled by default

  shadows_enabled = RT_SHADOWS_OFF; // disable shadows by default 
  aa_samples = 0;            // no AA samples by default

  ao_samples = 0;            // no AO samples by default
  ao_direct = 0.3f;          // AO direct contribution is 30%
  ao_ambient = 0.7f;         // AO ambient contribution is 70%

  dof_enabled = 0;           // disable DoF by default
  cam_dof_focal_dist = 2.0f;
  cam_dof_fnumber = 64.0f;

  fog_mode = RT_FOG_NONE;    // fog/cueing disabled by default
  fog_start = 0.0f;
  fog_end = 10.0f;
  fog_density = 0.32f;

  verbose = RT_VERB_MIN;  // keep console quiet except for perf/debugging cases
  check_verbose_env();    // see if the user has overridden verbose flag

  // clear all primitive lists
  trimesh_v3f_n3f_c3f.clear();
  spheres.clear();
  spheres_color.clear();
  cylinders_color.clear();

  create_context();
  destroy_scene();        // zero out object counters, prepare for rendering
}
        
/// destructor
OSPRayRenderer::~OSPRayRenderer(void) {
  DBG();

  int lcnt = ospLights.size();
  for (int i = 0; i < lcnt; ++i) {
    delete ospLights[i];
  }
  ospLights.clear();

  if (context_created)
    destroy_context(); 
  wkf_timer_destroy(ort_timer);
}


void OSPRayRenderer::check_verbose_env() {
  DBG();

  char *verbstr = getenv("VMDOSPRAYVERBOSE");
  if (verbstr != NULL) {
//    printf("OSPRayRenderer) verbosity config request: '%s'\n", verbstr);
    if (!strupcmp(verbstr, "MIN")) {
      verbose = RT_VERB_MIN;
      printf("OSPRayRenderer) verbose setting: minimum\n");
    } else if (!strupcmp(verbstr, "TIMING")) {
      verbose = RT_VERB_TIMING;
      printf("OSPRayRenderer) verbose setting: timing data\n");
    } else if (!strupcmp(verbstr, "DEBUG")) {
      verbose = RT_VERB_DEBUG;
      printf("OSPRayRenderer) verbose setting: full debugging data\n");
    }
  }
}


void OSPRayRenderer::create_context() {
  DBG();

  time_ctx_create = 0;
  if (context_created)
    return;

  double starttime = wkf_timer_timenow(ort_timer);

  if (verbose == RT_VERB_DEBUG) printf("OSPRayRenderer) creating context...\n");

  // Create our objects and set state
  if (lasterror != 0 /* XXX SUCCESS */) {
    msgErr << "OSPRayRenderer) Failed to create OSPRay rendering context" << sendmsg;
    context_created=0;
    return;
  }

  //
  // allow runtime override of the default shader path for testing
  // this has to be done prior to all calls that load programs from
  // the shader PTX
  //
  if (getenv("VMDOSPRAYSHADERPATH")) {
    strcpy(shaderpath, getenv("VMDOSPRAYSHADERPATH"));
    if (verbose == RT_VERB_DEBUG) 
      printf("OSPRayRenderer) user-override shaderpath: '%s'\n", shaderpath);
  }

  //
  // create the main renderer object needed early on for 
  // instantiation of materials, lights, etc.
  // 
  if ((ospRenderer = ospNewRenderer("scivis")) == NULL) {
    printf("OSPRayRenderer) Failed to load OSPRay renderer 'scivis'!\n");
  }

  // load and initialize all of the materials
  init_materials();

  // XXX lots of missing stuff here stil...

  time_ctx_create = wkf_timer_timenow(ort_timer) - starttime;
  
  if (verbose == RT_VERB_TIMING || verbose == RT_VERB_DEBUG) {
    printf("OSPRayRenderer) context creation time: %.2f\n", time_ctx_create);
  }

  context_created = 1;
}


void OSPRayRenderer::setup_context(int w, int h) {
  DBG();
  double starttime = wkf_timer_timenow(ort_timer);
  time_ctx_setup = 0;

  lasterror = 0; /* XXX SUCCESS; */ // clear any error state
  width = w;
  height = h;

  if (!context_created)
    return;

  check_verbose_env(); // update verbose flag if changed since last run

  if (getenv("VMDOSPRAYMAXDEPTH")) {
    int maxdepth = atoi(getenv("VMDOSPRAYMAXDEPTH"));
    if (maxdepth > 0 && maxdepth <= 20) {
      printf("OSPRayRenderer) Setting maxdepth to %d...\n", maxdepth);
      ospSet1f(ospRenderer, "maxDepth", maxdepth);  
    } else {
      printf("OSPRayRenderer) ignoring out-of-range maxdepth to %d...\n", maxdepth);
    }
  } else {
    ospSet1f(ospRenderer, "maxDepth", 20);  
  }

  float scene_epsilon = 5.e-5f;
  if (getenv("VMDOSPRAYSCENEEPSILON") != NULL) {
    scene_epsilon = atof(getenv("VMDOSPRAYSCENEEPSILON"));
    printf("OSPRayRenderer) user override of scene epsilon: %g\n", scene_epsilon);
  }
  ospSet1f(ospRenderer, "epsilon", scene_epsilon);

  // Implore OSPRay to correctly handle lighting through transparent 
  // surfaces when AO is enabled
  ospSet1i(ospRenderer, "aoTransparencyEnabled", 1);

  // Current accumulation subframe count, used as part of generating
  // AA and AO random number sequences
  // XXX set OSPRay accum count

  time_ctx_setup = wkf_timer_timenow(ort_timer) - starttime;
}


void OSPRayRenderer::destroy_scene() {
  DBG();

  double starttime = wkf_timer_timenow(ort_timer);
  time_ctx_destroy_scene = 0;

  // zero out all object counters
  cylinder_array_cnt = 0;
  cylinder_array_color_cnt = 0;
  ring_array_color_cnt = 0;
  sphere_array_cnt = 0;
  sphere_array_color_cnt = 0;
  tricolor_cnt = 0;
  trimesh_c4u_n3b_v3f_cnt = 0;
  trimesh_n3b_v3f_cnt = 0;
  trimesh_n3f_v3f_cnt = 0;
  trimesh_v3f_cnt = 0;

  if (context_created && scene_created) {
    // XXX destroy OSPRay objects...
    trimesh_v3f_n3f_c3f.clear();
    spheres.clear();

    // clear lists of primitives
    spheres_color.clear();
    cylinders_color.clear();
  }

  materialcache.clear();

  double endtime = wkf_timer_timenow(ort_timer);
  time_ctx_destroy_scene = endtime - starttime;

  scene_created = 0; // scene has been destroyed
}


void OSPRayRenderer::update_rendering_state(int interactive) {
  DBG();
  if (!context_created)
    return;

  wkf_timer_start(ort_timer);

  // Set interactive/progressive rendering flag so that we wire up
  // the most appropriate renderer for the task.  For batch rendering
  // with AO, we would choose the largest possible sample batch size,
  // but for interactive we will always choose a batch size of 1 or maybe 2
  // to yield the best interactivity.
  interactive_renderer = interactive;

  // XXX set OSPRay rendering state

  long totaltris = tricolor_cnt + trimesh_c4u_n3b_v3f_cnt + 
                   trimesh_n3b_v3f_cnt + trimesh_n3f_v3f_cnt + trimesh_v3f_cnt;

  if (verbose == RT_VERB_TIMING || verbose == RT_VERB_DEBUG) {
    printf("OSPRayRenderer) cyl %ld, ring %ld, sph %ld, tri %ld, tot: %ld  lt %ld\n",
           cylinder_array_cnt + cylinder_array_color_cnt,
           ring_array_color_cnt,
           sphere_array_cnt + sphere_array_color_cnt,
           totaltris,
           cylinder_array_cnt +  cylinder_array_color_cnt + ring_array_color_cnt + sphere_array_cnt + sphere_array_color_cnt + totaltris,
           directional_lights.num() + positional_lights.num());
  }

  if (verbose == RT_VERB_DEBUG) {
    printf("OSPRayRenderer) using fully general shader and materials.\n");
  }

  // XXX set OSPRay background color

  if (verbose == RT_VERB_DEBUG) {
    printf("OSPRayRenderer) scene bg mode: %d\n", scene_background_mode);

    printf("OSPRayRenderer) scene bgsolid: %.2f %.2f %.2f\n", 
           scene_bg_color[0], scene_bg_color[1], scene_bg_color[2]);

    printf("OSPRayRenderer) scene bggradT: %.2f %.2f %.2f\n", 
           scene_bg_grad_top[0], scene_bg_grad_top[1], scene_bg_grad_top[2]);

    printf("OSPRayRenderer) scene bggradB: %.2f %.2f %.2f\n", 
           scene_bg_grad_bot[0], scene_bg_grad_bot[1], scene_bg_grad_bot[2]);
  
    printf("OSPRayRenderer) bg gradient: %f %f %f  top: %f  bot: %f\n",
           scene_gradient[0], scene_gradient[1], scene_gradient[2],
           scene_gradient_topval, scene_gradient_botval);
  }

  // update in case the caller changed top/bottom values since last recalc
  scene_gradient_invrange = 1.0f / (scene_gradient_topval - scene_gradient_botval);
  // XXX set OSPRay background gradient

  // XXX set OSPRay fog mode

  if (verbose == RT_VERB_DEBUG) {
    printf("OSPRayRenderer) adding lights: dir: %ld  pos: %ld\n", 
           directional_lights.num(), positional_lights.num());
  }

  // XXX set OSPRay lights

  if (verbose == RT_VERB_DEBUG) 
    printf("OSPRayRenderer) Finalizing OSPRay scene graph...\n");

  // create group to hold instances

  // XXX we should create an acceleration object the instance shared
  //     by multiple PBC images


  // XXX OSPRay AS builder initialization if there's any customization...
 

  // do final state variable updates before rendering begins
  if (verbose == RT_VERB_DEBUG) {
    printf("OSPRayRenderer) cam zoom factor %f\n", cam_zoom);
    printf("OSPRayRenderer) cam stereo eye separation  %f\n", cam_stereo_eyesep);
    printf("OSPRayRenderer) cam stereo convergence distance %f\n", 
           cam_stereo_convergence_dist);
    printf("OSPRayRenderer) cam DoF focal distance %f\n", cam_dof_focal_dist);
    printf("OSPRayRenderer) cam DoF f/stop %f\n", cam_dof_fnumber);
  }

  // define all of the standard camera params
  // XXX set OSPRay camera state

  // define stereoscopic camera parameters
  // XXX set OSPRay camera state

  // define camera DoF parameters
  // XXX set OSPRay camera state

  // XXX set OSPRay AO sample counts and light scaling factors

  if (verbose == RT_VERB_DEBUG) {
    printf("OSPRayRenderer) setting sample counts:  AA %d  AO %d\n", aa_samples, ao_samples);
    printf("OSPRayRenderer) setting AO factors:  AOA %f  AOD %f\n", ao_ambient, ao_direct);
  }

  //
  // Handle AA samples either internally with loops internal to 
  // each ray launch point thread, or externally by iterating over
  // multiple launches, adding each sample to an accumulation buffer,
  // or a hybrid combination of the two.  
  //
#if 1
  ext_aa_loops = 1;
#else
  ext_aa_loops = 1;
  if (ao_samples > 0 || (aa_samples > 4)) {
    // if we have too much work for a single-pass rendering, we need to 
    // break it up into multiple passes of the right counts in each pass
    ext_aa_loops = 1 + aa_samples;
    // XXX set OSPRay sample counts per launch...
  } else { 
    // if the scene is simple, e.g. no AO rays and AA sample count is small,
    // we can run it in a single pass and get better performance
    // XXX set OSPRay sample counts per launch...
  }
  // XXX set OSPRay accum buf normalization scaling factors
#endif

  if (verbose == RT_VERB_DEBUG) {
    if (ext_aa_loops > 1)
      printf("OSPRayRenderer) Running OSPRay multi-pass: %d loops\n", ext_aa_loops);
    else
      printf("OSPRayRenderer) Running OSPRay single-pass: %d total samples\n", 1+aa_samples);
  }

  // set the ray generation program to the active camera code...
  // XXX set OSPRay camera mode and clear accum buf
  // set the active color accumulation ray gen program based on the 
  // camera/projection mode, stereoscopic display mode, 
  // and depth-of-field state
  // XXX set OSPRay camera mode and accum buf mode
  // XXX set OSPRay "miss" shading mode (solid or gradient)
}


void OSPRayRenderer::config_framebuffer(int fbwidth, int fbheight) {
  if (!context_created)
    return;

  // allocate and resize buffers to match request
  if (buffers_allocated) {
    // if the buffers already exist and match the current 
    // progressive/non-progressive rendering mode, just resize them
    if (verbose == RT_VERB_DEBUG) {
      printf("OSPRayRenderer) resizing framebuffer\n");
    }
    resize_framebuffer(fbwidth, fbheight);
  } else {
    // (re)allocate framebuffer and associated accumulation buffers if they
    // don't already exist or if they weren't bound properly for
    // current progressive/non-progressive rendering needs.
    if (verbose == RT_VERB_DEBUG) {
      printf("OSPRayRenderer) creating framebuffer and accum. buffer\n");
    }

    // create intermediate output and accumulation buffers
    osp::vec2i fbsize = { width, height };   
    ospFrameBuffer = ospNewFrameBuffer(fbsize, OSP_FB_RGBA8, OSP_FB_COLOR | OSP_FB_ACCUM);
    ospFrameBufferClear(ospFrameBuffer, OSP_FB_COLOR | /* OSP_FB_DEPTH | */ OSP_FB_ACCUM);

    buffers_allocated = 1;
  }
}


void OSPRayRenderer::resize_framebuffer(int fbwidth, int fbheight) {
  if (!context_created)
    return;

  if (buffers_allocated) {
    if (verbose == RT_VERB_DEBUG) 
      printf("OSPRayRenderer) resize_framebuffer(%d x %d)\n", fbwidth, fbheight);
    destroy_framebuffer();
  }

  osp::vec2i fbsize = { fbwidth, fbheight };   
  ospFrameBuffer = ospNewFrameBuffer(fbsize, OSP_FB_RGBA8, OSP_FB_COLOR | OSP_FB_ACCUM);

  ospFrameBufferClear(ospFrameBuffer, OSP_FB_COLOR | /* OSP_FB_DEPTH | */ OSP_FB_ACCUM);
}


void OSPRayRenderer::destroy_framebuffer() {
  if (!context_created)
    return;

  if (buffers_allocated) {
    if (ospFrameBuffer)
#if OSPRAY_VERSION_MAJOR >= 1 && OSPRAY_VERSION_MINOR >= 6
      ospRelease(ospFrameBuffer);
#else
      ospFreeFrameBuffer(ospFrameBuffer);
#endif
  }
  buffers_allocated = 0;
}


void OSPRayRenderer::render_compile_and_validate(void) {
  int i;

  DBG();
  if (!context_created)
    return;

  //
  // finalize context validation, compilation, and AS generation 
  //
  double startctxtime = wkf_timer_timenow(ort_timer);

  // XXX any last OSPRay state updates/checks

  if ((ospModel = ospNewModel()) == NULL) {
    printf("OSPRayRenderer) Failed to create new model!\n");
  }


  //
  // commit sphere geometry
  //
  OSPData data = ospNewData(5 * spheres.size(), OSP_FLOAT, &spheres[0]);

//FIXME: Incorrect
//    std::vector<OSPMaterial> sphereMaterials;
//    for(int i = 0; i < spheres.size(); i++){
// XXX 
//      sphereMaterials.push_back(createMaterialFromIndex(spheres[i].type));
//      spheres[i].type = i;
//    }
//    OSPData sphereMatData = ospNewData(sphereMaterials.size(), OSP_OBJECT, &sphereMaterials[0], 0);
  OSPGeometry sphere_geom = ospNewGeometry("spheres");
//  ospSetData(sphere_geom, "materialList", sphereMatData);
//  set_material(sphere_geom, matindex, NULL);

  ospSet1i(sphere_geom, "bytes_per_sphere", sizeof(Sphere));
  ospSet1i(sphere_geom, "offset_center", 0);
  ospSet1i(sphere_geom, "offset_radius", sizeof(vec3));
  ospSet1i(sphere_geom, "offset_materialID", sizeof(vec3) + sizeof(float));
  ospSetData(sphere_geom, "spheres", data);
  ospCommit(sphere_geom);
  ospAddGeometry(ospModel, sphere_geom);
 
  if (verbose == RT_VERB_DEBUG)
    printf("OSPRayReenderer) num spheres = %ld\n", spheres.size());


  // 
  // Set camera parms
  // 
  float cam_pos_orig[3] = {0.0f, 0.0f, 2.0f};
  float cam_U_orig[3] = {1.0f, 0.0f, 0.0f};
  float cam_V_orig[3] = {0.0f, 1.0f, 0.0f};
  float cam_W_orig[3] = {0.0f, 0.0f, -1.0f};

  float cam_pos[3], cam_U[3], cam_V[3], cam_W[3];
  vec_copy(cam_pos, cam_pos_orig);
  vec_copy(cam_U, cam_U_orig);
  vec_copy(cam_V, cam_V_orig);
  vec_copy(cam_W, cam_W_orig);

  if (camera_projection == OSPRayRenderer::RT_ORTHOGRAPHIC) {
    if(!ospCamera) ospCamera = ospNewCamera("orthographic");
 
   ospSet1f(ospCamera, "aspect", width / ((float) height));
    float orthoheight = 2.0f * cam_zoom;
    ospSet1f(ospCamera, "height", orthoheight);

    if (dof_enabled) {
      msgWarn << "OSPRayRenderer) DoF not implemented for orthographic camera!" << sendmsg;
    }
  } else {
    if(!ospCamera) ospCamera = ospNewCamera("perspective");

    ospSet1f(ospCamera, "aspect", width / ((float) height));
    //  ospSet1f(ospCamera, "fovy", 180.0f*atanf((ySize/2.0f)/zDist));
    ospSet1f(ospCamera, "fovy", 2.0f*180.0f*(atanf(cam_zoom)/M_PI));

    if (dof_enabled) {
      ospSet1f(ospCamera, "focusDistance", cam_dof_focal_dist);
      ospSet1f(ospCamera, "apertureRadius", cam_dof_focal_dist / (2.0f * cam_zoom * cam_dof_fnumber));
    } else {
      ospSet1f(ospCamera, "apertureRadius", 0.0f);
    }
  }

  if (ospCamera) {
    ospSet3fv(ospCamera, "pos", cam_pos);
    ospSet3fv(ospCamera, "dir",   cam_W);
    ospSet3fv(ospCamera,  "up",   cam_V);
    ospCommit(ospCamera);
    //  ospSet1f(ospCamera, "nearClip", 0.5f);
    //  ospSet1f(ospCamera, "far", 10.0f);
  }

  // 
  // Set framebuffer 
  // 
  config_framebuffer(width, height);

  //
  // Set all lights
  //

  // The direct lighting scaling factor all of the other lights.
  float lightscale = 1.0f;
  if (ao_samples != 0)
    lightscale = ao_direct;

  for (i = 0; i < directional_lights.num(); ++i) {
#if OSPRAY_VERSION_MAJOR >= 1 && OSPRAY_VERSION_MINOR >= 7
    OSPLight light = ospNewLight3("distant");
#elif OSPRAY_VERSION_MAJOR >= 1 && OSPRAY_VERSION_MINOR >= 6
    OSPLight light = ospNewLight2("scivis", "distant");
#elif 1
    OSPLight light = ospNewLight(ospRenderer, "distant");
#else
    OSPLight light = ospNewLight(ospRenderer, "DirectionalLight");
#endif

    // The direct lighting scaling factor is applied to the lights here.
    ospSet1f(light, "intensity", lightscale);
    ospSet3fv(light, "color", directional_lights[i].color);

    float lightDir[3];
    vec_copy(lightDir, directional_lights[i].dir);
    vec_normalize(lightDir); // just for good measure

    // OSPRay uses a light direction vector opposite to VMD and Tachyon 
    ospSet3f(light, "direction", -lightDir[0], -lightDir[1], -lightDir[2]);

    ospCommit(light);
    ospLights.push_back(light);
  }

  // AO scaling factor is applied to a special ambient light.
  if (ao_samples != 0) {
#if OSPRAY_VERSION_MAJOR >= 1 && OSPRAY_VERSION_MINOR >= 7
    OSPLight light = ospNewLight3("ambient");
#elif OSPRAY_VERSION_MAJOR >= 1 && OSPRAY_VERSION_MINOR >= 6
    OSPLight light = ospNewLight2("scivis", "ambient");
#else
    OSPLight light = ospNewLight(ospRenderer, "ambient");
#endif

    // AO scaling factor is applied to the special ambient light
    ospSet1f(light, "intensity", ao_ambient);
    ospSet3f(light, "color", 1.0f, 1.0f, 1.0f);

    ospCommit(light);
    ospLights.push_back(light); // add AO ambient light
  } 

  //
  // attach all lights to the scene
  //
#if 1
  ospLightData = ospNewData(ospLights.size(), OSP_OBJECT, &ospLights[0], 0);
#else
  ospLightData = ospNewData(ospLights.size(), OSP_OBJECT, &ospLights[0], OSP_DATA_SHARED_BUFFER);
#endif

  // 
  // update renderer state
  //
  if (ao_samples && interactive_renderer) {
    ospSet1i(ospRenderer, "aoSamples", 1);
    ospSet1i(ospRenderer, "spp", 1);
  } else {
    ospSet1i(ospRenderer, "spp", aa_samples);
    ospSet1i(ospRenderer, "aoSamples", ao_samples);
  }

  if (getenv("VMDOSPRAYAOMAXDIST")) {
    float tmp = atof(getenv("VMDOSPRAYAOMAXDIST"));
    if (verbose == RT_VERB_DEBUG) {
      printf("OSPRayRenderer) setting AO maxdist: %f\n", tmp);
    }
    ospSet1f(ospRenderer, "aoOcclusionDistance", tmp);
  }

  float scene_epsilon = 5.e-5f;
  if (getenv("VMDOSPRAYSCENEEPSILON") != NULL) {
    scene_epsilon = atof(getenv("VMDOSPRAYSCENEEPSILON"));
    printf("OSPRayRenderer) User override of scene epsilon: %g\n", scene_epsilon);
  }
  ospSet1f(ospRenderer, "epsilon", scene_epsilon);

  // render with/without shadows
  if (shadows_enabled || ao_samples) {
    if (shadows_enabled && !ao_samples)
      msgInfo << "Shadow rendering enabled." << sendmsg;

    ospSet1i(ospRenderer, "shadowsEnabled", 1);
  } else {
    ospSet1i(ospRenderer, "shadowsEnabled", 0);
  }

  // render with ambient occlusion, but only if shadows are also enabled
  if (ao_samples) {
    msgInfo << "Ambient occlusion enabled." << sendmsg;
    msgInfo << "Shadow rendering enabled." << sendmsg;
  }

  // commit triangle mesh geometry after assigning materials
  for (i=0; i<trimesh_v3f_n3f_c3f.num(); i++) {
    if (verbose == RT_VERB_DEBUG)
      printf("OSPRayRenderer) Adding triangle mesh[%d]: %d tris ...\n", 
             i, trimesh_v3f_n3f_c3f[i].num);

    ospAddGeometry(ospModel, trimesh_v3f_n3f_c3f[i].geom);
    ospSetMaterial(trimesh_v3f_n3f_c3f[i].geom, 
                   materialcache[trimesh_v3f_n3f_c3f[i].matindex].mat);
  } 

  // commit sphere geometry after assigning materials
  for (i=0; i<spheres_color.num(); i++) {
    if (verbose == RT_VERB_DEBUG)
      printf("OSPRayRenderer) Adding sphere_color array [%d]: %d spheres ...\n",
             i, spheres_color[i].num);

    ospAddGeometry(ospModel, spheres_color[i].geom);
    ospSetMaterial(spheres_color[i].geom, materialcache[spheres_color[i].matindex].mat);
  } 

  // commit cylinder geometry after assigning materials
  for (i=0; i<cylinders_color.num(); i++) {
    if (verbose == RT_VERB_DEBUG)
      printf("OSPRayRenderer) Adding cylinders_color array [%d]: %d cylinders...\n",
             i, cylinders_color[i].num);

    ospAddGeometry(ospModel, cylinders_color[i].geom);
    ospSetMaterial(cylinders_color[i].geom, materialcache[cylinders_color[i].matindex].mat);
  }

  // commit the completed scene
  ospCommit(ospModel);

  ospSetData(ospRenderer, "lights", ospLightData);
  ospSet3fv(ospRenderer, "bgColor", scene_bg_color);
  ospSetObject(ospRenderer, "camera", ospCamera);
  ospSetObject(ospRenderer, "world", ospModel);
//  ospSetObject(ospRenderer, "model", ospModel);
  ospCommit(ospRenderer);


  if (verbose == RT_VERB_DEBUG) printf("OSPRayRenderer) Finalizing OSPRay rendering kernels...\n");
  // XXX any last OSPRay state updates/checks

  double contextinittime = wkf_timer_timenow(ort_timer);
  time_ctx_validate = contextinittime - startctxtime;

  //
  // Force OSPRay to build the acceleration structure _now_, so we can time it
  //
  // XXX No way to force-build OSPRay AS for timing?

  time_ctx_AS_build = wkf_timer_timenow(ort_timer) - contextinittime;
  if (verbose == RT_VERB_DEBUG) {
    printf("OSPRayRenderer) launching render: %d x %d\n", width, height);
  }
}


#if defined(VMDOSPRAY_INTERACTIVE_OPENGL)

static void *createospraywindow(const char *wintitle, int width, int height) {
  printf("OSPRayRenderer) Creating OSPRay window: %d x %d...\n", width, height);

  void *win = glwin_create(wintitle, width, height);
  while (glwin_handle_events(win, GLWIN_EV_POLL_NONBLOCK) != 0);

  glDrawBuffer(GL_BACK);
  glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
  glClearColor(0.0, 0.0, 0.0, 1.0); /* black */
  glViewport(0, 0, width, height);
  glClear(GL_COLOR_BUFFER_BIT);

  glShadeModel(GL_FLAT);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.0, width, height, 0.0, -1.0, 1.0);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glDrawBuffer(GL_BACK);
  glClear(GL_COLOR_BUFFER_BIT);

  glwin_swap_buffers(win);

  return win;
}


static void interactive_viewer_usage(void *win) {
  printf("OSPRayRenderer) VMD TachyonL-OSPRay Interactive Ray Tracer help:\n");
  printf("OSPRayRenderer) ================================================\n");

  // check for Spaceball/SpaceNavigator/Magellan input devices
  int havespaceball = ((glwin_spaceball_available(win)) && (getenv("VMDDISABLESPACEBALLXDRV") == NULL));
  printf("OSPRayRenderer) Spaceball/SpaceNavigator/Magellan: %s\n",
         (havespaceball) ? "Available" : "Not available");

  // check for stereo-capable display
  int havestereo, havestencil;
  glwin_get_wininfo(win, &havestereo, &havestencil);
  printf("OSPRayRenderer) Stereoscopic display: %s\n",
         (havestereo) ? "Available" : "Not available");

  // check for vertical retrace sync
  int vsync=0, rc=0;
  if ((rc = glwin_query_vsync(win, &vsync)) == GLWIN_SUCCESS) {
    printf("OSPRayRenderer) Vert retrace sync: %s\n", (vsync) ? "On" : "Off");
  } else {
    printf("OSPRayRenderer) Vert retrace sync: indeterminate\n");
  }

  printf("OSPRayRenderer)\n");
  printf("OSPRayRenderer) General controls:\n");
  printf("OSPRayRenderer)   space: save numbered snapshot image\n");
  printf("OSPRayRenderer)       =: reset to initial view\n");
  printf("OSPRayRenderer)       h: print this help info\n");
  printf("OSPRayRenderer)       p: print current rendering parameters\n");
  printf("OSPRayRenderer)   ESC,q: quit viewer\n");
  printf("OSPRayRenderer)\n");
  printf("OSPRayRenderer) Display controls\n");
  printf("OSPRayRenderer)      F1: override shadows on/off (off=AO off too)\n");
  printf("OSPRayRenderer)      F2: override AO on/off\n");
  printf("OSPRayRenderer)      F3: override DoF on/off\n");
  printf("OSPRayRenderer)      F4: override Depth cueing on/off\n");
// Not currently applicable to OSPRay
// #ifdef USE_REVERSE_SHADOW_RAYS
//   printf("OSPRayRenderer)      F5: enable/disable shadow ray optimizations\n");
// #endif
  printf("OSPRayRenderer)     F12: toggle full-screen display on/off\n");
  printf("OSPRayRenderer)   1-9,0: override samples per update auto-FPS off\n");
  printf("OSPRayRenderer)      Up: increase DoF focal distance\n");
  printf("OSPRayRenderer)    Down: decrease DoF focal distance\n");
  printf("OSPRayRenderer)    Left: decrease DoF f/stop\n");
  printf("OSPRayRenderer)   Right: increase DoF f/stop\n");
  printf("OSPRayRenderer)       S: toggle stereoscopic display on/off (if avail)\n");
  printf("OSPRayRenderer)       a: toggle AA/AO auto-FPS tuning on/off (on)\n");
  printf("OSPRayRenderer)       g: toggle gradient sky xforms on/off (on)\n");
  printf("OSPRayRenderer)       l: toggle light xforms on/off (on)\n");
  printf("OSPRayRenderer)\n");
  printf("OSPRayRenderer) Mouse controls:\n");
  printf("OSPRayRenderer)       f: mouse depth-of-field mode\n");
  printf("OSPRayRenderer)       r: mouse rotation mode\n");
  printf("OSPRayRenderer)       s: mouse scaling mode\n");
  printf("OSPRayRenderer)       t: mouse translation mode\n");

  int movie_recording_enabled = (getenv("VMDOSPRAYLIVEMOVIECAPTURE") != NULL);
  if (movie_recording_enabled) {
    printf("OSPRayRenderer)\n");
    printf("OSPRayRenderer) Movie recording controls:\n");
    printf("OSPRayRenderer)       R: start/stop movie recording\n");
    printf("OSPRayRenderer)       F: toggle movie FPS (24, 30, 60)\n");
  }
}


void OSPRayRenderer::render_to_glwin(const char *filename) {
  DBG();
  int i;

  if (!context_created)
    return;

  enum RtMouseMode { RTMM_ROT=0, RTMM_TRANS=1, RTMM_SCALE=2, RTMM_DOF=3 };
  enum RtMouseDown { RTMD_NONE=0, RTMD_LEFT=1, RTMD_MIDDLE=2, RTMD_RIGHT=3 };
  RtMouseMode mm = RTMM_ROT;
  RtMouseDown mousedown = RTMD_NONE;

  // flags to interactively enable/disable shadows, AO, DoF
  int gl_shadows_on=(shadows_enabled) ? RT_SHADOWS_ON : RT_SHADOWS_OFF;

  int gl_fs_on=0; // fullscreen window state
  int owsx=0, owsy=0; // store last win size before fullscreen
  int gl_ao_on=(ao_samples > 0);
  int gl_dof_on, gl_dof_on_old;
  gl_dof_on=gl_dof_on_old=dof_enabled; 
  int gl_fog_on=(fog_mode != RT_FOG_NONE);

  // Enable live recording of a session to a stream of image files indexed
  // by their display presentation time, mapped to the nearest frame index
  // in a fixed-frame-rate image sequence (e.g. 24, 30, or 60 FPS), 
  // to allow subsequent encoding into a standard movie format.
  // XXX this feature is disabled by default at present, to prevent people
  //     from accidentally turning it on during a live demo or the like
  int movie_recording_enabled = (getenv("VMDOSPRAYLIVEMOVIECAPTURE") != NULL);
  int movie_recording_on = 0;
  double movie_recording_start_time = 0.0;
  int movie_recording_fps = 30;
  int movie_framecount = 0;
  int movie_lastframeindex = 0;
  const char *movie_recording_filebase = "vmdlivemovie.%05d.tga";
  if (getenv("VMDOSPRAYLIVEMOVIECAPTUREFILEBASE"))
    movie_recording_filebase = getenv("VMDOSPRAYLIVEMOVIECAPTUREFILEBASE");

  // Enable/disable Spaceball/SpaceNavigator/Magellan input 
  int spaceballenabled=(getenv("VMDDISABLESPACEBALLXDRV") == NULL) ? 1 : 0;
  int spaceballmode=0;       // default mode is rotation/translation
  int spaceballflightmode=0; // 0=moves object, 1=camera fly
  if (getenv("VMDOSPRAYSPACEBALLFLIGHT"))
    spaceballflightmode=1;


  // total AA/AO sample count
  int totalsamplecount=0;

  // counter for snapshots of live image...
  int snapshotcount=0;

  // flag to enable automatic AO sample count adjustment for FPS rate control
  int autosamplecount=1;

  // flag to enable transformation of lights and gradient sky sphere, 
  // so that they track camera orientation as they do in the VMD OpenGL display
  int xformlights=1, xformgradientsphere=1;

  //
  // allocate or reconfigure the framebuffer, accumulation buffer, 
  // and output streams required for progressive rendering, either
  // using the new progressive APIs, or using our own code.
  //
  // Unless overridden by environment variables, we use the incoming
  // window size parameters from VMD to initialize the RT image dimensions.
  // If image size is overridden, often when using HMDs, the incoming 
  // dims are window dims are used to size the GL window, but the image size
  // is set independently.
  int wsx=width, wsy=height;
  const char *imageszstr = getenv("VMDOSPRAYIMAGESIZE");
  if (imageszstr) {
    if (sscanf(imageszstr, "%d %d", &width, &height) != 2) {
      width=wsx;
      height=wsy;
    } 
  } 
  config_framebuffer(width, height);

  // prepare the majority of OSPRay rendering state before we go into 
  // the interactive rendering loop
  update_rendering_state(1);
  render_compile_and_validate();

  // make a copy of state we're going to interactively manipulate,
  // so that we can recover to the original state on-demand
  int samples_per_pass = 1;
  int cur_aa_samples = aa_samples;
  int cur_ao_samples = ao_samples;
  float cam_zoom_orig = cam_zoom;
  float scene_gradient_orig[3] = {0.0f, 1.0f, 0.0f};
  vec_copy(scene_gradient_orig, scene_gradient);

  float cam_pos_orig[3] = {0.0f, 0.0f, 2.0f};
  float cam_U_orig[3] = {1.0f, 0.0f, 0.0f};
  float cam_V_orig[3] = {0.0f, 1.0f, 0.0f};
  float cam_W_orig[3] = {0.0f, 0.0f, -1.0f};
  float cam_pos[3], cam_U[3], cam_V[3], cam_W[3];
  float hmd_U[3], hmd_V[3], hmd_W[3];

  vec_copy(cam_pos, cam_pos_orig);
  vec_copy(cam_U, cam_U_orig);
  vec_copy(cam_V, cam_V_orig);
  vec_copy(cam_W, cam_W_orig);

  // copy light directions
  ort_directional_light *cur_dlights = (ort_directional_light *) calloc(1, directional_lights.num() * sizeof(ort_directional_light));
  for (i=0; i<directional_lights.num(); i++) {
    vec_copy((float*)&cur_dlights[i].color, directional_lights[i].color);
    vec_copy((float*)&cur_dlights[i].dir, directional_lights[i].dir);
    vec_normalize((float*)&cur_dlights[i].dir);
  }

  // create the display window
  void *win = createospraywindow("VMD TachyonL-OSPRay Interactive Ray Tracer", width, height);
  interactive_viewer_usage(win);
  
  // check for stereo-capable display
  int havestereo=0, havestencil=0;
  int stereoon=0, stereoon_old=0;
  glwin_get_wininfo(win, &havestereo, &havestencil);

  // Override AA/AO sample counts since we're doing progressive rendering.
  // Choosing an initial AO sample count of 1 will give us the peak progressive 
  // display update rate, but we end up wasting time on re-tracing many
  // primary rays.  The automatic FPS optimization scheme below will update
  // the number of samples per rendering pass and assign the best values for
  // AA/AO samples accordingly.
  cur_aa_samples = samples_per_pass;
  if (cur_ao_samples > 0) {
    cur_aa_samples = 1;
    cur_ao_samples = samples_per_pass;
  }

  const char *statestr = "|/-\\.";
  int done=0, winredraw=1, accum_count=0;
  int state=0, mousedownx=0, mousedowny=0;
  float cur_cam_zoom = cam_zoom_orig;

  double fpsexpave=0.0; 
  double mapbuftotaltime=0.0;
  
  double oldtime = wkf_timer_timenow(ort_timer);
  while (!done) { 
    int winevent=0;

    while ((winevent = glwin_handle_events(win, GLWIN_EV_POLL_NONBLOCK)) != 0) {
      int evdev, evval;
      char evkey;

      glwin_get_lastevent(win, &evdev, &evval, &evkey);
      glwin_get_winsize(win, &wsx, &wsy);

      if (evdev == GLWIN_EV_WINDOW_CLOSE) {
        printf("OSPRayRenderer) display window closed, exiting...\n");
        done = 1;
        winredraw = 0;
      } else if (evdev == GLWIN_EV_KBD) {
        switch (evkey) {
          case  '1': autosamplecount=0; samples_per_pass=1; winredraw=1; break;
          case  '2': autosamplecount=0; samples_per_pass=2; winredraw=1; break;
          case  '3': autosamplecount=0; samples_per_pass=3; winredraw=1; break;
          case  '4': autosamplecount=0; samples_per_pass=4; winredraw=1; break;
          case  '5': autosamplecount=0; samples_per_pass=5; winredraw=1; break;
          case  '6': autosamplecount=0; samples_per_pass=6; winredraw=1; break;
          case  '7': autosamplecount=0; samples_per_pass=7; winredraw=1; break;
          case  '8': autosamplecount=0; samples_per_pass=8; winredraw=1; break;
          case  '9': autosamplecount=0; samples_per_pass=9; winredraw=1; break;
          case  '0': autosamplecount=0; samples_per_pass=10; winredraw=1; break;

          case  '=': /* recover back to initial state */
            vec_copy(scene_gradient, scene_gradient_orig);
            cam_zoom = cam_zoom_orig;
            vec_copy(cam_pos, cam_pos_orig);
            vec_copy(cam_U, cam_U_orig);
            vec_copy(cam_V, cam_V_orig);
            vec_copy(cam_W, cam_W_orig);

            // restore original light directions
            for (i=0; i<directional_lights.num(); i++) {
              vec_copy((float*)&cur_dlights[i].dir, directional_lights[i].dir);
              vec_normalize((float*)&cur_dlights[i].dir);
            }
            winredraw = 1;
            break;
 
          case  ' ': /* spacebar saves current image with counter */
            {
              char snapfilename[256];
              sprintf(snapfilename, "vmdsnapshot.%04d.tga", snapshotcount);
              const unsigned char *FB = (const unsigned char*)ospMapFrameBuffer(ospFrameBuffer, OSP_FB_COLOR);
              if (write_image_file_rgb4u(snapfilename, FB, width, height)) {
                printf("OSPRayRenderer) Failed to write output image!\n");
              } else {
                printf("OSPRayRenderer) Saved snapshot to '%s'             \n",
                       snapfilename);
              }
              ospUnmapFrameBuffer(FB, ospFrameBuffer);
              snapshotcount++; 
            }
            break;

          case  'a': /* toggle automatic sample count FPS tuning */
            autosamplecount = !(autosamplecount);
            printf("\nOSPRayRenderer) Automatic AO sample count FPS tuning %s\n",
                   (autosamplecount) ? "enabled" : "disabled");
            break;

          case  'f': /* DoF mode */
            mm = RTMM_DOF;
            printf("\nOSPRayRenderer) Mouse DoF aperture and focal dist. mode\n");
            break;

          case  'g': /* toggle gradient sky sphere xforms */
            xformgradientsphere = !(xformgradientsphere);
            printf("\nOSPRayRenderer) Gradient sky sphere transformations %s\n",
                   (xformgradientsphere) ? "enabled" : "disabled");
            break;

          case  'h': /* print help message */
            printf("\n");
            interactive_viewer_usage(win);
            break;

          case  'l': /* toggle lighting xforms */
            xformlights = !(xformlights);
            printf("\nOSPRayRenderer) Light transformations %s\n",
                   (xformlights) ? "enabled" : "disabled");
            break;

          case  'p': /* print current RT settings */
            printf("\nOSPRayRenderer) Current Ray Tracing Parameters:\n"); 
            printf("OSPRayRenderer) -------------------------------\n"); 
            printf("OSPRayRenderer) Camera zoom: %f\n", cur_cam_zoom);
            printf("OSPRayRenderer) Shadows: %s  Ambient occlusion: %s\n",
                   (gl_shadows_on) ? "on" : "off",
                   (gl_ao_on) ? "on" : "off");
            printf("OSPRayRenderer) Antialiasing samples per-pass: %d\n",
                   cur_aa_samples);
            printf("OSPRayRenderer) Ambient occlusion samples per-pass: %d\n",
                   cur_ao_samples);
            printf("OSPRayRenderer) Depth-of-Field: %s f/num: %.1f  Foc. Dist: %.2f\n",
                   (gl_dof_on) ? "on" : "off", 
                   cam_dof_fnumber, cam_dof_focal_dist);
            printf("OSPRayRenderer) Image size: %d x %d\n", width, height);
            break;

          case  'r': /* rotate mode */
            mm = RTMM_ROT;
            printf("\nOSPRayRenderer) Mouse rotation mode\n");
            break;

          case  's': /* scaling mode */
            mm = RTMM_SCALE;
            printf("\nOSPRayRenderer) Mouse scaling mode\n");
            break;

          case  'F': /* toggle live movie recording FPS (24, 30, 60) */
            if (movie_recording_enabled) {
              switch (movie_recording_fps) {
                case 24: movie_recording_fps = 30; break;
                case 30: movie_recording_fps = 60; break;
                case 60:
                default: movie_recording_fps = 24; break;
              }
              printf("\nOSPRayRenderer) Movie recording FPS rate: %d\n", 
                     movie_recording_fps);
            } else {
              printf("\nOSPRayRenderer) Movie recording not available.\n");
            }
            break;

          case  'R': /* toggle live movie recording mode on/off */
            if (movie_recording_enabled) {
              movie_recording_on = !(movie_recording_on);
              printf("\nOSPRayRenderer) Movie recording %s\n",
                     (movie_recording_on) ? "STARTED" : "STOPPED");
              if (movie_recording_on) {
                movie_recording_start_time = wkf_timer_timenow(ort_timer);
                movie_framecount = 0;
                movie_lastframeindex = 0;
              } else {
                printf("OSPRayRenderer) Encode movie with:\n");
                printf("OSPRayRenderer)   ffmpeg -f image2 -i vmdlivemovie.%%05d.tga -c:v libx264 -profile:v baseline -level 3.0 -pix_fmt yuv420p -b:v 15000000 output.mp4\n");
              }
            } else {
              printf("\nOSPRayRenderer) Movie recording not available.\n");
            }
            break;

          case  'S': /* toggle stereoscopic display mode */
            if (havestereo) {
              stereoon = (!stereoon);
              printf("\nOSPRayRenderer) Stereoscopic display %s\n",
                     (stereoon) ? "enabled" : "disabled");
              winredraw = 1;
            } else {
              printf("\nOSPRayRenderer) Stereoscopic display unavailable\n");
            }
            break;
 
          case  't': /* translation mode */
            mm = RTMM_TRANS;
            printf("\nOSPRayRenderer) Mouse translation mode\n");
            break;
            
          case  'q': /* 'q' key */
          case  'Q': /* 'Q' key */
          case 0x1b: /* ESC key */
            printf("\nOSPRayRenderer) Exiting on user input.               \n");
            done=1; /* exit from interactive RT window */
            break;
        }
      } else if (evdev != GLWIN_EV_NONE) {
        switch (evdev) {
          case GLWIN_EV_KBD_F1: /* turn shadows on/off */
            gl_shadows_on=(!gl_shadows_on) ? RT_SHADOWS_ON : RT_SHADOWS_OFF;
            // gl_shadows_on = (!gl_shadows_on);
            printf("\n");
            printf("OSPRayRenderer) Shadows %s\n",
                   (gl_shadows_on) ? "enabled" : "disabled");
            winredraw = 1; 
            break;

          case GLWIN_EV_KBD_F2: /* turn AO on/off */
            gl_ao_on = (!gl_ao_on); 
            printf("\n");
            printf("OSPRayRenderer) Ambient occlusion %s\n",
                   (gl_ao_on) ? "enabled" : "disabled");
            winredraw = 1; 
            break;

          case GLWIN_EV_KBD_F3: /* turn DoF on/off */
            gl_dof_on = (!gl_dof_on);
            printf("\n");
            if ((camera_projection == RT_ORTHOGRAPHIC) && gl_dof_on) {
              gl_dof_on=0; 
              printf("OSPRayRenderer) Depth-of-field not available in orthographic mode\n");
            }
            printf("OSPRayRenderer) Depth-of-field %s\n",
                   (gl_dof_on) ? "enabled" : "disabled");
            winredraw = 1;
            break;

          case GLWIN_EV_KBD_F4: /* turn fog/depth cueing on/off */
            gl_fog_on = (!gl_fog_on); 
            printf("\n");
            printf("OSPRayRenderer) Depth cueing %s\n",
                   (gl_fog_on) ? "enabled" : "disabled");
            winredraw = 1; 
            break;

          case GLWIN_EV_KBD_F12: /* toggle full-screen window on/off */
            gl_fs_on = (!gl_fs_on);
            printf("\nOSPRayRenderer) Toggling fullscreen window %s\n",
                   (gl_fs_on) ? "on" : "off");
            if (gl_fs_on) { 
              if (glwin_fullscreen(win, gl_fs_on, 0) == 0) {
                owsx = wsx;
                owsy = wsy;
                glwin_get_winsize(win, &wsx, &wsy);
              } else {
                printf("OSPRayRenderer) Fullscreen mode note available\n");
              }
            } else {
              glwin_fullscreen(win, gl_fs_on, 0);
              glwin_resize(win, owsx, owsy);
            }
            winredraw = 1; 
            break;

          case GLWIN_EV_KBD_UP: /* change depth-of-field focal dist */
            cam_dof_focal_dist *= 1.02f; 
            printf("\nOSPRayRenderer) DoF focal dist: %f\n", cam_dof_focal_dist);
            winredraw = 1; 
            break;

          case GLWIN_EV_KBD_DOWN: /* change depth-of-field focal dist */
            cam_dof_focal_dist *= 0.96f; 
            if (cam_dof_focal_dist < 0.02f) cam_dof_focal_dist = 0.02f;
            printf("\nOSPRayRenderer) DoF focal dist: %f\n", cam_dof_focal_dist);
            winredraw = 1; 
            break;

          case GLWIN_EV_KBD_RIGHT: /* change depth-of-field f/stop number */
            cam_dof_fnumber += 1.0f; 
            printf("\nOSPRayRenderer) DoF f/stop: %f\n", cam_dof_fnumber);
            winredraw = 1; 
            break;

          case GLWIN_EV_KBD_LEFT: /* change depth-of-field f/stop number */
            cam_dof_fnumber -= 1.0f; 
            if (cam_dof_fnumber < 1.0f) cam_dof_fnumber = 1.0f;
            printf("\nOSPRayRenderer) DoF f/stop: %f\n", cam_dof_fnumber);
            winredraw = 1; 
            break;

          case GLWIN_EV_MOUSE_MOVE:
            if (mousedown != RTMD_NONE) {
              int x, y;
              glwin_get_mousepointer(win, &x, &y);

              float zoommod = 2.0f*cur_cam_zoom/cam_zoom_orig;
              float txdx = (x - mousedownx) * zoommod / wsx;
              float txdy = (y - mousedowny) * zoommod / wsy;
              if (mm != RTMM_SCALE) {
                mousedownx = x;
                mousedowny = y;
              }

              if (mm == RTMM_ROT) {
                Matrix4 rm;
                if (mousedown == RTMD_LEFT) {
                  // when zooming in further from the initial view, we
                  // rotate more slowly so control remains smooth
                  rm.rotate_axis(cam_V, -txdx);
                  rm.rotate_axis(cam_U, -txdy);
                } else if (mousedown == RTMD_MIDDLE || 
                           mousedown == RTMD_RIGHT) {
                  rm.rotate_axis(cam_W, txdx);
                }
                rm.multpoint3d(cam_pos, cam_pos);
                rm.multnorm3d(cam_U, cam_U);
                rm.multnorm3d(cam_V, cam_V);
                rm.multnorm3d(cam_W, cam_W);

                if (xformgradientsphere) {
                  rm.multnorm3d(scene_gradient, scene_gradient);
                }
 
                if (xformlights) {
                  // update light directions (comparatively costly)
                  for (i=0; i<directional_lights.num(); i++) {
                    rm.multnorm3d((float*)&cur_dlights[i].dir, (float*)&cur_dlights[i].dir);
                  }
                }

                winredraw = 1;
              } else if (mm == RTMM_TRANS) {
                if (mousedown == RTMD_LEFT) {
                  float dU[3], dV[3];
                  vec_scale(dU, -txdx, cam_U);
                  vec_scale(dV,  txdy, cam_V);
                  vec_add(cam_pos, cam_pos, dU); 
                  vec_add(cam_pos, cam_pos, dV); 
                } else if (mousedown == RTMD_MIDDLE || 
                           mousedown == RTMD_RIGHT) {
                  float dW[3];
                  vec_scale(dW, txdx, cam_W);
                  vec_add(cam_pos, cam_pos, dW); 
                } 
                winredraw = 1;
              } else if (mm == RTMM_SCALE) {
                float txdx = (x - mousedownx) * 2.0 / wsx;
                float zoominc = 1.0 - txdx;
                if (zoominc < 0.01) zoominc = 0.01;
                cam_zoom = cur_cam_zoom * zoominc;
                winredraw = 1;
              } else if (mm == RTMM_DOF) {
                cam_dof_fnumber += txdx * 20.0f;
                if (cam_dof_fnumber < 1.0f) cam_dof_fnumber = 1.0f;
                cam_dof_focal_dist += -txdy; 
                if (cam_dof_focal_dist < 0.01f) cam_dof_focal_dist = 0.01f;
                winredraw = 1;
              }
            }
            break;

          case GLWIN_EV_MOUSE_LEFT:
          case GLWIN_EV_MOUSE_MIDDLE:
          case GLWIN_EV_MOUSE_RIGHT:
            if (evval) {
              glwin_get_mousepointer(win, &mousedownx, &mousedowny);
              cur_cam_zoom = cam_zoom;

              if (evdev == GLWIN_EV_MOUSE_LEFT) mousedown = RTMD_LEFT;
              else if (evdev == GLWIN_EV_MOUSE_MIDDLE) mousedown = RTMD_MIDDLE;
              else if (evdev == GLWIN_EV_MOUSE_RIGHT) mousedown = RTMD_RIGHT;
            } else {
              mousedown = RTMD_NONE;
            }
            break;

          case GLWIN_EV_MOUSE_WHEELUP:
            cam_zoom /= 1.1f; winredraw = 1; break;

          case GLWIN_EV_MOUSE_WHEELDOWN:
            cam_zoom *= 1.1f; winredraw = 1; break;
        }
      }
    }


    //
    // Support for Spaceball/Spacenavigator/Magellan devices that use
    // X11 ClientMessage protocol....
    //
    //
    // Support for Spaceball/Spacenavigator/Magellan devices that use
    // X11 ClientMessage protocol....
    //
    if (spaceballenabled) {
      // Spaceball/Spacenavigator/Magellan event state variables
      int tx=0, ty=0, tz=0, rx=0, ry=0, rz=0, buttons=0;
      if (glwin_get_spaceball(win, &rx, &ry, &rz, &tx, &ty, &tz, &buttons)) {
        // negate directions if we're in flight mode...
        if (spaceballflightmode) {
          rx= -rx;
          ry= -ry;
          rz= -rz;

          tx= -tx;
          ty= -ty;
          tz= -tz;
        }

        // check for button presses to reset the view
        if (buttons & 1) {
          printf("OSPRayRenderer) spaceball button 1 pressed: reset view\n");
          vec_copy(scene_gradient, scene_gradient_orig);
          cam_zoom = cam_zoom_orig;
          vec_copy(cam_pos, cam_pos_orig);
          vec_copy(cam_U, cam_U_orig);
          vec_copy(cam_V, cam_V_orig);
          vec_copy(cam_W, cam_W_orig);

          // restore original light directions
          for (i=0; i<directional_lights.num(); i++) {
            vec_copy((float*)&cur_dlights[i].dir, directional_lights[i].dir);
            vec_normalize((float*)&cur_dlights[i].dir);
          }
          winredraw = 1;
        }

        // check for button presses to toggle spaceball mode
        if (buttons & 2) {
          spaceballmode = !(spaceballmode);
          printf("OSPRayRenderer) spaceball mode: %s                       \n",
                 (spaceballmode) ? "scaling" : "rotation/translation");
        }

        // rotation/translation mode
        if (spaceballmode == 0) {
          float zoommod = 2.0f*cam_zoom/cam_zoom_orig;
          float divlen = sqrtf(wsx*wsx + wsy*wsy) * 50;

          // check for rotation and handle it...
          if (rx != 0 || ry !=0 || rz !=0) {
            Matrix4 rm;
            rm.rotate_axis(cam_U, -rx * zoommod / divlen);
            rm.rotate_axis(cam_V, -ry * zoommod / divlen);
            rm.rotate_axis(cam_W, -rz * zoommod / divlen);

            rm.multpoint3d(cam_pos, cam_pos);
            rm.multnorm3d(cam_U, cam_U);
            rm.multnorm3d(cam_V, cam_V);
            rm.multnorm3d(cam_W, cam_W);

            if (xformgradientsphere) {
              rm.multnorm3d(scene_gradient, scene_gradient);
            }

            if (xformlights) {
              // update light directions (comparatively costly)
              for (i=0; i<directional_lights.num(); i++) {
                rm.multnorm3d((float*)&cur_dlights[i].dir, (float*)&cur_dlights[i].dir);
              }
            }
            winredraw = 1;
          }

          // check for translation and handle it...
          if (tx != 0 || ty !=0 || tz !=0) {
            float dU[3], dV[3], dW[3];
            vec_scale(dU, -tx * zoommod / divlen, cam_U);
            vec_scale(dV, -ty * zoommod / divlen, cam_V);
            vec_scale(dW, -tz * zoommod / divlen, cam_W);
            vec_add(cam_pos, cam_pos, dU);
            vec_add(cam_pos, cam_pos, dV);
            vec_add(cam_pos, cam_pos, dW);
            winredraw = 1;
          }
        }
    
        // scaling mode
        if (spaceballmode == 1) {
          const float sbscale = 1.0f / (1024.0f * 8.0f);
          float zoominc = 1.0f - (rz * sbscale);
          if (zoominc < 0.01) zoominc = 0.01;
            cam_zoom *= zoominc;
            winredraw = 1;
        }

      }
    }


    // if there is no HMD, we use the camera orientation directly  
    vec_copy(hmd_U, cam_U);
    vec_copy(hmd_V, cam_V);
    vec_copy(hmd_W, cam_W);

    // XXX HMD handling goes here

    //
    // handle window resizing, stereoscopic mode changes,
    // destroy and recreate affected OSPRay buffers
    //
    int resize_buffers=0;

    if (wsx != width) {
      width = wsx;
      resize_buffers=1;
    }
 
    if (wsy != height || (stereoon != stereoon_old)) {
      if (stereoon) {
        if (height != wsy * 2) {
          height = wsy * 2; 
          resize_buffers=1;
        }
      } else {
        height = wsy;
        resize_buffers=1;
      }
    }


    // check if stereo mode or DoF mode changed, both cases
    // require changing the active color accumulation ray gen program
    if ((stereoon != stereoon_old) || (gl_dof_on != gl_dof_on_old)) {
      // when stereo mode changes, we have to regenerate the
      // the RNG, accumulation buffer, and framebuffer
      if (stereoon != stereoon_old) {
        resize_buffers=1;
      }

      // update stereo and DoF state
      stereoon_old = stereoon;
      gl_dof_on_old = gl_dof_on;

      // set the active color accumulation ray gen mode based on the 
      // camera/projection mode, stereoscopic display mode, 
      // and depth-of-field state
      winredraw=1;
    }

    if (resize_buffers) {
      resize_framebuffer(width, height);

      // when movie recording is enabled, print the window size as a guide
      // since the user might want to precisely control the size or 
      // aspect ratio for a particular movie format, e.g. 1080p, 4:3, 16:9
      if (movie_recording_enabled) {
        printf("\rOSPRayRenderer) Window resize: %d x %d                               \n", width, height);
      }

      winredraw=1;
    }

    int frame_ready = 1; // Default to true
    unsigned int subframe_count = 1;
    if (!done) {
      //
      // If the user interacted with the window in a meaningful way, we
      // need to update the OSPRay rendering state, recompile and re-validate
      // the context, and then re-render...
      //
      if (winredraw) {
        // update camera parameters
        ospSet3fv(ospCamera, "pos", cam_pos);
        ospSet3fv(ospCamera, "dir",   hmd_W);
        ospSet3fv(ospCamera,  "up",   hmd_V);
        ospSet1f(ospCamera, "aspect", width / ((float) height));
        ospSet1f(ospCamera, "fovy", 2.0f*180.0f*(atanf(cam_zoom)/M_PI));
 
        // update shadow state 
        ospSet1i(ospRenderer, "shadowsEnabled", gl_shadows_on);

        // update AO state 
        if (gl_shadows_on && gl_ao_on) {
          ospSet1i(ospRenderer, "aoSamples", 1);
        } else {
          ospSet1i(ospRenderer, "aoSamples", 0);
        }

        // update depth cueing state
        // XXX update OSPRay depth cueing state
 
        // update/recompute DoF values 
        // XXX OSPRay only implements DoF for the perspective
        //     camera at the present time
        if (camera_projection == OSPRayRenderer::RT_PERSPECTIVE) {
          if (gl_dof_on) {
            ospSet1f(ospCamera, "focusDistance", cam_dof_focal_dist);
            ospSet1f(ospCamera, "apertureRadius", cam_dof_focal_dist / (2.0f * cam_zoom * cam_dof_fnumber));
          } else {
            ospSet1f(ospCamera, "apertureRadius", 0.0f);
          }
        }

        // commit camera updates once they're all done...
        ospCommit(ospCamera);

        //
        // Update light directions in the OSPRay light buffers
        //
        if (xformlights) {
          // AO scaling factor is applied at the renderer level, but
          // we apply the direct lighting scaling factor to the lights.
          float lightscale = 1.0f;
          if (ao_samples != 0)
            lightscale = ao_direct;

          // XXX assumes the only contents in the first part of the 
          //     light list are directional lights.  The new AO "ambient"
          //     light is the last light in the list now, so we can get
          //     away with this, but refactoring is still needed here.
          for (i=0; i<directional_lights.num(); i++) {
            ospSet1f(ospLights[i], "intensity", lightscale);

            ospSet3fv(ospLights[i], "color", cur_dlights[i].color);

            ospSet3f(ospLights[i], "direction",
                     -cur_dlights[i].dir[0],
                     -cur_dlights[i].dir[1],
                     -cur_dlights[i].dir[2]);

            ospCommit(ospLights[i]);
          }
        }

        // commit pending changes...
        ospCommit(ospRenderer);

        // reset accumulation buffer 
        accum_count=0;
        totalsamplecount=0;
        if (ospFrameBuffer != NULL) {
          ospFrameBufferClear(ospFrameBuffer, OSP_FB_COLOR | /* OSP_FB_DEPTH | */ OSP_FB_ACCUM);
        }

        // 
        // Sample count updates and OSPRay state must always remain in 
        // sync, so if we only update sample count state during redraw events,
        // that's the only time we should recompute the sample counts, since
        // they also affect normalization factors for the accumulation buffer
        //

        // Update sample counts to achieve target interactivity
        if (autosamplecount) {
          if (fpsexpave > 37)
            samples_per_pass++;
          else if (fpsexpave < 30) 
            samples_per_pass--;
    
          // clamp sample counts to a "safe" range
          if (samples_per_pass > 14)
            samples_per_pass=14;
          if (samples_per_pass < 1)
            samples_per_pass=1;
        } 

        // split samples per pass either among AA and AO, depending on
        // whether DoF and AO are enabled or not. 
        if (gl_shadows_on && gl_ao_on) {
          if (gl_dof_on) {
            if (samples_per_pass < 4) {
              cur_aa_samples=samples_per_pass;
              cur_ao_samples=1;
            } else {
              int s = (int) sqrtf(samples_per_pass);
              cur_aa_samples=s;
              cur_ao_samples=s;
            }
          } else {
            cur_aa_samples=1;
            cur_ao_samples=samples_per_pass;
          }
        } else {
          cur_aa_samples=samples_per_pass;
          cur_ao_samples=0;
        }

        // update the current AA/AO sample counts since they may be changing if
        // FPS autotuning is enabled...
        // XXX update OSPRay AA sample counts

        // observe latest AO enable/disable flag, and sample count
        if (gl_shadows_on && gl_ao_on) {
          // XXX update OSPRay AA/AO sample counts
        } else {
          cur_ao_samples = 0;
          // XXX update OSPRay AA/AO sample counts
        }
      } 


      // The accumulation buffer normalization factor must be updated
      // to reflect the total accumulation count before the accumulation
      // buffer is drawn to the output framebuffer
      // XXX update OSPRay accum buf normalization factor

      // The accumulation buffer subframe index must be updated to ensure that
      // the RNGs for AA and AO get correctly re-seeded
      // XXX update OSPRay accum subframe count

      // Force context compilation/validation
      // render_compile_and_validate();

      //
      // run the renderer 
      //
      frame_ready = 1; // Default to true
      subframe_count = 1;
      if (lasterror == 0 /* XXX SUCCESS */) {
        if (winredraw) {
          ospFrameBufferClear(ospFrameBuffer, OSP_FB_COLOR | /* OSP_FB_DEPTH | */ OSP_FB_ACCUM);
          winredraw=0;
        }

        // iterate, adding to the accumulation buffer...
//printf("OSPRayRenderer) ospRenderFrame(): [%d] ...\n", accum_sample);
        ospRenderFrame(ospFrameBuffer, ospRenderer, OSP_FB_COLOR | OSP_FB_ACCUM);
        subframe_count++; // increment subframe index
        totalsamplecount += samples_per_pass;
        accum_count += cur_aa_samples;

        // copy the accumulation buffer image data to the framebuffer and
        // perform type conversion and normaliztion on the image data...
        // XXX launch OSPRay accum copy/norm/finish

        if (lasterror == 0 /* XXX SUCCESS */) {
          if (frame_ready) {
            double bufnewtime = wkf_timer_timenow(ort_timer);

            // display output image
            const unsigned char * img;
            img = (const unsigned char*)ospMapFrameBuffer(ospFrameBuffer, OSP_FB_COLOR);

#if 0
            glwin_draw_image_tex_rgb3u(win, (stereoon!=0)*GLWIN_STEREO_OVERUNDER, width, height, img);
#else
            glwin_draw_image_rgb3u(win, (stereoon!=0)*GLWIN_STEREO_OVERUNDER, width, height, img);
#endif
            ospUnmapFrameBuffer(img, ospFrameBuffer);
            mapbuftotaltime = wkf_timer_timenow(ort_timer) - bufnewtime;


            // if live movie recording is on, we save every displayed frame
            // to a sequence sequence of image files, with each file numbered
            // by its frame index, which is computed by the multiplying image
            // presentation time by the image sequence fixed-rate-FPS value.
            if (movie_recording_enabled && movie_recording_on) {
              char moviefilename[2048];

              // compute frame number from wall clock time and the
              // current fixed-rate movie playback frame rate
              double now = wkf_timer_timenow(ort_timer);
              double frametime = now - movie_recording_start_time;
              int fidx = frametime * movie_recording_fps;

              // always force the first recorded frame to be 0
              if (movie_framecount==0)
                fidx=0;
              movie_framecount++;

#if defined(__linux)
              // generate symlinks for frame indices between the last written
              // frame and the current one so that video encoders such as
              // ffmpeg and mencoder can be fed the contiguous frame sequence
              // at a fixed frame rate, as they require
              sprintf(moviefilename, movie_recording_filebase,
                      movie_lastframeindex);
              int symidx;
              for (symidx=movie_lastframeindex; symidx<fidx; symidx++) {
                char symlinkfilename[2048];
                sprintf(symlinkfilename, movie_recording_filebase, symidx);
                symlink(moviefilename, symlinkfilename);
              }
#endif

              // write the new movie frame
              sprintf(moviefilename, movie_recording_filebase, fidx);
              const unsigned char *FB = (const unsigned char*)ospMapFrameBuffer(ospFrameBuffer, OSP_FB_COLOR);
              if (write_image_file_rgb4u(moviefilename, FB, width, height)) {
                movie_recording_on = 0;
                printf("\n");
                printf("OSPRayRenderer) ERROR during writing image during movie recording!\n");
                printf("OSPRayRenderer) Movie recording STOPPED\n");
              }
              ospUnmapFrameBuffer(FB, ospFrameBuffer);

              movie_lastframeindex = fidx; // update last frame index written
            }
          }
        } else {
          printf("OSPRayRenderer) An error occured during rendering. Rendering is aborted.\n");
          done=1;
          break;
        }
      } else {
        printf("OSPRayRenderer) An error occured in AS generation. Rendering is aborted.\n");
        done=1;
        break;
      }
    }

    if (!done && frame_ready) {
      double newtime = wkf_timer_timenow(ort_timer);
      double frametime = (newtime-oldtime) + 0.00001f;
      oldtime=newtime;

      // compute exponential moving average for exp(-1/10)
      double framefps = 1.0f/frametime;
      fpsexpave = (fpsexpave * 0.90) + (framefps * 0.10);

      printf("OSPRayRenderer) %c AA:%2d AO:%2d, %4d tot RT FPS: %.1f  %.4f s/frame sf: %d  \r",
             statestr[state], cur_aa_samples, cur_ao_samples, 
             totalsamplecount, fpsexpave, frametime, subframe_count);

      fflush(stdout);
      state = (state+1) & 3;
    }

  } // end of per-cycle event processing

  printf("\n");

  // write the output image upon exit...
  if (lasterror == 0 /* XXX SUCCESS */) {
    wkf_timer_start(ort_timer);
    // write output image
    const unsigned char *FB = (const unsigned char*)ospMapFrameBuffer(ospFrameBuffer, OSP_FB_COLOR);
    if (write_image_file_rgb4u(filename, FB, width, height)) {
      printf("OSPRayRenderer) Failed to write output image!\n");
    }
    ospUnmapFrameBuffer(FB, ospFrameBuffer);
    wkf_timer_stop(ort_timer);

    if (verbose == RT_VERB_TIMING || verbose == RT_VERB_DEBUG) {
      printf("OSPRayRenderer) image file I/O time: %f secs\n", wkf_timer_time(ort_timer));
    }
  }

  glwin_destroy(win);
}

#endif


void OSPRayRenderer::render_to_file(const char *filename) {
  DBG();
  if (!context_created)
    return;

  // Unless overridden by environment variables, we use the incoming
  // window size parameters from VMD to initialize the RT image dimensions.
  int wsx=width, wsy=height;
  const char *imageszstr = getenv("VMDOSPRAYIMAGESIZE");
  if (imageszstr) {
    if (sscanf(imageszstr, "%d %d", &width, &height) != 2) {
      width=wsx;
      height=wsy;
    } 
  } 

  // config/allocate framebuffer and accumulation buffer
  config_framebuffer(width, height);

  update_rendering_state(0);
  render_compile_and_validate();
  double starttime = wkf_timer_timenow(ort_timer);

  //
  // run the renderer 
  //
  if (lasterror == 0 /* XXX SUCCESS */) {
    // clear the accumulation buffer
    ospFrameBufferClear(ospFrameBuffer, OSP_FB_COLOR | /* OSP_FB_DEPTH | */ OSP_FB_ACCUM);

    // Render to the accumulation buffer for the required number of passes
    if (getenv("VMDOSPRAYNORENDER") == NULL) {
      int accum_sample;
      for (accum_sample=0; accum_sample<ext_aa_loops; accum_sample++) {
        // The accumulation subframe count must be updated to ensure that
        // any custom RNGs for AA and AO get correctly re-seeded
        ospRenderFrame(ospFrameBuffer, ospRenderer, OSP_FB_COLOR | OSP_FB_ACCUM);
      }
    }

    // copy the accumulation buffer image data to the framebuffer and perform
    // type conversion and normaliztion on the image data...
    double rtendtime = wkf_timer_timenow(ort_timer);
    time_ray_tracing = rtendtime - starttime;

    if (lasterror == 0 /* XXX SUCCESS */) {
      // write output image to a file unless we are benchmarking
      if (getenv("VMDOSPRAYNOSAVE") == NULL) {
        const unsigned char *FB = (const unsigned char*)ospMapFrameBuffer(ospFrameBuffer, OSP_FB_COLOR);
        if (write_image_file_rgb4u(filename, FB, width, height)) {
          printf("OSPRayRenderer) Failed to write output image!\n");
        }
        ospUnmapFrameBuffer(FB, ospFrameBuffer);
      }
      time_image_io = wkf_timer_timenow(ort_timer) - rtendtime;
    } else {
      printf("OSPRayRenderer) Error during rendering.  Rendering aborted.\n");
    }

    if (verbose == RT_VERB_TIMING || verbose == RT_VERB_DEBUG) {
      printf("OSPRayRenderer) ctx setup %.2f  valid %.2f  AS %.2f  RT %.2f io %.2f\n", time_ctx_setup, time_ctx_validate, time_ctx_AS_build, time_ray_tracing, time_image_io);
    }
  } else {
    printf("OSPRayRenderer) Error during AS generation.  Rendering aborted.\n");
  }
}


void OSPRayRenderer::destroy_context() {
  DBG();
  if (!context_created)
    return;

  destroy_framebuffer();
}


void OSPRayRenderer::add_material(int matindex,
                                 float ambient, float diffuse, float specular,
                                 float shininess, float reflectivity,
                                 float opacity, 
                                 float outline, float outlinewidth,
                                 int transmode) {
  int oldmatcount = materialcache.num();
  if (oldmatcount <= matindex) {
    ort_material m;
    memset(&m, 0, sizeof(m));

    // XXX do something noticable so we see that we got a bad entry...
    m.ambient = 0.5f;
    m.diffuse = 0.7f;
    m.specular = 0.0f;
    m.shininess = 10.0f;
    m.reflectivity = 0.0f;
    m.opacity = 1.0f;
    m.transmode = 0;

    materialcache.appendN(m, matindex - oldmatcount + 1);
  }
 
  if (materialcache[matindex].isvalid) {
    return;
  } else {
    if (verbose == RT_VERB_DEBUG) printf("OSPRayRenderer) Adding material[%d]\n", matindex);

    materialcache[matindex].ambient      = ambient;
    materialcache[matindex].diffuse      = diffuse; 
    materialcache[matindex].specular     = specular;
    materialcache[matindex].shininess    = shininess;
    materialcache[matindex].reflectivity = reflectivity;
    materialcache[matindex].opacity      = opacity;
    materialcache[matindex].outline      = outline;
    materialcache[matindex].outlinewidth = outlinewidth;
    materialcache[matindex].transmode    = transmode;

    // create an OSPRay material object too...
#if OSPRAY_VERSION_MAJOR >= 1 && OSPRAY_VERSION_MINOR >= 5
    OSPMaterial ospMat = ospNewMaterial2("scivis", "RaytraceMaterial");
#else
    OSPMaterial ospMat = ospNewMaterial(ospRenderer, "RaytraceMaterial");
#endif
    ospSet3f(ospMat, "Ka", materialcache[matindex].ambient, materialcache[matindex].ambient, materialcache[matindex].ambient);
    ospSet3f(ospMat, "Kd", materialcache[matindex].diffuse, materialcache[matindex].diffuse, materialcache[matindex].diffuse);
    ospSet3f(ospMat, "Ks", materialcache[matindex].specular, materialcache[matindex].specular, materialcache[matindex].specular);
    ospSet1f(ospMat, "d", materialcache[matindex].opacity);
    ospSet1f(ospMat, "Ns", materialcache[matindex].shininess);

    /// XXX The OSPRay path tracer supports filtered transparency 
    ///     with a "Tf" material value, but there are noteworthy
    ///     restrictions about energy conservation etc to worry about:
    /// https://www.ospray.org/documentation.html#materials

    ospCommit(ospMat);
    materialcache[matindex].mat = ospMat;

    materialcache[matindex].isvalid      = 1;
  }
}


void OSPRayRenderer::init_materials() {
  DBG();
  if (verbose == RT_VERB_DEBUG) printf("OSPRayRenderer: init_materials()\n");

}


void OSPRayRenderer::set_material(OSPGeometry &geom, int matindex, float *uniform_color) {
  if (!context_created)
    return;

  if (verbose == RT_VERB_DEBUG) printf("OSPRay: setting material\n");
  ospSetMaterial(geom, materialcache[matindex].mat);
}


void OSPRayRenderer::add_directional_light(const float *dir, const float *color) {
  DBG();
  ort_directional_light l;
  vec_copy(l.dir, dir);
  vec_copy(l.color, color);

  directional_lights.append(l);
}


void OSPRayRenderer::add_positional_light(const float *pos, const float *color) {
  DBG();
  ort_positional_light l;
  vec_copy(l.pos, pos);
  vec_copy(l.color, color);

  positional_lights.append(l);
}


void OSPRayRenderer::cylinder_array(Matrix4 *wtrans, float radius,
                                   float *uniform_color,
                                   int cylnum, float *points, int matindex) {
  DBG();
  if (!context_created) return;
  if (verbose == RT_VERB_DEBUG) printf("OSPRayRenderer) creating cylinder array: %d...\n", cylnum);
  cylinder_array_cnt += cylnum;
  
  ort_cylinder_array_color ca;
  memset(&ca, 0, sizeof(ca));
  ca.num = cylnum;

  // each cylinder contains 7 32bit values.
  // compute bytes per cylinder: sizeof(v0) + sizeof(v1) + sizeof(radius)
  int bytes_per_cylinder = 3*sizeof(float) + 3*sizeof(float) + sizeof(float);
  ca.cylinders = (float *) calloc(1, cylnum * bytes_per_cylinder);
  ca.colors = (float *) calloc(1, cylnum * 4 * sizeof(float));

  int i,ind4,ind6,ind7;
  const int rOffset = 6; // radius offset
  if (wtrans == NULL) {
    for (i=0,ind4=0,ind6=0,ind7=0; i<cylnum; i++,ind4+=4,ind6+=6,ind7+=7) {
      vec_copy(&ca.cylinders[ind7  ], &points[ind6  ]);
      vec_copy(&ca.cylinders[ind7+3], &points[ind6+3]);
      ca.cylinders[ind7+rOffset] = radius;
      vec_copy(&ca.colors[ind4], &uniform_color[0]);
      ca.colors[ind4 + 3] = 1.0f;
    }
  } else {
    for (i=0,ind4=0,ind6=0,ind7=0; i<cylnum; i++,ind4+=4,ind6+=6,ind7+=7) {
      // apply transforms on points, radii
      wtrans->multpoint3d(&points[ind6  ], &ca.cylinders[ind7  ]);
      wtrans->multpoint3d(&points[ind6+3], &ca.cylinders[ind7+3]);
      ca.cylinders[ind7+rOffset] = radius;
      vec_copy(&ca.colors[ind4], &uniform_color[0]);
      ca.colors[ind4 + 3] = 1.0f;
    }
  }

  ca.matindex = matindex;
  ca.geom  = ospNewGeometry("cylinders");
  ca.cyls = ospNewData(cylnum*bytes_per_cylinder, OSP_CHAR, ca.cylinders, 0);
  ca.cols = ospNewData(cylnum, OSP_FLOAT4, ca.colors, 0);
  ospSetData(ca.geom, "cylinders", ca.cyls);
  ospSet1i(ca.geom, "bytes_per_cylinder", bytes_per_cylinder);
  ospSet1i(ca.geom, "offset_v0", 0);
  ospSet1i(ca.geom, "offset_v1", 3*sizeof(float));
  ospSet1i(ca.geom, "offset_radius", rOffset*sizeof(float));
  ospSetData(ca.geom, "color", ca.cols);
  ospCommit(ca.geom);

  set_material(ca.geom, matindex, NULL);
  cylinders_color.append(ca);
}


void OSPRayRenderer::cylinder_array_color(Matrix4 & wtrans, float rscale,
                                         int cylnum, float *points,
                                         float *radii, float *colors,
                                         int matindex) {
  DBG();
  if (!context_created) return;
  if (verbose == RT_VERB_DEBUG) printf("OSPRayRenderer) creating color cylinder array: %d...\n", cylnum);
  cylinder_array_color_cnt += cylnum;

  ort_cylinder_array_color cac;
  memset(&cac, 0, sizeof(cac));
  cac.num = cylnum;

  // each cylinder contains 7 32bit values.
  // compute bytes per cylinder: sizeof(v0) + sizeof(v1) + sizeof(radius)
  int bytes_per_cylinder = 3*sizeof(float) + 3*sizeof(float) + sizeof(float);
  cac.cylinders = (float *) calloc(1, cylnum * bytes_per_cylinder);
  cac.colors = (float *) calloc(1, cylnum * 4 * sizeof(float));

  int i, ind3, ind4, ind6, ind7;
  const int rOffset = 6; // radius offset
  for (i=0,ind3=0,ind4=0,ind6=0,ind7=0; i<cylnum; i++,ind3+=3,ind4+=4,ind6+=6,ind7+=7) {
    // apply transforms on points, radii
    wtrans.multpoint3d(&points[ind6  ], &cac.cylinders[ind7  ]);
    wtrans.multpoint3d(&points[ind6+3], &cac.cylinders[ind7+3]);
    cac.cylinders[ind7+rOffset] = radii[i] * rscale; // radius
    vec_copy(&cac.colors[ind4], &colors[ind3]);
    cac.colors[ind4 + 3] = 1.0f;
  }

  cac.matindex = matindex;
  cac.geom  = ospNewGeometry("cylinders");
  cac.cyls = ospNewData(cylnum*bytes_per_cylinder, OSP_CHAR, cac.cylinders, 0);
  cac.cols = ospNewData(cylnum, OSP_FLOAT4, cac.colors, 0);
  ospSetData(cac.geom, "cylinders", cac.cyls);
  ospSet1i(cac.geom, "bytes_per_cylinder", bytes_per_cylinder);
  ospSet1i(cac.geom, "offset_v0", 0);
  ospSet1i(cac.geom, "offset_v1", 3*sizeof(float));
  ospSet1i(cac.geom, "offset_radius", rOffset*sizeof(float));
  ospSetData(cac.geom, "color", cac.cols);
  ospCommit(cac.geom);

  set_material(cac.geom, matindex, NULL);
  cylinders_color.append(cac);
}

#if 0
void OSPRayRenderer::ring_array_color(Matrix4 & wtrans, float rscale,
                                     int rnum, float *centers,
                                     float *norms, float *radii, 
                                     float *colors, int matindex) {
}
#endif


void OSPRayRenderer::sphere_array(Matrix4 *wtrans, float rscale,
                                 float *uniform_color,
                                 int numsp, float *centers,
                                 float *radii,
                                 int matindex) {
  DBG();
  if (!context_created) return;
  if (verbose == RT_VERB_DEBUG) printf("OSPRayRenderer) creating sphere array: %d...\n", numsp);
  sphere_array_cnt += numsp;

  const rgba c = { uniform_color[0], uniform_color[1], uniform_color[2], 1.0f};

  ort_sphere_array_color sp;
  memset(&sp, 0, sizeof(sp));
  sp.num = numsp;
  sp.xyzr = (float *) calloc(1, numsp * 4*sizeof(float));
  sp.colors = (float *) calloc(1, numsp * 4*sizeof(float));

  int i, ind3, ind4;
  if (wtrans == NULL) {
    if (radii == NULL) {
      for (i=0,ind3=0,ind4=0; i<numsp; i++,ind3+=3,ind4+=4) {
        // transform to eye coordinates
        vec_copy((float*) &sp.xyzr[ind4], &centers[ind3]);
        sp.xyzr[ind4+3] = rscale;
        memcpy((float*) &sp.colors[ind4], &c, 4*sizeof(float));
      }
    } else {
      for (i=0,ind3=0,ind4=0; i<numsp; i++,ind3+=3,ind4+=4) {
        // transform to eye coordinates
        vec_copy((float*) &sp.xyzr[ind4], &centers[ind3]);
        sp.xyzr[ind4+3] = radii[i] * rscale;
        memcpy((float*) &sp.colors[ind4], &c, 4*sizeof(float));
      }
    }
  } else {
    if (radii == NULL) {
      for (i=0,ind3=0,ind4=0; i<numsp; i++,ind3+=3,ind4+=4) {
        wtrans->multpoint3d(&centers[ind3], &sp.xyzr[ind4]);
        sp.xyzr[ind4+3] = rscale;
        memcpy((float*) &sp.colors[ind4], &c, 4*sizeof(float));
      }
    } else {
      for (i=0,ind3=0,ind4=0; i<numsp; i++,ind3+=3,ind4+=4) {
        // transform to eye coordinates
        wtrans->multpoint3d(&centers[ind3], &sp.xyzr[ind4]);
        sp.xyzr[ind4+3] = radii[i] * rscale;
        memcpy((float*) &sp.colors[ind4], &c, 4*sizeof(float));
      }
    }
  }

  sp.matindex = matindex;
  sp.geom  = ospNewGeometry("spheres");
  sp.cents = ospNewData(numsp, OSP_FLOAT4, sp.xyzr, 0);
  sp.cols  = ospNewData(numsp, OSP_FLOAT4, sp.colors, 0);
  ospSetData(sp.geom, "spheres", sp.cents);
  ospSet1i(sp.geom, "bytes_per_sphere", 4*sizeof(float));
  ospSet1i(sp.geom, "offset_center", 0);
  ospSet1i(sp.geom, "offset_radius", 3*sizeof(float));
  ospSetData(sp.geom, "color",  sp.cols);
  ospCommit(sp.geom);

  set_material(sp.geom, matindex, NULL);
  spheres_color.append(sp);
}


void OSPRayRenderer::sphere_array_color(Matrix4 & wtrans, float rscale,
                                       int numsp, float *centers,
                                       float *radii, float *colors,
                                       int matindex) {
  DBG();
  if (!context_created) return;
  if (verbose == RT_VERB_DEBUG) printf("OSPRayRenderer) creating sphere array color: %d...\n", numsp);
  sphere_array_color_cnt += numsp;

  ort_sphere_array_color sp;
  memset(&sp, 0, sizeof(sp));
  sp.num = numsp;
  sp.xyzr = (float *) calloc(1, numsp * 4*sizeof(float));
  sp.colors = (float *) calloc(1, numsp * 4*sizeof(float));

  int i, ind3, ind4;
  for (i=0,ind3=0,ind4=0; i<numsp; i++,ind3+=3,ind4+=4) {
    wtrans.multpoint3d(&centers[ind3], &sp.xyzr[ind4]);
    sp.xyzr[ind4+3] = radii[i] * rscale;
    vec_copy((float*) &sp.colors[ind4], &colors[ind3]);
    sp.colors[ind4 + 3] = 1.0f;
  }

  sp.matindex = matindex;
  sp.geom  = ospNewGeometry("spheres");
  sp.cents = ospNewData(numsp, OSP_FLOAT4, sp.xyzr, 0);
  sp.cols  = ospNewData(numsp, OSP_FLOAT4, sp.colors, 0);
  ospSetData(sp.geom, "spheres", sp.cents);
  ospSet1i(sp.geom, "bytes_per_sphere", 4*sizeof(float));
  ospSet1i(sp.geom, "offset_center", 0);
  ospSet1i(sp.geom, "offset_radius", 3*sizeof(float));
  ospSetData(sp.geom, "color",  sp.cols);
  ospCommit(sp.geom);

  set_material(sp.geom, matindex, NULL);
  spheres_color.append(sp);
}


void OSPRayRenderer::tricolor_list(Matrix4 & wtrans, int numtris, float *vnc,
                                  int matindex) {
  if (!context_created) return;
//if (verbose == RT_VERB_DEBUG) printf("OSPRayRenderer) creating tricolor list: %d...\n", numtris);
  tricolor_cnt += numtris;

  // create and fill the OSPRay trimesh memory buffer
  ort_trimesh_v3f_n3f_c3f mesh;
  memset(&mesh, 0, sizeof(mesh));
  mesh.num = numtris;
  mesh.v = (float *) calloc(1, numtris * 9*sizeof(float));
  mesh.n = (float *) calloc(1, numtris * 9*sizeof(float));
  mesh.c = (float *) calloc(1, numtris * 12*sizeof(float));
  mesh.f = (int *) calloc(1, numtris * 3*sizeof(int));
  
  float alpha = 1.0f;
//  alpha = materialcache[matindex].opacity;

  int i, ind, ind9, ind12;
  for (i=0,ind=0,ind9=0,ind12=0; i<numtris; i++,ind+=27,ind9+=9,ind12+=12) {
    // transform to eye coordinates
    wtrans.multpoint3d(&vnc[ind    ], (float*) &mesh.v[ind9    ]);
    wtrans.multpoint3d(&vnc[ind + 3], (float*) &mesh.v[ind9 + 3]);
    wtrans.multpoint3d(&vnc[ind + 6], (float*) &mesh.v[ind9 + 6]);

    wtrans.multnorm3d(&vnc[ind +  9], (float*) &mesh.n[ind9    ]);
    wtrans.multnorm3d(&vnc[ind + 12], (float*) &mesh.n[ind9 + 3]);
    wtrans.multnorm3d(&vnc[ind + 15], (float*) &mesh.n[ind9 + 6]);

    vec_copy(&mesh.c[ind12    ], &vnc[ind + 18]);
    mesh.c[ind12 +  3] = alpha;
    vec_copy(&mesh.c[ind12 + 4], &vnc[ind + 21]);
    mesh.c[ind12 +  7] = alpha;
    vec_copy(&mesh.c[ind12 + 8], &vnc[ind + 24]);
    mesh.c[ind12 + 11] = alpha;

    mesh.f[i*3  ] = i*3;
    mesh.f[i*3+1] = i*3 + 1;
    mesh.f[i*3+2] = i*3 + 2;
  }

  int numverts = numtris * 3;
  mesh.matindex = matindex;
  mesh.geom  = ospNewGeometry("triangles");
  mesh.verts = ospNewData(numverts, OSP_FLOAT3, mesh.v, 0);
  mesh.norms = ospNewData(numverts, OSP_FLOAT3, mesh.n, 0);
  mesh.cols  = ospNewData(numverts, OSP_FLOAT4, mesh.c, 0);
  mesh.ind   = ospNewData(numtris, OSP_INT3,   mesh.f, 0);
  ospSetData(mesh.geom, "vertex", mesh.verts);
  ospSetData(mesh.geom, "vertex.normal", mesh.norms);
  ospSetData(mesh.geom, "index",  mesh.ind);
  ospSetData(mesh.geom, "vertex.color",  mesh.cols);
  ospCommit(mesh.geom);

  set_material(mesh.geom, matindex, NULL);
  trimesh_v3f_n3f_c3f.append(mesh); 
}


void OSPRayRenderer::trimesh_c4n3v3(Matrix4 & wtrans, int numverts,
                                   float *cnv, int numfacets, int * facets,
                                   int matindex) {
  if (!context_created) return;
  if (verbose == RT_VERB_DEBUG) printf("OSPRayRenderer) creating trimesh_c4n3v3: %d...\n", numfacets);
  trimesh_c4u_n3b_v3f_cnt += numfacets;

  // create and fill the OSPRay trimesh memory buffer
  ort_trimesh_v3f_n3f_c3f mesh;
  memset(&mesh, 0, sizeof(mesh));
  mesh.num = numfacets;
  mesh.v = (float *) calloc(1, numfacets * 9*sizeof(float));
  mesh.n = (float *) calloc(1, numfacets * 9*sizeof(float));
  mesh.c = (float *) calloc(1, numfacets * 12*sizeof(float));
  mesh.f = (int *) calloc(1, numfacets * 3*sizeof(int));
 
  float alpha = 1.0f;
//  alpha = materialcache[matindex].opacity;

  // XXX we are currently converting to triangle soup for ease of
  // initial implementation, but this is clearly undesirable long-term
  int i, ind, ind9, ind12;
  for (i=0,ind=0,ind9=0,ind12=0; i<numfacets; i++,ind+=3,ind9+=9,ind12+=12) {
    int v0 = facets[ind    ] * 10;
    int v1 = facets[ind + 1] * 10;
    int v2 = facets[ind + 2] * 10;

    // transform to eye coordinates
    wtrans.multpoint3d(cnv + v0 + 7, (float*) &mesh.v[ind9    ]);
    wtrans.multpoint3d(cnv + v1 + 7, (float*) &mesh.v[ind9 + 3]);
    wtrans.multpoint3d(cnv + v2 + 7, (float*) &mesh.v[ind9 + 6]);

    wtrans.multnorm3d(cnv + v0 + 4, (float*) &mesh.n[ind9    ]);
    wtrans.multnorm3d(cnv + v1 + 4, (float*) &mesh.n[ind9 + 3]);
    wtrans.multnorm3d(cnv + v2 + 4, (float*) &mesh.n[ind9 + 6]);

    vec_copy(&mesh.c[ind12    ], cnv + v0);
    mesh.c[ind12 +  3] = alpha;
    vec_copy(&mesh.c[ind12 + 4], cnv + v1);
    mesh.c[ind12 +  7] = alpha;
    vec_copy(&mesh.c[ind12 + 8], cnv + v2);
    mesh.c[ind12 + 11] = alpha;

    mesh.f[i*3  ] = i*3;
    mesh.f[i*3+1] = i*3 + 1;
    mesh.f[i*3+2] = i*3 + 2;
  }

  int Nnumverts = numfacets * 3;
  mesh.matindex = matindex;
  mesh.geom  = ospNewGeometry("triangles");
  mesh.verts = ospNewData(Nnumverts, OSP_FLOAT3, mesh.v, 0);
  mesh.norms = ospNewData(Nnumverts, OSP_FLOAT3, mesh.n, 0);
  mesh.cols  = ospNewData(Nnumverts, OSP_FLOAT4, mesh.c, 0);
  mesh.ind   = ospNewData(numfacets, OSP_INT3,   mesh.f, 0);
  ospSetData(mesh.geom, "vertex", mesh.verts);
  ospSetData(mesh.geom, "vertex.normal", mesh.norms);
  ospSetData(mesh.geom, "index",  mesh.ind);
  ospSetData(mesh.geom, "vertex.color",  mesh.cols);
  ospCommit(mesh.geom);

  set_material(mesh.geom, matindex, NULL);
  trimesh_v3f_n3f_c3f.append(mesh);
}



// 
// This implementation translates from the most-compact host representation
// to the best that OSPRay allows
//
void OSPRayRenderer::trimesh_c4u_n3b_v3f(Matrix4 & wtrans, unsigned char *c, 
                                        char *n, float *v, int numfacets, 
                                        int matindex) {
  if (!context_created) return;
  if (verbose == RT_VERB_DEBUG) printf("OSPRayRenderer) creating trimesh_c4u_n3b_v3f: %d...\n", numfacets);
  trimesh_n3b_v3f_cnt += numfacets;

  // create and fill the OSPRay trimesh memory buffer
  ort_trimesh_v3f_n3f_c3f mesh;
  memset(&mesh, 0, sizeof(mesh));
  mesh.num = numfacets;
  mesh.v = (float *) calloc(1, numfacets * 9*sizeof(float));
  mesh.n = (float *) calloc(1, numfacets * 9*sizeof(float));
  mesh.c = (float *) calloc(1, numfacets * 12*sizeof(float));
  mesh.f = (int *) calloc(1, numfacets * 3*sizeof(int));
 
  float alpha = 1.0f;
//  alpha = materialcache[matindex].opacity;

  // XXX we are currently converting to triangle soup for ease of
  // initial implementation, but this is clearly undesirable long-term
  int i, ind, ind9, ind12;

  const float ci2f = 1.0f / 255.0f;
  const float cn2f = 1.0f / 127.5f;
  for (i=0,ind=0,ind9=0,ind12=0; i<numfacets; i++,ind+=3,ind9+=9,ind12+=12) {
    float norm[9];

    // conversion from GLbyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
    // float = (2c+1)/(2^8-1)
    norm[0] = n[ind9    ] * cn2f + ci2f;
    norm[1] = n[ind9 + 1] * cn2f + ci2f;
    norm[2] = n[ind9 + 2] * cn2f + ci2f;
    norm[3] = n[ind9 + 3] * cn2f + ci2f;
    norm[4] = n[ind9 + 4] * cn2f + ci2f;
    norm[5] = n[ind9 + 5] * cn2f + ci2f;
    norm[6] = n[ind9 + 6] * cn2f + ci2f;
    norm[7] = n[ind9 + 7] * cn2f + ci2f;
    norm[8] = n[ind9 + 8] * cn2f + ci2f;

    // transform to eye coordinates
    wtrans.multpoint3d(v + ind9    , (float*) &mesh.v[ind9    ]);
    wtrans.multpoint3d(v + ind9 + 3, (float*) &mesh.v[ind9 + 3]);
    wtrans.multpoint3d(v + ind9 + 6, (float*) &mesh.v[ind9 + 6]);

    wtrans.multnorm3d(norm    , (float*) &mesh.n[ind9    ]);
    wtrans.multnorm3d(norm + 3, (float*) &mesh.n[ind9 + 3]);
    wtrans.multnorm3d(norm + 6, (float*) &mesh.n[ind9 + 6]);

    float col[9];

    // conversion from GLubyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
    // float = c/(2^8-1)
    col[0] = c[ind12     ] * ci2f;
    col[1] = c[ind12 +  1] * ci2f;
    col[2] = c[ind12 +  2] * ci2f;
    col[3] = c[ind12 +  4] * ci2f;
    col[4] = c[ind12 +  5] * ci2f;
    col[5] = c[ind12 +  6] * ci2f;
    col[6] = c[ind12 +  8] * ci2f;
    col[7] = c[ind12 +  9] * ci2f;
    col[8] = c[ind12 + 10] * ci2f;

    vec_copy(&mesh.c[ind12    ], col    );
    mesh.c[ind12 +  3] = alpha;
    vec_copy(&mesh.c[ind12 + 4], col + 3);
    mesh.c[ind12 +  7] = alpha;
    vec_copy(&mesh.c[ind12 + 8], col + 6);
    mesh.c[ind12 + 11] = alpha;

    mesh.f[i*3  ] = i*3;
    mesh.f[i*3+1] = i*3 + 1;
    mesh.f[i*3+2] = i*3 + 2;
  }

  int Nnumverts = numfacets * 3;
  mesh.matindex = matindex;
  mesh.geom  = ospNewGeometry("triangles");
  mesh.verts = ospNewData(Nnumverts, OSP_FLOAT3, mesh.v, 0);
  mesh.norms = ospNewData(Nnumverts, OSP_FLOAT3, mesh.n, 0);
  mesh.cols  = ospNewData(Nnumverts, OSP_FLOAT4, mesh.c, 0);
  mesh.ind   = ospNewData(numfacets, OSP_INT3,   mesh.f, 0);
  ospSetData(mesh.geom, "vertex", mesh.verts);
  ospSetData(mesh.geom, "vertex.normal", mesh.norms);
  ospSetData(mesh.geom, "index",  mesh.ind);
  ospSetData(mesh.geom, "vertex.color",  mesh.cols);
  ospCommit(mesh.geom);

  set_material(mesh.geom, matindex, NULL);
  trimesh_v3f_n3f_c3f.append(mesh);
}



void OSPRayRenderer::trimesh_c4u_n3f_v3f(Matrix4 & wtrans, unsigned char *c, 
                                        float *n, float *v, int numfacets, 
                                        int matindex) {
  if (!context_created) return;
  if (verbose == RT_VERB_DEBUG) printf("OSPRayRenderer) creating trimesh_c4u_n3f_v3f: %d...\n", numfacets);
  tricolor_cnt += numfacets;

  // create and fill the OSPRay trimesh memory buffer
  ort_trimesh_v3f_n3f_c3f mesh;
  memset(&mesh, 0, sizeof(mesh));
  mesh.num = numfacets;
  mesh.v = (float *) calloc(1, numfacets * 9*sizeof(float));
  mesh.n = (float *) calloc(1, numfacets * 9*sizeof(float));
  mesh.c = (float *) calloc(1, numfacets * 12*sizeof(float));
  mesh.f = (int *) calloc(1, numfacets * 3*sizeof(int));
 
  float alpha = 1.0f;
//  alpha = materialcache[matindex].opacity;

  // XXX we are currently converting to triangle soup for ease of
  // initial implementation, but this is clearly undesirable long-term
  int i, ind, ind9, ind12;

  const float ci2f = 1.0f / 255.0f;
  for (i=0,ind=0,ind9=0,ind12=0; i<numfacets; i++,ind+=3,ind9+=9,ind12+=12) {
    // transform to eye coordinates
    wtrans.multpoint3d(v + ind9    , (float*) &mesh.v[ind9    ]);
    wtrans.multpoint3d(v + ind9 + 3, (float*) &mesh.v[ind9 + 3]);
    wtrans.multpoint3d(v + ind9 + 6, (float*) &mesh.v[ind9 + 6]);

    wtrans.multnorm3d(n + ind9    , &mesh.n[ind9    ]);
    wtrans.multnorm3d(n + ind9 + 3, &mesh.n[ind9 + 3]);
    wtrans.multnorm3d(n + ind9 + 6, &mesh.n[ind9 + 3]);

    // conversion from GLubyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
    // float = c/(2^8-1)
    float col[9];
    col[0] = c[ind12     ] * ci2f;
    col[1] = c[ind12 +  1] * ci2f;
    col[2] = c[ind12 +  2] * ci2f;
    col[3] = c[ind12 +  4] * ci2f;
    col[4] = c[ind12 +  5] * ci2f;
    col[5] = c[ind12 +  6] * ci2f;
    col[6] = c[ind12 +  8] * ci2f;
    col[7] = c[ind12 +  9] * ci2f;
    col[8] = c[ind12 + 10] * ci2f;

    vec_copy(&mesh.c[ind12    ], col    );
    mesh.c[ind12 +  3] = alpha;
    vec_copy(&mesh.c[ind12 + 4], col + 3);
    mesh.c[ind12 +  7] = alpha;
    vec_copy(&mesh.c[ind12 + 8], col + 6);
    mesh.c[ind12 + 11] = alpha;

    mesh.f[i*3  ] = i*3;
    mesh.f[i*3+1] = i*3 + 1;
    mesh.f[i*3+2] = i*3 + 2;
  }

  int Nnumverts = numfacets * 3;
  mesh.matindex = matindex;
  mesh.geom  = ospNewGeometry("triangles");
  mesh.verts = ospNewData(Nnumverts, OSP_FLOAT3, mesh.v, 0);
  mesh.norms = ospNewData(Nnumverts, OSP_FLOAT3, mesh.n, 0);
  mesh.cols  = ospNewData(Nnumverts, OSP_FLOAT4, mesh.c, 0);
  mesh.ind   = ospNewData(numfacets, OSP_INT3,   mesh.f, 0);
  ospSetData(mesh.geom, "vertex", mesh.verts);
  ospSetData(mesh.geom, "vertex.normal", mesh.norms);
  ospSetData(mesh.geom, "index",  mesh.ind);
  ospSetData(mesh.geom, "vertex.color",  mesh.cols);
  ospCommit(mesh.geom);

  set_material(mesh.geom, matindex, NULL);
  trimesh_v3f_n3f_c3f.append(mesh);
}


void OSPRayRenderer::trimesh_n3b_v3f(Matrix4 & wtrans, float *uniform_color, 
                                    char *n, float *v, int numfacets, 
                                    int matindex) {
  if (!context_created) return;
  if (verbose == RT_VERB_DEBUG) printf("OSPRayRenderer) creating trimesh_n3b_v3f: %d...\n", numfacets);
  trimesh_n3b_v3f_cnt += numfacets;

  // create and fill the OSPRay trimesh memory buffer
  ort_trimesh_v3f_n3f_c3f mesh;
  memset(&mesh, 0, sizeof(mesh));
  mesh.num = numfacets;
  mesh.v = (float *) calloc(1, numfacets * 9*sizeof(float));
  mesh.n = (float *) calloc(1, numfacets * 9*sizeof(float));
  mesh.c = (float *) calloc(1, numfacets * 12*sizeof(float));
  mesh.f = (int *) calloc(1, numfacets * 3*sizeof(int));
 
  float alpha = 1.0f;

  // XXX we are currently converting to triangle soup for ease of
  // initial implementation, but this is clearly undesirable long-term
  int i, ind, ind9, ind12;

  const float ci2f = 1.0f / 255.0f;
  const float cn2f = 1.0f / 127.5f;
  for (i=0,ind=0,ind9=0,ind12=0; i<numfacets; i++,ind+=3,ind9+=9,ind12+=12) {
    float norm[9];

    // conversion from GLbyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
    // float = (2c+1)/(2^8-1)
    norm[0] = n[ind9    ] * cn2f + ci2f;
    norm[1] = n[ind9 + 1] * cn2f + ci2f;
    norm[2] = n[ind9 + 2] * cn2f + ci2f;
    norm[3] = n[ind9 + 3] * cn2f + ci2f;
    norm[4] = n[ind9 + 4] * cn2f + ci2f;
    norm[5] = n[ind9 + 5] * cn2f + ci2f;
    norm[6] = n[ind9 + 6] * cn2f + ci2f;
    norm[7] = n[ind9 + 7] * cn2f + ci2f;
    norm[8] = n[ind9 + 8] * cn2f + ci2f;

    // transform to eye coordinates
    wtrans.multpoint3d(v + ind9    , (float*) &mesh.v[ind9    ]);
    wtrans.multpoint3d(v + ind9 + 3, (float*) &mesh.v[ind9 + 3]);
    wtrans.multpoint3d(v + ind9 + 6, (float*) &mesh.v[ind9 + 6]);

    wtrans.multnorm3d(norm    , (float*) &mesh.n[ind9    ]);
    wtrans.multnorm3d(norm + 3, (float*) &mesh.n[ind9 + 3]);
    wtrans.multnorm3d(norm + 6, (float*) &mesh.n[ind9 + 6]);

    vec_copy(&mesh.c[ind12    ], uniform_color);
    mesh.c[ind12 +  3] = alpha;
    vec_copy(&mesh.c[ind12 + 4], uniform_color);
    mesh.c[ind12 +  7] = alpha;
    vec_copy(&mesh.c[ind12 + 8], uniform_color);
    mesh.c[ind12 + 11] = alpha;

    mesh.f[i*3  ] = i*3;
    mesh.f[i*3+1] = i*3 + 1;
    mesh.f[i*3+2] = i*3 + 2;
  }

  int Nnumverts = numfacets * 3;
  mesh.matindex = matindex;
  mesh.geom  = ospNewGeometry("triangles");
  mesh.verts = ospNewData(Nnumverts, OSP_FLOAT3, mesh.v, 0);
  mesh.norms = ospNewData(Nnumverts, OSP_FLOAT3, mesh.n, 0);
  mesh.cols  = ospNewData(Nnumverts, OSP_FLOAT4, mesh.c, 0);
  mesh.ind   = ospNewData(numfacets, OSP_INT3,   mesh.f, 0);
  ospSetData(mesh.geom, "vertex", mesh.verts);
  ospSetData(mesh.geom, "vertex.normal", mesh.norms);
  ospSetData(mesh.geom, "index",  mesh.ind);
  ospSetData(mesh.geom, "vertex.color",  mesh.cols);
  ospCommit(mesh.geom);

  set_material(mesh.geom, matindex, NULL);
  trimesh_v3f_n3f_c3f.append(mesh);
}


// XXX At present we have to build/populate a per-vertex color arrays,
//     but that should go away as soon as OSPRay allows it.
void OSPRayRenderer::trimesh_n3f_v3f(Matrix4 & wtrans, float *uniform_color, 
                                    float *n, float *v, int numfacets, 
                                    int matindex) {
  DBG();
  if (!context_created) return;
  if (verbose == RT_VERB_DEBUG) printf("OSPRayRenderer) creating trimesh_n3f_v3f: %d...\n", numfacets);
  trimesh_n3f_v3f_cnt += numfacets;

  // create and fill the OSPRay trimesh memory buffer
  ort_trimesh_v3f_n3f_c3f mesh;
  memset(&mesh, 0, sizeof(mesh));
  mesh.num = numfacets;
  mesh.v = (float *) calloc(1, numfacets * 9*sizeof(float));
  mesh.n = (float *) calloc(1, numfacets * 9*sizeof(float));
  mesh.c = (float *) calloc(1, numfacets * 12*sizeof(float));
  mesh.f = (int *) calloc(1, numfacets * 3*sizeof(int));

  float alpha = 1.0f;

  // create and fill the OSPRay trimesh memory buffer
  int i, ind, ind9, ind12;

  for (i=0,ind=0,ind9=0,ind12=0; i<numfacets; i++,ind+=3,ind9+=9,ind12+=12) {
    // transform to eye coordinates
    wtrans.multpoint3d(v + ind9    , (float*) &mesh.v[ind9    ]);
    wtrans.multpoint3d(v + ind9 + 3, (float*) &mesh.v[ind9 + 3]);
    wtrans.multpoint3d(v + ind9 + 6, (float*) &mesh.v[ind9 + 6]);

    wtrans.multnorm3d(n + ind9    , (float*) &mesh.n[ind9    ]);
    wtrans.multnorm3d(n + ind9 + 3, (float*) &mesh.n[ind9 + 3]);
    wtrans.multnorm3d(n + ind9 + 6, (float*) &mesh.n[ind9 + 6]);

    vec_copy(&mesh.c[ind12    ], uniform_color);
    mesh.c[ind12 +  3] = alpha;
    vec_copy(&mesh.c[ind12 + 4], uniform_color);
    mesh.c[ind12 +  7] = alpha;
    vec_copy(&mesh.c[ind12 + 8], uniform_color);
    mesh.c[ind12 + 11] = alpha;

    mesh.f[i*3  ] = i*3;
    mesh.f[i*3+1] = i*3 + 1;
    mesh.f[i*3+2] = i*3 + 2;
  }

  int Nnumverts = numfacets * 3;
  mesh.matindex = matindex;
  mesh.geom  = ospNewGeometry("triangles");
  mesh.verts = ospNewData(Nnumverts, OSP_FLOAT3, mesh.v, 0);
  mesh.norms = ospNewData(Nnumverts, OSP_FLOAT3, mesh.n, 0);
  mesh.cols  = ospNewData(Nnumverts, OSP_FLOAT4, mesh.c, 0);
  mesh.ind   = ospNewData(numfacets, OSP_INT3,   mesh.f, 0);
  ospSetData(mesh.geom, "vertex", mesh.verts);
  ospSetData(mesh.geom, "vertex.normal", mesh.norms);
  ospSetData(mesh.geom, "index",  mesh.ind);
  ospSetData(mesh.geom, "vertex.color",  mesh.cols);
  ospCommit(mesh.geom);

  set_material(mesh.geom, matindex, NULL);
  trimesh_v3f_n3f_c3f.append(mesh);
}


#if 0
void OSPRayRenderer::trimesh_v3f(Matrix4 & wtrans, float *uniform_color, 
                                float *v, int numfacets, int matindex) {
  if (!context_created) return;
  if (verbose == RT_VERB_DEBUG) printf("OSPRayRenderer) creating trimesh_v3f: %d...\n", numfacets);
  trimesh_v3f_cnt += numfacets;

  set_material(geom, matindex, NULL);
  append_objects(buf, geom, instance);
}

#endif

void OSPRayRenderer::tristrip(Matrix4 & wtrans, int numverts, const float * cnv,
                             int numstrips, const int *vertsperstrip,
                             const int *facets, int matindex) {
  if (!context_created) return;
  int i;
  int numfacets = 0;
  for (i=0; i<numstrips; i++) 
    numfacets += (vertsperstrip[i] - 2);  

  if (verbose == RT_VERB_DEBUG) printf("OSPRayRenderer) creating tristrip: %d...\n", numfacets);
  tricolor_cnt += numfacets;

  // create and fill the OSPRay trimesh memory buffer
  ort_trimesh_v3f_n3f_c3f mesh;
  memset(&mesh, 0, sizeof(mesh));
  mesh.num = numfacets;
  mesh.v = (float *) calloc(1, numfacets * 9*sizeof(float));
  mesh.n = (float *) calloc(1, numfacets * 9*sizeof(float));
  mesh.c = (float *) calloc(1, numfacets * 12*sizeof(float));
  mesh.f = (int *) calloc(1, numfacets * 3*sizeof(int));

  float alpha = 1.0f;
//  alpha = materialcache[matindex].opacity;

  // XXX we are currently converting to triangle soup for ease of
  // initial implementation, but this is clearly undesirable long-term

  // render triangle strips one triangle at a time
  // triangle winding order is:
  //   v0, v1, v2, then v2, v1, v3, then v2, v3, v4, etc.
  int strip, t, v = 0;
  int stripaddr[2][3] = { {0, 1, 2}, {1, 0, 2} };

  // loop over all of the triangle strips
  i=0; // set triangle index to 0
  int ind9, ind12;
  for (strip=0,ind9=0,ind12=0; strip < numstrips; strip++) {
    // loop over all triangles in this triangle strip
    for (t = 0; t < (vertsperstrip[strip] - 2); t++) {
      // render one triangle, using lookup table to fix winding order
      int v0 = facets[v + (stripaddr[t & 0x01][0])] * 10;
      int v1 = facets[v + (stripaddr[t & 0x01][1])] * 10;
      int v2 = facets[v + (stripaddr[t & 0x01][2])] * 10;

      // transform to eye coordinates
      wtrans.multpoint3d(cnv + v0 + 7, (float*) &mesh.v[ind9    ]);
      wtrans.multpoint3d(cnv + v1 + 7, (float*) &mesh.v[ind9 + 3]);
      wtrans.multpoint3d(cnv + v2 + 7, (float*) &mesh.v[ind9 + 6]);

      wtrans.multnorm3d(cnv + v0 + 4, (float*) &mesh.n[ind9    ]);
      wtrans.multnorm3d(cnv + v1 + 4, (float*) &mesh.n[ind9 + 3]);
      wtrans.multnorm3d(cnv + v2 + 4, (float*) &mesh.n[ind9 + 6]);

      vec_copy(&mesh.c[ind12    ], cnv + v0);
      mesh.c[ind12 +  3] = alpha;
      vec_copy(&mesh.c[ind12 + 4], cnv + v1);
      mesh.c[ind12 +  7] = alpha;
      vec_copy(&mesh.c[ind12 + 8], cnv + v2);
      mesh.c[ind12 + 11] = alpha;

      mesh.f[i*3  ] = i*3;
      mesh.f[i*3+1] = i*3 + 1;
      mesh.f[i*3+2] = i*3 + 2;

      v++; // move on to next vertex
      i++; // next triangle
      ind9+=9;
      ind12+=12;
    }
    v+=2; // last two vertices are already used by last triangle
  }

  int Nnumverts = numfacets * 3;
  mesh.matindex = matindex;
  mesh.geom  = ospNewGeometry("triangles");
  mesh.verts = ospNewData(Nnumverts, OSP_FLOAT3, mesh.v, 0);
  mesh.norms = ospNewData(Nnumverts, OSP_FLOAT3, mesh.n, 0);
  mesh.cols  = ospNewData(Nnumverts, OSP_FLOAT4, mesh.c, 0);
  mesh.ind   = ospNewData(numfacets, OSP_INT3,   mesh.f, 0);
  ospSetData(mesh.geom, "vertex", mesh.verts);
  ospSetData(mesh.geom, "vertex.normal", mesh.norms);
  ospSetData(mesh.geom, "index",  mesh.ind);
  ospSetData(mesh.geom, "vertex.color",  mesh.cols);
  ospCommit(mesh.geom);

  set_material(mesh.geom, matindex, NULL);
  trimesh_v3f_n3f_c3f.append(mesh);
}



