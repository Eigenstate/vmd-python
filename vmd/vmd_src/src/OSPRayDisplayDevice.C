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
*      $RCSfile: OSPRayDisplayDevice.C
*      $Author: johns $      $Locker:  $               $State: Exp $
*      $Revision: 1.12 $         $Date: 2016/11/06 05:53:43 $
*
***************************************************************************
* DESCRIPTION:
*   FileRenderer type for the OSPRay interface.
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

#include "VMDApp.h"    // needed for accelerator memory management
#include "QuickSurf.h" // needed for accelerator memory management

#include "DispCmds.h"  // CYLINDER_TRAILINGCAP, etc..
#include "OSPRayDisplayDevice.h"
#include "OSPRayRenderer.h"
#include "config.h"    // needed for default image viewer
#include "Hershey.h"   // needed for Hershey font rendering fctns


// The default radius for points and lines (which are displayed
// as small spheres or cylinders, respectively)
#define DEFAULT_RADIUS  0.0025f


// Global OSPRay initialization routine -- call it only ONCE...
void OSPRayDisplayDevice::OSPRay_Global_Init(void) {
  OSPRayRenderer::OSPRay_Global_Init(); // call only ONCE
}


/// constructor ... initialize some variables
OSPRayDisplayDevice::OSPRayDisplayDevice(VMDApp *app, int interactive) : FileRenderer((interactive) ? 
               "TachyonLOSPRayInteractive" : "TachyonLOSPRayInternal", 
               (interactive) ? 
               "TachyonL-OSPRay (interactive, SSE+AVX-accelerated)" : "TachyonL-OSPRay (internal, in-memory, SSE+AVX-accelerated)",
               "vmdscene.ppm", DEF_VMDIMAGEVIEWER) {
  vmdapp = app; // save VMDApp handle for init or memory management routines

  reset_vars(); // initialize material cache

  // flag interactive or not
  isinteractive = interactive;

  // Add supported file formats
  formats.add_name("PPM", 0);

  // Default image format depends on platform
  curformat = 0;

  // Set default aa level
  has_aa = TRUE;
  aasamples = 12;
  aosamples = 12;

  ort = new OSPRayRenderer();
  ort_timer = wkf_timer_create();
}
        
/// destructor
OSPRayDisplayDevice::~OSPRayDisplayDevice(void) {
  delete ort;
  wkf_timer_destroy(ort_timer);
}

void OSPRayDisplayDevice::add_material(void) {
  ort->add_material(materialIndex,
                    mat_ambient, mat_diffuse, mat_specular, mat_shininess,
                    mat_mirror, mat_opacity, mat_outline, mat_outlinewidth, 
                    mat_transmode > 0.5f);
}


/// (re)initialize cached state variables used to track material changes 
void OSPRayDisplayDevice::reset_vars(void) {
  reset_cylinder_buffer();
  reset_triangle_buffer();
}


void OSPRayDisplayDevice::send_cylinder_buffer() {
  if (cylinder_vert_buffer.num() > 0) {
    // send the cylinders...
    ort->cylinder_array_color(*cylinder_xform, cylinder_radius_scalefactor,
                              cylinder_vert_buffer.num()/6,
                              &cylinder_vert_buffer[0],
                              &cylinder_radii_buffer[0],
                              &cylinder_color_buffer[0],
                              cylinder_matindex);

#if 0
    // send the cylinder end caps, if any
    if (cylcap_vert_buffer.num() > 0) {
      ort->ring_array_color(*cylinder_xform, cylinder_radius_scalefactor,
                              cylcap_vert_buffer.num()/3,
                              &cylcap_vert_buffer[0],
                              &cylcap_norm_buffer[0],
                              &cylcap_radii_buffer[0],
                              &cylcap_color_buffer[0],
                              cylinder_matindex);
    }
#endif

    delete cylinder_xform;
    cylinder_xform=NULL;
  }
  reset_cylinder_buffer();
}


#if defined(VMDOSPRAY_ENABLE_CYLINDERS)

// draw a cylinder
void OSPRayDisplayDevice::cylinder(float *a, float *b, float r, int filled) {
  // if we have a change in transformation matrix, color, or material state,
  // we have to emit all accumulated cylinders to OSPRay and begin a new batch
  if (cylinder_xform != NULL && ((cylinder_matindex != materialIndex) || (memcmp(cylinder_xform->mat, transMat.top().mat, sizeof(cylinder_xform->mat))))) {
    send_cylinder_buffer(); // render the accumulated cylinder buffer...
  }

  // record all transformation/material/color state on first cylinder call
  if (cylinder_xform == NULL) {
    // record material, color, and transformation state
    cylinder_matindex = materialIndex;
    cylinder_xform = new Matrix4(transMat.top());
    cylinder_radius_scalefactor = scale_factor();
    add_material(); // cause OSPRay to cache the current material
  }

  // record vertex data
  cylinder_vert_buffer.append2x3(&a[0], &b[0]);
  cylinder_radii_buffer.append(r);
  cylinder_color_buffer.append3(&matData[colorIndex][0]);

#if 0
  // XXX Cylinder caps not yet supported in OSPRay
  // Cylinder caps?
  if (filled) {
    float norm[3];
    norm[0] = b[0] - a[0];
    norm[1] = b[1] - a[1];
    norm[2] = b[2] - a[2];

    float div = 1.0f / sqrtf(norm[0]*norm[0] + norm[1]*norm[1] + norm[2]*norm[2]);
    norm[0] *= div;
    norm[1] *= div;
    norm[2] *= div;

    if (filled & CYLINDER_TRAILINGCAP) {
      cylcap_vert_buffer.append3(&a[0]);
      cylcap_norm_buffer.append3(&norm[0]);
      cylcap_radii_buffer.append2(0.0f, r);
      cylcap_color_buffer.append3(&matData[colorIndex][0]);
    }

    if (filled & CYLINDER_LEADINGCAP) {
      cylcap_vert_buffer.append3(&b[0]);
      norm[0] *= -1;
      norm[1] *= -1;
      norm[2] *= -1;
      cylcap_norm_buffer.append3(&norm[0]);
      cylcap_radii_buffer.append2(0.0f, r);
      cylcap_color_buffer.append3(&matData[colorIndex][0]);
    }
  }
#endif
}
#endif


// draw a sphere array
void OSPRayDisplayDevice::sphere_array(int spnum, int spres, float *centers, 
                                      float *radii, float *colors) {
  add_material();
  ort->sphere_array_color(transMat.top(), scale_factor(), spnum, 
                          centers, radii, colors, materialIndex);
  
  // set final color state after array has been drawn
  int ind=(spnum-1)*3;
  super_set_color(nearest_index(colors[ind], colors[ind+1], colors[ind+2]));
}


#if defined(VMDOSPRAY_ENABLE_CYLINDERS)

//
// XXX text needs cylinders to be performant...
//
void OSPRayDisplayDevice::text(float *pos, float size, float thickness,
                              const char *str) {
  float textpos[3];
  float textsize, textthickness;
  hersheyhandle hh;

  // transform the world coordinates
  (transMat.top()).multpoint3d(pos, textpos);
  textsize = size * 1.5f;
  textthickness = thickness*DEFAULT_RADIUS;

  ResizeArray<float> text_spheres;
  ResizeArray<float> text_cylinders;

  while (*str != '\0') {
    float lm, rm, x, y, ox, oy;
    int draw, odraw;
    ox=oy=x=y=0.0f;
    draw=odraw=0;

    hersheyDrawInitLetter(&hh, *str, &lm, &rm);
    textpos[0] -= lm * textsize;

    while (!hersheyDrawNextLine(&hh, &draw, &x, &y)) {
      float oldpt[3], newpt[3];
      if (draw) {
        newpt[0] = textpos[0] + textsize * x;
        newpt[1] = textpos[1] + textsize * y;
        newpt[2] = textpos[2];

        if (odraw) {
          // if we have both previous and next points, connect them...
          oldpt[0] = textpos[0] + textsize * ox;
          oldpt[1] = textpos[1] + textsize * oy;
          oldpt[2] = textpos[2];

          text_cylinders.append2x3(&oldpt[0], &newpt[0]);
          text_spheres.append3(&newpt[0]);
        } else {
          // ...otherwise, just draw the next point
          text_spheres.append3(&newpt[0]);
        }
      }

      ox=x;
      oy=y;
      odraw=draw;
    }
    textpos[0] += rm * textsize;

    str++;
  }

  add_material();
  // add spheres, which are already in world coordinates
  if (text_cylinders.num() > 0) {
    ort->cylinder_array(NULL, textthickness, matData[colorIndex],
                        text_cylinders.num() / 6, &text_cylinders[0],
                        materialIndex);
  }
  if (text_spheres.num() > 0) {
    ort->sphere_array(NULL, textthickness, matData[colorIndex],
                      text_spheres.num() / 3, &text_spheres[0], NULL, 
                      materialIndex);
  }
}

#endif


void OSPRayDisplayDevice::send_triangle_buffer() {
  if (triangle_vert_buffer.num() > 0) {
    ort->trimesh_n3f_v3f(*triangle_xform, 
                         matData[triangle_cindex],
                         &triangle_norm_buffer[0],
                         &triangle_vert_buffer[0],
                         triangle_vert_buffer.num()/9,
                         triangle_matindex);
    delete triangle_xform;
    triangle_xform=NULL;
  }
  reset_triangle_buffer();
}


// draw a triangle
void OSPRayDisplayDevice::triangle(const float *a, const float *b, const float *c, const float *n1, const float *n2, const float *n3) {
  // if we have a change in transformation matrix, color, or material state,
  // we have to emit all accumulated triangles to OSPRay and begin a new batch
  if (triangle_xform != NULL && ((triangle_cindex != colorIndex) || (triangle_matindex != materialIndex) || (memcmp(triangle_xform->mat, transMat.top().mat, sizeof(triangle_xform->mat))))) {
    send_triangle_buffer(); // render the accumulated triangle buffer...
  } 

  // record all transformation/material/color state on first triangle call
  if (triangle_xform == NULL) {
    // record material, color, and transformation state
    triangle_cindex = colorIndex;
    triangle_matindex = materialIndex;
    triangle_xform = new Matrix4(transMat.top());
    add_material(); // cause OSPRay to cache the current material
  }

  // record vertex data 
  triangle_vert_buffer.append3x3(&a[0], &b[0], &c[0]);

  // record normal data 
  triangle_norm_buffer.append3x3(&n1[0], &n2[0], &n3[0]);
}


// draw a tricolor
void OSPRayDisplayDevice::tricolor(const float *a, const float *b, const float *c,
                      const float *n1, const float *n2, const float *n3,
                      const float *c1, const float *c2, const float *c3) {
  add_material();

  float vnc[27];
  vec_copy(&vnc[ 0], a);
  vec_copy(&vnc[ 3], b);
  vec_copy(&vnc[ 6], c);

  vec_copy(&vnc[ 9], n1);
  vec_copy(&vnc[12], n2);
  vec_copy(&vnc[15], n3);

  vec_copy(&vnc[18], c1);
  vec_copy(&vnc[21], c2);
  vec_copy(&vnc[24], c3);

  ort->tricolor_list(transMat.top(), 1, vnc, materialIndex);
}


void OSPRayDisplayDevice::trimesh_c4u_n3b_v3f(unsigned char *c, char *n,
                                             float *v, int numfacets) {
  add_material();
  ort->trimesh_c4u_n3b_v3f(transMat.top(), c, n, v, numfacets, materialIndex);
}


void OSPRayDisplayDevice::trimesh_c4u_n3f_v3f(unsigned char *c, float *n,
                                             float *v, int numfacets) {
  add_material();
  ort->trimesh_c4u_n3f_v3f(transMat.top(), c, n, v, numfacets, materialIndex);
}

void OSPRayDisplayDevice::trimesh_c4n3v3(int numverts, float * cnv,
                                             int numfacets, int * facets) {
  add_material();
  ort->trimesh_c4n3v3(transMat.top(), numverts, cnv, numfacets, facets, 
                      materialIndex);
}

void OSPRayDisplayDevice::trimesh_n3b_v3f(char *n, float *v, int numfacets) {
  add_material();
  ort->trimesh_n3b_v3f(transMat.top(), matData[colorIndex], n, v, numfacets,  materialIndex);
}

void OSPRayDisplayDevice::trimesh_n3f_v3f(float *n, float *v, int numfacets) {
  add_material();
  ort->trimesh_n3f_v3f(transMat.top(), matData[colorIndex], n, v, numfacets,  materialIndex);
}


#if 0
void OSPRayDisplayDevice::trimesh_n3fopt_v3f(float *n, float *v, int numfacets) {
  add_material();
  ort->trimesh_v3f(transMat.top(), matData[colorIndex], v, numfacets,  materialIndex);
}

#endif

void OSPRayDisplayDevice::tristrip(int numverts, const float * cnv,
                         int numstrips, const int *vertsperstrip,
                         const int *facets) {
  add_material();
  ort->tristrip(transMat.top(), numverts, cnv, numstrips, vertsperstrip, 
                facets, materialIndex);
}



void OSPRayDisplayDevice::write_lights() {
  int i;
  int lightcount = 0;

  // clear all existing lights before (re)appending the current lights,
  // otherwise if the OSPRay context is reused, we will crash and burn.
  ort->clear_all_lights();

  // directional lights
  for (i=0; i<DISP_LIGHTS; i++) {
    if (lightState[i].on) {
      ort->add_directional_light(lightState[i].pos, lightState[i].color);
      lightcount++;
    }
  }

#if 0
  // advanced positional lights
  for (i=0; i<DISP_LIGHTS; i++) {
    if (advLightState[i].on) {
      float pos[3];

      // always use world coordinates for now
      vec_copy(pos, advLightState[i].pos);

      if (advLightState[i].spoton) {
        printf("OSPRayDisplayDevice) SpotLight not implemented yet ...\n");
      } else {
        apitexture tex;
        memset(&tex, 0, sizeof(apitexture));

        tex.col.r=advLightState[i].color[0];
        tex.col.g=advLightState[i].color[1];
        tex.col.b=advLightState[i].color[2];

        void *l = rt_light(rtscene,
                           rt_texture(rtscene, &tex),
                           /* negate position to correct handedness... */
                           rt_vector(pos[0], pos[1], -pos[2]), 0.0f);

        /* emit light attentuation parameters if needed */
        if (advLightState[i].constfactor != 1.0f ||
            advLightState[i].linearfactor != 0.0f ||
            advLightState[i].quadfactor != 0.0f) {
          rt_light_attenuation(l,
                               advLightState[i].constfactor,
                               advLightState[i].linearfactor,
                               advLightState[i].quadfactor);
        }
      }

      lightcount++;
    }
  }
#endif

  if (lightcount < 1) {
    msgWarn << "No lights defined in molecular scene!!" << sendmsg;
  }
}


void OSPRayDisplayDevice::write_materials() {
  ort->set_bg_color(backColor);

  // Specify OSPRay background sky sphere if background gradient
  // mode is enabled.
  if (backgroundmode == 1) {
    float bspheremag = 0.5f;

    // compute positive/negative magnitude of sphere gradient
    switch (projection()) {
      case DisplayDevice::ORTHOGRAPHIC:
        // For orthographic views, Tachyon uses the dot product between
        // the incident ray origin and the sky sphere gradient "up" vector,
        // since all camera rays have the same direction and differ only
        // in their origin.
        bspheremag = vSize / 4.0f;
        break;

      case DisplayDevice::PERSPECTIVE:
      default:
        // For perspective views, Tachyon uses the dot product between
        // the incident ray and the sky sphere gradient "up" vector,
        // so for larger values of vSize, we have to clamp the maximum
        // magnitude to 1.0.
        bspheremag = (vSize / 2.0f) / (eyePos[2] - zDist);
        if (bspheremag > 1.0f)
          bspheremag = 1.0f;
        break;
    }

    if (projection() == DisplayDevice::ORTHOGRAPHIC)
      ort->set_bg_mode(OSPRayRenderer::RT_BACKGROUND_TEXTURE_SKY_ORTHO_PLANE);
    else
      ort->set_bg_mode(OSPRayRenderer::RT_BACKGROUND_TEXTURE_SKY_SPHERE);

    float updir[3] = { 0.0f, 1.0f, 0.0f };
    ort->set_bg_color_grad_top(backgradienttopcolor);
    ort->set_bg_color_grad_bot(backgradientbotcolor);
    ort->set_bg_gradient(updir);
    ort->set_bg_gradient_topval(bspheremag);
    ort->set_bg_gradient_botval(-bspheremag);
  } else {
    ort->set_bg_mode(OSPRayRenderer::RT_BACKGROUND_TEXTURE_SOLID);
  }
}


///////////////////// public virtual routines

void OSPRayDisplayDevice::write_header() {
  wkf_timer_start(ort_timer);

  ort->setup_context(xSize, ySize);
  write_materials();
  write_lights();

  ort->set_aa_samples(aasamples); // set with current FileRenderer values

  // render with/without shadows
  if (shadows_enabled() || ao_enabled()) {
    if (shadows_enabled() && !ao_enabled())
      msgInfo << "Shadow rendering enabled." << sendmsg;

    ort->shadows_on(1); // shadowing mode required
  } else {
    ort->shadows_on(0); // disable shadows by default
  }

  // render with ambient occlusion, but only if shadows are also enabled
  if (ao_enabled()) {
    msgInfo << "Ambient occlusion enabled." << sendmsg;
    msgInfo << "Shadow rendering enabled." << sendmsg;
    ort->set_ao_samples(aosamples); // set with current FileRenderer values
  } else {
    ort->set_ao_samples(0); // disable AO rendering entirely
  }

  // Always set the AO parameters, that way the user can enable/disable
  // AO on-the-fly in the interactive renderer
  ort->set_ao_ambient(get_ao_ambient());
  ort->set_ao_direct(get_ao_direct());

  // render with depth of field, but only for perspective projection
  if (dof_enabled()) {
    msgInfo << "DoF focal blur enabled." << sendmsg;
    ort->dof_on(1); // enable DoF rendering
    ort->set_camera_dof_fnumber(get_dof_fnumber());
    ort->set_camera_dof_focal_dist(get_dof_focal_dist());
  } else {
    ort->dof_on(0); // disable DoF rendering
  }

  // set depth cueing parameters
  float start = get_cue_start();
  float end = get_cue_end();
  float density = get_cue_density();
  if (cueingEnabled) {
    switch (cueMode) {
      case CUE_LINEAR:
        ort->set_cue_mode(OSPRayRenderer::RT_FOG_LINEAR, start, end, density);
        break;

      case CUE_EXP:
        ort->set_cue_mode(OSPRayRenderer::RT_FOG_EXP, start, end, density);
        break;

      case CUE_EXP2:
        ort->set_cue_mode(OSPRayRenderer::RT_FOG_EXP2, start, end, density);
        break;

      case NUM_CUE_MODES:
        // this should never happen
        break;
    }
  } else {
    ort->set_cue_mode(OSPRayRenderer::RT_FOG_NONE, start, end, density);
  }
}


void OSPRayDisplayDevice::write_trailer(void){
  send_cylinder_buffer(); // send any unsent accumulated cylinder buffer...
  send_triangle_buffer(); // send any unsent accumulated triangle buffer...

#if 0
  printf("OSPRay: z: %f zDist: %f vSize %f\n", eyePos[2], zDist, vSize);
#endif
  switch (projection()) {
    case DisplayDevice::ORTHOGRAPHIC:
      ort->set_camera_projection(OSPRayRenderer::RT_ORTHOGRAPHIC);
      ort->set_camera_zoom(0.5f / (1.0 / (vSize / 2.0)));
      break;

    case DisplayDevice::PERSPECTIVE:
    default:
      ort->set_camera_projection(OSPRayRenderer::RT_PERSPECTIVE);
      // ort->set_camera_zoom(0.5f / ((eyePos[2] - zDist) / vSize));
      ort->set_camera_zoom(0.5f * vSize / (eyePos[2] - zDist));
  }

  // set stereoscopic display parameters
  ort->set_camera_stereo_eyesep(eyeSep);
  ort->set_camera_stereo_convergence_dist(eyeDist);

  char *verbstr = getenv("VMDOSPRAYVERBOSE");
  if (verbstr != NULL) {
    if (!strupcmp(verbstr, "TIMING") || !strupcmp(verbstr, "DEBUG")) {
      double time_scene_graph = wkf_timer_timenow(ort_timer);
      printf("OSPRayDisplayDevice) scene graph construction time %.2f\n",
             time_scene_graph);
    }
  }

#if defined(VMDOSPRAY_INTERACTIVE_OPENGL)
  if (isinteractive)
    ort->render_to_glwin(my_filename); // interactive progressive ray tracer
  else
#endif
    ort->render_to_file(my_filename);  // render the scene in batch mode...


  if (getenv("VMDOSPRAYNODESTROYCONTEXT") == NULL) {
    // destroy the current context, because we haven't done enough to ensure
    // that we're managing memory well without tearing it all down.
    delete ort;

    // make a new OSPRayRenderer object so we're ready for the next run...
    ort = new OSPRayRenderer();
  } else {
    // reset internal state between renders
    // reinitialize material cache, clean context state
    ort->destroy_scene();
  }

  wkf_timer_stop(ort_timer);
  printf("OSPRayDisplayDevice) Total rendering time: %.2f sec\n", wkf_timer_time(ort_timer));

  reset_vars(); 
}





