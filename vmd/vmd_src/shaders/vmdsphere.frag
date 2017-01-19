/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2011 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: vmdsphere.frag,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.35 $       $Date: 2011/04/20 21:14:36 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  This file contains the VMD OpenGL fragment shader implementing 
 *  a ray-traced sphere primitive with per-pixel lighting,
 *  phong highlights, etc.  The sphere is drawn within the confines of a 
 *  correctly transformed bounding cube or viewer-aligned billboard.
 *  Much of the shading code is shared with the main VMD fragment shader,
 *  with a few optimizations that simplify the sphere shader due to the
 *  way it is specifically known to be used within VMD.  (certain texturing
 *  modes can't actually occur in practice, so the shader can be simplified
 *  relative to what the main fragment shader must implement.
 ***************************************************************************/

// requires GLSL version 1.10
// XXX If/when we implement usage of ARB_sample_shading for 
//     antialiasing of the sphere shader, we will have to 
//     crank the GLSL fragment shader version to 130.
#version 110

// macro to enable one-sided spheres for improved speed
#define FASTONESIDEDSPHERES     1

//
// Fragment shader varying and uniform variable definitions for data 
// supplied by VMD and/or the vertex shader
//
varying vec3 oglcolor;         // interpolated color from the vertex shader
varying vec3 V;                // view direction vector
varying vec3 spherepos;
varying vec3 rayorigin;
varying float sphereradsq;

uniform vec3 vmdlight0;        // VMD directional lights
uniform vec3 vmdlight1;
uniform vec3 vmdlight2;
uniform vec3 vmdlight3;

uniform vec3 vmdlight0H;       // Blinn halfway vectors for all four lights
uniform vec3 vmdlight1H;
uniform vec3 vmdlight2H;
uniform vec3 vmdlight3H;

uniform vec4 vmdlightscale;    // VMD light on/off state for all 4 VMD lights,
                               // represented as a scaling constant.  Could be
                               // done with on/off flags but ATI doesn't deal
                               // well with branching constructs, so this value
                               // is simply multiplied by the light's 
                               // contribution.  Hacky, but it works for now.

uniform vec4 vmdmaterial;      // VMD material properties
                               // [0] is ambient (white ambient light only)
                               // [1] is diffuse
                               // [2] is specular
                               // [3] is shininess

uniform int vmdprojectionmode; // perspective=1 orthographic=0
uniform vec4 vmdprojparms;     // VMD projection parameters
                               // [0] is nearClip
                               // [1] is farClip
                               // [2] is 0.5 * (farClip + nearClip)
                               // [3] is 1.0 / (farClip - nearClip)

uniform float vmdopacity;      // VMD global alpha value

uniform float vmdoutline;      // VMD outline shading
uniform float vmdoutlinewidth; // VMD outline shading width
uniform int vmdtransmode;      // VMD transparency mode
uniform int vmdfogmode;        // VMD depth cueing / fog mode
uniform int vmdtexturemode;    // VMD texture mode 0=off 1=modulate 2=replace
uniform sampler3D vmdtex0;     // active 3-D texture map


//
// VMD Sphere Fragment Shader (not for normal geometry)
//
void main(void) {
  vec3 raydir = normalize(V);
  vec3 spheredir = spherepos - rayorigin;
   
  //
  // Perform ray-sphere intersection tests based on the code in Tachyon
  //
  float b = dot(raydir, spheredir);
  float disc = b*b + sphereradsq - dot(spheredir, spheredir);

#if defined(FASTONESIDEDSPHERES)
  // only calculate the nearest intersection, for speed
  if (disc <= 0.0)
    discard; // ray missed sphere entirely, discard fragment

  // calculate closest intersection
  float tnear = b - sqrt(disc);

  if (tnear < 0.0)
    discard;
#else
  // calculate and sort both intersections
  if (disc <= 0.0)
    discard; // ray missed sphere entirely, discard fragment

  disc = sqrt(disc);

  // calculate farthest intersection
  float t2 = b + disc;
  if (t2 <= 0.0)
    discard; // farthest intersection is behind the eye, discard fragment

  // calculate closest intersection
  float t1 = b - disc;

  // select closest intersection in front of the eye
  float tnear;
  if (t1 > 0.0)
    tnear = t1;
  else
    tnear = t2;
#endif

  // calculate hit point and resulting surface normal
  vec3 pnt = rayorigin + tnear * raydir;
  vec3 N = normalize(pnt - spherepos);

  // Output the ray-sphere intersection point as the fragment depth 
  // rather than the depth of the bounding box polygons.
  // The eye coordinate Z value must be transformed to normalized device 
  // coordinates before being assigned as the final fragment depth.
  if (vmdprojectionmode == 1) {
    // perspective projection = 0.5 + (hfpn + (f * n / pnt.z)) / diff
    gl_FragDepth = 0.5 + (vmdprojparms[2] + (vmdprojparms[1] * vmdprojparms[0] / pnt.z)) * vmdprojparms[3];
  } else {
    // orthographic projection = 0.5 + (-hfpn - pnt.z) / diff
    gl_FragDepth = 0.5 + (-vmdprojparms[2] - pnt.z) * vmdprojparms[3];
  }

  // Done with ray-sphere intersection test and normal calculation
  // beginning of shading calculations
  float ambient = vmdmaterial[0];   // ambient
  float diffuse = 0.0;
  float specular = 0.0;
  float shininess = vmdmaterial[3]; // shininess 
  vec3 objcolor = oglcolor;         // default if texturing is disabled

  // perform texturing operations for volumetric data
  // The only texturing mode that applies to the sphere shader
  // is the GL_MODULATE texturing mode with texture coordinate generation.
  // The other texturing mode is never encountered in the sphere shader.
  // Note: texsture fetches cannot occur earlier than this point since we
  // don't know the fragment depth until this poing.
  if (vmdtexturemode == 1) {
    // emulate GL_MODULATE
    // The 3-D texture coordinates must be updated based
    // on the eye coordinate of the ray-sphere intersection in this case
    vec3 spheretexcoord;
    vec4 specpos = vec4(pnt, 1.0);
    spheretexcoord.s = dot(specpos, gl_EyePlaneS[0]);
    spheretexcoord.t = dot(specpos, gl_EyePlaneT[0]);
    spheretexcoord.p = dot(specpos, gl_EyePlaneR[0]);
    objcolor = oglcolor * vec3(texture3D(vmdtex0, spheretexcoord));
  }

  // calculate diffuse lighting contribution
  diffuse += max(0.0, dot(N, vmdlight0)) * vmdlightscale[0];
  diffuse += max(0.0, dot(N, vmdlight1)) * vmdlightscale[1];
  diffuse += max(0.0, dot(N, vmdlight2)) * vmdlightscale[2];
  diffuse += max(0.0, dot(N, vmdlight3)) * vmdlightscale[3];
  diffuse *= vmdmaterial[1]; // diffuse scaling factor

  // compute edge outline if enabled
  if (vmdoutline > 0.0) {
    float edgefactor = dot(N,V);
    edgefactor = 1.0 - (edgefactor*edgefactor);
    edgefactor = 1.0 - pow(edgefactor, (1.0-vmdoutlinewidth)*32.0);
    diffuse = mix(diffuse, diffuse * edgefactor, vmdoutline);
  }

  // Calculate specular lighting contribution with Phong highlights, based
  // on Blinn's halfway vector variation of Phong highlights
  specular += pow(max(0.0, dot(N, vmdlight0H)), shininess) * vmdlightscale[0];
  specular += pow(max(0.0, dot(N, vmdlight1H)), shininess) * vmdlightscale[1];
  specular += pow(max(0.0, dot(N, vmdlight2H)), shininess) * vmdlightscale[2];
  specular += pow(max(0.0, dot(N, vmdlight3H)), shininess) * vmdlightscale[3];
  specular *= vmdmaterial[2]; // specular scaling factor

  // Fog computations
  const float Log2E = 1.442695; // = log2(2.718281828)
  float fog = 1.0;
  if (vmdfogmode == 1) {
    // linear fog
    fog = (gl_Fog.end - gl_FogFragCoord) * gl_Fog.scale;
  } else if (vmdfogmode == 2) {
    // exponential fog
    fog = exp2(-gl_Fog.density * gl_FogFragCoord * Log2E);
  } else if (vmdfogmode == 3) {
    // exponential-squared fog
    fog = exp2(-gl_Fog.density * gl_Fog.density * gl_FogFragCoord * gl_FogFragCoord * Log2E);
  }
  fog = clamp(fog, 0.0, 1.0);       // clamp the final fog parameter [0->1)

  vec3 color = objcolor * vec3(diffuse) + vec3(ambient + specular);

  float alpha = vmdopacity;
  
  // Emulate Raster3D's angle-dependent surface opacity if enabled
  if (vmdtransmode==1) {
    alpha = 1.0 + cos(3.1415926 * (1.0-alpha) * dot(N,V));
    alpha = alpha*alpha * 0.25;
  }

  gl_FragColor = vec4(mix(vec3(gl_Fog.color), color, fog), alpha);
}


