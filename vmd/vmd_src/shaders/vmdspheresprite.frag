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
 *      $RCSfile: vmdspheresprite.frag,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.2 $       $Date: 2011/06/03 17:55:54 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  This file contains the VMD OpenGL fragment shader implementing 
 *  per-pixel lighting with phong highlights etc.
 ***************************************************************************/

// requires GLSL version 1.10
#version 110

//
// Fragment shader varying and uniform variable definitions for data 
// supplied by VMD and/or the vertex shader
//
varying vec3 oglcolor;        // interpolated color from the vertex shader
varying vec3 V;               // view direction vector

uniform vec3 vmdlight0;       // VMD directional lights
uniform vec3 vmdlight1;
uniform vec3 vmdlight2;
uniform vec3 vmdlight3;

uniform vec3 vmdlight0H;      // Blinn halfway vectors for all four lights
uniform vec3 vmdlight1H;
uniform vec3 vmdlight2H;
uniform vec3 vmdlight3H;

uniform vec4 vmdlightscale;   // VMD light on/off state for all 4 VMD lights,
                              // represented as a scaling constant.  Could be
                              // done with on/off flags but ATI doesn't deal
                              // well with branching constructs, so this value
                              // is simply multiplied by the light's 
                              // contribution.  Hacky, but it works for now.

uniform vec4 vmdmaterial;     // VMD material properties
                              // [0] is ambient (white ambient light only)
                              // [1] is diffuse
                              // [2] is specular
                              // [3] is shininess

uniform float vmdopacity;     // VMD global alpha value

uniform float vmdoutline;     // VMD outline shading

uniform float vmdoutlinewidth;// VMD outline shading width

uniform int vmdtransmode;     // VMD transparency mode

uniform int vmdfogmode;       // VMD depth cueing / fog mode

uniform int vmdtexturemode;   // VMD texture mode 0=off 1=modulate 2=replace
uniform sampler3D vmdtex0;    // active 3-D texture map

//
// VMD Fragment Shader
//
void main(void) {
  // Texture coords for point sprites range from 0,0 to 1,1, so we use this
  // to compute which pixels fall on the sphere (circle)
  vec2 pos = gl_TexCoord[0].xy - vec2(0.5, 0.5);
  float r2 = pos.x*pos.x + pos.y*pos.y;
  if (r2 > 0.25)
    discard; // Discard fragments outside of the sphere (circle)

#if 0
  vec3 texcolor;                    // texture color if needed
  // perform texturing operations for volumetric data start texture
  // fetch as early as possible to hide memory latency
  if (vmdtexturemode != 0) {
    texcolor = vec3(texture3D(vmdtex0, gl_TexCoord[0].xyz));
  }
#endif
  
  // Compute the surface normal for fragments on the sphere
  //  vec3 N = normalize(vec3(pos.x, -pos.y, sqrt(0.25 - r2)));
  vec3 N = 2.0*vec3(pos.x, -pos.y, sqrt(0.25 - r2));

  // beginning of shading calculations
  float ambient = vmdmaterial[0];   // ambient
  float diffuse = 0.0;
  float specular = 0.0;
  float shininess = vmdmaterial[3]; // shininess 

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

  // calculate specular lighting contribution with Phong highlights, based
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

  vec3 objcolor = oglcolor * vec3(diffuse);         // texturing is disabled
#if 0
  if (vmdtexturemode == 1) {
    objcolor = oglcolor * texcolor * vec3(diffuse); // emulate GL_MODULATE
  } else if (vmdtexturemode == 2) {
    objcolor = texcolor;                            // emulate GL_REPLACE
  } 
#endif

  vec3 color = objcolor + vec3(ambient + specular);

  float alpha = vmdopacity;

  // Emulate Raster3D's angle-dependent surface opacity if enabled
  if (vmdtransmode==1) {
    alpha = 1.0 + cos(3.1415926 * (1.0-alpha) * dot(N,V));
    alpha = alpha*alpha * 0.25;
  }

  gl_FragColor = vec4(mix(vec3(gl_Fog.color), color, fog), alpha);
}


