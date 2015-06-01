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
 *      $RCSfile: vmdsphere.vert,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.17 $       $Date: 2010/12/16 04:10:13 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  This file contains the VMD OpenGL vertex shader implementing
 *  a ray-traced sphere primitive with per-pixel lighting,
 *  phong highlights, etc.  The sphere is drawn within the confines of a
 *  correctly transformed bounding cube or viewer-aligned billboard.
 *  Much of the shading code is shared with the main VMD vertex shader,
 *  with a few optimizations that simplify the sphere shader due to the
 *  way it is specifically known to be used within VMD.  (certain texturing
 *  modes can't actually occur in practice, so the shader can be simplified
 *  relative to what the main fragment shader must implement.
 ***************************************************************************/

// requires GLSL version 1.10
#version 110

//
// Vertex shader varying and uniform variable definitions for data
// supplied by VMD. 
//
uniform int vmdprojectionmode;   // perspective=1 orthographic=0
uniform int vmdtexturemode;      // VMD texture mode

// 
// Outputs to fragment shader
//
varying vec3 oglcolor;           // output interpolated color to frag shader
varying vec3 V;                  // output view direction vector
varying vec3 spherepos;          // output transformed sphere position
varying vec3 rayorigin;          // output ray origin
varying float sphereradsq;       // output transformed sphere radius squared

//
// VMD Sphere Vertex Shader
//
void main(void) {
  // transform vertex to Eye space for user clipping plane calculations
  vec4 ecpos = gl_ModelViewMatrix * gl_Vertex;
  gl_ClipVertex = ecpos;

  // pass along vertex color for use fragment shading,
  // fragment shader will get an interpolated color.
  oglcolor = vec3(gl_Color);

  // Sphere-specific rendering calculations
  // Transform sphere location
  vec4 spos = gl_ModelViewMatrix * vec4(0, 0, 0, 1.0);
  spherepos = vec3(spos) / spos.w;

  // setup fog coordinate for fragment shader, use sphere center
  gl_FogFragCoord = abs(spos.z);

  // transform sphere radius
  vec4 rspos = gl_ModelViewMatrix * vec4(1.0, 0, 0, 1.0);
  sphereradsq = length(spherepos - (vec3(rspos) / rspos.w));
  sphereradsq *= sphereradsq; // square it, to save time in frag shader

  if (vmdprojectionmode == 1) {
    // set view direction vector from eye coordinate of vertex, for 
    // perspective views
    V = normalize(vec3(ecpos) / ecpos.w);
    rayorigin = vec3(0,0,0);
  } else {
    // set view direction vector with constant eye coordinate, used for
    // orthographic views
    V = vec3(0.0, 0.0, -1.0);
    rayorigin = vec3((ecpos.xy / ecpos.w), 0.0);  
  }

  // transform vertex to Clip space
  gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}



