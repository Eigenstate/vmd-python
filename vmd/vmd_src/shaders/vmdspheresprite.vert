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
 *      $RCSfile: vmdspheresprite.vert,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.2 $       $Date: 2011/06/03 17:55:55 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  This file contains the VMD OpenGL vertex shader implementing
 *  the vertex portion of per-pixel lighting with phong highlights etc.
 ***************************************************************************/

// requires GLSL version 1.10
#version 110

//
// Vertex shader varying and uniform variable definitions for data
// supplied by VMD. 
//
uniform int vmdprojectionmode;    // perspective=1 orthographic=0
uniform int   vmdtexturemode;     // VMD texture mode
uniform float vmdspritesize;      // VMD sprite sphere size 

// 
// Outputs to fragment shader
//
varying vec3 oglcolor;           // output interpolated color to frag shader
varying vec3 V;                  // output view direction vector

//
// VMD Vertex Shader
//
void main(void) {
  // transform vertex to Eye space for user clipping plane calculations
  vec4 ecpos = gl_ModelViewMatrix * gl_Vertex;
  gl_ClipVertex = ecpos;

  // pass along vertex color for use fragment shading,
  // fragment shader will get an interpolated color.
  oglcolor = vec3(gl_Color);

  // setup fog coordinate for fragment shader
  gl_FogFragCoord = abs(ecpos.z);

  // compute sphere radius scaling factor
#if 0
  float spscale = vmdspritesize * 10.0;
#else
  vec4 ospos = gl_ModelViewMatrix * vec4(0.0, 0, 0, 1.0);
  vec4 rspos = gl_ModelViewMatrix * vec4(1.0, 0, 0, 1.0);
  float spscale = vmdspritesize * 250.0 * 
                  length((vec3(ospos) / ospos.w) - (vec3(rspos) / rspos.w));
#endif
  if (vmdprojectionmode == 1) {
    // set view direction vector from eye coordinate of vertex, for 
    // perspective views
    V = normalize(vec3(ecpos) / ecpos.w);

    // compute point size
    gl_PointSize = max(1.0, min(1024.0, spscale / -(ecpos.z/ecpos.w)));
  } else {
    // set view direction vector with constant eye coordinate, used for
    // orthographic views
    V = vec3(0.0, 0.0, -1.0);

    // compute point size
    gl_PointSize = max(1.0, min(1024.0, spscale));
  }

#if 0
  // mode 0 disables texturing
  // mode 1 enables texturing, emulating GL_MODULATE, with linear texgen
  // mode 2 enables texturing, emulating GL_REPLACE, with linear texgen
  if (vmdtexturemode != 0) {
    // transform texture coordinates as would be done by linear texgen
    gl_TexCoord[0].s = dot(ecpos, gl_EyePlaneS[0]);
    gl_TexCoord[0].t = dot(ecpos, gl_EyePlaneT[0]);
    gl_TexCoord[0].p = dot(ecpos, gl_EyePlaneR[0]);
    gl_TexCoord[0].q = dot(ecpos, gl_EyePlaneQ[0]);
  }
#endif

  // transform vertex to Clip space
#if 1
  // not all drivers support ftransform() yet.
  gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
#else
  // We should exactly duplicate the fixed-function pipeline transform 
  // since VMD renders the scene in multiple passes, some of which must
  // continue to use the fixed-function pipeline.
  gl_Position = ftransform(); 
#endif


}



