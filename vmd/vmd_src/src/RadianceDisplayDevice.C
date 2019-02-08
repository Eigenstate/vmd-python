
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
 *	$RCSfile: RadianceDisplayDevice.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.48 $	$Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Writes to the format for Radiance.  For more information about that
 * package, see http://radsite.lbl.gov/radiance/HOME.html .  It provides
 * conversion programs to go from its format to something normal (eg,
 * ra_ps, ra_tiff, and ra_gif).
 *
 ***************************************************************************/
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "RadianceDisplayDevice.h"
#include "Matrix4.h"
#include "DispCmds.h"
#include "Inform.h"
#include "utilities.h"

#define DEFAULT_RADIUS 0.002
#define DASH_LENGTH 0.02

// Be careful when you modify the coordinates.  To make things view the
// right way, I have to rotate everything around the (x,y,z) = (1,1,1)
// vector so that x->z, y->x, and z->y

#define ORDER(x,y,z) -z, -x, y

///////////////////////// constructor and destructor

// constructor ... initialize some variables
RadianceDisplayDevice::RadianceDisplayDevice() 
: FileRenderer("Radiance", "Radiance 4.0", "vmdscene.rad", 
 "oconv %s > %s.oct; rview -pe 100 -vp -3.5 0 0 -vd 1 0 0 %s.oct") {
  reset_vars(); // initialize state variables
}
               
//destructor
RadianceDisplayDevice::~RadianceDisplayDevice(void) { }

void RadianceDisplayDevice::reset_vars(void) {
  // clear out the r/g/b/t arrays
  red.clear();
  green.clear();
  blue.clear();
  trans.clear();

  cur_color = 0;
}


///////////////////////// protected nonvirtual routines

// draw a point
void RadianceDisplayDevice::point(float * spdata) {
  float vec[3];

  // transform the world coordinates
  (transMat.top()).multpoint3d(spdata, vec);
   
  // draw the sphere
  set_color(colorIndex);

  fprintf(outfile, "color%d sphere ball\n0\n0\n4 %4f %4f %4f %4f\n",
          cur_color, ORDER(vec[0], vec[1], vec[2]), 
          float(lineWidth) * DEFAULT_RADIUS);
}

// draw a sphere
void RadianceDisplayDevice::sphere(float * spdata) {
  
  float vec[3];
  float radius;
    
  // transform the world coordinates
  (transMat.top()).multpoint3d(spdata, vec);
  radius = scale_radius(spdata[3]);
   
  // draw the sphere
  set_color(colorIndex);

  fprintf(outfile, "color%d sphere ball\n0\n0\n4 %4f %4f %4f %4f\n",
          cur_color, ORDER(vec[0], vec[1], vec[2]), radius);
}

// draw a line (cylinder) from a to b
void RadianceDisplayDevice::line(float *a, float*b) {
  int i, j, test;
  float dirvec[3], unitdirvec[3];
  float from[3], to[3], tmp1[3], tmp2[3];

  if (lineStyle == ::SOLIDLINE) {
    // transform the world coordinates
    (transMat.top()).multpoint3d(a, from);
    (transMat.top()).multpoint3d(b, to);
    
    // draw the cylinder
    set_color(colorIndex);
    fprintf(outfile, "color%d cylinder cyl\n0\n0\n7 ", cur_color);
    fprintf(outfile, "%4f %4f %4f ", 
            ORDER(from[0], from[1], from[2])); // first point
    fprintf(outfile, "%4f %4f %4f ", 
            ORDER(to[0], to[1], to[2])); // second point
    fprintf(outfile, "%4f\n", float(lineWidth)*DEFAULT_RADIUS); // radius
        
  } else if (lineStyle == ::DASHEDLINE) {
    // transform the world coordinates
    (transMat.top()).multpoint3d(a, tmp1);
    (transMat.top()).multpoint3d(b, tmp2);

    // how to create a dashed line
    vec_sub(dirvec, tmp2, tmp1);  // vector from a to b
    vec_copy(unitdirvec, dirvec);
    vec_normalize(unitdirvec);    // unit vector from a to b
    test = 1;
    i = 0;
    while (test == 1) {
      for (j=0; j<3; j++) {
        from[j] = (float) (tmp1[j] + (2*i    )*DASH_LENGTH*unitdirvec[j]);
          to[j] = (float) (tmp1[j] + (2*i + 1)*DASH_LENGTH*unitdirvec[j]);
      }
      if (fabsf(tmp1[0] - to[0]) >= fabsf(dirvec[0])) {
        vec_copy(to, tmp2);
        test = 0;
      }

      // draw the cylinder
      set_color(colorIndex);
      fprintf(outfile, "color%d cylinder cyl\n0\n0\n7 ", cur_color);
      // first point
      fprintf(outfile, "%4f %4f %4f ", ORDER(from[0], from[1], from[2]));
      // second point
      fprintf(outfile, "%4f %4f %4f ", ORDER(to[0], to[1], to[2])); 
      // radius
      fprintf(outfile, "%4f\n", float(lineWidth)*DEFAULT_RADIUS);
      i++;
    }
  } else {
    msgErr << "RadianceDisplayDevice: Unknown line style " << lineStyle << sendmsg;
  }
}

// draw a cylinder
void RadianceDisplayDevice::cylinder(float *a, float *b, float r,int /*filled*/) {

  float vec1[3], vec2[3];
  
  // transform the world coordinates
  (transMat.top()).multpoint3d(a, vec1);
  (transMat.top()).multpoint3d(b, vec2);
    
  // draw the cylinder

  set_color(colorIndex);

  fprintf(outfile, "color%d cylinder cyl\n0\n0\n7 ", cur_color);
  // first point
  fprintf(outfile, "%4f %4f %4f ", 
    ORDER(vec1[0], vec1[1], vec1[2]));
  // second point
  fprintf(outfile, "%4f %4f %4f ", 
    ORDER(vec2[0], vec2[1], vec2[2])); 
  // radius
  fprintf(outfile, "%4f\n", scale_radius(r));

  // and fill in the ends
  float normal[3];
  vec_sub(normal, vec1, vec2);
  vec_normalize(normal);

  // one end
  set_color(colorIndex);

  fprintf(outfile, "color%d ring cyl_end\n0\n0\n8 ", cur_color);
  fprintf(outfile, "%4f %4f %4f ",         // location
    ORDER(vec1[0], vec1[1], vec1[2]));
  fprintf(outfile, "%4f %4f %4f ",         // normal
    ORDER(normal[0], normal[1], normal[2]));
  fprintf(outfile, "0 %4f\n", scale_radius(r)); // radii

  // the other end
  normal[0] = -normal[0];
  normal[1] = -normal[1];
  normal[2] = -normal[2];
  set_color(colorIndex);

  fprintf(outfile, "color%d ring cyl_end\n0\n0\n8 ", cur_color);
  fprintf(outfile, "%4f %4f %4f ",         // location
    ORDER(vec2[0], vec2[1], vec2[2]));
  fprintf(outfile, "%4f %4f %4f ",         // normal
    ORDER(normal[0], normal[1], normal[2]));
  fprintf(outfile, "0 %4f\n", scale_radius(r)); // radii
  
}

// draw a two radius cone
void RadianceDisplayDevice::cone_trunc(float *a, float *b, float rad1, float rad2, int /* resolution */) {

  float vec1[3], vec2[3];
  
  // transform the world coordinates
  (transMat.top()).multpoint3d(a, vec1);
  (transMat.top()).multpoint3d(b, vec2);
    
  set_color(colorIndex);

  fprintf(outfile, "color%d cone a_cone\n0\n0\n8 ", cur_color);
  // first point
  fprintf(outfile, "%4f %4f %4f ", 
    ORDER(vec2[0], vec2[1], vec2[2]));
  // second point
  fprintf(outfile, "%4f %4f %4f ", 
    ORDER(vec1[0], vec1[1], vec1[2])); 
  // radius
  fprintf(outfile, "%4f %4f\n", scale_radius(rad2), scale_radius(rad1));
}

// draw a triangle
void RadianceDisplayDevice::triangle(const float *a, const float *b, const float *c, const float *, const float *, const float *) {

  float vec1[3], vec2[3], vec3[3];
  
  // transform the world coordinates
  (transMat.top()).multpoint3d(a, vec1);
  (transMat.top()).multpoint3d(b, vec2);
  (transMat.top()).multpoint3d(c, vec3);

  // draw the triangle

  set_color(colorIndex);

  fprintf(outfile, "color%d polygon poly\n0\n0\n9 ", cur_color); // triangle
  fprintf(outfile, "%4f %4f %4f ", 
    ORDER(vec1[0], vec1[1], vec1[2])); // point one
  fprintf(outfile, "%4f %4f %4f ", 
    ORDER(vec2[0], vec2[1], vec2[2])); // point two
  fprintf(outfile, "%4f %4f %4f\n", 
    ORDER(vec3[0], vec3[1], vec3[2])); // point three
}

// draw a square
void RadianceDisplayDevice::square(float *, float *a, float *b, float *c, float *d) {
  
  float vec1[3], vec2[3], vec3[3], vec4[3];
  
  // transform the world coordinates
  (transMat.top()).multpoint3d(a, vec1);
  (transMat.top()).multpoint3d(b, vec2);
  (transMat.top()).multpoint3d(c, vec3);
  (transMat.top()).multpoint3d(d, vec4);

  // draw the square

  set_color(colorIndex);

  fprintf(outfile, "color%d polygon poly\n0\n0\n12 ", cur_color); // triangle
  fprintf(outfile, "%4f %4f %4f ", 
    ORDER(vec1[0], vec1[1], vec1[2])); // point one
  fprintf(outfile, "%4f %4f %4f ", 
    ORDER(vec2[0], vec2[1], vec2[2])); // point two
  fprintf(outfile, "%4f %4f %4f ", 
    ORDER(vec3[0], vec3[1], vec3[2])); // point three
  fprintf(outfile, "%4f %4f %4f\n", 
    ORDER(vec4[0], vec4[1], vec4[2])); // point four

}

///////////////////// the color routines

void RadianceDisplayDevice::set_color(int cIndex)
{
  int num = red.num();
  int i;

  float r = matData[cIndex][0],
        g = matData[cIndex][0 + 1],
        b = matData[cIndex][0 + 2],
#if 0  /// XXX
        t = 1.0f - matData[cIndex][ALPHA_INDEX];
#else
        t = 1.0f;
#endif

  for (i = 0; i < num; i++) {
    if (r == red[i] && g == green[i] && b == blue[i] && t == trans[i]) {
      break;
    }
  }

  if (i == num) { // create a new color category
    red.append(r);
    green.append(g);
    blue.append(b);
    trans.append(t);
    // define it for radiance
    if (t != 0) {
      fprintf(outfile, "void trans color%d\n0\n0\n7 ", i);
      fprintf(outfile, "%f %f %f .05 .00 %f 1.0\n", r, g, b, t);
    }
    else {
      fprintf(outfile, "void plastic color%d\n0\n0\n5 ", i);
      fprintf(outfile, "%f %f %f .05 .05\n", r, g, b);
    }
  }
  //else {
  //  // the color is 'i' so print it
  //  fprintf(outfile, "color%d ", i);
  //}

  // Save the current color
  cur_color = i;
}
       

// write comment to file
void RadianceDisplayDevice::comment(const char *s) {
  fprintf (outfile, "# %s\n", s);
}

///////////////////// public virtual routines

// initialize the file for output
void RadianceDisplayDevice::write_header() {
    int i;

    // clear out the r/g/b/t arrays
    red.clear();
    green.clear();
    blue.clear();
    trans.clear();

    fprintf(outfile, "#\n");
    fprintf(outfile, "# Radiance input script: %s\n",my_filename);
    fprintf(outfile, "#\n");


    // write the light sources
    fprintf(outfile, "void dielectric invisible\n0\n0\n5 1 1 1 1 0\n");
    fprintf(outfile, "void illum bright\n1 invisible\n0\n"
            "3 10000 10000 10000\n");
    
    // do this instead of the right way (see later)
    // fprintf(outfile, "bright sphere fixture\n0\n0\n4  -10 0 0  .01\n");
    
    // background color is black until I figure out how to set it
    // interactively.  I'm thinking of having a glowing sphere or plane

    for (i = 0; i < DISP_LIGHTS; i++) {
        if (lightState[i].on) {
            float vec[3];

            (transMat.top()).multpoint3d(lightState[i].pos, vec);

            fprintf(outfile,
                    "bright sphere fixture\n0\n0\n4 %f %f %f .01\n",
                    ORDER(10 * vec[0], 10 * vec[1], 10 * vec[2]));
        }
    }
}

    
// clean up after yourself
void RadianceDisplayDevice::write_trailer() {
  msgInfo << "Radiance file generation finished" << sendmsg;
  reset_vars(); // reset state variables
}



