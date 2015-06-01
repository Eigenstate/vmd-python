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
 *	$RCSfile: VMDTitle.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.58 $	$Date: 2011/03/05 17:14:51 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * A flashy title object which is displayed when the program starts up,
 * until a molecule is loaded.
 *
 ***************************************************************************/
#include <math.h>
#include "VMDTitle.h"
#include "DisplayDevice.h"
#include "config.h"
#include "utilities.h"
#include "Scene.h"

// functions which put V, M, and D on the display list
// The max height is y+1 (and the min is y+0, since they
// are all capital letters).  The leftmost (on x)
// graphics starts at x.  However, different letters
// are different widths.
//  The width of a stroke is STROKE_WIDTH.
#define STROKE_WIDTH (0.2f + 0.0f)


// given the front surface coordinates(4) in counter clockwise direction
// draw the solid box that goes around the box with that
// surface, and a depth of 0.25 deep
// The optional field requests that the ends also be covered
static void draw_outsides(float c[8][3], VMDDisplayList *disp, int draw_ends) {
  float zoffset[3] = { 0, 0, 0.25}; // make the other 4 coords
  for (int i=0; i<4; i++)
    vec_sub(c[i+4], c[i], zoffset);

  DispCmdTriangle tri;
  // draw the front parallegram
  tri.putdata(c[0], c[1], c[2], disp);
  tri.putdata(c[2], c[3], c[0], disp);

  // draw the rear parallegram
  tri.putdata(c[5], c[4], c[6], disp);
  tri.putdata(c[6], c[4], c[7], disp);
  
  // draw the left face
  tri.putdata(c[0], c[3], c[7], disp);
  tri.putdata(c[7], c[4], c[0], disp);

  // draw the right face
  tri.putdata(c[1], c[5], c[6], disp);
  tri.putdata(c[6], c[2], c[1], disp);

  if (!draw_ends)
    return;
    
  // draw the top face
  tri.putdata(c[0], c[5], c[4], disp);
  tri.putdata(c[5], c[0], c[1], disp);
  
  // draw the bottom face
  tri.putdata(c[2], c[7], c[3], disp);
  tri.putdata(c[7], c[2], c[6], disp);
}


// draw a parallelpiped from start to end
//  the top and bottoms are flat
//  the front parallegram marks out the regions
//   (startx-STROKE_WIDTH/2, starty) to 
//   (startx+STROKE_WIDTH/2, starty) to 
//   (endx-STROKE_WIDTH/2, endy) to 
//   (endx+STROKE_WIDTH/2, endy) and back
static void draw_stroke(float startx, float starty, float endx, float endy, 
                        VMDDisplayList *disp) {
  float c[8][3];
  c[0][0] = startx - (STROKE_WIDTH+0.0f)/(2.0f);
  c[0][1] = starty; 
  c[0][2] = 0;
  c[1][0] = startx + (STROKE_WIDTH+0.0f)/(2.0f);
  c[1][1] = starty; 
  c[1][2] = 0;
  c[2][0] = endx + (STROKE_WIDTH+0.0f)/(2.0f);
  c[2][1] = endy; 
  c[2][2] = 0;
  c[3][0] = endx - (STROKE_WIDTH+0.0f)/(2.0f);
  c[3][1] = endy; 
  c[3][2] = 0;
  draw_outsides(c, disp, TRUE);
}

static void draw_letter_V(float x, float y, VMDDisplayList *disp) {
  // draw the down stroke (note the order, this is because of the 
  //  way I define the coordinates in the draw_stroke routine for width)
  draw_stroke(x+STROKE_WIDTH*(1.5f), y+0, x+STROKE_WIDTH/2.0f, y+1.0f, disp);
  // and the up stroke   
  draw_stroke(x+STROKE_WIDTH*(1.5f), y+0, x+STROKE_WIDTH*2.5f, y+1.0f, disp);
}

static float letter_V_width(void) {
 return STROKE_WIDTH*2.0;
}

static void draw_letter_M(float x, float y, VMDDisplayList *disp) {
  // up
  draw_stroke(x+STROKE_WIDTH/2.0f, y, x+STROKE_WIDTH*1.5f, y+1.0f, disp);
  // down
  draw_stroke(x+STROKE_WIDTH*2.5f, y, x+STROKE_WIDTH*1.5f, y+1.0f, disp);
  // up
  draw_stroke(x+STROKE_WIDTH*2.5f, y, 
              x+STROKE_WIDTH*3.5f, y+1.0f, disp);
  // down
  draw_stroke(x+STROKE_WIDTH*4.5f, y,
              x+STROKE_WIDTH*3.5f, y+1.0f, disp);
}

static float letter_M_width(void) {
  return STROKE_WIDTH*5.0f;
}

// given the angle and the major/minor radii for the ellipse
//   angle = 0 => along major axis
//   increasing angle is in the clockwise direction
// find the x, y point of the axis at that angle (with z=0)
static void find_ellipse_coords(float angle, float major, float minor, float *data) {
  data[0] = minor * sinf(angle);
  data[1] = major * cosf(angle);
  data[2] = 0;
}


// This one is tricky
static void draw_letter_D(float x, float y, VMDDisplayList *disp) {
  // up
  draw_stroke(x+STROKE_WIDTH/2.0f, y, x+STROKE_WIDTH/2.0f, y+1.0f, disp);
  // and the curve
  float stepsize = (float) VMD_PI/60.0f;
  float offset[3] = {STROKE_WIDTH, 0.5f, 0.0f};
  offset[0] += x;
  offset[1] += y;
  float c[8][3];
  int i, j;
  for (i=0; i<60; i++) {
    // inside coords
    find_ellipse_coords(stepsize*i, 0.5f - STROKE_WIDTH, STROKE_WIDTH, c[1]);
    find_ellipse_coords(stepsize*(i+1), 0.5f - STROKE_WIDTH, STROKE_WIDTH, c[2]);
    // outside coords
    find_ellipse_coords(stepsize * i, 0.5 , STROKE_WIDTH * 2, c[0]);
    find_ellipse_coords(stepsize * (i+1), 0.5 , STROKE_WIDTH * 2, c[3]);
    // add the offsets
    for (j=0; j<4; j++)
      vec_add(c[j], c[j], offset);
    draw_outsides(c, disp, FALSE);
  }
}

/// And now the class definition
VMDTitle::VMDTitle(DisplayDevice *d, Displayable *par) 
: Displayable(par), disp(d) {
  // displayable characteristics
  rot_off();
  scale_off();
  glob_trans_off();
  cent_trans_off();
  letterson = TRUE;
  redraw_list();
  
  glob_trans_on();
  set_glob_trans(-0.1f, 0.65f, 0.10f);
  glob_trans_off();
  starttime = time_of_day();
}

void VMDTitle::redraw_list(void) {
  reset_disp_list();
  float x;  // offset to center VMD
  x = letter_V_width() + STROKE_WIDTH/2.0f + 
      letter_M_width() + STROKE_WIDTH/2.0f +
      letter_V_width(); // same width as 'V'
  x=x/2.0f; // center on the screen

  append(DMATERIALON);
  color.putdata(REGRED, cmdList);
  draw_letter_V(-x, -0.4f, cmdList);
  color.putdata(REGGREEN, cmdList);
  draw_letter_M(-x + STROKE_WIDTH/2.0f + letter_V_width(), -0.4f, cmdList);
  color.putdata(REGBLUE, cmdList);
  draw_letter_D(-x + STROKE_WIDTH/2.0f + letter_V_width() +
                     STROKE_WIDTH/2.0f + letter_M_width(), -0.4f, cmdList);
  append(DMATERIALOFF);

  if (letterson) {
    color.putdata(REGWHITE, cmdList);
    float pos[3];
    pos[0] = -x + 0.2f;
    pos[1] = -0.6f; 
    pos[2] =  0.0f;

    DispCmdText txt;
    DispCmdTextSize txtsize;
    txtsize.putdata(0.8f, cmdList);
    txt.putdata(pos, "Theoretical and Computational Biophysics Group", 1.0f, cmdList);
    pos[1] -= 0.1f;
    txt.putdata(pos, "NIH Resource for Macromolecular Modeling and Bioinformatics", 1.0f, cmdList);
    pos[1] -= 0.1f;
    txt.putdata(pos, "University of Illinois at Urbana-Champaign", 1.0f, cmdList);
    pos[1] -= 0.1f;
    txt.putdata(pos, VMD_AUTHORS_LINE1, 1.0f, cmdList);
    pos[1] -= 0.1f;
    txt.putdata(pos, VMD_AUTHORS_LINE2, 1.0f, cmdList);
    pos[1] -= 0.1f;
    txt.putdata(pos, VMD_AUTHORS_LINE3, 1.0f, cmdList);
  }
}

// simply fits a 2rd degree polynomial so that v(d=0)=0=v(d=t)
// and x(d=0)=start and x(d=t)=end
static float solve_position(float d, float t, float start, float end) {
  float a = 6.0f*(end-start)/(t*t*t);
  return a*d*d*(t/2.0f-d/3.0f)+start;
}

void VMDTitle::prepare() {
  double elapsed = time_of_day() - starttime;
  double delta;

  // Prevent the title screen from hogging the CPU/GPU when there's
  // nothing else going on.  This is particularly important for users
  // that start VMD and immediately start using Multiseq with no structure
  // data loaded at all.
  vmd_msleep(1); // sleep for 1 millisecond or more

  if (elapsed < 5 + 3) {  // display the title screen, no animation
    if (!letterson) {
      letterson = TRUE;
      redraw_list();
    }
    return;
  }

  elapsed -= 3;
  if (letterson) {
    letterson = FALSE;
    redraw_list();
  }

  if (elapsed < 30) { // just spin the VMD logo
    delta = elapsed - 5;
    rot_on();
    set_rot(solve_position((float) delta, 25.0f, 0.0f, 360.0f*8.0f), 'y');
    rot_off();
  }

  if (elapsed < 15) { 
    delta = elapsed - 5;
    scale_on();
    set_scale( 1.0f/(1.0f+ ((float) delta)/3.0f)); // and getting smaller
    scale_off();
    glob_trans_on();

    // and moving up
    set_glob_trans(0, 0.5f, solve_position((float) delta, 10.0f, 0.0f, 0.5f)); 
    glob_trans_off();
    return;
  }

  if (elapsed < 20) {
    return;
  }

  // I am at          ( 0  ,  0.5, 0.5)
  // I want to get to ( -.7  ,  0.9  , 0.5) in 10 secs
  if (elapsed < 30) {
    delta = elapsed - 20;
    glob_trans_on();
    set_glob_trans(
       solve_position((float) delta, 10.0f, 0.0f, -0.6f * disp->aspect()),
       solve_position((float) delta, 10.0f, 0.5f, 0.8f),
       solve_position((float) delta, 10.0f, 0.5f, 0.5f));
    glob_trans_off();
    scale_on();
    set_scale(solve_position((float) delta, 10.0f, 1.0f/(1.0f+10.0f/3.0f), 0.25f));
    scale_off();
    return;
  }

  if (elapsed < 35) 
    return;

  // just spin the VMD logo
  delta = elapsed - 35;
  rot_on();
  set_rot((float) delta * 360.0f / 6.0f, 'y');  
  rot_off();
}

