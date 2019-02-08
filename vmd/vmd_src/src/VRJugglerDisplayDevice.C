/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2019 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 *cr VRJuggler patches contributed by Martijn Kragtwijk: m.kragtwijk@rug.nl
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: VRJugglerDisplayDevice.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.4 $       $Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * a VRJuggler specific display device for VMD
 ***************************************************************************/
#include "VRJugglerDisplayDevice.h" 
#include "Inform.h"
//#include <vrjuggler_ogl.h>          // include vrjuggler library access

// static string storage used for returning stereo modes
static const char *vrjugglerStereoNameStr[1] = {"VRJuggler"};

///////////////////////////////  constructor
VRJugglerDisplayDevice::VRJugglerDisplayDevice(void) : OpenGLRenderer("VRJuggler") {
  stereoNames = vrjugglerStereoNameStr;
  stereoModes = 1;
  doneGLInit = FALSE;    
  num_display_processes  = 0; //VRJugglerConfig->ActiveWalls;
							  // hebben we deze variabele wel nodig?
    
  // leave everything else up to the vrjuggler_gl_init_fn
}

///////////////////////////////  destructor
VRJugglerDisplayDevice::~VRJugglerDisplayDevice(void) {
  // nothing to do
}


/////////////////////////////  public routines  //////////////////////////

// set up the graphics on the seperate VRJuggler displays
void VRJugglerDisplayDevice::vrjuggler_gl_init_fn(void) {
  setup_initial_opengl_state();     // do all OpenGL setup/initialization now

  // follow up with mode settings
  aaAvailable = TRUE;               // enable antialiasing
  cueingAvailable = FALSE;          // disable depth cueing
  cullingAvailable = FALSE;         // disable culling 
  // XXX need to test this still
  // ext->hasstereo = VRJugglerInStereo();  // stereo availability test
  ext->hasstereo = TRUE;            // stereo is on initially
  ext->stereodrawforced = FALSE;    // no need for forced stereo draws
 
  glClearColor(0.0, 0.0, 0.0, 0.0); // set clear color to black

  aa_on();                          // force antialiasing on if possible
  cueing_off();                     // force depth cueing off

  // set default settings 
  set_sphere_mode(sphereMode);
  set_sphere_res(sphereRes);
  set_line_width(lineWidth);
  set_line_style(lineStyle);

  clear();                          // clear screen
  update();                         // swap buffers

  // we want the VRJuggler to be centered at the origin, and in the range -1, +1
  (transMat.top()).translate(0.0, 3.0, -2.0);
  (transMat.top()).scale(PI);

  doneGLInit = TRUE;                // only do this once
}

void VRJugglerDisplayDevice::set_stereo_mode(int) {
  // cannot change to stereo mode in the VRJuggler, it is setup at init time
}
  
void VRJugglerDisplayDevice::normal(void) {
  // prevent the OpenGLRenderer implementation of this routine
  // from overriding the projection matrices provided by the 
  // VRJuggler library.
}

// special render routine to check for graphics initialization
void VRJugglerDisplayDevice::render(const VMDDisplayList *cmdlist) {
  //msgInfo << ".";

  if(!doneGLInit) {
    vrjuggler_gl_init_fn();
  }

  // prepare for rendering
  glPushMatrix();
  multmatrix((transMat.top()));  // add our VRJuggler adjustment transformation

  // update the cached transformation matrices for use in text display, etc.
  // In the VRJuggler, we have to do this separately for all of the processors.  
  // Would be nice to do this outside of the render routine however, 
  // amortized over several Displayables.
  glGetFloatv(GL_PROJECTION_MATRIX, ogl_pmatrix);
  glGetFloatv(GL_MODELVIEW_MATRIX, ogl_mvmatrix);
  ogl_textMat.identity();
  ogl_textMat.multmatrix(ogl_pmatrix);
  ogl_textMat.multmatrix(ogl_mvmatrix);

  // call OpenGLRenderer to do the rest of the rendering the normal way
  OpenGLRenderer::render(cmdlist);
  glPopMatrix();
}

// update after drawing
void VRJugglerDisplayDevice::update(int do_update) {
}

