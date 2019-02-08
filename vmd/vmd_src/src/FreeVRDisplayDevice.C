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
 *      $RCSfile: FreeVRDisplayDevice.C,v $
 *      $Author: johns $        $Locker:  $                $State: Exp $
 *      $Revision: 1.33 $      $Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * a FreeVR specific display device for VMD
 ***************************************************************************/

#include <freevr.h> // include FreeVR library prototypes
#include "Inform.h"
#include "FreeVRDisplayDevice.h"

// static string storage used for returning stereo modes
static const char *freevrStereoNameStr[1] = {"FreeVR"};

///////////////////////////////  constructor
FreeVRDisplayDevice::FreeVRDisplayDevice(void) : OpenGLRenderer("FreeVR") {
  stereoNames = freevrStereoNameStr;
  stereoModes = 1;
  doneGLInit = FALSE;    
  num_display_processes  = vrContext->config->num_windows;

  // XXX migrated some initialization code into the constructor due
  //     to the order of initialization in CAVE/FreeVR builds
  aaAvailable = TRUE;               // enable antialiasing
  cueingAvailable = FALSE;          // disable depth cueing
  cullingAvailable = FALSE;         // disable culling
  ext->hasstereo = TRUE;            // stereo is on initially
  ext->stereodrawforced = FALSE;    // no need for force stereo draws

  ogl_useblendedtrans = 0;
  ogl_transpass = 0;
  ogl_useglslshader = 0;
  ogl_acrobat3dcapture = 0;
  ogl_lightingenabled = 0;
  ogl_rendstateserial = 1;    // force GLSL update on 1st pass
  ogl_glslserial = 0;         // force GLSL update on 1st pass
  ogl_glsltoggle = 1;         // force GLSL update on 1st pass
  ogl_glslmaterialindex = -1; // force GLSL update on 1st pass
  ogl_glslprojectionmode = DisplayDevice::PERSPECTIVE;
  ogl_glsltexturemode = 0;    // initialize GLSL projection to off


  // leave everything else up to the freevr_gl_init_fn
}

///////////////////////////////  destructor
FreeVRDisplayDevice::~FreeVRDisplayDevice(void) {
  // nothing to do
}


/////////////////////////////  public routines  //////////////////////////

// set up the graphics on the seperate FreeVR displays
void FreeVRDisplayDevice::freevr_gl_init_fn(void) {
  setup_initial_opengl_state();     // do all OpenGL setup/initialization now

  // follow up with mode settings
  aaAvailable = TRUE;               // enable antialiasing
  cueingAvailable = FALSE;          // disable depth cueing
  cullingAvailable = FALSE;         // disable culling
  ext->hasstereo = TRUE;            // stereo is on initially
  ext->stereodrawforced = FALSE;    // no need for force stereo draws

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

  // we want the CAVE to be centered at the origin, and in the range -1, +1
  (transMat.top()).translate(0.0, 3.0, -2.0);
  (transMat.top()).scale(VMD_PI);

  doneGLInit = TRUE;                // only do this once
}

void FreeVRDisplayDevice::set_stereo_mode(int) {
  // cannot change to stereo mode in FreeVR, it is setup at init time
}

void FreeVRDisplayDevice::normal(void) {
  // prevent the OpenGLRenderer implementation of this routine
  // from overriding the projection matrices provided by the
  // FreeVR library.
}

// special render routine to check for graphics initialization
void FreeVRDisplayDevice::render(const VMDDisplayList *cmdlist) {
  if(!doneGLInit) {
    freevr_gl_init_fn();
  }

  // prepare for rendering
  glPushMatrix();
  multmatrix((transMat.top()));  // add our FreeVR adjustment transformation

  // update the cached transformation matrices for use in text display, etc.
  // In FreeVR, we have to do this separately for all of the processors.
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
void FreeVRDisplayDevice::update(int do_update) {
  // XXX don't do buffer swaps in FreeVR!!!
  //     Though not well documented, it is implicitly illegal 
  //     to call glxSwapBuffers() or to call glDrawBuffer() 
  //     in a FreeVR application, since FreeVR does this for you.
#if 0 /* BS: my proof of concept code */
  vrSystemSetStatusDescription("in update!");
  if (vrGet2switchValue(2)) {
    vrUserTravelTranslate3d(VR_ALLUSERS, 0.0, 0.0, 0.1);
  }
#endif

  /****************************************************/
  /* Do some (global) world navigation                */
  /* BS: this is from my ex12_travel.c tutorial code. */
#define	WAND_SENSOR	1	/* TODO: this should really be based on the concept of props */
#define JS_EPSILON	0.125				/* the dead-zone before joystick movement is recognized */
#define MOVE_FACTOR	2.5				/* a scale factor to tune travel movement               */
#define TURN_FACTOR	25.0				/* a scale factor to tune travel movement               */
#define SUB_JS_EPSILON(a)	((a)-copysign(JS_EPSILON,(a)))	/* a macro to remove the joystick epsilon range */
static	vrTime		last_time = -1.0;		/* initially set to an invalid value as a flag */
	vrTime		sim_time = vrCurrentSimTime();	/* the current time of the simulation  */
	double		delta_time;			/* time since last update (in seconds) */
	double		delta_move;			/* distance an object should be moved */
	double		joy_x, joy_y;			/* Joystick values */
	vrVector	wand_rw_pointvec;		/* current vector pointing out of the wand (in RW coords) */

  /* These variables are for the grab-the-world travel */
static	vrMatrix	wand_at_grab;			/* the wand matrix as it was when the grab action was triggered */
static	vrMatrix	travel_at_grab;			/* the value of the travel matrix when the grab action was triggered */
static	int		world_grabbed = 0;		/* whether or not we're in the middle of a grab action */
	vrMatrix	new_world_matrix;		/* the matrix into which we calculate the world transformation */
	vrMatrix	current_wand_matrix;
	vrMatrix	invmat;

	/*****************************************************/
	/** Determine delta time from last simulation frame **/
	/*****************************************************/
	if (last_time == -1.0)
		delta_time = 0.0;
	else	delta_time = sim_time - last_time;
	last_time = sim_time;				/* now that delta_time has been calculated, we won't use last_time until next time */

	/* skip the update if the delta isn't big enough */
	if (delta_time <= 0.0)	/* can also choose a non-zero epsilon */
		return;

	/****************************/
	/** Handle Travel via wand **/
	/****************************/

	/* pressing the right wand button resets us to the initial position */
	if (vrGet2switchValue(3)) {
		vrUserTravelReset(VR_ALLUSERS);
	}

	/* use wand joystick to fly through world */
	/*  (but not while the world is grabbed!) */
	joy_x = vrGetValuatorValue(0);
	joy_y = vrGetValuatorValue(1);

	if (fabs(joy_x) > JS_EPSILON && !world_grabbed)
		vrUserTravelRotateId(VR_ALLUSERS, VR_Y, (delta_time * SUB_JS_EPSILON(joy_x) * -TURN_FACTOR));

	if (fabs(joy_y) > JS_EPSILON && !world_grabbed) {
		delta_move = delta_time * SUB_JS_EPSILON(joy_y) * MOVE_FACTOR;
		vrVectorGetRWFrom6sensorDir(&wand_rw_pointvec, WAND_SENSOR, VRDIR_FORE);
		vrUserTravelTranslate3d(VR_ALLUSERS,
			wand_rw_pointvec.v[VR_X] * delta_move,
			wand_rw_pointvec.v[VR_Y] * delta_move,
			wand_rw_pointvec.v[VR_Z] * delta_move);
	}

	/* New full-grab-the-world with matrix operations travel example */
	/* BS: NOTE that this can probably be done more cleanly/efficiently, and I will work on that in the future */
	switch (vrGet2switchDelta(2)) {
		case 1: /* just pressed */
			/* store the current location of the wand */
			/* TODO: we should "subtract-out" the current travel matrix */
			vrMatrixGet6sensorValues(&wand_at_grab, WAND_SENSOR);
			vrMatrixGetUserTravel(&travel_at_grab, 0);
			world_grabbed = 1;
			break;
		case 0: /* no change in state */
			if (world_grabbed) {
				/* move the world with the wand */
				vrMatrixGet6sensorValues(&current_wand_matrix, WAND_SENSOR);
				vrMatrixInvert(&invmat, &wand_at_grab);
				vrMatrixPostMult(&invmat, &current_wand_matrix);
				new_world_matrix = invmat;

				/* use locks to make the travel manipulations atomic */
				vrUserTravelLockSet(VR_ALLUSERS);
				vrUserTravelReset(VR_ALLUSERS);
				vrUserTravelTransformMatrix(VR_ALLUSERS, &travel_at_grab);	/* first put back the inital travel */
				vrUserTravelTransformMatrix(VR_ALLUSERS, &new_world_matrix);	/* now add in the wand-delta */
				vrUserTravelLockRelease(VR_ALLUSERS);
			}
			break;
		case -1: /* just released */
			world_grabbed = 0;
			break;
	}

}

