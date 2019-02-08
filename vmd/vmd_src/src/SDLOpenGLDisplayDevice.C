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
 *	$RCSfile: SDLOpenGLDisplayDevice.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.40 $	$Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Subclass of DisplayDevice, this object has routines used by all the
 * different display devices that are OpenGL-specific.  Will render drawing
 * commands into a single X window.
 *
 ***************************************************************************/

#include <stdlib.h>
#include <math.h>
#if defined(__APPLE__)
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif
#include <SDL.h>

#include "OpenGLDisplayDevice.h"
#include "Inform.h"
#include "utilities.h"
#include "config.h"   // VMD version strings etc

// static data for this object
static const char *glStereoNameStr[OPENGL_STEREO_MODES] = 
 { "Off", 
   "QuadBuffered", 
   "HDTV SideBySide",
   "Checkerboard",
   "ColumnInterleaved",
   "RowInterleaved",
   "Anaglyph", 
   "SideBySide", 
   "AboveBelow",
   "Left", 
   "Right" };

static const char *glRenderNameStr[OPENGL_RENDER_MODES] =
{ "Normal",
  "GLSL",
  "Acrobat3D" };

static const char *glCacheNameStr[OPENGL_CACHE_MODES] =
{ "Off",
  "On" };


/////////////////////////  constructor and destructor  

OpenGLDisplayDevice::OpenGLDisplayDevice()
: OpenGLRenderer((char *) "VMD " VMDVERSION " OpenGL Display") {

  // set up data possible before opening window
  stereoNames = glStereoNameStr;
  stereoModes = OPENGL_STEREO_MODES;

  renderNames = glRenderNameStr;
  renderModes = OPENGL_RENDER_MODES;

  cacheNames = glCacheNameStr;
  cacheModes = OPENGL_CACHE_MODES;

  memset(&sdlsrv, 0, sizeof(sdlsrv));
  have_window = FALSE;
  screenX = screenY = 0;
  vmdapp = NULL;
}

int OpenGLDisplayDevice::init(int argc, char **argv, VMDApp* app, int *size, int *loc) {

  // open the window
  sdlsrv.windowID = open_window(name, size, loc, argc, argv);
  if (!have_window) return FALSE;

  // set flags for the capabilities of this display
  // whether we can do antialiasing or not.
  if (ext->hasmultisample) 
    aaAvailable = TRUE;  // we use multisampling over other methods
  else
    aaAvailable = FALSE; // no non-multisample implementation yet

  // set default settings
  if (ext->hasmultisample) {
    aa_on();  // enable fast multisample based antialiasing by default
              // other antialiasing techniques are slow, so only multisample
              // makes sense to enable by default.
  } 

  cueingAvailable = TRUE;
  cueing_on(); // leave depth cueing on by default, despite the speed hit.

  cullingAvailable = TRUE;
  culling_off();

  set_sphere_mode(sphereMode);
  set_sphere_res(sphereRes);
  set_line_width(lineWidth);
  set_line_style(lineStyle);

  // reshape and clear the display, which initializes some other variables
  reshape();
  normal();
  clear();
  update();

  // successfully opened window.
  return TRUE;
}

// destructor ... close the window
OpenGLDisplayDevice::~OpenGLDisplayDevice(void) {
  if (have_window) {
    free_opengl_ctx(); // free display lists, textures, etc
  }

  SDL_Quit();
}


/////////////////////////  protected nonvirtual routines  


// create a new window and set it's characteristics
int OpenGLDisplayDevice::open_window(char *nm, int *size, int *loc,
                                     int argc, char** argv) {
  int SX = 100, SY = 100, W, H;
 
  char *dispname = NULL;
  if ((dispname = getenv("VMDGDISPLAY")) == NULL)
    dispname = getenv("DISPLAY");

  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    msgErr << "Exiting due to SDL window creation failure." << sendmsg;
    SDL_Quit();
    return -1;
  }
  // get info about root window
  screenX = 1280; // XXX hack
  screenY = 1024;
  W = size[0];
  H = size[1];
  if (loc) {
    SX = loc[0];
    // The X11 screen uses Y increasing from upper-left corner down; this is
    // opposite to what GL does, which is the way VMD was set up originally
    SY = (screenY - loc[1]) - H;
  }
  ext->hasstereo = FALSE;        // stereo is off until we find out otherwise.
  ext->stereodrawforced = FALSE; // don't force stereo draws initially.
  ext->hasmultisample = FALSE;   // multisample is off until we find out otherwise.
  
  if (SDL_SetVideoMode(W, H, 32, SDL_OPENGL) == NULL) {
    msgInfo << "Tried 32 bit color and failed..." << sendmsg;
    if (SDL_SetVideoMode(W, H, 24, SDL_OPENGL) == NULL) { 
      msgInfo << "Tried 24 bit color and failed..." << sendmsg;
      if (SDL_SetVideoMode(W, H, 16, SDL_OPENGL) == NULL) {
        msgInfo << "Tried 16 bit color and failed..." << sendmsg;
        msgErr << "Cannot open display.  Exiting ..." << sendmsg;
        SDL_Quit();
        return -1;
      }
    }
  } 

  SDL_WM_SetCaption("VMD " VMDVERSION " OpenGL Display", NULL);

  // (9) configure the rendering properly
  setup_initial_opengl_state();  // setup initial OpenGL state

  // Tell init that we successfully created a window.
  have_window = TRUE;

  return 0; // return window id
}


/////////////////////////  public virtual routines  

void OpenGLDisplayDevice::do_resize_window(int width, int height) {
  // not implemented yet
}

void OpenGLDisplayDevice::do_reposition_window(int xpos, int ypos) {
  // not implemented yet
}

//
// get the current state of the device's pointer (i.e. cursor if it has one)
//

// abs pos of cursor from lower-left corner of display
int OpenGLDisplayDevice::x(void) {
  int rx;

  rx = 0;

  return rx;
}


// same, for y direction
int OpenGLDisplayDevice::y(void) {
  int ry;

  ry = 0;

  return ry;
}

// return the current state of the shift, control, and alt keys
int OpenGLDisplayDevice::shift_state(void) {
  int retval = 0;

  // return the result
  return retval;
}

// return the spaceball state, if any
int OpenGLDisplayDevice::spaceball(int *rx, int *ry, int *rz, int *tx, int *ty,
int *tz, int *buttons) {
  // not implemented yet
  return 0;
}


// set the Nth cursor shape as the current one.  If no arg given, the
// default shape (n=0) is used.
void OpenGLDisplayDevice::set_cursor(int n) {
  // unimplemented
}


//
// event handling routines
//

// queue the standard events (need only be called once ... but this is
// not done automatically by the window because it may not be necessary or
// even wanted)
void OpenGLDisplayDevice::queue_events(void) {
}

// read the next event ... returns an event type (one of the above ones),
// and a value.  Returns success, and sets arguments.
int OpenGLDisplayDevice::read_event(long &retdev, long &retval) {
  int done = 0;
  SDL_Event event;

  SDL_PollEvent(&event);

  if ( event.type == SDL_KEYDOWN ) {
    if ( event.key.keysym.sym == SDLK_ESCAPE ) {
printf("ESC pressed!!\n");
      done = 1;
    }
  }

  return FALSE;
}


//
// virtual routines for preparing to draw, drawing, and finishing drawing
//

// reshape the display after a shape change
void OpenGLDisplayDevice::reshape(void) {
  xSize = 512;
  ySize = 512;
  xOrig = 0;
  yOrig = 0;

  switch (inStereo) {
    case OPENGL_STEREO_SIDE:
      set_screen_pos(0.5f * (float)xSize / (float)ySize);
      break;

    case OPENGL_STEREO_ABOVEBELOW:
      set_screen_pos(2.0f * (float)xSize / (float)ySize);
      break;

    case OPENGL_STEREO_STENCIL_CHECKERBOARD:
    case OPENGL_STEREO_STENCIL_COLUMNS:
    case OPENGL_STEREO_STENCIL_ROWS:
      enable_stencil_stereo(inStereo);
      set_screen_pos((float)xSize / (float)ySize);
      break;
 
    default:
      set_screen_pos((float)xSize / (float)ySize);
      break;
  }
}

unsigned char * OpenGLDisplayDevice::readpixels_rgb3u(int &x, int &y) {
  unsigned char * img;

  x = xSize;
  y = ySize;

  if ((img = (unsigned char *) malloc(x * y * 3)) != NULL) {
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, x, y, GL_RGB, GL_UNSIGNED_BYTE, img);
  } else {
    x = 0;
    y = 0;
  } 

  return img; 
}

unsigned char * OpenGLDisplayDevice::readpixels_rgba4u(int &x, int &y) {
  unsigned char * img;

  x = xSize;
  y = ySize;

  if ((img = (unsigned char *) malloc(x * y * 4)) != NULL) {
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, x, y, GL_RGBA, GL_UNSIGNED_BYTE, img);
  } else {
    x = 0;
    y = 0;
  } 

  return img; 
}


// update after drawing
void OpenGLDisplayDevice::update(int do_update) {
  if(do_update)
    SDL_GL_SwapBuffers();

  glDrawBuffer(GL_BACK);
}


