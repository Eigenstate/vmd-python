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
 *	$RCSfile: OpenGLPbufferDisplayDevice.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.5 $	$Date: 2014/11/08 06:52:30 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Subclass of OpenGLRenderer.
 * Will render drawing commands into a windowless off-screen rendering buffer.
 * This requires a windowing system display connection, but does not
 * require creation of any visible on-screen windows.
 *
 ***************************************************************************/
#ifndef OPENGLPBUFFERDISPLAYDEVICE_H
#define OPENGLPBUFFERDISPLAYDEVICE_H

#include "OpenGLRenderer.h"

class VMDApp;

#if !defined(_MSC_VER) && !(defined(VMDSDL) && defined(__APPLE__))
#include <GL/glx.h>

// NOTE: you may have to get copies of the latest OpenGL extension headers
// from the OpenGL web site if your Linux machine lacks them:
//   http://oss.sgi.com/projects/ogl-sample/registry/
#if defined(ARCH_LINUX) && !defined(VMDMESA)
#include <GL/glxext.h>
#endif
#endif

/// GLX-specific low-level handles and window IDs
typedef struct {
  Display *dpy;              ///< X server display connection
  int dpyScreen;             ///< X server screen number
  Window rootWindowID;       ///< ID of the root window
  Window windowID;           ///< ID of the graphics window
  GLXContext cx;             ///< GLX graphics context
} glxpbufferdata;


/// Subclass of OpenGLRenderer, this object has routines used by all the
/// different display devices that are OpenGL-specific.  Will render drawing
/// commands into a single X window.
class OpenGLPbufferDisplayDevice : public OpenGLRenderer {
private:
  unsigned int PbufferMaxXsz; ///< maximum Pbuffer X image dimension
  unsigned int PbufferMaxYsz; ///< maximum Pbuffer Y image dimension

public:
  glxpbufferdata glxsrv;

protected:
  // flag for whether a window was successfully created by open_window
  int have_window;

  // create a new window and set it's characteristics
  Window open_window(char *, int *, int *, int, char **);
  virtual void do_resize_window(int, int);
  virtual void do_reposition_window(int, int) {};

public:
  // constructor - trivial variable initialization, no window opened yet.
  OpenGLPbufferDisplayDevice();

  // real initialization; return TRUE if the window was successfully opened
  // or FALSE if it wasn't.  Pass argc/argv from main, and size and location
  // for the window, if known.  size must NOT be NULL.
  int init(int argc, char **argv, VMDApp *app, int *size, int *loc = NULL);

  virtual ~OpenGLPbufferDisplayDevice(void);

  // All display device subclasses from OpenGLRenderer (except OpenGLPbuffer)
  // support GUI's.
  virtual int supports_gui() { return FALSE; }

  //
  // get the current state of the device's pointer (i.e. cursor if it has one)
  //

  virtual int x(void) { return 0; } // abs pos of cursor from lower-left corner
  virtual int y(void) { return 0; } // same, for y direction

  // return the shift state (ORed of the enum in DisplayDevice)
  virtual int shift_state(void) { return 0; }

  // get the current state of the Spaceball if one is available
  // returns rx ry rz, tx ty tz, buttons
  virtual int spaceball(int *, int *, int *, int *, int *, int *, int *) {
    return 0;
  }

  // set the Nth cursor shape as the current one.
  virtual void set_cursor(int) {} 

  //
  // event handling routines
  //

  // queue the standard events (need only be called once ... but this is
  // not done automatically by the window because it may not be necessary or
  // even wanted)
  virtual void queue_events(void) {}

  // read the next event ... returns an event type (one of the above ones),
  // and a value.  Returns success, and sets arguments.
  virtual int read_event(long &retdev, long &retval) {
    retdev = WIN_NOEVENT;
    retval = 0;
    return (retdev != WIN_NOEVENT);
  }

  //
  // virtual routines for preparing to draw, drawing, and finishing drawing
  //
  virtual void update(int do_update = TRUE);	// finish up after drawing
  virtual void reshape(void);			// refresh device after change

  // virtual routine for capturing the screen to a packed RGB array
  virtual unsigned char * readpixels(int &x, int &y);
};

#endif

