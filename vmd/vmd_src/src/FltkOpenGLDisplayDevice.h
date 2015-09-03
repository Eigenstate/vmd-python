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
 *	$RCSfile: FltkOpenGLDisplayDevice.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.20 $	$Date: 2011/01/28 20:08:39 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Subclass of OpenGLRenderer, this object has routines used by all the
 * different display devices that are OpenGL-specific.  Will render drawing
 * commands into a single X window.
 *
 ***************************************************************************/
#ifndef OPENGLDISPLAYDEVICE_H
#define OPENGLDISPLAYDEVICE_H

#include "OpenGLRenderer.h"

class VMDApp;
class Fl_Gl_Window;

/// OpenGLRenderer subclass, displays to an FLTK window
class FltkOpenGLDisplayDevice : public OpenGLRenderer {

private:
  Fl_Gl_Window *window;
 
public:
  // constructor/destructor
  // arguments: argc/argv from main, 
  // handle to VMDApp,
  // and the size and location for the window, if known
  FltkOpenGLDisplayDevice(int, char **, VMDApp *,int *size = NULL, int *loc = NULL);
  virtual ~FltkOpenGLDisplayDevice(void);


  //
  // get the current state of the device's pointer (i.e. cursor if it has one)
  //
  int lastevent;   // last FLTK event we processed
  int lastkeycode; // last keyboard keycode pressed
  int lastbtn;     // last mouse button pressed 
  int lastzdelta;  // last mouse wheel delta (windows parlance)

  virtual int x(void);		// abs pos of cursor from lower-left corner
  virtual int y(void);		// same, for y direction
  virtual int shift_state(void);// return the shift state (ORed of the
				// enum in DisplayDevice)

  // get the current state of the Spaceball if one is available
  // returns rx ry rz, tx ty tz, buttons
  virtual int spaceball(int *, int *, int *, int *, int *, int *, int *);

  // set the Nth cursor shape as the current one.  If no arg given, the
  // default shape (n=0) is used.
  virtual void set_cursor(int);

  // resize window
  virtual void do_resize_window(int, int);

  // reposition window
  virtual void do_reposition_window(int xpos, int ypos);

  //
  // event handling routines
  //

  // read the next event ... returns an event type (one of the above ones),
  // and a value.  Returns success, and sets arguments.
  virtual int read_event(long &, long &);
 
  //
  // virtual routines for preparing to draw, drawing, and finishing drawing
  //
  virtual void update(int do_update = TRUE);	// finish up after drawing
  virtual void reshape(void);			// refresh device after change

  // virtual routine for capturing the screen to a packed RGB array
  virtual unsigned char * readpixels(int &x, int &y);

  // Update xOrig and yOrig before computing screen position
  virtual void rel_screen_pos(float &x, float &y) {
    reshape();
    DisplayDevice::rel_screen_pos(x, y);
  }
};

#endif

