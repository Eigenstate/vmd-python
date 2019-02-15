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
 *	$RCSfile: OpenGLDisplayDevice.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.57 $	$Date: 2019/01/17 21:21:00 $
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

#if !defined(_MSC_VER) && !(defined(VMDSDL) && defined(__APPLE__))
#include <GL/glx.h>

// NOTE: you may have to get copies of the latest OpenGL extension headers
// from the OpenGL web site if your Linux machine lacks them:
//   http://oss.sgi.com/projects/ogl-sample/registry/
#if defined(ARCH_LINUX) && !defined(VMDMESA)
#include <GL/glxext.h>
#endif
#endif

#if defined(VMDSPACEWARE) && defined(WIN32)
#define OS_WIN32 1
#include "spwmacro.h"           // Spaceware include files
#include "si.h"                 // Spaceware include files
#endif

#if defined(VMDSDL) 

/// SDL-specific low-level handles and window IDs
typedef struct {
  int windowID;
} sdldata;

#else
#if defined(_MSC_VER)

/// Win32-specific low-level handles and window IDs
typedef struct {
  HWND hWnd;
  HDC hDC;
  HGLRC hRC;
  long scrwidth;
  long scrheight;
  int width;
  int height;

  // internal state variables we need to track
  int cursornum;
  long MouseX;
  long MouseY;
  long MouseFlags;
  int  WEvents;
  char KeyFlag;
  int  PFDisStereo;

#ifdef VMDSPACEWARE
  SiHdl sball;             // Spaceware handle
  SiSpwEvent spwevent;     // Spaceware event
  SiGetEventData spwedata; // Spaceware event data
#endif

} wgldata;

#else

#define SBALL_COMMAND_NONE                0
#define SBALL_COMMAND_APP_WINDOW      27695
#define SBALL_COMMAND_APP_SENSITIVITY 27696

// Xlib ClientMessage-based Spaceball/Magellan handle data
typedef struct {
  Display *dpy;
  Window drv_win;
  Window app_win;
  Atom ev_motion;
  Atom ev_button_press;
  Atom ev_button_release;
  Atom ev_command;
} spaceballhandle;

// Spaceball event data
typedef struct {
  int event;
  int rx;
  int ry;
  int rz;
  int tx;
  int ty; 
  int tz;
  int buttons;
  int period;
} spaceballevent;

/// GLX-specific low-level handles and window IDs
typedef struct {
  Display *dpy;              ///< X server display connection
  int dpyScreen;             ///< X server screen number
  Window rootWindowID;       ///< ID of the root window

  Window windowID;           ///< ID of the graphics window
  XSizeHints sizeHints;      ///< size hints for opening graphics window
  Cursor cursor[5];          ///< graphics window cursor type
  GLXContext cx;             ///< GLX graphics context
  int havefocus;             ///< Flag indicating mouse/kbd focus

  void *xinp;                ///< XInput-based Spaceball, Dial box, etc.

  spaceballhandle *sball;    ///< Spaceball/Magellan/SpaceNavigator handle
  spaceballevent sballevent; ///< Most recent spaceball event status

} glxdata;
#endif
#endif


/// Subclass of OpenGLRenderer, this object has routines used by all the
/// different display devices that are OpenGL-specific.  Will render drawing
/// commands into a single X window.
class OpenGLDisplayDevice : public OpenGLRenderer {
public:
#if defined(VMDSDL)
  sdldata sdlsrv;
#else
#if defined(_MSC_VER)
  wgldata glwsrv;
#else
  glxdata glxsrv;
#endif
#endif

protected:
  // flag for whether a window was successfully created by open_window
  int have_window;

  // create a new window and set it's characteristics
#if defined(VMDSDL)
  int open_window(char *, int *, int *, int, char **);
#else
#if defined(_MSC_VER)
  int open_window(char *, int *, int *, int, char **);
#else
  Window open_window(char *, int *, int *, int, char **);
#endif
#endif

  virtual void do_resize_window(int, int);
  virtual void do_reposition_window(int, int);

public:
  // constructor - trivial variable initialization, no window opened yet.
  OpenGLDisplayDevice();

  // real initialization; return TRUE if the window was successfully opened
  // or FALSE if it wasn't.  Pass argc/argv from main, and size and location
  // for the window, if known.  size must NOT be NULL.
  int init(int argc, char **argv, VMDApp *app, int *size, int *loc = NULL);

  virtual ~OpenGLDisplayDevice(void);

  //
  // get the current state of the device's pointer (i.e. cursor if it has one)
  //

  virtual int x(void);           // abs pos of cursor from lower-left corner
  virtual int y(void);           // same, for y direction
  virtual int shift_state(void); // return the shift state (ORed of the
                                 // enum in DisplayDevice)

  // get the current state of the Spaceball if one is available
  // returns rx ry rz, tx ty tz, buttons
  virtual int spaceball(int *, int *, int *, int *, int *, int *, int *);

  // set the Nth cursor shape as the current one.
  virtual void set_cursor(int);

  //
  // event handling routines
  //

  // queue the standard events (need only be called once ... but this is
  // not done automatically by the window because it may not be necessary or
  // even wanted)
  virtual void queue_events(void);

  // read the next event ... returns an event type (one of the above ones),
  // and a value.  Returns success, and sets arguments.
  virtual int read_event(long &, long &);

  //
  // virtual routines for preparing to draw, drawing, and finishing drawing
  //

  // process draw cmd list but force-restore our GLX context first
  virtual int prepare3D(int do_clear = TRUE); ///< ready to draw 3D
  virtual void update(int do_update = TRUE);  // finish up after drawing
  virtual void reshape(void);                 // refresh device after change

  // virtual routine for capturing the screen to a packed RGB array
  virtual unsigned char * readpixels_rgb3u(int &x, int &y);
  virtual unsigned char * readpixels_rgba4u(int &x, int &y);

  /// virtual routine for drawing the screen from a packed RGBA array
  virtual int drawpixels_rgba4u(unsigned char *rgba, int &x, int &y);

};

#endif

