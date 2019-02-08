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
 *	$RCSfile: OpenGLDisplayDevice.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.210 $	$Date: 2019/01/17 21:21:00 $
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
#include <GL/gl.h>
#include <GL/glx.h>
#include <X11/Xlib.h>
#include <X11/cursorfont.h>
#include <X11/keysym.h>

#if defined(VMDXINERAMA)
#include <X11/extensions/Xinerama.h>
#endif

#include "OpenGLDisplayDevice.h"
#include "Inform.h"
#include "utilities.h"
#include "config.h"   // VMD version strings etc

#include "VMDApp.h"
#include "VideoStream.h"

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

// determine if all of the ARB multisample extension routines are available
#if defined(GL_ARB_multisample) && defined(GLX_SAMPLES_ARB) && defined(GLX_SAMPLE_BUFFERS_ARB)
#define USEARBMULTISAMPLE 1
#endif

// colors for cursors
static XColor cursorFG = { 0, 0xffff,      0,      0, 
                         DoRed | DoGreen | DoBlue, 0 };
static XColor cursorBG = { 0, 0xffff, 0xffff, 0xffff, 
                         DoRed | DoGreen | DoBlue, 0 };

////////////////////////// static helper functions.

#if defined(VMDXINPUT)

#if defined(VMDXINPUT)
#include <X11/extensions/XI.h>
#include <X11/extensions/XInput.h>

typedef struct {
  XDevice *dev;
  int motionevent;
  int motioneventclass;
  int buttonpressevent;
  int buttonpresseventclass;
  int buttonreleaseevent;
  int buttonreleaseeventclass;
  XEventClass evclasses[3];
} xidevhandle;

typedef struct {
  Display *dpy;
  Window win;
  xidevhandle *dev_spaceball;
  xidevhandle *dev_dialbox;
} xinputhandle;

#endif


static xidevhandle * xinput_open_device(xinputhandle *handle, XID devinfo) {
  xidevhandle *xdhandle = (xidevhandle *) malloc(sizeof(xidevhandle));
  memset(xdhandle, 0, sizeof(xidevhandle));
  xdhandle->dev = XOpenDevice(handle->dpy, devinfo);
  if (xdhandle->dev == NULL) {
    free(xdhandle);
    return NULL;
  }

  DeviceMotionNotify(xdhandle->dev, xdhandle->motionevent, xdhandle->motioneventclass); 
  DeviceButtonPress(xdhandle->dev, xdhandle->buttonpressevent, xdhandle->buttonpresseventclass); 
  DeviceButtonRelease(xdhandle->dev, xdhandle->buttonreleaseevent, xdhandle->buttonreleaseeventclass); 

  xdhandle->evclasses[0] = xdhandle->motioneventclass;
  xdhandle->evclasses[1] = xdhandle->buttonpresseventclass;
  xdhandle->evclasses[2] = xdhandle->buttonreleaseeventclass;

  XSelectExtensionEvent(handle->dpy, handle->win, xdhandle->evclasses, 3);

  return xdhandle;
}


static void xinput_close_device(xinputhandle *handle, xidevhandle *xdhandle) {
  if (handle == NULL || xdhandle == NULL)
    return;

  if (xdhandle->dev != NULL) {
    XCloseDevice(handle->dpy, xdhandle->dev);
  }
  free(xdhandle);
}


static int xinput_device_decode_event(xinputhandle *handle, xidevhandle *dev,
                                      XEvent *xev, spaceballevent *sballevent) {
  if (xev->type == dev->motionevent) {
    XDeviceMotionEvent *mptr = (XDeviceMotionEvent *) xev;

    // We assume that the axis mappings are in the order below,as this is
    // the axis ordering used by a few other applications as well.
    // We add the current control inputs to whatever we had previously,
    // so that we can process all queued events and not drop any inputs
    sballevent->tx     += mptr->axis_data[0]; // X translation
    sballevent->ty     += mptr->axis_data[1]; // Y translation
    sballevent->tz     += mptr->axis_data[2]; // Z translation
    sballevent->rx     += mptr->axis_data[3]; // A rotation
    sballevent->ry     += mptr->axis_data[4]; // B rotation
    sballevent->rz     += mptr->axis_data[5]; // C rotation
    sballevent->period += 50; // Period in milliseconds
    sballevent->event = 1;
    return 1;
  } else if (xev->type == dev->buttonpressevent) {
//    XDeviceButtonEvent *bptr = (XDeviceButtonEvent *) xev;;
//    sballevent->buttons |= (1 << xev->xclient.data.s[2]);
    sballevent->buttons |= 1;
    sballevent->event = 1;
    return 1;
  } else if (xev->type == dev->buttonreleaseevent) {
//    XDeviceButtonEvent *bptr = (XDeviceButtonEvent *) xev;;
//    sballevent->buttons &= ~(1 << xev->xclient.data.s[2]);
    sballevent->buttons &= ~1;
    sballevent->event = 1;
    return 1;
  }
 
  return 0;
}


static int xinput_decode_event(xinputhandle *handle, XEvent *xev,
                               spaceballevent *sballevent) {
  if (handle == NULL)
    return 0;

  if (handle->dev_spaceball != NULL) {
    return xinput_device_decode_event(handle, handle->dev_spaceball, xev, sballevent);
  }

  return 0;
}


// enable 6DOF input devices that use XInput 
static xinputhandle * xinput_enable(Display *dpy, Window win) {
  xinputhandle *handle = NULL;
  int i, numdev, numextdev;
  XDeviceInfoPtr list;
  int ximajor, xiev, xierr;
  Atom sballdevtype;
//  Atom dialboxdevtype; 
  xidevhandle *dev_spaceball = NULL;
//  xidevhandle *dev_dialbox = NULL;

  /* check for availability of the XInput extension */
  if(!XQueryExtension(dpy,"XInputExtension", &ximajor, &xiev, &xierr)) {
    msgInfo << "X-Windows XInput extension unavailable." << sendmsg; 
    return NULL;
  }

  sballdevtype = XInternAtom(dpy, XI_SPACEBALL, True);
//  dialboxdevtype = XInternAtom(dpy, XI_KNOB_BOX, True);

  /* Get the list of input devices attached to the display */
  list = (XDeviceInfoPtr) XListInputDevices(dpy, &numdev);
 
  numextdev = 0; 
  for (i = 0; i < numdev; i++) {
    if (list[i].use == IsXExtensionDevice) {
      // skip Xorg 'evdev brain' device
      if (!strupncmp(list[i].name, "evdev brain", strlen("evdev brain")))
        continue;

      numextdev++;
    }
  }
 
  if (numextdev > 0) {
    handle = (xinputhandle *) malloc(sizeof(xinputhandle));
    memset(handle, 0, sizeof(xinputhandle));
    handle->dpy = dpy;
    handle->win = win;

    msgInfo << "Detected " << numdev << " XInput devices, " 
            << numextdev << " usable device" 
            << ((numextdev > 1) ? "s:" : ":") << sendmsg;

    for (i = 0; i < numdev; i++) {
      if (list[i].use == IsXExtensionDevice) {
        // skip Xorg 'evdev brain' device
        if (!strupncmp(list[i].name, "evdev brain", strlen("evdev brain")))
          continue;

        /* list promising looking devices  */
        msgInfo << "  [" << list[i].id << "] " << list[i].name 
                << ", type: " << (int) list[i].type 
                << ", classes: " << (int) list[i].num_classes << sendmsg;

        /* Tag the first Spaceball device we find */
        if ((dev_spaceball == NULL) &&
            (((sballdevtype != None) && (list[i].type == sballdevtype)) ||
            !strupncmp(list[i].name, "SPACEBALL", strlen("SPACEBALL")) ||
            !strupncmp(list[i].name, "MAGELLAN", strlen("MAGELLAN")))) {
          dev_spaceball = xinput_open_device(handle, list[i].id);
        }
 
#if 0 
        /* Tag the first dial box device we find */
        if ((dev_dialbox == NULL) &&
            ((dialboxdevtype != None) && (list[i].type == dialboxdevtype))) {
          dev_dialbox = xinput_open_device(handle, list[i].id);
        }
#endif
      }
    }
    XFreeDeviceList(list);
  } else {
    // msgInfo << "No XInput devices found." << sendmsg; 
    XFreeDeviceList(list);
    return NULL;
  }

  if (dev_spaceball) {
    msgInfo << "Attached to XInput Spaceball" << sendmsg;
  }
//  if (dev_dialbox) {
//    msgInfo << "Attached to XInput Dial Box" << sendmsg;
//  }

  if (dev_spaceball != NULL /* || dev_dialbox != NULL */) {
    handle->dev_spaceball = dev_spaceball;
//    handle->dev_dialbox   = dev_dialbox;
  } else {
    free(handle);
    return NULL;
  }

  return handle;
}

void xinput_close(xinputhandle *handle) {
  if (handle != NULL) {
    xinput_close_device(handle, handle->dev_spaceball);
//    xinput_close_device(handle, handle->dev_dialbox);
    free(handle);
  }
}

#endif


// enable 3Dxware Spaceball / Magellan / SpaceNavigator events
static spaceballhandle * spaceball_enable(Display *dpy, Window win) {
  // allocate and clear handle data structure
  spaceballhandle *handle = (spaceballhandle *) malloc(sizeof(spaceballhandle));
  memset(handle, 0, sizeof(spaceballhandle));  

  // find and store X atoms for the event types we care about
  handle->ev_motion         = XInternAtom(dpy, "MotionEvent", True);
  handle->ev_button_press   = XInternAtom(dpy, "ButtonPressEvent", True);
  handle->ev_button_release = XInternAtom(dpy, "ButtonReleaseEvent", True);
  handle->ev_command        = XInternAtom(dpy, "CommandEvent", True);

  if (!handle->ev_motion || !handle->ev_button_press || 
      !handle->ev_button_release || !handle->ev_command) {
    free(handle);
    return NULL; /* driver is not running */
  }

  // Find the root window of the driver
  Window root = RootWindow(dpy, DefaultScreen(dpy)); 

  // Find the driver's window
  Atom ActualType;
  int ActualFormat;
  unsigned long NItems, BytesReturn;
  unsigned char *PropReturn = NULL;
  XGetWindowProperty(dpy, root, handle->ev_command, 0, 1, FALSE,
                     AnyPropertyType, &ActualType, &ActualFormat, &NItems,
                     &BytesReturn, &PropReturn );
  if (PropReturn == NULL) {
    free(handle);
    return NULL;
  }
  handle->drv_win = *(Window *) PropReturn;
  XFree(PropReturn);

  XTextProperty sball_drv_winname;
  if (XGetWMName(dpy, handle->drv_win, &sball_drv_winname) != 0) {
    if (!strcmp("Magellan Window", (char *) sball_drv_winname.value)) {
      /* Send the application window to the Spaceball/Magellan driver */
      XEvent msg;
      msg.type = ClientMessage;
      msg.xclient.format = 16;
      msg.xclient.send_event = FALSE;
      msg.xclient.display = dpy;
      msg.xclient.window = handle->drv_win;
      msg.xclient.message_type = handle->ev_command;

      msg.xclient.data.s[0] = (short) (((win)>>16)&0x0000FFFF); // High 16
      msg.xclient.data.s[1] = (short) (((win))    &0x0000FFFF); // Low 16
      msg.xclient.data.s[2] = SBALL_COMMAND_APP_WINDOW; // 27695

      int rc = XSendEvent(dpy, handle->drv_win, FALSE, 0x0000, &msg);
      XFlush(dpy);
      if (rc == 0) {
        free(handle); 
        return NULL;
      }
    }

    XFree(sball_drv_winname.value);
  } 

  return handle;
}


static void spaceball_close(spaceballhandle *handle) {
  free(handle);
}


static int spaceball_decode_event(spaceballhandle *handle, const XEvent *xev, spaceballevent *sballevent) {
  unsigned int evtype;

  if (handle == NULL || xev == NULL || sballevent == NULL)
    return 0;

  if (xev->type != ClientMessage)
    return 0;

  evtype = xev->xclient.message_type;

  if (evtype == handle->ev_motion) {
    // We add the current control inputs to whatever we had previously,
    // so that we can process all queued events and not drop any inputs
    // xev->xclient.data.s[0] is Device Window High 16-bits 
    // xev->xclient.data.s[1] is Device Window Low 16-bits 
    sballevent->tx     += xev->xclient.data.s[2]; // X translation
    sballevent->ty     += xev->xclient.data.s[3]; // Y translation
    sballevent->tz     += xev->xclient.data.s[4]; // Z translation
    sballevent->rx     += xev->xclient.data.s[5]; // A rotation
    sballevent->ry     += xev->xclient.data.s[6]; // B rotation
    sballevent->rz     += xev->xclient.data.s[7]; // C rotation
    sballevent->period += xev->xclient.data.s[8]; // Period in milliseconds
    sballevent->event = 1;
    return 1;
  } else if (evtype == handle->ev_button_press) {
    // xev->xclient.data.s[0] is Device Window High 16-bits 
    // xev->xclient.data.s[1] is Device Window Low 16-bits 
    sballevent->buttons |= (1 << xev->xclient.data.s[2]);
    sballevent->event = 1;
    return 1;
  } else if (evtype == handle->ev_button_release) {
    // xev->xclient.data.s[0] is Device Window High 16-bits 
    // xev->xclient.data.s[1] is Device Window Low 16-bits 
    sballevent->buttons &= ~(1 << xev->xclient.data.s[2]);
    sballevent->event = 1;
    return 1;
  }

  return 0;
}


static void spaceball_init_event(spaceballevent *sballevent) {
  memset(sballevent, 0, sizeof(spaceballevent));
}


static void spaceball_clear_event(spaceballevent *sballevent) {
  sballevent->tx = 0;
  sballevent->ty = 0;
  sballevent->tz = 0;
  sballevent->rx = 0;
  sballevent->ry = 0;
  sballevent->rz = 0;
  sballevent->period = 0;
  sballevent->event = 0;
}


static XVisualInfo * vmd_get_visual(glxdata *glxsrv, int *stereo, int *msamp, int *numsamples) {
  // we want double-buffered RGB with a Z buffer (possibly with stereo)
  XVisualInfo *vi;
  int ns, dsize;
  int simplegraphics = 0;
  int disablestereo = 0;
  vi = NULL;
  *numsamples = 0;
  *msamp = FALSE; 
  *stereo = FALSE;

  if (getenv("VMDSIMPLEGRAPHICS")) {
    simplegraphics = 1;
  }

  if (getenv("VMDDISABLESTEREO")) {
    disablestereo = 1;
  } 

  // check for user-override of maximum antialiasing sample count
  int maxaasamples=4;
  const char *maxaasamplestr = getenv("VMDMAXAASAMPLES");
  if (maxaasamplestr) {
    int aatmp;
    if (sscanf(maxaasamplestr, "%d", &aatmp) == 1) {
      if (aatmp >= 0) {
        maxaasamples=aatmp;
        msgInfo << "User-requested OpenGL antialiasing sample depth: " 
                << maxaasamples << sendmsg;

        if (maxaasamples < 2) {
          maxaasamples=1; 
          msgInfo << "OpenGL antialiasing disabled by user override."
                  << sendmsg;
        }
      } else {
        msgErr << "Ignoring user-requested OpenGL antialiasing sample depth: " 
               << aatmp << sendmsg;
      }
    } else {
      msgErr << "Unable to parse override of OpenGL antialiasing" << sendmsg;
      msgErr << "sample depth: '" << maxaasamplestr << "'" << sendmsg;
    }
  }


  // loop over a big range of depth buffer sizes, starting with biggest 
  // and working our way down from there.
  for (dsize=32; dsize >= 16; dsize-=4) { 

// Try the OpenGL ARB multisample extension if available
#if defined(USEARBMULTISAMPLE) 
    if (!simplegraphics && !disablestereo && (!vi || (vi->c_class != TrueColor))) {
      // Stereo, multisample antialising, stencil buffer
      for (ns=maxaasamples; ns>1; ns--) {
        int conf[]  = {GLX_DOUBLEBUFFER, GLX_RGBA, GLX_DEPTH_SIZE, dsize, 
                       GLX_STEREO,
                       GLX_STENCIL_SIZE, 1, 
                       GLX_SAMPLE_BUFFERS_ARB, 1, GLX_SAMPLES_ARB, ns, None};
        vi = glXChooseVisual(glxsrv->dpy, glxsrv->dpyScreen, conf);
  
        if (vi && (vi->c_class == TrueColor)) {
          *numsamples = ns;
          *msamp = TRUE;
          *stereo = TRUE;
          break; // exit loop if we got a good visual
        } 
      }
    }
#endif

    if (getenv("VMDPREFERSTEREO") != NULL && !disablestereo) {
      // The preferred 24-bit color, quad buffered stereo mode.
      // This hack allows NVidia Quadro users to avoid the mutually-exclusive
      // antialiasing/stereo options on their cards with current drivers.
      // This forces VMD to skip looking for multisample antialiasing capable
      // X visuals and look for stereo instead.
      if (!simplegraphics && (!vi || (vi->c_class != TrueColor))) {
        int conf[] = {GLX_DOUBLEBUFFER, GLX_RGBA, GLX_DEPTH_SIZE, dsize, 
                      GLX_STEREO,
                      GLX_STENCIL_SIZE, 1, 
                      GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, None};
        vi = glXChooseVisual(glxsrv->dpy, glxsrv->dpyScreen, conf);
        ns = 0; // no multisample antialiasing
        *numsamples = ns;
        *msamp = FALSE; 
        *stereo = TRUE; 
      }
    } 
#if defined(USEARBMULTISAMPLE) 
    else {
      // Try the OpenGL ARB multisample extension if available
      if (!simplegraphics && (!vi || (vi->c_class != TrueColor))) {
        // Non-Stereo, multisample antialising, stencil buffer
        for (ns=maxaasamples; ns>1; ns--) {
          int conf[]  = {GLX_DOUBLEBUFFER, GLX_RGBA, GLX_DEPTH_SIZE, dsize, 
                         GLX_STENCIL_SIZE, 1, 
                         GLX_SAMPLE_BUFFERS_ARB, 1, GLX_SAMPLES_ARB, ns, None};
          vi = glXChooseVisual(glxsrv->dpy, glxsrv->dpyScreen, conf);
    
          if (vi && (vi->c_class == TrueColor)) {
            *numsamples = ns;
            *msamp = TRUE;
            *stereo = FALSE; 
            break; // exit loop if we got a good visual
          } 
        }
      }
    }
#endif

  } // end of loop over a wide range of depth buffer sizes

  // Ideally we should fall back to accumulation buffer based antialiasing
  // here, but not currently implemented.  At this point no multisample
  // antialiasing mode is available.

  // The preferred 24-bit color, quad buffered stereo mode
  if (!simplegraphics && !disablestereo && (!vi || (vi->c_class != TrueColor))) {
    int conf[] = {GLX_DOUBLEBUFFER, GLX_RGBA, GLX_DEPTH_SIZE, 16, GLX_STEREO,
                  GLX_STENCIL_SIZE, 1, 
                  GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, None};
    vi = glXChooseVisual(glxsrv->dpy, glxsrv->dpyScreen, conf);
    ns = 0; // no multisample antialiasing
    *numsamples = ns;
    *msamp = FALSE; 
    *stereo = TRUE; 
  }

  // Mode for machines that provide stereo only in modes with 16-bit color.
  if (!simplegraphics && !disablestereo && (!vi || (vi->c_class != TrueColor))) {
    int conf[] = {GLX_DOUBLEBUFFER, GLX_RGBA, GLX_DEPTH_SIZE, 16, GLX_STEREO,
                  GLX_STENCIL_SIZE, 1, 
                  GLX_RED_SIZE, 1, GLX_GREEN_SIZE, 1, GLX_BLUE_SIZE, 1, None};
    vi = glXChooseVisual(glxsrv->dpy, glxsrv->dpyScreen, conf);
    ns = 0; // no multisample antialiasing
    *numsamples = ns;
    *msamp = FALSE; 
    *stereo = TRUE; 
  }

  // Mode for machines that provide stereo only without a stencil buffer, 
  // and with reduced color precision.  Examples of this are the SGI Octane2
  // machines with V6 graphics, with recent IRIX patch levels.
  // Without this configuration attempt, these machines won't get stereo.
  if (!simplegraphics && !disablestereo && (!vi || (vi->c_class != TrueColor))) {
    int conf[] = {GLX_DOUBLEBUFFER, GLX_RGBA, GLX_DEPTH_SIZE, 16, GLX_STEREO,
                  GLX_RED_SIZE, 1, GLX_GREEN_SIZE, 1, GLX_BLUE_SIZE, 1, None};
    vi = glXChooseVisual(glxsrv->dpy, glxsrv->dpyScreen, conf);
    ns = 0; // no multisample antialiasing
    *numsamples = ns;
    *msamp = FALSE; 
    *stereo = TRUE; 
  }

  // This mode gives up on trying to get stereo, and goes back to trying
  // to get a high quality non-stereo visual.
  if (!simplegraphics && (!vi || (vi->c_class != TrueColor))) {
    int conf[] = {GLX_DOUBLEBUFFER, GLX_RGBA, GLX_DEPTH_SIZE, 16, 
                  GLX_STENCIL_SIZE, 1, 
                  GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, None};
    vi = glXChooseVisual(glxsrv->dpy, glxsrv->dpyScreen, conf);
    ns = 0; // no multisample antialiasing
    *numsamples = ns;
    *msamp = FALSE; 
    *stereo = FALSE;
  }
  
  // check if we have a TrueColor visual.
  if(!vi || (vi->c_class != TrueColor)) {
    // still no TrueColor.  Try again, with a very basic request ...
    // This is a catch all, we're desperate for any truecolor
    // visual by this point.  We've given up hoping for 24-bit
    // color or stereo by this time.
    int conf[] = {GLX_DOUBLEBUFFER, GLX_RGBA, GLX_DEPTH_SIZE, 16, 
                  GLX_RED_SIZE, 1, GLX_GREEN_SIZE, 1, GLX_BLUE_SIZE, 1, None};
    vi = glXChooseVisual(glxsrv->dpy, glxsrv->dpyScreen, conf);
    ns = 0; // no multisample antialiasing
    *numsamples = ns;
    *msamp = FALSE; 
    *stereo = FALSE;
  }

  if (!vi || (vi->c_class != TrueColor)) {
    // complete failure
    ns = 0; // no multisample antialiasing
    *numsamples = ns;
    *msamp = FALSE; 
    *stereo = FALSE;
  }

  return vi;
}


// make an X11 window full-screen, or return it to normal state
static void setfullscreen(int fson, Display *dpy, Window win, int xinescreen) {
  struct {
    unsigned long flags;
    unsigned long functions;
    unsigned long decorations;
    long inputMode;
    unsigned long status;
  } wmhints;
  
  memset(&wmhints, 0, sizeof(wmhints));
  wmhints.flags = 2;       // changing window decorations
  if (fson) {
    wmhints.decorations = 0; // 0 (false) no window decorations
  } else {
    wmhints.decorations = 1; // 1 (true) window decorations enabled
  }

#if !defined(VMD_NANOHUB)
  Atom wmproperty = XInternAtom(dpy, "_MOTIF_WM_HINTS", True);
#else
  // Intern the atom even if it doesn't exist (no wm).
  Atom wmproperty = XInternAtom(dpy, "_MOTIF_WM_HINTS", False);
#endif
  XChangeProperty(dpy, win, wmproperty, wmproperty, 32, 
                  PropModeReplace, (unsigned char *) &wmhints, 5);

  // resize window to size of either the whole X display screen,
  // or to the size of one of the Xinerama component displays
  // if Xinerama is enabled, and xinescreen is not -1.
  if (fson) {
    int dpyScreen = DefaultScreen(dpy);
  
    XSizeHints sizeHints;
    memset((void *) &(sizeHints), 0, sizeof(sizeHints));
    sizeHints.flags |= USSize;
    sizeHints.flags |= USPosition;

    sizeHints.width = DisplayWidth(dpy, dpyScreen);
    sizeHints.height = DisplayHeight(dpy, dpyScreen);
    sizeHints.x = 0;
    sizeHints.y = 0;

#if defined(VMDXINERAMA)
    if (xinescreen != -1) {
      int xinerr, xinevent, xinenumscreens;
      if (XineramaQueryExtension(dpy, &xinevent, &xinerr) &&
          XineramaIsActive(dpy)) {
        XineramaScreenInfo *screens = 
          XineramaQueryScreens(dpy, &xinenumscreens);
        if (xinescreen >= 0 && xinescreen < xinenumscreens) {
          sizeHints.width = screens[xinescreen].width;
          sizeHints.height = screens[xinescreen].height;
          sizeHints.x = screens[xinescreen].x_org;
          sizeHints.y = screens[xinescreen].y_org;
        }
        XFree(screens);
      }
    }
#endif
 
    XMoveWindow(dpy, win, sizeHints.x, sizeHints.y);
    XResizeWindow(dpy, win, sizeHints.width, sizeHints.height);
  }
}


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

  memset(&glxsrv, 0, sizeof(glxsrv));
  glxsrv.dpy = NULL;
  glxsrv.dpyScreen = 0;
  glxsrv.xinp  = NULL;
  glxsrv.sball = NULL;
  glxsrv.havefocus = 0;
  have_window = FALSE;
  screenX = screenY = 0;
}

int OpenGLDisplayDevice::init(int argc, char **argv, VMDApp *app, int *size, int *loc) {
  vmdapp = app; // save VMDApp handle for use by drag-and-drop handlers
                // and GPU memory management routines

  // open the window
  glxsrv.windowID = open_window(name, size, loc, argc, argv);
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

  // We have a window, return success.
  return TRUE;
}

// destructor ... close the window
OpenGLDisplayDevice::~OpenGLDisplayDevice(void) {
  if (have_window) {
#if defined(VMDXINPUT)
    // detach from XInput devices
    if (glxsrv.xinp != NULL) {
      xinput_close((xinputhandle *) glxsrv.xinp); 
    }
#endif
    
    // detach from Xlib ClientMessage-based spaceball
    if (glxsrv.sball != NULL) {
      spaceball_close(glxsrv.sball); 
    }

    free_opengl_ctx(); // free display lists, textures, etc
 
    // close and delete windows, contexts, and display connections
    XUnmapWindow(glxsrv.dpy, glxsrv.windowID);
    glXDestroyContext(glxsrv.dpy, glxsrv.cx);
    XDestroyWindow(glxsrv.dpy, glxsrv.windowID);
    XCloseDisplay(glxsrv.dpy);
  }
}


/////////////////////////  protected nonvirtual routines  


// create a new window and set it's characteristics
Window OpenGLDisplayDevice::open_window(char *nm, int *size, int *loc,
					int argc, char** argv
) {
  Window win;
  int i, SX = 100, SY = 100, W, H;
 
  char *dispname;
  if ((dispname = getenv("VMDGDISPLAY")) == NULL)
    dispname = getenv("DISPLAY");

  if(!(glxsrv.dpy = XOpenDisplay(dispname))) {
    msgErr << "Exiting due to X-Windows OpenGL window creation failure." << sendmsg;
    if (dispname != NULL) {
      msgErr << "Failed to open display: " << dispname << sendmsg;
    }
    return (Window)0; 
  }


  //
  // Check for "Composite" extension and any others that might cause
  // stability issues and warn the user about any potential problems...
  //
  char **xextensionlist;
  int nextensions, xtn;
  int warncompositeext=0;
  xextensionlist = XListExtensions(glxsrv.dpy, &nextensions);
  for (xtn=0; xtn<nextensions; xtn++) {
//    printf("xtn[%d]: '%s'\n", xtn, xextensionlist[xtn]);
    if (xextensionlist[xtn] && !strcmp(xextensionlist[xtn], "Composite")) {
      warncompositeext=1;
    }
  }
  if (warncompositeext) {
    msgWarn << "Detected X11 'Composite' extension: if incorrect display occurs" << sendmsg;
    msgWarn << "try disabling this X server option.  Most OpenGL drivers" << sendmsg;
    msgWarn << "disable stereoscopic display when 'Composite' is enabled." << sendmsg;
  }
  XFreeExtensionList(xextensionlist);


  //
  // get info about root window
  //
  glxsrv.dpyScreen = DefaultScreen(glxsrv.dpy);
  glxsrv.rootWindowID = RootWindow(glxsrv.dpy, glxsrv.dpyScreen);
  screenX = DisplayWidth(glxsrv.dpy, glxsrv.dpyScreen);
  screenY = DisplayHeight(glxsrv.dpy, glxsrv.dpyScreen);
  W = size[0];
  H = size[1];
  if (loc) {
    SX = loc[0];
    // The X11 screen uses Y increasing from upper-left corner down; this is
    // opposite to what GL does, which is the way VMD was set up originally
    SY = (screenY - loc[1]) - H;
  }

  // (3) make sure the GLX extension is available
  if (!glXQueryExtension(glxsrv.dpy, NULL, NULL)) {
    msgErr << "The X server does not support the OpenGL GLX extension." 
           << "   Exiting ..." << sendmsg;
    XCloseDisplay(glxsrv.dpy);
    return (Window)0;
  }

  ext->hasstereo = TRUE;         // stereo on until we find out otherwise.
  ext->stereodrawforced = FALSE; // no need for force stereo draws initially
  ext->hasmultisample = TRUE;    // multisample on until we find out otherwise.

  // (4) find an appropriate X-Windows GLX-capable visual and colormap ...
  XVisualInfo *vi;
  vi =  vmd_get_visual(&glxsrv, &ext->hasstereo, &ext->hasmultisample, &ext->nummultisamples);

  // make sure we have what we want, darnit ...
  if (!vi) {
    msgErr << "A TrueColor visual is required, but not available." << sendmsg;
    msgErr << "The X server is not capable of displaying double-buffered," << sendmsg;
    msgErr << "RGB images with a Z buffer.   Exiting ..." << sendmsg;
    XCloseDisplay(glxsrv.dpy);
    return (Window)0;
  }

  // (5) create an OpenGL rendering context
  if(!(glxsrv.cx = glXCreateContext(glxsrv.dpy, vi, None, GL_TRUE))) {
    msgErr << "Could not create OpenGL rendering context-> Exiting..." 
           << sendmsg;
    return (Window)0;
  }

  // (6) setup cursors, icons, iconized mode title, etc.
  glxsrv.cursor[0] = XCreateFontCursor(glxsrv.dpy, XC_left_ptr);
  glxsrv.cursor[1] = XCreateFontCursor(glxsrv.dpy, XC_fleur);
  glxsrv.cursor[2] = XCreateFontCursor(glxsrv.dpy, XC_sb_h_double_arrow);
  glxsrv.cursor[3] = XCreateFontCursor(glxsrv.dpy, XC_crosshair);
  glxsrv.cursor[4] = XCreateFontCursor(glxsrv.dpy, XC_watch);
  for(i=0; i < 5; i++)
    XRecolorCursor(glxsrv.dpy, glxsrv.cursor[i], &cursorFG, &cursorBG);


  //
  // Create the window
  //
  XSetWindowAttributes swa;

  //   For StaticGray , StaticColor, and TrueColor,
  //   alloc must be AllocNone , or a BadMatch error results
  swa.colormap = XCreateColormap(glxsrv.dpy, glxsrv.rootWindowID, 
                                 vi->visual, AllocNone);

  swa.background_pixmap = None;
  swa.border_pixel=0;
  swa.event_mask = ExposureMask;
  swa.cursor = glxsrv.cursor[0];

  win = XCreateWindow(glxsrv.dpy, glxsrv.rootWindowID, SX, SY, W, H, 0,
                      vi->depth, InputOutput, vi->visual,
                      CWBorderPixel | CWColormap | CWEventMask, &swa);
  XInstallColormap(glxsrv.dpy, swa.colormap);

  XFree(vi); // free visual info

  //
  // create size hints for new window
  //
  memset((void *) &(glxsrv.sizeHints), 0, sizeof(glxsrv.sizeHints));
  glxsrv.sizeHints.flags |= USSize;
  glxsrv.sizeHints.flags |= USPosition;
  glxsrv.sizeHints.width = W;
  glxsrv.sizeHints.height = H;
  glxsrv.sizeHints.x = SX;
  glxsrv.sizeHints.y = SY;

  XSetStandardProperties(glxsrv.dpy, win, nm, "VMD", None, argv, argc, &glxsrv.sizeHints);
  XWMHints *wmHints = XAllocWMHints();
  wmHints->initial_state = NormalState;
  wmHints->flags = StateHint;
  XSetWMHints(glxsrv.dpy, win, wmHints);
  XFree(wmHints);

  // Cause X11 to generate a ClientMessage event for WM window closure
#if !defined(VMD_NANOHUB)
  Atom wmDeleteWindow = XInternAtom(glxsrv.dpy, "WM_DELETE_WINDOW", False);
#else
  Atom wmDeleteWindow = XInternAtom(glxsrv.dpy, "WM_DELETE_WINDOW", True);
#endif
  XSetWMProtocols(glxsrv.dpy, win, &wmDeleteWindow, 1);

  // (7) bind the rendering context to the window
  glXMakeCurrent(glxsrv.dpy, win, glxsrv.cx);


  // (8) actually request the window to be displayed
  XSelectInput(glxsrv.dpy, win, 
               KeyPressMask | ButtonPressMask | ButtonReleaseMask | 
               StructureNotifyMask | ExposureMask | 
               EnterWindowMask | LeaveWindowMask | FocusChangeMask);
  XMapRaised(glxsrv.dpy, win);

  // If we have acquired a multisample buffer with GLX, we
  // still need to test to see if we can actually use it.
  if (ext->hasmultisample) {
    int msampeext = 0;

    // check for ARB multisampling
    if (ext->vmdQueryExtension("GL_ARB_multisample")) {
      msampeext = 1;
    }

    if (!msampeext) {
      ext->hasmultisample = FALSE;
      ext->nummultisamples = 0;
    }
  }

  // (9) configure the rendering properly
  setup_initial_opengl_state();  // setup initial OpenGL state

#if defined(VMDXINPUT)
  // (10) check for XInput based 6DOF controllers etc
  if (getenv("VMDDISABLEXINPUT") == NULL) {
    glxsrv.xinp = xinput_enable(glxsrv.dpy, win);
  }
#endif

  // (11) Enable receiving Xlib ClientMessage-based Spaceball 
  //      events to this window
  if (getenv("VMDDISABLESPACEBALLXDRV") == NULL) {
    if (getenv("VMDSPACEBALLXDRVGLOBALFOCUS") == NULL) {
      // the driver will do focus processing for us
      glxsrv.sball = spaceball_enable(glxsrv.dpy, InputFocus);
    } else {
      // we'll do focus processing for ourselves
      glxsrv.sball = spaceball_enable(glxsrv.dpy, win);
    }
  }
  if (glxsrv.sball != NULL) {
    msgInfo << "X-Windows ClientMessage-based Spaceball device available." 
            << sendmsg;
  } 


  // initialize spaceball event structure to known state
  spaceball_init_event(&glxsrv.sballevent);

  // normal return: window was successfully created
  have_window = TRUE;

  // return window id
  return win;
}


int OpenGLDisplayDevice::prepare3D(int do_clear) {
  // force reset of OpenGL context back to ours in case something
  // else modified the OpenGL state
  glXMakeCurrent(glxsrv.dpy, glxsrv.windowID, glxsrv.cx);

  return OpenGLRenderer::prepare3D(do_clear);
}


void OpenGLDisplayDevice::do_resize_window(int w, int h) {
  if (getenv("VMDFULLSCREEN")) {
    int xinescreen=0;
    if (getenv("VMDXINESCREEN")) {
      xinescreen = atoi(getenv("VMDXINESCREEN"));
    }
    setfullscreen(1, glxsrv.dpy, glxsrv.windowID, xinescreen);
  } else {
    setfullscreen(0, glxsrv.dpy, glxsrv.windowID, -1);
    XResizeWindow(glxsrv.dpy, glxsrv.windowID, w, h);
  }
}

void OpenGLDisplayDevice::do_reposition_window(int xpos, int ypos) {
  XMoveWindow(glxsrv.dpy, glxsrv.windowID, xpos, ypos);
}

/////////////////////////  public virtual routines  

//
// get the current state of the device's pointer (i.e. cursor if it has one)
//

// abs pos of cursor from lower-left corner of display
int OpenGLDisplayDevice::x(void) {
  Window rw, cw;
  int rx, ry, wx, wy;
  unsigned int keymask;

  // get pointer info
  XQueryPointer(glxsrv.dpy, glxsrv.windowID, &rw, &cw, &rx, &ry, &wx, &wy, &keymask);

  // return value
  return rx;
}


// same, for y direction
int OpenGLDisplayDevice::y(void) {
  Window rw, cw;
  int rx, ry, wx, wy;
  unsigned int keymask;

  // get pointer info
  XQueryPointer(glxsrv.dpy, glxsrv.windowID, &rw, &cw, &rx, &ry, &wx, &wy, &keymask);

  // return value
  // return value ... must subtract position from total size since
  // X is opposite to GL in sizing the screen
  return screenY - ry;
}

// return the current state of the shift, control, and alt keys
int OpenGLDisplayDevice::shift_state(void) {
  int retval = 0;

  // get pointer info
  Window rw, cw;
  int rx, ry, wx, wy;
  unsigned int keymask;
  XQueryPointer(glxsrv.dpy, glxsrv.windowID, &rw, &cw, &rx, &ry, &wx, &wy, &keymask);

  // determine state of keys, and OR results together
  if ((keymask & ShiftMask) != 0)
    retval |= SHIFT;

  if ((keymask & ControlMask) != 0)
    retval |= CONTROL;

  if ((keymask & Mod1Mask) != 0)
    retval |= ALT;

  // return the result
  return retval;
}


// return the spaceball state, if any
int OpenGLDisplayDevice::spaceball(int *rx, int *ry, int *rz, int *tx, int *ty,
int *tz, int *buttons) {
  // return event state we have from X11 windowing system events
  if ((glxsrv.sball != NULL || glxsrv.xinp != NULL)
       && glxsrv.sballevent.event == 1) {
    *rx = glxsrv.sballevent.rx;
    *ry = glxsrv.sballevent.ry;
    *rz = glxsrv.sballevent.rz;
    *tx = glxsrv.sballevent.tx;
    *ty = glxsrv.sballevent.ty;
    *tz = glxsrv.sballevent.tz;
    *buttons = glxsrv.sballevent.buttons;
    return 1;
  }

  return 0;
}


// set the Nth cursor shape as the current one.  If no arg given, the
// default shape (n=0) is used.
void OpenGLDisplayDevice::set_cursor(int n) {
  int cursorindex;

  switch (n) {
    default:
    case DisplayDevice::NORMAL_CURSOR: cursorindex = 0; break;
    case DisplayDevice::TRANS_CURSOR:  cursorindex = 1; break;
    case DisplayDevice::SCALE_CURSOR:  cursorindex = 2; break;
    case DisplayDevice::PICK_CURSOR:   cursorindex = 3; break;
    case DisplayDevice::WAIT_CURSOR:   cursorindex = 4; break;
  }

  XDefineCursor(glxsrv.dpy, glxsrv.windowID, glxsrv.cursor[cursorindex]);
}


//
// event handling routines
//

// queue the standard events (need only be called once ... but this is
// not done automatically by the window because it may not be necessary or
// even wanted)
void OpenGLDisplayDevice::queue_events(void) {
  XSelectInput(glxsrv.dpy, glxsrv.windowID, 
               KeyPressMask | ButtonPressMask | ButtonReleaseMask | 
               StructureNotifyMask | ExposureMask | 
               EnterWindowMask | LeaveWindowMask | FocusChangeMask);
}


// This version of read_event flushes the entire queue before returning the
// last event to the caller.  It fixes buggy window resizing behavior on 
// Linux when using the Nvidia OpenGL drivers.  
int OpenGLDisplayDevice::read_event(long &retdev, long &retval) {
  XEvent xev;
  char keybuf[10];
  int keybuflen = 9;
  KeySym keysym;
  XComposeStatus comp;

  memset(keybuf, 0, sizeof(keybuf)); // clear keyboard input buffer

  // clear previous spaceball event state, except for button state which
  // must be left alone.
  spaceball_clear_event(&glxsrv.sballevent);

  retdev = WIN_NOEVENT;
  // read all events, handling the ones that need to be handled internally,
  // and returning the last one for processing.
  int need_reshape = FALSE;
  while (XPending(glxsrv.dpy)) {
    XNextEvent(glxsrv.dpy, &xev);

    // find what kind of event it was
    switch(xev.type) {
    case Expose:
    case ConfigureNotify:
    case ReparentNotify:
    case MapNotify:
      need_reshape = TRUE; // Probably not needed for Expose or Map
      _needRedraw = 1;
      // retdev not set; we handle this ourselves.
      break;
    case KeyPress:
      {
        int k = XLookupString(&(xev.xkey), keybuf, keybuflen,  &keysym, &comp);
        // handle all strictly alphanumeric keys here
        if (k > 0 && *keybuf != '\0') {
          retdev = WIN_KBD;
          retval = *keybuf;
        } else {
          switch (keysym) {
            case XK_Escape:      retdev = WIN_KBD_ESCAPE;    break;
            case XK_Up:          retdev = WIN_KBD_UP;        break;
            case XK_Down:        retdev = WIN_KBD_DOWN;      break;
            case XK_Left:        retdev = WIN_KBD_LEFT;      break;
            case XK_Right:       retdev = WIN_KBD_RIGHT;     break;
            case XK_Page_Up:     retdev = WIN_KBD_PAGE_UP;   break;
            case XK_Page_Down:   retdev = WIN_KBD_PAGE_UP;   break;
            case XK_Home:        retdev = WIN_KBD_HOME;      break;
            case XK_End:         retdev = WIN_KBD_END;       break;
            case XK_Insert:      retdev = WIN_KBD_INSERT;    break;
            case XK_Delete:      retdev = WIN_KBD_DELETE;    break;
            case XK_F1:          retdev = WIN_KBD_F1;        break;
            case XK_F2:          retdev = WIN_KBD_F2;        break;
            case XK_F3:          retdev = WIN_KBD_F3;        break;
            case XK_F4:          retdev = WIN_KBD_F4;        break;
            case XK_F5:          retdev = WIN_KBD_F5;        break;
            case XK_F6:          retdev = WIN_KBD_F6;        break;
            case XK_F7:          retdev = WIN_KBD_F7;        break;
            case XK_F8:          retdev = WIN_KBD_F8;        break;
            case XK_F9:          retdev = WIN_KBD_F9;        break;
            case XK_F10:         retdev = WIN_KBD_F10;       break;
            case XK_F11:         retdev = WIN_KBD_F11;       break;
            case XK_F12:         retdev = WIN_KBD_F12;       break;
          } 
        } 
        break;
      }
    case ButtonPress:
    case ButtonRelease:
      {
        unsigned int button = xev.xbutton.button;
        retval = (xev.type == ButtonPress);
        switch (button) {
          case Button1:
            retdev = WIN_LEFT;
            break;
          case Button2:
            retdev = WIN_MIDDLE;
            break;
          case Button3:
            retdev = WIN_RIGHT;
            break;
          case Button4:
            retdev = WIN_WHEELUP;
            break;
          case Button5:
            retdev = WIN_WHEELDOWN;
            break;
        }
        break;
      }
      break;

    case FocusIn:
    case EnterNotify:
      glxsrv.havefocus=1;
      break;

    case FocusOut:
    case LeaveNotify:
      glxsrv.havefocus=0;
      break;

    case ClientMessage:
#if 1
      // let the spaceball driver take care of focus processing
      // if we have mouse/keyboard focus, then translate spaceball events
      spaceball_decode_event(glxsrv.sball, &xev, &glxsrv.sballevent);
#else
      // do our own focus handling
      // if we have mouse/keyboard focus, then translate spaceball events
      if (glxsrv.havefocus) {
        spaceball_decode_event(glxsrv.sball, &xev, &glxsrv.sballevent);
      }
#endif
      break;

    default:
#if defined(VMDXINPUT)
      if (glxsrv.xinp != NULL) {
        if (xinput_decode_event((xinputhandle *) glxsrv.xinp, &xev, 
                                 &glxsrv.sballevent)) {
          break;
        }
      }
#endif

#if 0
      msgWarn << "Unrecognized X11 event" << xev.type << sendmsg;
#endif      
      break;

    } 
  } 

  if (need_reshape) 
    reshape();

  return (retdev != WIN_NOEVENT);
}

//
// virtual routines for preparing to draw, drawing, and finishing drawing
//

// reshape the display after a shape change
void OpenGLDisplayDevice::reshape(void) {

  // get and store size of window
  XWindowAttributes xwa;
  Window childwin;                  // not used, just needed for X call
  int rx, ry;

  // 
  // XXX WireGL notes: 
  //   WireGL doesn't have a variable window size like normal 
  // OpenGL windows do.  Not only that, but the size values reported
  // by X11 will be widly different from those reported by 
  // the glGetIntegerv(GL_VIEWPORT) call, and cause schizophrenic
  // behavior.  For now, we don't do anything about this, but 
  // the default window that comes up on the tiled display is not
  // locked to the same size and aspect ratio as the host display,
  // so spheres can look rather egg shaped if the X window on the 
  // host display isn't adjusted. 
  //

  XGetWindowAttributes(glxsrv.dpy, glxsrv.windowID, &xwa);
  XTranslateCoordinates(glxsrv.dpy, glxsrv.windowID, glxsrv.rootWindowID, -xwa.border_width,
			-xwa.border_width, &rx, &ry, &childwin);

  xSize = xwa.width;
  ySize = xwa.height;
  xOrig = rx;
  yOrig = screenY - ry - ySize;
  
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
#if defined(VMD_NANOHUB)
  init_offscreen_framebuffer(xSize, ySize);
#endif
}


unsigned char * OpenGLDisplayDevice::readpixels_rgb3u(int &xs, int &ys) {
  unsigned char * img = NULL;
  xs = xSize;
  ys = ySize;

  // fall back to normal glReadPixels() if better methods fail
  if ((img = (unsigned char *) malloc(xs * ys * 3)) != NULL) {
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, xs, ys, GL_RGB, GL_UNSIGNED_BYTE, img);
    return img; 
  }

  // else bail out
  xs = 0;
  ys = 0;
  return NULL;
}

unsigned char * OpenGLDisplayDevice::readpixels_rgba4u(int &xs, int &ys) {
  unsigned char * img = NULL;
  xs = xSize;
  ys = ySize;

  // fall back to normal glReadPixels() if better methods fail
  if ((img = (unsigned char *) malloc(xs * ys * 4)) != NULL) {
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, xs, ys, GL_RGBA, GL_UNSIGNED_BYTE, img);
    return img; 
  }

  // else bail out
  xs = 0;
  ys = 0;
  return NULL;
}


int OpenGLDisplayDevice::drawpixels_rgba4u(unsigned char *rgba, int &xs, int &ys) {

#if 0
//  glDrawBuffer(GL_BACK);
//  glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
//  glClearColor(0.0, 0.0, 0.0, 1.0); /* black */
//  glClear(GL_COLOR_BUFFER_BIT);

  glPushMatrix();
  glDisable(GL_DEPTH_TEST);

  glViewport(0, 0, xs, ys);

  glShadeModel(GL_FLAT);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.0, xs, 0.0, ys, -1.0, 1.0);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glPixelZoom(1.0, 1.0);

  glRasterPos2i(0, 0);
  glDrawPixels(xs, ys, GL_RGBA, GL_UNSIGNED_BYTE, rgba);

  glEnable(GL_DEPTH_TEST);
  glPopMatrix();
#elif 1
//  glDrawBuffer(GL_BACK);
//  glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
//  glClearColor(0.0, 0.0, 0.0, 1.0); /* black */
//  glClear(GL_COLOR_BUFFER_BIT);

  glPushMatrix();
  glDisable(GL_DEPTH_TEST);

  glViewport(0, 0, xs, ys);

  glShadeModel(GL_FLAT);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.0, xs, 0.0, ys, -1.0, 1.0);
  glMatrixMode(GL_MODELVIEW);

  GLuint texName = 0;
  GLfloat texborder[4] = {0.0, 0.0, 0.0, 1.0};
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glBindTexture(GL_TEXTURE_2D, texName);

  /* black borders if we go rendering anything beyond texture coordinates */
  glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, texborder);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

  glLoadIdentity();
  glColor3f(1.0, 1.0, 1.0);

#if 1
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, xs, ys, 0,
               GL_RGBA, GL_UNSIGNED_BYTE, rgba);
  glEnable(GL_TEXTURE_2D);
#endif

  glBegin(GL_QUADS);
  glTexCoord2f(0.0f, 0.0f);
  glVertex2f(0, 0);
  glTexCoord2f(0.0f, 1.0f);
  glVertex2f(0, ys);
  glTexCoord2f(1.0f, 1.0f);
  glVertex2f(xs, ys);
  glTexCoord2f(1.0f, 0.0f);
  glVertex2f(xs, 0);
  glEnd();

#if 1
  glDisable(GL_TEXTURE_2D);
#endif

  glEnable(GL_DEPTH_TEST);
  glPopMatrix();
#endif

  update();

  return 0;
}


// update after drawing
void OpenGLDisplayDevice::update(int do_update) {
  if (wiregl) {
    glFinish(); // force cluster to synchronize before buffer swap, 
                // this gives much better results than if the 
                // synchronization is done implicitly by glXSwapBuffers.
  }

#if 1
  // push latest frame into the video streaming pipeline
  // and pump the event handling mechanism afterwards
  if (vmdapp->uivs && vmdapp->uivs->srv_connected()) {
    // if no frame was provided, we grab the GL framebuffer
    int xs, ys;
    unsigned char *img = NULL;
    img = readpixels_rgba4u(xs, ys);
    if (img != NULL) {
      // srv_send_frame(img, xs * 4, xs, ys, vs_forceIframe);
      vmdapp->uivs->video_frame_pending(img, xs, ys);
      vmdapp->uivs->check_event();
      free(img);
    }
  }
#endif

#if !defined(VMD_NANOHUB)
  // Normal contexts are double-buffered, but Nanohub uses a FBO that is not.
  if (do_update)
    glXSwapBuffers(glxsrv.dpy, glxsrv.windowID);
#endif

  glDrawBuffer(GL_BACK);
}

