/*
 * glwin.c -- Simple self-contained code for opening an 
 *            OpenGL-capable display window with a double buffered 
 *            mono or stereoscopic visual, and for receiving
 *            and decoding window-system events from
 *            Spaceball/SpaceNavigator/Magellen 6DOF input devices.
 *
 *            This code is primarily meant for 2-D image display
 *            or for trivial 3-D rendering usage without any GLX/WGL 
 *            extensions that have to be enumerated prior to 
 *            window creation.
 *
 *            This file is part of the Tachyon ray tracer.
 *            John E. Stone - john.stone@gmail.com
 *
 * $Id: glwin.c,v 1.1 2015/05/19 16:05:08 johns Exp $
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "glwin.h"

/* Support for compilation as part of VMD for interactive */
/* ray tracing display.                                   */
#if defined(VMDOPENGL)
#define USEOPENGL
#endif

#if defined(USEOPENGL)
#if defined(WIN32) && defined(_MSC_VER)
/*
 * Win32
 */
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0400     /* hack for definition of wheel event codes */
#endif
#include <windows.h>
#include <winuser.h>            /* mouse wheel event codes */
#include <GL/gl.h>
/* 3DxWare driver */
#if defined(VMDSPACEWARE) && defined(WIN32)
#define OS_WIN32 1
#include "spwmacro.h"           /* Spaceware include files */
#include "si.h"                 /* Spaceware include files */
#endif
#else
/*
 * X11
 */
#include <X11/Xlib.h>
#include <X11/keysym.h>
#include <GL/gl.h>
#include <GL/glx.h>
#endif


/* 
 * Spaceball/Magellan/SpaceNavigator handle data structures
 */

/* Window system event handling data */
typedef struct {
#if defined(WIN32) && defined(_MSC_VER)
#if defined(USESPACEWARE)
  /* 3DxWare driver */
  SiHdl sball;             /**< Spaceware handle      */
  SiSpwEvent spwevent;     /**< Spaceware event       */
  SiGetEventData spwedata; /**< Spaceware event data  */
#else
  int foo;                 /**< placeholder           */
#endif
#else
  /* Xlib ClientMessage-based driver */
  Display *dpy;
  Window drv_win;
  Window app_win;
  Atom ev_motion;
  Atom ev_button_press;
  Atom ev_button_release;
  Atom ev_command;
#endif
} spaceballhandle;


/* Platform-independent Spaceball event record */
typedef struct {
  int event;                 /**< event record          */
  int rx;                    /**< X-axis rotation       */
  int ry;                    /**< Y-axis rotation       */
  int rz;                    /**< Z-axis rotation       */
  int tx;                    /**< X-axis translation    */
  int ty;                    /**< Y-axis translation    */
  int tz;                    /**< Z-axis translation    */
  int buttons;               /**< button presses        */
  int period;                /**< time since last event */
} spaceballevent;


/* OS and windowing system-specific handle data */
typedef struct {
#if defined(WIN32) && defined(_MSC_VER)
  HWND hWnd;                 /**< window handle */
  HDC hDC;                   /**< device handle */
  HGLRC hRC;                 /**< WGL OpenGL window context */
  long scrwidth;             /**< screen width in pixels  */
  long scrheight;            /**< screen height in pixels */
  long MouseFlags;           /**< mouse event handling state */
#else
  int scrnum;                /**< X11 screen number */
  Display *dpy;              /**< X11 display handle */
  Window root;               /**< X11 root window */
  Window win;                /**< X11 window handle */
  Atom wmDeleteWindow;       /**< X11 ClientMessage window close event type */
  GLXContext ctx;            /**< GLX context for OpenGL window */
#endif

  int havestencil;           /**< stencil buffer available  */
  int instereo;              /**< stereo-capable GL context */

  int width;                 /**< width of window           */
  int height;                /**< height of window          */
  int xpos;                  /**< x position of window      */
  int ypos;                  /**< y position of window      */
  int mousex;                /**< x position of mouse       */
  int mousey;                /**< y position of mouse       */
  int evdev;                 /**< event device class        */
  int evval;                 /**< value of the event        */
  char evkey;                /**< keypress ASCII character  */

  int havefocus;             /**< Mouse/Kbd/Spaceball focus state          */
  spaceballhandle *sball;    /**< Spaceball/Magellan/SpaceNavigator handle */
  spaceballevent sballevent; /**< Most recent spaceball event status       */
} oglhandle;


#if !defined(WIN32) && !defined(_MSC_VER)
/*
 * X11 version 
 */

/*
 * Spaceball event handling routines
 */
#define SBALL_COMMAND_NONE                0
#define SBALL_COMMAND_APP_WINDOW      27695
#define SBALL_COMMAND_APP_SENSITIVITY 27696

/* enable 3Dxware Spaceball / Magellan / SpaceNavigator events */
static spaceballhandle * spaceball_attach(Display *dpy, Window win) {
  /* allocate and clear handle data structure */
  spaceballhandle *handle = (spaceballhandle *) malloc(sizeof(spaceballhandle));
  memset(handle, 0, sizeof(spaceballhandle));

  /* find and store X atoms for the event types we care about */
  handle->ev_motion         = XInternAtom(dpy, "MotionEvent", True);
  handle->ev_button_press   = XInternAtom(dpy, "ButtonPressEvent", True);
  handle->ev_button_release = XInternAtom(dpy, "ButtonReleaseEvent", True);
  handle->ev_command        = XInternAtom(dpy, "CommandEvent", True);

  if (!handle->ev_motion || !handle->ev_button_press ||
      !handle->ev_button_release || !handle->ev_command) {
    free(handle);
    return NULL; /* driver is not running */
  }

  /* Find the root window of the driver */
  Window root = RootWindow(dpy, DefaultScreen(dpy));

  /* Find the driver's window */
  Atom ActualType;
  int ActualFormat;
  unsigned long NItems, BytesReturn;
  unsigned char *PropReturn = NULL;
  XGetWindowProperty(dpy, root, handle->ev_command, 0, 1, 0,
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
      msg.xclient.send_event = 0;
      msg.xclient.display = dpy;
      msg.xclient.window = handle->drv_win;
      msg.xclient.message_type = handle->ev_command;

      msg.xclient.data.s[0] = (short) (((win)>>16)&0x0000FFFF); /* High 16 */
      msg.xclient.data.s[1] = (short) (((win))    &0x0000FFFF); /* Low 16  */
      msg.xclient.data.s[2] = SBALL_COMMAND_APP_WINDOW;         /* 27695   */

      int rc = XSendEvent(dpy, handle->drv_win, 0, 0x0000, &msg);
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
    /* We add the current control inputs to whatever we had previously, */
    /* so that we can process all queued events and not drop any inputs */
    sballevent->tx     += xev->xclient.data.s[2];  /* X translation          */
    sballevent->ty     += xev->xclient.data.s[3];  /* Y translation          */
    sballevent->tz     += xev->xclient.data.s[4];  /* Z translation          */
    sballevent->rx     += xev->xclient.data.s[5];  /* A rotation             */
    sballevent->ry     += xev->xclient.data.s[6];  /* B rotation             */
    sballevent->rz     += xev->xclient.data.s[7];  /* C rotation             */
    sballevent->period += xev->xclient.data.s[8];  /* Period in milliseconds */
    sballevent->event = 1;
    return 1;
  } else if (evtype == handle->ev_button_press) {
    sballevent->buttons |= (1 << xev->xclient.data.s[2]);
    sballevent->event = 1;
    return 1;
  } else if (evtype == handle->ev_button_release) {
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


/*
 * X11 implementation of glwin routines
 */
void * glwin_create(const char * wintitle, int width, int height) {
  oglhandle * handle; 
  XSetWindowAttributes attr;
  unsigned long mask;
  XVisualInfo *vis;
  XSizeHints sizeHints;
  GLint stencilbits;

  int glxstereoattrib[] =   { GLX_RGBA, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8,
                              GLX_BLUE_SIZE, 8, GLX_DEPTH_SIZE, 16, 
                              GLX_STENCIL_SIZE, 1,
                              GLX_STEREO, GLX_DOUBLEBUFFER, None };
  int glxnormalattrib[] =   { GLX_RGBA, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, 
                              GLX_BLUE_SIZE, 8, GLX_DEPTH_SIZE, 16, 
                              GLX_STENCIL_SIZE, 1,
                              GLX_DOUBLEBUFFER, None };
  int glxfailsafeattrib[] = { GLX_RGBA, GLX_RED_SIZE, 1, GLX_GREEN_SIZE, 1, 
                              GLX_BLUE_SIZE, 1, GLX_DEPTH_SIZE, 16, 
                              GLX_DOUBLEBUFFER, None };

  handle = (oglhandle *) malloc(sizeof(oglhandle));
  if (handle == NULL)
    return NULL;

  handle->width = width;
  handle->height = height;

#if defined(VMDOPENGL)
  /* If this code is compiled into VMD, we honor the VMD scheme for */
  /* directing the VMD graphics window to a particular display      */
  if (getenv("VMDGDISPLAY") != NULL) {
    handle->dpy = XOpenDisplay(getenv("VMDGDISPLAY"));
  } else {
    handle->dpy = XOpenDisplay(getenv("DISPLAY"));
  }
#else
  handle->dpy = XOpenDisplay(getenv("DISPLAY"));
#endif
  if (handle->dpy == NULL) {
    free(handle);
    return NULL;
  } 

  handle->scrnum = DefaultScreen(handle->dpy);
  handle->root = RootWindow(handle->dpy, handle->scrnum);
  handle->havestencil = 0;
  handle->instereo = 0;
  handle->evdev = GLWIN_EV_NONE;
  handle->havefocus = 0;

  /* try for full stereo w/ features first */
  handle->havestencil = 1;
  handle->instereo = 1;
  vis=glXChooseVisual(handle->dpy, handle->scrnum, glxstereoattrib);
  if (vis == NULL) {
    handle->havestencil = 1;
    handle->instereo = 0;
    /* try non-stereo w/ full features next */
    vis=glXChooseVisual(handle->dpy, handle->scrnum, glxnormalattrib);
    if (vis == NULL) {
      handle->havestencil = 0;
      handle->instereo = 0;
      /* try minimal features last */
      vis=glXChooseVisual(handle->dpy, handle->scrnum, glxfailsafeattrib);
      if (vis == NULL) {
        free(handle);
        return NULL;
      }
    }
  }

  /* window attributes */
  attr.background_pixel = 0;
  attr.border_pixel = 0;
  attr.colormap = XCreateColormap(handle->dpy, handle->root, 
                                  vis->visual, AllocNone);

  attr.event_mask = StructureNotifyMask | ExposureMask;
  mask = CWBackPixel | CWBorderPixel | CWColormap | CWEventMask;

  handle->win = XCreateWindow(handle->dpy, handle->root, 0, 0, width, height,
                              0, vis->depth, InputOutput,
                              vis->visual, mask, &attr );

  handle->ctx = glXCreateContext( handle->dpy, vis, NULL, True );

  glXMakeCurrent( handle->dpy, handle->win, handle->ctx );

  XStoreName(handle->dpy, handle->win, wintitle);

  XSelectInput(handle->dpy, handle->win,
               KeyPressMask | ButtonPressMask | ButtonReleaseMask |
               PointerMotionMask | StructureNotifyMask | ExposureMask |
               EnterWindowMask | LeaveWindowMask | FocusChangeMask);

  /* set window manager size and position hints */ 
  memset((void *) &(sizeHints), 0, sizeof(sizeHints));
  sizeHints.flags |= USSize;
  sizeHints.flags |= USPosition;
  sizeHints.width = width;
  sizeHints.height = height;
  sizeHints.x = 0;
  sizeHints.y = 0;
  XSetWMNormalHints(handle->dpy, handle->win, &sizeHints); 

  /* cause X11 to generate a ClientMessage event when the window manager */
  /* wants to close window, so we can gracefully exit rather than crash  */
  handle->wmDeleteWindow = XInternAtom(handle->dpy, "WM_DELETE_WINDOW", False);
  XSetWMProtocols(handle->dpy, handle->win, &handle->wmDeleteWindow, 1);

  XMapRaised(handle->dpy, handle->win);

  /* Enable Spaceball events to this window */
#if 0
  /* do focus processing for ourselves      */
  handle->sball = spaceball_attach(handle->dpy, handle->win);
#else
  /* driver will do focus processing for us */
  handle->sball = spaceball_attach(handle->dpy, InputFocus);
#endif
  /* initialize spaceball event structure   */
  spaceball_init_event(&handle->sballevent);
  spaceball_clear_event(&handle->sballevent);

  glwin_handle_events(handle, GLWIN_EV_POLL_BLOCK);

  /* check for an OpenGL stencil buffer */
  glGetIntegerv(GL_STENCIL_BITS, &stencilbits);
  if (stencilbits > 0) {
    handle->havestencil = 1;
  }

  glClearColor(0.0, 0.0, 0.0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT);
  glwin_swap_buffers(handle);
  glClear(GL_COLOR_BUFFER_BIT);
  glwin_swap_buffers(handle);


  XFlush(handle->dpy);

  return handle;
}


void glwin_destroy(void * voidhandle) {
  oglhandle * handle = (oglhandle *) voidhandle;
  if (handle == NULL)
    return;

  /* detach spaceball        */
  if (handle->sball != NULL) {
    spaceball_close(handle->sball);
    handle->sball = NULL;
  }

  /* close and delete window */
  XUnmapWindow(handle->dpy, handle->win);
  glXMakeCurrent(handle->dpy, None, NULL);
  XDestroyWindow(handle->dpy, handle->win);
  XCloseDisplay(handle->dpy); 
}

 
void glwin_swap_buffers(void * voidhandle) {
  oglhandle * handle = (oglhandle *) voidhandle;

  if (handle != NULL)
    glXSwapBuffers(handle->dpy, handle->win);
}


int glwin_handle_events(void * voidhandle, int evblockmode) {
  oglhandle * handle = (oglhandle *) voidhandle;
  int rc=0;
  char keybuf[10];
  int keybuflen = 9;
  KeySym keysym;
  XComposeStatus comp;

  if (handle == NULL)
    return 0;

  /* clear previous spaceball event state, except for button state which  */
  /* must be left alone.                                                  */
  spaceball_clear_event(&handle->sballevent);

  /* only process a single recognized event and then return to the caller */
  /* otherwise it is too easy to drop mouse button press/release events   */
  while (!rc && (evblockmode || XPending(handle->dpy))) {
    int k; 
    unsigned int button;
    XEvent event;

    evblockmode = GLWIN_EV_POLL_NONBLOCK; /* subsequent loops don't block */
    XNextEvent(handle->dpy, &event);
    handle->evdev = GLWIN_EV_NONE;
    handle->evval = 0;
    handle->evkey = '\0';

    switch(event.type) {
      case Expose:
      case ReparentNotify:
      case MapNotify:
        rc=1; /* we got one event */
        break;

      case ConfigureNotify:
        handle->width = event.xconfigure.width;
        handle->height = event.xconfigure.height;
        handle->xpos = event.xconfigure.x;
        handle->ypos = event.xconfigure.y;
        rc=1; /* we got one event */
        break;

      case KeyPress:
        handle->mousex = event.xbutton.x;
        handle->mousey = event.xbutton.y;
        k = XLookupString(&(event.xkey), keybuf, keybuflen,  &keysym, &comp);
        if (k > 0 && keybuf[0] != '\0') {
          handle->evdev = GLWIN_EV_KBD;
          handle->evkey = keybuf[0];
          rc=1; /* we got one event */
        } else {
          handle->evdev = GLWIN_EV_NONE;
          handle->evkey = 0;
          switch (keysym) {
            case XK_Up:          handle->evdev = GLWIN_EV_KBD_UP;        break;
            case XK_Down:        handle->evdev = GLWIN_EV_KBD_DOWN;      break;
            case XK_Left:        handle->evdev = GLWIN_EV_KBD_LEFT;      break;
            case XK_Right:       handle->evdev = GLWIN_EV_KBD_RIGHT;     break;
            case XK_Page_Up:     handle->evdev = GLWIN_EV_KBD_PAGE_UP;   break;
            case XK_Page_Down:   handle->evdev = GLWIN_EV_KBD_PAGE_UP;   break;
            case XK_Home:        handle->evdev = GLWIN_EV_KBD_HOME;      break;
            case XK_End:         handle->evdev = GLWIN_EV_KBD_END;       break;
            case XK_Insert:      handle->evdev = GLWIN_EV_KBD_INSERT;    break;
            case XK_Delete:      handle->evdev = GLWIN_EV_KBD_DELETE;    break;

            case XK_F1:          handle->evdev = GLWIN_EV_KBD_F1;        break;
            case XK_F2:          handle->evdev = GLWIN_EV_KBD_F2;        break;
            case XK_F3:          handle->evdev = GLWIN_EV_KBD_F3;        break;
            case XK_F4:          handle->evdev = GLWIN_EV_KBD_F4;        break;
            case XK_F5:          handle->evdev = GLWIN_EV_KBD_F5;        break;
            case XK_F6:          handle->evdev = GLWIN_EV_KBD_F6;        break;
            case XK_F7:          handle->evdev = GLWIN_EV_KBD_F7;        break;
            case XK_F8:          handle->evdev = GLWIN_EV_KBD_F8;        break;
            case XK_F9:          handle->evdev = GLWIN_EV_KBD_F9;        break;
            case XK_F10:         handle->evdev = GLWIN_EV_KBD_F10;       break;
            case XK_F11:         handle->evdev = GLWIN_EV_KBD_F11;       break;
            case XK_F12:         handle->evdev = GLWIN_EV_KBD_F12;       break;

            case XK_Escape:      handle->evdev = GLWIN_EV_KBD_F12;       break;
          }
          if (handle->evdev != GLWIN_EV_NONE) 
            rc=1;
        }
        break;

      case MotionNotify:
        handle->evdev = GLWIN_EV_MOUSE_MOVE;
        handle->mousex = event.xmotion.x;
        handle->mousey = event.xmotion.y;
        rc=1; /* we got one event */
        break; 

      case ButtonPress:
      case ButtonRelease:
        button = event.xbutton.button;
        handle->evval = (event.type == ButtonPress);
        handle->mousex = event.xbutton.x;
        handle->mousey = event.xbutton.y;
        switch (button) {
          case Button1:
            handle->evdev = GLWIN_EV_MOUSE_LEFT;
            rc=1; /* we got one event */
            break;
          case Button2:
            handle->evdev = GLWIN_EV_MOUSE_MIDDLE;
            rc=1; /* we got one event */
            break;
          case Button3:
            handle->evdev = GLWIN_EV_MOUSE_RIGHT;
            rc=1; /* we got one event */
            break;
          case Button4:
            handle->evdev = GLWIN_EV_MOUSE_WHEELUP;
            rc=1; /* we got one event */
            break;
          case Button5:
            handle->evdev = GLWIN_EV_MOUSE_WHEELDOWN;
            rc=1; /* we got one event */
            break;
        }
        break;

      case FocusIn:
      case EnterNotify:
        handle->havefocus=1;
        break;

      case FocusOut:
      case LeaveNotify:
        handle->havefocus=0;
        break;

      case ClientMessage:
        /* handle window close events */
        if (event.xclient.data.l[0] == handle->wmDeleteWindow) {
          handle->evdev = GLWIN_EV_WINDOW_CLOSE;
          rc=1;
        } else {
#if 1
          /* let the spaceball driver take care of focus processing           */
          /* if we have mouse/keyboard focus, then translate spaceball events */
          spaceball_decode_event(handle->sball, &event, &handle->sballevent);
#else
          /* do our own focus handling                                        */
          /* if we have mouse/keyboard focus, then translate spaceball events */
          if (handle->havefocus) {
            spaceball_decode_event(handle->sball, &event, &handle->sballevent);
          }
#endif
        } 
        break;

    } 
  }

  return rc;
} 


int glwin_resize(void *voidhandle, int width, int height) {
  oglhandle * handle = (oglhandle *) voidhandle;
  if (handle == NULL)
    return -1;

  XResizeWindow(handle->dpy, handle->win, width, height);

  return 0;
}


int glwin_reposition(void *voidhandle, int xpos, int ypos) {
  oglhandle * handle = (oglhandle *) voidhandle;
  if (handle == NULL)
    return -1;

  XMoveWindow(handle->dpy, handle->win, xpos, ypos);

  return 0;
}


int glwin_fullscreen(void * voidhandle, int fson, int xinescreen) {
  struct {
    unsigned long flags;
    unsigned long functions;
    unsigned long decorations;
    long inputMode;
    unsigned long status;
  } wmhints;
  Atom wmproperty;

  oglhandle * handle = (oglhandle *) voidhandle;

  memset(&wmhints, 0, sizeof(wmhints));
  wmhints.flags = 2;         /* changing window decorations */
  if (fson) {
    wmhints.decorations = 0; /* 0 (false) no window decorations */
  } else {
    wmhints.decorations = 1; /* 1 (true) window decorations enabled */
  }
  wmproperty = XInternAtom(handle->dpy, "_MOTIF_WM_HINTS", True);
  XChangeProperty(handle->dpy, handle->win, wmproperty, wmproperty, 32,
                  PropModeReplace, (unsigned char *) &wmhints, 5);

  /* resize window to size of either the whole X display screen, */
  /* or to the size of one of the Xinerama component displays    */
  /* if Xinerama is enabled, and xinescreen is not -1.           */
  if (fson) {
    int dpyScreen = DefaultScreen(handle->dpy);

    XSizeHints sizeHints;
    memset((void *) &(sizeHints), 0, sizeof(sizeHints));
    sizeHints.flags |= USSize;
    sizeHints.flags |= USPosition;

    sizeHints.width = DisplayWidth(handle->dpy, dpyScreen);
    sizeHints.height = DisplayHeight(handle->dpy, dpyScreen);
    sizeHints.x = 0;
    sizeHints.y = 0;

#if defined(USEXINERAMA)
    if (xinescreen != -1) {
      int xinerr, xinevent, xinenumscreens;
      if (XineramaQueryExtension(handle->dpy, &xinevent, &xinerr) &&
          XineramaIsActive(handle->dpy)) {
        XineramaScreenInfo *screens =
          XineramaQueryScreens(handle->dpy, &xinenumscreens);
        if (xinescreen >= 0 && xinescreen < xinenumscreens) {
          sizeHints.width = screens[xinescreen].width;
          sizeHints.height = screens[xinescreen].height;
          sizeHints.x = screens[xinescreen].x_org;
          sizeHints.y = screens[xinescreen].y_org;
#if 1 || defined(DEBUGOUTPUT)
          printf("*** OpenGL Stereo: Xinerama screen %d, +%d+%dx%dx%d\n",
                 xinescreen, sizeHints.x, sizeHints.y,
                 sizeHints.width, sizeHints.height);
#endif
        } else {
          printf("*** OpenGL Stereo: no such Xinerama screen index %d\n",
                 xinescreen);
        }
        XFree(screens);
      }
    }
#endif

    XMoveWindow(handle->dpy, handle->win, sizeHints.x, sizeHints.y);
    XResizeWindow(handle->dpy, handle->win, sizeHints.width, sizeHints.height);
  }

  return 0;
}


#else

/* 
 *  Win32 Version 
 */

/*
 * Spaceball event handling routines
 */

#if defined(USESPACEWARE)

static spaceballhandle * spaceball_attach(HWND hWnd) {
  SiOpenData oData;
  enum SpwRetVal res;

  switch (SiInitialize()) {
    case SPW_NO_ERROR:         /* init succeeded */
      break;

    case SPW_DLL_LOAD_ERROR:   /* driver not installed */
    default:                   /* error prevented init */
      return NULL;
  }

  /* allocate and clear handle data structure */
  spaceballhandle *handle = (spaceballhandle *) malloc(sizeof(spaceballhandle));
  memset(handle, 0, sizeof(spaceballhandle));

  SiOpenWinInit(&oData, hWnd); /* init win platform data */
  SiSetUiMode(handle->sball, SI_UI_ALL_CONTROLS); /* config softbutton display */

  /* start a connection to the device now that the UI mode */
  /* and window system data are setup.                              */
  handle->sball = SiOpen("OpenGL", SI_ANY_DEVICE, SI_NO_MASK, SI_EVENT, &oData);
  if ((handle->sball == NULL) || (handle->sball == SI_NO_HANDLE)) {
    SiTerminate(); /* shutdown spaceware input library */
    free(handle);
    return NULL;
  }

  res = SiBeep(handle->sball, "CcCc"); // beep the spaceball
  if ((handle->sball != NULL) && (handle->sball != SI_NO_HANDLE))
    return handle;

  free(handle);
  return NULL;
}


static void spaceball_close(spaceballhandle *handle) {
  if (handle == NULL)
    return;

  if (handle->sball != NULL) {
    enum SpwRetVal res;
    res = SiClose(handle->sball); /* close spaceball device */
    if (res != SPW_NO_ERROR)
      printf("An error occured during Spaceball shutdown.\n");
    SiTerminate(); /* shutdown spaceware input library */
  }

  free(handle);
}


static int spaceball_decode_event(spaceballhandle *handle, spaceballevent *sballevent, UINT msg, WPARAM wParam, LPARAM lParam) {
  if (handle == NULL)
    return 0;

  if (handle->sball == NULL)
    return 0; /* no spaceball attached/running */

  /* Check to see if this message is a spaceball message */
  SiGetEventWinInit(&handle->spwedata, msg, wParam, lParam);

  if (SiGetEvent(handle->sball, 0, &handle->spwedata, &handle->spwevent) == SI_IS_EVENT) {
    return 1;
  }

  return 0;
}

#endif /* USESPACEWARE */



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


/*
 * declaration of myWindowProc()
 */
LRESULT WINAPI myWindowProc( HWND, UINT, WPARAM, LPARAM );

static const char *szClassName = "OpenGLWindow";

static int OpenWin32Connection(oglhandle * handle) {
  WNDCLASS  wc;
  HINSTANCE hInstance = GetModuleHandle(NULL);

  /* Clear (important!) and then fill in the window class structure. */
  memset(&wc, 0, sizeof(WNDCLASS));
  wc.style         = CS_OWNDC;
  wc.lpfnWndProc   = (WNDPROC) myWindowProc;
  wc.hInstance     = hInstance;
  wc.hIcon         = LoadIcon(NULL, IDI_WINLOGO);
  wc.hCursor       = LoadCursor(hInstance, IDC_ARROW);
  wc.hbrBackground = NULL; /* Default color */
  wc.lpszMenuName  = NULL;
  wc.lpszClassName = szClassName;

  if(!RegisterClass(&wc)) {
    printf("Cannot register window class.\n");
    return -1;
  }

  handle->scrwidth  = GetSystemMetrics(SM_CXSCREEN);
  handle->scrheight = GetSystemMetrics(SM_CYSCREEN); 

  return 0;
}


static HGLRC SetupOpenGL(oglhandle * handle) {
  int nMyPixelFormatID;
  HDC hDC;
  HGLRC hRC;
  PIXELFORMATDESCRIPTOR checkpfd;
  static PIXELFORMATDESCRIPTOR pfd = {
        sizeof (PIXELFORMATDESCRIPTOR), /* struct size      */
        1,                              /* Version number   */
        PFD_DRAW_TO_WINDOW      /* Flags, draw to a window, */
          | PFD_DOUBLEBUFFER    /* Requires Doublebuffer hw */
          | PFD_STEREO          /* we want stereo if possible */
          | PFD_SUPPORT_OPENGL, /* use OpenGL               */
        PFD_TYPE_RGBA,          /* RGBA pixel values        */
        24,                     /* 24-bit color             */
        0, 0, 0,                /* RGB bits & shift sizes.  */
        0, 0, 0,                /* Don't care about them    */
        0, 0,                   /* No alpha buffer info     */
        0, 0, 0, 0, 0,          /* No accumulation buffer   */
        16,                     /* 16-bit depth buffer      */
        1,                      /* Want stencil buffer      */
        0,                      /* No auxiliary buffers     */
        PFD_MAIN_PLANE,         /* Layer type               */
        0,                      /* Reserved (must be 0)     */
        0,                      /* No layer mask            */
        0,                      /* No visible mask          */
        0                       /* No damage mask           */
  };

  hDC = GetDC(handle->hWnd);
  nMyPixelFormatID = ChoosePixelFormat(hDC, &pfd);

  /* 
   * catch errors here.
   * If nMyPixelFormat is zero, then there's
   * something wrong... most likely the window's
   * style bits are incorrect (in CreateWindow() )
   * or OpenGL isn't installed on this machine
   *
   */
  if (nMyPixelFormatID == 0) {
    printf("Error selecting OpenGL Pixel Format!!\n");
    return NULL;
  }

  /* check for stereo window */
  DescribePixelFormat(hDC, nMyPixelFormatID, 
                      sizeof(PIXELFORMATDESCRIPTOR), &checkpfd);
  if (checkpfd.dwFlags & PFD_STEREO)
    handle->instereo = 1;
  else 
    handle->instereo = 0;
 
  SetPixelFormat(hDC, nMyPixelFormatID, &pfd);

  hRC = wglCreateContext(hDC);
  ReleaseDC(handle->hWnd, hDC);

  return hRC;
}


static int myCreateWindow(oglhandle * handle, const char * wintitle,
                          int xpos, int ypos, int xs, int ys) {
  /* Create a main window for this application instance. */
  handle->hWnd = 
        CreateWindow(
              szClassName,        /* Window class name */
              wintitle,           /* Text for window title bar */
              WS_OVERLAPPEDWINDOW /* Window style */
               | WS_CLIPCHILDREN
               | WS_CLIPSIBLINGS, /* NEED THESE for OpenGL calls to work! */
              xpos, ypos,
              xs, ys,
              NULL,                  /* no parent window                */
              NULL,                  /* Use the window class menu.      */
              GetModuleHandle(NULL), /* This instance owns this window  */
              handle                 /* We don't use any extra data     */
        );

  if (!handle->hWnd) {
    printf("Couldn't Open Window!!\n");
    return -1;
  }

  handle->hDC = GetDC(handle->hWnd);
  wglMakeCurrent(handle->hDC, handle->hRC);

  /* Make the window visible & update its client area */
  ShowWindow( handle->hWnd, SW_SHOW);  /* Show the window         */
  UpdateWindow( handle->hWnd );        /* Sends WM_PAINT message  */

  return 0;
}


static void win32decodemouse(oglhandle *handle, LPARAM lParam) {
  int x, y;
  x = LOWORD(lParam);
  y = HIWORD(lParam);
  /* handle mouse capture in negative range */
  if (x & 1 << 15) x -= (1 << 16);
  if (y & 1 << 15) y -= (1 << 16);
  handle->mousex = x;
  handle->mousey = y;
}


LRESULT WINAPI myWindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
  PAINTSTRUCT   ps; /* Paint structure. */
  oglhandle *handle;

  /* Upon first window creation, immediately set our user-data field */
  /* to store caller-provided handles for this window instance       */
  if (msg == WM_NCCREATE) {
#if defined(_M_X64) || defined(_WIN64) || defined(_Wp64)
    SetWindowLongPtr(hwnd, GWLP_USERDATA, (LONG_PTR) (((CREATESTRUCT *) lParam)->lpCreateParams));
#elif 1
    SetWindowLong(hwnd, GWL_USERDATA, (LONG) (((CREATESTRUCT *) lParam)->lpCreateParams));
#else
    SetProp(hwnd, "OGLHANDLE", (((CREATESTRUCT *) lParam)->lpCreateParams));
#endif
  }

  /* check to make sure we have a valid window data structure in case */
  /* it is destroyed while there are still pending messages...        */
#if defined(_M_X64) || defined(_WIN64) || defined(_Wp64)
  handle = (oglhandle *) GetWindowLongPtr(hwnd, GWLP_USERDATA);
#elif 1
  handle = (oglhandle *) GetWindowLong(hwnd, GWL_USERDATA);
#else
  handle = (oglhandle *) GetProp(hwnd, "OGLHANDLE");
#endif
  if (handle == NULL)
    return DefWindowProc(hwnd, msg, wParam, lParam);

  switch (msg) {
    case WM_CREATE:
      handle->hWnd = hwnd; /* must be set before we do anything else */
      handle->hRC = SetupOpenGL(handle);
      return 0;

    case WM_MOVE:
      wglMakeCurrent(handle->hDC, handle->hRC);
      handle->xpos = LOWORD(lParam);
      handle->ypos = HIWORD(lParam);
      return 0;

    case WM_SIZE:
      wglMakeCurrent(handle->hDC, handle->hRC);
      handle->width  = LOWORD(lParam);
      handle->height = HIWORD(lParam);
      return 0;

    case WM_KEYDOWN:
      handle->evdev = GLWIN_EV_KBD;
      /* try to map to ASCII first */
      /* handle->evkey = MapVirtualKey((UINT) wParam, MAPVK_VK_TO_CHAR); */
      handle->evkey = MapVirtualKey((UINT) wParam, 2);
      if (handle->evkey == 0) {
        /* if no ASCII code, try mapping to a virtual key scan code,  */
        /* but don't bother distinguishing which left/right key it is */
        /*unsigned int keysym = MapVirtualKey((UINT) wParam, MAPVK_VK_TO_VSC);*/
        unsigned int keysym = wParam;

        switch (keysym) {
          case VK_UP:          handle->evdev = GLWIN_EV_KBD_UP;        break;
          case VK_DOWN:        handle->evdev = GLWIN_EV_KBD_DOWN;      break;
          case VK_LEFT:        handle->evdev = GLWIN_EV_KBD_LEFT;      break;
          case VK_RIGHT:       handle->evdev = GLWIN_EV_KBD_RIGHT;     break;
          case VK_PRIOR:       handle->evdev = GLWIN_EV_KBD_PAGE_UP;   break;
          case VK_NEXT:        handle->evdev = GLWIN_EV_KBD_PAGE_UP;   break;
          case VK_HOME:        handle->evdev = GLWIN_EV_KBD_HOME;      break;
          case VK_END:         handle->evdev = GLWIN_EV_KBD_END;       break;
          case VK_INSERT:      handle->evdev = GLWIN_EV_KBD_INSERT;    break;
          case VK_DELETE:      handle->evdev = GLWIN_EV_KBD_DELETE;    break;

          case VK_F1:          handle->evdev = GLWIN_EV_KBD_F1;        break;
          case VK_F2:          handle->evdev = GLWIN_EV_KBD_F2;        break;
          case VK_F3:          handle->evdev = GLWIN_EV_KBD_F3;        break;
          case VK_F4:          handle->evdev = GLWIN_EV_KBD_F4;        break;
          case VK_F5:          handle->evdev = GLWIN_EV_KBD_F5;        break;
          case VK_F6:          handle->evdev = GLWIN_EV_KBD_F6;        break;
          case VK_F7:          handle->evdev = GLWIN_EV_KBD_F7;        break;
          case VK_F8:          handle->evdev = GLWIN_EV_KBD_F8;        break;
          case VK_F9:          handle->evdev = GLWIN_EV_KBD_F9;        break;
          case VK_F10:         handle->evdev = GLWIN_EV_KBD_F10;       break;
          case VK_F11:         handle->evdev = GLWIN_EV_KBD_F11;       break;
          case VK_F12:         handle->evdev = GLWIN_EV_KBD_F12;       break;

          case VK_ESCAPE:      handle->evdev = GLWIN_EV_KBD_ESC;       break;

          default:
            handle->evdev = GLWIN_EV_NONE;
            break;
        }
      }
      return 0;

    case WM_MOUSEMOVE:
      win32decodemouse(handle, lParam);
      handle->evdev = GLWIN_EV_MOUSE_MOVE;
      handle->MouseFlags = (long) wParam;
      if (!(handle->MouseFlags & (MK_LBUTTON | MK_MBUTTON | MK_RBUTTON)))
        ReleaseCapture();
      return 0;

    case WM_MOUSEWHEEL:
      {
        int wheeldelta = ((short) HIWORD(wParam));
        if (wheeldelta > (WHEEL_DELTA / 2)) {
          handle->evdev = GLWIN_EV_MOUSE_WHEELUP;
        } else if (wheeldelta < -(WHEEL_DELTA / 2)) {
          handle->evdev = GLWIN_EV_MOUSE_WHEELDOWN;
        }
      }
      return 0;

    case WM_LBUTTONDOWN:
      SetCapture(hwnd);
      win32decodemouse(handle, lParam);
      handle->MouseFlags = (long) wParam;
      handle->evdev = GLWIN_EV_MOUSE_LEFT;
      handle->evval = 1;
      return 0;

    case WM_LBUTTONUP:
      win32decodemouse(handle, lParam);
      handle->MouseFlags = (long) wParam;
      handle->evdev = GLWIN_EV_MOUSE_LEFT;
      handle->evval = 0;
      if (!(handle->MouseFlags & (MK_LBUTTON | MK_MBUTTON | MK_RBUTTON)))
        ReleaseCapture();
      return 0;

    case WM_MBUTTONDOWN:
      SetCapture(hwnd);
      win32decodemouse(handle, lParam);
      handle->MouseFlags = (long) wParam;
      handle->evdev = GLWIN_EV_MOUSE_MIDDLE;
      handle->evval = 1;
      return 0;

    case WM_MBUTTONUP:
      win32decodemouse(handle, lParam);
      handle->MouseFlags = (long) wParam;
      handle->evdev = GLWIN_EV_MOUSE_MIDDLE;
      handle->evval = 0;
      if (!(handle->MouseFlags & (MK_LBUTTON | MK_MBUTTON | MK_RBUTTON)))
        ReleaseCapture();
      return 0;

    case WM_RBUTTONDOWN:
      SetCapture(hwnd);
      win32decodemouse(handle, lParam);
      handle->MouseFlags = (long) wParam;
      handle->evdev = GLWIN_EV_MOUSE_RIGHT;
      handle->evval = 1;
      return 0;

    case WM_RBUTTONUP:
      win32decodemouse(handle, lParam);
      handle->MouseFlags = (long) wParam;
      handle->evdev = GLWIN_EV_MOUSE_RIGHT;
      handle->evval = 0;
      if (!(handle->MouseFlags & (MK_LBUTTON | MK_MBUTTON | MK_RBUTTON)))
        ReleaseCapture();
      return 0;

    case WM_CLOSE:
      PostQuitMessage(0);
      return 0;

    case WM_PAINT:
      BeginPaint(hwnd, &ps);
      EndPaint(hwnd, &ps);
      return 0;

    case WM_SIZING:
      glClear(GL_COLOR_BUFFER_BIT);
      SwapBuffers(handle->hDC);
      glDrawBuffer(GL_BACK);
      return 0;

    case WM_SETCURSOR:
      if (LOWORD(lParam) == HTCLIENT) {
        SetCursor(LoadCursor(NULL, IDC_ARROW));
        return 0;
      }
      return DefWindowProc(hwnd, msg, wParam, lParam);

    default:
      return DefWindowProc(hwnd, msg, wParam, lParam);
  }

  return 0;
}


void * glwin_create(const char * wintitle, int width, int height) {
  oglhandle * handle;
  int rc;
  GLint stencilbits;

  handle = (oglhandle *) malloc(sizeof(oglhandle));
  if (handle == NULL)
    return NULL;

  handle->havestencil=0; /* initialize stencil state */
  handle->instereo=0;    /* mark this as a non-stereo window */

  handle->width = width;
  handle->height = height;
  handle->evdev = GLWIN_EV_NONE;

  rc = OpenWin32Connection(handle);
  if (rc != 0) {
    printf("OpenWin32Connection() returned an error!\n");
    free(handle);
    return NULL;
  } 

  handle->width = width;
  handle->height = height;
  
  rc = myCreateWindow(handle, wintitle, 0, 0, width, height); 
  if (rc != 0) {
    printf("CreateWindow() returned an error!\n");
    free(handle);
    return NULL;
  } 

  /* Enable Spaceball events to this window */
#if 0
  handle->sball = spaceball_attach(handle->win);
#endif
  /* initialize spaceball event structure   */
  spaceball_init_event(&handle->sballevent);
  spaceball_clear_event(&handle->sballevent);

  /* check for an OpenGL stencil buffer */
  glGetIntegerv(GL_STENCIL_BITS, &stencilbits);
  if (stencilbits > 0) {
    handle->havestencil = 1;
  }

  return handle;
}


void glwin_destroy(void * voidhandle) {
  oglhandle * handle = (oglhandle *) voidhandle;

  wglDeleteContext(handle->hRC);
  PostQuitMessage( 0 );

  /* glwin_handle_events(handle, GLWIN_EV_POLL_NONBLOCK); */
}


void glwin_swap_buffers(void * voidhandle) {
  oglhandle * handle = (oglhandle *) voidhandle;
  glFlush();
  SwapBuffers(handle->hDC);
  glDrawBuffer(GL_BACK);
}


int glwin_handle_events(void * voidhandle, int evblockmode) {
  oglhandle * handle = (oglhandle *) voidhandle;
  MSG msg;
  int rc=0;
  int pending=0;

  handle->evdev = GLWIN_EV_NONE;
  handle->evval = 0;
  handle->evkey = '\0';

  /* This pumps the Windows message queue, forcing events to be updated */
  /* by the time we return from DispatchMessage.                        */
  pending=PeekMessage(&msg, NULL, 0, 0, PM_REMOVE);
  while (!rc && (evblockmode || pending)) {
    if (pending) {
      TranslateMessage(&msg); /* translate the message          */
      DispatchMessage(&msg);  /* fire it off to the window proc */
      pending=0;
    } else if (evblockmode == GLWIN_EV_POLL_BLOCK) {
      if (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
      }
    } else {
      if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
        TranslateMessage(&msg); /* translate the message          */
        DispatchMessage(&msg);  /* fire it off to the window proc */
      }
    }
    if (handle->evdev != GLWIN_EV_NONE)
      rc=1;
  }

  return rc;
}


int glwin_resize(void *voidhandle, int width, int height) {
  RECT rcClient, rcWindow;
  POINT ptDiff;
  oglhandle * handle = (oglhandle *) voidhandle;
  if (handle == NULL)
    return -1;

  GetClientRect(handle->hWnd, &rcClient);
  GetWindowRect(handle->hWnd, &rcWindow);
  ptDiff.x = (rcWindow.right - rcWindow.left) - rcClient.right;
  ptDiff.y = (rcWindow.bottom - rcWindow.top) - rcClient.bottom;
  MoveWindow(handle->hWnd, rcWindow.left, rcWindow.top, width + ptDiff.x, height + ptDiff.y, TRUE);

  return 0;
}


int glwin_reposition(void *voidhandle, int xpos, int ypos) {
  RECT rcClient, rcWindow;
  oglhandle * handle = (oglhandle *) voidhandle;
  if (handle == NULL)
    return -1;

  GetClientRect(handle->hWnd, &rcClient);
  GetWindowRect(handle->hWnd, &rcWindow);
  MoveWindow(handle->hWnd, xpos, ypos, rcWindow.right-rcWindow.left, rcWindow.bottom-rcWindow.top, TRUE);

  return 0;
}


int glwin_fullscreen(void * voidhandle, int fson, int xinescreen) {
  return -1;
}  


#endif


/* 
 * Code used for both Windows and Linux 
 */
int glwin_query_extension(void *voidhandle, const char *extname) {
  char *ext;
  char *endext;
  if (!extname)
    return 0;

  /* search for extension in list of available extensions */
  ext = (char *) glGetString(GL_EXTENSIONS);
  if (ext != NULL) {
    endext = ext + strlen(ext);
    while (ext < endext) {
      size_t n = strcspn(ext, " ");
      if ((strlen(extname) == n) && (strncmp(extname, ext, n) == 0)) {
        return 1; /* extension is available */
        break;
      }
      ext += (n + 1);
    }
  }

  return 0; /* False, extension is not available */
}


void glwin_draw_image(void * voidhandle, int xsize, int ysize, unsigned char * img) {
  glRasterPos2i(0, 0);
  glDrawPixels(xsize, ysize, GL_RGB, GL_UNSIGNED_BYTE, img);
  glwin_swap_buffers(voidhandle);
}


int glwin_get_wininfo(void * voidhandle, int *instereo, int *havestencil) {
  oglhandle * handle = (oglhandle *) voidhandle;
  if (handle == NULL)
    return -1;

  if (instereo != NULL)
    *instereo = handle->instereo;

  if (havestencil != NULL)
    *havestencil = handle->havestencil;

  return 0;
}


int glwin_get_winsize(void * voidhandle, int *xsize, int *ysize) {
  oglhandle * handle = (oglhandle *) voidhandle;
  if (handle == NULL)
    return -1;

  if (xsize != NULL)
    *xsize = handle->width;

  if (ysize != NULL)
    *ysize = handle->height;

  return 0;
}


int glwin_get_winpos(void * voidhandle, int *xpos, int *ypos) {
  oglhandle * handle = (oglhandle *) voidhandle;
  if (handle == NULL)
    return -1;

  if (xpos != NULL)
    *xpos = handle->xpos;

  if (ypos != NULL)
    *ypos = handle->ypos;

  return 0;
}


int glwin_get_mousepointer(void *voidhandle, int *x, int *y) {
  oglhandle * handle = (oglhandle *) voidhandle;
  if (handle == NULL)
    return -1;

  if (x != NULL)
    *x = handle->mousex;

  if (y != NULL)
    *y = handle->mousey;

  return 0;
}


int glwin_get_lastevent(void * voidhandle, int *evdev, int *evval, char *evkey) {
  oglhandle * handle = (oglhandle *) voidhandle;
  if (handle == NULL)
    return -1;

  if (evdev != NULL)
    *evdev = handle->evdev;

  if (evval != NULL)
    *evval = handle->evval;

  if (evkey != NULL)
    *evkey = handle->evkey;

  return 0;
}


int glwin_spaceball_available(void *voidhandle) {
  oglhandle * handle = (oglhandle *) voidhandle;

  /* check to see if we have a spaceball attached */
  if (handle->sball != NULL)
    return 1;

  return 0;
}


int glwin_get_spaceball(void *voidhandle, int *rx, int *ry, int *rz, int *tx, int *ty, int *tz, int *buttons) {
  oglhandle * handle = (oglhandle *) voidhandle;
  if (handle == NULL)
    return 0;

  if ((handle->sball != NULL) && (handle->sballevent.event == 1)) {
    *rx = handle->sballevent.rx;
    *ry = handle->sballevent.ry;
    *rz = handle->sballevent.rz;
    *tx = handle->sballevent.tx;
    *ty = handle->sballevent.ty;
    *tz = handle->sballevent.tz;
    *buttons = handle->sballevent.buttons;
    return 1;
  }

  return 0;
}


#else


/* 
 * stub code to allow linkage
 */
void * glwin_create(const char * wintitle, int width, int height) {
  return NULL;
}

void glwin_destroy(void * voidhandle) {
  return;
}

void glwin_swap_buffers(void * voidhandle) {
  return;
}

int glwin_handle_events(void * voidhandle, int evblockmode) {
  return 0;
}

int glwin_get_wininfo(void * voidhandle, int *instereo, int *havestencil) {
  return -1;
}

int glwin_get_winsize(void * voidhandle, int *xsize, int *ysize) {
  return -1;
}

int glwin_get_winpos(void * voidhandle, int *xpos, int *ypos) {
  return -1;
}

int glwin_get_mousepointer(void *voidhandle, int *x, int *y) {
  return -1;
}

int glwin_get_lastevent(void * voidhandle, int *evdev, int *evval, char *evkey) {
  return -1;
}

int glwin_query_extension(void *voidhandle, const char *extname) {
  return 0;
}

void glwin_draw_image(void * voidhandle, int xsize, int ysize, unsigned char * img) {
  return;
}

int glwin_resize(void *voidhandle, int width, int height) {
  return -1;
}

int glwin_reposition(void *voidhandle, int xpos, int ypos) {
  return -1;
}

int glwin_fullscreen(void * voidhandle, int fson, int xinescreen) {
  return -1;
}  

int glwin_spaceball_available(void *voidhandle) {
  return 0;
}

int glwin_get_spaceball(void *voidhandle, int *rx, int *ry, int *rz, int *tx, int *ty, int *tz, int *buttons) {
  return 0;
}

#endif

