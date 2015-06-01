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
 *	$RCSfile: OpenGLPbufferDisplayDevice.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.9 $	$Date: 2014/12/29 02:33:17 $
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

#include "OpenGLPbufferDisplayDevice.h"
#include "Inform.h"
#include "utilities.h"
#include "config.h"   // VMD version strings etc

// Request a Pbuffer just larger than standard Ultra-HD "4K" resolution
#define DEF_PBUFFER_XRES 4096
#define DEF_PBUFFER_YRES 2400

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

////////////////////////// static helper functions.
static GLXFBConfig * vmd_get_fbconfig(glxpbufferdata *glxsrv, int *stereo, int *msamp, int *numsamples) {
  // we want double-buffered RGB with a Z buffer (possibly with stereo)
  int ns, dsize;
  int simplegraphics = 0;
  int disablestereo = 0;
  GLXFBConfig *fbc = NULL;
  int nfbc = 0;

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
    if (!simplegraphics && !disablestereo && (!fbc && nfbc < 1)) {
      // Stereo, multisample antialising, stencil buffer
      for (ns=maxaasamples; ns>1; ns--) {
        int conf[]  = {GLX_DRAWABLE_TYPE, GLX_PBUFFER_BIT,
                       GLX_DOUBLEBUFFER, 1,
                       GLX_RENDER_TYPE, GLX_RGBA_BIT,
                       GLX_DEPTH_SIZE, dsize, 
                       GLX_STEREO, 1,
                       GLX_STENCIL_SIZE, 1, 
                       GLX_SAMPLE_BUFFERS_ARB, 1, 
                       GLX_SAMPLES_ARB, ns, 
                       GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8,
                       None};

        fbc = glXChooseFBConfig(glxsrv->dpy, glxsrv->dpyScreen, conf, &nfbc);

        if (fbc && nfbc > 0) {
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
      if (!simplegraphics && (!fbc && nfbc < 1)) {
        int conf[]  = {GLX_DRAWABLE_TYPE, GLX_PBUFFER_BIT,
                       GLX_DOUBLEBUFFER, 1,
                       GLX_RENDER_TYPE, GLX_RGBA_BIT,
                       GLX_DEPTH_SIZE, dsize, 
                       GLX_STEREO, 1,
                       GLX_STENCIL_SIZE, 1, 
                       GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8,
                       None};

        fbc = glXChooseFBConfig(glxsrv->dpy, glxsrv->dpyScreen, conf, &nfbc);

        ns = 0; // no multisample antialiasing
        *numsamples = ns;
        *msamp = FALSE; 
        *stereo = TRUE; 
      }
    } 
#if defined(USEARBMULTISAMPLE) 
    else {
      // Try the OpenGL ARB multisample extension if available
      if (!simplegraphics && (!fbc && nfbc < 1)) {
        // Non-Stereo, multisample antialising, stencil buffer
        for (ns=maxaasamples; ns>1; ns--) {
          int conf[]  = {GLX_DRAWABLE_TYPE, GLX_PBUFFER_BIT,
                         GLX_DOUBLEBUFFER, 1,
                         GLX_RENDER_TYPE, GLX_RGBA_BIT,
                         GLX_DEPTH_SIZE, dsize, 
                         GLX_STENCIL_SIZE, 1, 
                         GLX_SAMPLE_BUFFERS_ARB, 1, 
                         GLX_SAMPLES_ARB, ns, 
                         GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8,
                         None};

          fbc = glXChooseFBConfig(glxsrv->dpy, glxsrv->dpyScreen, conf, &nfbc);
    
          if (fbc && nfbc > 0) {
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
  if (!simplegraphics && !disablestereo && (!fbc && nfbc < 1)) {
    int conf[]  = {GLX_DRAWABLE_TYPE, GLX_PBUFFER_BIT,
                   GLX_DOUBLEBUFFER, 1,
                   GLX_RENDER_TYPE, GLX_RGBA_BIT,
                   GLX_DEPTH_SIZE, 16, 
                   GLX_STEREO, 1,
                   GLX_STENCIL_SIZE, 1, 
                   GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8,
                   None};

    fbc = glXChooseFBConfig(glxsrv->dpy, glxsrv->dpyScreen, conf, &nfbc);
    
    ns = 0; // no multisample antialiasing
    *numsamples = ns;
    *msamp = FALSE; 
    *stereo = TRUE; 
  }

  // Mode for machines that provide stereo only in modes with 16-bit color.
  if (!simplegraphics && !disablestereo && (!fbc && nfbc < 1)) {
    int conf[]  = {GLX_DRAWABLE_TYPE, GLX_PBUFFER_BIT,
                   GLX_DOUBLEBUFFER, 1,
                   GLX_RENDER_TYPE, GLX_RGBA_BIT,
                   GLX_DEPTH_SIZE, 16, 
                   GLX_STEREO, 1,
                   GLX_STENCIL_SIZE, 1, 
                   GLX_RED_SIZE, 1, GLX_GREEN_SIZE, 1, GLX_BLUE_SIZE, 1,
                   None};

    fbc = glXChooseFBConfig(glxsrv->dpy, glxsrv->dpyScreen, conf, &nfbc);

    ns = 0; // no multisample antialiasing
    *numsamples = ns;
    *msamp = FALSE; 
    *stereo = TRUE; 
  }

  // Mode for machines that provide stereo only without a stencil buffer, 
  // and with reduced color precision.  Examples of this are the SGI Octane2
  // machines with V6 graphics, with recent IRIX patch levels.
  // Without this configuration attempt, these machines won't get stereo.
  if (!simplegraphics && !disablestereo && (!fbc && nfbc < 1)) {
    int conf[]  = {GLX_DRAWABLE_TYPE, GLX_PBUFFER_BIT,
                   GLX_DOUBLEBUFFER, 1,
                   GLX_RENDER_TYPE, GLX_RGBA_BIT,
                   GLX_DEPTH_SIZE, 16, 
                   GLX_STEREO, 1,
                   GLX_RED_SIZE, 1, GLX_GREEN_SIZE, 1, GLX_BLUE_SIZE, 1,
                   None};

    fbc = glXChooseFBConfig(glxsrv->dpy, glxsrv->dpyScreen, conf, &nfbc);

    ns = 0; // no multisample antialiasing
    *numsamples = ns;
    *msamp = FALSE; 
    *stereo = TRUE; 
  }

  // This mode gives up on trying to get stereo, and goes back to trying
  // to get a high quality non-stereo visual.
  if (!simplegraphics && (!fbc && nfbc < 1)) {
    int conf[]  = {GLX_DRAWABLE_TYPE, GLX_PBUFFER_BIT,
                   GLX_DOUBLEBUFFER, 1,
                   GLX_RENDER_TYPE, GLX_RGBA_BIT,
                   GLX_DEPTH_SIZE, 16, 
                   GLX_STENCIL_SIZE, 1, 
                   GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8,
                   None};

    fbc = glXChooseFBConfig(glxsrv->dpy, glxsrv->dpyScreen, conf, &nfbc);

    ns = 0; // no multisample antialiasing
    *numsamples = ns;
    *msamp = FALSE; 
    *stereo = FALSE;
  }
  
  // check if we have a TrueColor visual.
  if (!fbc && nfbc < 1) {
    // still no TrueColor.  Try again, with a very basic request ...
    // This is a catch all, we're desperate for any truecolor
    // visual by this point.  We've given up hoping for 24-bit
    // color or stereo by this time.
    int conf[]  = {GLX_DRAWABLE_TYPE, GLX_PBUFFER_BIT,
                   GLX_DOUBLEBUFFER, 1,
                   GLX_RENDER_TYPE, GLX_RGBA_BIT,
                   GLX_DEPTH_SIZE, 16, 
                   GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8,
                   None};

    fbc = glXChooseFBConfig(glxsrv->dpy, glxsrv->dpyScreen, conf, &nfbc);

    ns = 0; // no multisample antialiasing
    *numsamples = ns;
    *msamp = FALSE; 
    *stereo = FALSE;
  }

  if (!fbc && nfbc < 1) {
    // complete failure
    ns = 0; // no multisample antialiasing
    *numsamples = ns;
    *msamp = FALSE; 
    *stereo = FALSE;
  }

  // return NULL if the config count is less than one
  if (nfbc < 1)
    return NULL;

  return fbc;
}


/////////////////////////  constructor and destructor  

OpenGLPbufferDisplayDevice::OpenGLPbufferDisplayDevice()
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
  have_window = FALSE;
  screenX = screenY = 0;
}

int OpenGLPbufferDisplayDevice::init(int argc, char **argv, VMDApp *app, int *size, int *loc) {
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
OpenGLPbufferDisplayDevice::~OpenGLPbufferDisplayDevice(void) {
  if (have_window) {
    free_opengl_ctx(); // free display lists, textures, etc
 
    // close and delete windows, contexts, and display connections
    glXDestroyContext(glxsrv.dpy, glxsrv.cx);
    XCloseDisplay(glxsrv.dpy);
  }
}


/////////////////////////  protected nonvirtual routines  


// create a new window and set it's characteristics
Window OpenGLPbufferDisplayDevice::open_window(char *nm, int *size, int *loc,
					int argc, char** argv
) {
  char *dispname;
  if ((dispname = getenv("VMDGDISPLAY")) == NULL)
    dispname = getenv("DISPLAY");

  if(!(glxsrv.dpy = XOpenDisplay(dispname))) {
    msgErr << "Exiting due to X-Windows GLX/OpenGL Pbuffer creation failure." << sendmsg;
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

  // Find the best matching OpenGL framebuffer config for our purposes
  GLXFBConfig *fbc;
  fbc = vmd_get_fbconfig(&glxsrv, &ext->hasstereo, &ext->hasmultisample, &ext->nummultisamples);
  if (fbc == NULL) {
    msgErr << "No OpenGL Pbuffer configurations available" << sendmsg;
    return (Window)0;
  }

  // Create the OpenGL Pbuffer and associated GLX context
  const int pbconf[] = {GLX_PBUFFER_WIDTH, DEF_PBUFFER_XRES, 
                        GLX_PBUFFER_HEIGHT, DEF_PBUFFER_YRES,
                        GLX_LARGEST_PBUFFER, 1,
                        GLX_PRESERVED_CONTENTS, 1, 
                        None};
  GLXPbuffer PBuffer = glXCreatePbuffer(glxsrv.dpy, fbc[0], pbconf);
  glxsrv.cx = glXCreateNewContext(glxsrv.dpy, fbc[0], GLX_RGBA_TYPE, 0, GL_TRUE);
  if (PBuffer == 0 || glxsrv.cx == NULL) {
    msgErr << "A TrueColor OpenGL Pbuffer is required, but not available." << sendmsg;
    msgErr << "The X server is not capable of displaying double-buffered," << sendmsg;
    msgErr << "RGB images with a Z buffer.   Exiting ..." << sendmsg;
    XCloseDisplay(glxsrv.dpy);
    return (Window)0;
  }

  // set maximum allowable rendered image size for the Pbuffer
  // that was actually allocated, which may be smaller than we hoped...
  PbufferMaxXsz = DEF_PBUFFER_XRES;
  PbufferMaxYsz = DEF_PBUFFER_YRES;
  glXQueryDrawable(glxsrv.dpy, PBuffer, GLX_WIDTH,  &PbufferMaxXsz);
  glXQueryDrawable(glxsrv.dpy, PBuffer, GLX_HEIGHT, &PbufferMaxYsz);

  msgInfo << "OpenGL Pbuffer size: " 
          << PbufferMaxXsz << "x"
          << PbufferMaxYsz << sendmsg;

  // set default image size to incoming values, when possible.
  xSize = size[0];
  ySize = size[1];
  if (xSize < 0 || xSize > PbufferMaxXsz || 
      ySize < 0 || ySize > PbufferMaxYsz) {
    msgWarn << "Ignored out-of-range OpenGL Pbuffer image dimension request: " 
            << xSize << "x" << ySize 
            << " (max: " 
            << PbufferMaxXsz << "x" << PbufferMaxYsz << ")" << sendmsg;
    xSize = PbufferMaxXsz;
    ySize = PbufferMaxYsz;
  }

  // make the Pbuffer active
  glXMakeContextCurrent(glxsrv.dpy, PBuffer, PBuffer, glxsrv.cx);
  glXMakeCurrent(glxsrv.dpy, PBuffer, glxsrv.cx);

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

  // normal return: window was successfully created
  have_window = TRUE;

  // return window id
  return PBuffer;
}

/////////////////////////  public virtual routines  

//
// virtual routines for preparing to draw, drawing, and finishing drawing
//
void OpenGLPbufferDisplayDevice::do_resize_window(int w, int h) {
  if ((w > 0) && (w <= PbufferMaxXsz)) {
    xSize = w;
  } else {
    msgWarn << "Ignored out-of-range OpenGL Pbuffer X dimension request: " 
            << w << " (max: " << PbufferMaxXsz << ")" << sendmsg;
  }
  if ((h > 0) && (h <= PbufferMaxYsz)) {
    ySize = h;
  } else {
    msgWarn << "Ignored out-of-range OpenGL Pbuffer Y dimension request: " 
            << h << " (max: " << PbufferMaxYsz << ")" << sendmsg;
  }

  // force OpenGL frustum recalc
  reshape();

  // display now needs to be redrawn from scratch
  _needRedraw = 1;
}


// reshape the display after a shape change
void OpenGLPbufferDisplayDevice::reshape(void) {
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


unsigned char * OpenGLPbufferDisplayDevice::readpixels(int &xs, int &ys) {
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


// update after drawing
void OpenGLPbufferDisplayDevice::update(int do_update) {
  if (wiregl) {
    glFinish(); // force cluster to synchronize before buffer swap, 
                // this gives much better results than if the 
                // synchronization is done implicitly by glXSwapBuffers.
  }

  if(do_update)
    glXSwapBuffers(glxsrv.dpy, glxsrv.windowID);

  glDrawBuffer(GL_BACK);
}

