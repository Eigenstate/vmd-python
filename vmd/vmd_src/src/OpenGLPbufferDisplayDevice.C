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
 *	$RCSfile: OpenGLPbufferDisplayDevice.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.27 $	$Date: 2019/01/17 21:21:00 $
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

#if defined(VMDGLXPBUFFER)
#include <GL/glx.h>
#include <X11/Xlib.h>
#endif

#include "OpenGLPbufferDisplayDevice.h"
#include "Inform.h"
#include "utilities.h"
#include "config.h"   // VMD version strings etc
#include "VMDApp.h"
#include "VideoStream.h" 

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


//
// GLX-related static helper functions
//
#if defined(VMDGLXPBUFFER)

// determine if all of the ARB multisample extension routines are available
// when using GLX APIs
#if defined(GL_ARB_multisample) && defined(GLX_SAMPLES_ARB) && defined(GLX_SAMPLE_BUFFERS_ARB)
#define USEARBMULTISAMPLE 1
#endif

static GLXFBConfig * vmd_get_glx_fbconfig(glxpbufferdata *glxsrv, int *stereo, int *msamp, int *numsamples) {
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

#endif


//
// EGL-related static helper functions
//
#if defined(VMDEGLPBUFFER)

// determine if all of the ARB multisample extension routines are available
// when using EGL APIs
#if defined(GL_ARB_multisample)
#define USEARBMULTISAMPLE 1
#endif

static int vmd_get_egl_fbconfig(eglpbufferdata *eglsrv, int *stereo, int *msamp, int *numsamples) {
  // we want double-buffered RGB with a Z buffer (possibly with stereo)
  int ns, dsize;
  int simplegraphics = 0;
  // XXX standard EGL doesn't support stereo visuals at present,
  // int disablestereo = 0;
  int fbc = 0;
  int nfbc = 0;

  *numsamples = 0;
  *msamp = FALSE; 
  *stereo = FALSE;

  if (getenv("VMDSIMPLEGRAPHICS")) {
    simplegraphics = 1;
  }

  // XXX standard EGL doesn't support stereo visuals at present,
  // if (getenv("VMDDISABLESTEREO")) {
  //   disablestereo = 1;
  // } 

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
    //
    // XXX standard EGL doesn't support stereo visuals at present,
    // so the tsts we would normally do for stereo are omitted here
    //

// Try the OpenGL ARB multisample extension if available
#if defined(USEARBMULTISAMPLE) 
    if (!simplegraphics && (!fbc && nfbc < 1)) {
      // Non-Stereo, multisample antialising, stencil buffer
      for (ns=maxaasamples; ns>1; ns--) {
        int conf[]  = {EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                       EGL_DEPTH_SIZE, dsize, 
                       EGL_STENCIL_SIZE, 1, 
                       EGL_SAMPLE_BUFFERS, 1, 
                       EGL_SAMPLES, ns, 
                       EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8,
                       EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_NONE };

        if (eglChooseConfig(eglsrv->dpy, conf, &eglsrv->conf, 1, &nfbc) == EGL_TRUE) {
          if (nfbc > 0) {
            fbc=1; // flag that we got a config
            *numsamples = ns;
            *msamp = TRUE;
            *stereo = FALSE; // XXX EGL doesn't support stereo at present
            break; // exit loop if we got a good visual
          } 
        }
      }
    }
#endif

    //
    // XXX standard EGL doesn't support stereo visuals at present,
    // so the tsts we would normally do for stereo are omitted here
    //
  } // end of loop over a wide range of depth buffer sizes

  // Ideally we should fall back to accumulation buffer based antialiasing
  // here, but not currently implemented.  At this point no multisample
  // antialiasing mode is available.

  //
  // XXX standard EGL doesn't support stereo visuals at present,
  // so the tests we would normally do for stereo are omitted here
  //

  // This mode gives up on trying to get stereo, and goes back to trying
  // to get a high quality non-stereo visual.
  if (!simplegraphics && (!fbc && nfbc < 1)) {
    int conf[]  = {EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                   EGL_DEPTH_SIZE, 16, 
                   EGL_STENCIL_SIZE, 1, 
                   EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8,
                   EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_NONE };

    if (eglChooseConfig(eglsrv->dpy, conf, &eglsrv->conf, 1, &nfbc) == EGL_TRUE) {
      if (nfbc > 0) {
        fbc=1; // flag that we got a config
        ns = 0; // no multisample antialiasing
        *numsamples = ns;
        *msamp = FALSE; 
        *stereo = FALSE; // XXX EGL doesn't support stereo at present
      } 
    }
  }
  
  // check if we have a TrueColor visual.
  if (!fbc && nfbc < 1) {
    // still no TrueColor.  Try again, with a very basic request ...
    // This is a catch all, we're desperate for any truecolor
    // visual by this point.  We've given up hoping for 24-bit
    // color or stereo by this time.
    int conf[]  = {EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                   EGL_DEPTH_SIZE, 16, 
                   EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8,
                   EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_NONE };

    if (eglChooseConfig(eglsrv->dpy, conf, &eglsrv->conf, 1, &nfbc) == EGL_TRUE) {
      if (nfbc > 0) {
        fbc=1; // flag that we got a config
        ns = 0; // no multisample antialiasing
        *numsamples = ns;
        *msamp = FALSE; 
        *stereo = FALSE; // XXX EGL doesn't support stereo at present
      } 
    }
  }

  if (!fbc && nfbc < 1) {
    // complete failure
    ns = 0; // no multisample antialiasing
    *numsamples = ns;
    *msamp = FALSE; 
    *stereo = FALSE; // XXX EGL doesn't support stereo at present
  }

  // return false if the config count is less than one
  if (nfbc < 1)
    return 0;

  return 1;
}

#endif









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

#if defined(VMDEGLPBUFFER)
  memset(&eglsrv, 0, sizeof(eglsrv));
#endif
#if defined(VMDGLXPBUFFER)
  memset(&glxsrv, 0, sizeof(glxsrv));
  glxsrv.dpy = NULL;
  glxsrv.dpyScreen = 0;
#endif

  have_window = FALSE;
  screenX = screenY = 0;
}


int OpenGLPbufferDisplayDevice::init(int argc, char **argv, VMDApp *app, int *size, int *loc) {
  vmdapp = app; // save VMDApp handle for use by drag-and-drop handlers
                // and GPU memory management routines

  // Try and create a pbuffer using GLX first, and if that doesn't work or
  // the code is compiled with EGL only, then we fall back to EGL.

#if defined(VMDGLXPBUFFER)
  int haveglxwin = 0;
  haveglxwin = glx_open_window(name, size, loc, argc, argv);
  if (haveglxwin)
    msgInfo << "Created GLX OpenGL Pbuffer for off-screen rendering" << sendmsg;
#endif

#if defined(VMDEGLPBUFFER)
  int haveeglwin = 0;
#if defined(VMDGLXPBUFFER) 
  if (!haveglxwin)
#endif
    haveeglwin = egl_open_window(name, size, loc, argc, argv);

  if (haveeglwin)
    msgInfo << "Created EGL OpenGL Pbuffer for off-screen rendering" << sendmsg;
#endif

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
#if defined(VMDEGLPBUFFER)
#endif
#if defined(VMDGLXPBUFFER)
    if (glxsrv.cx) {
      glXDestroyContext(glxsrv.dpy, glxsrv.cx);
      XCloseDisplay(glxsrv.dpy);
    }
#endif

  }
}


/////////////////////////  protected nonvirtual routines  


#if defined(VMDEGLPBUFFER)

// static helper fctn to convert error state into a human readable mesg string
static const char* vmd_get_egl_errorstring(void) {    
  EGLint errcode = eglGetError();
  switch (errcode) {
    case EGL_SUCCESS:              return "No error"; break;
    case EGL_NOT_INITIALIZED:      return "EGL not initialized"; break;
    case EGL_BAD_ACCESS:           return "EGL bad access"; break;
    case EGL_BAD_ALLOC:            return "EGL bad alloc"; break;
    case EGL_BAD_ATTRIBUTE:        return "EGL bad attribute"; break;
    case EGL_BAD_CONTEXT:          return "EGL bad context"; break;
    case EGL_BAD_CONFIG:           return "EGL bad config"; break;
    case EGL_BAD_CURRENT_SURFACE:  return "EGL bad cur context"; break;
    case EGL_BAD_DISPLAY:          return "EGL bad display"; break;
    case EGL_BAD_SURFACE:          return "EGL bad surface"; break;
    case EGL_BAD_MATCH:            return "EGL bad match"; break;
    case EGL_BAD_PARAMETER:        return "EGL bad parameter"; break;
    case EGL_BAD_NATIVE_PIXMAP:    return "EGL bad native pixmap"; break;
    case EGL_BAD_NATIVE_WINDOW:    return "EGL bad native window"; break;
    case EGL_CONTEXT_LOST:         return "EGL context lost"; break;
    default:
      return "Unrecognized EGL error"; break;
  }
}


// create a new window and set it's characteristics
int OpenGLPbufferDisplayDevice::egl_open_window(char *nm, int *size, int *loc,
                                                int argc, char** argv) {
  // Clear display before we try and attach
  eglsrv.dpy = EGL_NO_DISPLAY;
  eglsrv.numdevices = 0;
  eglsrv.devindex = 0;

#if defined(EGL_EXT_platform_base) && (EGL_EGLEXT_VERSION >= 20160000)
  // 
  // enumerate all GPUs and bind to the one that matches our MPI node rank
  // 

  // load the function pointers for the device,platform extensions            
  PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT;
  PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT;
  eglQueryDevicesEXT = (PFNEGLQUERYDEVICESEXTPROC) eglGetProcAddress("eglQueryDevicesEXT");     
  eglGetPlatformDisplayEXT = (PFNEGLGETPLATFORMDISPLAYEXTPROC) eglGetProcAddress("eglGetPlatformDisplayEXT");     

  // try and bind to a non-default display if we have all required fctn ptrs
  if (eglQueryDevicesEXT != NULL && eglGetPlatformDisplayEXT != NULL) {
    static const int MAX_DEVICES = 16;
    EGLDeviceEXT devicelist[MAX_DEVICES];
    eglQueryDevicesEXT(MAX_DEVICES, devicelist, &eglsrv.numdevices);

    // compute EGL device index to use via round-robin assignment
    eglsrv.devindex = vmdapp->noderank % eglsrv.numdevices;
    eglsrv.dpy = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, devicelist[eglsrv.devindex], 0);
  }
#endif

  // emit console message with node rank and bound EGL device
  if (eglsrv.dpy != EGL_NO_DISPLAY) {
    printf("Info) EGL: node[%d] bound to display[%d], %d %s total\n", 
            vmdapp->noderank, eglsrv.devindex, eglsrv.numdevices, 
            (eglsrv.numdevices == 1) ? "display" : "displays");
  } else {
    // use default display
    eglsrv.dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  
    if (eglsrv.dpy != EGL_NO_DISPLAY)
      msgInfo << "EGL context bound to default display." << sendmsg;
  }

  // if we still have no display, we have no choice but to abort
  if (eglsrv.dpy == EGL_NO_DISPLAY) {
    msgErr << "Exiting due to EGL Pbuffer creation failure." << sendmsg;
    return 0; 
  }

  EGLint eglmaj, eglmin;
  if (eglInitialize(eglsrv.dpy, &eglmaj, &eglmin) == EGL_FALSE) {
    msgErr << "Exiting due to EGL initialization failure." << sendmsg;
    msgErr << "  " << vmd_get_egl_errorstring() << sendmsg;
    return 0; 
  } 
  msgInfo << "EGL version " << eglmaj << "." << eglmin << sendmsg;

  int fbc = vmd_get_egl_fbconfig(&eglsrv, &ext->hasstereo, &ext->hasmultisample, &ext->nummultisamples);
  if (!fbc) {
    msgErr << "Exiting due to EGL config failure." << sendmsg;
    msgErr << "  " << vmd_get_egl_errorstring() << sendmsg;
    return 0; 
  }

  EGLint vid;
  if (eglGetConfigAttrib(eglsrv.dpy, eglsrv.conf, EGL_NATIVE_VISUAL_ID, &vid) == EGL_FALSE) {
    msgErr << "Exiting due to eglGetConfigAttrib() failure." << sendmsg;
    msgErr << "  " << vmd_get_egl_errorstring() << sendmsg;
    return 0; 
  }

  // bind to OpenGL API since some implementations don't do this by default
  if (eglBindAPI(EGL_OPENGL_API) == EGL_FALSE) {
    msgErr << "Exiting due to EGL OpenGL binding failure." << sendmsg;
    msgErr << "  " << vmd_get_egl_errorstring() << sendmsg;
    return 0; 
  }

  eglsrv.ctx = eglCreateContext(eglsrv.dpy, eglsrv.conf, EGL_NO_CONTEXT, NULL);

  // create the size we ask for, or fail
  static const EGLint pbuffer_fixedsz_attribs[] = {
    EGL_WIDTH, DEF_PBUFFER_XRES,
    EGL_HEIGHT, DEF_PBUFFER_YRES,
    EGL_NONE,
  };

  // if we don't get the size we ask for, try for the max size and
  // then tell VMD what it ended up being.  We probably need a sanity
  // check in the case we get a really lame small size buffer back.
  static const EGLint pbuffer_defsz_attribs[] = {
    EGL_WIDTH, DEF_PBUFFER_XRES,
    EGL_HEIGHT, DEF_PBUFFER_YRES,
    EGL_LARGEST_PBUFFER, EGL_TRUE,
    EGL_NONE,
  };

  // Try for a HUGE size and then tell VMD what it ended up being.  
  static const EGLint pbuffer_maxsz_attribs[] = {
    EGL_WIDTH, 10000,
    EGL_HEIGHT, 10000,
    EGL_LARGEST_PBUFFER, EGL_TRUE,
    EGL_NONE,
  };

  // Demonstrate bugs in implementations that don't do the right thing
  // with the EGL_LARGEST_BUFFER parameter
  static const EGLint pbuffer_hugesz_attribs[] = {
    EGL_WIDTH, 30000,
    EGL_HEIGHT, 30000,
    EGL_LARGEST_PBUFFER, EGL_TRUE,
    EGL_NONE,
  };

  EGLint const *pbuffer_attrs = pbuffer_defsz_attribs;
  if (getenv("VMDEGLUSEFIXEDSZ") != NULL) {
    pbuffer_attrs = pbuffer_fixedsz_attribs;
  }
  if (getenv("VMDEGLUSEMAXSZ") != NULL) {
    pbuffer_attrs = pbuffer_maxsz_attribs;
  }
  if (getenv("VMDEGLUSEHUGESZ") != NULL) {
    pbuffer_attrs = pbuffer_hugesz_attribs;
  }

  eglsrv.surf = eglCreatePbufferSurface(eglsrv.dpy, eglsrv.conf, pbuffer_attrs);
  if (eglsrv.surf == EGL_NO_SURFACE) {
    msgErr << "Exiting due to EGL Pbuffer surface creation failure." << sendmsg;
    msgErr << "  " << vmd_get_egl_errorstring() << sendmsg;
    return 0; 
  }

  EGLint surface_xsize, surface_ysize;
  if ((eglQuerySurface(eglsrv.dpy, eglsrv.surf, EGL_WIDTH, &surface_xsize) != EGL_TRUE) ||
      (eglQuerySurface(eglsrv.dpy, eglsrv.surf, EGL_HEIGHT, &surface_ysize) != EGL_TRUE)) {
    msgErr << "Exiting due to EGL Pbuffer surface dimensions query failure." << sendmsg;
    msgErr << "  " << vmd_get_egl_errorstring() << sendmsg;
    return 0; 
  }

  // set maximum allowable rendered image size for the Pbuffer
  // that was actually allocated, which may be smaller than we hoped...
  PbufferMaxXsz = surface_xsize;
  PbufferMaxYsz = surface_ysize;

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

  EGLBoolean ctxstatus = EGL_FALSE;
  ctxstatus = eglMakeCurrent(eglsrv.dpy, eglsrv.surf, eglsrv.surf, eglsrv.ctx);
  if (ctxstatus != EGL_TRUE) {
    msgErr << "Exiting due to EGL failure during eglMakeCurrent()." << sendmsg;
    msgErr << "  " << vmd_get_egl_errorstring() << sendmsg;
    return 0;
  }


  EGLint Context_RendererType=0;
  eglQueryContext(eglsrv.dpy, eglsrv.ctx, EGL_CONTEXT_CLIENT_TYPE, &Context_RendererType);

#if 0
  const char *glstring="uninitialized";
  char buf[1024];
  switch (Context_RendererType) {
    case    EGL_OPENGL_API: glstring = "OpenGL"; break;
    case EGL_OPENGL_ES_API: glstring = "OpenGL ES"; break;
    case    EGL_OPENVG_API: glstring = "OpenVG???"; break;
    default:
      sprintf(buf, "Unknown API: %x", Context_RendererType);
      glstring=buf;
      break;
  }
  msgInfo << "EGL_CONTEXT_CLIENT_TYPE: %s\n", glstring);
#endif

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

  return 1; // return success
}
#endif



#if defined(VMDGLXPBUFFER)

// create a new window and set it's characteristics
int OpenGLPbufferDisplayDevice::glx_open_window(char *nm, int *size, int *loc,
                                                int argc, char** argv) {
  char *dispname;
  if ((dispname = getenv("VMDGDISPLAY")) == NULL)
    dispname = getenv("DISPLAY");

  if(!(glxsrv.dpy = XOpenDisplay(dispname))) {
    msgErr << "Exiting due to X-Windows GLX/OpenGL Pbuffer creation failure." << sendmsg;
    if (dispname != NULL) {
      msgErr << "Failed to open display: " << dispname << sendmsg;
    }
    return 0; 
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
    return 0;
  }

  ext->hasstereo = TRUE;         // stereo on until we find out otherwise.
  ext->stereodrawforced = FALSE; // no need for force stereo draws initially
  ext->hasmultisample = TRUE;    // multisample on until we find out otherwise.

  // Find the best matching OpenGL framebuffer config for our purposes
  GLXFBConfig *fbc;
  fbc = vmd_get_glx_fbconfig(&glxsrv, &ext->hasstereo, &ext->hasmultisample, &ext->nummultisamples);
  if (fbc == NULL) {
    msgErr << "No OpenGL Pbuffer configurations available" << sendmsg;
    return 0;
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
    return 0;
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

  // set window id
  glxsrv.windowID = PBuffer;

  return 1; // return success
}
#endif


/////////////////////////  public virtual routines  

//
// virtual routines for preparing to draw, drawing, and finishing drawing
//
void OpenGLPbufferDisplayDevice::do_resize_window(int w, int h) {
  if ((w > 0) && (w <= ((int) PbufferMaxXsz))) {
    xSize = w;
  } else {
    msgWarn << "Ignored out-of-range OpenGL Pbuffer X dimension request: " 
            << w << " (max: " << PbufferMaxXsz << ")" << sendmsg;
  }
  if ((h > 0) && (h <= ((int) PbufferMaxYsz))) {
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


unsigned char * OpenGLPbufferDisplayDevice::readpixels_rgb3u(int &xs, int &ys) {
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

unsigned char * OpenGLPbufferDisplayDevice::readpixels_rgba4u(int &xs, int &ys) {
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


// update after drawing
void OpenGLPbufferDisplayDevice::update(int do_update) {
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

#if defined(VMDEGLPBUFFER)
#if 1
  if (do_update)
    eglSwapBuffers(eglsrv.dpy, eglsrv.surf);
#else
  EGLBoolean eglrc = EGL_TRUE;
  if (do_update)
    eglrc = eglSwapBuffers(eglsrv.dpy, eglsrv.surf);

  if (eglrc != EGL_TRUE) {
    printf("eglSwapBuffers(): EGLrc: %d\n", eglrc);
  }
#endif
#endif

#if defined(VMDGLXPBUFFER)
  if (do_update)
    glXSwapBuffers(glxsrv.dpy, glxsrv.windowID);
#endif

  glDrawBuffer(GL_BACK);
}

