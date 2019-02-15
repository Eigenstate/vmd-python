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
 *	$RCSfile: FltkOpenGLDisplayDevice.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.69 $	$Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Subclass of DisplayDevice, this object has routines used by all the
 * different display devices that are OpenGL-specific.  Will render drawing
 * commands into a single X window.
 *
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "FltkOpenGLDisplayDevice.h"
#include "Inform.h"
#include "utilities.h"
#include "config.h"   // VMD version strings etc
#include "VMDApp.h"  
#include "FL/Fl.H"
#include "FL/Fl_Gl_Window.H"
#include "FL/forms.H"

/// Fl_Gl_Window subclass that implements an OpenGL rendering surface
/// for use by the FltkOpenGLDisplayDevice class.
class myglwindow : public Fl_Gl_Window {
  FltkOpenGLDisplayDevice *dispdev;
  VMDApp *app;     // cached VMDApp handle for use in drag-and-drop
  int dragpending; // flag indicating incoming PASTE event is drag-and-drop

public:
  myglwindow(int wx, int wy, int width, int height, const char *nm, 
    FltkOpenGLDisplayDevice *d, VMDApp *vmdapp) 
  : Fl_Gl_Window(wx, wy, width, height, nm), dispdev(d), app(vmdapp), dragpending(0) {
    // XXX this may not be reliable on MacOS X with recent revs of FLTK
    size_range(1,1,0,0); // resizable to full screen
  }


  int handle(int event) {
    // handle paste operations
    if (event == FL_PASTE) {
      // ignore paste operations that weren't due to drag-and-drop
      // since they could be any arbitrary data/text, and not just filenames.
      if (dragpending) {
        int len = Fl::event_length();

        // ignore zero-length paste events (why do these occur???)
        if (len > 0) {
          int numfiles, i;
          const char *lastc;
          int lasti;
          FileSpec spec;
          const char *ctext = Fl::event_text();
          char *filename = (char *) malloc((1 + len) * sizeof(char));

          for (lasti=0,lastc=ctext,numfiles=0,i=0; i<len; i++) {
            // parse out all but last filename, which doesn't have a CR
            if (ctext[i] == '\n') {
              memcpy(filename, lastc, (i-lasti)*sizeof(char));
              filename[i-lasti] = '\0';
  
              // attempt to load the file into a new molecule
              app->molecule_load(-1, filename, NULL, &spec);
  
              lasti=i+1;
              lastc=&ctext[lasti];
              numfiles++;
            }
  
            // special-case last filename, since there's no CR
            if (i == (len-1)) {
              memcpy(filename, lastc, (1+i-lasti)*sizeof(char));
              filename[1+i-lasti] = '\0';
  
              // attempt to load the file into a new molecule
              app->molecule_load(-1, filename, NULL, &spec);
              numfiles++;
            }
          }
  
          free(filename);
        }
  
        dragpending = 0; // no longer waiting for drag-and-drop paste
      }
  
      return 1; // indicate that we handled the paste operation
    }

    // handle drag-and-drop operations
    if (event == FL_DND_ENTER || event == FL_DND_DRAG) {
      return 1; // indicate that we want the drag-and-drop operation
    }
    if (event == FL_DND_RELEASE) {
      Fl::paste(*this);
      dragpending = 1; // flag to expect incoming paste due to DND operation
      return 1;
    }
    // end of cut-paste and drag-and-drop handling

    switch (event) {
      case FL_MOUSEWHEEL:
        dispdev->lastevent = event;
        dispdev->lastzdelta = Fl::event_dy();
        break;
      case FL_PUSH:
        dispdev->lastevent = event;
        dispdev->lastbtn = Fl::event_button();
        if (dispdev->lastbtn == FL_LEFT_MOUSE && Fl::event_state(FL_META)) {
          dispdev->lastbtn = FL_MIDDLE_MOUSE; 
        }
        break;
      case FL_DRAG:
        dispdev->lastevent = event;
        break;
      case FL_RELEASE:
        dispdev->lastevent = event;
        break;
#if (FL_MAJOR_VERSION >= 1) && (FL_MINOR_VERSION >= 1)
      case FL_KEYDOWN:
#else
      // This event code is superceded by FL_KEYDOWN in newer revs of FLTK
      case FL_KEYBOARD:
#endif
        dispdev->lastevent = event;
        dispdev->lastkeycode = Fl::event_key();
        dispdev->lastbtn = *Fl::event_text();
        break; 
      default:
        return Fl_Gl_Window::handle(event);
    }
    return 1;
  }
  void draw() {
    dispdev->reshape();
    dispdev->_needRedraw = 1;
    app->VMDupdate(VMD_IGNORE_EVENTS);
  }    
  // override the hide() method since we have no way of getting it back
  void hide() {
    if (fl_show_question("Really Quit?", 0))
      app->VMDexit("",0,0);
  }
};
 

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

// constructor ... open a window and set initial default values
FltkOpenGLDisplayDevice::FltkOpenGLDisplayDevice(int argc, char **argv, 
  VMDApp *vmdapp_p, int *size, int *loc)
    : OpenGLRenderer((char *) "VMD " VMDVERSION " OpenGL Display") {

  vmdapp = vmdapp_p; // save VMDApp handle for use by drag-and-drop handlers, 
                     // and GPU memory management routines

  // set up data possible before opening window
  stereoNames = glStereoNameStr;
  stereoModes = OPENGL_STEREO_MODES;

  // GLSL is only available on MacOS X 10.4 and later.
  renderNames = glRenderNameStr;
  renderModes = OPENGL_RENDER_MODES;

  cacheNames = glCacheNameStr;
  cacheModes = OPENGL_CACHE_MODES;

  // open the window
  int SX = 100, SY = 100, W, H;

  W = size[0];
  H = size[1];
  if (loc) {
    SX = loc[0];
    SY = loc[1];
  }
  window = new myglwindow(SX, SY, W, H, name, this, vmdapp_p);

  ext->hasstereo = FALSE;         // stereo is off initially
  ext->stereodrawforced = FALSE;  // stereo not forced initially
  ext->hasmultisample = FALSE;    // multisample is off initially

  int rc=0;
// FLTK stereo support only started working for MacOS X at around version 1.1.7
#if (FL_MAJOR_VERSION >= 1) && (((FL_MINOR_VERSION >= 1) && (FL_PATCH_VERSION >= 7)) || ((FL_MINOR_VERSION >= 1) && (FL_PATCH_VERSION >= 7)))
  // find an appropriate visual and colormap ...
  if (getenv("VMDPREFERSTEREO") != NULL) {
    // Stereo limps along with FLTK 1.1.7 on MacOS X
    rc = window->mode(FL_RGB8 | FL_DOUBLE | FL_STENCIL | FL_STEREO);
    ext->hasstereo = TRUE;
#if defined(__APPLE__)
    ext->stereodrawforced = TRUE; // forced draw in stereo all the time when on
#endif
  // FLTK multisample antialiasing still doesn't actually work in 
  // MacOS X as of FLTK 1.1.10...
#if !defined(__APPLE__)
  //  } else if (getenv("VMDPREFERMULTISAMPLE") != NULL) {
  } else if (rc != 0) {
    rc = window->mode(FL_RGB8 | FL_DOUBLE | FL_STENCIL | FL_MULTISAMPLE);
    ext->hasmultisample = TRUE; // FLTK only does SGI multisample, no ARB yet
#endif
  } else {
    rc = window->mode(FL_RGB8 | FL_DOUBLE | FL_STENCIL);
  }
#else
  // find an appropriate visual and colormap ...
  rc = window->mode(FL_RGB8 | FL_DOUBLE | FL_STENCIL);
#endif

  window->show();
  // (7) bind the rendering context to the window
  window->make_current();

  // (8) actually request the window to be displayed
  screenX = Fl::w();
  screenY = Fl::h();  

  // (9) configure the rendering properly
  setup_initial_opengl_state();  // setup initial OpenGL state
  
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
}

// destructor ... close the window
FltkOpenGLDisplayDevice::~FltkOpenGLDisplayDevice(void) {
  free_opengl_ctx(); // free display lists, textures, etc
  delete window;
}

/////////////////////////  public virtual routines  

//
// get the current state of the device's pointer (i.e. cursor if it has one)
//

// abs pos of cursor from lower-left corner of display
int FltkOpenGLDisplayDevice::x(void) {
  //Fl::check();
#if 1
  return Fl::event_x_root();  
#else
  int x, y;
  Fl::get_mouse(x, y);
  return x;  
#endif
}


// same, for y direction
int FltkOpenGLDisplayDevice::y(void) {
  //Fl::check();
#if 1
  return screenY - Fl::event_y_root();
#else
  int x, y;
  Fl::get_mouse(x, y);
  return screenY - y;  
#endif
}

// return the current state of the shift, control, and alt keys
int FltkOpenGLDisplayDevice::shift_state(void) {
  Fl::check();
   
  int retval = 0;
  int keymask = (int) Fl::event_state();
  if (keymask & FL_SHIFT)
    retval |= SHIFT;
  if (keymask & FL_CTRL)
    retval |= CONTROL;
  if (keymask & FL_ALT)
    retval |= ALT;
  return retval;
}

// return the spaceball state, if any
int FltkOpenGLDisplayDevice::spaceball(int *rx, int *ry, int *rz, int *tx, int *ty,
int *tz, int *buttons) {
  // not implemented yet
  return 0;
}


// set the Nth cursor shape as the current one.  If no arg given, the
// default shape (n=0) is used.
void FltkOpenGLDisplayDevice::set_cursor(int n) {
  switch (n) {
    default:
    case DisplayDevice::NORMAL_CURSOR: window->cursor(FL_CURSOR_ARROW); break;
    case DisplayDevice::TRANS_CURSOR: window->cursor(FL_CURSOR_MOVE); break;
    case DisplayDevice::SCALE_CURSOR: window->cursor(FL_CURSOR_WE); break;
    case DisplayDevice::PICK_CURSOR: window->cursor(FL_CURSOR_CROSS); break;
    case DisplayDevice::WAIT_CURSOR: window->cursor(FL_CURSOR_WAIT); break;
  }
}


//
// event handling routines
//

// read the next event ... returns an event type (one of the above ones),
// and a value.  Returns success, and sets arguments.
int FltkOpenGLDisplayDevice::read_event(long &retdev, long &retval) {
#if !defined(__APPLE__)
    // disabled on OSX to avoid problems with Tcl/Tk mishandling events.
    // XXX this code was previously being used on MacOS X for Intel, but
    //     it seems that it should match what we do on PowerPC so we 
    //     do the same in all MacOS X cases now.
    Fl::check();
#endif
  
  switch (lastevent) {
    case FL_MOUSEWHEEL:
      // XXX tests on the Mac show that FLTK is using a coordinate system 
      // backwards from what is used on Windows' zDelta value.
      if (lastzdelta < 0) {
        retdev = WIN_WHEELUP;
      } else {
        retdev = WIN_WHEELDOWN;
      }
      break;
    case FL_PUSH:
    case FL_DRAG:
    case FL_RELEASE:
      if (lastbtn == FL_LEFT_MOUSE) retdev = WIN_LEFT;
      else if (lastbtn == FL_MIDDLE_MOUSE) retdev = WIN_MIDDLE;
      else if (lastbtn == FL_RIGHT_MOUSE) retdev = WIN_RIGHT;
      else {
        //printf("unknown button: %d\n", lastbtn);
      }
      retval = (lastevent == FL_PUSH || lastevent == FL_DRAG);
      break;
  
#if (FL_MAJOR_VERSION >= 1) && (FL_MINOR_VERSION >= 1)
    case FL_KEYDOWN:
#else
    // This event code is superceded by FL_KEYDOWN in newer revs of FLTK
    case FL_KEYBOARD:
#endif
      // check function keys first
      if (lastkeycode >= FL_F && lastkeycode <= FL_F_Last) {
        retdev = (lastkeycode - FL_F) + ((int) WIN_KBD_F1);
      } else {
        switch(lastkeycode) {
          case FL_Escape:      retdev = WIN_KBD_ESCAPE;    break;
          case FL_Up:          retdev = WIN_KBD_UP;        break;
          case FL_Down:        retdev = WIN_KBD_DOWN;      break;
          case FL_Left:        retdev = WIN_KBD_LEFT;      break;
          case FL_Right:       retdev = WIN_KBD_RIGHT;     break;
          case FL_Page_Up:     retdev = WIN_KBD_PAGE_UP;   break;
          case FL_Page_Down:   retdev = WIN_KBD_PAGE_UP;   break;
          case FL_Home:        retdev = WIN_KBD_HOME;      break;
          case FL_End:         retdev = WIN_KBD_END;       break;
          case FL_Insert:      retdev = WIN_KBD_INSERT;    break;
          case FL_Delete:      retdev = WIN_KBD_DELETE;    break;

          default:
            retdev = WIN_KBD;
            break;
        }
      }
      retval = lastbtn;
      break;

    default:
      return 0;
  }
  lastevent = 0;
  return 1;
}

//
// virtual routines for preparing to draw, drawing, and finishing drawing
//

// reshape the display after a shape change
void FltkOpenGLDisplayDevice::reshape(void) {

  xSize = window->w();
  ySize = window->h();
  xOrig = window->x();
  yOrig = screenY - window->y() - ySize;

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

unsigned char * FltkOpenGLDisplayDevice::readpixels_rgb3u(int &x, int &y) {
  unsigned char * img;

  x = xSize;
  y = ySize;

  if ((img = (unsigned char *) malloc(x * y * 3)) != NULL) {
#if !defined(WIREGL) 
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, x, y, GL_RGB, GL_UNSIGNED_BYTE, img);
#endif
  } else {
    x = 0;
    y = 0;
  } 

  return img; 
}

unsigned char * FltkOpenGLDisplayDevice::readpixels_rgba4u(int &x, int &y) {
  unsigned char * img;

  x = xSize;
  y = ySize;

  if ((img = (unsigned char *) malloc(x * y * 4)) != NULL) {
#if !defined(WIREGL) 
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, x, y, GL_RGBA, GL_UNSIGNED_BYTE, img);
#endif
  } else {
    x = 0;
    y = 0;
  } 

  return img; 
}


// update after drawing
void FltkOpenGLDisplayDevice::update(int do_update) {
  if(do_update)
    window->swap_buffers();

  glDrawBuffer(GL_BACK);
}

void FltkOpenGLDisplayDevice::do_resize_window(int w, int h) {
  window->size(w, h);
  // XXX this may not be reliable on MacOS X with recent revs of FLTK
  window->size_range(1,1,0,0); // resizable to full screen
}

void FltkOpenGLDisplayDevice::do_reposition_window(int xpos, int ypos) {
  window->position(xpos, ypos);
  // XXX this may not be reliable on MacOS X with recent revs of FLTK
  window->size_range(1,1,0,0); // resizable to full screen
}


