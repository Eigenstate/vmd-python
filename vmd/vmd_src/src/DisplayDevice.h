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
 *	$RCSfile: DisplayDevice.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.143 $	$Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************/
#ifndef DISPLAYDEVICE_H
#define DISPLAYDEVICE_H

#include <stdio.h>
#include <math.h>

#include "Matrix4.h"
#include "Stack.h"
#include "ResizeArray.h"
#include "utilities.h"

class VMDApp;
class VMDDisplayList;

/// Abstract base class for objects which
/// can process a list of drawing commands and render the drawing
/// to some device (screen, file, preprocessing script, etc.),
/// and provide mouse and keyboard events.
class DisplayDevice {
public:
  VMDApp *vmdapp; ///< VMDApp object ptr, for drag-and-drop handlers,
                  ///< and to allow subclasses to access to
                  ///< GPU management APIs, e.g. to force-free large 
                  ///< memory buffers in GPU global memory prior to 
                  ///< launching GPU ray tracing, etc.

  char *name; ///< name of this display device

  /// enum for left or right eye
  enum DisplayEye { NOSTEREO, LEFTEYE, RIGHTEYE };

  /// enum for the mouse buttons, function keys, and special meta keys
  enum Buttons { B_LEFT, B_MIDDLE, B_RIGHT, B2_LEFT, B2_MIDDLE, B2_RIGHT,
                 B_F1, B_F2, B_F3, B_F4,  B_F5,  B_F6,
                 B_F7, B_F8, B_F9, B_F10, B_F11, B_F12,
                 B_ESC, TOTAL_BUTTONS };
  
  /// enum for window events
  enum EventCodes  { WIN_REDRAW, WIN_LEFT, WIN_MIDDLE, WIN_RIGHT,
                     WIN_WHEELUP, WIN_WHEELDOWN, WIN_MOUSEX, WIN_MOUSEY, 
                     WIN_KBD, 
                     WIN_KBD_ESCAPE,
                     WIN_KBD_UP, 
                     WIN_KBD_DOWN, 
                     WIN_KBD_LEFT, 
                     WIN_KBD_RIGHT, 
                     WIN_KBD_PAGE_UP, 
                     WIN_KBD_PAGE_DOWN, 
                     WIN_KBD_HOME, 
                     WIN_KBD_END, 
                     WIN_KBD_INSERT,
                     WIN_KBD_DELETE,
                     WIN_KBD_F1,  WIN_KBD_F2,  WIN_KBD_F3,  WIN_KBD_F4,
                     WIN_KBD_F5,  WIN_KBD_F6,  WIN_KBD_F7,  WIN_KBD_F8,
                     WIN_KBD_F9,  WIN_KBD_F10, WIN_KBD_F11, WIN_KBD_F12,
                     WIN_NOEVENT };

  /// enum for cursor types
  enum CursorCodes { NORMAL_CURSOR, TRANS_CURSOR, SCALE_CURSOR, 
                     PICK_CURSOR, WAIT_CURSOR };

  /// display state has changed, requiring a redraw
  int needRedraw(void) const { return _needRedraw; }
  int _needRedraw; ///< XXX public so FltkOpenGLDisplayDevice can access it

  int num_display_processes;   ///< number of CAVE/FreeVR display processes
  int renderer_process;        ///< true of we're a rendering process

  /// return the number of display processes
  virtual int get_num_processes()   { return num_display_processes; } 
  virtual int is_renderer_process() { return renderer_process; }
 
protected:
  //@{
  /// Capability and state flags for antialiasing, depth cueing,
  /// backface culling, shadows, ambient occlusion, and depth of field
  int aaAvailable, cueingAvailable, cullingAvailable;
  int aaEnabled,   cueingEnabled,   cullingEnabled;
  int aoEnabled,   shadowEnabled,   dofEnabled;
  int aaPrevious;
  //@}

  /// depth-sorting flag for tranparent objects?  default is FALSE
  int my_depth_sort;

  /// cached pointer to color data; must be in shared memory for Cave-like
  //displays.
  const float *colorData;
  virtual void do_use_colors() {}

  /// Depth cueing mode enumerations
  enum CueMode { CUE_LINEAR, CUE_EXP, CUE_EXP2, NUM_CUE_MODES };
  CueMode cueMode;              ///< fog/cueing mode 
  float cueStart, cueEnd;       ///< fog start/end, used for linear fog/cueing
  float cueDensity;             ///< fog density, used for exp/exp2 fog/cueing

  /// Shadow rendering mode enumerations
  enum ShadowMode { SHADOWS_OFF, SHADOWS_ON };

  // ambient occlusion
  float aoAmbient;              ///< AO ambient lighting factor
  float aoDirect;               ///< AO direct lighting rescaling factor

  // depth of field
  float dofFNumber;             ///< DoF aperture
  float dofFocalDist;           ///< DoF focal plane distance

public:
  /// enum for the possible projection modes 
  enum Projection {PERSPECTIVE, ORTHOGRAPHIC, NUM_PROJECTIONS};
  Projection projection() const { return my_projection; }

  /// position and size of device, in 'pixels'
  long xOrig, yOrig, xSize, ySize;

protected:
  /// current transformation matrix for the display (NOT the projection matrix)
  Stack<Matrix4> transMat;

  /// display background clearing mode
  int backgroundmode; ///< 0=normal, 1=gradient

  //@{
  /// drawing characteristics ... line style, sphere resolution, etc.
  int lineStyle, lineWidth;
  int sphereRes, sphereMode;
  int cylinderRes;
  //@}


  //
  // eye position, clipping planes, and viewing geometry data
  //
  float eyePos[3];              ///< current location of the viewer's eye
  DisplayEye whichEye;          ///< the eye we are currently drawing to.
  float nearClip, farClip;      ///< dist from eye to near and far clip plane

  float vSize;                  ///< vertical size of 'screen'
  float zDist;                  ///< distance to 'screen' relative to origin

  float Aspect;	                ///< current window/image aspect ratio
                                ///< This is the width of the generated image
                                ///< (in pixels) divided by its height.
                                ///< NOT the aspect ratio of the pixels on 
                                ///< the target display device.

  //@{
  /// distances to near frustum base, which defines the view volume
  float cpUp, cpDown, cpLeft, cpRight;	
  //@}


  // stereo display data
  int inStereo;			///< current stereo mode (0 = non-stereo)
  int stereoSwap;               ///< whether left/right eyes are swapped
  int stereoModes;		///< total number of stereo modes (inc mono)
  const char **stereoNames;     ///< pointer to stereo mode names

  // display list caching data
  int cacheMode;                ///< current caching mode
  int cacheModes;               ///< total number of caching modes
  const char **cacheNames;      ///< pointer to rendering mode names

  // rendering mode data
  int renderMode;               ///< current rendering mode
  int renderModes;              ///< total number of rendering modes
  const char **renderNames;     ///< pointer to rendering mode names

  // display camera/eye parameters
  float eyeSep;			///< distance between eyes for stereo display
  float eyeDist;		///< distance from eye to focal point
  float eyeDir[3];		///< direction viewer is looking
  float upDir[3];		///< direction which is 'up'
  float eyeSepDir[3];		///< vector from eye position to right eye
  				///< magnitude is 1/2 eyeSep 

public:  
  /// virtual routines to deal with light sources, return success/failure
  virtual int do_define_light(int, float *, float *) { return TRUE; } 
  virtual int do_activate_light(int, int) { return TRUE; } 

#if 0
  // XXX need to implement advanced lights still
#endif

  /// Use this for colors
  void use_colors(const float *c) {
    colorData = c;
    do_use_colors();
  }

protected:
  int mouseX;                   ///< Mouse X position
  int mouseY;                   ///< mouse Y position

  /**
   * calculate the position of the near frustum plane, based on curr values
   * of Aspect, vSize, zDist, nearClip and eyePosition
   */
  void calc_frustum(void);

  /**
   * calculate eyeSepDir, based on up vector and look vector
   * eyeSepDir = 1/2 * eyeSep * (lookdir x updir) / mag(lookdir x updir)
   */
  void calc_eyedir(void);

  /**
   * Do device-specific resizing or positioning of window
   */
  virtual void do_resize_window(int w, int h);
  virtual void do_reposition_window(int xpos, int ypos) {}

  /// total size of the screen, in pixels ... MUST BE SET BY DERIVED CLASS
  int screenX, screenY;

  /// Find transformations corresponding to the periodic boundary conditions
  /// specified in the given display list.  Append these transformations to
  /// the given array.
  void find_pbc_images(const VMDDisplayList *, ResizeArray<Matrix4> &);

  /// Find the periodic cells that are turned on by the display list.  Return
  /// in the ResizeArray with 3 values per cell, in the form na nb nc, where
  /// nx is the number of times to apply transform x to get to the cell.
  void find_pbc_cells(const VMDDisplayList *, ResizeArray<int> &);

  /// Find transformations corresponding to the list of active molecule  
  /// instances specified in the given display list.  
  /// Append these transformations to the given array.
  void find_instance_images(const VMDDisplayList *, ResizeArray<Matrix4> &);

public:
  DisplayDevice(const char *);
  virtual ~DisplayDevice(void);
  
  /// copies over all relevant properties from one DisplayDevice to another
  DisplayDevice& operator=(DisplayDevice &);

  /// do actual window construction here.  Return true if the window was 
  /// successfully created; false if not.
  virtual int init(int argc, char **argv, VMDApp *app, int *size, int *loc) {
    if (size != NULL)
      resize_window(size[0], size[1]);

    vmdapp=app; // set VMDApp ptr to allow DisplayDevice control over use of
                // GPU memory resources during ray tracing, etc.

    return TRUE;
  }

  /// Does this display device support GUI's?  The default stub display 
  /// does not.
  virtual int supports_gui() { return FALSE; }


  //
  // event handling routines
  //

  /**
    * queue the standard events (need only be called once ... but this is
    * not done automatically by the window because it may not be necessary or
    * even wanted)
    */
  virtual void queue_events(void);

  /** 
    * read the next event ... returns an event type (one of the above ones),
    * and a value.  Returns success, and sets arguments.
    * NOTE: THIS SHOULD NOT BLOCK ... IT SHOUULD RETURN FALSE IF NO EVENT TO
    * READ.
    */
  virtual int read_event(long &, long &);

  // get the current state of the device's pointer (i.e. cursor if it has one)
  virtual int x(void); /// absolute position of cursor from lower-left corner
  virtual int y(void); /// absolute position of cursor from lower-left corner

  /// Mouse and keyboard shift state, joystick/sball aux key/button state  
  enum { 
    SHIFT = 1, 
    CONTROL = 2, 
    ALT = 4, 
    AUX = 8
  }; 

  virtual int shift_state(void); /// return the shift state (ORed enums)

  /** 
    * get the current state of the Spaceball if one is available
    * returns rx ry rz, tx ty tz, buttons
    */
  virtual int spaceball(int *, int *, int *, int *, int *, int *, int *) { return 0; }

  /** 
    * set the Nth cursor shape as the current one.  If no arg given, the
    * default shape (n=0) is used.
    */
  virtual void set_cursor(int);

private:
  Projection my_projection;     ///< viewing projection mode used
  static const char *projNames[NUM_PROJECTIONS];
  static const char *cueModeNames[NUM_CUE_MODES];

public:
  //@{ 
  /// routines to deal with the clipping planes and eye position
  float aspect(void) { return Aspect; }
  float near_clip(void) const { return nearClip; }
  float far_clip(void) const { return farClip; }
  float clip_width(void) const { return (farClip - nearClip); }
  float addto_near_clip(float ac) { return set_near_clip(nearClip + ac); }
  float addto_far_clip(float ac) { return set_far_clip(farClip + ac); }
  float set_near_clip(float nc) {
    // near clip plane must be > 0.0, and less than far clip plane
    if (nc < farClip && nc > 0.0) {
      nearClip = nc;
      calc_frustum();
      _needRedraw = 1;
    } 
    return nearClip;
  }

  float set_far_clip(float fc) {
    if(fc > nearClip) {
      farClip = fc;
      _needRedraw = 1;
    }
    return farClip;
  }
  //@}


  //@{
  /// routines to deal with depth cueing / fog
  virtual void cueing_on(void);
  virtual void cueing_off(void);
  int cueing_available(void) { return cueingAvailable; }
  int cueing_enabled(void) { return cueingEnabled; }

  const char *get_cue_mode() const { return cueModeNames[cueMode]; }
  int num_cue_modes() const { return NUM_CUE_MODES; }
  const char *cue_mode_name(int i) const {
    if (i < 0 || i >= NUM_CUE_MODES) return NULL;
    return cueModeNames[i];
  }
  int set_cue_mode(const char *mode) { 
    for (int i=0; i<NUM_CUE_MODES; i++) {
      if (!strupcmp(mode, cueModeNames[i])) {
        cueMode = (CueMode)i;
        _needRedraw = 1;
        return TRUE;
      }
    }
    return FALSE; // no match
  }

  float get_cue_start() const { return cueStart; }
  float set_cue_start(float s) { 
    if (s < cueEnd && s >= 0.0) { 
      cueStart = s; 
      _needRedraw = 1;
      return TRUE;
    } 
    return FALSE;
  }   

  float get_cue_end() const { return cueEnd; }
  float set_cue_end(float e) { 
    if (e > cueStart) { 
      cueEnd = e; 
      _needRedraw = 1;
      return TRUE;
    } 
    return FALSE;
  }   
  
  float get_cue_density() const { return cueDensity; }
  int set_cue_density(float d) {
    // XXX check for legal value here
    cueDensity = d;
    _needRedraw = 1;
    return TRUE;
  }
  //@}


  //@{
  /// function for antialiasing control
  virtual void aa_on(void);
  virtual void aa_off(void);
  int aa_available(void) { return aaAvailable; }
  int aa_enabled(void) { return aaEnabled; }
  //@}


  //@{
  /// virtual function for controlling backface culling
  virtual void culling_on(void);
  virtual void culling_off(void);
  int culling_available(void) { return cullingAvailable; }
  int culling_enabled(void) { return cullingEnabled; }
  //@}


  //@{
  /// routines for controlling shadow rendering
  int set_shadow_mode(int onoff) {
    if (onoff)
      shadowEnabled=1;
    else
      shadowEnabled=0;
    _needRedraw = 1;
    return TRUE;
  }  
  int shadows_enabled(void) { return shadowEnabled; }
  //@}


  //@{
  /// routines for controlling ambient occlusion rendering
  int set_ao_mode(int onoff) {
    if (onoff)
      aoEnabled=1;
    else
      aoEnabled=0;
    _needRedraw = 1;
    return TRUE;
  }  
  int ao_enabled(void) { return aoEnabled; }

  float get_ao_ambient() const { return aoAmbient; }
  int set_ao_ambient(float a) {
    aoAmbient = a;
    _needRedraw = 1;
    return TRUE;
  }

  float get_ao_direct() const { return aoDirect; }
  int set_ao_direct(float d) {
    aoDirect = d;
    _needRedraw = 1;
    return TRUE;
  }
  //@}


  //@{
  /// routines for controlling depth of field rendering
  int set_dof_mode(int onoff) {
    if (onoff)
      dofEnabled=1;
    else
      dofEnabled=0;
    _needRedraw = 1;
    return TRUE;
  }  
  int dof_enabled(void) { return dofEnabled; }

  float get_dof_fnumber() const { return dofFNumber; }
  int set_dof_fnumber(float f) {
    dofFNumber = f;
    _needRedraw = 1;
    return TRUE;
  }

  float get_dof_focal_dist() const { return dofFocalDist; }
  int set_dof_focal_dist(float d) {
    dofFocalDist = d;
    _needRedraw = 1;
    return TRUE;
  }
  //@}


  //
  // camera and view frustum parameters 
  //

  /// return/set the distance from the origin to the screen
  float distance_to_screen(void) { return zDist; }
  void distance_to_screen(float zd) {
    zDist = zd;
    calc_frustum();
    _needRedraw = 1;
  }
  
  /// return the height of the screen
  float screen_height(void) { return vSize; }
  void screen_height(float vs) {
    if(vs > 0.0) {
      vSize = vs;
      calc_frustum();
      _needRedraw = 1;
    }
  }

  /// a) specify aspect ratio
  void set_screen_pos(float vsize, float zdist, float asp) {
    Aspect = asp;  vSize = vsize;  zDist = zdist;
    calc_frustum();    
  }
  
  /// b) have device provide aspect ratio
  void set_screen_pos(float vs, float zd) { set_screen_pos(vs, zd, aspect()); }
  
  /// c) just specify aspect ratio
  void set_screen_pos(float asp) { set_screen_pos(vSize, zDist, asp); }
  
  /// d) do not specify anything
  void set_screen_pos(void) { set_screen_pos(vSize, zDist); }

  /// Resize the window
  void resize_window(int w, int h) {
    if (w > 0 && h > 0) {
      do_resize_window(w, h);
    }
  }
  /// Reposition the window
  void reposition_window(int xpos, int ypos) {
    if (xpos >= 0 && xpos < screenX && ypos >= 0 && ypos < screenY) {
      do_reposition_window(xpos, -1+screenY-ypos);
    }
  } 
#if 0
  // XXX This doesn't work...
  void window_position(int *xpos, int *ypos) {
    *xpos = xOrig;
    //*ypos = screenY - 1 - yOrig;
    *ypos = yOrig + screenY - 1;
  }
#endif

  //
  // routines to deal with stereo display
  //
  
  /// change to a different stereo mode (0 means 'off')
  virtual void set_stereo_mode(int = 0);

  /// current stereo mode ... 0 means non-stereo, others device-specific
  int stereo_mode(void) { return inStereo; }

  /// swap left/right eyes when rendering in stereo (0 means not swapped)
  void set_stereo_swap(int onoff) { 
     stereoSwap = (!(onoff == 0)); 
    _needRedraw = 1;
  }

  /// current stereo swap mode ... 0 means not swapped
  int stereo_swap(void) { return stereoSwap; };
 
  /// whether we must force mono draws in stereo or not
  virtual int forced_stereo_draws(void) { return 0; }
 
  /// number of different stereo modes supported ... 0 means no stereo
  int num_stereo_modes(void) { return stereoModes; }
  
  /// return stereo name string, if possible
  const char *stereo_name(int n) {
    const char *retval = stereoNames[0];
    if(n >= 0 && n < stereoModes)
      retval = stereoNames[n];
    return retval;
  }

  //
  // routines to deal with special rendering modes/features
  //
  
  /// change to a different caching mode (0 means 'off')
  virtual void set_cache_mode(int = 0);

  /// return current caching mode
  int cache_mode(void) { return cacheMode; }

  /// number of different caching modes supported ... 0 means 'off'
  int num_cache_modes(void) { return cacheModes; }
  
  /// return caching name string, if possible
  const char *cache_name(int n) {
    const char *retval = cacheNames[0];
    if(n >= 0 && n < cacheModes)
      retval = cacheNames[n];
    return retval;
  }
  

  /// change to a different rendering mode (0 means 'normal')
  virtual void set_render_mode(int = 0);

  /// return current rendering mode
  int render_mode(void) { return renderMode; }
  
  /// number of different rendering modes supported ... 0 means 'normal'
  int num_render_modes(void) { return renderModes; }
  
  /// return rendering name string, if possible
  const char *render_name(int n) {
    const char *retval = renderNames[0];
    if(n >= 0 && n < renderModes)
      retval = renderNames[n];
    return retval;
  }


  /// set default eye position, orientation information
  int set_eye_defaults(void);

  /// set eye position
  int set_eye_pos(float * pos) {
    if (!pos) return FALSE;
    vec_copy(&eyePos[0], pos); 
    _needRedraw = 1;
    return TRUE; 
  }

  /// get eye position
  int get_eye_pos(float * pos) {
    if (!pos) return FALSE;
    vec_copy(pos, &eyePos[0]); 
    return TRUE; 
  }
 
  /// set eye direction
  int set_eye_dir(float * dir) {
    if (!dir) return FALSE;
    vec_copy(&eyeDir[0], dir); 
    _needRedraw = 1;
    return TRUE; 
  }

  /// get eye direction
  int get_eye_dir(float * dir) {
    if (!dir) return FALSE;
    vec_copy(dir, &eyeDir[0]); 
    return TRUE; 
  }
 
  /// set the eye up direction
  int set_eye_up(float *updir) {
    if (!updir) return FALSE;
    vec_copy(&upDir[0], updir); 
    calc_eyedir(); // recalculate related eye parameters
    _needRedraw = 1;
    return TRUE; 
  }

  /// query the eye up direction
  int get_eye_up(float *updir) {
    if (!updir) return FALSE;
    vec_copy(updir, &upDir[0]); 
    return TRUE; 
  }

  /// return focal length
  float eye_dist(void) const { return eyeDist; }

  /// change focal length; this means adjusting eyeDir and updating 
  /// eyeDist accordingly (eyeDist is just norm(eyeDir)).
  int set_eye_dist(float flen) {
    if (!eyeDist) return FALSE; // XXX is this possible?
    float fl = flen/eyeDist;
    for (int i=0; i<3; i++) eyeDir[i] *= fl;
    eyeDist = flen; 
    calc_eyedir(); // XXX should really be called calc_eyesep 
    _needRedraw = 1;
    return TRUE;
  }
   
  /// return eye separation
  float eyesep(void) const { return eyeSep; }

  /// set eye separation
  float set_eyesep(float newsep) {
    if(newsep >= 0.0) {
      eyeSep = newsep;
      calc_eyedir();
    }
    _needRedraw = 1;
    return eyeSep;
  }

  /// find the direction to the right eye position
  void right_eye_dir(float& x, float& y, float& z) const {
    x = eyeSepDir[0];  y = eyeSepDir[1];  z = eyeSepDir[2];
  }  

  /// get/change the viewing matrix
  int num_projections() const { return NUM_PROJECTIONS; }
  const char *projection_name(int i) const {
    if (i < 0 || i >= NUM_PROJECTIONS) return NULL;
    return projNames[i];
  }
  const char *get_projection() const { return projNames[my_projection]; }
  int set_projection(const char *proj) {
    if (!proj) return FALSE;
    for (int i=0; i<NUM_PROJECTIONS; i++) {
      if (!strupcmp(proj, projNames[i])) {
        my_projection = (Projection)i;
        _needRedraw = 1;
        return TRUE;
      }
    }
    return FALSE;
  }

  //
  // virtual routines to find characteristics of display itself
  //

  /// return normalized absolut 3D screen coordinates, given 3D world coordinates.
  /// (i.e the second parameter will return a 3D coordinate, with a normalized z)
  virtual void abs_screen_loc_3D(float *, float *);

  /// return absolute 2D screen coordinates, given 2D world coordinates.
  virtual void abs_screen_loc_2D(float *, float *);

  /// convert 2D absolute screen coords into relative coords, 0 ... 1
  virtual void rel_screen_pos(float &x, float &y) {
    x = (x - (float)xOrig) / ((float)xSize);
    y = (y - (float)yOrig) / ((float)ySize);
  }

  /// convert 2D relative screen coords into absolute coords
  void abs_screen_pos(float &x, float &y) {
    x = x * (float)xSize + (float)xOrig;
    y = y * (float)ySize + (float)yOrig;
  }

  // Given a 3D point (pos A),
  // and a 2D rel screen pos point (for pos B), computes the 3D point
  // which goes with the second 2D point at pos B.  Result returned in 3rd arg.
  virtual void find_3D_from_2D(const float *, const float *, float *) {}


  //
  // virtual routines to affect the device's transformation matrix
  //
  virtual void loadmatrix(const Matrix4 &); ///< replace transformation matrix 
  virtual void multmatrix(const Matrix4 &); ///< multiply transformation matrix


  //
  // set the background colors
  //
  void set_background_mode(int newmode) { backgroundmode = newmode; } ///< set display background type
  int background_mode(void) { return backgroundmode; } ///< return active background mode
  virtual void set_background(const float *) {} ///< set main background color
  virtual void set_backgradient(const float *, const float *) {} ///< set gradient colors

 
  //
  // virtual routines for preparing to draw, drawing, and finishing drawing
  //
  virtual int prepare3D(int = TRUE);    ///< ready to draw 3D
  virtual int prepareOpaque(void) { return 1; } ///< draw opaque objects
  virtual int prepareTrans(void){ return 0; }   ///< draw transparent objects
  virtual void clear(void);		///< erase the device
  virtual void left(void);		///< ready to draw left eye
  virtual void right(void);		///< ready to draw right eye
  virtual void normal(void);		///< ready to draw non-stereo
  virtual void update(int = TRUE);	///< finish up after drawing
  virtual void reshape(void);		///< refresh device after change

  /// virtual routine for capturing the screen to a packed RGB array
  virtual unsigned char * readpixels_rgb3u(int &x, int &y);

  /// virtual routine for capturing the screen to a packed RGBA array
  virtual unsigned char * readpixels_rgba4u(int &x, int &y);

  /// virtual routine for drawing the screen from a packed RGBA array
  virtual int drawpixels_rgba4u(unsigned char *rgba, int &x, int &y) { return -1; };
  
  
  /// process list of draw commands
  virtual void render(const VMDDisplayList *) { _needRedraw = 0; } 
                                          
  virtual void render_done() {}
  
  // pick objects based on given list of draw commands.
  // arguments are dimension of picking (2 or 3), position of pointer,
  // draw command list, and returned distance from object to eye position.
  // Returns ID code ('tag') for item closest to pointer, or (-1) if no pick.
  // If an object is picked, the eye distance argument is set to the distance
  // from the display's eye position to the object (after its position has been
  // found from the transformation matrix).  If the value of the argument when
  // 'pick' is called is <= 0, a pick will be generated if any item is near the
  // pointer.  If the value of the argument is > 0, a pick will be generated
  // only if an item is closer to the eye position than the value of the
  // argument.
  // For 2D picking, coordinates are relative position in window from
  //	lower-left corner (both in range 0 ... 1)
  // For 3D picking, coordinates are the world coords of the pointer.  They
  //	are the coords of the pointer after its transformation matrix has been
  //	applied, and these coordinates are compared to the coords of the objects
  //	when their transformation matrices are applied.
  // The window_size argument tells pick how close the picked item must be.
  virtual int pick(int, const float *, const VMDDisplayList *, float &, int *,
                   float window_size); 

};

#endif

