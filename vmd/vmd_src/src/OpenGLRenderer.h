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
 *	$RCSfile: OpenGLRenderer.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.150 $	$Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Subclass of DisplayDevice, this object has routines used by all the
 * different display devices that use OpenGL for rendering.
 * Will render drawing commands into a window.
 * This is not the complete definition,
 * however, of a DisplayDevice; something must provide routines to open
 * windows, reshape, clear, set perspective, etc.  This object contains the
 * code to render a display command list.
 *
 ***************************************************************************/
#ifndef OPENGLRENDERER_H
#define OPENGLRENDERER_H

// Starting with VMD 1.9.3, we disable the use of the old 
// OpenGL GLU library, in favor of VMD-internal replacements
// for GLU routines that set viewing matrices, project/unproject
// points to/from viewport coordinates, and draw spheres, conics,
// and end cap discs.  The GLU library is deprecated on OSes such as 
// MacOS X, and it's not suited for the latest revs of OpenGL.
//#define VMDUSELIBGLU 1  // macro to enable use of libGLU instead of built-ins

#include "DisplayDevice.h"
#include "Scene.h"
#include "OpenGLExtensions.h"
#include "OpenGLCache.h"

#if defined(VMDUSEOPENGLSHADER)
#include "OpenGLShader.h"
#endif

#if defined(_MSC_VER)
#include <windows.h>
#endif

// The Linux OpenGL ABI 1.0 spec requires that we define
// GL_GLEXT_PROTOTYPES before including gl.h or glx.h for extensions
// in order to get prototypes:
//   http://oss.sgi.com/projects/ogl-sample/ABI/index.html
#define GL_GLEXT_PROTOTYPES 1

#if defined(__APPLE__) && !defined (VMDMESA)
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

// include GLU if we use it
#if defined(VMDUSELIBGLU)
#if defined(__APPLE__) && !defined (VMDMESA)
#include <OpenGL/glu.h>
#else
#include <GL/glu.h>
#endif
#endif

// NOTE: you may have to get copies of the latest OpenGL extension headers
// from the OpenGL web site if your Linux machine lacks them:
//   http://oss.sgi.com/projects/ogl-sample/registry/
#if (defined(__linux) || defined(_MSC_VER)) && !defined(VMDMESA)
#include <GL/glext.h>
#endif
#if defined(__APPLE__) && !defined (VMDMESA)
#include <OpenGL/glext.h>
#endif

// required for Win32 calling conventions to work correctly
#ifndef APIENTRY
#define APIENTRY
#endif
#ifndef GLAPI
#define GLAPI extern
#endif

// simple defines for stereo modes
#define OPENGL_STEREO_OFF                  0
#define OPENGL_STEREO_QUADBUFFER           1
#define OPENGL_STEREO_HDTVSIDE             2
#define OPENGL_STEREO_STENCIL_CHECKERBOARD 3
#define OPENGL_STEREO_STENCIL_COLUMNS      4
#define OPENGL_STEREO_STENCIL_ROWS         5
#define OPENGL_STEREO_ANAGLYPH             6
#define OPENGL_STEREO_SIDE                 7
#define OPENGL_STEREO_ABOVEBELOW           8
#define OPENGL_STEREO_LEFT                 9
#define OPENGL_STEREO_RIGHT               10
#define OPENGL_STEREO_MODES               11 

// simple defines for rendering modes
#define OPENGL_RENDER_NORMAL               0
#define OPENGL_RENDER_GLSL                 1
#define OPENGL_RENDER_ACROBAT3D            2
#define OPENGL_RENDER_MODES                3

// simple defines for caching modes
#define OPENGL_CACHE_OFF                   0
#define OPENGL_CACHE_ON                    1
#define OPENGL_CACHE_MODES                 2

class OpenGLRenderer; //<! forward declaration of classes here
class VMDDisplayList; //<! forward declaration of classes here


/// DisplayDevice subclass implementing the low-level OpenGL rendering
/// functions used by several derived DisplayDevice subclasses.
/// This class renders drawing commands into a window provided by 
/// one of the further subclasses.
class OpenGLRenderer : public DisplayDevice {
#if defined(VMD_NANOHUB)
protected:
  GLuint _finalFbo, _finalColorTex, _finalDepthRb;
  bool init_offscreen_framebuffer(int width, int height);
#endif

public: 
  void setup_initial_opengl_state(void); ///< initialize VMD's OpenGL state

protected:
  //@{
  /// quadric objects and display lists for spheres, cylinders, and disks
#if defined(VMDUSELIBGLU)
  GLUquadricObj *pointsQuadric;
  GLUquadricObj *objQuadric;
#endif

  /// one sphere display list for each supported resolution
  ResizeArray<GLuint> solidSphereLists;
  ResizeArray<GLuint> pointSphereLists;

  /// the current sphere display list
  GLuint SphereList;
  //@}

  
  //@{
  /// cached copies of most recently used OpenGL state, materials, etc
  // used to eliminate unnecessary OpenGL state changes at draw time
  OpenGLCache displaylistcache;  // display list cache
  OpenGLCache texturecache;      // texture object cache
  int     oglmaterialindex;      // material index for fast matching
  float   oglopacity;
  float   oglambient;
  float   oglspecular;
  float   ogldiffuse;
  float   oglshininess;
  float   ogloutline;
  float   ogloutlinewidth;
  int     ogltransmode;
  GLfloat ogl_pmatrix[16];        // perspective matrix
  GLfloat ogl_mvmatrix[16];       // model view matrix
  Matrix4 ogl_textMat;            // text rendering matrix
  GLint   ogl_viewport[4];        // viewport setting
  GLint   ogl_fogmode;            // active fog mode
  int     ogl_lightingenabled;    // lighting on/off 
  int     ogl_useblendedtrans;    // flag to use alpha-blended transparency
  int     ogl_useglslshader;      // flag to use GLSL programmable shading
  int     ogl_glslserial;         // last rendering state GLSL used
  int     ogl_glsltoggle;         // GLSL state must be re-sent, when off 
  int     ogl_glslmaterialindex;  // last material rendered by GLSL
  int     ogl_glslprojectionmode; // which projection mode is in use
  int     ogl_glsltexturemode;    // whether shader perform texturing
  int     ogl_transpass;          // which rendering pass (solid/transparent)
  int     ogl_rendstateserial;    // light/fog/material state combo serial num
  int     ogl_clipmode[VMD_MAX_CLIP_PLANE];
  int     ogl_lightstate[DISP_LIGHTS];
  GLfloat ogl_lightcolor[DISP_LIGHTS][4];
  GLfloat ogl_lightpos[DISP_LIGHTS][4];
  GLfloat ogl_backgradient[2][4]; // background gradient colors
  int     ogl_acrobat3dcapture;   // don't cache anything, for 3-D capture
  //@}

  /// display list caching state variables
  int ogl_cacheenabled;       ///< flag to enable display list caching
  int ogl_cachedebug;         ///< flag to enable printing of debug messages
  GLint ogl_cachelistbase;    ///< base index for display list cache

  int dpl_initialized;        ///< have we initialized display lists?

protected:
  /// font info to use for our display ... MUST BE SET BY DERIVED CLASS
  GLuint font1pxListBase;     ///< 1-pixel wide non-AA font display list
  GLuint fontNpxListBase;     ///< N-pixel wide antialiased font display list

  OpenGLExtensions *ext;      ///< OpenGL Extensions class

#if defined(VMDUSEOPENGLSHADER)
  OpenGLShader *mainshader;   ///< Main OpenGL Vertex/Fragment Shader
  OpenGLShader *sphereshader; ///< Sphere-only OpenGL Vertex/Fragment Shader
  OpenGLShader *spherespriteshader; ///< Sphere-only OpenGL Vertex/Fragment Shader
#endif

  int simplegraphics;    ///< Force use of simplest OpenGL primitives
  int wiregl;            ///< Using Stanford's WireGL library 
  int intelswr;          ///< Intel's OpenSWR software rasterizer
  int immersadeskflip;   ///< Immersadesk right-eye X-axis reflection mode
  int shearstereo;       ///< Use shear matrix stereo rather than eye rotation

  //@{
  /// 2D texturing features
  int hastex2d;
  GLint max2DtexX;
  GLint max2DtexY;
  GLint max2DtexSize;
  //@}

  //@{
  /// 3D texturing features
  int hastex3d;
  GLint max3DtexX;
  GLint max3DtexY;
  GLint max3DtexZ;
  GLint max3DtexSize;
  //@}

  //
  // routines to perform various OGL-specific initializations.
  //
  /// Update the OpenGL sphere/cylinder/etc display lists
  void update_lists(void);
  void update_shader_uniforms(void *, int forceupdate);

  //@{
  /// routines to perform various OGL-specific graphics operations
  void set_line_width(int);
  void set_line_style(int);
  void set_sphere_res(int);
  void set_sphere_mode(int);
  void cylinder(float *, float *, int, float, float);  // slow cylinder version
  void require_volume_texture(unsigned long ID, 
    unsigned xsize, unsigned ysize, unsigned zsize, unsigned char *texmap);
  int build3Dmipmaps(int, int, int, unsigned char *tx);
  void draw_background_gradient(void);
  //@}
  
 
  //@{ 
  /// routines to deal with light sources at device level, return success/fail
  virtual int do_define_light(int n, float *color, float *position);
  virtual int do_activate_light(int n, int turnon);
  //@}

public:
  /// constructor/destructor
  OpenGLRenderer(const char *);
  virtual ~OpenGLRenderer(void);

  // All display device subclasses from OpenGLRenderer (with the notable
  // exception of OpenGLPbufferDisplayDevice) support GUIs.
  virtual int supports_gui() { return TRUE; }

  //@{
  /// virtual routines to affect the devices transformation matrix
  virtual void loadmatrix(const Matrix4&); // replace trans matrix w. given one
  virtual void multmatrix(const Matrix4&); // multiply trans matrix w. given one
  //@}


  //
  // virtual routines to find characteristics of display itself
  //

  //@{
  /// return normalized absolut 3D screen coordinates, given 3D world coordinates.
  virtual void abs_screen_loc_3D(float *, float *);
  /// return absolute 2D screen coordinates, given 2D world coordinates.
  virtual void abs_screen_loc_2D(float *, float *);
  //@}

  // Given a 3D point (pos A),
  // and a 2D rel screen pos point (for pos B), computes the 3D point
  // which goes with the second 2D point at pos B.  Result returned in B3D.
  virtual void find_3D_from_2D(const float *A3D, const float *B2D, float *B3D);

  //@{
  /// functions to control depth cueing, culling, and antialiasing
  virtual void aa_on(void);
  virtual void aa_off(void);
  virtual void cueing_on(void);
  virtual void cueing_off(void);
  virtual void culling_on(void);
  virtual void culling_off(void);
  //@}

  // get/set the background color
  virtual void set_background(const float *);      ///< set bg color
  virtual void set_backgradient(const float *, const float *); ///< set bg grad

  // virtual routines for preparing to draw, drawing, and finishing drawing
  virtual void enable_stencil_stereo(int newMode); ///< turn on stencil stereo
  virtual void disable_stencil_stereo(void);       ///< turn off stencil stereo
  virtual void left(void);                         ///< ready to draw left eye
  virtual void right(void);                        ///< ready to draw right eye
  virtual void normal(void);                       ///< ready to draw non-stereo
  virtual void set_persp(DisplayEye = NOSTEREO);   ///< set view configuration
  virtual int prepare3D(int do_clear = TRUE);      ///< ready to draw 3D
  virtual int prepareOpaque();                     ///< draw opaque objects
  virtual int prepareTrans();                      ///< draw transparent objects
  virtual void clear(void);                        ///< erase the device
  virtual void render(const VMDDisplayList *);     ///< process draw cmd list
  virtual void render_done();                      ///< post-rendering ops
  void free_opengl_ctx();                          ///< free gl context rsrcs

  /// whether we must force mono draws in stereo or not
  virtual int forced_stereo_draws(void) { return ext->stereodrawforced; }

  virtual void set_stereo_mode(int = 0);           ///< set stereo mode, 0==off
  virtual void set_cache_mode(int);                ///< set caching mode, 0==off
  virtual void set_render_mode(int);               ///< set render mode, 0==norm
};

#endif

