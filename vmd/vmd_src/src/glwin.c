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
 * $Id: glwin.c,v 1.29 2015/12/19 06:28:15 johns Exp $
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "glwin.h"

/* Support for compilation as part of VMD for interactive */
/* ray tracing display.                                   */
#if defined(VMDOPENGL)
#define USEOPENGL
#endif

#if defined(USEOPENGL)

/* The Linux OpenGL ABI 1.0 spec requires that that GL_GLEXT_PROTOTYPES be 
 * defined before including gl.h or glx.h for extensions in order to get 
 * prototypes:  http://oss.sgi.com/projects/ogl-sample/ABI/index.html
 */
#define GL_GLEXT_PROTOTYPES   1
#define GLX_GLXEXT_PROTOTYPES 1

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
#include <X11/Xatom.h>

#if defined(USEEGL)
/* use EGL for window management */
#include <EGL/egl.h>
#else
/* use GLX for window management */
#include <GL/glx.h>
#endif

/* use full OpenGL */
#include <GL/gl.h>
#endif

/*
 * Optionally enable advanced OpenGL shaders for HMD usage, etc.
 */
#if defined(USEGLEXT)

/* NOTE: you may have to get copies of the latest OpenGL extension headers
 * from the OpenGL web site if your Linux machine lacks them:
 *   http://oss.sgi.com/projects/ogl-sample/registry/
 */
#if (defined(__linux) || defined(_MSC_VER))
#include <GL/glext.h>
#endif

/* not needed with recent OSX 10.9 revs */
#if 0 && defined(__APPLE__)
#include <OpenGL/glext.h>
#endif

/* required for Win32 calling conventions to work correctly */
#ifndef APIENTRY
#define APIENTRY
#endif
#ifndef GLAPI
#define GLAPI extern
#endif

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


/* GLSL shader state */
typedef struct {
  int isvalid;                      /**< succesfully compiled shader flag */
  GLhandleARB ProgramObject;        /**< ARB program object handle */
  GLhandleARB VertexShaderObject;   /**< ARB vertex shader object handle */
  GLhandleARB FragmentShaderObject; /**< ARB fragment shader object handle */
  int lastshader;                   /**< last shader index/state used */
} glsl_shader;


/* struct to provide access to key GLSL shader routines */
typedef struct {
  int oglmajor;         /**< major version of OpenGL renderer */
  int oglminor;         /**< minor version of OpenGL renderer */
  int oglrelease;       /**< release of OpenGL renderer       */

  int hasglshaderobjectsarb;
  int hasglvertexshaderarb;
  int hasglfragmentshaderarb;
  int hasglgeometryshader4arb;
  int hasglsampleshadingarb;
  int hasglshadinglangarb;
  int hasglfborendertarget;
  int hasgetvideosyncsgi;

/* when extensions are found in the headers, we include them in the struct */
#if defined(GL_ARB_shader_objects)
  /* GLSL fctn ptrs  */
  GLhandleARB (APIENTRY *p_glCreateShaderObjectARB)(GLenum shaderType);
  GLhandleARB (APIENTRY *p_glCreateProgramObjectARB)(void);
  void (APIENTRY *p_glUseProgramObjectARB)(GLhandleARB programObj);
  void (APIENTRY *p_glDetachObjectARB)(GLhandleARB containerObj, GLhandleARB attachedObj);
  void (APIENTRY *p_glGetInfoLogARB)(GLhandleARB obj,GLsizei maxLength, GLsizei *length, GLcharARB *infoLog);
  void (APIENTRY *p_glGetObjectParameterivARB)(GLhandleARB obj, GLenum pname, GLint *params);
  void (APIENTRY *p_glLinkProgramARB)(GLhandleARB programObj);
  void (APIENTRY *p_glDeleteObjectARB)(GLhandleARB obj);
  void (APIENTRY *p_glAttachObjectARB)(GLhandleARB containerObj, GLhandleARB obj);
  void (APIENTRY *p_glCompileShaderARB)(GLhandleARB shaderObj);
  void (APIENTRY *p_glShaderSourceARB)(GLhandleARB shaderObj, GLsizei count, const GLcharARB **strings, const GLint *length);
  GLint (APIENTRY *p_glGetUniformLocationARB)(GLhandleARB programObject, const GLcharARB *name);
  void (APIENTRY *p_glUniform1iARB)(GLint location, GLint v0);
  void (APIENTRY *p_glUniform1fvARB)(GLint location, GLsizei count, GLfloat *value);
  void (APIENTRY *p_glUniform2fvARB)(GLint location, GLsizei count, GLfloat *value);
  void (APIENTRY *p_glUniform3fvARB)(GLint location, GLsizei count, GLfloat *value);
  void (APIENTRY *p_glUniform4fvARB)(GLint location, GLsizei count, GLfloat *value);

  /* FBO and render target fctns */
  void (APIENTRY *p_glGenFramebuffers)(GLsizei n, GLuint * framebuffers);
  void (APIENTRY *p_glBindFramebuffer)(GLenum target, GLuint framebuffer); 
  void (APIENTRY *p_glGenRenderbuffers)(GLsizei n, GLuint * renderbuffers);
  void (APIENTRY *p_glBindRenderbuffer)(GLenum target, GLuint renderbuffer);
  void (APIENTRY *p_glRenderbufferStorage)(GLenum target, GLenum internalformat, GLsizei width, GLsizei height);
  void (APIENTRY *p_glFramebufferTexture2D)(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level);
  void (APIENTRY *p_glFramebufferRenderbuffer)(GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer);
  GLenum (APIENTRY *p_glCheckFramebufferStatus)(GLenum target);
  void (APIENTRY *p_glDeleteRenderbuffers)(GLsizei n, const GLuint * renderbuffers);
  void (APIENTRY *p_glDeleteFramebuffers)(GLsizei n, const GLuint * framebuffers);
  void (APIENTRY *p_glDrawBuffers)(GLsizei n, const GLenum *bufs);
#endif

  /* SGI video sync query extension */
  int (APIENTRY *p_glXGetVideoSyncSGI)(GLuint *count);
} glwin_ext_fctns;


#if defined(GL_ARB_shader_objects)
/* GLSL shader functions */
#define GLCREATESHADEROBJECTARB   ext->p_glCreateShaderObjectARB
#define GLCREATEPROGRAMOBJECTARB  ext->p_glCreateProgramObjectARB
#define GLUSEPROGRAMOBJECTARB     ext->p_glUseProgramObjectARB
#define GLDETACHOBJECTARB         ext->p_glDetachObjectARB
#define GLGETINFOLOGARB           ext->p_glGetInfoLogARB
#define GLGETOBJECTPARAMETERIVARB ext->p_glGetObjectParameterivARB
#define GLLINKPROGRAMARB          ext->p_glLinkProgramARB
#define GLDELETEOBJECTARB         ext->p_glDeleteObjectARB
#define GLATTACHOBJECTARB         ext->p_glAttachObjectARB
#define GLCOMPILESHADERARB        ext->p_glCompileShaderARB
#define GLSHADERSOURCEARB         ext->p_glShaderSourceARB
#define GLGETUNIFORMLOCATIONARB   ext->p_glGetUniformLocationARB
#define GLUNIFORM1IARB            ext->p_glUniform1iARB
#define GLUNIFORM1FVARB           ext->p_glUniform1fvARB
#define GLUNIFORM2FVARB           ext->p_glUniform2fvARB
#define GLUNIFORM3FVARB           ext->p_glUniform3fvARB
#define GLUNIFORM4FVARB           ext->p_glUniform4fvARB

/* FBO and render buffer management functions */
#define GLGENFRAMEBUFFERS         ext->p_glGenFramebuffers
#define GLBINDFRAMEBUFFER         ext->p_glBindFramebuffer
#define GLGENRENDERBUFFERS        ext->p_glGenRenderbuffers
#define GLBINDRENDERBUFFER        ext->p_glBindRenderbuffer
#define GLRENDERBUFFERSTORAGE     ext->p_glRenderbufferStorage
#define GLFRAMEBUFFERTEXTURE2D    ext->p_glFramebufferTexture2D
#define GLFRAMEBUFFERRENDERBUFFER ext->p_glFramebufferRenderbuffer
#define GLCHECKFRAMEBUFFERSTATUS  ext->p_glCheckFramebufferStatus
#define GLDELETERENDERBUFFERS     ext->p_glDeleteRenderbuffers
#define GLDELETEFRAMEBUFFERS      ext->p_glDeleteFramebuffers
#define GLDRAWBUFFERS             ext->p_glDrawBuffers

/* video sync extensions */
#define GLXGETVIDEOSYNCSGI        ext->p_glXGetVideoSyncSGI
#endif


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
#if defined(USEEGL)
  EGLDisplay egldpy;         /**< EGL display for OpenGL window */
  EGLConfig  eglconf;        /**< EGL config  for OpenGL window */
  EGLSurface eglsurf;        /**< EGL surface for OpenGL window */
  EGLContext eglctx;         /**< EGL context for OpenGL window */
#else
  GLXContext ctx;            /**< GLX context for OpenGL window */
#endif
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

  glwin_ext_fctns *ext;   /**< OpenGL extension fctn pointers */
} oglhandle;


#if defined(USEOPENGL) && !defined(USEEGL) && !defined(WIN32) && !defined(_MSC_VER)
static int glx_query_extension(Display *dpy, const char *extname) {
  char *ext;
  char *endext;
  if (!extname)
    return 0;

  /* check for GLX extensions too */
  ext = (char *) glXQueryExtensionsString(dpy, 0);
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
#endif


/* static helper routines for handling quaternions from HMDs */
static void quat_rot_matrix(float *m, const float *q) {
  m[ 0] = 1.0f - 2.0f * (q[1] * q[1] + q[2] * q[2]);
  m[ 1] = 2.0f * (q[0] * q[1] - q[2] * q[3]);
  m[ 2] = 2.0f * (q[2] * q[0] + q[1] * q[3]);
  m[ 3] = 0.0f;

  m[ 4] = 2.0f * (q[0] * q[1] + q[2] * q[3]);
  m[ 5] = 1.0f - 2.0f * (q[2] * q[2] + q[0] * q[0]);
  m[ 6] = 2.0f * (q[1] * q[2] - q[0] * q[3]);
  m[ 7] = 0.0f;

  m[ 8] = 2.0f * (q[2] * q[0] - q[1] * q[3]);
  m[ 9] = 2.0f * (q[1] * q[2] + q[0] * q[3]);
  m[10] = 1.0f - 2.0f * (q[1] * q[1] + q[0] * q[0]);
  m[11] = 0.0f;

  m[12] = 0.0f;
  m[13] = 0.0f;
  m[14] = 0.0f;
  m[15] = 1.0f;
}


/* prevent vendor-specific header file clashes */
typedef void (APIENTRY *glwin_fctnptr)(void);

/* static helper to query GL fctn pointers */
void * glwin_get_procaddress(const char * procname) {
  void *fctn = NULL;
  if (!procname)
    return NULL;

#if defined(_MSC_VER)
  /* NOTE: wgl returns a context-dependent function pointer
   *       the function can only be called within the same wgl
   *       context in which it was generated.
   */
  fctn = (glwin_fctnptr) wglGetProcAddress((LPCSTR) procname);
#else

#if !defined(_MSC_VER) && !defined(__APPLE__)
  /* GLX 1.4 form found on commercial Unix systems that
   * don't bother providing the ARB extension version that Linux prefers.
   */
  fctn = glXGetProcAddressARB((const GLubyte *) procname);
#if 0
  printf("GL fctn '%s' %s\n", procname, (fctn) ? "available" : "NULL");
#endif
#endif


#if defined(GLX_ARB_get_proc_address)
  /* NOTE: GLX returns a context-independent function pointer that
   *       can be called anywhere, no special handling is required.
   *       This method is used on Linux
   */
  if (fctn == NULL) {
    fctn = glXGetProcAddressARB((const GLubyte *) procname);
#if 0
    printf("GLARB fctn '%s' %s\n", procname, (fctn) ? "available" : "NULL");
#endif
  }
#endif
#endif

#if 0
  printf("GL fctn '%s' %s\n", procname, (fctn) ? "available" : "NULL");
#endif

  return fctn;
}


/* static helper routine to init OpenGL ext fctn pointers to NULL */
void glwin_init_exts(void * voidhandle) {
  oglhandle * handle = (oglhandle *) voidhandle;
  if (handle == NULL)
    return;
  glwin_ext_fctns *ext = handle->ext;

  /* clear everything to zeros, so all ptrs are NULL, */
  /* and all flags are false.                         */
  memset(ext, 0, sizeof(glwin_ext_fctns));

#if defined(GL_ARB_shading_language_100)
  /* check for the OpenGL Shading Language extension */
  if (glwin_query_extension("GL_ARB_shading_language_100")) {
    ext->hasglshadinglangarb = 1;
  }
#endif

#if defined(GL_ARB_shader_objects)
  if (glwin_query_extension("GL_ARB_shader_objects")) {
    ext->p_glCreateShaderObjectARB = (GLhandleARB (APIENTRY *)(GLenum)) glwin_get_procaddress("glCreateShaderObjectARB");
    ext->p_glCreateProgramObjectARB = (GLhandleARB (APIENTRY *)(void)) glwin_get_procaddress("glCreateProgramObjectARB");
    ext->p_glUseProgramObjectARB = (void (APIENTRY *)(GLhandleARB)) glwin_get_procaddress("glUseProgramObjectARB");
    ext->p_glDetachObjectARB = (void (APIENTRY *)(GLhandleARB, GLhandleARB)) glwin_get_procaddress("glDetachObjectARB");
    ext->p_glGetInfoLogARB = (void (APIENTRY *)(GLhandleARB, GLsizei, GLsizei *, GLcharARB *)) glwin_get_procaddress("glGetInfoLogARB");
    ext->p_glGetObjectParameterivARB = (void (APIENTRY *)(GLhandleARB, GLenum, GLint *)) glwin_get_procaddress("glGetObjectParameterivARB");
    ext->p_glLinkProgramARB = (void (APIENTRY *)(GLhandleARB)) glwin_get_procaddress("glLinkProgramARB");
    ext->p_glDeleteObjectARB = (void (APIENTRY *)(GLhandleARB)) glwin_get_procaddress("glDeleteObjectARB");
    ext->p_glAttachObjectARB = (void (APIENTRY *)(GLhandleARB, GLhandleARB)) glwin_get_procaddress("glAttachObjectARB");
    ext->p_glCompileShaderARB = (void (APIENTRY *)(GLhandleARB)) glwin_get_procaddress("glCompileShaderARB");
    ext->p_glShaderSourceARB = (void (APIENTRY *)(GLhandleARB, GLsizei, const GLcharARB **, const GLint *)) glwin_get_procaddress("glShaderSourceARB");
    ext->p_glGetUniformLocationARB = (GLint (APIENTRY *)(GLhandleARB programObject, const GLcharARB *name)) glwin_get_procaddress("glGetUniformLocationARB");
    ext->p_glUniform1iARB = (void (APIENTRY *)(GLint location, GLint v0)) glwin_get_procaddress("glUniform1iARB");
    ext->p_glUniform1fvARB = (void (APIENTRY *)(GLint location, GLsizei count, GLfloat *value)) glwin_get_procaddress("glUniform1fvARB");
    ext->p_glUniform2fvARB = (void (APIENTRY *)(GLint location, GLsizei count, GLfloat *value)) glwin_get_procaddress("glUniform2fvARB");
    ext->p_glUniform3fvARB = (void (APIENTRY *)(GLint location, GLsizei count, GLfloat *value)) glwin_get_procaddress("glUniform3fvARB");
    ext->p_glUniform4fvARB = (void (APIENTRY *)(GLint location, GLsizei count, GLfloat *value)) glwin_get_procaddress("glUniform4fvARB");

    if (ext->p_glCreateShaderObjectARB != NULL && ext->p_glCreateProgramObjectARB != NULL &&
        ext->p_glUseProgramObjectARB != NULL && ext->p_glDetachObjectARB != NULL &&
        ext->p_glGetInfoLogARB != NULL && ext->p_glGetObjectParameterivARB != NULL &&
        ext->p_glLinkProgramARB != NULL && ext->p_glDeleteObjectARB != NULL &&
        ext->p_glAttachObjectARB != NULL && ext->p_glCompileShaderARB != NULL &&
        ext->p_glShaderSourceARB != NULL && ext->p_glGetUniformLocationARB != NULL &&
        ext->p_glUniform1iARB != NULL && ext->p_glUniform1fvARB != NULL &&
        ext->p_glUniform2fvARB != NULL && ext->p_glUniform3fvARB != NULL &&
        ext->p_glUniform4fvARB  != NULL) {
      ext->hasglshaderobjectsarb = 1;
    } 
  }
#endif

#if defined(GL_ARB_vertex_shader)
  if (glwin_query_extension("GL_ARB_vertex_shader")) {
    ext->hasglvertexshaderarb = 1;
  }
#endif

#if defined(GL_ARB_fragment_shader)
  if (glwin_query_extension("GL_ARB_fragment_shader")) {
    ext->hasglfragmentshaderarb = 1;
  }
#endif

#if defined(GL_ARB_geometry_shader4)
  if (glwin_query_extension("GL_ARB_geometry_shader4")) {
    ext->hasglgeometryshader4arb = 1;
  }
#endif

  if (glwin_query_extension("GL_ARB_sample_shading")) {
    ext->hasglsampleshadingarb = 1;
  }

#if defined(GL_ARB_framebuffer_object)
  /* routines for managing FBOs and render targets */
  ext->p_glGenFramebuffers = (void (APIENTRY *)(GLsizei n, GLuint * framebuffers)) glwin_get_procaddress("glGenFramebuffers");
  ext->p_glBindFramebuffer = (void (APIENTRY *)(GLenum target, GLuint framebuffer)) glwin_get_procaddress("glBindFramebuffer"); 
  ext->p_glGenRenderbuffers = (void (APIENTRY *)(GLsizei n, GLuint * renderbuffers)) glwin_get_procaddress("glGenRenderbuffers");
  ext->p_glBindRenderbuffer = (void (APIENTRY *)(GLenum target, GLuint renderbuffer)) glwin_get_procaddress("glBindRenderbuffer");
  ext->p_glRenderbufferStorage = (void (APIENTRY *)(GLenum target, GLenum internalformat, GLsizei width, GLsizei height)) glwin_get_procaddress("glRenderbufferStorage");
  ext->p_glFramebufferTexture2D = (void (APIENTRY *)(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level)) glwin_get_procaddress("glFramebufferTexture2D");
  ext->p_glFramebufferRenderbuffer = (void (APIENTRY *)(GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer)) glwin_get_procaddress("glFramebufferRenderbuffer");
  ext->p_glCheckFramebufferStatus = (GLenum (APIENTRY *)(GLenum target)) glwin_get_procaddress("glCheckFramebufferStatus");
  ext->p_glDeleteRenderbuffers = (void (APIENTRY *)(GLsizei n, const GLuint * renderbuffers)) glwin_get_procaddress("glDeleteRenderbuffers");
  ext->p_glDeleteFramebuffers = (void (APIENTRY *)(GLsizei n, const GLuint * framebuffers)) glwin_get_procaddress("glDeleteFramebuffers");
  ext->p_glDrawBuffers = (void (APIENTRY *)(GLsizei n, const GLenum *bufs)) glwin_get_procaddress("glDrawBuffers");

  if (ext->p_glGenFramebuffers != NULL && 
      ext->p_glBindFramebuffer != NULL &&
      ext->p_glGenRenderbuffers != NULL &&
      ext->p_glBindRenderbuffer != NULL && 
      ext->p_glRenderbufferStorage != NULL &&
      ext->p_glFramebufferTexture2D != NULL &&
      ext->p_glFramebufferRenderbuffer != NULL &&
      ext->p_glCheckFramebufferStatus != NULL &&
      ext->p_glDeleteRenderbuffers != NULL &&
      ext->p_glDeleteFramebuffers != NULL &&
      ext->p_glDrawBuffers != NULL) {
    ext->hasglfborendertarget = 1;
  }
#endif

  if (glx_query_extension(handle->dpy, "GLX_SGI_video_sync")) {
    ext->p_glXGetVideoSyncSGI = (int (APIENTRY *)(GLuint *count)) glwin_get_procaddress("glXGetVideoSyncSGI");

    if (ext->p_glXGetVideoSyncSGI != NULL)
      ext->hasgetvideosyncsgi = 1;
  }
}


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
  spaceballhandle *handle = (spaceballhandle *) calloc(1, sizeof(spaceballhandle));

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


static oglhandle *glwin_alloc_init(void) {
  oglhandle * handle = (oglhandle *) calloc(1, sizeof(oglhandle));
  if (handle == NULL)
    return NULL;

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

  handle->ext = (glwin_ext_fctns *) calloc(1, sizeof(glwin_ext_fctns));

  return handle;
}


/*
 * X11 implementation of glwin routines
 */
void * glwin_create(const char * wintitle, int width, int height) {
#if defined(USEEGL)
  /* EGL */
  oglhandle * handle; 
  XSetWindowAttributes attr;
  unsigned long mask;
  int num_visuals;
  XVisualInfo vistemplate;
  XVisualInfo *vis=NULL;
  XSizeHints sizeHints;
  GLint stencilbits;
  EGLint eglmaj, eglmin;
  EGLint num_config=0;

#if defined(USEOPENGLES2)
#define GLWIN_RENDERABLE_TYPE EGL_OPENGL_ES2_BIT
#else
#define GLWIN_RENDERABLE_TYPE EGL_OPENGL_BIT
#endif

  EGLint eglnormalattrib[] =   { EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, 
                                 EGL_BLUE_SIZE, 8, 
                                 EGL_DEPTH_SIZE, 16, 
                                 EGL_STENCIL_SIZE, 1,
/*                                 EGL_NATIVE_RENDERABLE,  */
                                 EGL_RENDERABLE_TYPE, GLWIN_RENDERABLE_TYPE,
/*                                 EGL_SURFACE_TYPE, EGL_WINDOW_BIT, */
                                 EGL_NONE };
  EGLint eglfailsafeattrib[] = { EGL_RED_SIZE, 1, EGL_GREEN_SIZE, 1, 
                                 EGL_BLUE_SIZE, 1, 
/*                                 EGL_DEPTH_SIZE, 16,  */
/*                                 EGL_NATIVE_RENDERABLE, */
                                 EGL_RENDERABLE_TYPE, GLWIN_RENDERABLE_TYPE,
/*                                 EGL_SURFACE_TYPE, EGL_WINDOW_BIT, */
                                 EGL_NONE };

  handle = glwin_alloc_init();
  handle->width = width;
  handle->height = height;

  /* setup EGL state */
#if 1
  handle->egldpy = eglGetDisplay(handle->dpy);
#else
  handle->egldpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
#endif
  if (handle->egldpy == EGL_NO_DISPLAY) {
    printf("glwin_create(): Failed to connect to EGL display\n"); 
    free(handle);
    return NULL;
  }

  if (eglInitialize(handle->egldpy, &eglmaj, &eglmin) == EGL_FALSE) {
    printf("glwin_create(): Failed to initialize EGL display connection\n"); 
    free(handle);
    return NULL;
#if 1
  } else {
    printf("EGL init dpy version: %d.%d\n", eglmaj, eglmin);
#endif
  }

  /* try for full stereo w/ features first */
  handle->havestencil = 1;
  handle->instereo = 0;
  if (eglChooseConfig(handle->egldpy, eglnormalattrib, &handle->eglconf, 1, &num_config) == EGL_FALSE) {
    printf("eglChooseConfig(1) %d configs\n", num_config);
    if (eglChooseConfig(handle->egldpy, eglfailsafeattrib, &handle->eglconf, 1, &num_config) == EGL_FALSE) {
      printf("Error: eglChooseConfig() failed\n");
      free(handle);
      return NULL;
    }
    handle->havestencil = 0;
  }
  printf("eglChooseConfig() %d configs\n", num_config);


  EGLint vid;
  if (eglGetConfigAttrib(handle->egldpy, handle->eglconf, EGL_NATIVE_VISUAL_ID, &vid) == EGL_FALSE) {
    printf("Error: eglGetConfigAttrib() failed\n");
    return NULL;
  }

  vistemplate.visualid = vid;
  vis = XGetVisualInfo(handle->dpy, VisualIDMask, &vistemplate, &num_visuals);
  if (vis == NULL) {
    printf("Error: failed to obtain EGL-compatible X visual...\n");
    free(handle);
    return NULL;
  }


#if defined(USEOPENGLES2)
  /* bind to OpenGL API since some implementations don't do this by default */
  if (eglBindAPI(EGL_OPENGL_ES_API) == EGL_FALSE) {
    printf("Error: failed to bind OpenGL ES API\n");
    free(handle);
    return NULL;
  }
#else
  /* bind to OpenGL API since some implementations don't do this by default */
  if (eglBindAPI(EGL_OPENGL_API) == EGL_FALSE) {
    printf("Error: failed to bind full OpenGL API\n");
    free(handle);
    return NULL;
  }
#endif


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

#if 0
  /* XXX untested EWMH code to enable compositor bypass  */
  Atom bypasscomp = X11_XInternAtom(handle->dpy, "_NET_WM_BYPASS_COMPOSITOR", False);
  const long bypasscomp_on = 1;
  X11_XChangeProperty(handle->dpy, handle->win, bypasscomp, XA_CARDINAL, 32,
                      PropModeReplace, (unsigned char *) bypasscomp_on, 1);
#endif

  handle->eglctx = eglCreateContext(handle->dpy, handle->eglconf, EGL_NO_CONTEXT, NULL);

  handle->eglsurf = eglCreateWindowSurface(handle->egldpy, handle->eglconf, handle->win, NULL);
  eglMakeCurrent(handle->dpy, handle->eglsurf, handle->eglsurf, handle->eglctx);

  /* initialize extensions once window has been made current... */
  glwin_init_exts(handle);

  EGLint Context_RendererType=0;
  eglQueryContext(handle->egldpy, handle->eglctx, EGL_CONTEXT_CLIENT_TYPE, &Context_RendererType);

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
  printf("EGL_CONTEXT_CLIENT_TYPE: %s\n", glstring);


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

#if 0
  /* check for an OpenGL stencil buffer */
  glGetIntegerv(GL_STENCIL_BITS, &stencilbits);
  if (stencilbits > 0) {
    handle->havestencil = 1;
  }
#endif

  glClearColor(1.0, 0.0, 0.0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT);
  glwin_swap_buffers(handle);
  glClear(GL_COLOR_BUFFER_BIT);
  glwin_swap_buffers(handle);


  XFlush(handle->dpy);

  return handle;
#else
  /* GLX */
  oglhandle * handle; 
  XSetWindowAttributes attr;
  unsigned long mask;
  XVisualInfo *vis=NULL;
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

  handle = glwin_alloc_init();
  handle->width = width;
  handle->height = height;

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

#if 0
  /* XXX untested EWMH code to enable compositor bypass */
  Atom bypasscomp = X11_XInternAtom(handle->dpy, "_NET_WM_BYPASS_COMPOSITOR", False);
  const long bypasscomp_on = 1;
  X11_XChangeProperty(handle->dpy, handle->win, bypasscomp, XA_CARDINAL, 32,
                      PropModeReplace, (unsigned char *) bypasscomp_on, 1);
#endif

  handle->ctx = glXCreateContext( handle->dpy, vis, NULL, True );

  glXMakeCurrent( handle->dpy, handle->win, handle->ctx );

  /* initialize extensions once window has been made current... */
  glwin_init_exts(handle);

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
#endif
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

  /* destroy GLSL and FBO extension fctns */
  if (handle->ext != NULL) {
    free(handle->ext);
  }

  /* close and delete window */
  XUnmapWindow(handle->dpy, handle->win);

#if defined(USEEGL)
  /* EGL code path */
  eglMakeCurrent(handle->egldpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
  eglDestroyContext(handle->egldpy, handle->eglctx);
  eglDestroySurface(handle->egldpy, handle->eglsurf);
  eglTerminate(handle->egldpy);
#else
  /* GLX code path */
  glXMakeCurrent(handle->dpy, None, NULL);
#endif

  XDestroyWindow(handle->dpy, handle->win);
  XCloseDisplay(handle->dpy); 
}

 
void glwin_swap_buffers(void * voidhandle) {
  oglhandle * handle = (oglhandle *) voidhandle;

  if (handle != NULL)
#if defined(USEEGL)
    /* EGL code path */
    eglSwapBuffers(handle->egldpy, handle->eglsurf);
#else
    /* GLX code path */
    glXSwapBuffers(handle->dpy, handle->win);
#endif
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

#if 0
  if (handle) { 
    /* check window size */
    XWindowAttributes xwa;
    XGetWindowAttributes(handle->dpy, handle->win, &xwa);
    handle->width = xwa.width;
    handle->height = xwa.height;
  }
#endif

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

#if 0
  XFlush(handle->dpy);
#endif

  return 0;
}


int glwin_reposition(void *voidhandle, int xpos, int ypos) {
  oglhandle * handle = (oglhandle *) voidhandle;
  if (handle == NULL)
    return -1;

  XMoveWindow(handle->dpy, handle->win, xpos, ypos);

  return 0;
}


#if 0
int glwin_switch_fullscreen_video_mode(void * voidhandle, mode...) {
  XF86VidModeSwitchToMode(display,defaultscreen,video_mode);
  XF86VidModeSetViewPort(display,DefaultScreen,0,0);
  XMoveResizeWindow(display,window,0,0,width,height);
  XMapRaised(display,window);
  XGrabPointer(display,window,True,0,GrabModeAsync,GrabModeAsync,window,0L,CurrentTime);
  XGrabKeyboard(display,window,False,GrabModeAsync,GrabModeAsync,CurrentTime);
}
#endif


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

#if 0
  {
    XSetWindowAttributes xswa;
    xswa.override_redirect = False;
    XChangeWindowAttributes(handle->dpy, handle->win, CWOverrideRedirect, &xswa);
  }
#endif

#if 1 && defined(__linux)
  /* support EWMH methods for setting full-screen mode            */
  /* http://standards.freedesktop.org/wm-spec/wm-spec-latest.html */
  Atom fsatom = XInternAtom(handle->dpy, "_NET_WM_STATE_FULLSCREEN", True);
  Atom stateatom = XInternAtom(handle->dpy, "_NET_WM_STATE", True);
  if (fsatom != None && stateatom != None) {
#if 0
    XChangeProperty(handle->dpy, handle->win, stateatom,
                    XA_ATOM, 32, PropModeReplace, (unsigned char*) &fsatom, 1);
#endif

    XEvent xev;
    memset(&xev, 0, sizeof(xev));
    xev.type = ClientMessage;
    xev.xclient.window = handle->win;
    xev.xclient.message_type = stateatom;
    xev.xclient.format = 32;
    if (fson)
      xev.xclient.data.l[0] = 1; /* _NET_WM_STATE_ADD    */
    else 
      xev.xclient.data.l[0] = 0; /* _NET_WM_STATE_REMOVE */
    xev.xclient.data.l[1] = fsatom;
    xev.xclient.data.l[2] = 0;

    XSendEvent(handle->dpy, handle->root, False,
               SubstructureRedirectMask | SubstructureNotifyMask, &xev);

    XFlush(handle->dpy);
  } 
#if 0
  else {
    printf("*** failed to obtain full screen X11 atom\n");
  }
#endif
#endif

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

#if 0
  {
    XSetWindowAttributes xswa;
    xswa.override_redirect = True;
    XChangeWindowAttributes(handle->dpy, handle->win, CWOverrideRedirect, &xswa);
  }
#endif

#if 0
  XSync(handle->dpy, 0);
#endif

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
  spaceballhandle *handle = (spaceballhandle *) calloc(1, sizeof(spaceballhandle));

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

  res = SiBeep(handle->sball, "CcCc"); /* beep the spaceball */
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

  handle = (oglhandle *) calloc(1, sizeof(oglhandle));
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
int glwin_query_extension(const char *extname) {
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


int glwin_query_vsync(void *voidhandle, int *onoff) {
  oglhandle * handle = (oglhandle *) voidhandle;
  if (handle == NULL)
    return GLWIN_ERROR; 

#if defined(GLX_EXT_swap_control)
  if (glx_query_extension(handle->dpy, "GLX_EXT_swap_control")) {
    int interval = 0;
    unsigned int tmp = -1;
    glXQueryDrawable(handle->dpy, glXGetCurrentDrawable(), GLX_SWAP_INTERVAL_EXT, &tmp);
    interval = tmp;
    if (interval > 0) {
      *onoff = 1;
    } else { 
      *onoff = 0;
    }
    return GLWIN_SUCCESS;
  }
#elif 0
  GLuint count = 0;
  glwin_ext_fctns *ext = handle->ext;
  if (ext->hasgetvideosyncsgi) {
    /* XXX this doesn't help much since we get non-zero counts */
    /* even when using a screen with vsync disabled            */
    if (GLXGETVIDEOSYNCSGI(&count) == 0) {
      if (count > 0) {
        *onoff = 1;
      } else { 
        *onoff = 0;
      }
    }
    return GLWIN_SUCCESS;
  }
#endif

  return GLWIN_NOT_IMPLEMENTED;
}


#if 1

typedef struct {
  int isvalid;
  GLenum drawbufs[16];
  GLuint fbo;
  GLuint tex;
  GLuint depth;
} glwin_fbo_target;


int glwin_fbo_target_bind(void *voidhandle, void *voidtarget) {
  oglhandle * handle = (oglhandle *) voidhandle;
  glwin_fbo_target * fb = (glwin_fbo_target *) voidtarget;
  if (handle == NULL || fb == NULL)
    return GLWIN_ERROR; 
  glwin_ext_fctns *ext = handle->ext;

  GLBINDFRAMEBUFFER(GL_FRAMEBUFFER, fb->fbo); /* bind FBO */

  return GLWIN_SUCCESS; 
}


int glwin_fbo_target_unbind(void *voidhandle, void *voidtarget) {
  oglhandle * handle = (oglhandle *) voidhandle;
  glwin_fbo_target * fb = (glwin_fbo_target *) voidtarget;
  if (handle == NULL || fb == NULL)
    return GLWIN_ERROR; 
  glwin_ext_fctns *ext = handle->ext;

  GLBINDFRAMEBUFFER(GL_FRAMEBUFFER, 0); /* bind the normal framebuffer */

  return GLWIN_SUCCESS; 
}


int glwin_fbo_target_destroy(void *voidhandle, void *voidtarget) {
  oglhandle * handle = (oglhandle *) voidhandle;
  glwin_fbo_target * fb = (glwin_fbo_target *) voidtarget;
  if (handle == NULL || fb == NULL)
    return GLWIN_ERROR; 
  glwin_ext_fctns *ext = handle->ext;

  GLDELETERENDERBUFFERS(1, &fb->depth);
  GLDELETEFRAMEBUFFERS(1, &fb->fbo);
  glDeleteTextures(1, &fb->tex);
  free(fb);

  return GLWIN_SUCCESS; 
}


int glwin_fbo_target_resize(void *voidhandle, void *voidtarget, int wsx, int wsy) {
  oglhandle * handle = (oglhandle *) voidhandle;
  glwin_fbo_target * fb = (glwin_fbo_target *) voidtarget;
  if (handle == NULL || fb == NULL)
    return GLWIN_ERROR;
  glwin_ext_fctns *ext = handle->ext;

#if 0
  printf("\nglwin_fbo_target_resize(): W %d x %d\n", wsx, wsy); 
#endif

  GLBINDFRAMEBUFFER(GL_FRAMEBUFFER, fb->fbo);

  glBindTexture(GL_TEXTURE_2D, fb->tex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, wsx, wsy, 0,
               GL_RGBA, GL_UNSIGNED_BYTE, 0);
  GLFRAMEBUFFERTEXTURE2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 
                         GL_TEXTURE_2D, fb->tex, 0);
  GLBINDRENDERBUFFER(GL_RENDERBUFFER, fb->depth);
  GLRENDERBUFFERSTORAGE(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, wsx, wsy);
  GLFRAMEBUFFERRENDERBUFFER(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, 
                            GL_RENDERBUFFER, fb->depth);
  if (GLCHECKFRAMEBUFFERSTATUS(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    return GLWIN_ERROR;
  }

  fb->drawbufs[0] = GL_COLOR_ATTACHMENT0;
  GLDRAWBUFFERS(1, fb->drawbufs);

  return GLWIN_SUCCESS;
}


void *glwin_fbo_target_create(void *voidhandle, int wsx, int wsy) {
  oglhandle * handle = (oglhandle *) voidhandle;
  if (handle == NULL)
    return NULL;
  glwin_ext_fctns *ext = handle->ext;

  if (!ext->hasglfborendertarget)
    return NULL;  /* fail out if the required GL extensions aren't available */

  glwin_fbo_target *fb = 
    (glwin_fbo_target *) calloc(1, sizeof(glwin_fbo_target));

  if (fb != NULL) {
    /* create target texture */
    glGenTextures(1, &fb->tex);
    glBindTexture(GL_TEXTURE_2D, fb->tex);

    /* set tex mode to replace... */
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    /* we need to use nearest-pixel filtering... */
#if 1
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
#else
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
#endif
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
#if 1
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, wsx, wsy, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, 0);
#else
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, wsx, wsy, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, 0);
#endif
    glBindTexture(GL_TEXTURE_2D, 0); /* XXX may not need this */


    /* create RBO for depth buffer */
    GLGENRENDERBUFFERS(1, &fb->depth);
    GLBINDRENDERBUFFER(GL_RENDERBUFFER, fb->depth);
    GLRENDERBUFFERSTORAGE(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, wsx, wsy);
    GLBINDRENDERBUFFER(GL_RENDERBUFFER, 0); /* XXX not sure if necessary */


    /* create FBO */
    GLGENFRAMEBUFFERS(1, &fb->fbo);
    GLBINDFRAMEBUFFER(GL_FRAMEBUFFER, fb->fbo);

    /* Set "renderedTexture" as our colour attachement #0 */
    GLFRAMEBUFFERTEXTURE2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 
                           GL_TEXTURE_2D, fb->tex, 0);

    /* attach FBO to depth buffer attachment point */
    GLFRAMEBUFFERRENDERBUFFER(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                              GL_RENDERBUFFER, fb->depth);

    if (GLCHECKFRAMEBUFFERSTATUS(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      return NULL;
    }

    /* set the list of draw buffers. */
    fb->drawbufs[0] = GL_COLOR_ATTACHMENT0;
    GLDRAWBUFFERS(1, fb->drawbufs); /* 1 is number of bufs */
 
    /* switch back to window system framebuffer */
    GLBINDFRAMEBUFFER(GL_FRAMEBUFFER, 0);
  }

#if 0
  if (glwin_fbo_target_resize(voidhandle, fb, wsx, wsy) == GLWIN_ERROR) {
    glwin_fbo_target_destroy(voidhandle, fb);
    return NULL;
  }
#endif

  return fb;
}


int glwin_fbo_target_draw_normal(void *voidhandle, void *voidtarget) {
  oglhandle * handle = (oglhandle *) voidhandle;
  glwin_fbo_target * fb = (glwin_fbo_target *) voidtarget;
  if (handle == NULL || fb == NULL)
    return GLWIN_ERROR;
  glwin_ext_fctns *ext = handle->ext;

  GLBINDFRAMEBUFFER(GL_FRAMEBUFFER, 0); /* bind standard framebuffer */

  return GLWIN_SUCCESS;
}


int glwin_fbo_target_draw_fbo(void *voidhandle, void *voidtarget, int wsx, int wsy) {
  oglhandle * handle = (oglhandle *) voidhandle;
  glwin_fbo_target * fb = (glwin_fbo_target *) voidtarget;
  if (handle == NULL || fb == NULL)
    return GLWIN_ERROR;

  /* render to the screen */
  glwin_fbo_target_unbind(voidhandle, voidtarget);

  glBindTexture(GL_TEXTURE_2D, fb->tex);
  glEnable(GL_TEXTURE_2D);
  glColor3f(0.0, 0.0, 1.0);
  glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(0, 0);
    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(0, wsy);
    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(wsx, wsy);
    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(wsx, 0);
  glEnd();
  glDisable(GL_TEXTURE_2D);

  return GLWIN_SUCCESS;
}


/*
 * HMD-specific FBO renderer
 */

#define HMD_DIVCNT 10

static void hmd_compute_warped_coords(int divcnt, int wsx, int wsy, 
                                      float rscale, float wscale,
                                      float *xcrds, float *ycrds,
                                      const float *user_distort_coeff5) {
  float divs = (float) divcnt;
  float hwidth = wsx / 2.0f;
  float hwdiv = hwidth / divs;
  float hdiv = wsy / divs;
  float cx=hwidth / 2.0f;
  float cy=wsy / 2.0f;
  float x, y;

  /* assume Oculus DK2 coefficients if caller doesn't specify */
  const float dk2_warp_coeff[5] = { 1.000f, 0.000f, 0.220f, 0.000f, 0.240f };
#if 0
  const float msr_warp_coeff[5] = { 1.000f, 0.290f, 0.195f, 0.045f, 0.360f };
#endif
  const float *C = dk2_warp_coeff;

  /*
   * use caller-provided distortion correction coefficients when 
   * available, otherwise assume Oculus DK2 
   */
  if (user_distort_coeff5)
    C = user_distort_coeff5;

  int ix, iy;
  for (iy=0; iy<=divcnt; iy++) {
    for (ix=0; ix<=divcnt; ix++) {
      float drx, dry, r, r2, rnew;
      int addr = iy*(divcnt+1) + ix;
      x = ix * hwdiv;
      y = iy * hdiv;

      /* HMD image barrel distortion warping */
      float rnorm = wsy * 1.0f;
      drx = (x - cx) / rnorm;
      dry = (y - cy) / rnorm;
      r2 = drx*drx + dry*dry;
      r = sqrt(r2); 

      rnew = C[0] + r*C[1] + r2*C[2] + r*r2*C[3] + r2*r2*C[4];

      rnew = 1.0f/rnew;
      rnorm *= 1.0f * rscale;
      x = wscale * rnorm * rnew * drx + cx; 
      y =          rnorm * rnew * dry + cy; 

      xcrds[addr] = x;
      ycrds[addr] = y;
    }
  }
}


/* draw lines w/ HMD warp correction to check remaining optical distortions */
static void hmd_draw_eye_lines(int divcnt, int xoff, int width, int height, 
                               float *xcrds, float *ycrds) {
  float x, y;
  int ix, iy;

  glDisable(GL_TEXTURE_2D);
  glColor3f(1.0f, 1.0f, 1.0f);

  for (iy=0; iy<=divcnt; iy++) {
    for (ix=0; ix<divcnt; ix++) {
      int addr = iy*(divcnt+1) + ix;
      y = ycrds[addr];
      x = xcrds[addr] + xoff;
      float xn = xcrds[addr+1] + xoff;
      float yn = ycrds[addr+1];
      glBegin(GL_LINES);
      glVertex2f(x, y);
      glVertex2f(xn, yn);
      glEnd();
    }
  }
  for (ix=0; ix<=divcnt; ix++) {
    for (iy=0; iy<divcnt; iy++) {
      int addr = iy*(divcnt+1) + ix;
      x = xcrds[addr] + xoff;
      y = ycrds[addr];
      float xn = xcrds[addr + divcnt+1] + xoff;
      float yn = ycrds[addr + divcnt+1];
      glBegin(GL_LINES);
      glVertex2f(x, y);
      glVertex2f(xn, yn);
      glEnd();
    }
  }
}


/* draw quads w/ HMD warp correction */
static void hmd_draw_eye_texquads(int divcnt, int xoff, int width, int height, 
                                  float *xcrds, float *ycrds) {
  float divs = (float) divcnt;
  float xtxdiv = 0.5f / divs;
  float ytxdiv = 1.0f / divs;
  float tx, ty;
  int ix, iy;
  float txoff = xoff / ((float) width);

  glBegin(GL_QUADS);
  for (iy=0,ty=1.0f; iy<divcnt; iy++,ty-=ytxdiv) {
    float tyn = ty-ytxdiv;
    for (ix=0,tx=0.0f; ix<divcnt; ix++,tx+=xtxdiv) {
      float txn = tx+xtxdiv;
      int addr = iy*(divcnt+1) + ix;
      float xx0y0 = xcrds[addr] + xoff;
      float yx0y0 = ycrds[addr];
      float xx1y0 = xcrds[addr + 1] + xoff;
      float yx1y0 = ycrds[addr + 1];
      float xx0y1 = xcrds[addr     + divcnt+1] + xoff;
      float yx0y1 = ycrds[addr     + divcnt+1];
      float xx1y1 = xcrds[addr + 1 + divcnt+1] + xoff;
      float yx1y1 = ycrds[addr + 1 + divcnt+1];

      glTexCoord2f(tx+txoff, ty);
      glVertex2f(xx0y0, yx0y0);
      glTexCoord2f(tx+txoff, tyn);
      glVertex2f(xx0y1, yx0y1);
      glTexCoord2f(txn+txoff, tyn);
      glVertex2f(xx1y1, yx1y1);
      glTexCoord2f(txn+txoff, ty);
      glVertex2f(xx1y0, yx1y0);
    }
  }
  glEnd();
}


/*
 * Structures for HMD spheremap display and image warping for
 * eye lens distortion correction
 */

typedef struct {
  void *hmd_fbo;
  int divcnt;
  int wsx;       /* window dimensions */
  int wsy;   
  int wrot;      /* flag indicating that window is rotated vs. HMD image */
  int ixs;       /* image size */
  int iys;       
  float *xcrds;  /* use if there chromatic aberration is unnecessary */
  float *ycrds;

  /* chromatic aberration correction */
  float *Rxcrds;
  float *Rycrds;
  float *Gxcrds;
  float *Gycrds;
  float *Bxcrds;
  float *Bycrds;
} glwin_warp_hmd;


void glwin_spheremap_update_hmd_warp(void *vwin, void *voidwarp, 
                                     int wsx, int wsy, 
                                     int warpdivs, int ixs, int iys,
                                     const float *barrel_coeff, int force) {
  glwin_warp_hmd * warp = (glwin_warp_hmd *) voidwarp;
 
  if (force || warp->divcnt!=warpdivs || warp->wsx!=wsx || warp->wsy!=wsy) {
    const float Oculus_DK2_coeff[4] = { 1.0f, 0.22f, 0.24f, 0.0f };
    /* const float Oculus_DK1_coeff[4] = { 1.0f, 0.18f, 0.115f, 0.0f }; */
    if (!barrel_coeff) 
      barrel_coeff = Oculus_DK2_coeff;

#if 0
    printf("glwin_spheremap_update_hmd_warp(): W %d x %d, I %d x %d\n", 
           wsx, wsy, ixs, iys); 
    printf("warp: %.3f, %.3f, %.3f, %.3f\n",
           barrel_coeff[0], barrel_coeff[1], barrel_coeff[2], barrel_coeff[3]);
#endif


    /* update FBO target for new window size */
    if (glwin_fbo_target_resize(vwin, warp->hmd_fbo, wsx, wsy) == GLWIN_ERROR) {
      printf("\nglwin_spheremap_update_hmd_warp(): "
             "an error occured resizing the FBO!\n");
    }

    /*
     * recompute the warp mesh 
     */
    if (warp->xcrds != NULL)
      free(warp->xcrds);
    if (warp->ycrds != NULL)
      free(warp->ycrds);

    if (warp->Rxcrds != NULL)
      free(warp->Rxcrds);
    if (warp->Rycrds != NULL)
      free(warp->Rycrds);
    if (warp->Gxcrds != NULL)
      free(warp->Gxcrds);
    if (warp->Gycrds != NULL)
      free(warp->Gycrds);
    if (warp->Bxcrds != NULL)
      free(warp->Bxcrds);
    if (warp->Bycrds != NULL)
      free(warp->Bycrds);

    warp->wsx = wsx;
    warp->wsy = wsy;
    warp->divcnt = warpdivs;
    warp->xcrds  = (float *) calloc(1, warpdivs*warpdivs*sizeof(float)); 
    warp->ycrds  = (float *) calloc(1, warpdivs*warpdivs*sizeof(float)); 
    warp->Rxcrds = (float *) calloc(1, warpdivs*warpdivs*sizeof(float)); 
    warp->Rycrds = (float *) calloc(1, warpdivs*warpdivs*sizeof(float)); 
    warp->Gxcrds = (float *) calloc(1, warpdivs*warpdivs*sizeof(float)); 
    warp->Gycrds = (float *) calloc(1, warpdivs*warpdivs*sizeof(float)); 
    warp->Bxcrds = (float *) calloc(1, warpdivs*warpdivs*sizeof(float)); 
    warp->Bycrds = (float *) calloc(1, warpdivs*warpdivs*sizeof(float)); 

    /* plain image w/ no chromatic aberration correction */
    hmd_compute_warped_coords(warpdivs-1, wsx, wsy, 1.0f, 1.0f, 
                              warp->xcrds, warp->ycrds, barrel_coeff);

    /* set of RGB meshes for chromatic aberration correction */
    const float Rscale  = 1.015f;
    const float Gscale  = 1.000f;
    const float Bscale  = 0.980f;

    hmd_compute_warped_coords(warpdivs-1, wsx,wsy, Rscale, 1.0f, 
                              warp->Rxcrds, warp->Rycrds, barrel_coeff);
    hmd_compute_warped_coords(warpdivs-1, wsx,wsy, Gscale, 1.0f, 
                              warp->Gxcrds, warp->Gycrds, barrel_coeff);
    hmd_compute_warped_coords(warpdivs-1, wsx,wsy, Bscale, 1.0f, 
                              warp->Bxcrds, warp->Bycrds, barrel_coeff);
  }
}


void glwin_spheremap_destroy_hmd_warp(void *vwin, void *voidwarp) {
  glwin_warp_hmd * warp = (glwin_warp_hmd *) voidwarp;
  glwin_fbo_target_destroy(vwin, warp->hmd_fbo);

  if (warp->xcrds != NULL)
    free(warp->xcrds);
  if (warp->ycrds != NULL)
    free(warp->ycrds);

  if (warp->Rxcrds != NULL)
    free(warp->Rxcrds);
  if (warp->Rycrds != NULL)
    free(warp->Rycrds);
  if (warp->Gxcrds != NULL)
    free(warp->Gxcrds);
  if (warp->Gycrds != NULL)
    free(warp->Gycrds);
  if (warp->Bxcrds != NULL)
    free(warp->Bxcrds);
  if (warp->Bycrds != NULL)
    free(warp->Bycrds);
  free(warp);
}


void * glwin_spheremap_create_hmd_warp(void *vwin, int wsx, int wsy, int wrot,
                                       int warpdivs, int ixs, int iys,
                                       const float *user_coeff) {
  glwin_warp_hmd *warp = (glwin_warp_hmd *) calloc(1, sizeof(glwin_warp_hmd));
  warp->hmd_fbo = glwin_fbo_target_create(vwin, wsx, wsy);
  warp->wrot = wrot;
  glwin_spheremap_update_hmd_warp(vwin, warp, wsx, wsy, warpdivs, 
                                  ixs, iys, user_coeff, 1);
  return warp;
}


int glwin_spheremap_draw_hmd_warp(void *vwin, void *voidwarp, 
                                  int drawimage, int drawlines, int chromcorr,
                                  int wsx, int wsy, int ixs, int iys, 
                                  const float *hmdquat,
                                  float fov, float rad, int hmd_spres) {
  oglhandle * handle = (oglhandle *) vwin;
  glwin_warp_hmd * warp = (glwin_warp_hmd *) voidwarp;
  glwin_fbo_target * fb = (glwin_fbo_target *) warp->hmd_fbo;
  if (handle == NULL || warp == NULL)
    return GLWIN_ERROR;

  glBindTexture(GL_TEXTURE_2D, 0); /* bind the RT image before drawing */
  glEnable(GL_TEXTURE_2D);

  glwin_fbo_target_bind(vwin, warp->hmd_fbo);
  glwin_spheremap_draw_tex(vwin, GLWIN_STEREO_OVERUNDER, 
                           ixs, iys, hmdquat, fov, rad, hmd_spres);
  glwin_fbo_target_unbind(vwin, warp->hmd_fbo); /* render to the screen */

  glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
  glClearColor(0.0, 0.0, 0.0, 1.0); /* black */
  glViewport(0, 0, wsx, wsy);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glShadeModel(GL_SMOOTH);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.0, wsx, wsy, 0.0, -1.0, 1.0);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  float hw = wsx * 0.5f;
  int dm1 = warp->divcnt-1;


  /*
   * draw texture map FBO using precomputed warp meshes
   */
  if (drawimage) {
    glBindTexture(GL_TEXTURE_2D, fb->tex);
    glEnable(GL_TEXTURE_2D);

    /* if chromatic abberation correction is enabled, use the RGB meshes */
    if (chromcorr) {
      glColorMask(GL_TRUE, GL_FALSE, GL_FALSE, GL_TRUE);
      hmd_draw_eye_texquads(dm1,  0, wsx, wsy, warp->Rxcrds, warp->Rycrds);
      hmd_draw_eye_texquads(dm1, hw, wsx, wsy, warp->Rxcrds, warp->Rycrds);
      glColorMask(GL_FALSE, GL_TRUE, GL_FALSE, GL_TRUE);
      hmd_draw_eye_texquads(dm1,  0, wsx, wsy, warp->Gxcrds, warp->Gycrds);
      hmd_draw_eye_texquads(dm1, hw, wsx, wsy, warp->Gxcrds, warp->Gycrds);
      glColorMask(GL_FALSE, GL_FALSE, GL_TRUE, GL_TRUE);
      hmd_draw_eye_texquads(dm1,  0, wsx, wsy, warp->Bxcrds, warp->Bycrds);
      hmd_draw_eye_texquads(dm1, hw, wsx, wsy, warp->Bxcrds, warp->Bycrds);
      glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    } else {
      hmd_draw_eye_texquads(dm1,  0, wsx, wsy, warp->xcrds, warp->ycrds);
      hmd_draw_eye_texquads(dm1, hw, wsx, wsy, warp->xcrds, warp->ycrds);
    }
  } 
  glDisable(GL_TEXTURE_2D);

  /*
   * draw warp mesh grid lines over the top of the eye images if requested
   */
  if (drawlines) {
    /* if chromatic abberation correction is enabled, use the RGB meshes */
    if (chromcorr) {
      glColorMask(GL_TRUE, GL_FALSE, GL_FALSE, GL_TRUE);
      hmd_draw_eye_lines(dm1,  0, wsx, wsy, warp->Rxcrds, warp->Rycrds);
      hmd_draw_eye_lines(dm1, hw, wsx, wsy, warp->Rxcrds, warp->Rycrds);
      glColorMask(GL_FALSE, GL_TRUE, GL_FALSE, GL_TRUE);
      hmd_draw_eye_lines(dm1,  0, wsx, wsy, warp->Gxcrds, warp->Gycrds);
      hmd_draw_eye_lines(dm1, hw, wsx, wsy, warp->Gxcrds, warp->Gycrds);
      glColorMask(GL_FALSE, GL_FALSE, GL_TRUE, GL_TRUE);
      hmd_draw_eye_lines(dm1,  0, wsx, wsy, warp->Bxcrds, warp->Bycrds);
      hmd_draw_eye_lines(dm1, hw, wsx, wsy, warp->Bxcrds, warp->Bycrds);
      glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    } else {
      hmd_draw_eye_lines(dm1,  0, wsx, wsy, warp->xcrds, warp->ycrds);
      hmd_draw_eye_lines(dm1, hw, wsx, wsy, warp->xcrds, warp->ycrds);
    }
  }

  return GLWIN_SUCCESS;
}




/*
 * GLSL support routines
 */

static void glwin_print_glsl_infolog(void *voidhandle, GLhandleARB obj, 
                                     const char *msg) {
  oglhandle *handle = (oglhandle *) voidhandle;
  if (handle == NULL)
    return;
  glwin_ext_fctns *ext = handle->ext;

  GLint blen = 0;   /* length of buffer to allocate      */
  GLint slen = 0;   /* strlen actually written to buffer */
  GLcharARB *infoLog;

  GLGETOBJECTPARAMETERIVARB(obj, GL_OBJECT_INFO_LOG_LENGTH_ARB , &blen);
  if (blen > 1) {
    if ((infoLog = (GLcharARB *) calloc(1, blen)) == NULL) {
      printf("GLSL shader compiler could not allocate InfoLog buffer\n");
      return;
    }

    GLGETINFOLOGARB(obj, blen, &slen, infoLog);
    printf("  %s\n", msg);
    printf("    %s\n", (char *) infoLog);
    free(infoLog);
  }
}


static int glwin_compile_shaders(void *voidhandle, glsl_shader *sh, 
                                 const GLubyte *vertexShader, 
                                 const GLubyte *fragmentShader,
                                 int verbose) {
  oglhandle *handle = (oglhandle *) voidhandle;
  glwin_ext_fctns *ext = handle->ext;

  GLint  vert_compiled = 0;
  GLint  frag_compiled = 0;
  GLint shaders_linked = 0;
  GLint length;

  /* clear shader structure before proceeding */ 
  memset(sh, 0, sizeof(glsl_shader));
 
  /* bail out if we don't have valid pointers for shader source code */
  if (vertexShader == NULL || fragmentShader == NULL) {
    return GLWIN_ERROR;
  }

  /* Hand the source code strings to OpenGL. */
  length = strlen((const char *) vertexShader);
  GLSHADERSOURCEARB(sh->VertexShaderObject, 1, (const char **) &vertexShader, &length);

  length = strlen((const char *) fragmentShader);
  GLSHADERSOURCEARB(sh->FragmentShaderObject, 1, (const char **) &fragmentShader, &length);

  /* Compile the vertex and fragment shader, and print out */
  /* the compiler log file if one is available.            */
  GLCOMPILESHADERARB(sh->VertexShaderObject);
  GLGETOBJECTPARAMETERIVARB(sh->VertexShaderObject,
                            GL_OBJECT_COMPILE_STATUS_ARB, &vert_compiled);

  if (verbose)
    glwin_print_glsl_infolog(voidhandle, sh->VertexShaderObject, "OpenGL vertex shader compilation log: ");

  GLCOMPILESHADERARB(sh->FragmentShaderObject);
  GLGETOBJECTPARAMETERIVARB(sh->FragmentShaderObject,
                  GL_OBJECT_COMPILE_STATUS_ARB, &frag_compiled);

  if (verbose)
    glwin_print_glsl_infolog(voidhandle, sh->FragmentShaderObject, "OpenGL fragment shader compilation log: ");

  if (vert_compiled && frag_compiled) {
    /* Populate the program object with the two compiled shaders */
    GLATTACHOBJECTARB(sh->ProgramObject, sh->VertexShaderObject);
    GLATTACHOBJECTARB(sh->ProgramObject, sh->FragmentShaderObject);

    /* Link the whole thing together and print out the linker log file */
    GLLINKPROGRAMARB(sh->ProgramObject);
    GLGETOBJECTPARAMETERIVARB(sh->ProgramObject, GL_OBJECT_LINK_STATUS_ARB, &shaders_linked);

    if (verbose)
      glwin_print_glsl_infolog(voidhandle, sh->ProgramObject, "OpenGL shader linkage log: " );
  }

  /* We want the shaders to go away as soon as they are detached from   */
  /* the program object (or program objects) they are attached to. We   */
  /* can simply call delete now to achieve that. Note that calling      */
  /* delete on a program object will result in all shaders attached to  */
  /* that program object to be detached. If delete has been called for  */
  /* these shaders, calling delete on the program object will result in */
  /* the shaders being deleted as well.                                 */
  if (vert_compiled)
    GLDELETEOBJECTARB(sh->VertexShaderObject);
  if (frag_compiled)
    GLDELETEOBJECTARB(sh->FragmentShaderObject);

  if (vert_compiled && frag_compiled && shaders_linked) {
    sh->isvalid = 1;
    return GLWIN_SUCCESS;
  } else {
    memset(sh, 0, sizeof(glsl_shader));
    return GLWIN_ERROR;
  }
}


int glwin_destroy_shaders(void *voidhandle, glsl_shader *sh) {
  oglhandle *handle = (oglhandle *) voidhandle;
  glwin_ext_fctns *ext = handle->ext;
 
  /* destroy the GLSL shaders and associated state */
  if (sh->isvalid) {
    GLDELETEOBJECTARB(sh->ProgramObject);
    memset(sh, 0, sizeof(glsl_shader));

    return GLWIN_SUCCESS;
  }

  return GLWIN_ERROR;
}


/*
 * GLSL vertex and fragment shaders for HMD rendering
 */
const char *hmd_vert = 
  "// requires GLSL version 1.10                                           \n"
  "#version 110                                                            \n"
  "                                                                        \n"
  "                                                                        \n"
  "                                                                        \n"
  "void main(void) {                                                       \n"
  "  // transform vertex to Eye space for user clipping plane calculations \n"
  "  vec4 ecpos = gl_ModelViewMatrix * gl_Vertex;                          \n"
  "  gl_ClipVertex = ecpos;                                                \n"
  "                                                                        \n"
  "  // transform, normalize, and output normal.                           \n"
  "  oglnormal = normalize(gl_NormalMatrix * gl_Normal);                   \n"
  "                                                                        \n"
  "  // pass along vertex color for use fragment shading,                  \n"
  "  // fragment shader will get an interpolated color.                    \n"
  "  oglcolor = vec3(gl_Color);                                            \n"
  "                                                                        \n"
  "                                                                        \n"
#if 1
  "  gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;               \n"
#else
  "  gl_Position = ftransform();                                           \n"
#endif
  "                                                                        \n"
  "                                                                        \n"
  "                                                                        \n"
  "                                                                        \n"
  "}                                                                       \n"
  "                                                                        \n";


const char *hmd_frag = 
  "                                                                        \n"
  "                                                                        \n"
  "                                                                        \n"
  "                                                                        \n"
  "void main(void) {                                                       \n"
  "                                                                        \n"
#if 1
  "  // Flip the surface normal if it is facing away from the viewer,      \n"
  "  // determined by polygon winding order provided by OpenGL.            \n"
  "  vec3 N = normalize(oglnormal);                                        \n"
  "  if (!gl_FrontFacing) {                                                \n"
  "    N = -N;                                                             \n"
  "  }                                                                     \n"
#endif
  "                                                                        \n"
  "                                                                        \n"
  "                                                                        \n"
  "                                                                        \n"
  "                                                                        \n"
  "                                                                        \n"
  "}                                                                       \n"
  "                                                                        \n";


int glwin_compile_hmd_shaders(void *voidhandle, glsl_shader *sh) {
  int rc = glwin_compile_shaders(voidhandle, sh, 
                                 (GLubyte *) hmd_vert, (GLubyte *) hmd_frag, 1);
  return rc;
}

#endif


void glwin_draw_image(void * voidhandle, int ixs, int iys, unsigned char * img) {
  glRasterPos2i(0, 0);
  glDrawPixels(ixs, iys, GL_RGB, GL_UNSIGNED_BYTE, img);
  glwin_swap_buffers(voidhandle);
}


void glwin_draw_image_rgb3u(void *voidhandle, int stereomode, int ixs, int iys,
                            const unsigned char *rgb3u) {
  int wxs=0, wys=0;
  glwin_get_winsize(voidhandle, &wxs, &wys);
  glViewport(0, 0, wxs, wys);

  glDrawBuffer(GL_BACK);
  glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
  glClearColor(0.0, 0.0, 0.0, 1.0); /* black */
  glClear(GL_COLOR_BUFFER_BIT);

  glShadeModel(GL_FLAT);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.0, wxs, 0.0, wys, -1.0, 1.0);
  glMatrixMode(GL_MODELVIEW);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glPixelZoom(1.0, 1.0);

  if (stereomode == GLWIN_STEREO_OVERUNDER) {
    /* printf("wsz: %dx%d  bsz: %dx%d\n", wxs, wys, ixs, iys); */
    const unsigned char *leftimg = rgb3u;
    const unsigned char *rightimg = leftimg + ((ixs * (iys/2)) * 4);

    glDrawBuffer(GL_BACK_LEFT);
    glRasterPos2i(0, 0);
#if 0
    glColorMask(GL_TRUE, GL_TRUE, GL_FALSE, GL_TRUE); /* anaglyph or testing */
#endif
    glDrawPixels(ixs, iys/2, GL_RGBA, GL_UNSIGNED_BYTE, leftimg);

    glDrawBuffer(GL_BACK_RIGHT);
    glRasterPos2i(0, 0);
#if 0
    glColorMask(GL_FALSE, GL_TRUE, GL_TRUE, GL_TRUE); /* anaglyph or testing */
#endif
    glDrawPixels(ixs, iys/2, GL_RGBA, GL_UNSIGNED_BYTE, rightimg);
  } else {
    glRasterPos2i(0, 0);
    glDrawPixels(ixs, iys, GL_RGBA, GL_UNSIGNED_BYTE, rgb3u);
  }

  glwin_swap_buffers(voidhandle);
}


void glwin_draw_image_tex_rgb3u(void *voidhandle, 
                                int stereomode, int ixs, int iys,
                                const unsigned char *rgb3u) {
  int wxs=0, wys=0;
  glwin_get_winsize(voidhandle, &wxs, &wys);
  glViewport(0, 0, wxs, wys);

  glDrawBuffer(GL_BACK);
  glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
  glClearColor(0.0, 0.0, 0.0, 1.0); /* black */
  glClear(GL_COLOR_BUFFER_BIT);

  glShadeModel(GL_FLAT);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.0, wxs, 0.0, wys, -1.0, 1.0);
  glMatrixMode(GL_MODELVIEW);

  GLuint texName = 0;
  GLfloat texborder[4] = {0.0, 0.0, 0.0, 1.0};
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glBindTexture(GL_TEXTURE_2D, texName);

  /* black borders if we go rendering anything beyond texture coordinates */
  glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, texborder);
#if defined(GL_CLAMP_TO_BORDER)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
#else
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
#endif

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

  glLoadIdentity();
  glColor3f(1.0, 1.0, 1.0);

  if (stereomode == GLWIN_STEREO_OVERUNDER) {
    const unsigned char *leftimg = rgb3u;
    const unsigned char *rightimg = leftimg + ((ixs * (iys/2)) * 4);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ixs, iys, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, leftimg);
    glEnable(GL_TEXTURE_2D);

    glDrawBuffer(GL_BACK_LEFT);
#if 0
    glColorMask(GL_TRUE, GL_TRUE, GL_FALSE, GL_TRUE); /* anaglyph or testing */
#endif
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(0, 0);
    glTexCoord2f(0.0f, 0.5f);
    glVertex2f(0, wys);
    glTexCoord2f(1.0f, 0.5f);
    glVertex2f(wxs, wys);
    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(wxs, 0);
    glEnd();

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ixs, iys, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, rightimg);
    glEnable(GL_TEXTURE_2D);

    glDrawBuffer(GL_BACK_RIGHT);
#if 0
    glColorMask(GL_FALSE, GL_TRUE, GL_TRUE, GL_TRUE); /* anaglyph or testing */
#endif
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.5f);
    glVertex2f(0, 0);
    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(0, wys);
    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(wxs, wys);
    glTexCoord2f(1.0f, 0.5f);
    glVertex2f(wxs, 0);
    glEnd();
  } else {
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ixs, iys, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, rgb3u);
    glEnable(GL_TEXTURE_2D);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(0, 0);
    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(0, wys);
    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(wxs, wys);
    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(wxs, 0);
    glEnd();
  }

  glDisable(GL_TEXTURE_2D);

  glwin_swap_buffers(voidhandle);
}


/*
 * OpenGL texture mapped sphere rendering code needed for
 * display of spheremap textures w/ HMDs.
 * The texture must wraparound in the X/longitudinal dimension 
 * for the sphere to be drawn correctly.
 * The texture Y/latitude dimension doesn't have to wrap and this
 * code allows a single texture containing multiple images to be used
 * for drawing multiple spheres by offsetting the txlatstart/txlatend.
 */
#define SPHEREMAXRES 64
void glwin_draw_sphere_tex(float rad, int res, float txlatstart, float txlatend) {
  int i, j;
  float zLo, zHi, res_1;
  
  float sinLong[SPHEREMAXRES];
  float cosLong[SPHEREMAXRES];
  float sinLatVert[SPHEREMAXRES];
  float cosLatVert[SPHEREMAXRES];
  float sinLatNorm[SPHEREMAXRES];
  float cosLatNorm[SPHEREMAXRES];
  float texLat[SPHEREMAXRES];
  float texLong[SPHEREMAXRES];

  /* compute length of texture from start */
  float txlatsz = txlatend - txlatstart;

  if (res < 2)
    res = 2;

  if (res >= SPHEREMAXRES)
    res = SPHEREMAXRES-1;

  res_1 = 1.0f / res;
  
  /* longitudinal "slices" */
  float ang_twopi_res = 6.28318530718 * res_1;
  for (i=0; i<res; i++) {
    float angle = i * ang_twopi_res;
    sinLong[i] = sinf(angle);
    cosLong[i] = cosf(angle);
    texLong[i] = (res-i) * res_1;
  }
  /* ensure that longitude end point exactly matches start */
  sinLong[res] = 0.0f; /* sinLong[0]; */
  cosLong[res] = 1.0f; /* cosLong[0]; */
  texLong[res] = 0.0f;

  /* latitude "stacks" */
  float ang_pi_res = 3.14159265359 * res_1;
  for (i=0; i<=res; i++) {
      float angle = i * ang_pi_res;
    sinLatNorm[i] = sinf(angle);
    cosLatNorm[i] = cosf(angle);
    sinLatVert[i] = rad * sinLatNorm[i];
    cosLatVert[i] = rad * cosLatNorm[i];
        texLat[i] = txlatstart + (i * res_1 * txlatsz);
  }
  /* ensure top and bottom poles come to points */
  sinLatVert[0] = 0;
  sinLatVert[res] = 0;

  for (j=0; j<res; j++) {
    zLo = cosLatVert[j];
    zHi = cosLatVert[j+1];

    float stv1 = sinLatVert[j];
    float stv2 = sinLatVert[j+1];

    float stn1 = sinLatNorm[j];
    float ctn1 = cosLatNorm[j];
    float stn2 = sinLatNorm[j+1];
    float ctn2 = cosLatNorm[j+1];

    glBegin(GL_QUAD_STRIP);
    for (i=0; i<=res; i++) {
      glNormal3f(sinLong[i] * stn2, cosLong[i] * stn2, ctn2);
      glTexCoord2f(texLong[i], texLat[j+1]);
      glVertex3f(stv2 * sinLong[i], stv2 * cosLong[i], zHi);

      glNormal3f(sinLong[i] * stn1, cosLong[i] * stn1, ctn1);
      glTexCoord2f(texLong[i], texLat[j]);
      glVertex3f(stv1 * sinLong[i], stv1 * cosLong[i], zLo);
    }
    glEnd();
  }
}


void glwin_spheremap_draw_prepare(void *voidhandle) {
  glShadeModel(GL_FLAT);
  glDepthFunc(GL_LEQUAL);
  glEnable(GL_DEPTH_TEST);    /* use Z-buffer for hidden-surface removal */
  glClearDepth(1.0);

  glDrawBuffer(GL_BACK);
  glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
  glClearColor(0.0, 0.0, 0.0, 1.0); /* black */
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  GLuint texName = 0;
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glBindTexture(GL_TEXTURE_2D, texName);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
}


void glwin_spheremap_upload_tex_rgb3u(void *voidhandle, int ixs, int iys,
                                      const unsigned char *rgb3u) {
  glDisable(GL_TEXTURE_2D);
  GLuint texName = 0;
  glBindTexture(GL_TEXTURE_2D, texName);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ixs, iys, 0,
               GL_RGBA, GL_UNSIGNED_BYTE, rgb3u);
  glEnable(GL_TEXTURE_2D);
}


void glwin_spheremap_draw_tex(void *voidhandle, int stereomode, 
                              int ixs, int iys, const float *hmdquat,
                              float fov, float rad, int res) {
  int wxs=0, wys=0;
  float n, f, a, t, b, r, l;

  glwin_get_winsize(voidhandle, &wxs, &wys);
  glViewport(0, 0, wxs, wys); /* clear entire window prior to rendering */

  glDrawBuffer(GL_BACK);
  glClearColor(0.0, 0.0, 0.0, 1.0); /* black */
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  n = 1.0f;      /* near clipping plane */
  f = 15.0f;     /* far clipping plane  */
  a = wxs / wys; /* window aspect ratio */
  t = n * tanf(fov * 3.14159265359 / (180.0f*2.0f)); /* top */
  b = -t;        /* bottom */
  r = a * t;     /* right */
  l = -r;        /* left */
  glFrustum(l, r, b, t, n, f);
 
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  if (hmdquat != NULL) {
    float hmdmat[16];
    quat_rot_matrix(hmdmat, hmdquat);
    glMultMatrixf(hmdmat);
  }

  glRotatef(90.0f, 1.0f, 0.0f, 0.0f);
  glColor3f(1.0, 1.0, 1.0);

  /* window dims control viewport size, image dims control tex size, */
  /* since we will render from a spheremap that has different dims   */
  /* than the target window                                          */
  if (stereomode == GLWIN_STEREO_OVERUNDER) {
    /* right image is stored first */
    glViewport(wxs/2, 0, wxs/2, wys);
    glwin_draw_sphere_tex(rad, res, 0.0f, 0.5f);

    /* left image is stored second */
    glViewport(0, 0, wxs/2, wys);
    glwin_draw_sphere_tex(rad, res, 0.5f, 1.0f);
  } else {
    glwin_draw_sphere_tex(rad, res, 0.0f, 1.0f);
  }
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

#if 0
  if (handle) { 
    /* check window size */
    XWindowAttributes xwa;
    XGetWindowAttributes(handle->dpy, handle->win, &xwa);
    handle->width = xwa.width;
    handle->height = xwa.height;
  }
#endif

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

int glwin_query_extension(const char *extname) {
  return 0;
}

int glwin_query_vsync(void *voidhandle, int *onoff) {
  return GLWIN_NOT_IMPLEMENTED;
}

void glwin_draw_image(void * voidhandle, int xsize, int ysize, unsigned char * img) {
  return;
}

void glwin_draw_image_rgb3u(void *voidhandle, int stereomode, int ixs, int iys,
                            const unsigned char *rgb3u) {
  return;
}

void glwin_draw_image_tex_rgb3u(void *voidhandle, 
                                int stereomode, int ixs, int iys,
                                const unsigned char *rgb3u) {
  return;
}

void glwin_spheremap_draw_prepare(void *voidhandle) {
  return;
}

void glwin_spheremap_upload_tex_rgb3u(void *voidhandle, int ixs, int iys,
                                      const unsigned char *rgb3u) {
  return;
}

void glwin_draw_sphere_tex(float rad, int res, float txlatstart, float txlatend) {
  return;
}

void glwin_spheremap_draw(void *voidhandle, int stereomode, int ixs, int iys,
                          const float *hmdquat, float fov, float rad, int res) {
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

