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
 *	$RCSfile: OpenGLExtensions.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.55 $	$Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Class to store and handle enumeration and initialization of OpenGL 
 *   extensions and features.
 ***************************************************************************/
#ifndef OPENGLEXTENSIONS_H
#define OPENGLEXTENSIONS_H

#if defined(_MSC_VER)
#include <windows.h>
#endif

// The Linux OpenGL ABI 1.0 spec requires that we define
// GL_GLEXT_PROTOTYPES before including gl.h or glx.h for extensions
// in order to get prototypes:
//   http://oss.sgi.com/projects/ogl-sample/ABI/index.html
#define GL_GLEXT_PROTOTYPES   1
#define GLX_GLXEXT_PROTOTYPES 1

#if defined(__APPLE__) && !defined (VMDMESA) 
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

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
#if defined(__APPLE__) && !defined(VMDMESA) 
#include <OpenGL/glext.h>
#endif

// Add support for EGL contexts, eglGetProcAddress() 
#if defined(VMDEGLPBUFFER)
#include <EGL/egl.h>
#endif

// required for Win32 calling conventions to work correctly
#ifndef APIENTRY
#define APIENTRY
#endif
#ifndef GLAPI
#define GLAPI extern
#endif

// XXX enable OpenGL Shading Language support if it is available in the headers
#if 1 && defined(GL_ARB_shader_objects)
#define VMDUSEOPENGLSHADER 1
#endif

// prevent vendor-specific header file clashes
typedef void (APIENTRY *VMDGLXextFuncPtr)(void);

// XXX Newer OpenGL extensions cause problems on Linux/Windows/Mac because
//     they don't gaurantee runtime linkage, even for ARB extensions.
//     To use them, we must look them up at runtime with vmdProcAddress()
//      which wraps display- and system-dependent methods for doing so.
#if defined(_MSC_VER) || defined(__APPLE__) || defined(__irix) || (!defined(ARCH_SOLARIS2) && !defined(ARCH_SOLARIS2_64))

#define VMDUSEGETPROCADDRESS 1
#define GLLOCKARRAYSEXT           ext->p_glLockArraysEXT
#define GLUNLOCKARRAYSEXT         ext->p_glUnlockArraysEXT
#define GLMULTIDRAWELEMENTSEXT    ext->p_glMultiDrawElementsEXT
#define GLPOINTPARAMETERFARB      ext->p_glPointParameterfARB
#define GLPOINTPARAMETERFVARB     ext->p_glPointParameterfvARB

// OpenGL Shader Functions
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

#else

#define GLLOCKARRAYSEXT           glLockArraysEXT
#define GLUNLOCKARRAYSEXT         glUnlockArraysEXT
#define GLMULTIDRAWELEMENTSEXT    glMultiDrawElementsEXT
#define GLPOINTPARAMETERFARB      glPointParameterfARB
#define GLPOINTPARAMETERFVARB     glPointParameterfvARB

// OpenGL Shader Functions
#define GLCREATESHADEROBJECTARB   glCreateShaderObjectARB
#define GLCREATEPROGRAMOBJECTARB  glCreateProgramObjectARB
#define GLUSEPROGRAMOBJECTARB     glUseProgramObjectARB
#define GLDETACHOBJECTARB         glDetachObjectARB
#define GLGETINFOLOGARB           glGetInfoLogARB
#define GLGETOBJECTPARAMETERIVARB glGetObjectParameterivARB
#define GLLINKPROGRAMARB          glLinkProgramARB
#define GLDELETEOBJECTARB         glDeleteObjectARB
#define GLATTACHOBJECTARB         glAttachObjectARB
#define GLCOMPILESHADERARB        glCompileShaderARB
#define GLSHADERSOURCEARB         glShaderSourceARB
#define GLGETUNIFORMLOCATIONARB   glGetUniformLocationARB
#define GLUNIFORM1IARB            glUniform1iARB
#define GLUNIFORM1FVARB           glUniform1fvARB
#define GLUNIFORM2FVARB           glUniform2fvARB
#define GLUNIFORM3FVARB           glUniform3fvARB
#define GLUNIFORM4FVARB           glUniform4fvARB

#endif


// special case Linux and MacOS X as platforms that fail to use glProcAddress()
// on core functions like glTexImage3D().
#if defined(VMDUSEGETPROCADDRESS) && !defined(__linux) && !defined(__APPLE__)
#define GLTEXIMAGE3D              ext->p_glTexImage3D
#else
#define GLTEXIMAGE3D              glTexImage3D
#endif


/// Manages the use of OpenGL extensions, provides queries, 
/// OS-specific function pointer setup, and some OpenGL state management.
class OpenGLExtensions {
private:
  void *gllibraryhandle;

public: 
  // OpenGL buffers, extensions, and bonus features found on this display 
  int hasstereo;                 ///< whether we have stereo capable buffer
  int stereodrawforced;          ///< must always draw in stereo, buggy driver
  int hasmultisample;            ///< whether we have multisample extension
  int nummultisamples;           ///< number of multisample samples available
  int hasstencilbuffer;          ///< whether display has a stencil buffer
  int hastex2d;                  ///< whether renderer supports 2-D texturing
  int hastex3d;                  ///< whether renderer supports 3-D texturing
  int hasmultidrawext;           ///< ARB/Sun GL_EXT_multi_draw 
  int hascompiledvertexarrayext; ///< ARB GL_EXT_compiled_vertex_array 
  int hasrescalenormalext;       ///< ARB GL_RESCALE_NORMAL_EXT 
  GLint multitextureunits;       ///< number of multitexture texture units 
  int hasglarbtexnonpoweroftwo;  ///< OpenGL non-power-of-two texture ARB ext
  int hasglpointparametersext;   ///< glPointParameterfvARB
  int hasglpointspritearb;       ///< OpenGL point sprite ARB extension
  int hasglshadinglangarb;       ///< OpenGL Shading Language ARB extension
  int hasglshaderobjectsarb;     ///< OpenGL Shader Objects ARB extension
  int hasglvertexshaderarb;      ///< OpenGL Vertex Shader ARB extension
  int hasglfragmentshaderarb;    ///< OpenGL Fragment Shader ARB extension
  int hasglgeometryshader4arb;   ///< OpenGL Geometry Shader ARB extension
  int hasglsampleshadingarb;     ///< OpenGL Sample Shading ARB extension
 
  // OpenGL function pointers
  void (APIENTRY *p_glLockArraysEXT)(GLint, GLsizei);
  void (APIENTRY *p_glUnlockArraysEXT)(void);
  void (APIENTRY *p_glMultiDrawElementsEXT)(GLenum, const GLsizei *, GLenum, const GLvoid* *, GLsizei); 
  void (APIENTRY *p_glGlobalAlphaFactorfSUN)(GLfloat);
  void (APIENTRY *p_glPointParameterfARB)(GLenum, GLfloat);
  void (APIENTRY *p_glPointParameterfvARB)(GLenum, const GLfloat *);
  void (APIENTRY *p_glTexImage3D)(GLenum, GLint, GLint, 
                                  GLsizei, GLsizei, GLsizei, GLint, 
                                  GLenum, GLenum, const GLvoid *);


//
// Only enable OpenGL Shader code when we find exensions in the headers
//
#if defined(GL_ARB_shader_objects)
  // OpenGL Shader Function Pointers
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
#endif

  enum rendenum { ATI, NVIDIA, MSOFTGDI, MESAGL, WIREGL, INTELSWR, GENERIC };

  // OpenGL Renderer version information
  int oglmajor;         ///< major version of OpenGL renderer
  int oglminor;         ///< minor version of OpenGL renderer
  int oglrelease;       ///< release of OpenGL renderer
  rendenum oglrenderer; ///< OpenGL renderer ID tag for important boards

public:
  OpenGLExtensions(void);
  virtual ~OpenGLExtensions(void);
  void find_renderer(void);         ///< identify OpenGL accelerator/vendor 
  void find_extensions(void);       ///< initialize OpenGL extension state
  int vmdQueryExtension(const char *extname);  ///< query OpenGL extension
  void vmdQueryGLVersion(int *major, int *minor, int *release); ///< query OpenGL version
  VMDGLXextFuncPtr vmdGetProcAddress(const char *); ///< get extension proc addr
  void PrintExtensions(void);       ///< print out OpenGL extensions
};

#endif

