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
 *	$RCSfile: OpenGLExtensions.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.81 $	$Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Class to store and handle enumeration and initialization of OpenGL 
 *   extensions and features.
 ***************************************************************************/

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "OpenGLExtensions.h"
#include "Inform.h"
#include "utilities.h"
#include "vmddlopen.h"

#if !defined(_MSC_VER) && !(defined(__APPLE__) && !defined (VMDMESA)) && !defined(VMDEGLPBUFFER)
#include <GL/glx.h>     // needed for glxGetProcAddress() prototype
#endif

#if defined(__APPLE__)
#import <mach-o/dyld.h> // needed by the getProcAddress code
#import <string.h>
#include <AvailabilityMacros.h>
#endif

/////////////////////////  constructor and destructor  
// constructor ... initialize some variables
OpenGLExtensions::OpenGLExtensions(void) {
  // set handle NULL
  gllibraryhandle = NULL;

  // initialize OpenGL version info and feature detection
  multitextureunits = 0;
  hasmultisample = 0;
  hasmultidrawext = 0;
  hasstencilbuffer = 0;
  hastex2d = 0;
  hastex3d = 0;
  hasrescalenormalext = 0;
  hasglarbtexnonpoweroftwo = 0;
  hascompiledvertexarrayext = 0;
  hasglpointparametersext = 0;
  hasglpointspritearb = 0; 
  hasglshaderobjectsarb = 0;
  hasglshadinglangarb = 0;
  hasglvertexshaderarb = 0;
  hasglfragmentshaderarb = 0;
  hasglgeometryshader4arb = 0;
  hasglsampleshadingarb = 0;
}

// destructor
OpenGLExtensions::~OpenGLExtensions(void) {
  if (gllibraryhandle != NULL)
    vmddlclose(gllibraryhandle);
}

int OpenGLExtensions::vmdQueryExtension(const char *extname) {
  char *excl = getenv("VMD_EXCL_GL_EXTENSIONS");
  if (!extname)
    return 0;

  // search for extension in VMD's exclusion list  
  if (excl != NULL) {
    char *endexcl = excl + strlen(excl);
    while (excl < endexcl) {
      size_t n = strcspn(excl, " ");
      if ((strlen(extname) == n) && (strncmp(extname, excl, n) == 0)) {
        return 0; // extension is disabled and excluded
      }
      excl += (n + 1);
    }
  }

  // search for extension in list of available extensions
  char *ext = (char *) glGetString(GL_EXTENSIONS);
  if (ext != NULL) {
    char *endext = ext + strlen(ext);
    while (ext < endext) {
      size_t n = strcspn(ext, " ");
      if ((strlen(extname) == n) && (strncmp(extname, ext, n) == 0)) {
        return 1; // True, extension is available
        break;
      }
      ext += (n + 1);
    }
  }

  return 0; // False, extension is not available
}

VMDGLXextFuncPtr OpenGLExtensions::vmdGetProcAddress(const char * procname) {
  if (!procname)
    return NULL;

#if defined(_MSC_VER)
  // NOTE: wgl returns a context-dependent function pointer
  //       the function can only be called within the same wgl
  //       context in which it was generated.
  return (VMDGLXextFuncPtr) wglGetProcAddress((LPCSTR) procname);
#endif

#if defined(__APPLE__)
#if defined(ARCH_MACOSX)
  // MacOS X versions for PowerPC were unstable with the use of full 
  // OpenGL extensions.  So we turn them off at at runtime unless 
  // the user has specifically enabled them.
  if (getenv("VMDMACENABLEEEXTENSIONS") == NULL) {
    return NULL;
  }
#endif

  // According to the MacOS X documentation, they provide(d) statically 
  // linkable OpenGL extensions tied to to each specific rev of MacOS X.
#if defined(MAC_OS_X_VERSION_10_5) && (MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_X_VERSION_10_5)
  // MacOS X >= 10.4 apps are directed to use the Unix standard 
  // dlopen() family of routines  or to use explicit tests against 
  // weak-linked OpenGL API entry points instead.
  if (gllibraryhandle == NULL) {
    // Path to GL on OSX. True for at least versions 10.8 through 10.10.x.
    static const char * glLibPath = 
        "/System/Library/Frameworks/OpenGL.framework/Versions/Current/OpenGL";
    gllibraryhandle = vmddlopen(glLibPath);
  }

  if (gllibraryhandle != NULL) {
    return (VMDGLXextFuncPtr) vmddlsym(gllibraryhandle, procname);
  }
#else
  // This implementation is based off of the MacOS X developer information
  // provided by Apple, circa MacOS X 10.[23].
  //   http://developer.apple.com/technotes/tn2002/tn2080.html#TAN55
  //   http://developer.apple.com/technotes/tn2002/tn2080.html#TAN28
  NSSymbol symbol;
  char *symbolName;
  // Prepend a '_' for the Unix C symbol mangling convention
  symbolName = (char *) calloc(1, strlen(procname) + 2);
  strcpy(symbolName+1, procname);
  symbolName[0] = '_';
  symbol = NULL;
  if (NSIsSymbolNameDefined(symbolName))
    symbol = NSLookupAndBindSymbol (symbolName);
  free(symbolName);
  return (VMDGLXextFuncPtr) (symbol ? NSAddressOfSymbol(symbol) : NULL);
#endif
#endif

#if !defined(_MSC_VER) && !defined(__APPLE__)
#if !defined(__linux) && !defined(ARCH_FREEBSD) && !defined(ARCH_FREEBSDAMD64) && !defined(ARCH_SOLARISX86) && !defined(ARCH_SOLARISX86_64) && (defined(GLX_VERSION_1_4) || defined(ARCH_SOLARIS2))
  // GLX 1.4 form found on commercial Unix systems that
  // don't bother providing the ARB extension version that Linux prefers.
  //
  // From FreeBSD ports comment by stephen@math.missouri.edu:
  // Why the !defined(ARCH_FREEBSD)?:  Typically the X libraries that
  // come with FreeBSD work with glXGetProcAddress.  However, if the
  // nvidia-driver port is installed, it seems not to work.  But using
  // glXGetProcAddressARB seems to work whether or not the nvidia-driver
  // port is installed.
  //
  return glXGetProcAddress((const GLubyte *) procname);
#else

// XXX this is a workaround for a crash on early 64-bit NVidia drivers
#if defined(VMDEGLPBUFFER)
#if 0
  // When using EGL with full OpenGL we might have to use dlopen() to find our 
  // OpenGL API entry points if eglGetProcAddress doesn't work out.
  if (gllibraryhandle == NULL) {
    // Path to GL on Linux. 
    static const char * glLibPath = "/usr/lib64/libGL.so";
    gllibraryhandle = vmddlopen(glLibPath);
  }

  if (gllibraryhandle != NULL) {
    VMDGLXextFuncPtr fctn;
    fctn = (VMDGLXextFuncPtr) vmddlsym(gllibraryhandle, procname);
#if 0
    printf("proc: '%s' ptr: %p\n", procname, fctn);
#endif
    return fctn;
  }
#else
  // If we were using OpenGL ES we would use the EGL routine directly
  {
    VMDGLXextFuncPtr fctn;
    fctn = (VMDGLXextFuncPtr) eglGetProcAddress((const char *) procname);
#if 0
    printf("proc: '%s' ptr: %p\n", procname, fctn);
#endif
    return fctn;
  }
#endif
#else
#if defined(GLX_ARB_get_proc_address)
  // NOTE: GLX returns a context-independent function pointer that
  //       can be called anywhere, no special handling is required.
  //       This method is used on Linux
  return glXGetProcAddressARB((const GLubyte *) procname);
#endif
#endif

#endif
#endif

  return NULL;
}

void OpenGLExtensions::vmdQueryGLVersion(int *major, int *minor, int *release) {
  const char *p = (char *) glGetString(GL_VERSION);

  *major = 1;
  *minor = 0;
  *release = 0;

  if (p != NULL) {
    char *cp;
    cp = (char *) calloc(1, strlen(p) + 1);
    strcpy(cp, p); 
  
    char *np=cp;
    char *ep=cp;
 
    while (np < (np + strlen(p))) {
      if (*np == ' ' || *np == '\0') {
        *np = '\0';
        ep = np;
        break;
      }
      np++;
    }

    np = cp;
    char *lp=cp;
    int x=0; 
    while (np <= ep) {
      if (*np == '.' || *np == '\0') {
        *np = '\0';

        switch(x) {
          case 0:
            *major = atoi(lp);
            break;
 
          case 1:
            *minor = atoi(lp);
            break;
 
          case 2:
            *release = atoi(lp);
            break;
        }
        np++;
        lp = np;
        x++;
        continue;
      }           

      np++;
    }

    free(cp);
  }
}


void OpenGLExtensions::find_renderer(void) {
  // Identify the hardware we're rendering on
  oglrenderer = GENERIC;
  const char * rs = (const char *) glGetString(GL_RENDERER);
  const char * rv = (const char *) glGetString(GL_VENDOR);
  if ((rs != NULL) && (rv != NULL)) { 
    if (strstr(rv, "ATI") != NULL) {
      oglrenderer = ATI;
    }
    if (strstr(rv, "NVIDIA") != NULL) {
      oglrenderer = NVIDIA;
    }
    if (strstr(rs, "GDI Generic") != NULL) {
      oglrenderer = MSOFTGDI; // microsoft software renderer
    }
    if (strstr(rs, "Mesa") != NULL) {
      oglrenderer = MESAGL;
    }
    if (strstr(rs, "WireGL") != NULL) {
      oglrenderer = WIREGL;
    }
    if ((strstr(rv, "Intel") != NULL) && (strstr(rs, "SWR") != NULL)) {
      oglrenderer = INTELSWR;
    }
  }
}


void OpenGLExtensions::find_extensions(void) {
  // initialize OpenGL extension function pointers to NULL
  p_glLockArraysEXT = NULL;
  p_glUnlockArraysEXT = NULL;
  p_glMultiDrawElementsEXT = NULL;
  p_glGlobalAlphaFactorfSUN = NULL;
  p_glPointParameterfARB = NULL;
  p_glPointParameterfvARB = NULL;

#if defined(GL_ARB_shader_objects)
  // ARB OpenGL Shader functions
  p_glCreateShaderObjectARB = NULL;
  p_glCreateProgramObjectARB = NULL;
  p_glUseProgramObjectARB = NULL;
  p_glDetachObjectARB = NULL;
  p_glGetInfoLogARB = NULL;
  p_glGetObjectParameterivARB = NULL;
  p_glLinkProgramARB = NULL;
  p_glDeleteObjectARB = NULL;
  p_glAttachObjectARB = NULL;
  p_glCompileShaderARB = NULL;
  p_glShaderSourceARB = NULL;
  p_glGetUniformLocationARB = NULL;
  p_glUniform1iARB = NULL;
  p_glUniform1fvARB = NULL;
  p_glUniform2fvARB = NULL;
  p_glUniform3fvARB = NULL;
  p_glUniform4fvARB = NULL;
#endif

  vmdQueryGLVersion(&oglmajor, &oglminor, &oglrelease);

  // check for an OpenGL stencil buffer
  GLint stencilbits;
  glGetIntegerv(GL_STENCIL_BITS, &stencilbits);
  if (stencilbits > 0) {
    hasstencilbuffer = 1;
  }

  // Identify the hardware we're rendering on
  find_renderer();

#if defined(GL_ARB_multitexture)
  // perform tests for ARB multitexturing if we're on an
  // appropriate rev of OpenGL.
  if (((oglmajor >= 1) && (oglminor >= 2)) ||
      ((oglmajor >= 2) && (oglminor >= 0))) {
    // query to see if this machine supports the multitexture extension
    if (vmdQueryExtension("GL_ARB_multitexture")) {
      glGetIntegerv(GL_MAX_TEXTURE_UNITS_ARB, &multitextureunits);
    }
  }
#endif

#if defined(GL_VERSION_1_1)
  // Our implementation of 3-D texturing is only available on
  // OpenGL 1.1 or higher, so only enable/test if that is the case.
  if (((oglmajor >= 1) && (oglminor >= 1)) ||
      ((oglmajor >= 2) && (oglminor >= 0))) {
    hastex2d = 1;
  }
#endif

#if defined(GL_VERSION_1_2)
  // Our implementation of 3-D texturing is only available on
  // OpenGL 1.2 or higher, so only enable/test if that is the case.
  if (((oglmajor >= 1) && (oglminor >= 2)) ||
      ((oglmajor >= 2) && (oglminor >= 0))) {
#if defined(VMDUSEGETPROCADDRESS) && !defined(__linux) && !defined(__APPLE__)
    p_glTexImage3D = (void (APIENTRY *)(GLenum, GLint, GLint,
                            GLsizei, GLsizei, GLsizei, GLint,
                            GLenum, GLenum, const GLvoid *)) vmdGetProcAddress("glTexImage3D"); 
    if (p_glTexImage3D != NULL) {
      hastex3d = 1;
    }
#else
    hastex3d = 1;
#endif
  }
#endif

#if defined(GL_ARB_texture_non_power_of_two)
  // check for ARB non-power-of-two texture size extension
  if (vmdQueryExtension("GL_ARB_texture_non_power_of_two")) {
    hasglarbtexnonpoweroftwo = 1;
  }
#endif

#if defined(GL_EXT_multi_draw_arrays)
  // check for the Sun/ARB glMultiDraw...() extensions
  if (vmdQueryExtension("GL_EXT_multi_draw_arrays")) {
#if defined(VMDUSEGETPROCADDRESS)
    p_glMultiDrawElementsEXT = (void (APIENTRY *)(GLenum, const GLsizei *, GLenum, const GLvoid**, GLsizei)) vmdGetProcAddress("glMultiDrawElementsEXT");
    if (p_glMultiDrawElementsEXT != NULL) {
      hasmultidrawext = 1;
    } 
#else 
    hasmultidrawext = 1;
#endif    
  }
#endif

#if defined(GL_ARB_shading_language_100)
  // check for the OpenGL Shading Language extension
  if (vmdQueryExtension("GL_ARB_shading_language_100")) {
    hasglshadinglangarb = 1;
  }
#endif

#if defined(GL_ARB_shader_objects)
  if (vmdQueryExtension("GL_ARB_shader_objects")) {
#if defined(VMDUSEGETPROCADDRESS)
    p_glCreateShaderObjectARB = (GLhandleARB (APIENTRY *)(GLenum)) vmdGetProcAddress("glCreateShaderObjectARB");
    p_glCreateProgramObjectARB = (GLhandleARB (APIENTRY *)(void)) vmdGetProcAddress("glCreateProgramObjectARB");
    p_glUseProgramObjectARB = (void (APIENTRY *)(GLhandleARB)) vmdGetProcAddress("glUseProgramObjectARB");
    p_glDetachObjectARB = (void (APIENTRY *)(GLhandleARB, GLhandleARB)) vmdGetProcAddress("glDetachObjectARB");
    p_glGetInfoLogARB = (void (APIENTRY *)(GLhandleARB, GLsizei, GLsizei *, GLcharARB *)) vmdGetProcAddress("glGetInfoLogARB");
    p_glGetObjectParameterivARB = (void (APIENTRY *)(GLhandleARB, GLenum, GLint *)) vmdGetProcAddress("glGetObjectParameterivARB");
    p_glLinkProgramARB = (void (APIENTRY *)(GLhandleARB)) vmdGetProcAddress("glLinkProgramARB");
    p_glDeleteObjectARB = (void (APIENTRY *)(GLhandleARB)) vmdGetProcAddress("glDeleteObjectARB");
    p_glAttachObjectARB = (void (APIENTRY *)(GLhandleARB, GLhandleARB)) vmdGetProcAddress("glAttachObjectARB");
    p_glCompileShaderARB = (void (APIENTRY *)(GLhandleARB)) vmdGetProcAddress("glCompileShaderARB");
    p_glShaderSourceARB = (void (APIENTRY *)(GLhandleARB, GLsizei, const GLcharARB **, const GLint *)) vmdGetProcAddress("glShaderSourceARB");
    p_glGetUniformLocationARB = (GLint (APIENTRY *)(GLhandleARB programObject, const GLcharARB *name)) vmdGetProcAddress("glGetUniformLocationARB");
    p_glUniform1iARB = (void (APIENTRY *)(GLint location, GLint v0)) vmdGetProcAddress("glUniform1iARB");
    p_glUniform1fvARB = (void (APIENTRY *)(GLint location, GLsizei count, GLfloat *value)) vmdGetProcAddress("glUniform1fvARB");
    p_glUniform2fvARB = (void (APIENTRY *)(GLint location, GLsizei count, GLfloat *value)) vmdGetProcAddress("glUniform2fvARB");
    p_glUniform3fvARB = (void (APIENTRY *)(GLint location, GLsizei count, GLfloat *value)) vmdGetProcAddress("glUniform3fvARB");
    p_glUniform4fvARB = (void (APIENTRY *)(GLint location, GLsizei count, GLfloat *value)) vmdGetProcAddress("glUniform4fvARB");

    if (p_glCreateShaderObjectARB != NULL && p_glCreateProgramObjectARB != NULL &&
        p_glUseProgramObjectARB != NULL && p_glDetachObjectARB != NULL &&
        p_glGetInfoLogARB != NULL && p_glGetObjectParameterivARB != NULL &&
        p_glLinkProgramARB != NULL && p_glDeleteObjectARB != NULL &&
        p_glAttachObjectARB != NULL && p_glCompileShaderARB != NULL &&
        p_glShaderSourceARB != NULL && p_glGetUniformLocationARB != NULL &&
        p_glUniform1iARB != NULL && p_glUniform1fvARB != NULL &&
        p_glUniform2fvARB != NULL && p_glUniform3fvARB != NULL && 
        p_glUniform4fvARB  != NULL) {
      hasglshaderobjectsarb = 1;
    } else {
      hasglshaderobjectsarb = 0;
    }  
#else
    hasglshaderobjectsarb = 1;
#endif
  }
#endif

#if defined(GL_ARB_vertex_shader)
  if (vmdQueryExtension("GL_ARB_vertex_shader")) {
    hasglvertexshaderarb = 1;
  }
#endif

#if defined(GL_ARB_fragment_shader)
  if (vmdQueryExtension("GL_ARB_fragment_shader")) {
    hasglfragmentshaderarb = 1;
  }
#endif

#if defined(GL_ARB_geometry_shader4)
  if (vmdQueryExtension("GL_ARB_geometry_shader4")) {
    hasglgeometryshader4arb = 1;
  }
#endif

  if (vmdQueryExtension("GL_ARB_sample_shading")) {
    hasglsampleshadingarb = 1;
  }


#if defined(GL_EXT_compiled_vertex_array)
  // check for the compiled vertex array extension
  if (vmdQueryExtension("GL_EXT_compiled_vertex_array")) {
#if defined(VMDUSEGETPROCADDRESS)
    p_glLockArraysEXT   = (void (APIENTRY *)(GLint, GLsizei)) vmdGetProcAddress("glLockArraysEXT");
    p_glUnlockArraysEXT = (void (APIENTRY *)(void))           vmdGetProcAddress("glUnlockArraysEXT");
    if ((p_glLockArraysEXT != NULL) && (p_glUnlockArraysEXT != NULL)) { 
      hascompiledvertexarrayext = 1;
    }
#else
    hascompiledvertexarrayext = 1;
#endif
  }
#endif

#if defined(GL_ARB_point_parameters) 
  // check for glPointParameterfARB extension functions
  if (vmdQueryExtension("GL_ARB_point_parameters")) {
#if defined(VMDUSEGETPROCADDRESS)
    p_glPointParameterfARB = (void (APIENTRY *)(GLenum, GLfloat)) vmdGetProcAddress("glPointParameterfARB");
    p_glPointParameterfvARB = (void (APIENTRY *)(GLenum, const GLfloat *)) vmdGetProcAddress("glPointParameterfvARB");
    if (p_glPointParameterfARB != NULL && p_glPointParameterfvARB != NULL) {
      hasglpointparametersext = 1;
    }
#else
    hasglpointparametersext = 1;
#endif
  } 
#endif

#if defined(GL_ARB_point_sprite)
  if (vmdQueryExtension("GL_ARB_point_sprite")) {
    hasglpointspritearb = 1;
  }
#endif

}


void OpenGLExtensions::PrintExtensions(void) {
  const char * rs = (const char *) glGetString(GL_RENDERER);
  if (rs == NULL)
    rs = "ErrorUnknown";

  // Print renderer string for informational purposes
  msgInfo << "OpenGL renderer: " << rs << sendmsg;

  // print information on any OpenGL features found and used
  msgInfo << "  Features: ";
  if (hasstencilbuffer)
    msgInfo << "STENCIL ";

  if (hasstereo)
    msgInfo << "STEREO ";

  if (hasmultisample)
    msgInfo << "MSAA(" << nummultisamples << ") ";

  if (hasrescalenormalext)
    msgInfo << "RN ";

  if (hasmultidrawext)
    msgInfo << "MDE ";

  if (hascompiledvertexarrayext)
    msgInfo << "CVA ";

  if (multitextureunits > 0)
    msgInfo << "MTX ";

  if (hasglarbtexnonpoweroftwo)
    msgInfo << "NPOT ";

  if (hasglpointparametersext)
    msgInfo << "PP ";

  if (hasglpointspritearb)
    msgInfo << "PS "; 

  //
  // OpenGL Shading language extensions
  //
  if (hasglshadinglangarb) {
    msgInfo << "GLSL(";

    if (hasglshaderobjectsarb)
      msgInfo << "O"; 

    if (hasglvertexshaderarb)
      msgInfo << "V"; 

    if (hasglfragmentshaderarb)
      msgInfo << "F"; 

    if (hasglgeometryshader4arb)
      msgInfo << "G"; 

    if (hasglsampleshadingarb)
      msgInfo << "S"; 

    msgInfo << ") ";
  }
  msgInfo << sendmsg;
}

