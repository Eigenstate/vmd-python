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
 *	$RCSfile: OpenGLShader.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.14 $	$Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Class to store and handle enumeration and initialization of OpenGL 
 *   shaders.
 ***************************************************************************/
#ifndef OPENGLSHADER_H
#define OPENGLSHADER_H

#if defined(_MSC_VER)
#include <windows.h>
#endif

/// manages enumeration and initialization of OpenGL programmable shaders
class OpenGLShader {
public: 
  const OpenGLExtensions *ext;      ///< cached OpenGL extensions handle
  int isvalid;                      ///< succesfully compiled shader flag
  GLhandleARB ProgramObject;        ///< ARB program object handle
  GLhandleARB VertexShaderObject;   ///< ARB vertex shader object handle
  GLhandleARB FragmentShaderObject; ///< ARB fragment shader object handle
  int lastshader;                   ///< last shader index/state used

public:
  OpenGLShader(OpenGLExtensions *ext);
  virtual ~OpenGLShader(void);

  /// Load a named set of shader source files (no filename extensions)
  int LoadShader(const char * shaderpath);
  int reset(void); 

  /// Compile and link loaded shader source files
  int CompileShaders(GLubyte *vertexShader, GLubyte *fragmentshader);

  /// Print compilation log
  void PrintInfoLog(GLhandleARB ProgramObject, const char *msg);

  /// Read shader source code from  files into a byte array
  int ReadShaderSource(const char * filename, GLubyte **vs, GLubyte **fs);

  /// Make a valid compiled shader the active shader
  /// This replaces the fixed-function OpenGL pipeline and any other 
  /// active shader.
  void UseShader(int onoff);
};

#endif

