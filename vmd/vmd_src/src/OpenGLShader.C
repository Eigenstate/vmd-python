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
 *	$RCSfile: OpenGLShader.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.29 $	$Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Class to store and handle enumeration and initialization of OpenGL 
 *   shaders.
 ***************************************************************************/

#include <string.h>

#include "OpenGLExtensions.h"

#if defined(VMDUSEOPENGLSHADER)   // only compile this file if oglsl is enabled
#include "OpenGLShader.h"
#include "Inform.h"
#include "utilities.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/////////////////////////  constructor and destructor  
// constructor ... initialize some variables
OpenGLShader::OpenGLShader(OpenGLExtensions * glextptr) {
  ext = glextptr; 
  isvalid = 0;
  ProgramObject = 0;
  VertexShaderObject = 0;
  FragmentShaderObject = 0;
  lastshader = 0;
}

// destructor
OpenGLShader::~OpenGLShader(void) {
}

int OpenGLShader::LoadShader(const char * shaderpath) {
  int rc;
  GLubyte *vertexShader = NULL;
  GLubyte *fragmentShader = NULL;

  reset();
  rc = ReadShaderSource(shaderpath, &vertexShader, &fragmentShader);
  if (rc) {
    rc = CompileShaders(vertexShader, fragmentShader);
  }

  return rc;
}

void OpenGLShader::UseShader(int on) {
  if (lastshader != on) {
    if (on && isvalid) {
      GLUSEPROGRAMOBJECTARB(ProgramObject);
    } else if (!on && isvalid) {
      GLUSEPROGRAMOBJECTARB(0);
    }
  }
  lastshader = on;
}

int OpenGLShader::reset(void) {
  // delete any previous objects
  if (ProgramObject != 0) {
    if (VertexShaderObject != 0) {
      GLDETACHOBJECTARB(ProgramObject, VertexShaderObject);
    }
    if (FragmentShaderObject != 0) {
      GLDETACHOBJECTARB(ProgramObject, FragmentShaderObject);
    }

    ProgramObject = 0;
    VertexShaderObject = 0;
    FragmentShaderObject = 0;

    isvalid = 0;
    lastshader = 0;
  }

  // Create shader and program objects.
  ProgramObject        = GLCREATEPROGRAMOBJECTARB();
  VertexShaderObject   = GLCREATESHADEROBJECTARB(GL_VERTEX_SHADER_ARB);
  FragmentShaderObject = GLCREATESHADEROBJECTARB(GL_FRAGMENT_SHADER_ARB);

  return 1; // success
}

void OpenGLShader::PrintInfoLog(GLhandleARB obj, const char *msg) {
  GLint blen = 0;   /* length of buffer to allocate */
  GLint slen = 0;   /* strlen actually written to buffer */
  GLcharARB *infoLog;

  GLGETOBJECTPARAMETERIVARB(obj, GL_OBJECT_INFO_LOG_LENGTH_ARB , &blen);
  if (blen > 1) {
    if ((infoLog = (GLcharARB *) calloc(1, blen)) == NULL) {
      msgErr << "OpenGLShader could not allocate InfoLog buffer" << sendmsg;
      return;
    }

    GLGETINFOLOGARB(obj, blen, &slen, infoLog);
    msgInfo << "  " << msg << sendmsg;
    msgInfo << "    " << ((char *) infoLog) << sendmsg;
    free(infoLog);
  }
}


int OpenGLShader::CompileShaders(GLubyte *vertexShader, GLubyte *fragmentShader) {
  GLint vert_compiled = 0;
  GLint frag_compiled = 0;
  GLint linked = 0;
  GLint     length;
 
  int verbose = (getenv("VMDGLSLVERBOSE") != NULL);

  if (verbose)
    msgInfo << "Verbose GLSL shader compilation enabled..." << sendmsg;

  // Bail out if we don't have valid pointers for shader source code
  if (vertexShader == NULL || fragmentShader == NULL) {
    ProgramObject = 0;
    if (verbose)
      msgErr << "GLSL shader source incomplete during compilation" << sendmsg;
    return 0;
  }

  // Hand the source code strings to OpenGL.
  length = strlen((const char *) vertexShader);
  GLSHADERSOURCEARB(VertexShaderObject, 1, (const char **) &vertexShader, &length);
  free(vertexShader);   // OpenGL copies the shaders, we can free our copy

  length = strlen((const char *) fragmentShader);
  GLSHADERSOURCEARB(FragmentShaderObject, 1, (const char **) &fragmentShader, &length);
  free(fragmentShader); // OpenGL copies the shaders, we can free our copy

  // Compile the vertex and fragment shader, and print out
  // the compiler log file if one is available.
  GLCOMPILESHADERARB(VertexShaderObject);
  GLGETOBJECTPARAMETERIVARB(VertexShaderObject,
                  GL_OBJECT_COMPILE_STATUS_ARB, &vert_compiled);

  if (verbose)
    PrintInfoLog(VertexShaderObject, "OpenGL vertex shader compilation log: ");

  GLCOMPILESHADERARB(FragmentShaderObject);
  GLGETOBJECTPARAMETERIVARB(FragmentShaderObject,
                  GL_OBJECT_COMPILE_STATUS_ARB, &frag_compiled);

  if (verbose)
    PrintInfoLog(FragmentShaderObject, "OpenGL fragment shader compilation log: ");

  if (!vert_compiled || !frag_compiled) {
    if (verbose) {
      if (!vert_compiled)
        msgErr << "GLSL vertex shader failed to compile" << sendmsg;
      if (!frag_compiled)
        msgErr << "GLSL fragment shader failed to compile" << sendmsg;
    }
    ProgramObject = 0;
    return 0;
  }

  // Populate the program object with the two compiled shaders
  GLATTACHOBJECTARB(ProgramObject, VertexShaderObject);
  GLATTACHOBJECTARB(ProgramObject, FragmentShaderObject);

  // We want the shaders to go away as soon as they are detached from
  // the program object (or program objects) they are attached to. We
  // can simply call delete now to achieve that. Note that calling
  // delete on a program object will result in all shaders attached to
  // that program object to be detached. If delete has been called for
  // these shaders, calling delete on the program object will result in
  // the shaders being deleted as well.
  GLDELETEOBJECTARB(VertexShaderObject);
  GLDELETEOBJECTARB(FragmentShaderObject);

  // Link the whole thing together and print out the linker log file
  GLLINKPROGRAMARB(ProgramObject);
  GLGETOBJECTPARAMETERIVARB(ProgramObject, GL_OBJECT_LINK_STATUS_ARB, &linked);

  if (verbose)
    PrintInfoLog(ProgramObject, "OpenGL shader linkage log: " );

  if (vert_compiled && frag_compiled && linked) {
    isvalid = 1;
    return 1;
  } else {
    ProgramObject = 0;
    return 0;
  }
}

int OpenGLShader::ReadShaderSource(const char *filename, GLubyte **vertexShader, GLubyte **fragmentShader) {
  char *vsfilename, *fsfilename;
  FILE *vsfp, *fsfp;
  long vsize, fsize;

  vsfilename = (char *) calloc(1, strlen(filename) + strlen(".vert") + 1);
  strcpy(vsfilename, filename); 
  strcat(vsfilename, ".vert"); 
  vsfp = fopen(vsfilename, "r");

  fsfilename = (char *) calloc(1, strlen(filename) + strlen(".frag") + 1);
  strcpy(fsfilename, filename); 
  strcat(fsfilename, ".frag"); 
  fsfp = fopen(fsfilename, "r");

  if ((vsfp == NULL) || (fsfp == NULL)) {
    msgErr << "Failed to open OpenGL shader source files: " 
           << vsfilename << ", " << fsfilename << sendmsg;
    free(vsfilename);
    free(fsfilename);
    return 0;
  }

  // find size and load vertex shader
  fseek(vsfp, 0, SEEK_END);
  vsize = ftell(vsfp); 
  fseek(vsfp, 0, SEEK_SET);
  *vertexShader = (GLubyte *) calloc(1, vsize + 1);
  memset(*vertexShader, 0, vsize + 1);
  if (fread(*vertexShader, vsize, 1, vsfp) != 1) {
    msgErr << "Failed to read OpenGL vertex shader source file: " 
           << vsfilename << sendmsg;
    free(vsfilename);
    free(fsfilename);
    return 0;
  }

  // find size and load fragment shader
  fseek(fsfp, 0, SEEK_END);
  fsize = ftell(fsfp); 
  fseek(fsfp, 0, SEEK_SET);
  *fragmentShader = (GLubyte *) calloc(1, fsize + 1);
  memset(*fragmentShader, 0, fsize + 1);
  if (fread(*fragmentShader, fsize, 1, fsfp) != 1) {
    msgErr << "Failed to read OpenGL fragment shader source file: " 
           << fsfilename << sendmsg;
    free(vsfilename);
    free(fsfilename);
    return 0;
  }

  free(vsfilename);
  free(fsfilename);

  return 1;
}

#endif
