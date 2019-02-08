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
 *	$RCSfile: FileRenderList.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.96 $	$Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * The FileRenderList class maintains a list of available FileRenderer
 * objects
 *
 ***************************************************************************/

#include "config.h"  // create dependency so new compile options cause rebuild
#include "FileRenderList.h"
#include "VMDApp.h"
#include "CmdRender.h"
#include "CommandQueue.h"
#include "Scene.h"
#include "DisplayDevice.h"
#include "TextEvent.h"
#include "Inform.h"
#include <stdlib.h>  // for system()

//
// Supported external rendering programs
//
#include "ArtDisplayDevice.h"         // Art ray tracer
#include "GelatoDisplayDevice.h"      // nVidia Gelato
#if defined(VMDLIBOPTIX)
// We include the OptiX related headers first to prevent collisions between
// enums in OptiXRenderer, and preprocessor macros in Tachyon...
#include "OptiXDisplayDevice.h"       // Compiled-in OptiX ray tracer
#include "OptiXRenderer.h"            // Compiled-in OptiX ray tracer
#endif
#if defined(VMDLIBOSPRAY)
#include "OSPRayDisplayDevice.h"      // Compiled-in OSPRay ray tracer
#endif
#if defined(VMDLIBTACHYON) 
#include "LibTachyonDisplayDevice.h"  // Compiled-in Tachyon ray tracer
#endif
#if defined(VMDLIBGELATO)
#include "LibGelatoDisplayDevice.h"   // Compiled-in Gelato renderer
#endif
#include "MayaDisplayDevice.h"        // Autodesk Maya
#include "POV3DisplayDevice.h"        // POVRay 3.x 
#include "PSDisplayDevice.h"          // Postscript
#include "R3dDisplayDevice.h"         // Raster3D
#include "RadianceDisplayDevice.h"    // Radiance, unknown version 
#include "RayShadeDisplayDevice.h"    // Rayshade 4.0 
#include "RenderManDisplayDevice.h"   // RenderMan interface
#include "SnapshotDisplayDevice.h"    // Built-in snapshot capability
#include "STLDisplayDevice.h"         // Stereolithography files
#include "TachyonDisplayDevice.h"     // Tachyon ray tracer
#include "VrmlDisplayDevice.h"        // VRML 1.0 
#include "Vrml2DisplayDevice.h"       // VRML 2.0 / VRML97
#include "WavefrontDisplayDevice.h"   // Wavefront "OBJ" files
#include "X3DDisplayDevice.h"         // X3D (XML encoding)


// constructor, start off with the default means of rendering
FileRenderList::FileRenderList(VMDApp *vmdapp) : app(vmdapp) {
  add(new ArtDisplayDevice());
  add(new GelatoDisplayDevice());
#if defined(VMDLIBGELATO)
  // Only add the internally linked gelato display device to the
  // menu if the user has correctly set the GELATOHOME environment
  // variable.  If we allow them to use it otherwise, it may lead
  // to crashing, or failed renders.  This way they won't even see it
  // as an option unless they've got Gelato installed and the environment
  // at least minimally configured.
  if (getenv("GELATOHOME") != NULL) {
    add(new LibGelatoDisplayDevice());
  }
#endif
  // XXX until the native Maya ASCII export code is finished,
  // only show it when a magic environment variable is set.
  if (getenv("VMDENABLEMAYA") != NULL) {
    add(new MayaDisplayDevice());
  }
  add(new PSDisplayDevice());
  add(new R3dDisplayDevice());
  add(new RadianceDisplayDevice());
  add(new RayShadeDisplayDevice());
  add(new RenderManDisplayDevice());
  add(new SnapshotDisplayDevice(app->display));
  add(new STLDisplayDevice());
  add(new TachyonDisplayDevice());
#if defined(VMDLIBTACHYON)
  add(new LibTachyonDisplayDevice(vmdapp));
#endif

#if defined(VMDLIBOPTIX)
  // ensure special member variables are cleared
  cluster_dev = NULL;
  if (!getenv("VMDNOOPTIX")) {
    int optixdevcount=OptiXRenderer::device_count();

    // Only emit detailed OptiX GPU data if we're running on a single node.
    // Put the console informational text immediately prior to the creation of
    // FileRenderList objects where the renderers are actually instantiated, 
    // so that all of the OptiX GPU and compilation status info is together
    if (vmdapp->nodecount == 1) {
      if (optixdevcount > 0) {
        msgInfo << "Detected " << optixdevcount << " available TachyonL/OptiX ray tracing "
                << ((optixdevcount > 1) ? "accelerators" : "accelerator")
                << sendmsg;
      }
    }

    // check for user-supplied remote VCA rendering cluster URL, user, passwd,
    // if all three are set, we attach to the remote cluster and if successful,
    // the remote connection is used during OptiX context initialization.

#if defined(VMDOPTIX_PROGRESSIVEAPI)
    const char *vca_url  = getenv("VMDOPTIX_CLUSTER");
    const char *vca_user = getenv("VMDOPTIX_USER");
    const char *vca_pw   = getenv("VMDOPTIX_PASSWD");

    if (vca_url && vca_user && vca_pw) {
      cluster_dev = OptiXRenderer::remote_connect(vca_url, vca_user, vca_pw);
    }
#endif

    // Perform runtime check for OptiX availability before we add it to the
    // list of available renderers.
    if (optixdevcount > 0) { 
      // Emit a console message during OptiX renderer instantiation
      // since the JIT compilation and linkage of the 256+ shaders may
      // take several seconds on machines with several GPUs...
      if (vmdapp->nodecount == 1) {
        msgInfo << "  Compiling " 
                << OptiXRenderer::material_shader_table_size() 
                << " OptiX shaders on " << optixdevcount << " target GPU" 
                << ((optixdevcount > 1) ? "s" : "") << "..." << sendmsg;
      }

      add(new OptiXDisplayDevice(vmdapp, 0, cluster_dev));

      // Even if we have no GUI, then we now add interactive renderer
      // to be able to support remote rendering via video streaming
      // from clusters, supercomputers, etc.  The renderer launch code
      // will determine whether or not to launch a GUI-based interactive
      // renderer or a renderer intended solely for video streaming.
      add(new OptiXDisplayDevice(vmdapp, 1, cluster_dev));
    }
  }
#endif

#if defined(VMDLIBOSPRAY)
  if (!getenv("VMDNOOSPRAY")) {
    OSPRayDisplayDevice::OSPRay_Global_Init(); // call only ONCE
    add(new OSPRayDisplayDevice(vmdapp, 0));
#if defined(VMDOPENGL) && defined(VMDOSPRAY_INTERACTIVE_OPENGL)
    add(new OSPRayDisplayDevice(vmdapp, 1));
#endif
  }
#endif

  add(new POV3DisplayDevice());
  add(new VrmlDisplayDevice());
  add(new Vrml2DisplayDevice());
  add(new WavefrontDisplayDevice());
  add(new X3DDisplayDevice());
  add(new X3DOMDisplayDevice());
}


// destructor, deallocate all the info
FileRenderList::~FileRenderList(void) {
  // Ensure that renderer classes that contain OptiX contexts are 
  // destroyed prior to detaching from remote VCA rendering devices.
  for (int i=0;i<renderList.num();i++)
    delete renderList.data(i);

#if defined(VMDOPTIX_PROGRESSIVEAPI)
  // detach from remote VCA OptiX rendering cluster at shutdown if attached
  if (cluster_dev)
    OptiXRenderer::remote_detach(cluster_dev);
#endif
}


// add a new render class with its corresponding name
void FileRenderList::add(FileRenderer *newRenderer) {
  if (newRenderer)
    renderList.add_name(newRenderer->name, newRenderer);
}

// figure out how many render classes are installed
int FileRenderList::num(void) {
  return renderList.num();
}

// return the name for the ith class, returns NULL if out of range
const char *FileRenderList::name(int i) {
  if (i>=0 && i < renderList.num()) {
    return renderList.name(i);
  }
  return NULL;
}

// return the "pretty" name (used in GUIs) for the ith class.
// returns NULL if out of range
const char *FileRenderList::pretty_name(int i) {
  if (i>=0 && i < renderList.num()) {
    const FileRenderer * fr = renderList.data(i);
    return fr->pretty_name();
  }
  return NULL;
}

// find class (case-insensitive) for a renderer name, else return NULL  
FileRenderer *FileRenderList::find(const char *rname) {
  int indx = renderList.typecode(rname);
  
  if (indx >= 0)
    return renderList.data(indx);
  else
    return NULL;
}

// given a "pretty" render name, return the corresponding class
FileRenderer *FileRenderList::find_pretty_name(const char *pretty) {
  int i;
  for (i=0; i<renderList.num(); i++) {
    if (!strcmp(pretty_name(i), pretty)) {
      return renderList.data(i); 
    }
  }
  return NULL;
}

// given a "pretty" renderer name, return the short name
const char *FileRenderList::find_short_name_from_pretty_name(const char *pretty) {
  const FileRenderer *fr = find_pretty_name(pretty);
  if (fr)
    return fr->visible_name();
  return NULL;
}

int FileRenderList::render(const char *filename, const char *method,
                           const char *extcmd) {
  msgInfo << "Rendering current scene to '" << filename << "' ..." << sendmsg;

  FileRenderer *render = find(method);
  if (!render) {
    msgErr << "Invalid render method '" << method << sendmsg;
    return FALSE;
  }

  // XXX Snapshot grabs the wrong buffer, so if we're doing snapshot, swap
  // the buffers, render, then swap back.
  if (!strcmp(method, "snapshot")) app->display->update(TRUE);
  int retval = app->scene->filedraw(render, filename, app->display);
  if (!strcmp(method, "snapshot")) app->display->update(TRUE);

  // if successful, execute external command
  if (retval && extcmd && *extcmd != '\0') {
    JString strbuf(extcmd);
    strbuf.gsub("%s", filename);
    // substitute display %w and %h for display width and height
    int w=100, h=100;
    char buf[32];
    app->display_get_size(&w, &h);
    sprintf(buf, "%d", w);
    strbuf.gsub("%w", buf);
    sprintf(buf, "%d", h);
    strbuf.gsub("%h", buf);
    msgInfo << "Executing post-render cmd '" << (const char *)strbuf << "' ..." << sendmsg;
    vmd_system(strbuf);
  }

  // return result
  msgInfo << "Rendering complete." << sendmsg;
  return retval;
}

int FileRenderList::set_render_option(const char *method, const char *option) {
  FileRenderer *ren;
  ren = find(method);
  if (!ren) {
    msgErr << "No rendering method '" << method << "' available." << sendmsg;
    return FALSE;
  }
  ren->set_exec_string(option);
  return TRUE;
} 

int FileRenderList::has_antialiasing(const char *method) {
  FileRenderer *ren = find(method);
  if (ren) return ren->has_antialiasing();
  return 0;
}

int FileRenderList::aasamples(const char *method, int aasamples) {
  FileRenderer *ren = find(method);
  if (ren) return ren->set_aasamples(aasamples);
  return -1;
}

int FileRenderList::aosamples(const char *method, int aosamples) {
  FileRenderer *ren = find(method);
  if (ren) return ren->set_aosamples(aosamples);
  return -1;
}

int FileRenderList::imagesize(const char *method, int *w, int *h) {
  FileRenderer *ren = find(method);
  if (!ren) return FALSE;
  return ren->set_imagesize(w, h);
}

int FileRenderList::has_imagesize(const char *method) {
  FileRenderer *ren = find(method);
  if (!ren) return FALSE;
  return ren->has_imagesize();
}

int FileRenderList::aspectratio(const char *method, float *aspect) {
  FileRenderer *ren = find(method);
  if (!ren) return FALSE;
  *aspect = ren->set_aspectratio(*aspect);
  return TRUE;
}

int FileRenderList::numformats(const char *method) {
  FileRenderer *ren = find(method);
  if (!ren) return 0;
  return ren->numformats();
}

const char *FileRenderList::format(const char *method, int i) {
  FileRenderer *ren = find(method);
  if (!ren) return NULL;
  if (i < 0) return ren->format();
  return ren->format(i);
}

int FileRenderList::set_format(const char *method, const char *format) {
  FileRenderer *ren = find(method);
  if (!ren) return FALSE;
  return ren->set_format(format);
}

