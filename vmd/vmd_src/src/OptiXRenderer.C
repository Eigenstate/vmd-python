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
*      $RCSfile: OptiXRenderer.C
*      $Author: johns $      $Locker:  $               $State: Exp $
*      $Revision: 1.346 $         $Date: 2019/01/17 21:38:55 $
*
***************************************************************************
* DESCRIPTION:
*   VMD built-in Tachyon/OptiX renderer implementation.
*
* This work is described in:
*  "GPU-Accelerated Molecular Visualization on
*   Petascale Supercomputing Platforms"
*   John E. Stone, Kirby L. Vandivort, and Klaus Schulten.
*   UltraVis'13: Proceedings of the 8th International Workshop on
*   Ultrascale Visualization, pp. 6:1-6:8, 2013.
*   http://dx.doi.org/10.1145/2535571.2535595
*
*  "Atomic Detail Visualization of Photosynthetic Membranes with
*   GPU-Accelerated Ray Tracing"
*   John E. Stone, Melih Sener, Kirby L. Vandivort, Angela Barragan,
*   Abhishek Singharoy, Ivan Teo, João V. Ribeiro, Barry Isralewitz,
*   Bo Liu, Boon Chong Goh, James C. Phillips, Craig MacGregor-Chatwin,
*   Matthew P. Johnson, Lena F. Kourkoutis, C. Neil Hunter, and Klaus Schulten
*   J. Parallel Computing, 55:17-27, 2016.
*   http://dx.doi.org/10.1016/j.parco.2015.10.015
*
*  "Immersive Molecular Visualization with Omnidirectional
*   Stereoscopic Ray Tracing and Remote Rendering"
*   John E. Stone, William R. Sherman, and Klaus Schulten.
*   High Performance Data Analysis and Visualization Workshop,
*   2016 IEEE International Parallel and Distributed Processing
*   Symposium Workshops (IPDPSW), pp. 1048-1057, 2016.
*   http://dx.doi.org/10.1109/IPDPSW.2016.121
*
*  "Omnidirectional Stereoscopic Projections for VR"
*   John E. Stone.
*   In, William R. Sherman, editor,
*   VR Developer Gems, Taylor and Francis / CRC Press, Chapter 24, 2019.
*
*  "Interactive Ray Tracing Techniques for
*   High-Fidelity Scientific Visualization"
*   John E. Stone.
*   In, Eric Haines and Tomas Akenine-Möller, editors,
*   Ray Tracing Gems, Apress, 2019.
*
*  "A Planetarium Dome Master Camera"
*   John E. Stone.
*   In, Eric Haines and Tomas Akenine-Möller, editors,
*   Ray Tracing Gems, Apress, 2019.
*
* Portions of this code are derived from Tachyon:
*   "An Efficient Library for Parallel Ray Tracing and Animation"
*   John E. Stone.  Master's Thesis, University of Missouri-Rolla,
*   Department of Computer Science, April 1998
*
*   "Rendering of Numerical Flow Simulations Using MPI"
*   John Stone and Mark Underwood.
*   Second MPI Developers Conference, pages 138-141, 1996.
*   http://dx.doi.org/10.1109/MPIDC.1996.534105
*
***************************************************************************/

#include <optix.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#if defined(__linux)
#include <unistd.h>         // needed for symlink() in movie recorder
#endif

#include "VMDApp.h"         // needed for video streaming
#include "VideoStream.h"    // needed for video streaming
#include "DisplayDevice.h"  // needed for video streaming

#include "Inform.h"
#include "ImageIO.h"
#include "OptiXRenderer.h"
#include "OptiXShaders.h"
#include "Matrix4.h"
#include "utilities.h"
#include "WKFUtils.h"
#include "ProfileHooks.h"

// Enable HMD if VMD compiled with Oculus VR SDK or OpenHMD
#if defined(VMDUSEOPENHMD) 
#define VMDOPTIX_USE_HMD 1
#endif

#if defined(VMDOPTIX_USE_HMD)
#include "HMDMgr.h"
#endif

// support Linux event I/O based joystick/spaceball input
#if defined(VMDUSEEVENTIO)
#include "eventio.h"
#endif

// enable the interactive ray tracing capability
#if defined(VMDOPTIX_INTERACTIVE_OPENGL)
#include <GL/gl.h>
#endif

// the ORT_USE_TEMPLATE_SHADERS macro enables or disables the use of 
// an array of template-specialized shaders for every combination of
// scene-wide and material-specific shader features.
#if defined(ORT_USE_TEMPLATE_SHADERS)
static const char *onoffstr(int onoff) {
  return (onoff) ? "on" : "off";
}
#endif

// OptiX 5.2 HW triangle APIs
#define ORT_USE_HW_TRIANGLES 1

// check environment for verbose timing/debugging output flags
static OptiXRenderer::Verbosity get_verbose_flag(int inform=0) {
  OptiXRenderer::Verbosity verbose = OptiXRenderer::RT_VERB_MIN;
  char *verbstr = getenv("VMDOPTIXVERBOSE");
  if (verbstr != NULL) {
//    printf("OptiXRenderer) verbosity config request: '%s'\n", verbstr);
    if (!strupcmp(verbstr, "MIN")) {
      verbose = OptiXRenderer::RT_VERB_MIN;
      if (inform)
        printf("OptiXRenderer) verbose setting: minimum\n");
    } else if (!strupcmp(verbstr, "TIMING")) {
      verbose = OptiXRenderer::RT_VERB_TIMING;
      if (inform)
        printf("OptiXRenderer) verbose setting: timing data\n");
    } else if (!strupcmp(verbstr, "DEBUG")) {
      verbose = OptiXRenderer::RT_VERB_DEBUG;
      if (inform)
        printf("OptiXRenderer) verbose setting: full debugging data\n");
    }
  }
  return verbose;
}


#if 0
// Enable the use of OptiX timeout callbacks to help reduce the likelihood
// of kernel timeouts when rendering on GPUs that are also used for display
#define VMD_ENABLE_OPTIX_TIMEOUTS 1

static int vmd_timeout_init = 0;
static wkf_timerhandle cbtimer;
static float vmd_timeout_lastcallback = 0.0f;

static void vmd_timeout_reset(void) {
  if (vmd_timeout_init == 0) {
    vmd_timeout_init = 1;
    cbtimer = wkf_timer_create();
  }
  wkf_timer_start(cbtimer);
  vmd_timeout_lastcallback = wkf_timer_timenow(cbtimer);
}

static void vmd_timeout_time(float &deltat, float &totalt) {
  double now = wkf_timer_timenow(cbtimer);
  deltat = now - vmd_timeout_lastcallback;
  totalt = now;
  vmd_timeout_lastcallback = now;
}

static int vmd_timeout_cb(void) {
  int earlyexit = 0;
  float deltat, totalt;

  if (vmd_timeout_init == 0) 
    vmd_timeout_reset();

  vmd_timeout_time(deltat, totalt);
  printf("OptiXRenderer) timeout callback: since last %f sec, total %f sec\n",
         deltat, totalt); 
  return earlyexit; 
}

#endif


// assumes current scope has Context variable named 'ctx'
#define RTERR( func )                                                  \
  {                                                                    \
    RTresult code = func;                                              \
    if (code != RT_SUCCESS) {                                          \
      lasterror = code; /* preserve error code for subsequent tests */ \
      const char* message;                                             \
      rtContextGetErrorString(ctx, code, &message);                    \
      msgErr << "OptiXRenderer) ERROR: " << message << " ("            \
             << __FILE__ << ":" << __LINE__ << sendmsg;                \
    }                                                                  \
  }


// assumes current scope has Context variable named 'ctx'
// caller-provided 'code' error return value is used so that subsequent
// code can use that for its own purposes.
#define RTERR2( func, code )                                           \
  {                                                                    \
    code = func;                                                       \
    if (code != RT_SUCCESS) {                                          \
      lasterror = code; /* preserve error code for subsequent tests */ \
      const char* message;                                             \
      rtContextGetErrorString(ctx, code, &message);                    \
      msgErr << "OptiXRenderer) ERROR: " << message << " ("            \
             << __FILE__ << ":" << __LINE__ << sendmsg;                \
    }                                                                  \
  }


#if defined(ORT_USERTXAPIS)

//
// helper routines for OptiX RTX hardware triangle APIs
//

// helper function that tests triangles for degeneracy and 
// computes geometric normals
__forceinline__ int hwtri_test_calc_Ngeom(const __restrict__ float3 *vertices, 
                                          float3 &Ngeometric) {
  // Compute unnormalized geometric normal
  float3 Ng = cross(vertices[1]-vertices[0], vertices[2]-vertices[0]);

  // Cull any degenerate triangles
  float area = length(Ng); // we want non-zero parallelogram area
  if (area > 0.0f && !isinf(area)) {
    // finish normalizing the geometric vector
    Ngeometric = Ng * (1.0f / area);
    return 0; // return success
  }

  return 1; // cull any triangle that fails area test
}


// helper function to allocate and map RT buffers for hardware triangles
static void hwtri_alloc_bufs_v3f_n4u4_c4u(RTcontext ctx, int numfacets,
                                          RTbuffer &vbuf, float3 *&vertices,
                                          RTbuffer &nbuf, uint4 *&normals,
                                          RTbuffer &cbuf, int numcolors,
                                          uchar4 *&colors, 
                                          const float *uniform_color) {
  // Create and fill vertex/normal buffers
  rtBufferCreate(ctx, RT_BUFFER_INPUT, &vbuf);
  rtBufferSetFormat(vbuf, RT_FORMAT_FLOAT3);
  rtBufferSetSize1D(vbuf, numfacets * 3);

  rtBufferCreate(ctx, RT_BUFFER_INPUT, &nbuf);
  rtBufferSetFormat(nbuf, RT_FORMAT_UNSIGNED_INT4);
  rtBufferSetSize1D(nbuf, numfacets);

  rtBufferCreate(ctx, RT_BUFFER_INPUT, &cbuf);
  rtBufferSetFormat(cbuf, RT_FORMAT_UNSIGNED_BYTE4);
  rtBufferSetSize1D(cbuf, numcolors);

  rtBufferMap(vbuf, (void**) &vertices);
  rtBufferMap(nbuf, (void**) &normals);
  rtBufferMap(cbuf, (void**) &colors);

  if ((numcolors == 1) && (uniform_color != NULL)) {
    colors[0].x = uniform_color[0] * 255.0f;
    colors[0].y = uniform_color[1] * 255.0f;
    colors[0].z = uniform_color[2] * 255.0f;
    colors[0].w = 255;
  }
}
 

// helper function to set instance state variables to flag 
// the availability of per-vertex normals and colors for the triangle mesh
static void hwtri_set_vertex_flags(RTcontext ctx, 
                                   RTgeometryinstance instance_hwtri,
                                   RTbuffer nbuf, RTbuffer cbuf,
                                   int has_vertex_normals, 
                                   int has_vertex_colors) {
  RTresult lasterror = RT_SUCCESS;

  // register normal buffer
  RTvariable nbuf_v;
  RTERR( rtGeometryInstanceDeclareVariable(instance_hwtri, "normalBuffer", &nbuf_v) );
  RTERR( rtVariableSetObject(nbuf_v, nbuf) );

  // register color buffer
  RTvariable cbuf_v;
  RTERR( rtGeometryInstanceDeclareVariable(instance_hwtri, "colorBuffer", &cbuf_v) );
  RTERR( rtVariableSetObject(cbuf_v, cbuf) );

  // Enable/disable per-vertex normals (or use geometric normal)
  RTvariable has_vertex_normals_v;
  RTERR( rtGeometryInstanceDeclareVariable(instance_hwtri, "has_vertex_normals", &has_vertex_normals_v) );
  RTERR( rtVariableSet1i(has_vertex_normals_v, has_vertex_normals) );

  // Enable/disable per-vertex colors (or use uniform color) 
  RTvariable has_vertex_colors_v;
  RTERR( rtGeometryInstanceDeclareVariable(instance_hwtri, "has_vertex_colors", &has_vertex_colors_v) );
  RTERR( rtVariableSet1i(has_vertex_colors_v, has_vertex_colors) );
}

#endif


#if defined(VMDOPTIX_INTERACTIVE_OPENGL)

static void print_ctx_devices(RTcontext ctx) {
  unsigned int devcount = 0;
  rtContextGetDeviceCount(ctx, &devcount);
  if (devcount > 0) {
    int *devlist = (int *) calloc(1, devcount * sizeof(int));
    rtContextGetDevices(ctx, devlist);
    printf("OptiXRenderer) Using %d device%s:\n", 
           devcount, (devcount == 1) ? "" : "s");

    unsigned int d;
    for (d=0; d<devcount; d++) {
      char devname[20];
      int cudadev=-1, kto=-1;
      RTsize totalmem;
      memset(devname, 0, sizeof(devname));

      rtDeviceGetAttribute(devlist[d], RT_DEVICE_ATTRIBUTE_NAME, sizeof(devname), devname);
      rtDeviceGetAttribute(devlist[d], RT_DEVICE_ATTRIBUTE_EXECUTION_TIMEOUT_ENABLED, sizeof(int), &kto);
      rtDeviceGetAttribute(devlist[d], RT_DEVICE_ATTRIBUTE_CUDA_DEVICE_ORDINAL, sizeof(int), &cudadev);
      rtDeviceGetAttribute(devlist[d], RT_DEVICE_ATTRIBUTE_TOTAL_MEMORY, sizeof(totalmem), &totalmem);

      printf("OptiXRenderer) [%u] %-19s  CUDA[%d], %.1fGB RAM", 
             d, devname, cudadev, totalmem / (1024.0*1024.0*1024.0));
      if (kto) {
        printf(", KTO");
      }
      printf("\n");
    }
    printf("OptiXRenderer)\n");

    free(devlist); 
  }
}

#endif


static int query_meminfo_ctx_devices(RTcontext &ctx, unsigned long &freemem, unsigned long &physmem) {
  freemem=0;
  physmem=0;
  RTresult lasterror = RT_SUCCESS;

  unsigned int devcount = 0;
  RTERR( rtContextGetDeviceCount(ctx, &devcount) );
  if (devcount > 0) {
    int *devlist = (int *) calloc(1, devcount * sizeof(int));
    RTERR( rtContextGetDevices(ctx, devlist) );
    unsigned int d;
    for (d=0; d<devcount; d++) {
      RTsize freememsz=0;
      RTsize physmemsz=0;
      int ordinal = devlist[d];
      RTERR( rtContextGetAttribute(ctx, static_cast<RTcontextattribute>(RT_CONTEXT_ATTRIBUTE_AVAILABLE_DEVICE_MEMORY+ordinal), sizeof(freememsz), &freememsz) );
      if (lasterror != RT_SUCCESS) {
        free(devlist);
        return -1;
      }
     
      RTERR( rtDeviceGetAttribute(devlist[d], RT_DEVICE_ATTRIBUTE_TOTAL_MEMORY, sizeof(physmemsz), &physmemsz) );
      if (lasterror != RT_SUCCESS) {
        free(devlist);
        return -1;
      }

      if (d==0) {
        freemem = freememsz;
        physmem = physmemsz;
      } else {
        if (freemem < freememsz)
          freemem = freememsz;

        if (physmem < physmemsz)
          physmem = physmemsz;
      }
    }
    free(devlist); 
    return 0;
  }

  return -1;
}


int OptiXPrintRayStats(RTbuffer raystats1_buffer, RTbuffer raystats2_buffer,
                       double rtruntime) {
  int rc = 0;
  RTcontext ctx;
  RTresult result;
  RTsize buffer_width, buffer_height;
  const char* error;

  rtBufferGetContext(raystats1_buffer, &ctx);

  // buffer must be 2-D (for now)
  unsigned int bufdim;
  if (rtBufferGetDimensionality(raystats1_buffer, &bufdim) != RT_SUCCESS) {
    msgErr << "OptiXPrintRayStats: Failed to get ray stats buffer dimensions!" << sendmsg;
    return -1;
  }
  if (bufdim != 2) {
    msgErr << "OptiXPrintRayStats: Output buffer is not 2-D!" << sendmsg;
    return -1;
  }

  result = rtBufferGetSize2D(raystats1_buffer, &buffer_width, &buffer_height);
  if (result != RT_SUCCESS) {
    // Get error from context
    rtContextGetErrorString(ctx, result, &error);
    msgErr << "OptiXRenderer) Error getting dimensions of buffers: " << error << sendmsg;
    return -1;
  }

  volatile uint4 *raystats1, *raystats2;
  result = rtBufferMap(raystats1_buffer, (void**) &raystats1);   
  result = rtBufferMap(raystats2_buffer, (void**) &raystats2);   
  if (result != RT_SUCCESS) {
    rtContextGetErrorString(ctx, result, &error);
    msgErr << "OptiXPrintRayStats: Error mapping stats buffers: "
           << error << sendmsg;
    return -1;
  }

  // no stats data
  if (buffer_width < 1 || buffer_height < 1 || 
      raystats1 == NULL || raystats2 == NULL) {
    msgErr << "OptiXPrintRayStats: No data in ray stats buffers!" << sendmsg;
    return -1;
  }

  // collect and sum all per-pixel ray stats
  int i;
  int totalsz = buffer_width * buffer_height;
  unsigned long misses, transkips;
  unsigned long primaryrays, shadowlights, shadowao, transrays, reflrays;
  misses = transkips = primaryrays = shadowlights 
         = shadowao = transrays = reflrays = 0;

  // accumulate per-pixel ray stats into totals
  for (i=0; i<totalsz; i++) {
    primaryrays  += raystats1[i].x;
    shadowlights += raystats1[i].y;
    shadowao     += raystats1[i].z;
    misses       += raystats1[i].w;
    transrays    += raystats2[i].x;
    transkips    += raystats2[i].y;
    // XXX raystats2[i].z unused at present...
    reflrays     += raystats2[i].w;

  }
  unsigned long totalrays = primaryrays + shadowlights + shadowao 
                          + transrays + reflrays;

  printf("OptiXRenderer)\n");
  printf("OptiXRenderer) VMD/OptiX Scene Ray Tracing Statistics:\n");
  printf("OptiXRenderer) ----------------------------------------\n");
  printf("OptiXRenderer)                     Misses: %lu\n", misses);
  printf("OptiXRenderer) Transmission Any-Hit Skips: %lu\n", transkips);
  printf("OptiXRenderer) ----------------------------------------\n");
  printf("OptiXRenderer)               Primary Rays: %lu\n", primaryrays);
  printf("OptiXRenderer)      Dir-Light Shadow Rays: %lu\n", shadowlights);
  printf("OptiXRenderer)             AO Shadow Rays: %lu\n", shadowao);
  printf("OptiXRenderer)          Transmission Rays: %lu\n", transrays);
  printf("OptiXRenderer)            Reflection Rays: %lu\n", reflrays);
  printf("OptiXRenderer) ----------------------------------------\n");
  printf("OptiXRenderer)                 Total Rays: %lu\n", totalrays); 
  printf("OptiXRenderer)                 Total Rays: %g\n", totalrays * 1.0); 
  if (rtruntime > 0.0) {
    printf("OptiXRenderer)                   Rays/sec: %g\n", totalrays / rtruntime); 
  }
  printf("OptiXRenderer)\n");

  result = rtBufferUnmap(raystats1_buffer);
  result = rtBufferUnmap(raystats2_buffer);
  if (result != RT_SUCCESS) {
    rtContextGetErrorString(ctx, result, &error);
    msgErr << "OptiXPrintRayStats: Error unmapping ray stats buffer: "
           << error << sendmsg;
    return -1;
  }

  return rc;
}


int OptiXWriteImage(const char* filename, int writealpha,
                    RTbuffer buffer, RTformat buffer_format,
                    RTsize buffer_width, RTsize buffer_height) {
  RTresult result;

  void * imageData;
  result = rtBufferMap(buffer, &imageData);
  if (result != RT_SUCCESS) {
    RTcontext ctx;
    const char* error;
    rtBufferGetContext(buffer, &ctx);
    rtContextGetErrorString(ctx, result, &error);
    msgErr << "OptiXWriteImage: Error mapping image buffer: " 
           << error << sendmsg;
    return -1;
  }

  // no image data
  if (buffer_width < 1 || buffer_height < 1 || imageData == NULL) {
    msgErr << "OptiXWriteImage: No image data in output buffer!" << sendmsg;
    return -1;
  }

  // write the image to a file, according to the buffer format
  int xs = buffer_width;
  int ys = buffer_height;
  int rc = 0;
  if (buffer_format == RT_FORMAT_FLOAT4) {
    if (writealpha) {
//printf("Writing rgba4f alpha channel output image 2\n");
      if (write_image_file_rgba4f(filename, (const float *) imageData, xs, ys))
        rc = -1;
    } else {
      if (write_image_file_rgb4f(filename, (const float *) imageData, xs, ys))
        rc = -1;
    }
  } else if (buffer_format == RT_FORMAT_UNSIGNED_BYTE4) {
    if (writealpha) {
//printf("Writing rgba4u alpha channel output image 2\n");
      if (write_image_file_rgba4u(filename, (const unsigned char *) imageData, xs, ys))
        rc = -1;
    } else {
      if (write_image_file_rgb4u(filename, (const unsigned char *) imageData, xs, ys))
        rc = -1;
    }
  } else {
    rc = -1;
  }

  result = rtBufferUnmap(buffer);
  if (result != RT_SUCCESS) {
    RTcontext ctx;
    const char* error;
    rtBufferGetContext(buffer, &ctx);
    rtContextGetErrorString(ctx, result, &error);
    msgErr << "OptiXWriteImage: Error unmapping image buffer: " 
           << error << sendmsg;
    return -1;
  }

  return rc;
}


int OptiXWriteImage(const char* filename, int writealpha, RTbuffer buffer) {
  RTresult result;
  RTformat buffer_format;
  RTsize buffer_width, buffer_height;

  // buffer must be 2-D
  unsigned int bufdim;
  if (rtBufferGetDimensionality(buffer, &bufdim) != RT_SUCCESS) {
    msgErr << "OptiXWriteImage: Failed to get output buffer dimensions!" << sendmsg;
    return -1;
  }

  if (bufdim != 2) {
    msgErr << "OptiXWriteImage: Output buffer is not 2-D!" << sendmsg;
    return -1;
  }

  void * imageData;
  result = rtBufferMap(buffer, &imageData);
  if (result != RT_SUCCESS) {
    RTcontext ctx;
    const char* error;
    rtBufferGetContext(buffer, &ctx);
    rtContextGetErrorString(ctx, result, &error);
    msgErr << "OptiXWriteImage: Error mapping image buffer: " 
           << error << sendmsg;
    return -1;
  }

  // no image data
  if (imageData == NULL) {
    msgErr << "OptiXWriteImage: No image data in output buffer!" << sendmsg;
    return -1;
  }

  result = rtBufferGetSize2D(buffer, &buffer_width, &buffer_height);
  if (result != RT_SUCCESS) {
    // Get error from context
    RTcontext ctx;
    const char* error;
    rtBufferGetContext(buffer, &ctx);
    rtContextGetErrorString(ctx, result, &error);
    msgErr << "OptiXRenderer) Error getting dimensions of buffer: " << error << sendmsg;
    return -1;
  }

  if (rtBufferGetFormat(buffer, &buffer_format) != RT_SUCCESS) {
    msgErr << "OptiXWriteImage: failed to query output buffer format!" 
           << sendmsg;
    return -1;
  }

  // write the image to a file, according to the buffer format
  int xs = buffer_width;
  int ys = buffer_height;
  int rc = 0;

  if (buffer_format == RT_FORMAT_FLOAT4) {
    if (writealpha) {
//printf("Writing rgba4f alpha channel output image 1\n");
      if (write_image_file_rgba4f(filename, (const float *) imageData, xs, ys))
        rc = -1;
    } else {
      if (write_image_file_rgb4f(filename, (const float *) imageData, xs, ys))
        rc = -1;
    }
  } else if (buffer_format == RT_FORMAT_UNSIGNED_BYTE4) {
    if (writealpha) {
//printf("Writing rgba4u alpha channel output image 1\n");
      if (write_image_file_rgba4u(filename, (const unsigned char *) imageData, xs, ys))
        rc = -1;
    } else {
      if (write_image_file_rgb4u(filename, (const unsigned char *) imageData, xs, ys))
        rc = -1;
    }
  } else {
    rc = -1;
  }

  result = rtBufferUnmap(buffer);
  if (result != RT_SUCCESS) {
    RTcontext ctx;
    const char* error;
    rtBufferGetContext(buffer, &ctx);
    rtContextGetErrorString(ctx, result, &error);
    msgErr << "OptiXWriteImage: Error unmapping image buffer: "
           << error << sendmsg;
    return -1;
  }

  return rc;
}


/// constructor ... initialize some variables
OptiXRenderer::OptiXRenderer(VMDApp *vmdapp, void *rdev) {
  PROFILE_PUSH_RANGE("OptiXRenderer::OptiXRenderer()", 0);
  app = vmdapp;                   // store VMDApp ptr for video streaming
  ort_timer = wkf_timer_create(); // create and initialize timer
  wkf_timer_start(ort_timer);

  // copy remote device pointer
  remote_device = rdev;

  // setup path to pre-compiled shader PTX code
  const char *vmddir = getenv("VMDDIR");
  if (vmddir == NULL)
    vmddir = ".";
  sprintf(shaderpath, "%s/shaders/%s", vmddir, "OptiXShaders.ptx");

  // allow runtime override of the default shader path for testing
  if (getenv("VMDOPTIXSHADERPATH")) {
    strcpy(shaderpath, getenv("VMDOPTIXSHADERPATH"));
    msgInfo << "User-override of OptiX shader path: " << getenv("VMDOPTIXSHADERPATH") << sendmsg;
  }

#if defined(ORT_USERTXAPIS)
  hwtri_enabled = 1;           // RTX hardware triangle APIs enabled by default
#endif
  lasterror = RT_SUCCESS;      // begin with no error state set 
  context_created = 0;         // no context yet
  buffers_allocated = 0;       // flag no buffer allocated yet
  buffers_progressive = 0;     // buf bound using progressive API or not
  scene_created = 0;           // scene has been created

  // clear timers
  time_ctx_setup = 0.0;
  time_ctx_validate = 0.0;
  time_ctx_AS_build = 0.0;
  time_ray_tracing = 0.0;
  time_image_io = 0.0;

  // set default scene background state
  scene_background_mode = RT_BACKGROUND_TEXTURE_SOLID;
  memset(scene_bg_color,    0, sizeof(scene_bg_color));
  memset(scene_bg_grad_top, 0, sizeof(scene_bg_grad_top));
  memset(scene_bg_grad_bot, 0, sizeof(scene_bg_grad_bot));
  memset(scene_gradient,    0, sizeof(scene_gradient));
  scene_gradient_topval = 1.0f;
  scene_gradient_botval = 0.0f;
  // XXX this has to be recomputed prior to rendering..
  scene_gradient_invrange = 1.0f / (scene_gradient_topval - scene_gradient_botval);

  // zero out the array of material usage counts for the scene
  memset(material_special_counts, 0, sizeof(material_special_counts));

  cam_zoom = 1.0f;
  cam_stereo_eyesep = 0.06f;
  cam_stereo_convergence_dist = 2.0f;

  clipview_mode = RT_CLIP_NONE;      // VR HMD fade+clipping plane/sphere
  clipview_start = 1.0f;             // VR HMD fade+clipping radial start dist
  clipview_end = 0.2f;               // VR HMD fade+clipping radial end dist

  // check for VR headlight and HMD/camera view clipping plane/sphere
  if (getenv("VMDOPTIXCLIPVIEW")) {
    clipview_mode = RT_CLIP_SPHERE;
    msgInfo << "OptiXRenderer) Overriding default clipping mode with RT_CLIP_SPHERE" << sendmsg;
  }
  if (getenv("VMDOPTIXCLIPVIEWSTART")) {
    clipview_start = atof(getenv("VMDOPTIXCLIPVIEWSTART"));
    msgInfo << "OptiXRenderer) Overriding default clipping start: " 
            << clipview_start << sendmsg;
  }
  if (getenv("VMDOPTIXCLIPVIEWEND")) {
    clipview_start = atof(getenv("VMDOPTIXCLIPVIEWEND"));
    msgInfo << "OptiXRenderer) Overriding default clipping end: " 
            << clipview_start << sendmsg;
  }

  headlight_mode = RT_HEADLIGHT_OFF; // VR HMD headlight disabled by default
  if (getenv("VMDOPTIXHEADLIGHT")) {
    headlight_mode = RT_HEADLIGHT_ON;
    msgInfo << "OptiXRenderer) Overriding default headlight mode with RT_HEADLIGHT_ON" << sendmsg;
  }

  shadows_enabled = RT_SHADOWS_OFF;  // disable shadows by default 
  aa_samples = 0;                    // no AA samples by default

  ao_samples = 0;                    // no AO samples by default
  ao_direct = 0.3f;                  // AO direct contribution is 30%
  ao_ambient = 0.7f;                 // AO ambient contribution is 70%
  ao_maxdist = RT_DEFAULT_MAX;       // default is no max occlusion distance

  dof_enabled = 0;                   // disable DoF by default
  cam_dof_focal_dist = 2.0f;
  cam_dof_fnumber = 64.0f;

  fog_mode = RT_FOG_NONE;            // fog/cueing disabled by default
  fog_start = 0.0f;
  fog_end = 10.0f;
  fog_density = 0.32f;

  verbose = RT_VERB_MIN;  // keep console quiet except for perf/debugging cases
  check_verbose_env();    // see if the user has overridden verbose flag

  create_context();
  destroy_scene();        // zero out object counters, prepare for rendering

  PROFILE_POP_RANGE();
}
        
/// destructor
OptiXRenderer::~OptiXRenderer(void) {
  PROFILE_PUSH_RANGE("OptiXRenderer::~OptiXRenderer()", 0);

  if (context_created)
    destroy_context(); 
  wkf_timer_destroy(ort_timer);

  PROFILE_POP_RANGE();
}


void OptiXRenderer::check_verbose_env() {
  verbose = get_verbose_flag(1);
}


#if OPTIX_VERSION >= 3080
//
// routines for managing remote VCA rendering (OptiX >= 3.8 only)
//
RTRDev OptiXRenderer::remote_connect(const char *cluster,
                                     const char *user, 
                                     const char *passwd) {
  OptiXRenderer::Verbosity dl_verbose = get_verbose_flag();
  if (dl_verbose == RT_VERB_DEBUG)
     printf("OptiXRenderer) OptiXRenderer::remote_connect()\n");

  msgInfo << "OptiX VCA remote connection" << sendmsg;
  msgInfo << "  URL: '" << cluster << "'" << sendmsg;
  msgInfo << " User: '" << user << "'" << sendmsg;
  msgInfo << "   pw: '" << passwd << "'" << sendmsg;

  RTremotedevice *rdev = (RTremotedevice *) calloc(1, sizeof(RTremotedevice));
  RTresult code = rtRemoteDeviceCreate(cluster, user, passwd, rdev);
  if (code != RT_SUCCESS) {
    if (code == RT_ERROR_NETWORK_LOAD_FAILED) {
      msgErr << "OptiX VCA remote connect:" << sendmsg;
      msgErr << "  OptiX libdice and/or other shared libraries are missing" 
             << sendmsg;
      msgErr << "  from VMD installation directory!!!" << sendmsg;
    }
    msgErr << "OptiXRenderer) VCA: failed to login to remote VCA cluster" << sendmsg;
    free(rdev);
    return NULL;
  }
  msgInfo << "OptiXRenderer) VCA: remote connection established." << sendmsg;

  char clusturl[] = "unknown                                                  ";
  rtRemoteDeviceGetAttribute(*rdev, RT_REMOTEDEVICE_ATTRIBUTE_CLUSTER_URL,
                             sizeof(clusturl), &clusturl);
  msgInfo << "OptiXRenderer) VCA: remote connection cluster URL: "
          << clusturl << sendmsg;

  unsigned int numconfigs = 0;
  rtRemoteDeviceGetAttribute(*rdev,RT_REMOTEDEVICE_ATTRIBUTE_NUM_CONFIGURATIONS,
                             sizeof(numconfigs), &numconfigs);
  msgInfo << "OptiXRenderer) VCA: configuration count: " << numconfigs << sendmsg;

  unsigned int l; 
  const unsigned int badconfig = 999999;
  unsigned int vcaconfigidx=badconfig;
  const char optix38string[] = 
#if 1
    "NVIDIA OptiX 3.8.0 (Version 19623407 Bridge 231000.5154 Protocol 11)";
#elif 1
    "NVIDIA OptiX 3.8.0 (Version 19519708 Bridge 231000.5154 Protocol 11)";
#elif 1
    "NVIDIA OptiX 3.8.0-beta1 (Version 19398853 Bridge 231000.5154 Protocol 10)";
#else
    "NVIDIA OptiX Alpha (Version 19329292 Bridge 231000.5154 Protocol 7)";
#endif

  for (l=0; l<numconfigs; l++) {
    char VCA_config_name[256];
    memset(VCA_config_name, 0, sizeof(VCA_config_name));
    rtRemoteDeviceGetAttribute(*rdev, 
                               (RTremotedeviceattribute) (RT_REMOTEDEVICE_ATTRIBUTE_CONFIGURATIONS + l),
                               sizeof(VCA_config_name), VCA_config_name);
    if (!strcmp(VCA_config_name, optix38string)) {
      vcaconfigidx=l;
    }

    msgInfo << " [" << l << "] " << VCA_config_name << sendmsg;
  }

  if (getenv("VMDOPTIXVCACONFIG")) {
    unsigned int idx = atoi(getenv("VMDOPTIXVCACONFIG"));
    if (idx < numconfigs)
      vcaconfigidx = idx;

    printf("OptiXRenderer) User-specified OptiX VCA config index: %d\n",
            vcaconfigidx);
  }

  if (vcaconfigidx == badconfig && numconfigs > 0) {
    vcaconfigidx = 0;
    msgInfo << "OptiXRenderer) VCA: didn't match a config, trying config [0]" << sendmsg;
  }


  int resvnodes = 2;
  if (getenv("VMDOPTIXVCANODES") != NULL) {
    resvnodes = atoi(getenv("VMDOPTIXVCANODES"));
  }

  if (vcaconfigidx != badconfig) {
    msgInfo << "OptiXRenderer) VCA: reserving " << resvnodes << " nodes" << sendmsg;
    rtRemoteDeviceReserve(*rdev, resvnodes, vcaconfigidx); 

    // enter polling loop waiting until reservation is ready...
    int rdevready=0; 
    int pollcount=0;
    int existingresv=0;
    do {
      if ((existingresv == 0) && (pollcount > 10) && 
          (rdevready == RT_REMOTEDEVICE_STATUS_CONNECTED)) {
        existingresv=1; // prevent infinite loop
        printf("\n");
        msgInfo << "OptiXRenderer) VCA: may have a stuck reservation" << sendmsg;
        rtRemoteDeviceRelease(*rdev); 

        msgInfo << "OptiXRenderer) Restarting reservation process..." <<  sendmsg;
        pollcount = 0;  // reset poll counter 
        rtRemoteDeviceReserve(*rdev, resvnodes, vcaconfigidx); 
      }

      if (pollcount > 20)
        break;

      vmd_msleep(500);
      rtRemoteDeviceGetAttribute(*rdev, RT_REMOTEDEVICE_ATTRIBUTE_STATUS,
                                 sizeof(rdevready), &rdevready);
      char statec = '.';  
      switch (rdevready) {
        case RT_REMOTEDEVICE_STATUS_CONNECTED:
          statec = 'C';
          break;

        case RT_REMOTEDEVICE_STATUS_DISCONNECTED:
          statec = 'D';
          break;

        case RT_REMOTEDEVICE_STATUS_RESERVED:
          statec = 'r';
          break;
        
        case RT_REMOTEDEVICE_STATUS_READY:
          statec = 'R';
          break;

        default:
          statec = '.';
          break;
      }

      printf("%c", statec);
      fflush(stdout);

      pollcount++;
    } while (rdevready != RT_REMOTEDEVICE_STATUS_READY);
    printf("\n");

    if (rdevready != RT_REMOTEDEVICE_STATUS_READY) {
      msgErr << "OptiXRenderer) VCA: reservation timed out, closing connection" << sendmsg;
      rtRemoteDeviceRelease(*rdev); 
      rtRemoteDeviceDestroy(*rdev);  
      free(rdev);
      rdev=NULL;
      return rdev;
    }
    msgInfo << "OptiXRenderer) VCA: reservation ready." << sendmsg;

    // Once the remote device connection is established, we can 
    // query and print info about the node count, gpu count, and other
    // useful metadata...
    int rdevnodecount = 0;
    int rdevgpucount = 0;
    rtRemoteDeviceGetAttribute(*rdev, 
                               RT_REMOTEDEVICE_ATTRIBUTE_NUM_RESERVED_NODES,
                               sizeof(rdevnodecount), &rdevnodecount);
   
    rtRemoteDeviceGetAttribute(*rdev,
                               RT_REMOTEDEVICE_ATTRIBUTE_NUM_GPUS,
                               sizeof(rdevgpucount), &rdevgpucount);

    msgInfo << "OptiXRenderer) VCA node: " << rdevnodecount 
            << "  GPUs: " << rdevgpucount << sendmsg;
  } else {
    msgErr << "OptiXRenderer) VCA: unable to match a usable configuration!" << sendmsg;
  }

  return rdev;
}


void OptiXRenderer::remote_detach(RTRDev vrdev) {
  OptiXRenderer::Verbosity dl_verbose = get_verbose_flag();
  if (dl_verbose == RT_VERB_DEBUG)
     printf("OptiXRenderer) OptiXRenderer::remote_detach()\n");

  RTremotedevice *rdev = (RTremotedevice *) vrdev;

  msgInfo << "OptiXRenderer) VCA: remote connection detach" << sendmsg;
  if (rdev) {
    msgInfo << "OptiXRenderer) VCA: releasing reservation." << sendmsg;
    rtRemoteDeviceRelease(*rdev);
    msgInfo << "OptiXRenderer) VCA: destroying remote connection." << sendmsg;
    rtRemoteDeviceDestroy(*rdev);  
    free(rdev);
  }
}

#endif


//
// This routine enumerates the set of GPUs that are usable by OptiX,
// both in terms of their compatibility with the OptiX library we have
// compiled against, and also in terms of user preferences to exclude
// particular GPUs, GPUs that have displays attached, and so on.
//
unsigned int OptiXRenderer::device_list(int **devlist, char ***devnames) {
  OptiXRenderer::Verbosity dl_verbose = get_verbose_flag();
  if (dl_verbose == RT_VERB_DEBUG)
     printf("OptiXRenderer) OptiXRenderer::device_list()\n");

  unsigned int count=0;
  RTresult devcntresult = rtDeviceGetDeviceCount(&count);
  if (devcntresult != RT_SUCCESS) {
#if OPTIX_VERSION >= 50200
    if (devcntresult == RT_ERROR_OPTIX_NOT_LOADED) {
      printf("OptiXRenderer) ERROR: Failed to load the OptiX shared library.\n");
      printf("OptiXRenderer)        NVIDIA driver may be too old.\n");
      printf("OptiXRenderer)        Check/update NVIDIA driver\n");
    }
#endif

    if (dl_verbose == RT_VERB_DEBUG) {
      printf("OptiXRenderer)   rtDeviceGetDeviceCount() returned error %08x\n",
             devcntresult);
      printf("OptiXRenderer)   No GPUs available\n"); 
    }

    count = 0;
    if (devlist != NULL)
      *devlist = NULL;
    if (devnames != NULL)
      *devnames = NULL;

    return 0;
  }

  if (dl_verbose == RT_VERB_DEBUG) {
    printf("OptiXRenderer) OptiX rtDeviceGetDeviceCount() reports\n");
    printf("OptiXRenderer) that %d GPUs are available\n", count);
  }

  // check to see if the user wants to limit what device(s) are used
  unsigned int gpumask = 0xffffffff;
  const char *gpumaskstr = getenv("VMDOPTIXDEVICEMASK");
  if (gpumaskstr != NULL) {
    unsigned int tmp;
    if (sscanf(gpumaskstr, "%x", &tmp) == 1) {
      gpumask = tmp;
      msgInfo << "Using OptiX device mask '"
              << gpumaskstr << "'" << sendmsg;
    } else {
      msgInfo << "Failed to parse OptiX GPU device mask string '"
              << gpumaskstr << "'" << sendmsg;
    }
  }

  if (devlist != NULL) {
    *devlist = NULL;
    if (count > 0) {
      *devlist = (int *) calloc(1, count * sizeof(int));  
    }
  }
  if (devnames != NULL) {
    *devnames = NULL;
    if (count > 0) {
      *devnames = (char **) calloc(1, count * sizeof(char *));  
    }
  }

  // walk through the list of available devices and screen out
  // any that may cause problems with the version of OptiX we are using
  unsigned int i, goodcount;
  for (goodcount=0,i=0; i<count; i++) {
    // check user-defined GPU device mask for OptiX...
    if (!(gpumask & (1 << i))) {
      if (dl_verbose == RT_VERB_DEBUG) {
        char msgbuf[1024];
        sprintf(msgbuf, "  Excluded GPU[%d] due to user-specified device mask\n", i);
        msgInfo << msgbuf << sendmsg;
      }
      continue;
    } 

    // check user-requested exclusion of devices with display timeouts enabled
    int timeoutenabled;
    rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_EXECUTION_TIMEOUT_ENABLED, 
                         sizeof(int), &timeoutenabled);
    if (timeoutenabled && getenv("VMDOPTIXNODISPLAYGPUS")) {
      if (dl_verbose == RT_VERB_DEBUG) {
        char msgbuf[1024];
        sprintf(msgbuf, "  Excluded GPU[%d] due to user-specified display timeout exclusion \n", i);
        msgInfo << msgbuf << sendmsg;
      }
      continue;
    } 

    //
    // screen for viable compute capability for this version of OptiX
    //
    // XXX this should be unnecessary with OptiX 3.6.x and later (I hope)
    //
    int compute_capability[2];
    rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY, 
                         sizeof(compute_capability), compute_capability);
//    printf("OptiX GPU[%d] compute capability %d\n", i, compute_capability[0]);
#if OPTIX_VERSION <= 3051
    // exclude Maxwell and later GPUs if we're running OptiX 3.5.1 or earlier
    if (compute_capability[0] > 3) {
      if (dl_verbose == RT_VERB_DEBUG) {
        char msgbuf[1024];
        sprintf(msgbuf, "  Excluded GPU[%d] due to unsupported compute capability\n", i);
        msgInfo << msgbuf << sendmsg;
      }
      continue;
    }
#endif

    // record all usable GPUs we find...
    if (dl_verbose == RT_VERB_DEBUG) {
      char msgbuf[1024];
      sprintf(msgbuf, "Found usable GPU[%i]\n", i);
      msgInfo << msgbuf << sendmsg;
    }

    if (devlist != NULL) {
      if (dl_verbose == RT_VERB_DEBUG) {
        char msgbuf[1024];
        sprintf(msgbuf, "  Adding usable GPU[%i] to list[%d]\n", i, goodcount);
        msgInfo << msgbuf << sendmsg;
      }
      (*devlist)[goodcount] = i;
    }

    if (devnames != NULL) {
      char *namebuf = (char *) calloc(1, 65 * sizeof(char));
      rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_NAME, 
                           64*sizeof(char), namebuf);
      if (dl_verbose == RT_VERB_DEBUG) {
        char msgbuf[1024];
        sprintf(msgbuf, "  Adding usable GPU[%i] to list[%d]: '%s'\n", i, goodcount, namebuf);
        msgInfo << msgbuf << sendmsg;
      }
      (*devnames)[goodcount] = namebuf;
    }
    goodcount++;
  }

  return goodcount;
}


unsigned int OptiXRenderer::device_count(void) {
  OptiXRenderer::Verbosity dl_verbose = get_verbose_flag();
  if (dl_verbose == RT_VERB_DEBUG)
     printf("OptiXRenderer) OptiXRenderer::device_count()\n");

#if 1
  return device_list(NULL, NULL);
#else
  unsigned int count=0;
  if (rtDeviceGetDeviceCount(&count) != RT_SUCCESS)
    count = 0;
  return count;
#endif
}


unsigned int OptiXRenderer::optix_version(void) {
  OptiXRenderer::Verbosity dl_verbose = get_verbose_flag();
  if (dl_verbose == RT_VERB_DEBUG)
     msgInfo << "OptiXRenderer) OptiXRenderer::optix_version()" << sendmsg;

  unsigned int version=0;
  if (rtGetVersion(&version) != RT_SUCCESS)
    version = 0;
  return version;
}


int OptiXRenderer::material_shader_table_size(void) {
  // used for initialization info printed to console
#if defined(ORT_USE_TEMPLATE_SHADERS)
  return ORTMTABSZ;
#else
  return 1;
#endif
}


void OptiXRenderer::create_context() {
  time_ctx_create = 0;
  if (context_created)
    return;

  double starttime = wkf_timer_timenow(ort_timer);

  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) creating context...\n");

#if defined(ORT_USERTXAPIS)
  // OptiX 5.2 allows runtime manipulation of OptiX RTX execution strategy
  // using a new API flag.  RTX mode is only supported on Maxwell and later
  // GPUs, earlier Kepler hardware do not support RTX mode.
  int rtxonoff = 1;
  if (getenv("VMDOPTIXNORTX") != NULL) {
    rtxonoff = 0;

    // if the RTX mode isn't on, then we can't use the hardware triangle APIs.
    hwtri_enabled = 0;
  }

  // XXX Presently, the execution strategy mode has to be set very early on,
  //     otherwise the API call will return without error, but the OptiX
  //     runtime will ignore the state change and continue with the existing
  //     execution strategy (e.g. the default of '0' if set too late...)
  if (rtGlobalSetAttribute(RT_GLOBAL_ATTRIBUTE_ENABLE_RTX, sizeof(rtxonoff), &rtxonoff) != RT_SUCCESS) {
    printf("OptiXRenderer) Error setting RT_GLOBAL_ATTRIBUTE_ENABLE_RTX!!!\n");
  } else {
    if (verbose == RT_VERB_TIMING || verbose == RT_VERB_DEBUG)
      printf("OptiXRenderer) OptiX RTX execution mode is %s.\n",
             (rtxonoff) ? "on" : "off");
  }
#elif 0 && defined(ORT_USERTXAPIS)
  // XXX this needs to go away when the production OptiX revs no longer 
  //     allow runtime manipulation of OptiX RTX execution strategy
  //     (it will be forced on/off by detection of the GPU hardware type)
  int expexeconoff = 0;
  if (getenv("VMDOPTIXEXPEXECSTRATEGY") != NULL) {
    expexeconoff = 1;
  } else {
    // if the experimental execution strategy isn't on, then we can't
    // use the hardware triangle APIs.
    hwtri_enabled = 0;
  }

  // XXX Presently, the execution strategy mode has to be set very early on,
  //     otherwise the API call will return without error, but the OptiX
  //     runtime will ignore the state change and continue with the existing
  //     execution strategy (e.g. the default of '0' if set too late...)
  if (rtGlobalSetAttribute(RT_GLOBAL_ATTRIBUTE_EXPERIMENTAL_EXECUTION_STRATEGY, sizeof(expexeconoff), &expexeconoff) != RT_SUCCESS) {
    printf("OptiXRenderer) Error setting RT_GLOBAL_ATTRIBUTE_EXPERIMENTAL_EXECUTION_STRATEGY!!!\n");
  } else {
    if (verbose == RT_VERB_TIMING || verbose == RT_VERB_DEBUG)
      printf("OptiXRenderer) OptiX 5.2 experimental execution mode is %s.\n",
             (expexeconoff) ? "on" : "off");
  }
#endif

  // Create our objects and set state
  RTresult ctxrc;
  RTERR2( rtContextCreate(&ctx), ctxrc );
  if (ctxrc != RT_SUCCESS) {
    msgErr << "OptiXRenderer) Failed to create OptiX rendering context" << sendmsg;
    context_created=0;
    return;
  }

#if defined(VMDOPTIX_PROGRESSIVEAPI)
  // If we have a connection to a remote rendering device such as a VCA,
  // we ensure it is used by the active OptiX context.
  if (remote_device) {
    if (verbose == RT_VERB_DEBUG) 
      printf("OptiXRenderer) attaching context to remote device...\n");

    RTremotedevice *rdev = (RTremotedevice *) remote_device;
    rtContextSetRemoteDevice(ctx, *rdev);
  }
#endif

  // screen and set what GPU device(s) are used for this context
  // We shouldn't need the compute capability exclusions post-OptiX 3.6.x,
  // but this will benefit from other updates.
  if (getenv("VMDOPTIXDEVICEMASK") != NULL) {
    int *optixdevlist;
    int optixdevcount = device_list(&optixdevlist, NULL);
    if (optixdevcount > 0) {
      RTERR( rtContextSetDevices(ctx, optixdevcount, optixdevlist) );
    }
  } else if (getenv("VMDOPTIXDEVICE") != NULL) {
    int optixdev = atoi(getenv("VMDOPTIXDEVICE"));
    msgInfo << "Setting OptiX GPU device to: " << optixdev << sendmsg;
    RTERR( rtContextSetDevices(ctx, 1, &optixdev) );
  }

  // register ray types for both shadow and radiance rays
  RTERR( rtContextSetRayTypeCount(ctx, RT_RAY_TYPE_COUNT) );

  // flag to indicate whether we're running in progressive mode or not
  RTERR( rtContextDeclareVariable(ctx, "progressive_enabled", &progressive_enabled_v) );
  RTERR( rtVariableSet1i(progressive_enabled_v, 0) );

  // declare various internal state variables
  RTERR( rtContextDeclareVariable(ctx, "max_depth", &max_depth_v) );
  RTERR( rtContextDeclareVariable(ctx, "max_trans", &max_trans_v) );
  RTERR( rtContextDeclareVariable(ctx, "radiance_ray_type", &radiance_ray_type_v) );
  RTERR( rtContextDeclareVariable(ctx, "shadow_ray_type", &shadow_ray_type_v) );
  RTERR( rtContextDeclareVariable(ctx, "scene_epsilon", &scene_epsilon_v) );

  // create light buffers/variables now, populate at render time...
#if defined(VMDOPTIX_LIGHTUSEROBJS)
  RTERR( rtContextDeclareVariable(ctx, "dir_light_list", &dir_light_list_v) );
  RTERR( rtContextDeclareVariable(ctx, "pos_light_list", &pos_light_list_v) );
#else
  RTERR( rtContextDeclareVariable(ctx, "dir_lights", &dir_lightbuffer_v) );
  RTERR( rtBufferCreate(ctx, RT_BUFFER_INPUT, &dir_lightbuffer) );
  RTERR( rtBufferSetFormat(dir_lightbuffer, RT_FORMAT_USER) );
  RTERR( rtBufferSetElementSize(dir_lightbuffer, sizeof(DirectionalLight)) );

  RTERR( rtContextDeclareVariable(ctx, "pos_lights", &pos_lightbuffer_v) );
  RTERR( rtBufferCreate(ctx, RT_BUFFER_INPUT, &pos_lightbuffer) );
  RTERR( rtBufferSetFormat(pos_lightbuffer, RT_FORMAT_USER) );
  RTERR( rtBufferSetElementSize(pos_lightbuffer, sizeof(PositionalLight)) );
#endif

  // Current accumulation subframe count, used as part of generating
  // AA and AO random number sequences
  RTERR( rtContextDeclareVariable(ctx, "accumCount", &accum_count_v) );
  RTERR( rtVariableSet1ui(accum_count_v, 0) );

  // AO direct lighting scale factors, max occlusion distance
  RTERR( rtContextDeclareVariable(ctx, "ao_direct", &ao_direct_v) );
  RTERR( rtContextDeclareVariable(ctx, "ao_ambient", &ao_ambient_v) );
  RTERR( rtContextDeclareVariable(ctx, "ao_maxdist", &ao_maxdist_v) );

  // shadows, antialiasing, ambient occlusion
  RTERR( rtContextDeclareVariable(ctx, "shadows_enabled", &shadows_enabled_v) );
  RTERR( rtContextDeclareVariable(ctx, "aa_samples", &aa_samples_v) );
  RTERR( rtContextDeclareVariable(ctx, "ao_samples", &ao_samples_v) );

  // background color / gradient
  RTERR( rtContextDeclareVariable(ctx, "scene_bg_color", &scene_bg_color_v) );
  RTERR( rtContextDeclareVariable(ctx, "scene_bg_color_grad_top", &scene_bg_grad_top_v) );
  RTERR( rtContextDeclareVariable(ctx, "scene_bg_color_grad_bot", &scene_bg_grad_bot_v) );
  RTERR( rtContextDeclareVariable(ctx, "scene_gradient", &scene_gradient_v) );
  RTERR( rtContextDeclareVariable(ctx, "scene_gradient_topval", &scene_gradient_topval_v) );
  RTERR( rtContextDeclareVariable(ctx, "scene_gradient_botval", &scene_gradient_botval_v) );
  RTERR( rtContextDeclareVariable(ctx, "scene_gradient_invrange", &scene_gradient_invrange_v) );

  // VR HMD variables
  RTERR( rtContextDeclareVariable(ctx, "clipview_mode", &clipview_mode_v) );
  RTERR( rtContextDeclareVariable(ctx, "clipview_start", &clipview_start_v) );
  RTERR( rtContextDeclareVariable(ctx, "clipview_end", &clipview_end_v) );
  RTERR( rtContextDeclareVariable(ctx, "headlight_mode", &headlight_mode_v) );

  // cueing/fog variables
  RTERR( rtContextDeclareVariable(ctx, "fog_mode", &fog_mode_v) );
  RTERR( rtContextDeclareVariable(ctx, "fog_start", &fog_start_v) );
  RTERR( rtContextDeclareVariable(ctx, "fog_end", &fog_end_v) );
  RTERR( rtContextDeclareVariable(ctx, "fog_density", &fog_density_v) );

  // variables for top level scene graph objects
  RTERR( rtContextDeclareVariable(ctx, "root_object", &root_object_v) );
  RTERR( rtContextDeclareVariable(ctx, "root_shadower", &root_shadower_v) );

  // define all of the standard camera params
  RTERR( rtContextDeclareVariable(ctx, "cam_zoom", &cam_zoom_v) );
  RTERR( rtContextDeclareVariable(ctx, "cam_pos", &cam_pos_v) );
  RTERR( rtContextDeclareVariable(ctx, "cam_U", &cam_U_v) );
  RTERR( rtContextDeclareVariable(ctx, "cam_V", &cam_V_v) );
  RTERR( rtContextDeclareVariable(ctx, "cam_W", &cam_W_v) );

  // define stereoscopic camera parameters
  RTERR( rtContextDeclareVariable(ctx, "cam_stereo_eyesep", &cam_stereo_eyesep_v) );
  RTERR( rtContextDeclareVariable(ctx, "cam_stereo_convergence_dist", &cam_stereo_convergence_dist_v) );

  // define camera DoF parameters
  RTERR( rtContextDeclareVariable(ctx, "cam_dof_focal_dist", &cam_dof_focal_dist_v) );
  RTERR( rtContextDeclareVariable(ctx, "cam_dof_aperture_rad", &cam_dof_aperture_rad_v) );

  RTERR( rtContextDeclareVariable(ctx, "accumulation_normalization_factor", &accum_norm_v) );


  //
  // allow runtime override of the default shader path for testing
  // this has to be done prior to all calls that load programs from
  // the shader PTX
  //
  if (getenv("VMDOPTIXSHADERPATH")) {
    strcpy(shaderpath, getenv("VMDOPTIXSHADERPATH"));
    if (verbose == RT_VERB_DEBUG) 
      printf("OptiXRenderer) user-override shaderpath: '%s'\n", shaderpath);
  }

  if (verbose >= RT_VERB_TIMING) {
    printf("OptiXRenderer) creating shader programs...\n");
    fflush(stdout);
  }

  // load and initialize all of the material programs
  init_materials();

  double time_materials = wkf_timer_timenow(ort_timer); 
  if (verbose >= RT_VERB_TIMING) {
    printf("OptiXRenderer)   ");
    printf("materials(%.1f) ", time_materials - starttime);
    fflush(stdout);
  }

#if defined(ORT_RAYSTATS)
  // program for clearing the raystats buffers
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath, "clear_raystats_buffers", &clear_raystats_buffers_pgm) );
#endif

  // program for clearing the accumulation buffer
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath, "clear_accumulation_buffer", &clear_accumulation_buffer_pgm) );

  // program for copying the accumulation buffer to the framebuffer
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath, "draw_accumulation_buffer", &draw_accumulation_buffer_pgm) );

  // empty placeholder program for copying the accumulation buffer
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath, "draw_accumulation_buffer_stub", &draw_accumulation_buffer_stub_pgm) );

  double time_fbops = wkf_timer_timenow(ort_timer);
  if (verbose >= RT_VERB_TIMING) {
    printf("fbops(%.1f) ", time_fbops - time_materials);
    fflush(stdout);
  }

  // register cubemap VR camera ray gen programs
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath,
         "vmd_camera_cubemap", &ray_gen_pgm_cubemap) );
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath,
         "vmd_camera_cubemap_dof", &ray_gen_pgm_cubemap_dof) );
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath,
         "vmd_camera_cubemap_stereo", &ray_gen_pgm_cubemap_stereo) );
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath,
         "vmd_camera_cubemap_stereo_dof", &ray_gen_pgm_cubemap_stereo_dof) );

  // register planetarium dome master camera ray gen programs
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath,
         "vmd_camera_dome_master", &ray_gen_pgm_dome_master) );
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath,
         "vmd_camera_dome_master_dof", &ray_gen_pgm_dome_master_dof) );
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath,
         "vmd_camera_dome_master_stereo", &ray_gen_pgm_dome_master_stereo) );
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath,
         "vmd_camera_dome_master_stereo_dof", &ray_gen_pgm_dome_master_stereo_dof) );

  // register 360-degree equirectantular projection of spherical camera
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath,
         "vmd_camera_equirectangular", &ray_gen_pgm_equirectangular) );
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath,
         "vmd_camera_equirectangular_dof", &ray_gen_pgm_equirectangular_dof) );
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath,
         "vmd_camera_equirectangular_stereo", &ray_gen_pgm_equirectangular_stereo) );
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath,
         "vmd_camera_equirectangular_stereo_dof", &ray_gen_pgm_equirectangular_stereo_dof) );

  // register Oculus Rift projection
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath,
         "vmd_camera_oculus_rift", &ray_gen_pgm_oculus_rift) );
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath,
         "vmd_camera_oculus_rift_dof", &ray_gen_pgm_oculus_rift_dof) );
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath,
         "vmd_camera_oculus_rift_stereo", &ray_gen_pgm_oculus_rift_stereo) );
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath,
         "vmd_camera_oculus_rift_stereo_dof", &ray_gen_pgm_oculus_rift_stereo_dof) );

  // register perspective camera ray gen programs
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath,
         "vmd_camera_perspective", &ray_gen_pgm_perspective) );
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath,
         "vmd_camera_perspective_dof", &ray_gen_pgm_perspective_dof) );
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath,
         "vmd_camera_perspective_stereo", &ray_gen_pgm_perspective_stereo) );
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath,
         "vmd_camera_perspective_stereo_dof", &ray_gen_pgm_perspective_stereo_dof) );

  // register othographic camera ray gen programs
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath,
         "vmd_camera_orthographic", &ray_gen_pgm_orthographic) );
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath,
         "vmd_camera_orthographic_dof", &ray_gen_pgm_orthographic_dof) );
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath,
         "vmd_camera_orthographic_stereo", &ray_gen_pgm_orthographic_stereo) );
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath,
         "vmd_camera_orthographic_stereo_dof", &ray_gen_pgm_orthographic_stereo_dof) );

  // miss programs for background (solid, gradient sphere/plane)
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath, "miss_gradient_bg_sky_sphere", &miss_pgm_sky_sphere) );
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath, "miss_gradient_bg_sky_plane", &miss_pgm_sky_ortho_plane) );
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath, "miss_solid_bg", &miss_pgm_solid) );

  // exception handler program
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath, "exception", &exception_pgm) );

  double time_cambgops = wkf_timer_timenow(ort_timer);
  if (verbose >= RT_VERB_TIMING) {
    printf("cambgops(%.1f) ", time_cambgops - time_fbops);
    fflush(stdout);
  }

  // cylinder array programs
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath, "cylinder_array_bounds", &cylinder_array_bbox_pgm) );
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath, "cylinder_array_intersect", &cylinder_array_isct_pgm) );

  // color-per-cylinder array programs
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath, "cylinder_array_color_bounds", &cylinder_array_color_bbox_pgm) );
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath, "cylinder_array_color_intersect", &cylinder_array_color_isct_pgm) );

  // color-per-ring array programs
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath, "ring_array_color_bounds", &ring_array_color_bbox_pgm) );
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath, "ring_array_color_intersect", &ring_array_color_isct_pgm) );

  // sphere array programs
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath, "sphere_array_bounds", &sphere_array_bbox_pgm) );
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath, "sphere_array_intersect", &sphere_array_isct_pgm) );

  // color-per-sphere array programs
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath, "sphere_array_color_bounds", &sphere_array_color_bbox_pgm) );
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath, "sphere_array_color_intersect", &sphere_array_color_isct_pgm) );

  // tricolor list programs
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath, "tricolor_bounds", &tricolor_bbox_pgm) );
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath, "tricolor_intersect", &tricolor_isct_pgm) );

  // c4u_n3b_v3f
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath, "trimesh_c4u_n3b_v3f_bounds", &trimesh_c4u_n3b_v3f_bbox_pgm) );
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath, "trimesh_c4u_n3b_v3f_intersect", &trimesh_c4u_n3b_v3f_isct_pgm) );

  // n3f_v3f
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath, "trimesh_n3f_v3f_bounds", &trimesh_n3f_v3f_bbox_pgm) );
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath, "trimesh_n3f_v3f_intersect", &trimesh_n3f_v3f_isct_pgm) );

  // n3b_v3f
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath, "trimesh_n3b_v3f_bounds", &trimesh_n3b_v3f_bbox_pgm) );
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath, "trimesh_n3b_v3f_intersect", &trimesh_n3b_v3f_isct_pgm) );

  // v3f
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath, "trimesh_v3f_bounds", &trimesh_v3f_bbox_pgm) );
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath, "trimesh_v3f_intersect", &trimesh_v3f_isct_pgm) );

  double time_geompgms = wkf_timer_timenow(ort_timer);
  if (verbose >= RT_VERB_TIMING) {
    printf("geompgms(%.1f) ", time_geompgms - time_cambgops);
    fflush(stdout);
  }

  if (verbose >= RT_VERB_TIMING) {
    printf("\n");
  }

  time_ctx_create = wkf_timer_timenow(ort_timer) - starttime;
  
  if (verbose == RT_VERB_TIMING || verbose == RT_VERB_DEBUG) {
    printf("OptiXRenderer) context creation time: %.2f\n", time_ctx_create);
  }

  context_created = 1;
}


void OptiXRenderer::setup_context(int w, int h) {
  double starttime = wkf_timer_timenow(ort_timer);
  time_ctx_setup = 0;

  lasterror = RT_SUCCESS; // clear any error state
  width = w;
  height = h;

  if (!context_created)
    return;

  check_verbose_env(); // update verbose flag if changed since last run

  int maxdepth = 20;
  if (getenv("VMDOPTIXMAXDEPTH")) {
    maxdepth = atoi(getenv("VMDOPTIXMAXDEPTH"));
    if (maxdepth > 0 && maxdepth <= 30) {
      printf("OptiXRenderer) Setting maxdepth to %d...\n", maxdepth);
      RTERR( rtVariableSet1i(max_depth_v, maxdepth) );
    } else {
      printf("OptiXRenderer) ignoring out-of-range maxdepth to %d...\n", maxdepth);
    }
  } 

  int maxtrans = maxdepth;
  if (getenv("VMDOPTIXMAXTRANS")) {
    maxtrans = atoi(getenv("VMDOPTIXMAXTRANS"));
    if (maxtrans > 0 && maxtrans <= 30) {
      printf("OptiXRenderer) Setting maxtrans to %d...\n", maxtrans);
      RTERR( rtVariableSet1i(max_trans_v, maxtrans) );
    } else {
      printf("OptiXRenderer) ignoring out-of-range maxtrans to %d...\n", maxtrans);
    }
  }

  // set maxdepth and maxtrans with new values
  RTERR( rtVariableSet1i(max_depth_v, maxdepth) );
  RTERR( rtVariableSet1i(max_trans_v, maxtrans) );

  // assign indices to ray types
  RTERR( rtVariableSet1ui(radiance_ray_type_v, 0u) );
  RTERR( rtVariableSet1ui(shadow_ray_type_v, 1u) );

  // set default scene epsilon
  float scene_epsilon = 5.e-5f;
  RTERR( rtVariableSet1f(scene_epsilon_v, scene_epsilon) );

  // Current accumulation subframe count, used as part of generating
  // AA and AO random number sequences
  RTERR( rtVariableSet1ui(accum_count_v, 0) );

   // zero out the array of material usage counts for the scene
  memset(material_special_counts, 0, sizeof(material_special_counts));
  time_ctx_setup = wkf_timer_timenow(ort_timer) - starttime;
}


void OptiXRenderer::report_context_stats() {
  if (!context_created)
    return;

  unsigned int ctx_varcount=0;
  RTERR( rtContextGetVariableCount(ctx, &ctx_varcount) );
  printf("OptiXRenderer) ctx var cnt: %u\n", ctx_varcount);
}


void OptiXRenderer::destroy_scene() {
  double starttime = wkf_timer_timenow(ort_timer);
  time_ctx_destroy_scene = 0;

  // zero out all object counters
  cylinder_array_cnt = 0;
  cylinder_array_color_cnt = 0;
  ring_array_color_cnt = 0;
  sphere_array_cnt = 0;
  sphere_array_color_cnt = 0;
  tricolor_cnt = 0;
  trimesh_c4u_n3b_v3f_cnt = 0;
  trimesh_n3b_v3f_cnt = 0;
  trimesh_n3f_v3f_cnt = 0;
  trimesh_v3f_cnt = 0;

  if (!context_created)
    return;

  if (scene_created) {
    int i;

    RTERR( rtAccelerationDestroy(acceleration) );
    RTERR( rtAccelerationDestroy(root_acceleration) );
    RTERR( rtGroupDestroy(root_group) );
    RTERR( rtGeometryGroupDestroy(geometrygroup) );

    int instcount = geominstancelist.num();
    for (i=0; i<instcount; i++) {
      RTERR( rtGeometryInstanceDestroy(geominstancelist[i]) );
    }

    int geomcount = geomlist.num();
    for (i=0; i<geomcount; i++) {
      RTERR( rtGeometryDestroy(geomlist[i]) );
    }

    geominstancelist.clear();
    geomlist.clear();

#if defined(ORT_USERTXAPIS)
    // OptiX RTX hardware-accelerated triangles API
#if 0
    // XXX this crashes the OptiX 5.2 DEV build w/ triangle API 0.3
    //  -- reported to NVIDIA on 8/2/2018.  
    int insttrianglescount = geomtrianglesinstancelist.num();
    for (i=0; i<insttrianglescount; i++) {
      RTERR( rtGeometryInstanceDestroy(geomtrianglesinstancelist[i]) );
    }
#endif

    int geomtrianglescount = geomtriangleslist.num();
    for (i=0; i<geomtrianglescount; i++) {
      RTERR( rtGeometryTrianglesDestroy(geomtriangleslist[i]) );
    }

    geomtrianglesinstancelist.clear();
    geomtriangleslist.clear();
#endif

    int bufcount = bufferlist.num();
    for (i=0; i<bufcount; i++) {
      RTERR( rtBufferDestroy(bufferlist[i]) );
    }

    bufferlist.clear();
  }

  materialcache.clear(); // ensure no materials live across renderings

  double endtime = wkf_timer_timenow(ort_timer);
  time_ctx_destroy_scene = endtime - starttime;

  scene_created = 0; // scene has been destroyed
}


int OptiXRenderer::set_accum_raygen_pgm(CameraProjection &proj, 
                                        int stereo_on, int dof_on) {
  //
  // XXX The ray tracing engine supports a number of camera models that
  //     are extremely difficult to implement effectively in OpenGL,
  //     particularly in the context of interactive rasterization.
  //     The control over use of these camera models is currently implemented
  //     solely through environment variables, which is undesirable, but
  //     necessary in the very short term until we come up with a way of
  //     exposing this in the VMD GUIs.  The environment variables currently
  //     override the incoming projection settings from VMD.
  //


  // VR cubemap
  if (getenv("VMDOPTIXCUBEMAP") != NULL) {
    msgInfo << "Overriding VMD camera projection mode with VR cubemap" << sendmsg;
    proj = RT_CUBEMAP;
  }

  // planetarium dome master
  if (getenv("VMDOPTIXDOMEMASTER") != NULL) {
    msgInfo << "Overriding VMD camera projection mode with planetarium dome master" << sendmsg;
    proj = RT_DOME_MASTER;
  }

  // 360-degree spherical projection into a rectangular (2w x 1h) image
  if (getenv("VMDOPTIXEQUIRECTANGULAR") != NULL) {
    msgInfo << "Overriding VMD camera projection mode with spherical equirectangular projection" << sendmsg;
    proj = RT_EQUIRECTANGULAR;
  }

  // Oculus Rift w/ barrel distortion applied
  if (getenv("VMDOPTIXOCULUSRIFT") != NULL) {
    msgInfo << "Overriding VMD camera projection mode with Oculus Rift projection" << sendmsg;
    proj = RT_OCULUS_RIFT;
  }

  // override stereo if an environment variable is set
  if (getenv("VMDOPTIXSTEREO") != NULL) {
    msgInfo << "Overriding VMD camera, enabling stereo" << sendmsg;
    stereo_on = 1;
  }
    
  // set the active ray gen program based on the active projection mode
  switch (proj) {
    default:
      msgErr << "OptiXRenderer) Illegal projection mode! Using perspective." << sendmsg;
      // XXX fall through to perspective is intentional...

    case RT_PERSPECTIVE:
      if (stereo_on) {
        if (dof_on) {
          RTERR( rtContextSetRayGenerationProgram(ctx, RT_RAY_GEN_ACCUMULATE, ray_gen_pgm_perspective_stereo_dof) );
        } else {
          RTERR( rtContextSetRayGenerationProgram(ctx, RT_RAY_GEN_ACCUMULATE, ray_gen_pgm_perspective_stereo) );
        }
      } else {
        if (dof_on) {
          RTERR( rtContextSetRayGenerationProgram(ctx, RT_RAY_GEN_ACCUMULATE, ray_gen_pgm_perspective_dof) );
        } else {
          RTERR( rtContextSetRayGenerationProgram(ctx, RT_RAY_GEN_ACCUMULATE, ray_gen_pgm_perspective) );
        }
      }
      break;

    case RT_ORTHOGRAPHIC:
      if (stereo_on) {
        if (dof_on) {
          RTERR( rtContextSetRayGenerationProgram(ctx, RT_RAY_GEN_ACCUMULATE, ray_gen_pgm_orthographic_stereo_dof) );
        } else {
          RTERR( rtContextSetRayGenerationProgram(ctx, RT_RAY_GEN_ACCUMULATE, ray_gen_pgm_orthographic_stereo) );
        }
      } else {
        if (dof_on) {
          RTERR( rtContextSetRayGenerationProgram(ctx, RT_RAY_GEN_ACCUMULATE, ray_gen_pgm_orthographic_dof) );
        } else {
          RTERR( rtContextSetRayGenerationProgram(ctx, RT_RAY_GEN_ACCUMULATE, ray_gen_pgm_orthographic) );
        }
      }
      break;

    case RT_CUBEMAP:
      if (stereo_on) {
        if (dof_on) {
          RTERR( rtContextSetRayGenerationProgram(ctx, RT_RAY_GEN_ACCUMULATE, ray_gen_pgm_cubemap_stereo_dof) );
        } else {
          RTERR( rtContextSetRayGenerationProgram(ctx, RT_RAY_GEN_ACCUMULATE, ray_gen_pgm_cubemap_stereo) );
        }
      } else {
        if (dof_on) {
          RTERR( rtContextSetRayGenerationProgram(ctx, RT_RAY_GEN_ACCUMULATE, ray_gen_pgm_cubemap_dof) );
        } else {
          RTERR( rtContextSetRayGenerationProgram(ctx, RT_RAY_GEN_ACCUMULATE, ray_gen_pgm_cubemap) );
        }
      }
      break;

    case RT_DOME_MASTER:
      if (stereo_on) {
        if (dof_on) {
          RTERR( rtContextSetRayGenerationProgram(ctx, RT_RAY_GEN_ACCUMULATE, ray_gen_pgm_dome_master_stereo_dof) );
        } else {
          RTERR( rtContextSetRayGenerationProgram(ctx, RT_RAY_GEN_ACCUMULATE, ray_gen_pgm_dome_master_stereo) );
        }
      } else {
        if (dof_on) {
          RTERR( rtContextSetRayGenerationProgram(ctx, RT_RAY_GEN_ACCUMULATE, ray_gen_pgm_dome_master_dof) );
        } else {
          RTERR( rtContextSetRayGenerationProgram(ctx, RT_RAY_GEN_ACCUMULATE, ray_gen_pgm_dome_master) );
        }
      }
      break;

    case RT_EQUIRECTANGULAR:
      if (stereo_on) {
        if (dof_on) {
          RTERR( rtContextSetRayGenerationProgram(ctx, RT_RAY_GEN_ACCUMULATE, ray_gen_pgm_equirectangular_stereo_dof) );
        } else {
          RTERR( rtContextSetRayGenerationProgram(ctx, RT_RAY_GEN_ACCUMULATE, ray_gen_pgm_equirectangular_stereo) );
        }
      } else {
        if (dof_on) {
          RTERR( rtContextSetRayGenerationProgram(ctx, RT_RAY_GEN_ACCUMULATE, ray_gen_pgm_equirectangular_dof) );
        } else {
          RTERR( rtContextSetRayGenerationProgram(ctx, RT_RAY_GEN_ACCUMULATE, ray_gen_pgm_equirectangular) );
        }
      }
      break;

    case RT_OCULUS_RIFT:
      if (stereo_on) {
        if (dof_on) {
          RTERR( rtContextSetRayGenerationProgram(ctx, RT_RAY_GEN_ACCUMULATE, ray_gen_pgm_oculus_rift_stereo_dof) );
        } else {
          RTERR( rtContextSetRayGenerationProgram(ctx, RT_RAY_GEN_ACCUMULATE, ray_gen_pgm_oculus_rift_stereo) );
        }
      } else {
        if (dof_on) {
          RTERR( rtContextSetRayGenerationProgram(ctx, RT_RAY_GEN_ACCUMULATE, ray_gen_pgm_oculus_rift_dof) );
        } else {
          RTERR( rtContextSetRayGenerationProgram(ctx, RT_RAY_GEN_ACCUMULATE, ray_gen_pgm_oculus_rift) );
        }
      }
      break;
  }

  return 0;
}


void OptiXRenderer::update_rendering_state(int interactive) {
  if (!context_created)
    return;

#if defined(ORT_USERTXAPIS)
  // permit the hardware triangle API to be disabled at runtime
  if (getenv("VMDOPTIXNOHWTRIANGLES") != NULL) {
    hwtri_enabled = 0;
  }
#endif

  // update scene epsilon if necessary
  if (getenv("VMDOPTIXSCENEEPSILON") != NULL) {
    float scene_epsilon = atof(getenv("VMDOPTIXSCENEEPSILON"));
    printf("OptiXRenderer) user override of scene epsilon: %g\n", scene_epsilon);
    RTERR( rtVariableSet1f(scene_epsilon_v, scene_epsilon) );
  }

  int i;
  wkf_timer_start(ort_timer);

  // set interactive/progressive rendering flag
  RTERR( rtVariableSet1i(progressive_enabled_v, interactive) );

  long totaltris = tricolor_cnt + trimesh_c4u_n3b_v3f_cnt + 
                   trimesh_n3b_v3f_cnt + trimesh_n3f_v3f_cnt + trimesh_v3f_cnt;

  if (verbose == RT_VERB_TIMING || verbose == RT_VERB_DEBUG) {
    printf("OptiXRenderer) cyl %ld, ring %ld, sph %ld, tri %ld, tot: %ld  lt %ld\n",
           cylinder_array_cnt + cylinder_array_color_cnt,
           ring_array_color_cnt,
           sphere_array_cnt + sphere_array_color_cnt,
           totaltris,
           cylinder_array_cnt +  cylinder_array_color_cnt + ring_array_color_cnt + sphere_array_cnt + sphere_array_color_cnt + totaltris,
           directional_lights.num() + positional_lights.num());
  }

  if (verbose == RT_VERB_DEBUG) {
#if defined(ORT_USE_TEMPLATE_SHADERS)
    if (getenv("VMDOPTIXFORCEGENERALSHADER") == NULL) {
      printf("OptiXRenderer) using template-specialized shaders and materials:\n");
      int i;
      for (i=0; i<ORTMTABSZ; i++) {
        if (material_special_counts[i] > 0) {
          printf("OptiXRenderer) material_special[%d] usage count: %d\n", 
                 i, material_special_counts[i]); 
    
          printf("OptiXRenderer)   "
                 "ClipView %s, "
                 "Headlight %s, "
                 "Fog %s, "
                 "Shadows %s, "
                 "AO %s, "
                 "Outline %s, "
                 "Refl %s, "
                 "Trans %s\n",
#if defined(VMDOPTIX_VCA_TABSZHACK)
                 onoffstr(1),
                 onoffstr(1),
#else
                 onoffstr(i & 128),
                 onoffstr(i &  64),
#endif
                 onoffstr(i &  32),
                 onoffstr(i &  16),
                 onoffstr(i &   8),
                 onoffstr(i &   4),
                 onoffstr(i &   2),
                 onoffstr(i &   1));
        }
      }
      printf("OptiXRenderer)\n");
    } else {
      printf("OptiXRenderer) using fully general shader and materials.\n");
    }
#else
    printf("OptiXRenderer) using fully general shader and materials.\n");
#endif
  }

  RTERR( rtVariableSet3fv(scene_bg_color_v, scene_bg_color) );
  RTERR( rtVariableSet3fv(scene_bg_grad_top_v, scene_bg_grad_top) );
  RTERR( rtVariableSet3fv(scene_bg_grad_bot_v, scene_bg_grad_bot) );
  RTERR( rtVariableSet3fv(scene_gradient_v, scene_gradient) );
  RTERR( rtVariableSet1f(scene_gradient_topval_v, scene_gradient_topval) );
  RTERR( rtVariableSet1f(scene_gradient_botval_v, scene_gradient_botval) );

  if (verbose == RT_VERB_DEBUG) {
    printf("OptiXRenderer) HMD/camera view clipping mode: %d start: %.2f end: %.2f\n",
           clipview_mode, clipview_start, clipview_end);

    printf("OptiXRenderer) HMD/camera headlight mode: %d\n", headlight_mode);

    printf("OptiXRenderer) scene bg mode: %d\n", scene_background_mode);

    printf("OptiXRenderer) scene bgsolid: %.2f %.2f %.2f\n", 
           scene_bg_color[0], scene_bg_color[1], scene_bg_color[2]);

    printf("OptiXRenderer) scene bggradT: %.2f %.2f %.2f\n", 
           scene_bg_grad_top[0], scene_bg_grad_top[1], scene_bg_grad_top[2]);

    printf("OptiXRenderer) scene bggradB: %.2f %.2f %.2f\n", 
           scene_bg_grad_bot[0], scene_bg_grad_bot[1], scene_bg_grad_bot[2]);
  
    printf("OptiXRenderer) bg gradient: %f %f %f  top: %f  bot: %f\n",
           scene_gradient[0], scene_gradient[1], scene_gradient[2],
           scene_gradient_topval, scene_gradient_botval);
  }

  // update in case the caller changed top/bottom values since last recalc
  scene_gradient_invrange = 1.0f / (scene_gradient_topval - scene_gradient_botval);
  RTERR( rtVariableSet1f(scene_gradient_invrange_v, scene_gradient_invrange) );

  RTERR( rtVariableSet1i(clipview_mode_v, clipview_mode) );
  RTERR( rtVariableSet1f(clipview_start_v, clipview_start) );
  RTERR( rtVariableSet1f(clipview_end_v, clipview_end) );
  RTERR( rtVariableSet1i(headlight_mode_v, (int) headlight_mode) );

  RTERR( rtVariableSet1i(fog_mode_v, (int) fog_mode) );
  RTERR( rtVariableSet1f(fog_start_v, fog_start) );
  RTERR( rtVariableSet1f(fog_end_v, fog_end) );
  RTERR( rtVariableSet1f(fog_density_v, fog_density) );

  if (verbose == RT_VERB_DEBUG) {
    printf("OptiXRenderer) adding lights: dir: %ld  pos: %ld\n", 
           directional_lights.num(), positional_lights.num());
  }

#if defined(VMDOPTIX_LIGHTUSEROBJS)
  DirectionalLightList dir_lights;
  memset(&dir_lights, 0, sizeof(DirectionalLightList));
  dir_lights.num_lights = directional_lights.num();
  int dlcount = directional_lights.num();
  dlcount = (dlcount > DISP_LIGHTS) ? DISP_LIGHTS : dlcount;
  for (i=0; i<dlcount; i++) {
    vec_copy((float*)(&dir_lights.dirs[i]), directional_lights[i].dir);
    vec_normalize((float*)&dir_lights.dirs[i]);
  }
  RTERR( rtVariableSetUserData(dir_light_list_v, sizeof(DirectionalLightList), &dir_lights) );

  PositionalLightList pos_lights;
  memset(&pos_lights, 0, sizeof(PositionalLightList));
  pos_lights.num_lights = positional_lights.num();
  int plcount = positional_lights.num();
  plcount = (plcount > DISP_LIGHTS) ? DISP_LIGHTS : plcount;
  for (i=0; i<plcount; i++) {
    vec_copy((float*)(&pos_lights.posns[i]), positional_lights[i].pos);
  }
  RTERR( rtVariableSetUserData(pos_light_list_v, sizeof(PositionalLightList), &pos_lights) );
#else
  DirectionalLight *dlbuf;
  RTERR( rtBufferSetSize1D(dir_lightbuffer, directional_lights.num()) );
  RTERR( rtBufferMap(dir_lightbuffer, (void **) &dlbuf) );
  for (i=0; i<directional_lights.num(); i++) {
    vec_copy((float*)&dlbuf[i].dir, directional_lights[i].dir);
    vec_normalize((float*)&dlbuf[i].dir);
  }
  RTERR( rtBufferUnmap(dir_lightbuffer) );
  RTERR( rtVariableSetObject(dir_lightbuffer_v, dir_lightbuffer) );

  PositionalLight *plbuf;
  RTERR( rtBufferSetSize1D(pos_lightbuffer, positional_lights.num()) );
  RTERR( rtBufferMap(pos_lightbuffer, (void **) &plbuf) );
  for (i=0; i<positional_lights.num(); i++) {
    vec_copy((float*)&plbuf[i].pos, positional_lights[i].pos);
  }
  RTERR( rtBufferUnmap(pos_lightbuffer) );
  RTERR( rtVariableSetObject(pos_lightbuffer_v, pos_lightbuffer) );
#endif

  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) Finalizing OptiX scene graph...\n");

  // create group to hold instances
  int instcount = geominstancelist.num();
  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) instance objects: %d\n", instcount);

  RTERR( rtGeometryGroupCreate(ctx, &geometrygroup) );
  RTERR( rtGeometryGroupSetChildCount(geometrygroup, instcount) );
  for (i=0; i<instcount; i++) {
    RTERR( rtGeometryGroupSetChild(geometrygroup, i, geominstancelist[i]) );
  }

#if defined(ORT_USERTXAPIS)
  // OptiX RTX hardware-accelerated triangles API
  // create separate group for triangle instances
  int insttrianglescount = geomtrianglesinstancelist.num();
  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) triangle instance objects: %d\n", insttrianglescount);

  RTERR( rtGeometryGroupCreate(ctx, &geometrytrianglesgroup) );
  RTERR( rtGeometryGroupSetChildCount(geometrytrianglesgroup, insttrianglescount) );
  for (i=0; i<insttrianglescount; i++) {
    RTERR( rtGeometryGroupSetChild(geometrytrianglesgroup, i, geomtrianglesinstancelist[i]) );
  }
#endif


  // XXX we should create an acceleration object the instance shared
  //     by multiple PBC images

  // acceleration object for the geometrygroup
  RTERR( rtAccelerationCreate(ctx, &acceleration) );

  // Allow runtime override of acceleration builder and traverser
  // for performance testing/tuning
  const char *ort_builder   = getenv("VMDOPTIXBUILDER");
  const char *ort_traverser = getenv("VMDOPTIXTRAVERSER");
  if (ort_builder && ort_traverser) {
    RTERR( rtAccelerationSetBuilder(acceleration, ort_builder) );
    RTERR( rtAccelerationSetTraverser(acceleration, ort_traverser) );
    if (verbose == RT_VERB_DEBUG) {
      printf("OptiXRenderer) user-override of AS: builder: '%s' traverser '%s'\n",
             ort_builder, ort_traverser);
    }
  } else if (ort_builder) {
    RTERR( rtAccelerationSetBuilder(acceleration, ort_builder) );
    RTERR( rtAccelerationSetTraverser(acceleration, "Bvh") );
    if (verbose == RT_VERB_DEBUG) {
      printf("OptiXRenderer) user-override of AS builder: '%s' (def traverser '%s')\n",
             ort_builder, "Bvh");
    }
  } else {
#if (OPTIX_VERSION >= 3050) && (OPTIX_VERSION < 3060) || (OPTIX_VERSION == 3063) || (OPTIX_VERSION == 3080)
    // OptiX 3.5.0 was the first to include the new fast "Trbvh" AS builder
    // OptiX 3.6.3 fixed Trbvh bugs on huge models
    // OptiX 3.8.0 has cured all known Trbvh bugs for VMD so far
    RTERR( rtAccelerationSetBuilder(acceleration, "Trbvh") );
    RTERR( rtAccelerationSetTraverser(acceleration, "Bvh") );
#else
    // For older revs of OptiX (or those with bugs in Trbvh), 
    // the "MedianBvh" AS builder gives the best compromise between 
    // builder speed and ray tracing speed.
    // OptiX 3.6.[012] and 3.7.0 had Trbvh bugs on huge models that 
    // could cause VMD to crash in some cases
//    RTERR( rtAccelerationSetBuilder(acceleration, "Sbvh") );
//    RTERR( rtAccelerationSetBuilder(acceleration, "Bvh") );
    RTERR( rtAccelerationSetBuilder(acceleration, "MedianBvh") );
    RTERR( rtAccelerationSetTraverser(acceleration, "Bvh") );
#endif
  }

 
  // allow user-override of the builder type (e.g. "GPU", or "CPU") when
  // the AS builder provides more than one choice.
  if (getenv("VMDOPTIXBUILDTYPE") != NULL) {
    const char *buildtypestr = getenv("VMDOPTIXBUILDTYPE");
    const char *curbuilderstr = NULL;
    RTERR( rtAccelerationGetBuilder(acceleration, &curbuilderstr) );
    if (!strcmp(curbuilderstr, "Trbvh")) {
      msgInfo << "OptiXRenderer) user-override of Trbvh AS build type: " 
              << buildtypestr << sendmsg;
      RTERR( rtAccelerationSetProperty(acceleration, "build_type", buildtypestr) );
    } else {
      msgErr << "OptiXRenderer) Can't set build type for AS builders other than Trbvh" << sendmsg; 
    }
  } 


  RTERR( rtGeometryGroupSetAcceleration(geometrygroup, acceleration) );
  RTERR( rtAccelerationMarkDirty(acceleration) );

#if defined(ORT_USERTXAPIS)
  // OptiX RTX hardware-accelerated triangles API
  // Acceleration structure for triangle geometry
  RTERR( rtAccelerationCreate(ctx, &trianglesacceleration) );
  RTERR( rtAccelerationSetBuilder(trianglesacceleration, "Trbvh") );
  RTERR( rtAccelerationSetTraverser(trianglesacceleration, "Bvh") );
  RTERR( rtGeometryGroupSetAcceleration(geometrytrianglesgroup, trianglesacceleration) );
  RTERR( rtAccelerationMarkDirty(trianglesacceleration) );
#endif


  // create the root node of the scene graph
  RTERR( rtGroupCreate(ctx, &root_group) );
#if defined(ORT_USERTXAPIS)
  RTERR( rtGroupSetChildCount(root_group, 2) );
  RTERR( rtGroupSetChild(root_group, 0, geometrygroup) );
  RTERR( rtGroupSetChild(root_group, 1, geometrytrianglesgroup) );
#else
  RTERR( rtGroupSetChildCount(root_group, 1) );
  RTERR( rtGroupSetChild(root_group, 0, geometrygroup) );
#endif
  RTERR( rtVariableSetObject(root_object_v, root_group) );
  RTERR( rtVariableSetObject(root_shadower_v, root_group) );

  // create an acceleration object for the entire scene graph
  RTERR( rtAccelerationCreate(ctx, &root_acceleration) );
  RTERR( rtAccelerationSetBuilder(root_acceleration,"NoAccel") );
  RTERR( rtAccelerationSetTraverser(root_acceleration,"NoAccel") );
  RTERR( rtGroupSetAcceleration(root_group, root_acceleration) );
  RTERR( rtAccelerationMarkDirty(root_acceleration) );
  scene_created=1;


  // do final state variable updates before rendering begins
  if (verbose == RT_VERB_DEBUG) {
    printf("OptiXRenderer) cam zoom factor %f\n", cam_zoom);
    printf("OptiXRenderer) cam stereo eye separation  %f\n", cam_stereo_eyesep);
    printf("OptiXRenderer) cam stereo convergence distance %f\n", 
           cam_stereo_convergence_dist);
    printf("OptiXRenderer) cam DoF focal distance %f\n", cam_dof_focal_dist);
    printf("OptiXRenderer) cam DoF f/stop %f\n", cam_dof_fnumber);
  }

  // define all of the standard camera params
  RTERR( rtVariableSet1f(cam_zoom_v,  cam_zoom) );
  RTERR( rtVariableSet3f( cam_pos_v,  0.0f,  0.0f,  2.0f) );
  RTERR( rtVariableSet3f(   cam_U_v,  1.0f,  0.0f,  0.0f) );
  RTERR( rtVariableSet3f(   cam_V_v,  0.0f,  1.0f,  0.0f) );
  RTERR( rtVariableSet3f(   cam_W_v,  0.0f,  0.0f, -1.0f) );

  // define stereoscopic camera parameters
  RTERR( rtVariableSet1f(cam_stereo_eyesep_v, cam_stereo_eyesep) );
  RTERR( rtVariableSet1f(cam_stereo_convergence_dist_v, cam_stereo_convergence_dist) );

  // define camera DoF parameters
  RTERR( rtVariableSet1f(cam_dof_focal_dist_v, cam_dof_focal_dist) );
  RTERR( rtVariableSet1f(cam_dof_aperture_rad_v, cam_dof_focal_dist / (2.0f * cam_zoom * cam_dof_fnumber)) );

  // for batch mode rendering, we prefer correctness to speed, so 
  // we currently ignore USE_REVERSE_SHADOW_RAYS_DEFAULT except when
  // running interactively.  When the reverse ray optimizatoin is 100%
  // bulletproof, we will use it for batch rendering also.
  RTERR( rtVariableSet1i(shadows_enabled_v, 
                         (shadows_enabled) ? RT_SHADOWS_ON : RT_SHADOWS_OFF) );

  RTERR( rtVariableSet1i(ao_samples_v, ao_samples) );
  RTERR( rtVariableSet1f(ao_ambient_v, ao_ambient) );
  RTERR( rtVariableSet1f(ao_direct_v, ao_direct) );
  RTERR( rtVariableSet1f(ao_maxdist_v, ao_maxdist) );
  if (getenv("VMDOPTIXAOMAXDIST")) {
    float tmp = atof(getenv("VMDOPTIXAOMAXDIST"));
    if (verbose == RT_VERB_DEBUG) {
      printf("OptiXRenderer) setting AO maxdist: %f\n", tmp);
    }
    RTERR( rtVariableSet1f(ao_maxdist_v, tmp) );
  }

  if (verbose == RT_VERB_DEBUG) {
    printf("OptiXRenderer) setting sample counts:  AA %d  AO %d\n", aa_samples, ao_samples);
    printf("OptiXRenderer) setting AO factors:  AOA %f  AOD %f\n", ao_ambient, ao_direct);
  }

  //
  // Handle AA samples either internally with loops internal to 
  // each ray launch point thread, or externally by iterating over
  // multiple launches, adding each sample to an accumulation buffer,
  // or a hybrid combination of the two.  The final framebuffer output
  // is written by launching a special accumulation buffer drawing 
  // program that range clamps and converts the pixel data while copying
  // the GPU-local accumulation buffer to the final output buffer...
  //
  ext_aa_loops = 1;
  if (ao_samples > 0 || (aa_samples > 4)) {
    // if we have too much work for a single-pass rendering, we need to 
    // break it up into multiple passes or we risk having kernel timeouts
    ext_aa_loops = 1 + aa_samples;
    RTERR( rtVariableSet1i(aa_samples_v, 1) );
  } else { 
    // if the scene is simple, e.g. no AO rays and AA sample count is small,
    // we can run it in a single pass and get better performance
    RTERR( rtVariableSet1i(aa_samples_v, aa_samples + 1) );
  }
  RTERR( rtVariableSet1f(accum_norm_v, 1.0f / float(1 + aa_samples)) );

  if (verbose == RT_VERB_DEBUG) {
    if (ext_aa_loops > 1)
      printf("OptiXRenderer) Running OptiX multi-pass: %d loops\n", ext_aa_loops);
    else
      printf("OptiXRenderer) Running OptiX single-pass: %d total samples\n", 1+aa_samples);
  }

  // set the ray generation program to the active camera code...
  RTERR( rtContextSetEntryPointCount(ctx, RT_RAY_GEN_COUNT) );
  RTERR( rtContextSetRayGenerationProgram(ctx, RT_RAY_GEN_CLEAR_ACCUMULATION_BUFFER, clear_accumulation_buffer_pgm) );
#if defined(ORT_RAYSTATS)
  RTERR( rtContextSetRayGenerationProgram(ctx, RT_RAY_GEN_CLEAR_RAYSTATS, clear_raystats_buffers_pgm) );
#endif

  // set the active color accumulation ray gen program based on the 
  // camera/projection mode, stereoscopic display mode, 
  // and depth-of-field state
  set_accum_raygen_pgm(camera_projection, 0, dof_enabled);

  //
  // set the ray gen program to use for the copy/finish operations
  //
#if defined(VMDOPTIX_PROGRESSIVEAPI)
  if (interactive) {
    RTERR( rtContextSetRayGenerationProgram(ctx, RT_RAY_GEN_COPY_FINISH, draw_accumulation_buffer_stub_pgm) );
  } else
#endif
  {
    RTERR( rtContextSetRayGenerationProgram(ctx, RT_RAY_GEN_COPY_FINISH, draw_accumulation_buffer_pgm) );
  }

  // Link up miss program depending on background rendering mode
  switch (scene_background_mode) {
    case RT_BACKGROUND_TEXTURE_SKY_SPHERE:
      RTERR( rtContextSetMissProgram(ctx, RT_RAY_TYPE_RADIANCE, miss_pgm_sky_sphere) );
      break;

    case RT_BACKGROUND_TEXTURE_SKY_ORTHO_PLANE:
      RTERR( rtContextSetMissProgram(ctx, RT_RAY_TYPE_RADIANCE, miss_pgm_sky_ortho_plane) );
      break;

    case RT_BACKGROUND_TEXTURE_SOLID:
    default:
      RTERR( rtContextSetMissProgram(ctx, RT_RAY_TYPE_RADIANCE, miss_pgm_solid) );
      break;
  }

  // enable exception handling for all defined entry points
  unsigned int epcnt=0;
  RTERR( rtContextGetEntryPointCount(ctx, &epcnt) );
  unsigned int epidx;
  for (epidx=0; epidx<epcnt; epidx++) { 
    RTERR( rtContextSetExceptionProgram(ctx, epidx, exception_pgm) );
  }

  // enable all exceptions for debugging if requested
  if (getenv("VMDOPTIXDEBUG")) {
    printf("OptiXRenderer) Enabling all OptiX exceptions\n");
    RTERR( rtContextSetExceptionEnabled(ctx, RT_EXCEPTION_ALL, 1) );
  }

  // increase default OptiX stack size to prevent runtime failures
  RTsize ssz;
  rtContextGetStackSize(ctx, &ssz);
  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) original stack size: %ld\n", ssz);

  // a decent default stack size is 7KB
  long newstacksize = 7 * 1024;

  // allow runtime user override of the OptiX stack size in 
  // case we need to render a truly massive scene
  if (getenv("VMDOPTIXSTACKSIZE")) {
    newstacksize = atoi(getenv("VMDOPTIXSTACKSIZE"));
    if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) user stack size override: %ld\n", newstacksize);
  }
  rtContextSetStackSize(ctx, newstacksize);
  rtContextGetStackSize(ctx, &ssz);
  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) new stack size: %ld\n", ssz);
#if !defined(VMDOPTIX_PROGRESSIVEAPI)
  rtContextSetPrintEnabled(ctx, 1);
  rtContextSetPrintBufferSize(ctx, 1*1024*1024); 
#endif

#if defined(VMD_ENABLE_OPTIX_TIMEOUTS)
  // Add a custom OptiX timeout callback to see if we can overcome
  // some of the timeout issues we've had previously
  double timeoutlimit = 0.5;
  const char *tmstr = getenv("VMDOPTIXTIMEOUTLIMIT");
  if (tmstr) {
    timeoutlimit = atof(tmstr);
    printf("Setting OptiX timeout: %f sec\n", timeoutlimit);
  }

  if (verbose == RT_VERB_DEBUG)
    printf("Setting OptiX timeout: %f sec\n", timeoutlimit);
  
  RTERR( rtContextSetTimeoutCallback(ctx, vmd_timeout_cb, timeoutlimit) );
#endif
}


void OptiXRenderer::config_framebuffer(int fbwidth, int fbheight,
                                       int interactive) {
  if (!context_created)
    return;

#ifdef VMDOPTIX_PROGRESSIVEAPI
  // If VMD is using the progressive APIs, we have to check that
  // the requested framebuffer config matches the existing one in 
  // terms of bindings for streaming output, otherwise we have to 
  // destroy and re-create the framebuffer and any needed streaming
  // bindings.
  if (buffers_progressive != (interactive != 0)) {
    if (verbose == RT_VERB_DEBUG) {
      printf("OptiXRenderer) switching between progressive/non-progressive mode\n");
      printf("OptiXRenderer) remaking framebuffer\n");
    }
    destroy_framebuffer();
  }
#endif

  // allocate and resize buffers to match request
  if (buffers_allocated) {
    // if the buffers already exist and match the current 
    // progressive/non-progressive rendering mode, just resize them
    if (verbose == RT_VERB_DEBUG) {
      printf("OptiXRenderer) resizing framebuffer\n");
    }
    resize_framebuffer(fbwidth, fbheight);
  } else {
    // (re)allocate framebuffer and associated accumulation buffers if they
    // don't already exist or if they weren't bound properly for
    // current progressive/non-progressive rendering needs.
    if (verbose == RT_VERB_DEBUG) {
      printf("OptiXRenderer) creating framebuffer and accum. buffer\n");
    }

    // create intermediate GPU-local accumulation buffer
    RTERR( rtContextDeclareVariable(ctx, "accumulation_buffer", &accumulation_buffer_v) );

#ifdef VMDOPTIX_PROGRESSIVEAPI
    if (interactive) {
      RTERR( rtBufferCreate(ctx, RT_BUFFER_OUTPUT, &accumulation_buffer) );
      buffers_progressive = 1;
    } else 
#endif
    {
      RTERR( rtBufferCreate(ctx, RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, &accumulation_buffer) );
      buffers_progressive = 0;
    }

    RTERR( rtBufferSetFormat(accumulation_buffer, RT_FORMAT_FLOAT4) );
    RTERR( rtBufferSetSize2D(accumulation_buffer, fbwidth, fbheight) );
    RTERR( rtVariableSetObject(accumulation_buffer_v, accumulation_buffer) );

#if defined(ORT_RAYSTATS)
    // (re)create intermediate GPU-local ray stats buffers
    // the ray stat buffers get cleared when clearing the accumulation buffer
    RTERR( rtContextDeclareVariable(ctx, "raystats1_buffer", &raystats1_buffer_v) );
    RTERR( rtBufferCreate(ctx, RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, &raystats1_buffer) );
    RTERR( rtBufferSetFormat(raystats1_buffer, RT_FORMAT_UNSIGNED_INT4) );
    RTERR( rtBufferSetSize2D(raystats1_buffer, fbwidth, fbheight) );
    RTERR( rtVariableSetObject(raystats1_buffer_v, raystats1_buffer) );

    RTERR( rtContextDeclareVariable(ctx, "raystats2_buffer", &raystats2_buffer_v) );
    RTERR( rtBufferCreate(ctx, RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, &raystats2_buffer) );
    RTERR( rtBufferSetFormat(raystats2_buffer, RT_FORMAT_UNSIGNED_INT4) );
    RTERR( rtBufferSetSize2D(raystats2_buffer, fbwidth, fbheight) );
    RTERR( rtVariableSetObject(raystats2_buffer_v, raystats2_buffer) );
#endif

    // create output framebuffer
#ifdef VMDOPTIX_PROGRESSIVEAPI
    if (interactive) {
      RTERR( rtBufferCreate(ctx, RT_BUFFER_PROGRESSIVE_STREAM, &framebuffer) );

      // allow user-override of default 5 Mbit/s video encoding bit rate
      int stream_bitrate = 5000000;
      if (getenv("VMDOPTIXBITRATE"))
        stream_bitrate = atoi(getenv("VMDOPTIXBITRATE"));
      RTERR( rtBufferSetAttribute(framebuffer, RT_BUFFER_ATTRIBUTE_STREAM_BITRATE, sizeof(int), &stream_bitrate) );

      // allow user-override of default 30 FPS target frame rate
      int stream_fps = 30;
      if (getenv("VMDOPTIXFPS"))
        stream_fps = atoi(getenv("VMDOPTIXFPS"));
      RTERR( rtBufferSetAttribute(framebuffer, RT_BUFFER_ATTRIBUTE_STREAM_FPS, sizeof(int), &stream_fps) );
            
      // allow user-override of Gamma
      float stream_gamma = 1.0f;
      if (getenv("VMDOPTIXGAMMS"))
        stream_gamma = atoi(getenv("VMDOPTIXGAMMA"));
      RTERR( rtBufferSetAttribute(framebuffer, RT_BUFFER_ATTRIBUTE_STREAM_GAMMA, sizeof(float), &stream_gamma) );
    } else
#endif
    {
      RTERR( rtBufferCreate(ctx, RT_BUFFER_OUTPUT, &framebuffer) );
    }

    RTERR( rtBufferSetFormat(framebuffer, RT_FORMAT_UNSIGNED_BYTE4) );
    RTERR( rtBufferSetSize2D(framebuffer, fbwidth, fbheight) );

#ifdef VMDOPTIX_PROGRESSIVEAPI
    if (interactive) {
      RTERR( rtBufferBindProgressiveStream( framebuffer, accumulation_buffer) );
    } else
#endif 
    {
      RTERR( rtContextDeclareVariable(ctx, "framebuffer", &framebuffer_v) );
      RTERR( rtVariableSetObject(framebuffer_v, framebuffer) );
    }

    buffers_allocated = 1;
  }
}


void OptiXRenderer::resize_framebuffer(int fbwidth, int fbheight) {
  if (!context_created)
    return;

  if (buffers_allocated) {
    if (verbose == RT_VERB_DEBUG) 
      printf("OptiXRenderer) resize_framebuffer(%d x %d)\n", fbwidth, fbheight);

    RTERR( rtBufferSetSize2D(framebuffer, width, height) );
    RTERR( rtBufferSetSize2D(accumulation_buffer, width, height) );
#if defined(ORT_RAYSTATS)
    RTERR( rtBufferSetSize2D(raystats1_buffer, width, height) );
    RTERR( rtBufferSetSize2D(raystats2_buffer, width, height) );
#endif
  }
}


void OptiXRenderer::destroy_framebuffer() {
  if (!context_created)
    return;

  if (buffers_allocated) {
#if defined(ORT_RAYSTATS)
    RTERR( rtContextRemoveVariable(ctx, raystats1_buffer_v) );
    RTERR( rtBufferDestroy(raystats1_buffer) );
    RTERR( rtContextRemoveVariable(ctx, raystats2_buffer_v) );
    RTERR( rtBufferDestroy(raystats2_buffer) );
#endif
    RTERR( rtContextRemoveVariable(ctx, accumulation_buffer_v) );
    RTERR( rtBufferDestroy(accumulation_buffer) );
#ifndef VMDOPTIX_PROGRESSIVEAPI
    RTERR( rtContextRemoveVariable(ctx, framebuffer_v) );
#endif
    RTERR( rtBufferDestroy(framebuffer) );
  }
  buffers_allocated = 0;
  buffers_progressive = 0;
}


void OptiXRenderer::render_compile_and_validate(void) {
  if (!context_created)
    return;

  //
  // finalize context validation, compilation, and AS generation 
  //
  double startctxtime = wkf_timer_timenow(ort_timer);


  // 
  // XXX need to add heuristics to prevent out-of-memory AS build failures on 
  //     versions of OptiX that don't allow multiple AS build attempts
  //
  // The memory usage of Trbvh is roughly (from Keith's email):
  //   68Byte * num_tris * 1.3 + 0.1*total GPU memory + user prim buffers.
  //
  // VMD should use this as an estimate of peak Trbvh memory demand and
  // force a switch from the GPU builder to a CPU builder, or if it's 
  // still too much.  Trbvh built on CPU still uses 2x the memory that
  // the geometry does, so there will still be cases where the only
  // viable route is to switch away from Trbvh entirely, e.g. use MedianBvh.
#define ORT_AS_BUILD_MEM_HEURISTIC 1
#if defined(ORT_AS_BUILD_MEM_HEURISTIC)
  if (getenv("VMDOPTIXNOMEMHEURISTIC") == NULL) {
    const char *curbuilderstr = NULL;
    RTERR( rtAccelerationGetBuilder(acceleration, &curbuilderstr) );
    if (!strcmp(curbuilderstr, "Trbvh")) {
      long totaltris = tricolor_cnt + trimesh_c4u_n3b_v3f_cnt +
                       trimesh_n3b_v3f_cnt + trimesh_n3f_v3f_cnt + 
                       trimesh_v3f_cnt;

      long totalobjs = cylinder_array_cnt + cylinder_array_color_cnt +
                       ring_array_color_cnt +
                       sphere_array_cnt + sphere_array_color_cnt +
                       totaltris;

      long totaluserbufsz = 
          cylinder_array_cnt * sizeof(vmd_cylinder) +
          cylinder_array_color_cnt * sizeof(vmd_cylinder_color) + 
          ring_array_color_cnt * sizeof(vmd_ring_color) +
          sphere_array_cnt * sizeof(vmd_sphere) + 
          sphere_array_color_cnt * sizeof(vmd_sphere_color) + 
          tricolor_cnt * sizeof(vmd_tricolor) + 
          trimesh_c4u_n3b_v3f_cnt * sizeof(vmd_trimesh_c4u_n3b_v3f) +
          trimesh_n3b_v3f_cnt * sizeof(vmd_trimesh_n3b_v3f) + 
          trimesh_n3f_v3f_cnt * sizeof(vmd_trimesh_n3f_v3f) + 
          trimesh_v3f_cnt * sizeof(vmd_trimesh_v3f);

      // Query the current state of all GPUs in the OptiX context
      // to determine the smallest amount of free memory, and 
      // the smallest amount of physical memory among all GPUs.
      unsigned long mingpufreemem, mingpuphysmem;
      if (query_meminfo_ctx_devices(ctx, mingpufreemem, mingpuphysmem)) {
        // If the GPU hardware query fails for some reason, we blindly
        // assume we've got a mostly vacant K20-like GPU with 4GB free 
        mingpufreemem = 4L * 1024L * 1024L * 1024L;
        mingpuphysmem = 6L * 1024L * 1024L * 1024L;
      }
      unsigned long tenpctgpumem = mingpuphysmem / 10;

      // 1.3 * 68 bytes == ~88 bytes
      unsigned long trbvhmemsz = totalobjs * 90L + tenpctgpumem;
      unsigned long totaltrbvhmemsz = trbvhmemsz + totaluserbufsz;
      unsigned long totaltrbvhmemszmb = totaltrbvhmemsz / (1024L * 1024L);

      if (totaltrbvhmemsz > mingpufreemem) {
        // issue warning, and try to build the AS using a different builder
        msgWarn << "OptiXRenderer) Predicted Trbvh AS peak GPU memory requirement exceeds capacity" << sendmsg;
        msgWarn << "OptiXRenderer) Min free GPU mem: " << (mingpufreemem / (1024L * 1024L)) << "MB" << sendmsg;
        msgWarn << "OptiXRenderer) Predicted Trbvh AS peak mem during build: " << totaltrbvhmemszmb << "MB" << sendmsg;

#if 1
        // XXX the Trbvh CPU builder can segfault in some cases, e.g. on
        //     a 171M triangle mesh on a machine w/ 256GB physical mem.
        //     for now, we will only use MedianBvh as the fallback path until
        //     this issue is resolved with greater certainty.
        msgWarn << "OptiXRenderer) Switching to MedianBvh AS builder..." << sendmsg;   
        RTERR( rtAccelerationSetBuilder(acceleration, "MedianBvh") );
        RTERR( rtAccelerationSetTraverser(acceleration, "Bvh") );
#else

        if (trbvhmemsz + totaluserbufsz > maxgpufreemem) {
          msgWarn << "OptiXRenderer) Switching to MedianBvh AS builder..." << sendmsg;   
          RTERR( rtAccelerationSetBuilder(acceleration, "MedianBvh") );
          RTERR( rtAccelerationSetTraverser(acceleration, "Bvh") );
        } else { 
          msgWarn << "OptiXRenderer) Switching to Trbvh CPU-based AS builder..." << sendmsg;   
          RTERR( rtAccelerationSetBuilder(acceleration, "Trbvh") );
          RTERR( rtAccelerationSetProperty(acceleration, "build_type", "CPU") );
          RTERR( rtAccelerationSetTraverser(acceleration, "Bvh") );
        }
#endif

      }
    }
  }
#endif

  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) Finalizing OptiX rendering kernels...\n");
  RTERR( rtContextValidate(ctx) );
  if (lasterror != RT_SUCCESS) {
    printf("OptiXRenderer) An error occured validating the context. Rendering is aborted.\n");
    return;
  }

  RTERR( rtContextCompile(ctx) );
  if (lasterror != RT_SUCCESS) {
    printf("OptiXRenderer) An error occured compiling the context. Rendering is aborted.\n");
    return;
  }

  double contextinittime = wkf_timer_timenow(ort_timer);
  time_ctx_validate = contextinittime - startctxtime;

  //
  // Force OptiX to build the acceleration structure _now_ by using 
  // an empty launch.  This is done in OptiX sample 6...
  //
// #define ORT_RETRY_FAILED_AS_BUILD 1
#if defined(ORT_RETRY_FAILED_AS_BUILD)
  RTresult rc;
  rc = rtContextLaunch2D(ctx, RT_RAY_GEN_ACCUMULATE, 0, 0);
  RTERR( rc );
#else
  RTERR( rtContextLaunch2D(ctx, RT_RAY_GEN_ACCUMULATE, 0, 0) );
#endif

#if defined(ORT_RETRY_FAILED_AS_BUILD)
  // XXX this never works right, I just get another error saying:
  //    "Cannot set builder while asynchronous building is in progress"
  if (rc == RT_ERROR_MEMORY_ALLOCATION_FAILED) {
    const char *curbuilderstr = NULL;
    RTERR( rtAccelerationGetBuilder(acceleration, &curbuilderstr) );
    printf("Current OptiX builder str: '%s'\n", curbuilderstr);
    if (!strcmp(curbuilderstr, "Trbvh")) {
      // clear previous error so we don't abort immediately...
      lasterror = RT_SUCCESS;

      // issue warning, and try to rebuild the AS using a different builder
      printf("OptiXRenderer) Trbvh AS ran out of GPU memory, retrying with MedianBvh...\n");
      RTERR( rtAccelerationSetBuilder(acceleration, "MedianBvh") );
      RTERR( rtAccelerationSetTraverser(acceleration, "Bvh") );

      // try re-validating and re-compiling context after changing the
      // AS builder to something that can survive GPU memory shortages
      render_compile_and_validate(); 
    }
  }
#endif

  time_ctx_AS_build = wkf_timer_timenow(ort_timer) - contextinittime;
  if (verbose == RT_VERB_DEBUG) {
    printf("OptiXRenderer) launching render: %d x %d\n", width, height);
  }
}


#if defined(VMDOPTIX_INTERACTIVE_OPENGL)

static void *createoptixwindow(const char *wintitle, int width, int height) {
  printf("OptiXRenderer) Creating OptiX window: %d x %d...\n", width, height);

  void *win = glwin_create(wintitle, width, height);
  while (glwin_handle_events(win, GLWIN_EV_POLL_NONBLOCK) != 0);

  glDrawBuffer(GL_BACK);
  glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
  glClearColor(0.0, 0.0, 0.0, 1.0); /* black */
  glViewport(0, 0, width, height);
  glClear(GL_COLOR_BUFFER_BIT);

  glShadeModel(GL_FLAT);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.0, width, height, 0.0, -1.0, 1.0);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glDrawBuffer(GL_BACK);
  glClear(GL_COLOR_BUFFER_BIT);

  glwin_swap_buffers(win);

  return win;
}


static void interactive_viewer_usage(RTcontext ctx, void *win) {
  printf("OptiXRenderer) VMD TachyonL-OptiX Interactive Ray Tracer help:\n");
  printf("OptiXRenderer) ===============================================\n");

  print_ctx_devices(ctx);

  // check for Spaceball/SpaceNavigator/Magellan input devices
  int havespaceball = ((glwin_spaceball_available(win)) && (getenv("VMDDISABLESPACEBALLXDRV") == NULL));
  printf("OptiXRenderer) Spaceball/SpaceNavigator/Magellan: %s\n",
         (havespaceball) ? "Available" : "Not available");

  // check for stereo-capable display
  int havestereo, havestencil;
  glwin_get_wininfo(win, &havestereo, &havestencil);
  printf("OptiXRenderer) Stereoscopic display: %s\n",
         (havestereo) ? "Available" : "Not available");

  // check for vertical retrace sync
  int vsync=0, rc=0;
  if ((rc = glwin_query_vsync(win, &vsync)) == GLWIN_SUCCESS) {
    printf("OptiXRenderer) Vert retrace sync: %s\n", (vsync) ? "On" : "Off");
  } else {
    printf("OptiXRenderer) Vert retrace sync: indeterminate\n");
  }

  printf("OptiXRenderer)\n");
  printf("OptiXRenderer) General controls:\n");
  printf("OptiXRenderer)   space: save numbered snapshot image\n");
  printf("OptiXRenderer)       =: reset to initial view\n");
  printf("OptiXRenderer)       h: print this help info\n");
  printf("OptiXRenderer)       p: print current rendering parameters\n");
  printf("OptiXRenderer)   ESC,q: quit viewer\n");
  printf("OptiXRenderer)\n");
  printf("OptiXRenderer) Display controls\n");
  printf("OptiXRenderer)      F1: override shadows on/off (off=AO off too)\n");
  printf("OptiXRenderer)      F2: override AO on/off\n");
  printf("OptiXRenderer)      F3: override DoF on/off\n");
  printf("OptiXRenderer)      F4: override Depth cueing on/off\n");
#if defined(VMDOPTIX_USE_HMD)
  printf("OptiXRenderer)      F5: override HMD/camera clipping plane/sphere\n");
  printf("OptiXRenderer)      F6: override HMD/camera headlight\n");
  printf("OPtiXRenderer)      F7: toggle HMD interleaved drawing\n");
  printf("OPtiXRenderer)      F8: toggle HMD tex caching/update mode\n");
  printf("OPtiXRenderer)      F9: switch HMD lens distortion mode\n");
#endif
#ifdef USE_REVERSE_SHADOW_RAYS
  printf("OptiXRenderer)     F10: enable/disable shadow ray optimizations\n");
#endif
  printf("OptiXRenderer)     F12: toggle full-screen display on/off\n");
  printf("OptiXRenderer)   1-9,0: override samples per update auto-FPS off\n");
  printf("OptiXRenderer)      Up: increase DoF focal distance\n");
  printf("OptiXRenderer)    Down: decrease DoF focal distance\n");
  printf("OptiXRenderer)    Left: decrease DoF f/stop\n");
  printf("OptiXRenderer)   Right: increase DoF f/stop\n");
  printf("OptiXRenderer)       S: toggle stereoscopic display on/off (if avail)\n");
  printf("OptiXRenderer)       a: toggle AA/AO auto-FPS tuning on/off (on)\n");
  printf("OptiXRenderer)       g: toggle gradient sky xforms on/off (on)\n");
  printf("OptiXRenderer)       l: toggle light xforms on/off (on)\n");
  printf("OptiXRenderer)\n");
  printf("OptiXRenderer) Mouse controls:\n");
  printf("OptiXRenderer)       f: mouse depth-of-field mode\n");
  printf("OptiXRenderer)       r: mouse rotation mode\n");
  printf("OptiXRenderer)       s: mouse scaling mode\n");
  printf("OptiXRenderer)       t: mouse translation mode\n");

  int movie_recording_enabled = (getenv("VMDOPTIXLIVEMOVIECAPTURE") != NULL);
  if (movie_recording_enabled) {
    printf("OptiXRenderer)\n");
    printf("OptiXRenderer) Movie recording controls:\n");
    printf("OptiXRenderer)       R: start/stop movie recording\n");
    printf("OptiXRenderer)       F: toggle movie FPS (24, 30, 60)\n");
  }
}



void OptiXRenderer::render_to_glwin(const char *filename, int writealpha) {
  int i;

  if (!context_created)
    return;

  enum RtMouseMode { RTMM_ROT=0, RTMM_TRANS=1, RTMM_SCALE=2, RTMM_DOF=3 };
  enum RtMouseDown { RTMD_NONE=0, RTMD_LEFT=1, RTMD_MIDDLE=2, RTMD_RIGHT=3 };
  RtMouseMode mm = RTMM_ROT;
  RtMouseDown mousedown = RTMD_NONE;

  // initialize HMD free-run flag to off until HMD code enumerates the 
  // hardware and passes all the way through initialization
  int hmd_freerun = 0;
#if defined(VMDOPTIX_USE_HMD)
  void *hmd_warp = NULL;
#endif
  int hmd_warp_drawmode=1;

  // default HMD distortion correction coefficients assume an Oculus DK2 HMD
  int hmd_warp_coeff_update=0; // warp equation was changed
  int hmd_warp_coeff_edit=1;   // which power of r to edit
  int hmd_warp_coeff_set=0;    // sets: 0=DK2, 1=MSR, 2=User
  const float dk2_warp_coeff[5] = { 1.000f, 0.000f, 0.600f, 0.000f, 0.000f };
//  const float msr_warp_coeff[5] = { 1.000f, 0.290f, 0.195f, 0.045f, 0.360f };
  const float msr_warp_coeff[5] = { 0.950f, 0.330f, 0.195f, 0.045f, 0.360f };
  float user_warp_coeff[5]      = { 1.000f, 0.000f, 0.600f, 0.000f, 0.000f };
  float hmd_warp_coeff[5]       = { 1.000f, 0.000f, 0.600f, 0.000f, 0.000f };

  // obtain user-defined warp coefficients from environment variable if set
  if (getenv("VMDOPTIXHMDUSERWARPCOEFFS")) {
    printf("OptiXRenderer) user-override of default user-defined HMD warp coefficients\n");
    memset(user_warp_coeff, 0, sizeof(user_warp_coeff));
    int cnt=sscanf(getenv("VMDOPTIXHMDUSERWARPCOEFFS"), "%f %f %f %f %f",
                   &user_warp_coeff[0], &user_warp_coeff[1],
                   &user_warp_coeff[2], &user_warp_coeff[3],
                   &user_warp_coeff[4]);
    if (cnt != 5) {
      printf("OptiXRenderer) Warning: only parsed %d coefficients!\n", cnt);
    } else {
      printf("OptiXRenderer) user-defined warp coefficients: %d\n", cnt);
      printf("OptiXRenderer)  %.3f %.3f %.3f %.3f %.3f\n", 
             user_warp_coeff[0], user_warp_coeff[1], 
             user_warp_coeff[2], user_warp_coeff[3], user_warp_coeff[4]);
    }
  }

  // don't interleave extra HMD update/draw passes between buffer updates
  // unless the user wants us to, as a means of improving usability of
  // slow machines
  int hmd_interleave_draws = 0;
  if (getenv("VMDOPTIXHMDINTERLEAVEDRAWS")) {
    hmd_interleave_draws = 1;
    printf("OptiXRenderer) HMD GL draw call interleaving enabled\n");
  }

  int hmd_tex_caching = 0;
  if (getenv("VMDOPTIXHMDTEXCACHING")) {
    hmd_tex_caching = 1;
    printf("OptiXRenderer) HMD texture caching enabled\n");
  }

  // flag to skip the HMD GL draw calls for timing purposes
  int hmd_no_draw = 0;
  if (getenv("VMDOPTIXHMDNODRAW")) {
    hmd_no_draw = 1;
    printf("OptiXRenderer) HMD GL draw calls disabled\n");
  }
  
  // allow user-override of VR HMD sphere geometric res
  float hmd_fov = 95;
  if (getenv("VMDOPTIXHMDFOV")) {
    hmd_fov = atof(getenv("VMDOPTIXHMDFOV"));
    printf("OptiXRenderer) User-override of HMD FoV: %.2f\n", hmd_fov);
  }

  // allow user-override of VR HMD sphere geometric res
  int hmd_spres = 72; // 50 seems like the useful lower-bound spres setting
  if (getenv("VMDOPTIXHMDSPRES")) {
    hmd_spres = atoi(getenv("VMDOPTIXHMDSPRES"));
    printf("OptiXRenderer) User-override of HMD sph res: %d\n", hmd_spres);
  }
 
  // flags to interactively enable/disable shadows, AO, DoF
#if defined(USE_REVERSE_SHADOW_RAYS) && defined(USE_REVERSE_SHADOW_RAYS_DEFAULT)
  int gl_shadows_on=(shadows_enabled) ? RT_SHADOWS_ON_REVERSE : RT_SHADOWS_OFF;
#else
  int gl_shadows_on=(shadows_enabled) ? RT_SHADOWS_ON : RT_SHADOWS_OFF;
#endif

  int gl_fs_on=0; // fullscreen window state
  int fsowsx=0, fsowsy=0; // store last win size before fullscreen
  int owsx=0, owsy=0; // last win size
  int gl_ao_on=(ao_samples > 0);
  int gl_dof_on, gl_dof_on_old;
  gl_dof_on=gl_dof_on_old=dof_enabled; 
  int gl_fog_on=(fog_mode != RT_FOG_NONE);
  int gl_clip_on=(clipview_mode != RT_CLIP_NONE);
  int gl_headlight_on=(headlight_mode != RT_HEADLIGHT_OFF);

  // Enable live recording of a session to a stream of image files indexed
  // by their display presentation time, mapped to the nearest frame index
  // in a fixed-frame-rate image sequence (e.g. 24, 30, or 60 FPS), 
  // to allow subsequent encoding into a standard movie format.
  // XXX this feature is disabled by default at present, to prevent people
  //     from accidentally turning it on during a live demo or the like
  int movie_recording_enabled = (getenv("VMDOPTIXLIVEMOVIECAPTURE") != NULL);
  int movie_recording_on = 0;
  double movie_recording_start_time = 0.0;
  int movie_recording_fps = 30;
  int movie_framecount = 0;
  int movie_lastframeindex = 0;
  const char *movie_recording_filebase = "vmdlivemovie.%05d.tga";
  if (getenv("VMDOPTIXLIVEMOVIECAPTUREFILEBASE"))
    movie_recording_filebase = getenv("VMDOPTIXLIVEMOVIECAPTUREFILEBASE");

  // Enable/disable Spaceball/SpaceNavigator/Magellan input 
  int spaceballenabled=(getenv("VMDDISABLESPACEBALLXDRV") == NULL) ? 1 : 0;
  int spaceballmode=0;       // default mode is rotation/translation
  int spaceballflightmode=0; // 0=moves object, 1=camera fly
  if (getenv("VMDOPTIXSPACEBALLFLIGHT"))
    spaceballflightmode=1;

  // total AA/AO sample count
  int totalsamplecount=0;

  // counter for snapshots of live image...
  int snapshotcount=0;

  // flag to enable automatic AO sample count adjustment for FPS rate control
#if defined(VMDOPTIX_PROGRESSIVEAPI)
  int vcarunning=0;      // flag to indicate VCA streaming active/inactive
#if 1
  int autosamplecount=0; // leave disabled for now
#else
  int autosamplecount=1; // works partially in current revs of progressive API
  if (remote_device != NULL)
    autosamplecount=0;  // re-disable when targeting VCA for now
#endif
#else
  int autosamplecount=1;
#endif

  // flag to enable transformation of lights and gradient sky sphere, 
  // so that they track camera orientation as they do in the VMD OpenGL display
  int xformlights=1, xformgradientsphere=1;

  //
  // allocate or reconfigure the framebuffer, accumulation buffer, 
  // and output streams required for progressive rendering, either
  // using the new progressive APIs, or using our own code.
  //
  // Unless overridden by environment variables, we use the incoming
  // window size parameters from VMD to initialize the RT image dimensions.
  // If image size is overridden, often when using HMDs, the incoming 
  // dims are window dims are used to size the GL window, but the image size
  // is set independently.
  int wsx=width, wsy=height;
  const char *imageszstr = getenv("VMDOPTIXIMAGESIZE");
  if (imageszstr) {
    if (sscanf(imageszstr, "%d %d", &width, &height) != 2) {
      width=wsx;
      height=wsy;
    } 
  } 

  config_framebuffer(width, height, 1);

  // prepare the majority of OptiX rendering state before we go into 
  // the interactive rendering loop
  update_rendering_state(1);
  render_compile_and_validate();

  // make a copy of state we're going to interactively manipulate,
  // so that we can recover to the original state on-demand
  int samples_per_pass = 1;
  int force_ao_1 = 0; // whether or not to force AO count per pass to 1
  int cur_aa_samples = aa_samples;
  int cur_ao_samples = ao_samples;
  float cam_zoom_orig = cam_zoom;
  float scene_gradient_orig[3] = {0.0f, 1.0f, 0.0f};
  vec_copy(scene_gradient_orig, scene_gradient);

  float cam_pos_orig[3] = {0.0f, 0.0f, 2.0f};
  float cam_U_orig[3] = {1.0f, 0.0f, 0.0f};
  float cam_V_orig[3] = {0.0f, 1.0f, 0.0f};
  float cam_W_orig[3] = {0.0f, 0.0f, -1.0f};
  float cam_pos[3], cam_U[3], cam_V[3], cam_W[3];
  float hmd_U[3], hmd_V[3], hmd_W[3];

  vec_copy(cam_pos, cam_pos_orig);
  vec_copy(cam_U, cam_U_orig);
  vec_copy(cam_V, cam_V_orig);
  vec_copy(cam_W, cam_W_orig);

  // copy light directions
  DirectionalLight *cur_dlights = (DirectionalLight *) calloc(1, directional_lights.num() * sizeof(DirectionalLight));
  for (i=0; i<directional_lights.num(); i++) {
    vec_copy((float*)&cur_dlights[i].dir, directional_lights[i].dir);
    vec_normalize((float*)&cur_dlights[i].dir);
  }

#if defined(VMDOPTIX_USE_HMD)
  HMDMgr *hmd = NULL;
  // if using HMD for display, get head tracking data
  if (getenv("VMDOPTIXUSEHMD") != NULL) {
    hmd = new HMDMgr();
    if (hmd->device_count() < 1) {
      delete hmd;
      hmd = NULL;
    }
  }
  if (hmd) {
    autosamplecount=0; // disable when targeting HMDs
    msgInfo << "OptiXRenderer) HMD in use, disabling auto sample count adjustement," << sendmsg;
    msgInfo << "OptiXRenderer) optimizing for lowest rendering latency." << sendmsg;
  }

#if defined(VMDOPTIX_PROGRESSIVEAPI) 
  hmd_freerun = (hmd != NULL && camera_projection == RT_EQUIRECTANGULAR);
#endif

#if defined(VMDUSEEVENTIO)
  evio_handle eviodev = NULL;
  const char *eviodevname = getenv("VMDOPTIXEVIODEV");
  if (hmd && eviodevname) {
    msgInfo << "OptiXRenderer) Attempting to open '" 
            << eviodevname << "' for Linux event I/O input..." << sendmsg; 
    eviodev = evio_open(eviodevname);
    if (eviodev) {
      msgInfo << "OptiXRenderer) Using Linux event I/O input:" << sendmsg;
      evio_print_devinfo(eviodev);
    }
  }
#endif
#endif

  // create the display window
  const char *windowtitle;
#if 1
  windowtitle = "VMD TachyonL-OptiX Interactive Ray Tracer";
#else
  // useful for demos and perf comparisons
  if (getenv("VMDOPTIXNORTX") != NULL || hwtri_enabled==0) {
    windowtitle = "VMD TachyonL-OptiX Interactive Ray Tracer -- Turing RTX DISABLED";
  } else {
    windowtitle = "VMD TachyonL-OptiX Interactive Ray Tracer -- Turing RTX ENABLED";
  }
#endif

  void *win = createoptixwindow(windowtitle, width, height);
  interactive_viewer_usage(ctx, win);
  
  // check for stereo-capable display
  int havestereo=0, havestencil=0;
  int stereoon=0, stereoon_old=0;
  glwin_get_wininfo(win, &havestereo, &havestencil);

#if defined(VMDOPTIX_USE_HMD) && defined(VMDOPTIX_PROGRESSIVEAPI)
  if (hmd_freerun && hmd != NULL && camera_projection == RT_EQUIRECTANGULAR) {
    glwin_spheremap_draw_prepare(win);

    // enable HMD optical distortion correction, unless force-disabled
    if (hmd != NULL && !getenv("VMDOPTIXHMDNOWARP"))
      hmd_warp = glwin_spheremap_create_hmd_warp(win, wsx, wsy, 21, 0,
                                                 width, height, hmd_warp_coeff);

    // if an HMD is in use, we trigger full-screen display by default
    if (hmd) {
      if (glwin_fullscreen(win, 1, 0) == 0) {
        gl_fs_on = 1;
        fsowsx = wsx;
        fsowsy = wsy;
        glwin_get_winsize(win, &wsx, &wsy);
      } else {
        printf("OptiXRenderer) Fullscreen mode not available\n");
      }
    }
  }
#endif

  // Override AA/AO sample counts since we're doing progressive rendering.
  // Choosing an initial AO sample count of 1 will give us the peak progressive 
  // display update rate, but we end up wasting time on re-tracing many
  // primary rays.  The automatic FPS optimization scheme below will update
  // the number of samples per rendering pass and assign the best values for
  // AA/AO samples accordingly.
  cur_aa_samples = samples_per_pass;
  if (cur_ao_samples > 0) {
    cur_aa_samples = 1;
    cur_ao_samples = samples_per_pass;
  }

  const char *statestr = "|/-\\.";
  int done=0, winredraw=1, accum_count=0;
  int state=0, mousedownx=0, mousedowny=0;
  float cur_cam_zoom = cam_zoom_orig;

  double fpsexpave=0.0; 
  double hmdfpsexpave=0.0;
  double hmdgldrawtime=0.0;
  double mapbuftotaltime=0.0;
  
  double oldtime = wkf_timer_timenow(ort_timer);
#if defined(VMDOPTIX_USE_HMD)
  double hmdoldtime = oldtime;
#endif
  while (!done) { 
    int winevent=0;

#if 1
    if (app->uivs && app->uivs->srv_connected()) {
      if (app->uivs->srv_check_ui_event()) {
        int eventtype;
        app->uivs->srv_get_last_event_type(eventtype);
        switch (eventtype) {
          case VideoStream::VS_EV_ROTATE_BY:
            { int axis;
              float angle;
              app->uivs->srv_get_last_rotate_by(angle, axis);
              Matrix4 rm;

              switch (axis) {
                case 'x':
                  rm.rotate_axis(cam_U, -angle * M_PI/180.0f);
                  break;

                case 'y':
                  rm.rotate_axis(cam_V, -angle * M_PI/180.0f);
                  break;

                case 'z':
                  rm.rotate_axis(cam_W, -angle * M_PI/180.0f);
                  break;
              }
              rm.multpoint3d(cam_pos, cam_pos);
              rm.multnorm3d(cam_U, cam_U);
              rm.multnorm3d(cam_V, cam_V);
              rm.multnorm3d(cam_W, cam_W);

              if (xformgradientsphere) {
                rm.multnorm3d(scene_gradient, scene_gradient);
              }

              if (xformlights) {
                // update light directions (comparatively costly)
                for (i=0; i<directional_lights.num(); i++) {
                  rm.multnorm3d((float*)&cur_dlights[i].dir, (float*)&cur_dlights[i].dir);
                }
              }
              winredraw = 1;
            }
            break;

          case VideoStream::VS_EV_TRANSLATE_BY:
            {
              float dU[3], dV[3], dW[3];
              float tx, ty, tz;
              app->uivs->srv_get_last_translate_by(tx, ty, tz);
              vec_scale(dU, -tx, cam_U);
              vec_scale(dV, -ty, cam_V);
              vec_scale(dW, -tz, cam_W);
              vec_add(cam_pos, cam_pos, dU);
              vec_add(cam_pos, cam_pos, dV);
              vec_add(cam_pos, cam_pos, dW);
              winredraw = 1;
            }
            break;

          case VideoStream::VS_EV_SCALE_BY:
            { float zoominc;
              app->uivs->srv_get_last_scale_by(zoominc);
              cam_zoom *= 1.0f / zoominc;
              winredraw = 1;
            }
            break;

#if 0
              } else if (mm == RTMM_DOF) {
                cam_dof_fnumber += txdx * 20.0f;
                if (cam_dof_fnumber < 1.0f) cam_dof_fnumber = 1.0f;
                cam_dof_focal_dist += -txdy;
                if (cam_dof_focal_dist < 0.01f) cam_dof_focal_dist = 0.01f;
                winredraw = 1;
              }
            }
#endif

          case VideoStream::VS_EV_NONE: 
                  default:
            // should never happen...
            break;
        }
      }
    }              
#endif


    while ((winevent = glwin_handle_events(win, GLWIN_EV_POLL_NONBLOCK)) != 0) {
      int evdev, evval;
      char evkey;

      glwin_get_lastevent(win, &evdev, &evval, &evkey);
      glwin_get_winsize(win, &wsx, &wsy);

      if (evdev == GLWIN_EV_WINDOW_CLOSE) {
        printf("OptiXRenderer) display window closed, exiting...\n");
        done = 1;
        winredraw = 0;
      } else if (evdev == GLWIN_EV_KBD) {
        switch (evkey) {
          case '`': autosamplecount=0; samples_per_pass=1; 
                    force_ao_1 = (!force_ao_1); winredraw=1; 
                    printf("OptiXRenderer) Toggling forced single AO sample per pass: %s\n",
                           force_ao_1 ? "on" : "off");
                    break;

          // update HMD warp distortion coefficients
          case '|': hmd_warp_coeff_set = (hmd_warp_coeff_set + 1) % 3;
                   switch (hmd_warp_coeff_set) {
                     case 0: 
                       printf("\nDistortion correction: DK2 stock lens\n");
                       memcpy(hmd_warp_coeff, dk2_warp_coeff, 5*sizeof(float));
                       break;

                     case 1: 
                       printf("\nDistortion correction: DK2 w/ MSR lens\n");
                       memcpy(hmd_warp_coeff, msr_warp_coeff, 5*sizeof(float));
                       break;

                     case 2: 
                       printf("\nDistortion correction: User defined lens\n");
                       memcpy(hmd_warp_coeff, user_warp_coeff, 5*sizeof(float));
                       break;
                   }
                   printf("\nHMD warp coeff: %.3f, %.3f, %.3f, %.3f, %.3f\n", 
                          hmd_warp_coeff[0], hmd_warp_coeff[1], 
                          hmd_warp_coeff[2], hmd_warp_coeff[3],
                          hmd_warp_coeff[4]);
                   hmd_warp_coeff_update=1; winredraw=1; break;
                   break; 

          case '\\': hmd_warp_coeff_edit = (hmd_warp_coeff_edit + 1) % 5;
                    printf("\nHMD edit warp coeff: r^%d\n",hmd_warp_coeff_edit);
                    break; 

          case '[': hmd_warp_coeff[hmd_warp_coeff_edit]-=0.005; 
                    printf("\nHMD warp coeff: %.3f, %.3f, %.3f, %.3f, %.3f\n", 
                           hmd_warp_coeff[0], hmd_warp_coeff[1], 
                           hmd_warp_coeff[2], hmd_warp_coeff[3],
                           hmd_warp_coeff[4]);
                    memcpy(user_warp_coeff, hmd_warp_coeff, 5*sizeof(float));
                    hmd_warp_coeff_update=1; winredraw=1; break;

          case ']': hmd_warp_coeff[hmd_warp_coeff_edit]+=0.005; 
                    printf("\nHMD warp coeff: %.3f, %.3f, %.3f, %.3f, %.3f\n", 
                           hmd_warp_coeff[0], hmd_warp_coeff[1], 
                           hmd_warp_coeff[2], hmd_warp_coeff[3],
                           hmd_warp_coeff[4]);
                    memcpy(user_warp_coeff, hmd_warp_coeff, 5*sizeof(float));
                    hmd_warp_coeff_update=1; winredraw=1; break;

          // update sample counts
          case  '1': autosamplecount=0; samples_per_pass=1; winredraw=1; break;
          case  '2': autosamplecount=0; samples_per_pass=2; winredraw=1; break;
          case  '3': autosamplecount=0; samples_per_pass=3; winredraw=1; break;
          case  '4': autosamplecount=0; samples_per_pass=4; winredraw=1; break;
          case  '5': autosamplecount=0; samples_per_pass=5; winredraw=1; break;
          case  '6': autosamplecount=0; samples_per_pass=6; winredraw=1; break;
          case  '7': autosamplecount=0; samples_per_pass=7; winredraw=1; break;
          case  '8': autosamplecount=0; samples_per_pass=8; winredraw=1; break;
          case  '9': autosamplecount=0; samples_per_pass=9; winredraw=1; break;
          case  '0': autosamplecount=0; samples_per_pass=10; winredraw=1; break;

          case  '=': /* recover back to initial state */
            vec_copy(scene_gradient, scene_gradient_orig);
            cam_zoom = cam_zoom_orig;
            vec_copy(cam_pos, cam_pos_orig);
            vec_copy(cam_U, cam_U_orig);
            vec_copy(cam_V, cam_V_orig);
            vec_copy(cam_W, cam_W_orig);

            // restore original light directions
            for (i=0; i<directional_lights.num(); i++) {
              vec_copy((float*)&cur_dlights[i].dir, directional_lights[i].dir);
              vec_normalize((float*)&cur_dlights[i].dir);
            }
            winredraw = 1;
#if defined(VMDOPTIX_USE_HMD)
            // Handle HMD head orientation updates if we have one attached
            if (hmd) {
              hmd->reset_orientation();
              printf("\nOptiXRenderer) Resetting HMD orientation\n");
            }
#endif
            break;
 
          case  ' ': /* spacebar saves current image with counter */
            {
              char snapfilename[256];
              sprintf(snapfilename, "vmdsnapshot.%04d.tga", snapshotcount);
              if (OptiXWriteImage(snapfilename, writealpha, framebuffer) != -1) {
                printf("OptiXRenderer) Saved snapshot to '%s'             \n",
                       snapfilename);
              }
              snapshotcount++; 
            }
            break;

          case  'a': /* toggle automatic sample count FPS tuning */
            autosamplecount = !(autosamplecount);
            printf("\nOptiXRenderer) Automatic AO sample count FPS tuning %s\n",
                   (autosamplecount) ? "enabled" : "disabled");
            break;

          case  'f': /* DoF mode */
            mm = RTMM_DOF;
            printf("\nOptiXRenderer) Mouse DoF aperture and focal dist. mode\n");
            break;

          case  'g': /* toggle gradient sky sphere xforms */
            xformgradientsphere = !(xformgradientsphere);
            printf("\nOptiXRenderer) Gradient sky sphere transformations %s\n",
                   (xformgradientsphere) ? "enabled" : "disabled");
            break;

          case  'h': /* print help message */
            printf("\n");
            interactive_viewer_usage(ctx, win);

            // we have to force a redraw after querying OptiX context
            // info due to the current behavior of the progressive API,
            // which halts upon any API call other than the simplest queries
            winredraw = 1; 
            break;

          case  'l': /* toggle lighting xforms */
            xformlights = !(xformlights);
            printf("\nOptiXRenderer) Light transformations %s\n",
                   (xformlights) ? "enabled" : "disabled");
            break;

          case  'p': /* print current RT settings */
            printf("\nOptiXRenderer) Current Ray Tracing Parameters:\n"); 
            printf("OptiXRenderer) -------------------------------\n"); 
            printf("OptiXRenderer) Camera zoom: %f\n", cur_cam_zoom);
            printf("OptiXRenderer) Shadows: %s  Ambient occlusion: %s\n",
                   (gl_shadows_on) ? "on" : "off",
                   (gl_ao_on) ? "on" : "off");
            printf("OptiXRenderer) Antialiasing samples per-pass: %d\n",
                   cur_aa_samples);
            printf("OptiXRenderer) Ambient occlusion samples per-pass: %d\n",
                   cur_ao_samples);
            printf("OptiXRenderer) Depth-of-Field: %s f/num: %.1f  Foc. Dist: %.2f\n",
                   (gl_dof_on) ? "on" : "off", 
                   cam_dof_fnumber, cam_dof_focal_dist);
            printf("OptiXRenderer)   Win size: %d x %d\n", wsx, wsy);
            printf("OptiXRenderer) Image size: %d x %d\n", width, height);
            break;

          case  'r': /* rotate mode */
            mm = RTMM_ROT;
            printf("\nOptiXRenderer) Mouse rotation mode\n");
            break;

          case  's': /* scaling mode */
            mm = RTMM_SCALE;
            printf("\nOptiXRenderer) Mouse scaling mode\n");
            break;

          case  'F': /* toggle live movie recording FPS (24, 30, 60) */
            if (movie_recording_enabled) {
              switch (movie_recording_fps) {
                case 24: movie_recording_fps = 30; break;
                case 30: movie_recording_fps = 60; break;
                case 60:
                default: movie_recording_fps = 24; break;
              }
              printf("\nOptiXRenderer) Movie recording FPS rate: %d\n", 
                     movie_recording_fps);
            } else {
              printf("\nOptiXRenderer) Movie recording not available.\n");
            }
            break;

          case  'R': /* toggle live movie recording mode on/off */
            if (movie_recording_enabled) {
              movie_recording_on = !(movie_recording_on);
              printf("\nOptiXRenderer) Movie recording %s\n",
                     (movie_recording_on) ? "STARTED" : "STOPPED");
              if (movie_recording_on) {
                movie_recording_start_time = wkf_timer_timenow(ort_timer); 
                movie_framecount = 0;
                movie_lastframeindex = 0;
              } else {
                printf("OptiXRenderer) Encode movie with:\n");
                printf("OptiXRenderer)   ffmpeg -f image2 -i vmdlivemovie.%%05d.tga -c:v libx264 -profile:v baseline -level 3.0 -pix_fmt yuv420p -b:v 15000000 output.mp4\n");
              }
            } else {
              printf("\nOptiXRenderer) Movie recording not available.\n");
            }
            break;

          case  'S': /* toggle stereoscopic display mode */
            if (havestereo) {
              stereoon = (!stereoon);
              printf("\nOptiXRenderer) Stereoscopic display %s\n",
                     (stereoon) ? "enabled" : "disabled");
              winredraw = 1;
            } else {
              printf("\nOptiXRenderer) Stereoscopic display unavailable\n");
            }
            break;
 
          case  't': /* translation mode */
            mm = RTMM_TRANS;
            printf("\nOptiXRenderer) Mouse translation mode\n");
            break;
            
          case  'q': /* 'q' key */
          case  'Q': /* 'Q' key */
          case 0x1b: /* ESC key */
            printf("\nOptiXRenderer) Exiting on user input.               \n");
            done=1; /* exit from interactive RT window */
            break;
        }
      } else if (evdev != GLWIN_EV_NONE) {
        switch (evdev) {
          case GLWIN_EV_KBD_F1: /* turn shadows on/off */
#if defined(USE_REVERSE_SHADOW_RAYS) && defined(USE_REVERSE_SHADOW_RAYS_DEFAULT)
            gl_shadows_on=(!gl_shadows_on) ? RT_SHADOWS_ON_REVERSE : RT_SHADOWS_OFF;
#else
            gl_shadows_on=(!gl_shadows_on) ? RT_SHADOWS_ON : RT_SHADOWS_OFF;
            // gl_shadows_on = (!gl_shadows_on);
#endif

            printf("\n");
#if defined(USE_REVERSE_SHADOW_RAYS) && defined(USE_REVERSE_SHADOW_RAYS_DEFAULT)
            printf("OptiXRenderer) Shadows %s\n",
                   (gl_shadows_on) ? "enabled (reversal opt.)" : "disabled");
#else
            printf("OptiXRenderer) Shadows %s\n",
                   (gl_shadows_on) ? "enabled" : "disabled");
#endif
            winredraw = 1; 
            break;

          case GLWIN_EV_KBD_F2: /* turn AO on/off */
            gl_ao_on = (!gl_ao_on); 
            printf("\n");
            printf("OptiXRenderer) Ambient occlusion %s\n",
                   (gl_ao_on) ? "enabled" : "disabled");
            winredraw = 1; 
            break;

          case GLWIN_EV_KBD_F3: /* turn DoF on/off */
            gl_dof_on = (!gl_dof_on);
            printf("\n");
            printf("OptiXRenderer) Depth-of-field %s\n",
                   (gl_dof_on) ? "enabled" : "disabled");
            winredraw = 1;
            break;

          case GLWIN_EV_KBD_F4: /* turn fog/depth cueing on/off */
            gl_fog_on = (!gl_fog_on); 
            printf("\n");
            printf("OptiXRenderer) Depth cueing %s\n",
                   (gl_fog_on) ? "enabled" : "disabled");
            winredraw = 1; 
            break;

          case GLWIN_EV_KBD_F5: /* turn HMD/camera fade+clipping on/off */
            gl_clip_on = (!gl_clip_on);
            printf("\n");
            printf("OptiXRenderer) HMD/camera clipping plane/sphere %s\n",
                   (gl_clip_on) ? "enabled" : "disabled");
            winredraw = 1; 
            break;

          case GLWIN_EV_KBD_F6: /* turn HMD/camera headlight on/off */
            gl_headlight_on = (!gl_headlight_on); 
            printf("\n");
            printf("OptiXRenderer) HMD/camera headlight %s\n",
                   (gl_headlight_on) ? "enabled" : "disabled");
            winredraw = 1; 
            break;

          case GLWIN_EV_KBD_F7: /* turn HMD draw interleaving on/off */
            hmd_interleave_draws = (!hmd_interleave_draws); 
            printf("\n");
            printf("OptiXRenderer) HMD interleaved draws %s\n",
                   (hmd_interleave_draws) ? "enabled" : "disabled");
            break;

          case GLWIN_EV_KBD_F8: /* turn HMD tex caching on/off */
            hmd_tex_caching = (!hmd_tex_caching); 
            printf("\n");
            printf("OptiXRenderer) HMD tex caching %s\n",
                   (hmd_tex_caching) ? "enabled" : "disabled");
            break;

          case GLWIN_EV_KBD_F9: /* switch HMD lens distortion correction mode */
            hmd_warp_drawmode = (hmd_warp_drawmode+1) % 5;
            printf("\n");
            { const char *warpmodestr="Off";
              switch (hmd_warp_drawmode) {
                case 0: warpmodestr="Lens: Off  Chroma: Off  Grid: Off"; break;
                case 1: warpmodestr="Lens: On   Chroma: On   Grid: Off"; break;
                case 2: warpmodestr="Lens: On   Chroma: On   Grid: On "; break;
                case 3: warpmodestr="Lens: On   Chroma: Off  Grid: Off"; break;
                case 4: warpmodestr="Lens: On   Chroma: Off  Grid: On "; break;
              }
              printf("OptiXRenderer) HMD Corr.  %s\n", warpmodestr);
            }
            break;

#ifdef USE_REVERSE_SHADOW_RAYS
          case GLWIN_EV_KBD_F10: /* toggle shadow ray reversal on/off */
            if (gl_shadows_on == RT_SHADOWS_ON) 
              gl_shadows_on = RT_SHADOWS_ON_REVERSE;
            else if (gl_shadows_on == RT_SHADOWS_ON_REVERSE)
              gl_shadows_on = RT_SHADOWS_ON;
            printf("\n");
            printf("OptiXRenderer) Shadow ray reversal %s\n",
                   (gl_shadows_on==RT_SHADOWS_ON_REVERSE) ? "enabled" : "disabled");
            winredraw = 1; 
            break;
#endif

          case GLWIN_EV_KBD_F12: /* toggle full-screen window on/off */
            gl_fs_on = (!gl_fs_on);
            printf("\nOptiXRenderer) Toggling fullscreen window %s\n",
                   (gl_fs_on) ? "on" : "off");
            if (gl_fs_on) { 
              if (glwin_fullscreen(win, gl_fs_on, 0) == 0) {
                fsowsx = wsx;
                fsowsy = wsy;
                glwin_get_winsize(win, &wsx, &wsy);
              } else {
                printf("OptiXRenderer) Fullscreen mode note available\n");
              }
            } else {
              glwin_fullscreen(win, gl_fs_on, 0);
              glwin_resize(win, fsowsx, fsowsy);
            }
            winredraw = 1; 
            break;

          case GLWIN_EV_KBD_UP: /* change depth-of-field focal dist */
            cam_dof_focal_dist *= 1.02f; 
            printf("\nOptiXRenderer) DoF focal dist: %f\n", cam_dof_focal_dist);
            winredraw = 1; 
            break;

          case GLWIN_EV_KBD_DOWN: /* change depth-of-field focal dist */
            cam_dof_focal_dist *= 0.96f; 
            if (cam_dof_focal_dist < 0.02f) cam_dof_focal_dist = 0.02f;
            printf("\nOptiXRenderer) DoF focal dist: %f\n", cam_dof_focal_dist);
            winredraw = 1; 
            break;

          case GLWIN_EV_KBD_RIGHT: /* change depth-of-field f/stop number */
            cam_dof_fnumber += 1.0f; 
            printf("\nOptiXRenderer) DoF f/stop: %f\n", cam_dof_fnumber);
            winredraw = 1; 
            break;

          case GLWIN_EV_KBD_LEFT: /* change depth-of-field f/stop number */
            cam_dof_fnumber -= 1.0f; 
            if (cam_dof_fnumber < 1.0f) cam_dof_fnumber = 1.0f;
            printf("\nOptiXRenderer) DoF f/stop: %f\n", cam_dof_fnumber);
            winredraw = 1; 
            break;

          case GLWIN_EV_MOUSE_MOVE:
            if (mousedown != RTMD_NONE) {
              int x, y;
              glwin_get_mousepointer(win, &x, &y);

              float zoommod = 2.0f*cur_cam_zoom/cam_zoom_orig;
              float txdx = (x - mousedownx) * zoommod / wsx;
              float txdy = (y - mousedowny) * zoommod / wsy;
              if (mm != RTMM_SCALE) {
                mousedownx = x;
                mousedowny = y;
              }

              if (mm == RTMM_ROT) {
                Matrix4 rm;
                if (mousedown == RTMD_LEFT) {
                  // when zooming in further from the initial view, we
                  // rotate more slowly so control remains smooth
                  rm.rotate_axis(cam_V, -txdx);
                  rm.rotate_axis(cam_U, -txdy);
                } else if (mousedown == RTMD_MIDDLE || 
                           mousedown == RTMD_RIGHT) {
                  rm.rotate_axis(cam_W, txdx);
                }
                rm.multpoint3d(cam_pos, cam_pos);
                rm.multnorm3d(cam_U, cam_U);
                rm.multnorm3d(cam_V, cam_V);
                rm.multnorm3d(cam_W, cam_W);

                if (xformgradientsphere) {
                  rm.multnorm3d(scene_gradient, scene_gradient);
                }
 
                if (xformlights) {
                  // update light directions (comparatively costly)
                  for (i=0; i<directional_lights.num(); i++) {
                    rm.multnorm3d((float*)&cur_dlights[i].dir, (float*)&cur_dlights[i].dir);
                  }
                }

                winredraw = 1;
              } else if (mm == RTMM_TRANS) {
                if (mousedown == RTMD_LEFT) {
                  float dU[3], dV[3];
                  vec_scale(dU, -txdx, cam_U);
                  vec_scale(dV,  txdy, cam_V);
                  vec_add(cam_pos, cam_pos, dU); 
                  vec_add(cam_pos, cam_pos, dV); 
                } else if (mousedown == RTMD_MIDDLE || 
                           mousedown == RTMD_RIGHT) {
                  float dW[3];
                  vec_scale(dW, txdx, cam_W);
                  vec_add(cam_pos, cam_pos, dW); 
                } 
                winredraw = 1;
              } else if (mm == RTMM_SCALE) {
                float txdx = (x - mousedownx) * 2.0 / wsx;
                float zoominc = 1.0 - txdx;
                if (zoominc < 0.01) zoominc = 0.01;
                cam_zoom = cur_cam_zoom * zoominc;
                winredraw = 1;
              } else if (mm == RTMM_DOF) {
                cam_dof_fnumber += txdx * 20.0f;
                if (cam_dof_fnumber < 1.0f) cam_dof_fnumber = 1.0f;
                cam_dof_focal_dist += -txdy; 
                if (cam_dof_focal_dist < 0.01f) cam_dof_focal_dist = 0.01f;
                winredraw = 1;
              }
            }
            break;

          case GLWIN_EV_MOUSE_LEFT:
          case GLWIN_EV_MOUSE_MIDDLE:
          case GLWIN_EV_MOUSE_RIGHT:
            if (evval) {
              glwin_get_mousepointer(win, &mousedownx, &mousedowny);
              cur_cam_zoom = cam_zoom;

              if (evdev == GLWIN_EV_MOUSE_LEFT) mousedown = RTMD_LEFT;
              else if (evdev == GLWIN_EV_MOUSE_MIDDLE) mousedown = RTMD_MIDDLE;
              else if (evdev == GLWIN_EV_MOUSE_RIGHT) mousedown = RTMD_RIGHT;
            } else {
              mousedown = RTMD_NONE;
            }
            break;

          case GLWIN_EV_MOUSE_WHEELUP:
            cam_zoom /= 1.1f; winredraw = 1; break;

          case GLWIN_EV_MOUSE_WHEELDOWN:
            cam_zoom *= 1.1f; winredraw = 1; break;
        }
      }
    }


    //
    // Support for Spaceball/Spacenavigator/Magellan devices that use
    // X11 ClientMessage protocol....
    //
    if (spaceballenabled) {
      // Spaceball/Spacenavigator/Magellan event state variables
      int tx=0, ty=0, tz=0, rx=0, ry=0, rz=0, buttons=0;
      if (glwin_get_spaceball(win, &rx, &ry, &rz, &tx, &ty, &tz, &buttons)) {
        // negate directions if we're in flight mode...
        if (spaceballflightmode) {
          rx= -rx;
          ry= -ry;
          rz= -rz;

          tx= -tx;
          ty= -ty;
          tz= -tz;
        }

        // check for button presses to reset the view
        if (buttons & 1) {
          printf("OptiXRenderer) spaceball button 1 pressed: reset view\n");
          vec_copy(scene_gradient, scene_gradient_orig);
          cam_zoom = cam_zoom_orig;
          vec_copy(cam_pos, cam_pos_orig);
          vec_copy(cam_U, cam_U_orig);
          vec_copy(cam_V, cam_V_orig);
          vec_copy(cam_W, cam_W_orig);

          // restore original light directions
          for (i=0; i<directional_lights.num(); i++) {
            vec_copy((float*)&cur_dlights[i].dir, directional_lights[i].dir);
            vec_normalize((float*)&cur_dlights[i].dir);
          }
          winredraw = 1;
        }

        // check for button presses to toggle spaceball mode
        if (buttons & 2) {
          spaceballmode = !(spaceballmode);
          printf("OptiXRenderer) spaceball mode: %s                        \n",
                 (spaceballmode) ? "scaling" : "rotation/translation");
        }

        // rotation/translation mode
        if (spaceballmode == 0) {
          float zoommod = 2.0f*cam_zoom/cam_zoom_orig;
          float divlen = sqrtf(wsx*wsx + wsy*wsy) * 50;

          // check for rotation and handle it...
          if (rx != 0 || ry !=0 || rz !=0) {
            Matrix4 rm;
            rm.rotate_axis(cam_U, -rx * zoommod / divlen);
            rm.rotate_axis(cam_V, -ry * zoommod / divlen);
            rm.rotate_axis(cam_W, -rz * zoommod / divlen);

            rm.multpoint3d(cam_pos, cam_pos);
            rm.multnorm3d(cam_U, cam_U);
            rm.multnorm3d(cam_V, cam_V);
            rm.multnorm3d(cam_W, cam_W);

            if (xformgradientsphere) {
              rm.multnorm3d(scene_gradient, scene_gradient);
            }

            if (xformlights) {
              // update light directions (comparatively costly)
              for (i=0; i<directional_lights.num(); i++) {
                rm.multnorm3d((float*)&cur_dlights[i].dir, (float*)&cur_dlights[i].dir);
              }
            }
            winredraw = 1;
          }

          // check for translation and handle it...
          if (tx != 0 || ty !=0 || tz !=0) {
            float dU[3], dV[3], dW[3];
            vec_scale(dU, -tx * zoommod / divlen, cam_U);
            vec_scale(dV, -ty * zoommod / divlen, cam_V);
            vec_scale(dW, -tz * zoommod / divlen, cam_W);
            vec_add(cam_pos, cam_pos, dU);
            vec_add(cam_pos, cam_pos, dV);
            vec_add(cam_pos, cam_pos, dW);
            winredraw = 1;
          }
        }

        // scaling mode
        if (spaceballmode == 1) {
          const float sbscale = 1.0f / (1024.0f * 8.0f);
          float zoominc = 1.0f - (rz * sbscale);
          if (zoominc < 0.01) zoominc = 0.01;
            cam_zoom *= zoominc;
            winredraw = 1;
        }

      }
    }


    // if there is no HMD, we use the camera orientation directly  
    vec_copy(hmd_U, cam_U);
    vec_copy(hmd_V, cam_V);
    vec_copy(hmd_W, cam_W);

    //
    // Handle HMD head orientation by directly manipulating the 
    // view orientation if we have an HMD attached and we're using
    // the direct-to-HMD "Oculus Rift" camera mode.  For the other 
    // equirectangular or cubemap cameras, the head pose processing
    // and all HMD lens aberration correction must done in the OpenGL 
    // display code rather than in the RT code itself.  When using the
    // direct-drive HMD camera, we only run a single rendering pass before
    // triggering a fresh orientation, since the head pose changes constantly.
    //
#if defined(VMDOPTIX_USE_HMD)
    float hmd_U_new[3] = {1.0f, 0.0f, 0.0f};
    float hmd_V_new[3] = {0.0f, 1.0f, 0.0f};
    float hmd_W_new[3] = {0.0f, 0.0f, 1.0f};
#if defined(VMDUSEEVENTIO)
    if ((hmd && eviodev) || (hmd && camera_projection == RT_OCULUS_RIFT)) {
#else    
    if (hmd && camera_projection == RT_OCULUS_RIFT) {
#endif
      float hmd_U_orig[3] = {1.0f, 0.0f, 0.0f};
      float hmd_V_orig[3] = {0.0f, 1.0f, 0.0f};
      float hmd_W_orig[3] = {0.0f, 0.0f, 1.0f};

      // query the HMD head pose as late as possible to reduce latency 
      // between the sensor reads and presentation on the display
      hmd->update();
      hmd->rot_basis_quat(hmd_U_new, hmd_V_new, hmd_W_new, 
                          hmd_U_orig, hmd_V_orig, hmd_W_orig);

      // We use the HMD pose quaternion to transform the standard camera
      // orientation basis vectors to their new orientation, and then we
      // project those onto the current view basis vector to get the 
      // correct final camera view vector that we pass along to OptiX
      float hmdtmp[3];
      memset(hmdtmp, 0, sizeof(hmdtmp));
      vec_scaled_add(hmdtmp, hmd_U_new[0], cam_U);
      vec_scaled_add(hmdtmp, hmd_U_new[1], cam_V);
      vec_scaled_add(hmdtmp, hmd_U_new[2], cam_W);
      vec_copy(hmd_U_new, hmdtmp);

      memset(hmdtmp, 0, sizeof(hmdtmp));
      vec_scaled_add(hmdtmp, hmd_V_new[0], cam_U);
      vec_scaled_add(hmdtmp, hmd_V_new[1], cam_V);
      vec_scaled_add(hmdtmp, hmd_V_new[2], cam_W);
      vec_copy(hmd_V_new, hmdtmp);
 
      memset(hmdtmp, 0, sizeof(hmdtmp));
      vec_scaled_add(hmdtmp, hmd_W_new[0], cam_U);
      vec_scaled_add(hmdtmp, hmd_W_new[1], cam_V);
      vec_scaled_add(hmdtmp, hmd_W_new[2], cam_W);
      vec_copy(hmd_W_new, hmdtmp);

#if 0
      float q[4];
      hmd->get_rot_quat(q);
      printf("\nQ: %f %f %f %f\n", q[0], q[1], q[2], q[3]);
      printf("hmd_U: %.1f %.1f %.1f\n", hmd_U[0], hmd_U[1], hmd_U[2]);
      printf("hmd_V: %.1f %.1f %.1f\n", hmd_V[0], hmd_V[1], hmd_V[2]);
      printf("hmd_W: %.1f %.1f %.1f\n", hmd_W[0], hmd_W[1], hmd_W[2]);
#endif

      if (hmd && camera_projection == RT_OCULUS_RIFT) {
        vec_copy(hmd_U, hmd_U_new);
        vec_copy(hmd_V, hmd_V_new);
        vec_copy(hmd_W, hmd_W_new);

        // when using an HMD in direct-drive mode, we have to do 
        // a redraw on every rendering pass.
        winredraw = 1;
      }
    }
#endif


#if defined(VMDUSEEVENTIO)
    //
    // Handle Linux event-based input device I/O for joysticks/spaceball/etc
    //
    if (eviodev) {
      float ax1, ay1, ax2, ay2;
      int buttons;
      int rc=0;
      rc = evio_get_joystick_status(eviodev, &ax1, &ay1, &ax2, &ay2, &buttons);

      if (buttons) {
        printf("Joystick: %5.2f %5.2f  %5.2f %5.2f  0x%08x  \n",
               ax1, ay1, ax2, ay2, buttons);
      }

      float tx = ax1 + ax2;
      float ty = ay1;
      float tz = ay2;

      // check for translation and handle it...
      if (fabsf(tx) > 0.03 || fabsf(ty) > 0.03 || fabsf(tz) > 0.03) { 
        tx *= -500;
        ty *=  500;
        tz *=  500;

        // Re-use the HMD head pose info obtained above to apply 
        // head motion translations from joysticks or other controllers
        float zoommod = 2.0f*cam_zoom/cam_zoom_orig;
        float divlen = sqrtf(wsx*wsx + wsy*wsy) * 50;
        float dU[3], dV[3], dW[3];
        vec_scale(dU, -tx * zoommod / divlen, hmd_U_new);
        vec_scale(dV, -ty * zoommod / divlen, hmd_V_new);
        vec_scale(dW, -tz * zoommod / divlen, hmd_W_new);
        vec_add(cam_pos, cam_pos, dU);
        vec_add(cam_pos, cam_pos, dV);
        vec_add(cam_pos, cam_pos, dW);
        winredraw = 1;
      }

    }
#endif


    //
    // handle window resizing, stereoscopic mode changes,
    // destroy and recreate affected OptiX buffers
    //
    int resize_buffers=0;

#if defined(VMDOPTIX_USE_HMD)
    // when using spheremaps, we trigger redraw ops but we do not change
    // the spheremap image size 
    if (hmd_freerun) {
      if (wsx != owsx || wsy != owsy)
        winredraw=1;
    } 
    else
#endif
    {
      // only process image/window resizing when not drawing spheremaps
      if (wsx != width) {
        width = wsx;
        resize_buffers=1;
      }
 
      if (wsy != height || (stereoon != stereoon_old)) {
        if (stereoon) {
          if (height != wsy * 2) {
            height = wsy * 2; 
            resize_buffers=1;
          }
        } else {
          height = wsy;
          resize_buffers=1;
        }
      }
    }


// XXX Prior to OptiX 3.8, we had to manually stop progressive 
//     mode before changing any OptiX state.
#if defined(VMDOPTIX_PROGRESSIVEAPI) && OPTIX_VERSION < 3080
    // 
    // Check for all conditions that would require modifying OptiX state
    // and tell the VCA to stop progressive rendering before we modify 
    // the rendering state, 
    //
    if (done || winredraw || resize_buffers ||
        (stereoon != stereoon_old) || (gl_dof_on != gl_dof_on_old)) {
      // need to issue stop command before editing optix objects
      if (vcarunning) {
        rtContextStopProgressive(ctx);
        vcarunning=0;
      }
    }
#endif

    // check if stereo mode or DoF mode changed, both cases
    // require changing the active color accumulation ray gen program
    if ((stereoon != stereoon_old) || (gl_dof_on != gl_dof_on_old)) {
      // when stereo mode changes, we have to regenerate the
      // the RNG, accumulation buffer, and framebuffer
      if (stereoon != stereoon_old) {
        resize_buffers=1;
      }

      // update stereo and DoF state
      stereoon_old = stereoon;
      gl_dof_on_old = gl_dof_on;

      // set the active color accumulation ray gen program based on the 
      // camera/projection mode, stereoscopic display mode, 
      // and depth-of-field state
      set_accum_raygen_pgm(camera_projection, stereoon, gl_dof_on);
    }

    if (resize_buffers) {
      resize_framebuffer(width, height);

      // when movie recording is enabled, print the window size as a guide
      // since the user might want to precisely control the size or 
      // aspect ratio for a particular movie format, e.g. 1080p, 4:3, 16:9
      if (movie_recording_enabled) {
        printf("\rOptiXRenderer) Window resize: %d x %d                               \n", width, height);
      }

      winredraw=1;
    }

    int frame_ready = 1; // Default to true for the non-VCA case
    unsigned int subframe_count = 1;
    if (!done) {
#if defined(VMDOPTIX_USE_HMD)
      // update HMD lens distortion correction mesh and/or warp coefficients 
      if (hmd_warp && (winredraw || hmd_warp_coeff_update)) {
        glwin_spheremap_update_hmd_warp(win, hmd_warp, wsx, wsy, 21,
                                        width, height, hmd_warp_coeff,
                                        hmd_warp_coeff_update);
        hmd_warp_coeff_update=0;
      }
#endif

      //
      // If the user interacted with the window in a meaningful way, we
      // need to update the OptiX rendering state, recompile and re-validate
      // the context, and then re-render...
      //
      if (winredraw) {
        // update camera parameters
        RTERR( rtVariableSet1f( cam_zoom_v, cam_zoom) );
        RTERR( rtVariableSet3fv( cam_pos_v, cam_pos) );
        RTERR( rtVariableSet3fv(   cam_U_v, hmd_U) );
        RTERR( rtVariableSet3fv(   cam_V_v, hmd_V) );
        RTERR( rtVariableSet3fv(   cam_W_v, hmd_W) );
        RTERR( rtVariableSet3fv(scene_gradient_v, scene_gradient) );
 
        // update shadow state 
        RTERR( rtVariableSet1i(shadows_enabled_v, gl_shadows_on) );

        // update depth cueing state
        RTERR( rtVariableSet1i(fog_mode_v, 
                 (int) (gl_fog_on) ? fog_mode : RT_FOG_NONE) );

        // update clipping sphere state
        RTERR( rtVariableSet1i(clipview_mode_v, 
                 (int) (gl_clip_on) ? clipview_mode : RT_CLIP_NONE) );

        // update headlight state 
        RTERR( rtVariableSet1i(headlight_mode_v, 
                 (int) (gl_headlight_on) ? RT_HEADLIGHT_ON : RT_HEADLIGHT_OFF) );
 
        // update/recompute DoF values 
        RTERR( rtVariableSet1f(cam_dof_focal_dist_v, cam_dof_focal_dist) );
        RTERR( rtVariableSet1f(cam_dof_aperture_rad_v, cam_dof_focal_dist / (2.0f * cam_zoom * cam_dof_fnumber)) );

        //
        // Update light directions in the OptiX light buffer or user object.
        // Only update when xformlights is set, otherwise we take a 
        // speed hit when using a remote VCA cluster for rendering.
        //
        // We only transform directional lights, since our positional lights
        // are normally affixed to the model coordinate system rather than 
        // the camera.
        //
        if (xformlights) {
#if defined(VMDOPTIX_LIGHTUSEROBJS)
          DirectionalLightList dlights;
          memset(&dlights, 0, sizeof(DirectionalLightList) );
          dlights.num_lights = directional_lights.num();
          int dlcount = directional_lights.num();
          dlcount = (dlcount > DISP_LIGHTS) ? DISP_LIGHTS : dlcount;
          for (i=0; i<dlcount; i++) {
            //vec_copy( (float*)( &lights.dirs[i] ), cur_dlights[i].dir );
            dlights.dirs[i] = cur_dlights[i].dir;
          }
          RTERR( rtVariableSetUserData(dir_light_list_v, sizeof(DirectionalLightList), &dlights) );
#else
          DirectionalLight *dlbuf;
          RTERR( rtBufferMap(dir_lightbuffer, (void **) &dlbuf) );
          for (i=0; i<directional_lights.num(); i++) {
            vec_copy((float*)&dlbuf[i].dir, (float*)&cur_dlights[i].dir);
          }
          RTERR( rtBufferUnmap(dir_lightbuffer) );
#endif
        }

        // reset accumulation buffer 
        accum_count=0;
        totalsamplecount=0;


        // 
        // Sample count updates and OptiX state must always remain in 
        // sync, so if we only update sample count state during redraw events,
        // that's the only time we should recompute the sample counts, since
        // they also affect normalization factors for the accumulation buffer
        // in the non-VCA case.
        //

        // Update sample counts to achieve target interactivity
        if (autosamplecount) {
          if (fpsexpave > 37)
            samples_per_pass++;
          else if (fpsexpave < 30) 
            samples_per_pass--;
    
          // clamp sample counts to a "safe" range
          if (samples_per_pass > 14)
            samples_per_pass=14;
          if (samples_per_pass < 1)
            samples_per_pass=1;
        } 

        // split samples per pass either among AA and AO, depending on
        // whether DoF and AO are enabled or not. 
        if (force_ao_1) {
          cur_aa_samples = samples_per_pass;
          cur_ao_samples = 1;
        } else if (gl_shadows_on && gl_ao_on) {
          if (gl_dof_on) {
            if (samples_per_pass < 4) {
              cur_aa_samples=samples_per_pass;
              cur_ao_samples=1;
            } else {
              int s = (int) sqrtf(samples_per_pass);
              cur_aa_samples=s;
              cur_ao_samples=s;
            }
          } else {
            cur_aa_samples=1;
            cur_ao_samples=samples_per_pass;
          }
        } else {
          cur_aa_samples=samples_per_pass;
          cur_ao_samples=0;
        }

        // update the current AA/AO sample counts since they may be changing if
        // FPS autotuning is enabled...
        RTERR( rtVariableSet1i(aa_samples_v, cur_aa_samples) );

        // observe latest AO enable/disable flag, and sample count
        if (gl_shadows_on && gl_ao_on) {
          RTERR( rtVariableSet1i(ao_samples_v, cur_ao_samples) );
        } else {
          cur_ao_samples = 0;
          RTERR( rtVariableSet1i(ao_samples_v, 0) );
        }

#ifdef VMDOPTIX_PROGRESSIVEAPI
        RTERR( rtVariableSet1f(accum_norm_v, 1.0f / float(cur_aa_samples)) );
#endif

        // updated cached copy of previous window dimensions so we can
        // trigger updates on HMD spheremaps and FBOs as necessary
        owsx = wsx;
        owsy = wsy;
      } 


      //
      // The non-VCA code path must handle the accumulation buffer 
      // for itself, correctly rescaling the accumulated samples when
      // drawing to the output framebuffer.  
      //
      // The VCA code path takes care of normalization for itself.
      //
#ifndef VMDOPTIX_PROGRESSIVEAPI
      // The accumulation buffer normalization factor must be updated
      // to reflect the total accumulation count before the accumulation
      // buffer is drawn to the output framebuffer
      RTERR( rtVariableSet1f(accum_norm_v, 1.0f / float(cur_aa_samples + accum_count)) );

      // The accumulation buffer subframe index must be updated to ensure that
      // the RNGs for AA and AO get correctly re-seeded
      RTERR( rtVariableSet1ui(accum_count_v, accum_count) );

      // Force context compilation/validation
      // If no state has changed, there's no need to recompile/validate.
      // This call can be omitted since OptiX will do this automatically
      // at the next rtContextLaunchXXX() call.
//      render_compile_and_validate();
#endif


      //
      // run the renderer 
      //
      frame_ready = 1; // Default to true for the non-VCA case
      subframe_count = 1;
      if (lasterror == RT_SUCCESS) {
        if (winredraw) {
#ifdef VMDOPTIX_PROGRESSIVEAPI
          // start the VCA doing progressive rendering...
          RTERR( rtContextLaunchProgressive2D(ctx, RT_RAY_GEN_ACCUMULATE, width, height, 0) );
          vcarunning=1;
#else
          RTERR( rtContextLaunch2D(ctx, RT_RAY_GEN_CLEAR_ACCUMULATION_BUFFER, width, height) );
#endif
#if defined(ORT_RAYSTATS)
          RTERR( rtContextLaunch2D(ctx, RT_RAY_GEN_CLEAR_RAYSTATS, width, height) );
#endif
          winredraw=0;
        }

#ifdef VMDOPTIX_PROGRESSIVEAPI
        // Wait for the next frame to arrive
        RTERR( rtBufferGetProgressiveUpdateReady(framebuffer, &frame_ready, &subframe_count, 0) );
        if (frame_ready) 
          totalsamplecount = subframe_count * samples_per_pass;
#else
        // iterate, adding to the accumulation buffer...
        RTERR( rtContextLaunch2D(ctx, RT_RAY_GEN_ACCUMULATE, width, height) );
        subframe_count++; // increment subframe index
        totalsamplecount += samples_per_pass;
        accum_count += cur_aa_samples;

        // copy the accumulation buffer image data to the framebuffer and
        // perform type conversion and normaliztion on the image data...
        RTERR( rtContextLaunch2D(ctx, RT_RAY_GEN_COPY_FINISH, width, height) );
#endif

        if (lasterror == RT_SUCCESS) {
          if (frame_ready || hmd_freerun) {
            double bufnewtime = wkf_timer_timenow(ort_timer);


            // if an HMD is connected and one of the panoramic image formats
            // is in use, we use the appopriate OpenGL HMD viewer for the 
            // panoramic image format that's currently active, otherwise the
            // image is displayed as-is in a window.
#if defined(VMDOPTIX_USE_HMD)
            if (hmd_freerun) {
              double hmdnewtime = wkf_timer_timenow(ort_timer);
              double hmdframetime = (hmdnewtime-hmdoldtime) + 0.00001f;
              hmdoldtime=hmdnewtime;

              // compute exponential moving average for exp(-1/10)
              double hmdframefps = 1.0f/hmdframetime;
              hmdfpsexpave = (hmdfpsexpave * 0.90) + (hmdframefps * 0.10);

              float hmdquat[4];
              if (hmd_no_draw == 0) {
                int hmd_warp_on = (hmd_warp_drawmode!=0);
                int hmd_warp_lines = (hmd_warp_drawmode==2 || hmd_warp_drawmode==4);
                int hmd_chroma_on = (hmd_warp_drawmode==1 || hmd_warp_drawmode==2);

                // update when frame is ready, or when tex caching is disabled
                if (frame_ready || (!hmd_tex_caching)) {
                  // display output image
                  const unsigned char * img;
                  rtBufferMap(framebuffer, (void **) &img);

                  // minimize impact of OptiX buffer map and tex update steps
                  if (hmd_interleave_draws) {
                    // query HMD sensors immediately prior to draw...
                    hmd->get_rot_quat(hmdquat, 1);
                    if (hmd_warp && hmd_warp_drawmode != 0) {
                      glwin_spheremap_draw_hmd_warp(win, hmd_warp, 
                          hmd_warp_on, hmd_warp_lines, hmd_chroma_on,
                          wsx, wsy, width, height, hmdquat, hmd_fov, 15.0f, hmd_spres);
                    } else {
                      glwin_spheremap_draw_tex(win, GLWIN_STEREO_OVERUNDER, width, height, hmdquat, hmd_fov, 15.0f, hmd_spres);
                    }
                    glwin_swap_buffers(win);
                  }

                  glwin_spheremap_upload_tex_rgb3u(win, width, height, img);

                  // minimize impact of OptiX buffer map and tex update steps
                  if (hmd_interleave_draws) {
                    // query HMD sensors immediately prior to draw...
                    hmd->get_rot_quat(hmdquat, 1);
                    if (hmd_warp && hmd_warp_drawmode != 0) {
                      glwin_spheremap_draw_hmd_warp(win, hmd_warp, 
                          hmd_warp_on, hmd_warp_lines, hmd_chroma_on,
                          wsx, wsy, width, height, hmdquat, hmd_fov, 15.0f, hmd_spres);
                    } else {
                      glwin_spheremap_draw_tex(win, GLWIN_STEREO_OVERUNDER, width, height, hmdquat, hmd_fov, 15.0f, hmd_spres);
                    }
                    glwin_swap_buffers(win);
                  }
 
                  rtBufferUnmap(framebuffer);
                  mapbuftotaltime = wkf_timer_timenow(ort_timer) - bufnewtime;
                }

                // query HMD sensors immediately prior to draw...
                hmd->get_rot_quat(hmdquat, 1);
                if (hmd_warp && hmd_warp_drawmode != 0) {
                  glwin_spheremap_draw_hmd_warp(win, hmd_warp, 
                      hmd_warp_on, hmd_warp_lines, hmd_chroma_on,
                      wsx, wsy, width, height, hmdquat, hmd_fov, 15.0f, hmd_spres);
                } else {
                  glwin_spheremap_draw_tex(win, GLWIN_STEREO_OVERUNDER, width, height, hmdquat, hmd_fov, 15.0f, hmd_spres);
                }
                glwin_swap_buffers(win);
              }

              hmdgldrawtime = wkf_timer_timenow(ort_timer) - hmdnewtime;
            } else {
#endif
              // display output image
              const unsigned char * img;
              rtBufferMap(framebuffer, (void **) &img);

#if 0
              glwin_draw_image_tex_rgb3u(win, (stereoon!=0)*GLWIN_STEREO_OVERUNDER, width, height, img);
#else
              glwin_draw_image_rgb3u(win, (stereoon!=0)*GLWIN_STEREO_OVERUNDER, width, height, img);
#endif

#if 1
              // push latest frame into the video streaming pipeline
              // and pump the event handling mechanism afterwards
              if (app->uivs && app->uivs->srv_connected()) {
                app->uivs->video_frame_pending(img, width, height);             
                app->uivs->check_event();             
              }              
#endif

              rtBufferUnmap(framebuffer);
              mapbuftotaltime = wkf_timer_timenow(ort_timer) - bufnewtime;

#if defined(VMDOPTIX_USE_HMD)
            }
#endif


            // if live movie recording is on, we save every displayed frame
            // to a sequence sequence of image files, with each file numbered
            // by its frame index, which is computed by the multiplying image 
            // presentation time by the image sequence fixed-rate-FPS value.
            if (movie_recording_enabled && movie_recording_on) {
              char moviefilename[2048];

              // compute frame number from wall clock time and the
              // current fixed-rate movie playback frame rate
              double now = wkf_timer_timenow(ort_timer);
              double frametime = now - movie_recording_start_time;
              int fidx = frametime * movie_recording_fps;

              // always force the first recorded frame to be 0
              if (movie_framecount==0)
                fidx=0;
              movie_framecount++;

#if defined(__linux)
              // generate symlinks for frame indices between the last written
              // frame and the current one so that video encoders such as
              // ffmpeg and mencoder can be fed the contiguous frame sequence
              // at a fixed frame rate, as they require
              sprintf(moviefilename, movie_recording_filebase, 
                      movie_lastframeindex);
              int symidx;
              for (symidx=movie_lastframeindex; symidx<fidx; symidx++) {
                char symlinkfilename[2048];
                sprintf(symlinkfilename, movie_recording_filebase, symidx);
                symlink(moviefilename, symlinkfilename);
              }
#endif
               
              // write the new movie frame               
              sprintf(moviefilename, movie_recording_filebase, fidx);
              if (OptiXWriteImage(moviefilename, writealpha, framebuffer,
                                  RT_FORMAT_UNSIGNED_BYTE4, width, height) == -1) {
                movie_recording_on = 0;
                printf("\n");
                printf("OptiXRenderer) ERROR during writing image during movie recording!\n");
                printf("OptiXRenderer) Movie recording STOPPED\n");
              }

              movie_lastframeindex = fidx; // update last frame index written
            }
          }
        } else {
          printf("OptiXRenderer) An error occured during rendering. Rendering is aborted.\n");
          done=1;
          break;
        }
      } else {
        printf("OptiXRenderer) An error occured in AS generation. Rendering is aborted.\n");
        done=1;
        break;
      }
    }

    if (!done && frame_ready) {
      double newtime = wkf_timer_timenow(ort_timer);
      double frametime = (newtime-oldtime) + 0.00001f;
      oldtime=newtime;

      // compute exponential moving average for exp(-1/10)
      double framefps = 1.0f/frametime;
      fpsexpave = (fpsexpave * 0.90) + (framefps * 0.10);

      if (hmd_freerun) {
        printf("OptiXRenderer) %c AA%2d AO%2d %5d tot RT FPS %.1f HMD FPS %.0f GL%.4f MB%.4f \r",
               statestr[state], cur_aa_samples, cur_ao_samples, 
               totalsamplecount, fpsexpave, 
               hmdfpsexpave, hmdgldrawtime, mapbuftotaltime);
      } else {
        printf("OptiXRenderer) %c AA:%2d AO:%2d, %4d tot RT FPS: %.1f  %.4f s/frame sf: %d  \r",
               statestr[state], cur_aa_samples, cur_ao_samples, 
               totalsamplecount, fpsexpave, frametime, subframe_count);
      }

      fflush(stdout);
      state = (state+1) & 3;
    }

  } // end of per-cycle event processing

  printf("\n");

  // write the output image upon exit...
  if (lasterror == RT_SUCCESS) {
    wkf_timer_start(ort_timer);
    // write output image
    OptiXWriteImage(filename, writealpha, framebuffer);
#if defined(ORT_RAYSTATS)
    OptiXPrintRayStats(raystats1_buffer, raystats2_buffer, 0.0);
#endif

    wkf_timer_stop(ort_timer);

    if (verbose == RT_VERB_TIMING || verbose == RT_VERB_DEBUG) {
      printf("OptiXRenderer) image file I/O time: %f secs\n", wkf_timer_time(ort_timer));
    }
  }

#if defined(VMDOPTIX_USE_HMD)
  // if using HMD for display, get head tracking data
  if (hmd) {
    delete hmd;
    hmd = NULL;
  }

  if (hmd_warp != NULL) {
    glwin_spheremap_destroy_hmd_warp(win, hmd_warp);
  }
#endif

#if defined(VMDUSEEVENTIO)
  if (eviodev) {
    evio_close(eviodev);
  }
#endif

  glwin_destroy(win);
}

#endif



void OptiXRenderer::render_to_videostream(const char *filename, int writealpha) {
  int i;

  if (!context_created)
    return;

  // flags to interactively enable/disable shadows, AO, DoF
#if defined(USE_REVERSE_SHADOW_RAYS) && defined(USE_REVERSE_SHADOW_RAYS_DEFAULT)
  int gl_shadows_on=(shadows_enabled) ? RT_SHADOWS_ON_REVERSE : RT_SHADOWS_OFF;
#else
  int gl_shadows_on=(shadows_enabled) ? RT_SHADOWS_ON : RT_SHADOWS_OFF;
#endif

  int gl_fs_on=0; // fullscreen window state
  int fsowsx=0, fsowsy=0; // store last win size before fullscreen
  int owsx=0, owsy=0; // last win size
  int gl_ao_on=(ao_samples > 0);
  int gl_dof_on, gl_dof_on_old;
  gl_dof_on=gl_dof_on_old=dof_enabled; 
  int gl_fog_on=(fog_mode != RT_FOG_NONE);
  int gl_clip_on=(clipview_mode != RT_CLIP_NONE);
  int gl_headlight_on=(headlight_mode != RT_HEADLIGHT_OFF);

  // Enable live recording of a session to a stream of image files indexed
  // by their display presentation time, mapped to the nearest frame index
  // in a fixed-frame-rate image sequence (e.g. 24, 30, or 60 FPS), 
  // to allow subsequent encoding into a standard movie format.
  // XXX this feature is disabled by default at present, to prevent people
  //     from accidentally turning it on during a live demo or the like
  int movie_recording_enabled = (getenv("VMDOPTIXLIVEMOVIECAPTURE") != NULL);
  int movie_recording_on = 0;
  double movie_recording_start_time = 0.0;
  int movie_recording_fps = 30;
  int movie_framecount = 0;
  int movie_lastframeindex = 0;
  const char *movie_recording_filebase = "vmdlivemovie.%05d.tga";
  if (getenv("VMDOPTIXLIVEMOVIECAPTUREFILEBASE"))
    movie_recording_filebase = getenv("VMDOPTIXLIVEMOVIECAPTUREFILEBASE");

  // total AA/AO sample count
  int totalsamplecount=0;

  // counter for snapshots of live image...
  int snapshotcount=0;

  // flag to enable automatic AO sample count adjustment for FPS rate control
#if defined(VMDOPTIX_PROGRESSIVEAPI)
  int vcarunning=0;      // flag to indicate VCA streaming active/inactive
#if 1
  int autosamplecount=0; // leave disabled for now
#else
  int autosamplecount=1; // works partially in current revs of progressive API
  if (remote_device != NULL)
    autosamplecount=0;  // re-disable when targeting VCA for now
#endif
#else
  int autosamplecount=1;
#endif

  // flag to enable transformation of lights and gradient sky sphere, 
  // so that they track camera orientation as they do in the VMD OpenGL display
  int xformlights=1, xformgradientsphere=1;

  //
  // allocate or reconfigure the framebuffer, accumulation buffer, 
  // and output streams required for progressive rendering, either
  // using the new progressive APIs, or using our own code.
  //
  // Unless overridden by environment variables, we use the incoming
  // window size parameters from VMD to initialize the RT image dimensions.
  // If image size is overridden, often when using HMDs, the incoming 
  // dims are window dims are used to size the GL window, but the image size
  // is set independently.
  int wsx=width, wsy=height;
  const char *imageszstr = getenv("VMDOPTIXIMAGESIZE");
  if (imageszstr) {
    if (sscanf(imageszstr, "%d %d", &width, &height) != 2) {
      width=wsx;
      height=wsy;
    } 
  } 

  config_framebuffer(width, height, 1);

  // prepare the majority of OptiX rendering state before we go into 
  // the interactive rendering loop
  update_rendering_state(1);
  render_compile_and_validate();

  // make a copy of state we're going to interactively manipulate,
  // so that we can recover to the original state on-demand
  int samples_per_pass = 1;
  int force_ao_1 = 0; // whether or not to force AO count per pass to 1
  int cur_aa_samples = aa_samples;
  int cur_ao_samples = ao_samples;
  float cam_zoom_orig = cam_zoom;
  float scene_gradient_orig[3] = {0.0f, 1.0f, 0.0f};
  vec_copy(scene_gradient_orig, scene_gradient);

  float cam_pos_orig[3] = {0.0f, 0.0f, 2.0f};
  float cam_U_orig[3] = {1.0f, 0.0f, 0.0f};
  float cam_V_orig[3] = {0.0f, 1.0f, 0.0f};
  float cam_W_orig[3] = {0.0f, 0.0f, -1.0f};
  float cam_pos[3], cam_U[3], cam_V[3], cam_W[3];
  float hmd_U[3], hmd_V[3], hmd_W[3];

  vec_copy(cam_pos, cam_pos_orig);
  vec_copy(cam_U, cam_U_orig);
  vec_copy(cam_V, cam_V_orig);
  vec_copy(cam_W, cam_W_orig);

  // copy light directions
  DirectionalLight *cur_dlights = (DirectionalLight *) calloc(1, directional_lights.num() * sizeof(DirectionalLight));
  for (i=0; i<directional_lights.num(); i++) {
    vec_copy((float*)&cur_dlights[i].dir, directional_lights[i].dir);
    vec_normalize((float*)&cur_dlights[i].dir);
  }

  // check for stereo-capable display
  int havestereo=0, havestencil=0;
  int stereoon=0, stereoon_old=0;
  // glwin_get_wininfo(win, &havestereo, &havestencil);

  // Override AA/AO sample counts since we're doing progressive rendering.
  // Choosing an initial AO sample count of 1 will give us the peak progressive 
  // display update rate, but we end up wasting time on re-tracing many
  // primary rays.  The automatic FPS optimization scheme below will update
  // the number of samples per rendering pass and assign the best values for
  // AA/AO samples accordingly.
  cur_aa_samples = samples_per_pass;
  if (cur_ao_samples > 0) {
    cur_aa_samples = 1;
    cur_ao_samples = samples_per_pass;
  }

  const char *statestr = "|/-\\.";
  int done=0, winredraw=1, accum_count=0;
  int state=0, mousedownx=0, mousedowny=0;
  float cur_cam_zoom = cam_zoom_orig;

  double fpsexpave=0.0; 
  double mapbuftotaltime=0.0;
  
  double oldtime = wkf_timer_timenow(ort_timer);
  // Note: we immediately terminate the rendering loop if
  //       the videostream server loses its client connection(s)
  while (!done &&
         app->uivs && app->uivs->srv_connected()) { 
    int winevent=0;

#if 1
    if (app->uivs && app->uivs->srv_connected()) {
      if (app->uivs->srv_check_ui_event()) {
        int eventtype;
        app->uivs->srv_get_last_event_type(eventtype);
        switch (eventtype) {
          case VideoStream::VS_EV_ROTATE_BY:
            { int axis;
              float angle;
              app->uivs->srv_get_last_rotate_by(angle, axis);
              Matrix4 rm;

              switch (axis) {
                case 'x':
                  rm.rotate_axis(cam_U, -angle * M_PI/180.0f);
                  break;

                case 'y':
                  rm.rotate_axis(cam_V, -angle * M_PI/180.0f);
                  break;

                case 'z':
                  rm.rotate_axis(cam_W, -angle * M_PI/180.0f);
                  break;
              }
              rm.multpoint3d(cam_pos, cam_pos);
              rm.multnorm3d(cam_U, cam_U);
              rm.multnorm3d(cam_V, cam_V);
              rm.multnorm3d(cam_W, cam_W);

              if (xformgradientsphere) {
                rm.multnorm3d(scene_gradient, scene_gradient);
              }

              if (xformlights) {
                // update light directions (comparatively costly)
                for (i=0; i<directional_lights.num(); i++) {
                  rm.multnorm3d((float*)&cur_dlights[i].dir, (float*)&cur_dlights[i].dir);
                }
              }
              winredraw = 1;
            }
            break;

          case VideoStream::VS_EV_TRANSLATE_BY:
            {
              float dU[3], dV[3], dW[3];
              float tx, ty, tz;
              app->uivs->srv_get_last_translate_by(tx, ty, tz);
              vec_scale(dU, -tx, cam_U);
              vec_scale(dV, -ty, cam_V);
              vec_scale(dW, -tz, cam_W);
              vec_add(cam_pos, cam_pos, dU);
              vec_add(cam_pos, cam_pos, dV);
              vec_add(cam_pos, cam_pos, dW);
              winredraw = 1;
            }
            break;

          case VideoStream::VS_EV_SCALE_BY:
            { float zoominc;
              app->uivs->srv_get_last_scale_by(zoominc);
              cam_zoom *= 1.0f / zoominc;
              winredraw = 1;
            }
            break;

#if 0
              } else if (mm == RTMM_DOF) {
                cam_dof_fnumber += txdx * 20.0f;
                if (cam_dof_fnumber < 1.0f) cam_dof_fnumber = 1.0f;
                cam_dof_focal_dist += -txdy;
                if (cam_dof_focal_dist < 0.01f) cam_dof_focal_dist = 0.01f;
                winredraw = 1;
              }
            }
#endif
          case VideoStream::VS_EV_KEYBOARD:
            { int keydev, keyval, shift_state;
              app->uivs->srv_get_last_keyboard(keydev, keyval, shift_state);
              switch (keydev) {
                case DisplayDevice::WIN_KBD:
                  {
                    switch (keyval) {
                      // update sample counts
                      case  '1': autosamplecount=0; samples_per_pass=1; winredraw=1; break;
                      case  '2': autosamplecount=0; samples_per_pass=2; winredraw=1; break;
                      case  '3': autosamplecount=0; samples_per_pass=3; winredraw=1; break;
                      case  '4': autosamplecount=0; samples_per_pass=4; winredraw=1; break;
                      case  '5': autosamplecount=0; samples_per_pass=5; winredraw=1; break;
                      case  '6': autosamplecount=0; samples_per_pass=6; winredraw=1; break;
                      case  '7': autosamplecount=0; samples_per_pass=7; winredraw=1; break;
                      case  '8': autosamplecount=0; samples_per_pass=8; winredraw=1; break;
                      case  '9': autosamplecount=0; samples_per_pass=9; winredraw=1; break;
                      case  '0': autosamplecount=0; samples_per_pass=10; winredraw=1; break;

                      case  '=': /* recover back to initial state */
                        vec_copy(scene_gradient, scene_gradient_orig);
                        cam_zoom = cam_zoom_orig;
                        vec_copy(cam_pos, cam_pos_orig);
                        vec_copy(cam_U, cam_U_orig);
                        vec_copy(cam_V, cam_V_orig);
                        vec_copy(cam_W, cam_W_orig);

                        // restore original light directions
                        for (i=0; i<directional_lights.num(); i++) {
                          vec_copy((float*)&cur_dlights[i].dir, directional_lights[i].dir);
                          vec_normalize((float*)&cur_dlights[i].dir);
                        }
                        winredraw = 1;
                        break;

                      case  'q': /* 'q' key */
                      case  'Q': /* 'Q' key */
                        printf("\nOptiXRenderer) Exiting on user input.               \n");
                        done=1; /* exit from interactive RT window */
                        break;
                    }
                  }
                  break;

                case DisplayDevice::WIN_KBD_ESCAPE:
                  printf("\nOptiXRenderer) Exiting on user input.               \n");
                  done=1; /* exit from interactive RT window */
                  break;

                case DisplayDevice::WIN_KBD_F1:
#if defined(USE_REVERSE_SHADOW_RAYS) && defined(USE_REVERSE_SHADOW_RAYS_DEFAULT)
                  gl_shadows_on=(!gl_shadows_on) ? RT_SHADOWS_ON_REVERSE : RT_SHADOWS_OFF;
#else
                  gl_shadows_on=(!gl_shadows_on) ? RT_SHADOWS_ON : RT_SHADOWS_OFF;
                  // gl_shadows_on = (!gl_shadows_on);
#endif

                  printf("\n");
#if defined(USE_REVERSE_SHADOW_RAYS) && defined(USE_REVERSE_SHADOW_RAYS_DEFAULT)
                  printf("OptiXRenderer) Shadows %s\n",
                         (gl_shadows_on) ? "enabled (reversal opt.)" : "disabled");
#else
                  printf("OptiXRenderer) Shadows %s\n",
                         (gl_shadows_on) ? "enabled" : "disabled");
#endif
                  winredraw = 1;
                  break;

                case DisplayDevice::WIN_KBD_F2:
                  gl_ao_on = (!gl_ao_on);
                  printf("\n");
                  printf("OptiXRenderer) Ambient occlusion %s\n",
                         (gl_ao_on) ? "enabled" : "disabled");
                  winredraw = 1;
                  break;

                case DisplayDevice::WIN_KBD_F3:
                  gl_dof_on = (!gl_dof_on);
                  printf("\n");
                  printf("OptiXRenderer) Depth-of-field %s\n",
                         (gl_dof_on) ? "enabled" : "disabled");
                  winredraw = 1;
                  break;
              }
            }
            break;

          case VideoStream::VS_EV_NONE: 
                  default:
            // should never happen...
            break;
        }
      }
    }              
#endif

    // if there is no HMD, we use the camera orientation directly  
    vec_copy(hmd_U, cam_U);
    vec_copy(hmd_V, cam_V);
    vec_copy(hmd_W, cam_W);

    //
    // handle window resizing, stereoscopic mode changes,
    // destroy and recreate affected OptiX buffers
    //
    int resize_buffers=0;

    {
      // only process image/window resizing when not drawing spheremaps
      if (wsx != width) {
        width = wsx;
        resize_buffers=1;
      }
 
      if (wsy != height || (stereoon != stereoon_old)) {
        if (stereoon) {
          if (height != wsy * 2) {
            height = wsy * 2; 
            resize_buffers=1;
          }
        } else {
          height = wsy;
          resize_buffers=1;
        }
      }
    }


// XXX Prior to OptiX 3.8, we had to manually stop progressive 
//     mode before changing any OptiX state.
#if defined(VMDOPTIX_PROGRESSIVEAPI) && OPTIX_VERSION < 3080
    // 
    // Check for all conditions that would require modifying OptiX state
    // and tell the VCA to stop progressive rendering before we modify 
    // the rendering state, 
    //
    if (done || winredraw || resize_buffers ||
        (stereoon != stereoon_old) || (gl_dof_on != gl_dof_on_old)) {
      // need to issue stop command before editing optix objects
      if (vcarunning) {
        rtContextStopProgressive(ctx);
        vcarunning=0;
      }
    }
#endif

    // check if stereo mode or DoF mode changed, both cases
    // require changing the active color accumulation ray gen program
    if ((stereoon != stereoon_old) || (gl_dof_on != gl_dof_on_old)) {
      // when stereo mode changes, we have to regenerate the
      // the RNG, accumulation buffer, and framebuffer
      if (stereoon != stereoon_old) {
        resize_buffers=1;
      }

      // update stereo and DoF state
      stereoon_old = stereoon;
      gl_dof_on_old = gl_dof_on;

      // set the active color accumulation ray gen program based on the 
      // camera/projection mode, stereoscopic display mode, 
      // and depth-of-field state
      set_accum_raygen_pgm(camera_projection, stereoon, gl_dof_on);
    }

    if (resize_buffers) {
      resize_framebuffer(width, height);

      // when movie recording is enabled, print the window size as a guide
      // since the user might want to precisely control the size or 
      // aspect ratio for a particular movie format, e.g. 1080p, 4:3, 16:9
      if (movie_recording_enabled) {
        printf("\rOptiXRenderer) Window resize: %d x %d                               \n", width, height);
      }

      winredraw=1;
    }

    int frame_ready = 1; // Default to true for the non-VCA case
    unsigned int subframe_count = 1;
    if (!done) {
      //
      // If the user interacted with the window in a meaningful way, we
      // need to update the OptiX rendering state, recompile and re-validate
      // the context, and then re-render...
      //
      if (winredraw) {
        // update camera parameters
        RTERR( rtVariableSet1f( cam_zoom_v, cam_zoom) );
        RTERR( rtVariableSet3fv( cam_pos_v, cam_pos) );
        RTERR( rtVariableSet3fv(   cam_U_v, hmd_U) );
        RTERR( rtVariableSet3fv(   cam_V_v, hmd_V) );
        RTERR( rtVariableSet3fv(   cam_W_v, hmd_W) );
        RTERR( rtVariableSet3fv(scene_gradient_v, scene_gradient) );
 
        // update shadow state 
        RTERR( rtVariableSet1i(shadows_enabled_v, gl_shadows_on) );

        // update depth cueing state
        RTERR( rtVariableSet1i(fog_mode_v, 
                 (int) (gl_fog_on) ? fog_mode : RT_FOG_NONE) );

        // update clipping sphere state
        RTERR( rtVariableSet1i(clipview_mode_v, 
                 (int) (gl_clip_on) ? clipview_mode : RT_CLIP_NONE) );

        // update headlight state 
        RTERR( rtVariableSet1i(headlight_mode_v, 
                 (int) (gl_headlight_on) ? RT_HEADLIGHT_ON : RT_HEADLIGHT_OFF) );
 
        // update/recompute DoF values 
        RTERR( rtVariableSet1f(cam_dof_focal_dist_v, cam_dof_focal_dist) );
        RTERR( rtVariableSet1f(cam_dof_aperture_rad_v, cam_dof_focal_dist / (2.0f * cam_zoom * cam_dof_fnumber)) );

        //
        // Update light directions in the OptiX light buffer or user object.
        // Only update when xformlights is set, otherwise we take a 
        // speed hit when using a remote VCA cluster for rendering.
        //
        // We only transform directional lights, since our positional lights
        // are normally affixed to the model coordinate system rather than 
        // the camera.
        //
        if (xformlights) {
#if defined(VMDOPTIX_LIGHTUSEROBJS)
          DirectionalLightList dlights;
          memset(&dlights, 0, sizeof(DirectionalLightList) );
          dlights.num_lights = directional_lights.num();
          int dlcount = directional_lights.num();
          dlcount = (dlcount > DISP_LIGHTS) ? DISP_LIGHTS : dlcount;
          for (i=0; i<dlcount; i++) {
            //vec_copy( (float*)( &lights.dirs[i] ), cur_dlights[i].dir );
            dlights.dirs[i] = cur_dlights[i].dir;
          }
          RTERR( rtVariableSetUserData(dir_light_list_v, sizeof(DirectionalLightList), &dlights) );
#else
          DirectionalLight *dlbuf;
          RTERR( rtBufferMap(dir_lightbuffer, (void **) &dlbuf) );
          for (i=0; i<directional_lights.num(); i++) {
            vec_copy((float*)&dlbuf[i].dir, (float*)&cur_dlights[i].dir);
          }
          RTERR( rtBufferUnmap(dir_lightbuffer) );
#endif
        }

        // reset accumulation buffer 
        accum_count=0;
        totalsamplecount=0;


        // 
        // Sample count updates and OptiX state must always remain in 
        // sync, so if we only update sample count state during redraw events,
        // that's the only time we should recompute the sample counts, since
        // they also affect normalization factors for the accumulation buffer
        // in the non-VCA case.
        //

        // Update sample counts to achieve target interactivity
        if (autosamplecount) {
          if (fpsexpave > 37)
            samples_per_pass++;
          else if (fpsexpave < 30) 
            samples_per_pass--;
    
          // clamp sample counts to a "safe" range
          if (samples_per_pass > 14)
            samples_per_pass=14;
          if (samples_per_pass < 1)
            samples_per_pass=1;
        } 

        // split samples per pass either among AA and AO, depending on
        // whether DoF and AO are enabled or not. 
        if (force_ao_1) {
          cur_aa_samples = samples_per_pass;
          cur_ao_samples = 1;
        } else if (gl_shadows_on && gl_ao_on) {
          if (gl_dof_on) {
            if (samples_per_pass < 4) {
              cur_aa_samples=samples_per_pass;
              cur_ao_samples=1;
            } else {
              int s = (int) sqrtf(samples_per_pass);
              cur_aa_samples=s;
              cur_ao_samples=s;
            }
          } else {
            cur_aa_samples=1;
            cur_ao_samples=samples_per_pass;
          }
        } else {
          cur_aa_samples=samples_per_pass;
          cur_ao_samples=0;
        }

        // update the current AA/AO sample counts since they may be changing if
        // FPS autotuning is enabled...
        RTERR( rtVariableSet1i(aa_samples_v, cur_aa_samples) );

        // observe latest AO enable/disable flag, and sample count
        if (gl_shadows_on && gl_ao_on) {
          RTERR( rtVariableSet1i(ao_samples_v, cur_ao_samples) );
        } else {
          cur_ao_samples = 0;
          RTERR( rtVariableSet1i(ao_samples_v, 0) );
        }

#ifdef VMDOPTIX_PROGRESSIVEAPI
        RTERR( rtVariableSet1f(accum_norm_v, 1.0f / float(cur_aa_samples)) );
#endif

        // updated cached copy of previous window dimensions so we can
        // trigger updates on HMD spheremaps and FBOs as necessary
        owsx = wsx;
        owsy = wsy;
      } 


      //
      // The non-VCA code path must handle the accumulation buffer 
      // for itself, correctly rescaling the accumulated samples when
      // drawing to the output framebuffer.  
      //
      // The VCA code path takes care of normalization for itself.
      //
#ifndef VMDOPTIX_PROGRESSIVEAPI
      // The accumulation buffer normalization factor must be updated
      // to reflect the total accumulation count before the accumulation
      // buffer is drawn to the output framebuffer
      RTERR( rtVariableSet1f(accum_norm_v, 1.0f / float(cur_aa_samples + accum_count)) );

      // The accumulation buffer subframe index must be updated to ensure that
      // the RNGs for AA and AO get correctly re-seeded
      RTERR( rtVariableSet1ui(accum_count_v, accum_count) );

      // Force context compilation/validation
      // If no state has changed, there's no need to recompile/validate.
      // This call can be omitted since OptiX will do this automatically
      // at the next rtContextLaunchXXX() call.
//      render_compile_and_validate();
#endif


      //
      // run the renderer 
      //
      frame_ready = 1; // Default to true for the non-VCA case
      subframe_count = 1;
      if (lasterror == RT_SUCCESS) {
        if (winredraw) {
#ifdef VMDOPTIX_PROGRESSIVEAPI
          // start the VCA doing progressive rendering...
          RTERR( rtContextLaunchProgressive2D(ctx, RT_RAY_GEN_ACCUMULATE, width, height, 0) );
          vcarunning=1;
#else
          RTERR( rtContextLaunch2D(ctx, RT_RAY_GEN_CLEAR_ACCUMULATION_BUFFER, width, height) );
#endif
#if defined(ORT_RAYSTATS)
          RTERR( rtContextLaunch2D(ctx, RT_RAY_GEN_CLEAR_RAYSTATS, width, height) );
#endif
          winredraw=0;
        }

#ifdef VMDOPTIX_PROGRESSIVEAPI
        // Wait for the next frame to arrive
        RTERR( rtBufferGetProgressiveUpdateReady(framebuffer, &frame_ready, &subframe_count, 0) );
        if (frame_ready) 
          totalsamplecount = subframe_count * samples_per_pass;
#else
        // iterate, adding to the accumulation buffer...
        RTERR( rtContextLaunch2D(ctx, RT_RAY_GEN_ACCUMULATE, width, height) );
        subframe_count++; // increment subframe index
        totalsamplecount += samples_per_pass;
        accum_count += cur_aa_samples;

        // copy the accumulation buffer image data to the framebuffer and
        // perform type conversion and normaliztion on the image data...
        RTERR( rtContextLaunch2D(ctx, RT_RAY_GEN_COPY_FINISH, width, height) );
#endif

        if (lasterror == RT_SUCCESS) {
          if (frame_ready) {
            double bufnewtime = wkf_timer_timenow(ort_timer);

            // display output image
            const unsigned char * img;
            rtBufferMap(framebuffer, (void **) &img);

#if 1
            // push latest frame into the video streaming pipeline
            // and pump the event handling mechanism afterwards
            if (app->uivs && app->uivs->srv_connected()) {
              app->uivs->video_frame_pending(img, width, height);             
              app->uivs->check_event();             
            }              
#endif

            rtBufferUnmap(framebuffer);
            mapbuftotaltime = wkf_timer_timenow(ort_timer) - bufnewtime;


            // if live movie recording is on, we save every displayed frame
            // to a sequence sequence of image files, with each file numbered
            // by its frame index, which is computed by the multiplying image 
            // presentation time by the image sequence fixed-rate-FPS value.
            if (movie_recording_enabled && movie_recording_on) {
              char moviefilename[2048];

              // compute frame number from wall clock time and the
              // current fixed-rate movie playback frame rate
              double now = wkf_timer_timenow(ort_timer);
              double frametime = now - movie_recording_start_time;
              int fidx = frametime * movie_recording_fps;

              // always force the first recorded frame to be 0
              if (movie_framecount==0)
                fidx=0;
              movie_framecount++;

#if defined(__linux)
              // generate symlinks for frame indices between the last written
              // frame and the current one so that video encoders such as
              // ffmpeg and mencoder can be fed the contiguous frame sequence
              // at a fixed frame rate, as they require
              sprintf(moviefilename, movie_recording_filebase, 
                      movie_lastframeindex);
              int symidx;
              for (symidx=movie_lastframeindex; symidx<fidx; symidx++) {
                char symlinkfilename[2048];
                sprintf(symlinkfilename, movie_recording_filebase, symidx);
                symlink(moviefilename, symlinkfilename);
              }
#endif
               
              // write the new movie frame               
              sprintf(moviefilename, movie_recording_filebase, fidx);
              if (OptiXWriteImage(moviefilename, writealpha, framebuffer,
                                  RT_FORMAT_UNSIGNED_BYTE4, width, height) == -1) {
                movie_recording_on = 0;
                printf("\n");
                printf("OptiXRenderer) ERROR during writing image during movie recording!\n");
                printf("OptiXRenderer) Movie recording STOPPED\n");
              }

              movie_lastframeindex = fidx; // update last frame index written
            }
          }
        } else {
          printf("OptiXRenderer) An error occured during rendering. Rendering is aborted.\n");
          done=1;
          break;
        }
      } else {
        printf("OptiXRenderer) An error occured in AS generation. Rendering is aborted.\n");
        done=1;
        break;
      }
    }

    if (!done && frame_ready) {
      double newtime = wkf_timer_timenow(ort_timer);
      double frametime = (newtime-oldtime) + 0.00001f;
      oldtime=newtime;

      // compute exponential moving average for exp(-1/10)
      double framefps = 1.0f/frametime;
      fpsexpave = (fpsexpave * 0.90) + (framefps * 0.10);

      printf("OptiXRenderer) %c AA:%2d AO:%2d, %4d tot RT FPS: %.1f  %.4f s/frame sf: %d  \r",
             statestr[state], cur_aa_samples, cur_ao_samples, 
             totalsamplecount, fpsexpave, frametime, subframe_count);

      fflush(stdout);
      state = (state+1) & 3;
    }

  } // end of per-cycle event processing

  printf("\n");

  // write the output image upon exit...
  if (lasterror == RT_SUCCESS) {
    wkf_timer_start(ort_timer);
    // write output image
    OptiXWriteImage(filename, writealpha, framebuffer);
#if defined(ORT_RAYSTATS)
    OptiXPrintRayStats(raystats1_buffer, raystats2_buffer, 0.0);
#endif

    wkf_timer_stop(ort_timer);

    if (verbose == RT_VERB_TIMING || verbose == RT_VERB_DEBUG) {
      printf("OptiXRenderer) image file I/O time: %f secs\n", wkf_timer_time(ort_timer));
    }
  }
}


void OptiXRenderer::render_to_file(const char *filename, int writealpha) {
  if (!context_created)
    return;

  // Unless overridden by environment variables, we use the incoming
  // window size parameters from VMD to initialize the RT image dimensions.
  int wsx=width, wsy=height;
  const char *imageszstr = getenv("VMDOPTIXIMAGESIZE");
  if (imageszstr) {
    if (sscanf(imageszstr, "%d %d", &width, &height) != 2) {
      width=wsx;
      height=wsy;
    }
  }

  // config/allocate framebuffer and accumulation buffer
  config_framebuffer(width, height, 0);

  update_rendering_state(0);
  render_compile_and_validate();
  double starttime = wkf_timer_timenow(ort_timer);

  //
  // run the renderer 
  //
  if (lasterror == RT_SUCCESS) {
    // clear the accumulation buffer
    RTERR( rtContextLaunch2D(ctx, RT_RAY_GEN_CLEAR_ACCUMULATION_BUFFER, width, height) );
#if defined(ORT_RAYSTATS)
    RTERR( rtContextLaunch2D(ctx, RT_RAY_GEN_CLEAR_RAYSTATS, width, height) );
#endif

    // Render to the accumulation buffer for the required number of passes
    if (getenv("VMDOPTIXNORENDER") == NULL) {
      int accum_sample;
      for (accum_sample=0; accum_sample<ext_aa_loops; accum_sample++) {
        // The accumulation subframe count must be updated to ensure that
        // the RNGs for AA and AO get correctly re-seeded
        RTERR( rtVariableSet1ui(accum_count_v, accum_sample) );
  
        RTERR( rtContextLaunch2D(ctx, RT_RAY_GEN_ACCUMULATE, width, height) );
      }
    }

    // copy the accumulation buffer image data to the framebuffer and perform
    // type conversion and normaliztion on the image data...
    RTERR( rtContextLaunch2D(ctx, RT_RAY_GEN_COPY_FINISH, width, height) );
    double rtendtime = wkf_timer_timenow(ort_timer);
    time_ray_tracing = rtendtime - starttime;

    if (lasterror == RT_SUCCESS) {
      // write output image to a file unless we are benchmarking
      if (getenv("VMDOPTIXNOSAVE") == NULL) {
        OptiXWriteImage(filename, writealpha, framebuffer);
      }
#if defined(ORT_RAYSTATS)
      OptiXPrintRayStats(raystats1_buffer, raystats2_buffer, time_ray_tracing);
#endif
      time_image_io = wkf_timer_timenow(ort_timer) - rtendtime;
    } else {
      printf("OptiXRenderer) Error during rendering.  Rendering aborted.\n");
    }

    if (verbose == RT_VERB_TIMING || verbose == RT_VERB_DEBUG) {
      printf("OptiXRenderer) ctx setup %.2f  valid %.2f  AS %.2f  RT %.2f io %.2f\n", time_ctx_setup, time_ctx_validate, time_ctx_AS_build, time_ray_tracing, time_image_io);
    }
  } else {
    printf("OptiXRenderer) Error during AS generation.  Rendering aborted.\n");
  }
}


void OptiXRenderer::destroy_context() {
  if (!context_created)
    return;

#ifdef VMDOPTIX_PROGRESSIVEAPI
  // ensure that there's no way we could be leaving the VCA running
  rtContextStopProgressive(ctx);
#endif

  destroy_framebuffer();

  if ((lasterror = rtContextDestroy(ctx)) != RT_SUCCESS) {
    msgErr << "OptiXRenderer) An error occured while destroying the OptiX context" << sendmsg;
  }
}


void OptiXRenderer::add_material(int matindex,
                                 float ambient, float diffuse, float specular,
                                 float shininess, float reflectivity,
                                 float opacity, 
                                 float outline, float outlinewidth,
                                 int transmode) {
  int oldmatcount = materialcache.num();
  if (oldmatcount <= matindex) {
    ort_material m;
    memset(&m, 0, sizeof(m));

    // XXX do something noticable so we see that we got a bad entry...
    m.ambient = 0.5f;
    m.diffuse = 0.7f;
    m.specular = 0.0f;
    m.shininess = 10.0f;
    m.reflectivity = 0.0f;
    m.opacity = 1.0f;
    m.transmode = 0;

    materialcache.appendN(m, matindex - oldmatcount + 1);
  }
 
  if (materialcache[matindex].isvalid) {
    return;
  } else {
    if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) Adding material[%d]\n", matindex);

    materialcache[matindex].ambient      = ambient;
    materialcache[matindex].diffuse      = diffuse; 
    materialcache[matindex].specular     = specular;
    materialcache[matindex].shininess    = shininess;
    materialcache[matindex].reflectivity = reflectivity;
    materialcache[matindex].opacity      = opacity;
    materialcache[matindex].outline      = outline;
    materialcache[matindex].outlinewidth = outlinewidth;
    materialcache[matindex].transmode    = transmode;
    materialcache[matindex].isvalid      = 1;
  }
}


void OptiXRenderer::init_materials() {
  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) init_materials()\n");

  // pre-register all of the hit programs to be shared by all materials
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath, "closest_hit_radiance_general", &closest_hit_pgm_general) );
#if defined(ORT_USERTXAPIS)
  // OptiX RTX triangle API
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath, "closest_hit_radiance_general_hwtri", &closest_hit_pgm_general_hwtri) );
#endif

#if defined(ORT_USE_TEMPLATE_SHADERS)
  // build up the list of closest hit programs from all combinations
  // of shader parameters
  int i;
  for (i=0; i<ORTMTABSZ; i++) {
    char ch_program_name[256];
    snprintf(ch_program_name, sizeof(ch_program_name),
             "closest_hit_radiance_"
             "CLIP_VIEW_%s_"
             "HEADLIGHT_%s_"
             "FOG_%s_"
             "SHADOWS_%s_"
             "AO_%s_"
             "OUTLINE_%s_"
             "REFL_%s_"
             "TRANS_%s",
#if defined(VMDOPTIX_VCA_TABSZHACK)
             onoffstr(1),
             onoffstr(1),
#else
             onoffstr(i & 128),
             onoffstr(i &  64),
#endif
             onoffstr(i &  32),
             onoffstr(i &  16),
             onoffstr(i &   8),
             onoffstr(i &   4),
             onoffstr(i &   2),
             onoffstr(i &   1));

    RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath, ch_program_name, &closest_hit_pgm_special[i] ) );

#if defined(ORT_USERTXAPIS)
  #error OptiX RTX triangle API not implemented for template shader expansion
#endif

  } 
#endif

  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath, "any_hit_opaque", &any_hit_pgm_opaque) );
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath, "any_hit_transmission", &any_hit_pgm_transmission) );
  RTERR( rtProgramCreateFromPTXFile(ctx, shaderpath, "any_hit_clip_sphere", &any_hit_pgm_clip_sphere) );

  RTERR( rtMaterialCreate(ctx, &material_general) );
  RTERR( rtMaterialSetClosestHitProgram(material_general, RT_RAY_TYPE_RADIANCE, closest_hit_pgm_general) );
  RTERR( rtMaterialSetAnyHitProgram(material_general, RT_RAY_TYPE_SHADOW, any_hit_pgm_clip_sphere) );


#if defined(ORT_USERTXAPIS)
  RTERR( rtMaterialCreate(ctx, &material_general_hwtri) );
  RTERR( rtMaterialSetClosestHitProgram(material_general_hwtri, RT_RAY_TYPE_RADIANCE, closest_hit_pgm_general_hwtri) );
  RTERR( rtMaterialSetAnyHitProgram(material_general_hwtri, RT_RAY_TYPE_SHADOW, any_hit_pgm_clip_sphere) );
#endif


#if defined(ORT_USE_TEMPLATE_SHADERS)
  // build up the list of materials from all combinations of shader parameters
  for (i=0; i<ORTMTABSZ; i++) {
    RTERR( rtMaterialCreate(ctx, &material_special[i]) );
    RTERR( rtMaterialSetClosestHitProgram(material_special[i], RT_RAY_TYPE_RADIANCE, closest_hit_pgm_special[i]) );

    // select correct any hit program depending on opacity
    if (clipview_mode == RT_CLIP_SPHERE) {
      RTERR( rtMaterialSetAnyHitProgram(material_special[i], RT_RAY_TYPE_SHADOW, any_hit_pgm_clip_sphere) );
    } else {
      if (i & 1) {
        RTERR( rtMaterialSetAnyHitProgram(material_special[i], RT_RAY_TYPE_SHADOW, any_hit_pgm_transmission) );
      } else {
        RTERR( rtMaterialSetAnyHitProgram(material_special[i], RT_RAY_TYPE_SHADOW, any_hit_pgm_opaque) );
      }
    }

#if defined(ORT_USERTXAPIS)
  #error OptiX RTX triangle API not implemented for template shader expansion
#endif

    // zero out the array of material usage counts for the scene
    material_special_counts[i] = 0;
  }
#endif
}


void OptiXRenderer::set_material(RTgeometryinstance instance, int matindex, float *uniform_color, int hwtri) {
  if (!context_created)
    return;

//if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) setting material\n");
  RTvariable ka, kd, ks, phongexp, krefl;
  RTvariable opacity, outline, outlinewidth, transmode, uniform_col;
  RTmaterial material = material_general; 

#if defined(ORT_USERTXAPIS)
  // OptiX RTX hardware-triangle APIs require special material handling
  if (hwtri) {
    material = material_general_hwtri;
  }
#endif

#if defined(ORT_USE_TEMPLATE_SHADERS)
  if (getenv("VMDOPTIXFORCEGENERALSHADER") == NULL) {
    unsigned int specialized_material_index = 
      ((clipview_mode != RT_CLIP_NONE)             << 7) |   // VR clip pln/sph
      ((headlight_mode != RT_HEADLIGHT_OFF)        << 6) |   // VR headlight
      ((fog_mode != RT_FOG_NONE)                   << 5) |   // fog
      ((shadows_enabled != RT_SHADOWS_OFF)         << 4) |   // shadows
      ((ao_samples != 0)                           << 3) |   // AO
      ((materialcache[matindex].outline != 0)      << 2) |   // outline
      ((materialcache[matindex].reflectivity != 0) << 1) |   // reflection
      ((materialcache[matindex].opacity != 1)          );    // transmission

#if defined(VMDOPTIX_VCA_TABSZHACK)
    // XXX hack to mask down the material index down to the range 
    //     that works without creating trouble for the VCA
    if (specialized_material_index >= ORTMTABSZ) { 
      specialized_material_index &= (ORTMTABSZ - 1);
    }
#endif

    material = material_special[specialized_material_index];

    // increment material usage counter
    material_special_counts[specialized_material_index]++;
  }
#endif

  RTERR( rtGeometryInstanceSetMaterialCount(instance, 1) );
  RTERR( rtGeometryInstanceSetMaterial(instance, 0, material) );

  if (uniform_color != NULL) {
    RTERR( rtGeometryInstanceDeclareVariable(instance, "uniform_color", &uniform_col) );
    RTERR( rtVariableSet3fv(uniform_col, uniform_color) );
  }

  RTERR( rtGeometryInstanceDeclareVariable(instance, "Ka", &ka) );
  RTERR( rtGeometryInstanceDeclareVariable(instance, "Kd", &kd) );
  RTERR( rtGeometryInstanceDeclareVariable(instance, "Ks", &ks) );
  RTERR( rtGeometryInstanceDeclareVariable(instance, "phong_exp", &phongexp) );
  RTERR( rtGeometryInstanceDeclareVariable(instance, "Krefl", &krefl) );
  RTERR( rtGeometryInstanceDeclareVariable(instance, "opacity", &opacity) );
  RTERR( rtGeometryInstanceDeclareVariable(instance, "outline", &outline) );
  RTERR( rtGeometryInstanceDeclareVariable(instance, "outlinewidth", &outlinewidth) );
  RTERR( rtGeometryInstanceDeclareVariable(instance, "transmode", &transmode) );

  RTERR( rtVariableSet1f(ka, materialcache[matindex].ambient) );
  RTERR( rtVariableSet1f(kd, materialcache[matindex].diffuse) );
  RTERR( rtVariableSet1f(ks, materialcache[matindex].specular) );
  RTERR( rtVariableSet1f(phongexp, materialcache[matindex].shininess) );
  RTERR( rtVariableSet1f(krefl, materialcache[matindex].reflectivity) );
  RTERR( rtVariableSet1f(opacity, materialcache[matindex].opacity) );
  RTERR( rtVariableSet1f(outline, materialcache[matindex].outline) );
  RTERR( rtVariableSet1f(outlinewidth, materialcache[matindex].outlinewidth) );
  RTERR( rtVariableSet1i(transmode, materialcache[matindex].transmode) );
}


void OptiXRenderer::add_directional_light(const float *dir, const float *color) {
  ort_directional_light l;
  vec_copy(l.dir, dir);
  vec_copy(l.color, color);

  directional_lights.append(l);
}


void OptiXRenderer::add_positional_light(const float *pos, const float *color) {
  ort_positional_light l;
  vec_copy(l.pos, pos);
  vec_copy(l.color, color);

  positional_lights.append(l);
}


void OptiXRenderer::cylinder_array(Matrix4 *wtrans, float radius,
                                   float *uniform_color,
                                   int cylnum, float *points, int matindex) {
  if (!context_created) return;
  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) creating cylinder array: %d...\n", cylnum);
  cylinder_array_cnt += cylnum;

  int i, ind;
  RTbuffer buf;
  RTgeometry geom;
  RTgeometryinstance instance;
  vmd_cylinder *cyldata;

  // create and fill the OptiX cylinder array memory buffer
  rtBufferCreate(ctx, RT_BUFFER_INPUT, &buf);
  rtBufferSetFormat(buf, RT_FORMAT_USER);
  rtBufferSetElementSize(buf, sizeof(vmd_cylinder));
  rtBufferSetSize1D(buf, cylnum);
  // rtBufferValidate(buf);
  rtBufferMap(buf, (void **) &cyldata); // map buffer for writing by host

  if (wtrans == NULL) {
    for (i=0,ind=0; i<cylnum; i++,ind+=6) {
      // transform to eye coordinates
      vec_copy((float*) &cyldata[i].start, &points[ind]);
      cyldata[i].radius = radius;
      vec_sub((float*) &cyldata[i].axis, &points[ind+3], &points[ind]);
    }
  } else {
    for (i=0,ind=0; i<cylnum; i++,ind+=6) {
      // transform to eye coordinates
      wtrans->multpoint3d(&points[ind], (float*) &cyldata[i].start);
      cyldata[i].radius = radius;
      float ctmp[3];
      wtrans->multpoint3d(&points[ind+3], ctmp);
      vec_sub((float*) &cyldata[i].axis, ctmp, &points[ind]);
    }
  }
  rtBufferUnmap(buf); // cylinder array is complete, unmap buffer

  RTERR( rtGeometryCreate(ctx, &geom) );
  RTERR( rtGeometrySetPrimitiveCount(geom, cylnum) );
  RTERR( rtGeometrySetBoundingBoxProgram(geom, cylinder_array_bbox_pgm) );
  RTERR( rtGeometrySetIntersectionProgram(geom, cylinder_array_isct_pgm) );

  // this cyl buffer is associated only with this particular geometry node
  RTvariable buf_v;
  RTERR( rtGeometryDeclareVariable(geom, "cylinder_buffer", &buf_v) );
  RTERR( rtVariableSetObject(buf_v, buf) );

  // create a geometry instance node and bind materials to this geometry
  RTERR( rtGeometryInstanceCreate(ctx, &instance) );
  RTERR( rtGeometryInstanceSetGeometry(instance, geom) );

  set_material(instance, matindex, uniform_color);

  append_objects(buf, geom, instance);
}


void OptiXRenderer::cylinder_array_color(Matrix4 *wtrans, float rscale,
                                         int cylnum, float *points, 
                                         float *radii, float *colors,
                                         int matindex) {
  if (!context_created) return;
  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) creating cylinder color array: %d...\n", cylnum);
  cylinder_array_color_cnt += cylnum;

  int i, ind;
  RTbuffer buf;
  RTgeometry geom;
  RTgeometryinstance instance;
  vmd_cylinder_color *cyldata;

  // create and fill the OptiX cylinder array memory buffer
  rtBufferCreate(ctx, RT_BUFFER_INPUT, &buf);
  rtBufferSetFormat(buf, RT_FORMAT_USER);
  rtBufferSetElementSize(buf, sizeof(vmd_cylinder_color));
  rtBufferSetSize1D(buf, cylnum);
  // rtBufferValidate(buf);
  rtBufferMap(buf, (void **) &cyldata); // map buffer for writing by host

  if (wtrans == NULL) {
    // already transformed to eye coordinates
    if (radii == NULL) {
      for (i=0,ind=0; i<cylnum; i++,ind+=6) {
        vec_copy((float*) &cyldata[i].start, &points[ind]);
        cyldata[i].radius = rscale;
        vec_sub((float*) &cyldata[i].axis, &points[ind+3], &points[ind]);
        vec_copy((float*) &cyldata[i].color, &colors[i*3]);
      }
    } else {
      for (i=0,ind=0; i<cylnum; i++,ind+=6) {
        vec_copy((float*) &cyldata[i].start, &points[ind]);
        cyldata[i].radius = rscale * radii[i];
        vec_sub((float*) &cyldata[i].axis, &points[ind+3], &points[ind]);
        vec_copy((float*) &cyldata[i].color, &colors[i*3]);
      }
    }
  } else {
    // transform to eye coordinates
    if (radii == NULL) {
      for (i=0,ind=0; i<cylnum; i++,ind+=6) {
        wtrans->multpoint3d(&points[ind], (float*) &cyldata[i].start);
        cyldata[i].radius = rscale;
        float ctmp[3];
        wtrans->multpoint3d(&points[ind+3], ctmp);
        vec_sub((float*) &cyldata[i].axis, ctmp, (float*) &cyldata[i].start);
        vec_copy((float*) &cyldata[i].color, &colors[i*3]);
      }
    } else {
      for (i=0,ind=0; i<cylnum; i++,ind+=6) {
        wtrans->multpoint3d(&points[ind], (float*) &cyldata[i].start);
        cyldata[i].radius = rscale * radii[i];
        float ctmp[3];
        wtrans->multpoint3d(&points[ind+3], ctmp);
        vec_sub((float*) &cyldata[i].axis, ctmp, (float*) &cyldata[i].start);
        vec_copy((float*) &cyldata[i].color, &colors[i*3]);
      }
    }
  }
  rtBufferUnmap(buf); // cylinder array is complete, unmap buffer

  RTERR( rtGeometryCreate(ctx, &geom) );
  RTERR( rtGeometrySetPrimitiveCount(geom, cylnum) );
  RTERR( rtGeometrySetBoundingBoxProgram(geom, cylinder_array_color_bbox_pgm) );
  RTERR( rtGeometrySetIntersectionProgram(geom, cylinder_array_color_isct_pgm) );

  // this cyl buffer is associated only with this particular geometry node
  RTvariable buf_v;
  RTERR( rtGeometryDeclareVariable(geom, "cylinder_color_buffer", &buf_v) );
  RTERR( rtVariableSetObject(buf_v, buf) );

  // create a geometry instance node and bind materials to this geometry
  RTERR( rtGeometryInstanceCreate(ctx, &instance) );
  RTERR( rtGeometryInstanceSetGeometry(instance, geom) );

  set_material(instance, matindex, NULL);

  append_objects(buf, geom, instance);
}


void OptiXRenderer::ring_array_color(Matrix4 & wtrans, float rscale,
                                     int rnum, float *centers,
                                     float *norms, float *radii, 
                                     float *colors, int matindex) {
  if (!context_created) return;
  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) creating ring array color: %d...\n", rnum);
  ring_array_color_cnt += rnum;

  int i, ind;
  RTbuffer buf;
  RTgeometry geom;
  RTgeometryinstance instance;
  vmd_ring_color *rdata;

  // create and fill the OptiX ring array memory buffer
  rtBufferCreate(ctx, RT_BUFFER_INPUT, &buf);
  rtBufferSetFormat(buf, RT_FORMAT_USER);
  rtBufferSetElementSize(buf, sizeof(vmd_ring_color));
  rtBufferSetSize1D(buf, rnum);
  // rtBufferValidate(buf);
  rtBufferMap(buf, (void **) &rdata); // map buffer for writing by host

  for (i=0,ind=0; i<rnum; i++,ind+=3) {
    // transform to eye coordinates
    wtrans.multpoint3d(&centers[ind], (float*) &rdata[i].center);
    wtrans.multnorm3d(&norms[ind], (float*) &rdata[i].norm);
    vec_normalize((float*) &rdata[i].norm);
    rdata[i].inrad  = rscale * radii[i*2];
    rdata[i].outrad = rscale * radii[i*2+1];
    vec_copy((float*) &rdata[i].color, &colors[ind]);
    rdata[i].pad = 0.0f; // please valgrind  
  }
  rtBufferUnmap(buf); // ring array is complete, unmap buffer

  RTERR( rtGeometryCreate(ctx, &geom) );
  RTERR( rtGeometrySetPrimitiveCount(geom, rnum) );
  RTERR( rtGeometrySetBoundingBoxProgram(geom, ring_array_color_bbox_pgm) );
  RTERR( rtGeometrySetIntersectionProgram(geom, ring_array_color_isct_pgm) );

  // this ring buffer is associated only with this particular geometry node
  RTvariable buf_v;
  RTERR( rtGeometryDeclareVariable(geom, "ring_color_buffer", &buf_v) );
  RTERR( rtVariableSetObject(buf_v, buf) );

  // create a geometry instance node and bind materials to this geometry
  RTERR( rtGeometryInstanceCreate(ctx, &instance) );
  RTERR( rtGeometryInstanceSetGeometry(instance, geom) );

  set_material(instance, matindex, NULL);

  append_objects(buf, geom, instance);
}


void OptiXRenderer::sphere_array(Matrix4 *wtrans, float rscale,
                                 float *uniform_color,
                                 int spnum, float *centers,
                                 float *radii,
                                 int matindex) {
  if (!context_created) return;
  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) creating sphere array: %d...\n", spnum);
  sphere_array_cnt += spnum;

  int i, ind;
  RTbuffer buf;
  RTgeometry geom;
  RTgeometryinstance instance;
  vmd_sphere *spdata;

  // create and fill the OptiX sphere array memory buffer
  rtBufferCreate(ctx, RT_BUFFER_INPUT, &buf);
  rtBufferSetFormat(buf, RT_FORMAT_USER);
  rtBufferSetElementSize(buf, sizeof(vmd_sphere));
  rtBufferSetSize1D(buf, spnum);
  // rtBufferValidate(buf);
  rtBufferMap(buf, (void **) &spdata); // map buffer for writing by host

  if (wtrans == NULL) {
    if (radii == NULL) {
      for (i=0,ind=0; i<spnum; i++,ind+=3) {
        // transform to eye coordinates
        vec_copy((float*) &spdata[i].center, &centers[ind]);
        spdata[i].radius = rscale; // use "rscale" as radius...
      }
    } else {
      for (i=0,ind=0; i<spnum; i++,ind+=3) {
        // transform to eye coordinates
        vec_copy((float*) &spdata[i].center, &centers[ind]);
        spdata[i].radius = rscale * radii[i];
      }
    }
  } else {
    if (radii == NULL) {
      for (i=0,ind=0; i<spnum; i++,ind+=3) {
        // transform to eye coordinates
        wtrans->multpoint3d(&centers[ind], (float*) &spdata[i].center);
        spdata[i].radius = rscale; // use "rscale" as radius...
      }
    } else {
      for (i=0,ind=0; i<spnum; i++,ind+=3) {
        // transform to eye coordinates
        wtrans->multpoint3d(&centers[ind], (float*) &spdata[i].center);
        spdata[i].radius = rscale * radii[i];
      }
    }
  }
  rtBufferUnmap(buf); // sphere array is complete, unmap buffer

  RTERR( rtGeometryCreate(ctx, &geom) );
  RTERR( rtGeometrySetPrimitiveCount(geom, spnum) );
  RTERR( rtGeometrySetBoundingBoxProgram(geom, sphere_array_bbox_pgm) );
  RTERR( rtGeometrySetIntersectionProgram(geom, sphere_array_isct_pgm) );

  // this sphere buffer is associated only with this particular geometry node
  RTvariable buf_v;
  RTERR( rtGeometryDeclareVariable(geom, "sphere_buffer", &buf_v) );
  RTERR( rtVariableSetObject(buf_v, buf) );

  // create a geometry instance node and bind materials to this geometry
  RTERR( rtGeometryInstanceCreate(ctx, &instance) );
  RTERR( rtGeometryInstanceSetGeometry(instance, geom) );

  set_material(instance, matindex, uniform_color);

  append_objects(buf, geom, instance);
}


void OptiXRenderer::sphere_array_color(Matrix4 & wtrans, float rscale,
                                       int spnum, float *centers,
                                       float *radii, float *colors,
                                       int matindex) {
  if (!context_created) return;
  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) creating sphere array color: %d...\n", spnum);
  sphere_array_color_cnt += spnum;

  int i, ind;
  RTbuffer buf;
  RTgeometry geom;
  RTgeometryinstance instance;
  vmd_sphere_color *spdata;

  // create and fill the OptiX sphere array memory buffer
  rtBufferCreate(ctx, RT_BUFFER_INPUT, &buf);
  rtBufferSetFormat(buf, RT_FORMAT_USER);
  rtBufferSetElementSize(buf, sizeof(vmd_sphere_color));
  rtBufferSetSize1D(buf, spnum);
  // rtBufferValidate(buf);
  rtBufferMap(buf, (void **) &spdata); // map buffer for writing by host

  for (i=0,ind=0; i<spnum; i++,ind+=3) {
    // transform to eye coordinates
    wtrans.multpoint3d(&centers[ind], (float*) &spdata[i].center);
    spdata[i].radius = rscale * radii[i];
    vec_copy((float*) &spdata[i].color, &colors[ind]);
    spdata[i].pad = 0.0f; // please valgrind
  }
  rtBufferUnmap(buf); // sphere array is complete, unmap buffer

  RTERR( rtGeometryCreate(ctx, &geom) );
  RTERR( rtGeometrySetPrimitiveCount(geom, spnum) );
  RTERR( rtGeometrySetBoundingBoxProgram(geom, sphere_array_color_bbox_pgm) );
  RTERR( rtGeometrySetIntersectionProgram(geom, sphere_array_color_isct_pgm) );

  // this sphere buffer is associated only with this particular geometry node
  RTvariable buf_v;
  RTERR( rtGeometryDeclareVariable(geom, "sphere_color_buffer", &buf_v) );
  RTERR( rtVariableSetObject(buf_v, buf) );

  // create a geometry instance node and bind materials to this geometry
  RTERR( rtGeometryInstanceCreate(ctx, &instance) );
  RTERR( rtGeometryInstanceSetGeometry(instance, geom) );

  set_material(instance, matindex, NULL);

  append_objects(buf, geom, instance);
}


#if defined(ORT_USERTXAPIS)
void OptiXRenderer::tricolor_list_hwtri(Matrix4 & wtrans, int numtris, 
                                        float *vnc, int matindex) {
  if (!context_created) return;
//if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) creating tricolor list: %d...\n", numtris);
  tricolor_cnt += numtris;

  RTbuffer vbuf, nbuf, cbuf;
  RTgeometryinstance instance_hwtri;
  RTgeometrytriangles geom_hwtri;
  
  // Create and fill vertex/normal/color buffers
  float3 *vertices; 
  uint4 *normals;
  uchar4 *colors;
  hwtri_alloc_bufs_v3f_n4u4_c4u(ctx, numtris, vbuf, vertices, nbuf, normals, 
                                cbuf, 1, colors, NULL);

  int i, ind, tcnt;
  for (i=0,ind=0,tcnt=0; i<numtris; i++,ind+=27) {
    int taddr = 3 * tcnt;

    // transform to eye coordinates
    wtrans.multpoint3d(&vnc[ind     ], (float*) &vertices[taddr + 0]);
    wtrans.multpoint3d(&vnc[ind +  3], (float*) &vertices[taddr + 1]);
    wtrans.multpoint3d(&vnc[ind +  6], (float*) &vertices[taddr + 2]);

    // Compute geometric normal, detect and cull degenerate triangles
    float3 Ng;
    if (hwtri_test_calc_Ngeom(&vertices[taddr], Ng)) {
      continue; // cull any triangle that fails degeneracy tests
    }

    float3 n0, n1, n2;
    wtrans.multnorm3d(&vnc[ind +  9], (float*) &n0);
    wtrans.multnorm3d(&vnc[ind + 12], (float*) &n1);
    wtrans.multnorm3d(&vnc[ind + 15], (float*) &n2);

    // Pack normals
    normals[tcnt].x = packNormal(Ng);
    normals[tcnt].y = packNormal(n0);
    normals[tcnt].z = packNormal(n1);
    normals[tcnt].w = packNormal(n2);

    // convert color format
    colors[taddr + 0].x = vnc[ind + 18] * 255.0f;
    colors[taddr + 0].y = vnc[ind + 19] * 255.0f;
    colors[taddr + 0].z = vnc[ind + 20] * 255.0f;

    colors[taddr + 1].x = vnc[ind + 21] * 255.0f;
    colors[taddr + 1].y = vnc[ind + 22] * 255.0f;
    colors[taddr + 1].z = vnc[ind + 23] * 255.0f;

    colors[taddr + 2].x = vnc[ind + 24] * 255.0f;
    colors[taddr + 2].y = vnc[ind + 25] * 255.0f;
    colors[taddr + 2].z = vnc[ind + 26] * 255.0f;

    tcnt++; // count non-culled triangles
  }

  rtBufferUnmap(vbuf);
  rtBufferUnmap(nbuf);
  rtBufferUnmap(cbuf);

  RTERR( rtGeometryTrianglesCreate(ctx, &geom_hwtri) );
  RTERR( rtGeometryTrianglesSetTriangles(geom_hwtri, tcnt, vbuf, 
                                         0, sizeof(float3), RT_FORMAT_FLOAT3, 
                                         RT_GEOMETRYBUILDFLAGS_RELEASE_BUFFERS) 
);

  // create a geometry instance and bind materials to this geometry
  RTERR( rtGeometryInstanceCreate(ctx, &instance_hwtri) );
  RTERR( rtGeometryInstanceSetGeometryTriangles(instance_hwtri, geom_hwtri) );

  // Enable per-vertex normals, enable per-vertex colors
  hwtri_set_vertex_flags(ctx, instance_hwtri, nbuf, cbuf, 1, 1);

  // We have to pass the explicit hardware triangle parameter for this geometry
  set_material(instance_hwtri, matindex, NULL, 1);

  // The vertex buffer is released automatically after construction. 
  // We need to keep track of normal and color buffers for ourselves.
  append_objects(nbuf, cbuf, geom_hwtri, instance_hwtri);
}
#endif


void OptiXRenderer::tricolor_list(Matrix4 & wtrans, int numtris, float *vnc,
                                  int matindex) {
  if (!context_created) return;
//if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) creating tricolor list: %d...\n", numtris);
  tricolor_cnt += numtris;

  int i, ind;
  RTbuffer buf;
  RTgeometry geom;
  RTgeometryinstance instance;
  vmd_tricolor *trimesh;

  // create and fill the OptiX trimesh memory buffer
  rtBufferCreate(ctx, RT_BUFFER_INPUT, &buf);
  rtBufferSetFormat(buf, RT_FORMAT_USER);
  rtBufferSetElementSize(buf, sizeof(vmd_tricolor));
  rtBufferSetSize1D(buf, numtris);
  // rtBufferValidate(buf);
  rtBufferMap(buf, (void **) &trimesh); // map buffer for writing by host

  for (i=0,ind=0; i<numtris; i++,ind+=27) {
    // transform to eye coordinates
    wtrans.multpoint3d(&vnc[ind     ], (float*) &trimesh[i].v0);
    wtrans.multpoint3d(&vnc[ind +  3], (float*) &trimesh[i].v1);
    wtrans.multpoint3d(&vnc[ind +  6], (float*) &trimesh[i].v2);

    wtrans.multnorm3d(&vnc[ind +  9], (float*) &trimesh[i].n0);
    wtrans.multnorm3d(&vnc[ind + 12], (float*) &trimesh[i].n1);
    wtrans.multnorm3d(&vnc[ind + 15], (float*) &trimesh[i].n2);

    vec_copy((float*) &trimesh[i].c0, &vnc[ind + 18]);
    vec_copy((float*) &trimesh[i].c1, &vnc[ind + 21]);
    vec_copy((float*) &trimesh[i].c2, &vnc[ind + 24]);
  }
  rtBufferUnmap(buf); // triangle list is complete, unmap buffer

  RTERR( rtGeometryCreate(ctx, &geom) );
  RTERR( rtGeometrySetPrimitiveCount(geom, numtris) );
  RTERR( rtGeometrySetBoundingBoxProgram(geom, tricolor_bbox_pgm) );
  RTERR( rtGeometrySetIntersectionProgram(geom, tricolor_isct_pgm) );

  // this buffer is associated only with this particular geometry node
  RTvariable buf_v;
  RTERR( rtGeometryDeclareVariable(geom, "tricolor_buffer", &buf_v) );
  RTERR( rtVariableSetObject(buf_v, buf) );

  // create a geometry instance and bind materials to this geometry
  RTERR( rtGeometryInstanceCreate(ctx, &instance) );
  RTERR( rtGeometryInstanceSetGeometry(instance, geom) );
//  RTERR( rtGeometryInstanceSetMaterialCount(instance, 1) );

  set_material(instance, matindex, NULL);

  append_objects(buf, geom, instance);
}


#if defined(ORT_USERTXAPIS)
void OptiXRenderer::trimesh_c4n3v3_hwtri(Matrix4 & wtrans, int numverts,
                                         float *cnv, 
                                         int numfacets, int * facets,
                                         int matindex) {
  if (!context_created) return;
  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) creating trimesh_c4n3v3_hwtri: %d...\n", numfacets);
  trimesh_c4u_n3b_v3f_cnt += numfacets;

  RTbuffer vbuf, nbuf, cbuf;
  RTgeometryinstance instance_hwtri;
  RTgeometrytriangles geom_hwtri;

  // Create and fill vertex/normal/color buffers
  float3 *vertices; 
  uint4 *normals;
  uchar4 *colors;
  hwtri_alloc_bufs_v3f_n4u4_c4u(ctx, numfacets, vbuf, vertices, nbuf, normals, 
                                cbuf, numfacets * 3, colors, NULL);

  int i, ind, tcnt;
  for (i=0,ind=0,tcnt=0; i<numfacets; i++,ind+=3) {
    int taddr = 3 * tcnt;

    int v0 = facets[ind    ] * 10;
    int v1 = facets[ind + 1] * 10;
    int v2 = facets[ind + 2] * 10;

    // transform to eye coordinates
    wtrans.multpoint3d(cnv + v0 + 7, (float*) &vertices[taddr + 0]);
    wtrans.multpoint3d(cnv + v1 + 7, (float*) &vertices[taddr + 1]);
    wtrans.multpoint3d(cnv + v2 + 7, (float*) &vertices[taddr + 2]);

    // Compute geometric normal, detect and cull degenerate triangles
    float3 Ng;
    if (hwtri_test_calc_Ngeom(&vertices[taddr], Ng)) {
      continue; // cull any triangle that fails degeneracy tests
    }

    float3 n0, n1, n2;
    wtrans.multnorm3d(cnv + v0 + 4, (float*) &n0);
    wtrans.multnorm3d(cnv + v1 + 4, (float*) &n1);
    wtrans.multnorm3d(cnv + v2 + 4, (float*) &n2);

    // Pack normals
    normals[tcnt].x = packNormal(Ng);
    normals[tcnt].y = packNormal(n0);
    normals[tcnt].z = packNormal(n1);
    normals[tcnt].w = packNormal(n2);

    // convert color format
    colors[taddr + 0].x = cnv[v0 + 0] * 255.0f;
    colors[taddr + 0].y = cnv[v0 + 1] * 255.0f;
    colors[taddr + 0].z = cnv[v0 + 2] * 255.0f;

    colors[taddr + 1].x = cnv[v1 + 0] * 255.0f;
    colors[taddr + 1].y = cnv[v1 + 1] * 255.0f;
    colors[taddr + 1].z = cnv[v1 + 2] * 255.0f;

    colors[taddr + 2].x = cnv[v2 + 0] * 255.0f;
    colors[taddr + 2].y = cnv[v2 + 1] * 255.0f;
    colors[taddr + 2].z = cnv[v2 + 2] * 255.0f;

    tcnt++; // count non-culled triangles
  }

  rtBufferUnmap(vbuf);
  rtBufferUnmap(nbuf);
  rtBufferUnmap(cbuf);

  RTERR( rtGeometryTrianglesCreate(ctx, &geom_hwtri) );
  RTERR( rtGeometryTrianglesSetTriangles(geom_hwtri, tcnt, vbuf, 
                                         0, sizeof(float3), RT_FORMAT_FLOAT3, 
                                         RT_GEOMETRYBUILDFLAGS_RELEASE_BUFFERS) );

  // create a geometry instance and bind materials to this geometry
  RTERR( rtGeometryInstanceCreate(ctx, &instance_hwtri) );
  RTERR( rtGeometryInstanceSetGeometryTriangles(instance_hwtri, geom_hwtri) );

  // Enable per-vertex normals, enable per-vertex colors
  hwtri_set_vertex_flags(ctx, instance_hwtri, nbuf, cbuf, 1, 1);

  // We have to pass the explicit hardware triangle parameter for this geometry
  set_material(instance_hwtri, matindex, NULL, 1);

  // The vertex buffer is released automatically after construction. 
  // We need to keep track of normal and color buffers for ourselves.
  append_objects(nbuf, cbuf, geom_hwtri, instance_hwtri);
}
#endif


void OptiXRenderer::trimesh_c4n3v3(Matrix4 & wtrans, int numverts,
                                   float *cnv, int numfacets, int * facets,
                                   int matindex) {
  if (!context_created) return;

#if defined(ORT_USERTXAPIS)
  // OptiX 5.2 hardware-accelerated triangle API
  if (hwtri_enabled) {
    trimesh_c4n3v3_hwtri(wtrans, numverts, cnv, numfacets, facets, matindex);
    return;
  }
#endif

  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) creating trimesh_c4n3v3: %d...\n", numfacets);
  trimesh_c4u_n3b_v3f_cnt += numfacets;

  int i, ind;
  RTbuffer buf;
  RTgeometry geom;
  RTgeometryinstance instance;
  vmd_tricolor *trimesh;

  // create and fill the OptiX trimesh memory buffer
  rtBufferCreate(ctx, RT_BUFFER_INPUT, &buf);
  rtBufferSetFormat(buf, RT_FORMAT_USER);
  rtBufferSetElementSize(buf, sizeof(vmd_tricolor));
  rtBufferSetSize1D(buf, numfacets);
  // rtBufferValidate(buf);
  rtBufferMap(buf, (void **) &trimesh); // map buffer for writing by host

  for (i=0,ind=0; i<numfacets; i++,ind+=3) {
    int v0 = facets[ind    ] * 10;
    int v1 = facets[ind + 1] * 10;
    int v2 = facets[ind + 2] * 10;

    // transform to eye coordinates
    wtrans.multpoint3d(cnv + v0 + 7, (float*) &trimesh[i].v0);
    wtrans.multpoint3d(cnv + v1 + 7, (float*) &trimesh[i].v1);
    wtrans.multpoint3d(cnv + v2 + 7, (float*) &trimesh[i].v2);

    wtrans.multnorm3d(cnv + v0 + 4, (float*) &trimesh[i].n0);
    wtrans.multnorm3d(cnv + v1 + 4, (float*) &trimesh[i].n1);
    wtrans.multnorm3d(cnv + v2 + 4, (float*) &trimesh[i].n2);

    vec_copy((float*) &trimesh[i].c0, cnv + v0);
    vec_copy((float*) &trimesh[i].c1, cnv + v1);
    vec_copy((float*) &trimesh[i].c2, cnv + v2);
  }
  rtBufferUnmap(buf); // triangle list is complete, unmap buffer

  RTERR( rtGeometryCreate(ctx, &geom) );
  RTERR( rtGeometrySetPrimitiveCount(geom, numfacets) );
  RTERR( rtGeometrySetBoundingBoxProgram(geom, tricolor_bbox_pgm) );
  RTERR( rtGeometrySetIntersectionProgram(geom, tricolor_isct_pgm) );

  // this buffer is associated only with this particular geometry node
  RTvariable buf_v;
  RTERR( rtGeometryDeclareVariable(geom, "tricolor_buffer", &buf_v) );
  RTERR( rtVariableSetObject(buf_v, buf) );

  // create a geometry instance and bind materials to this geometry
  RTERR( rtGeometryInstanceCreate(ctx, &instance) );
  RTERR( rtGeometryInstanceSetGeometry(instance, geom) );

  set_material(instance, matindex, NULL);

  append_objects(buf, geom, instance);
}


#if defined(ORT_USERTXAPIS)
// 
// This implementation translates from the most-compact host representation
// to a GPU-specific organization that balances performance vs. memory 
// storage efficiency.
//
void OptiXRenderer::trimesh_c4u_n3b_v3f_hwtri(Matrix4 & wtrans, 
                                              unsigned char *c, char *n, 
                                              float *v, int numfacets, 
                                              int matindex) {
  if (!context_created) return;
  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) creating trimesh_c4u_n3b_v3f_hwtri: %d...\n", numfacets);
  trimesh_c4u_n3b_v3f_cnt += numfacets;

  RTbuffer vbuf, nbuf, cbuf;
  RTgeometryinstance instance_hwtri;
  RTgeometrytriangles geom_hwtri;

  // Create and fill vertex/normal/color buffers
  float3 *vertices; 
  uint4 *normals;
  uchar4 *colors;
  hwtri_alloc_bufs_v3f_n4u4_c4u(ctx, numfacets, vbuf, vertices, nbuf, normals, 
                                cbuf, numfacets * 3, colors, NULL);

  const float ci2f = 1.0f / 255.0f;
  const float cn2f = 1.0f / 127.5f;
  int i, j, ind, tcnt;
  for (ind=0,i=0,j=0,tcnt=0; ind<numfacets; ind++,i+=9,j+=12) {
    float norm[9];
    int taddr = 3 * tcnt;

    // transform to eye coordinates
    wtrans.multpoint3d(v + i    , (float*) &vertices[taddr + 0]);
    wtrans.multpoint3d(v + i + 3, (float*) &vertices[taddr + 1]);
    wtrans.multpoint3d(v + i + 6, (float*) &vertices[taddr + 2]);

    // Compute geometric normal, detect and cull degenerate triangles
    float3 Ng;
    if (hwtri_test_calc_Ngeom(&vertices[taddr], Ng)) {
      continue; // cull any triangle that fails degeneracy tests
    }

    // conversion from GLbyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
    // float = (2c+1)/(2^8-1)
    norm[0] = n[i    ] * cn2f + ci2f;
    norm[1] = n[i + 1] * cn2f + ci2f;
    norm[2] = n[i + 2] * cn2f + ci2f;
    norm[3] = n[i + 3] * cn2f + ci2f;
    norm[4] = n[i + 4] * cn2f + ci2f;
    norm[5] = n[i + 5] * cn2f + ci2f;
    norm[6] = n[i + 6] * cn2f + ci2f;
    norm[7] = n[i + 7] * cn2f + ci2f;
    norm[8] = n[i + 8] * cn2f + ci2f;

    // conversion to GLbyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
    float3 n0, n1, n2;
    wtrans.multnorm3d(&norm[0], (float*) &n0);
    wtrans.multnorm3d(&norm[3], (float*) &n1);
    wtrans.multnorm3d(&norm[6], (float*) &n2);

    // Pack normals
    normals[tcnt].x = packNormal(Ng);
    normals[tcnt].y = packNormal(n0);
    normals[tcnt].z = packNormal(n1);
    normals[tcnt].w = packNormal(n2);

    memcpy(&colors[tcnt * 3], &c[j], 12); // copy colors (same memory format)

    tcnt++; // count non-culled triangles
  }

  rtBufferUnmap(vbuf);
  rtBufferUnmap(nbuf);
  rtBufferUnmap(cbuf);

  RTERR( rtGeometryTrianglesCreate(ctx, &geom_hwtri) );
  RTERR( rtGeometryTrianglesSetTriangles(geom_hwtri, tcnt, vbuf, 
                                         0, sizeof(float3), RT_FORMAT_FLOAT3, 
                                         RT_GEOMETRYBUILDFLAGS_RELEASE_BUFFERS) );

  // create a geometry instance and bind materials to this geometry
  RTERR( rtGeometryInstanceCreate(ctx, &instance_hwtri) );
  RTERR( rtGeometryInstanceSetGeometryTriangles(instance_hwtri, geom_hwtri) );

  // Enable per-vertex normals, enable per-vertex colors
  hwtri_set_vertex_flags(ctx, instance_hwtri, nbuf, cbuf, 1, 1);

  // We have to pass the explicit hardware triangle parameter for this geometry
  set_material(instance_hwtri, matindex, NULL, 1);

  // The vertex buffer is released automatically after construction. 
  // We need to keep track of normal and color buffers for ourselves.
  append_objects(nbuf, cbuf, geom_hwtri, instance_hwtri);
}
#endif


// 
// This implementation translates from the most-compact host representation
// to a GPU-specific organization that balances performance vs. memory 
// storage efficiency.
//
void OptiXRenderer::trimesh_c4u_n3b_v3f(Matrix4 & wtrans, unsigned char *c, 
                                        char *n, float *v, int numfacets, 
                                        int matindex) {
  if (!context_created) return;

#if defined(ORT_USERTXAPIS)
  // OptiX RTX hardware-accelerated triangle API
  if (hwtri_enabled) {
    trimesh_c4u_n3b_v3f_hwtri(wtrans, c, n, v, numfacets, matindex);
    return;
  }
#endif

  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) creating trimesh_c4u_n3b_v3f: %d...\n", numfacets);
  trimesh_n3b_v3f_cnt += numfacets;

  int i, j, ind;
  RTbuffer buf;
  RTgeometry geom;
  RTgeometryinstance instance;
  vmd_trimesh_c4u_n3b_v3f *trimesh;

  // create and fill the OptiX trimesh memory buffer
  rtBufferCreate(ctx, RT_BUFFER_INPUT, &buf);
  rtBufferSetFormat(buf, RT_FORMAT_USER);
  rtBufferSetElementSize(buf, sizeof(vmd_trimesh_c4u_n3b_v3f));
  rtBufferSetSize1D(buf, numfacets);
  // rtBufferValidate(buf);
  rtBufferMap(buf, (void **) &trimesh); // map buffer for writing by host

  const float ci2f = 1.0f / 255.0f;
  const float cn2f = 1.0f / 127.5f;
  for (ind=0,i=0,j=0; ind<numfacets; ind++,i+=9,j+=12) {
    float norm[9];

    // transform to eye coordinates
    wtrans.multpoint3d(v + i    , (float*) &trimesh[ind].v0);
    wtrans.multpoint3d(v + i + 3, (float*) &trimesh[ind].v1);
    wtrans.multpoint3d(v + i + 6, (float*) &trimesh[ind].v2);

    // conversion from GLbyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
    // float = (2c+1)/(2^8-1)
    norm[0] = n[i    ] * cn2f + ci2f;
    norm[1] = n[i + 1] * cn2f + ci2f;
    norm[2] = n[i + 2] * cn2f + ci2f;
    norm[3] = n[i + 3] * cn2f + ci2f;
    norm[4] = n[i + 4] * cn2f + ci2f;
    norm[5] = n[i + 5] * cn2f + ci2f;
    norm[6] = n[i + 6] * cn2f + ci2f;
    norm[7] = n[i + 7] * cn2f + ci2f;
    norm[8] = n[i + 8] * cn2f + ci2f;

    // conversion to GLbyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
    float3 tmpn;
    wtrans.multnorm3d(&norm[0], (float*) &tmpn);
    tmpn = tmpn * 127.5f - 0.5f;
    trimesh[ind].n0 = make_char4(tmpn.x, tmpn.y, tmpn.z, 0);
    wtrans.multnorm3d(&norm[3], (float*) &tmpn);
    tmpn = tmpn * 127.5f - 0.5f;
    trimesh[ind].n1 = make_char4(tmpn.x, tmpn.y, tmpn.z, 0);
    wtrans.multnorm3d(&norm[6], (float*) &tmpn);
    tmpn = tmpn * 127.5f - 0.5f;
    trimesh[ind].n2 = make_char4(tmpn.x, tmpn.y, tmpn.z, 0);

    memcpy(&trimesh[ind].c0, &c[j  ], 4);
    memcpy(&trimesh[ind].c1, &c[j+4], 4);
    memcpy(&trimesh[ind].c2, &c[j+8], 4);
  }
  rtBufferUnmap(buf); // triangle list is complete, unmap buffer

  RTERR( rtGeometryCreate(ctx, &geom) );
  RTERR( rtGeometrySetPrimitiveCount(geom, numfacets) );
  RTERR( rtGeometrySetBoundingBoxProgram(geom, trimesh_c4u_n3b_v3f_bbox_pgm) );
  RTERR( rtGeometrySetIntersectionProgram(geom, trimesh_c4u_n3b_v3f_isct_pgm) );

  // this trimesh buffer is associated only with this particular geometry node
  RTvariable buf_v;
  RTERR( rtGeometryDeclareVariable(geom, "trimesh_c4u_n3b_v3f_buffer", &buf_v) );
  RTERR( rtVariableSetObject(buf_v, buf) );

  // create a geometry instance and bind materials to this geometry
  RTERR( rtGeometryInstanceCreate(ctx, &instance) );
  RTERR( rtGeometryInstanceSetGeometry(instance, geom) );

  set_material(instance, matindex, NULL);

  append_objects(buf, geom, instance);
}


#if defined(ORT_USERTXAPIS)
void OptiXRenderer::trimesh_c4u_n3f_v3f_hwtri(Matrix4 & wtrans, 
                                              unsigned char *c, 
                                              float *n, float *v, 
                                              int numfacets, 
                                              int matindex) {
  if (!context_created) return;
  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) creating trimesh_c4u_n3f_v3f: %d...\n", numfacets);
  tricolor_cnt += numfacets;

  RTbuffer vbuf, nbuf, cbuf;
  RTgeometryinstance instance_hwtri;
  RTgeometrytriangles geom_hwtri;
  
  // Create and fill vertex/normal/color buffers
  float3 *vertices; 
  uint4 *normals;
  uchar4 *colors;
  hwtri_alloc_bufs_v3f_n4u4_c4u(ctx, numfacets, vbuf, vertices, nbuf, normals, 
                                cbuf, 1, colors, NULL);

  const float ci2f = 1.0f / 255.0f;
  int i, j, ind, tcnt;
  for (ind=0,i=0,j=0,tcnt=0; ind<numfacets; ind++,i+=9,j+=12) {
    int taddr = 3 * tcnt;

    // transform to eye coordinates
    wtrans.multpoint3d(v + i    , (float*) &vertices[taddr + 0]);
    wtrans.multpoint3d(v + i + 3, (float*) &vertices[taddr + 1]);
    wtrans.multpoint3d(v + i + 6, (float*) &vertices[taddr + 2]);

    // Compute geometric normal, detect and cull degenerate triangles
    float3 Ng;
    if (hwtri_test_calc_Ngeom(&vertices[taddr], Ng)) {
      continue; // cull any triangle that fails degeneracy tests
    }

    float3 n0, n1, n2;
    wtrans.multnorm3d(n + i    , (float*) &n0);
    wtrans.multnorm3d(n + i + 3, (float*) &n1);
    wtrans.multnorm3d(n + i + 6, (float*) &n2);

    // Pack normals
    normals[tcnt].x = packNormal(Ng);
    normals[tcnt].y = packNormal(n0);
    normals[tcnt].z = packNormal(n1);
    normals[tcnt].w = packNormal(n2);

    memcpy(&colors[taddr + 0], &c[j  ], sizeof(uchar4));
    memcpy(&colors[taddr + 1], &c[j+4], sizeof(uchar4));
    memcpy(&colors[taddr + 2], &c[j+8], sizeof(uchar4));

    tcnt++; // count non-culled triangles
  }

  rtBufferUnmap(vbuf);
  rtBufferUnmap(nbuf);
  rtBufferUnmap(cbuf);

  RTERR( rtGeometryTrianglesCreate(ctx, &geom_hwtri) );
  RTERR( rtGeometryTrianglesSetTriangles(geom_hwtri, tcnt, vbuf, 
                                         0, sizeof(float3), RT_FORMAT_FLOAT3, 
                                         RT_GEOMETRYBUILDFLAGS_RELEASE_BUFFERS) 
);

  // create a geometry instance and bind materials to this geometry
  RTERR( rtGeometryInstanceCreate(ctx, &instance_hwtri) );
  RTERR( rtGeometryInstanceSetGeometryTriangles(instance_hwtri, geom_hwtri) );

  // Enable per-vertex normals, enable per-vertex colors
  hwtri_set_vertex_flags(ctx, instance_hwtri, nbuf, cbuf, 1, 1);

  // We have to pass the explicit hardware triangle parameter for this geometry
  set_material(instance_hwtri, matindex, NULL, 1);

  // The vertex buffer is released automatically after construction. 
  // We need to keep track of normal and color buffers for ourselves.
  append_objects(nbuf, cbuf, geom_hwtri, instance_hwtri);
}
#endif


void OptiXRenderer::trimesh_c4u_n3f_v3f(Matrix4 & wtrans, unsigned char *c, 
                                        float *n, float *v, int numfacets, 
                                        int matindex) {
  if (!context_created) return;

#if defined(ORT_USERTXAPIS)
  // OptiX RTX hardware-accelerated triangle API
  if (hwtri_enabled) {
    trimesh_c4u_n3f_v3f_hwtri(wtrans, c, n, v, numfacets, matindex);
    return;
  }
#endif

  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) creating trimesh_c4u_n3f_v3f: %d...\n", numfacets);
  tricolor_cnt += numfacets;

  int i, j, ind;
  RTbuffer buf;
  RTgeometry geom;
  RTgeometryinstance instance;
  vmd_tricolor *trimesh;

  // create and fill the OptiX trimesh memory buffer
  rtBufferCreate(ctx, RT_BUFFER_INPUT, &buf);
  rtBufferSetFormat(buf, RT_FORMAT_USER);
  rtBufferSetElementSize(buf, sizeof(vmd_tricolor));
  rtBufferSetSize1D(buf, numfacets);
  // rtBufferValidate(buf);
  rtBufferMap(buf, (void **) &trimesh); // map buffer for writing by host

  const float ci2f = 1.0f / 255.0f;
  for (ind=0,i=0,j=0; ind<numfacets; ind++,i+=9,j+=12) {
    // transform to eye coordinates
    wtrans.multpoint3d(v + i    , (float*) &trimesh[ind].v0);
    wtrans.multpoint3d(v + i + 3, (float*) &trimesh[ind].v1);
    wtrans.multpoint3d(v + i + 6, (float*) &trimesh[ind].v2);

    wtrans.multnorm3d(n + i    , (float*) &trimesh[ind].n0);
    wtrans.multnorm3d(n + i + 3, (float*) &trimesh[ind].n1);
    wtrans.multnorm3d(n + i + 6, (float*) &trimesh[ind].n2);

    // conversion from GLubyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
    // float = c/(2^8-1)
    float col[9];
    col[0] = c[j     ] * ci2f;
    col[1] = c[j +  1] * ci2f;
    col[2] = c[j +  2] * ci2f;
    col[3] = c[j +  4] * ci2f;
    col[4] = c[j +  5] * ci2f;
    col[5] = c[j +  6] * ci2f;
    col[6] = c[j +  8] * ci2f;
    col[7] = c[j +  9] * ci2f;
    col[8] = c[j + 10] * ci2f;

    vec_copy((float*) &trimesh[ind].c0, &col[0]);
    vec_copy((float*) &trimesh[ind].c1, &col[3]);
    vec_copy((float*) &trimesh[ind].c2, &col[6]);
  }
  rtBufferUnmap(buf); // triangle list is complete, unmap buffer

  RTERR( rtGeometryCreate(ctx, &geom) );
  RTERR( rtGeometrySetPrimitiveCount(geom, numfacets) );
  RTERR( rtGeometrySetBoundingBoxProgram(geom, tricolor_bbox_pgm) );
  RTERR( rtGeometrySetIntersectionProgram(geom, tricolor_isct_pgm) );

  // this buffer is associated only with this particular geometry node
  RTvariable buf_v;
  RTERR( rtGeometryDeclareVariable(geom, "tricolor_buffer", &buf_v) );
  RTERR( rtVariableSetObject(buf_v, buf) );

  // create a geometry instance and bind materials to this geometry
  RTERR( rtGeometryInstanceCreate(ctx, &instance) );
  RTERR( rtGeometryInstanceSetGeometry(instance, geom) );

  set_material(instance, matindex, NULL);

  append_objects(buf, geom, instance);
}


#if defined(ORT_USERTXAPIS)
void OptiXRenderer::trimesh_n3b_v3f_hwtri(Matrix4 & wtrans, 
                                          float *uniform_color, 
                                          char *n, float *v, int numfacets, 
                                          int matindex) {
  if (!context_created) return;
  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) creating trimesh_n3b_v3f_hwtri: %d...\n", numfacets);
  trimesh_n3b_v3f_cnt += numfacets;

  RTbuffer vbuf, nbuf, cbuf;
  RTgeometryinstance instance_hwtri;
  RTgeometrytriangles geom_hwtri;

  // Create and fill vertex/normal/color buffers
  float3 *vertices; 
  uint4 *normals;
  uchar4 *colors;
  hwtri_alloc_bufs_v3f_n4u4_c4u(ctx, numfacets, vbuf, vertices, nbuf, normals, 
                                cbuf, 1, colors, uniform_color);

  const float ci2f = 1.0f / 255.0f;
  const float cn2f = 1.0f / 127.5f;
  int i, ind, tcnt;
  for (ind=0,i=0,tcnt=0; ind<numfacets; ind++,i+=9) {
    float norm[9];
    int taddr = 3 * tcnt;

    // transform to eye coordinates
    wtrans.multpoint3d(v + i    , (float*) &vertices[taddr + 0]);
    wtrans.multpoint3d(v + i + 3, (float*) &vertices[taddr + 1]);
    wtrans.multpoint3d(v + i + 6, (float*) &vertices[taddr + 2]);

    // Compute geometric normal, detect and cull degenerate triangles
    float3 Ng;
    if (hwtri_test_calc_Ngeom(&vertices[taddr], Ng)) {
      continue; // cull any triangle that fails degeneracy tests
    }

    // conversion from GLbyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
    // float = (2c+1)/(2^8-1)
    norm[0] = n[i    ] * cn2f + ci2f;
    norm[1] = n[i + 1] * cn2f + ci2f;
    norm[2] = n[i + 2] * cn2f + ci2f;
    norm[3] = n[i + 3] * cn2f + ci2f;
    norm[4] = n[i + 4] * cn2f + ci2f;
    norm[5] = n[i + 5] * cn2f + ci2f;
    norm[6] = n[i + 6] * cn2f + ci2f;
    norm[7] = n[i + 7] * cn2f + ci2f;
    norm[8] = n[i + 8] * cn2f + ci2f;

    // conversion to GLbyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
    float3 n0, n1, n2;
    wtrans.multnorm3d(&norm[0], (float*) &n0);
    wtrans.multnorm3d(&norm[3], (float*) &n1);
    wtrans.multnorm3d(&norm[6], (float*) &n2);

    // Pack normals
    normals[tcnt].x = packNormal(Ng);
    normals[tcnt].y = packNormal(n0);
    normals[tcnt].z = packNormal(n1);
    normals[tcnt].w = packNormal(n2);

    tcnt++; // count non-culled triangles
  }

  rtBufferUnmap(vbuf);
  rtBufferUnmap(nbuf);
  rtBufferUnmap(cbuf);

  RTERR( rtGeometryTrianglesCreate(ctx, &geom_hwtri) );
  RTERR( rtGeometryTrianglesSetTriangles(geom_hwtri, tcnt, vbuf, 
                                         0, sizeof(float3), RT_FORMAT_FLOAT3, 
                                         RT_GEOMETRYBUILDFLAGS_RELEASE_BUFFERS) );

  // create a geometry instance and bind materials to this geometry
  RTERR( rtGeometryInstanceCreate(ctx, &instance_hwtri) );
  RTERR( rtGeometryInstanceSetGeometryTriangles(instance_hwtri, geom_hwtri) );

  // Enable per-vertex normals, disable per-vertex colors (use uniform color) 
  hwtri_set_vertex_flags(ctx, instance_hwtri, nbuf, cbuf, 1, 0);

  // We have to pass the explicit hardware triangle parameter for this geometry
  set_material(instance_hwtri, matindex, uniform_color, 1);

  // The vertex buffer is released automatically after construction. 
  // We need to keep track of normal and color buffers for ourselves.
  append_objects(nbuf, cbuf, geom_hwtri, instance_hwtri);
}
#endif


void OptiXRenderer::trimesh_n3b_v3f(Matrix4 & wtrans, float *uniform_color, 
                                    char *n, float *v, int numfacets, 
                                    int matindex) {
  if (!context_created) return;

#if defined(ORT_USERTXAPIS)
  // OptiX RTX hardware-accelerated triangle API
  if (hwtri_enabled) {
    trimesh_n3b_v3f_hwtri(wtrans, uniform_color, n, v, numfacets, matindex);
    return;
  }
#endif

  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) creating trimesh_n3b_v3f: %d...\n", numfacets);
  trimesh_n3b_v3f_cnt += numfacets;

  int i, ind;
  RTbuffer buf;
  RTgeometry geom;
  RTgeometryinstance instance;
  vmd_trimesh_n3b_v3f *trimesh;

  // create and fill the OptiX trimesh memory buffer
  rtBufferCreate(ctx, RT_BUFFER_INPUT, &buf);
  rtBufferSetFormat(buf, RT_FORMAT_USER);
  rtBufferSetElementSize(buf, sizeof(vmd_trimesh_n3b_v3f));
  rtBufferSetSize1D(buf, numfacets);
  // rtBufferValidate(buf);
  rtBufferMap(buf, (void **) &trimesh); // map buffer for writing by host

  const float ci2f = 1.0f / 255.0f;
  const float cn2f = 1.0f / 127.5f;
  for (ind=0,i=0; ind<numfacets; ind++,i+=9) {
    float norm[9];

    // transform to eye coordinates
    wtrans.multpoint3d(v + i    , (float*) &trimesh[ind].v0);
    wtrans.multpoint3d(v + i + 3, (float*) &trimesh[ind].v1);
    wtrans.multpoint3d(v + i + 6, (float*) &trimesh[ind].v2);

    // conversion from GLbyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
    // float = (2c+1)/(2^8-1)
    norm[0] = n[i    ] * cn2f + ci2f;
    norm[1] = n[i + 1] * cn2f + ci2f;
    norm[2] = n[i + 2] * cn2f + ci2f;
    norm[3] = n[i + 3] * cn2f + ci2f;
    norm[4] = n[i + 4] * cn2f + ci2f;
    norm[5] = n[i + 5] * cn2f + ci2f;
    norm[6] = n[i + 6] * cn2f + ci2f;
    norm[7] = n[i + 7] * cn2f + ci2f;
    norm[8] = n[i + 8] * cn2f + ci2f;

    // conversion to GLbyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
    float3 tmpn;
    wtrans.multnorm3d(&norm[0], (float*) &tmpn);
    tmpn = tmpn * 127.5f - 0.5f;
    trimesh[ind].n0 = make_char4(tmpn.x, tmpn.y, tmpn.z, 0);
    wtrans.multnorm3d(&norm[3], (float*) &tmpn);
    tmpn = tmpn * 127.5f - 0.5f;
    trimesh[ind].n1 = make_char4(tmpn.x, tmpn.y, tmpn.z, 0);
    wtrans.multnorm3d(&norm[6], (float*) &tmpn);
    tmpn = tmpn * 127.5f - 0.5f;
    trimesh[ind].n2 = make_char4(tmpn.x, tmpn.y, tmpn.z, 0);
  }
  rtBufferUnmap(buf); // triangle list is complete, unmap buffer

  RTERR( rtGeometryCreate(ctx, &geom) );
  RTERR( rtGeometrySetPrimitiveCount(geom, numfacets) );
  RTERR( rtGeometrySetBoundingBoxProgram(geom, trimesh_n3b_v3f_bbox_pgm) );
  RTERR( rtGeometrySetIntersectionProgram(geom, trimesh_n3b_v3f_isct_pgm) );

  // this buffer is associated only with this particular geometry node
  RTvariable buf_v;
  RTERR( rtGeometryDeclareVariable(geom, "trimesh_n3b_v3f_buffer", &buf_v) );
  RTERR( rtVariableSetObject(buf_v, buf) );

  // create a geometry instance and bind materials to this geometry
  RTERR( rtGeometryInstanceCreate(ctx, &instance) );
  RTERR( rtGeometryInstanceSetGeometry(instance, geom) );

  set_material(instance, matindex, uniform_color);

  append_objects(buf, geom, instance);
}


#if defined(ORT_USERTXAPIS)
void OptiXRenderer::trimesh_n3f_v3f_hwtri(Matrix4 & wtrans, 
                                          float *uniform_color,
                                          float *n, float *v, int numfacets,
                                          int matindex) {
  if (!context_created) return;
  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) creating trimesh_n3f_v3f_hwtri: %d...\n", numfacets);
  trimesh_n3f_v3f_cnt += numfacets;

  RTbuffer vbuf, nbuf, cbuf;
  RTgeometryinstance instance_hwtri;
  RTgeometrytriangles geom_hwtri;

  // Create and fill vertex/normal/color buffers
  float3 *vertices; 
  uint4 *normals;
  uchar4 *colors;
  hwtri_alloc_bufs_v3f_n4u4_c4u(ctx, numfacets, vbuf, vertices, nbuf, normals, 
                                cbuf, 1, colors, uniform_color);

  float3 n0, n1, n2;
  int i, tcnt;
  for (i=0, tcnt=0; i < numfacets; i++) {
    int taddr = 3 * tcnt;

    // transform to eye coordinates
    wtrans.multpoint3d(v + 9 * i + 0, (float*) &vertices[taddr + 0]);
    wtrans.multpoint3d(v + 9 * i + 3, (float*) &vertices[taddr + 1]);
    wtrans.multpoint3d(v + 9 * i + 6, (float*) &vertices[taddr + 2]);
   
    // Compute geometric normal, detect and cull degenerate triangles
    float3 Ng;
    if (hwtri_test_calc_Ngeom(&vertices[taddr], Ng)) {
      continue; // cull any triangle that fails degeneracy tests
    }

    wtrans.multnorm3d(n + 9 * i + 0, (float*) &n0.x);
    wtrans.multnorm3d(n + 9 * i + 3, (float*) &n1.x);
    wtrans.multnorm3d(n + 9 * i + 6, (float*) &n2.x);

    // Pack normals
    normals[i].x = packNormal(Ng);
    normals[i].y = packNormal(n0);
    normals[i].z = packNormal(n1);
    normals[i].w = packNormal(n2);

    tcnt++; // count non-culled triangles
  }

  rtBufferUnmap(vbuf);
  rtBufferUnmap(nbuf);
  rtBufferUnmap(cbuf);

  RTERR( rtGeometryTrianglesCreate(ctx, &geom_hwtri) );
  RTERR( rtGeometryTrianglesSetTriangles(geom_hwtri, tcnt, vbuf, 
                                         0, sizeof(float3), RT_FORMAT_FLOAT3, 
                                         RT_GEOMETRYBUILDFLAGS_RELEASE_BUFFERS) );

  // create a geometry instance and bind materials to this geometry
  RTERR( rtGeometryInstanceCreate(ctx, &instance_hwtri) );
  RTERR( rtGeometryInstanceSetGeometryTriangles(instance_hwtri, geom_hwtri) );

  // Enable per-vertex normals, disable per-vertex colors (use uniform color) 
  hwtri_set_vertex_flags(ctx, instance_hwtri, nbuf, cbuf, 1, 0);

  // We have to pass the explicit hardware triangle parameter for this geometry
  set_material(instance_hwtri, matindex, uniform_color, 1);

  // The vertex buffer is released automatically after construction. 
  // We need to keep track of normal and color buffers for ourselves.
  append_objects(nbuf, cbuf, geom_hwtri, instance_hwtri);
}
#endif


void OptiXRenderer::trimesh_n3f_v3f(Matrix4 & wtrans, float *uniform_color, 
                                    float *n, float *v, int numfacets, 
                                    int matindex) {
  if (!context_created) return;

#if defined(ORT_USERTXAPIS)
  // OptiX RTX hardware-accelerated triangle API
  if (hwtri_enabled) {
    trimesh_n3f_v3f_hwtri(wtrans, uniform_color, n, v, numfacets, matindex);
    return;
  }
#endif

  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) creating trimesh_n3f_v3f: %d...\n", numfacets);
  trimesh_n3f_v3f_cnt += numfacets;

  int i, ind;
  RTbuffer buf;
  RTgeometry geom;
  RTgeometryinstance instance;
  vmd_trimesh_n3f_v3f *trimesh;

  // create and fill the OptiX trimesh memory buffer
  rtBufferCreate(ctx, RT_BUFFER_INPUT, &buf);
  rtBufferSetFormat(buf, RT_FORMAT_USER);
  rtBufferSetElementSize(buf, sizeof(vmd_trimesh_n3f_v3f));
  rtBufferSetSize1D(buf, numfacets);
  // rtBufferValidate(buf);
  rtBufferMap(buf, (void **) &trimesh); // map buffer for writing by host

  for (ind=0,i=0; ind<numfacets; ind++,i+=9) {
    // transform to eye coordinates
    wtrans.multpoint3d(v + i    , (float*) &trimesh[ind].v0);
    wtrans.multpoint3d(v + i + 3, (float*) &trimesh[ind].v1);
    wtrans.multpoint3d(v + i + 6, (float*) &trimesh[ind].v2);

    wtrans.multnorm3d(n + i    , (float*) &trimesh[ind].n0);
    wtrans.multnorm3d(n + i + 3, (float*) &trimesh[ind].n1);
    wtrans.multnorm3d(n + i + 6, (float*) &trimesh[ind].n2);
  }
  rtBufferUnmap(buf); // triangle list is complete, unmap buffer

  RTERR( rtGeometryCreate(ctx, &geom) );
  RTERR( rtGeometrySetPrimitiveCount(geom, numfacets) );
  RTERR( rtGeometrySetBoundingBoxProgram(geom, trimesh_n3f_v3f_bbox_pgm) );
  RTERR( rtGeometrySetIntersectionProgram(geom, trimesh_n3f_v3f_isct_pgm) );

  // this buffer is associated only with this particular geometry node
  RTvariable buf_v;
  RTERR( rtGeometryDeclareVariable(geom, "trimesh_n3f_v3f_buffer", &buf_v) );
  RTERR( rtVariableSetObject(buf_v, buf) );

  // create a geometry instance and bind materials to this geometry
  RTERR( rtGeometryInstanceCreate(ctx, &instance) );
  RTERR( rtGeometryInstanceSetGeometry(instance, geom) );

  set_material(instance, matindex, uniform_color);

  append_objects(buf, geom, instance);
}


#if defined(ORT_USERTXAPIS)
void OptiXRenderer::trimesh_v3f_hwtri(Matrix4 & wtrans, float *uniform_color,
                                      float *v, int numfacets, int matindex) {
  if (!context_created) return;
  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) creating trimesh_v3f_hwtri: %d...\n", numfacets);
  trimesh_v3f_cnt += numfacets;

  RTbuffer vbuf, nbuf, cbuf;
  RTgeometryinstance instance_hwtri;
  RTgeometrytriangles geom_hwtri;
  
  // Create and fill vertex/normal/color buffers
  float3 *vertices; 
  uint4 *normals;
  uchar4 *colors;
  hwtri_alloc_bufs_v3f_n4u4_c4u(ctx, numfacets, vbuf, vertices, nbuf, normals, 
                                cbuf, 1, colors, uniform_color);

  int i, tcnt;
  for (i=0, tcnt=0; i < numfacets; i++) {
    int taddr = 3 * tcnt;

    // transform to eye coordinates
    wtrans.multpoint3d(v + 9 * i + 0, (float*) &vertices[taddr + 0]);
    wtrans.multpoint3d(v + 9 * i + 3, (float*) &vertices[taddr + 1]);
    wtrans.multpoint3d(v + 9 * i + 6, (float*) &vertices[taddr + 2]);

    // Compute geometric normal, detect and cull degenerate triangles
    float3 Ng;
    if (hwtri_test_calc_Ngeom(&vertices[taddr], Ng)) {
      continue; // cull any triangle that fails degeneracy tests
    }

    // Pack normal (we don't have per-vertex normals, so leave them empty)
    normals[i].x = packNormal(Ng);
    // XXX we could initialize the others for paranoia's sake...
    // normals[i].y = normals[i].z = normals[i].w = normals[i].x;

    tcnt++; // count non-culled triangles
  }

  rtBufferUnmap(vbuf);
  rtBufferUnmap(nbuf);
  rtBufferUnmap(cbuf);

  RTERR( rtGeometryTrianglesCreate(ctx, &geom_hwtri) );
  RTERR( rtGeometryTrianglesSetTriangles(geom_hwtri, tcnt, vbuf, 
                                         0, sizeof(float3), RT_FORMAT_FLOAT3, 
                                         RT_GEOMETRYBUILDFLAGS_RELEASE_BUFFERS) );

  // create a geometry instance and bind materials to this geometry
  RTERR( rtGeometryInstanceCreate(ctx, &instance_hwtri) );
  RTERR( rtGeometryInstanceSetGeometryTriangles(instance_hwtri, geom_hwtri) );

  // Disable per-vertex normals, disable per-vertex colors (use uniform color) 
  hwtri_set_vertex_flags(ctx, instance_hwtri, nbuf, cbuf, 0, 0);

  // We have to pass the explicit hardware triangle parameter for this geometry
  set_material(instance_hwtri, matindex, uniform_color, 1);

  // The vertex buffer is released automatically after construction. 
  // We need to keep track of normal and color buffers for ourselves.
  append_objects(nbuf, cbuf, geom_hwtri, instance_hwtri);
}
#endif


void OptiXRenderer::trimesh_v3f(Matrix4 & wtrans, float *uniform_color, 
                                float *v, int numfacets, int matindex) {
  if (!context_created) return;

#if defined(ORT_USERTXAPIS)
  // OptiX RTX hardware-accelerated triangle API
  if (hwtri_enabled) {
    trimesh_v3f_hwtri(wtrans, uniform_color, v, numfacets, matindex);
    return;
  }
#endif

  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) creating trimesh_v3f: %d...\n", numfacets);
  trimesh_v3f_cnt += numfacets;

  long i, ind;
  RTbuffer buf;
  RTgeometry geom;
  RTgeometryinstance instance;
  vmd_trimesh_v3f *trimesh;

  // create and fill the OptiX trimesh memory buffer
  rtBufferCreate(ctx, RT_BUFFER_INPUT, &buf);
  rtBufferSetFormat(buf, RT_FORMAT_USER);
  rtBufferSetElementSize(buf, sizeof(vmd_trimesh_v3f));
  rtBufferSetSize1D(buf, numfacets);
  // rtBufferValidate(buf);
  rtBufferMap(buf, (void **) &trimesh); // map buffer for writing by host

  for (ind=0,i=0; ind<numfacets; ind++,i+=9) {
    // transform to eye coordinates
    wtrans.multpoint3d(v + i    , (float*) &trimesh[ind].v0);
    wtrans.multpoint3d(v + i + 3, (float*) &trimesh[ind].v1);
    wtrans.multpoint3d(v + i + 6, (float*) &trimesh[ind].v2);
  }
  rtBufferUnmap(buf); // triangle list is complete, unmap buffer

  RTERR( rtGeometryCreate(ctx, &geom) );
  RTERR( rtGeometrySetPrimitiveCount(geom, numfacets) );
  RTERR( rtGeometrySetBoundingBoxProgram(geom, trimesh_v3f_bbox_pgm) );
  RTERR( rtGeometrySetIntersectionProgram(geom, trimesh_v3f_isct_pgm) );

  // this buffer is associated only with this particular geometry node
  RTvariable buf_v;
  RTERR( rtGeometryDeclareVariable(geom, "trimesh_v3f_buffer", &buf_v) );
  RTERR( rtVariableSetObject(buf_v, buf) );

  // create a geometry instance and bind materials to this geometry
  RTERR( rtGeometryInstanceCreate(ctx, &instance) );
  RTERR( rtGeometryInstanceSetGeometry(instance, geom) );

  set_material(instance, matindex, uniform_color);

  append_objects(buf, geom, instance);
}


#if defined(ORT_USERTXAPIS)
void OptiXRenderer::tristrip_hwtri(Matrix4 & wtrans, int numverts, 
                                   const float * cnv,
                                   int numstrips, const int *vertsperstrip,
                                   const int *facets, int matindex) {
  if (!context_created) return;
  int i;
  int numfacets = 0;
  for (i=0; i<numstrips; i++) 
    numfacets += (vertsperstrip[i] - 2);  

  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) creating tristrip_hwtri: %d...\n", numfacets);
  tricolor_cnt += numfacets;

  RTbuffer vbuf, nbuf, cbuf;
  RTgeometryinstance instance_hwtri;
  RTgeometrytriangles geom_hwtri;

  // Create and fill vertex/normal/color buffers
  float3 *vertices; 
  uint4 *normals;
  uchar4 *colors;
  hwtri_alloc_bufs_v3f_n4u4_c4u(ctx, numfacets, vbuf, vertices, nbuf, normals, 
                                cbuf, numfacets * 3, colors, NULL);

  // render triangle strips one triangle at a time
  // triangle winding order is:
  //   v0, v1, v2, then v2, v1, v3, then v2, v3, v4, etc.
  int strip, t, v = 0;
  int stripaddr[2][3] = { {0, 1, 2}, {1, 0, 2} };

  // loop over all of the triangle strips
  int tcnt=0; // set triangle index to 0
  for (strip=0; strip < numstrips; strip++) {
    // loop over all triangles in this triangle strip
    for (t = 0; t < (vertsperstrip[strip] - 2); t++) {
      int taddr = 3 * tcnt;

      // render one triangle, using lookup table to fix winding order
      int v0 = facets[v + (stripaddr[t & 0x01][0])] * 10;
      int v1 = facets[v + (stripaddr[t & 0x01][1])] * 10;
      int v2 = facets[v + (stripaddr[t & 0x01][2])] * 10;

      // transform to eye coordinates
      wtrans.multpoint3d(cnv + v0 + 7, (float*) &vertices[taddr + 0]);
      wtrans.multpoint3d(cnv + v1 + 7, (float*) &vertices[taddr + 1]);
      wtrans.multpoint3d(cnv + v2 + 7, (float*) &vertices[taddr + 2]);

      // Compute geometric normal, detect and cull degenerate triangles
      float3 Ng;
      if (hwtri_test_calc_Ngeom(&vertices[taddr], Ng)) {
        v++;      // move on to next vertex
        continue; // cull any triangle that fails degeneracy tests
      }

      float3 n0, n1, n2;
      wtrans.multnorm3d(cnv + v0 + 4, (float*) &n0);
      wtrans.multnorm3d(cnv + v1 + 4, (float*) &n1);
      wtrans.multnorm3d(cnv + v2 + 4, (float*) &n2);

      // Pack normals
      normals[tcnt].x = packNormal(Ng);
      normals[tcnt].y = packNormal(n0);
      normals[tcnt].z = packNormal(n1);
      normals[tcnt].w = packNormal(n2);

      // convert color format
      colors[taddr + 0].x = cnv[v0 + 0] * 255.0f;
      colors[taddr + 0].y = cnv[v0 + 1] * 255.0f;
      colors[taddr + 0].z = cnv[v0 + 2] * 255.0f;

      colors[taddr + 1].x = cnv[v1 + 0] * 255.0f;
      colors[taddr + 1].y = cnv[v1 + 1] * 255.0f;
      colors[taddr + 1].z = cnv[v1 + 2] * 255.0f;

      colors[taddr + 2].x = cnv[v2 + 0] * 255.0f;
      colors[taddr + 2].y = cnv[v2 + 1] * 255.0f;
      colors[taddr + 2].z = cnv[v2 + 2] * 255.0f;

      v++;    // move on to next vertex
      tcnt++; // count non-culled triangles
    }
    v+=2; // last two vertices are already used by last triangle
  }

  rtBufferUnmap(vbuf);
  rtBufferUnmap(nbuf);
  rtBufferUnmap(cbuf);

  RTERR( rtGeometryTrianglesCreate(ctx, &geom_hwtri) );
  RTERR( rtGeometryTrianglesSetTriangles(geom_hwtri, tcnt, vbuf, 
                                         0, sizeof(float3), RT_FORMAT_FLOAT3, 
                                         RT_GEOMETRYBUILDFLAGS_RELEASE_BUFFERS) );

  // create a geometry instance and bind materials to this geometry
  RTERR( rtGeometryInstanceCreate(ctx, &instance_hwtri) );
  RTERR( rtGeometryInstanceSetGeometryTriangles(instance_hwtri, geom_hwtri) );

  // Enable per-vertex normals, enable per-vertex colors
  hwtri_set_vertex_flags(ctx, instance_hwtri, nbuf, cbuf, 1, 1);

  // We have to pass the explicit hardware triangle parameter for this geometry
  set_material(instance_hwtri, matindex, NULL, 1);

  // The vertex buffer is released automatically after construction. 
  // We need to keep track of normal and color buffers for ourselves.
  append_objects(nbuf, cbuf, geom_hwtri, instance_hwtri);
}
#endif


void OptiXRenderer::tristrip(Matrix4 & wtrans, int numverts, const float * cnv,
                             int numstrips, const int *vertsperstrip,
                             const int *facets, int matindex) {
  if (!context_created) return;
  int i;
  int numfacets = 0;
  for (i=0; i<numstrips; i++) 
    numfacets += (vertsperstrip[i] - 2);  

#if defined(ORT_USERTXAPIS)
  // OptiX RTX hardware-accelerated triangle API
  if (hwtri_enabled) {
    tristrip_hwtri(wtrans, numverts, cnv, numstrips, 
                   vertsperstrip, facets, matindex);
    return;
  }
#endif

  if (verbose == RT_VERB_DEBUG) printf("OptiXRenderer) creating tristrip: %d...\n", numfacets);
  tricolor_cnt += numfacets;

  RTbuffer buf;
  RTgeometry geom;
  RTgeometryinstance instance;
  vmd_tricolor *trimesh;

  // create and fill the OptiX trimesh memory buffer
  rtBufferCreate(ctx, RT_BUFFER_INPUT, &buf);
  rtBufferSetFormat(buf, RT_FORMAT_USER);
  rtBufferSetElementSize(buf, sizeof(vmd_tricolor));
  rtBufferSetSize1D(buf, numfacets);
  // rtBufferValidate(buf);
  rtBufferMap(buf, (void **) &trimesh); // map buffer for writing by host

  // render triangle strips one triangle at a time
  // triangle winding order is:
  //   v0, v1, v2, then v2, v1, v3, then v2, v3, v4, etc.
  int strip, t, v = 0;
  int stripaddr[2][3] = { {0, 1, 2}, {1, 0, 2} };

  // loop over all of the triangle strips
  i=0; // set triangle index to 0
  for (strip=0; strip < numstrips; strip++) {
    // loop over all triangles in this triangle strip
    for (t = 0; t < (vertsperstrip[strip] - 2); t++) {
      // render one triangle, using lookup table to fix winding order
      int v0 = facets[v + (stripaddr[t & 0x01][0])] * 10;
      int v1 = facets[v + (stripaddr[t & 0x01][1])] * 10;
      int v2 = facets[v + (stripaddr[t & 0x01][2])] * 10;

      // transform to eye coordinates
      wtrans.multpoint3d(cnv + v0 + 7, (float*) &trimesh[i].v0);
      wtrans.multpoint3d(cnv + v1 + 7, (float*) &trimesh[i].v1);
      wtrans.multpoint3d(cnv + v2 + 7, (float*) &trimesh[i].v2);

      wtrans.multnorm3d(cnv + v0 + 4, (float*) &trimesh[i].n0);
      wtrans.multnorm3d(cnv + v1 + 4, (float*) &trimesh[i].n1);
      wtrans.multnorm3d(cnv + v2 + 4, (float*) &trimesh[i].n2);

      vec_copy((float*) &trimesh[i].c0, cnv + v0);
      vec_copy((float*) &trimesh[i].c1, cnv + v1);
      vec_copy((float*) &trimesh[i].c2, cnv + v2);

      v++; // move on to next vertex
      i++; // next triangle
    }
    v+=2; // last two vertices are already used by last triangle
  }
  rtBufferUnmap(buf); // triangle list is complete, unmap buffer

  RTERR( rtGeometryCreate(ctx, &geom) );
  RTERR( rtGeometrySetPrimitiveCount(geom, numfacets) );
  RTERR( rtGeometrySetBoundingBoxProgram(geom, tricolor_bbox_pgm) );
  RTERR( rtGeometrySetIntersectionProgram(geom, tricolor_isct_pgm) );

  // this buffer is associated only with this particular geometry node
  RTvariable buf_v;
  RTERR( rtGeometryDeclareVariable(geom, "tricolor_buffer", &buf_v) );
  RTERR( rtVariableSetObject(buf_v, buf) );

  // create a geometry instance and bind materials to this geometry
  RTERR( rtGeometryInstanceCreate(ctx, &instance) );
  RTERR( rtGeometryInstanceSetGeometry(instance, geom) );

  set_material(instance, matindex, NULL);

  append_objects(buf, geom, instance);
}


#if !defined(VMDOPENGL)
// A hack to prevent VMD from having to be linked to libGL.so to resolve
// OptiX dependencies for OpenGL interop, e.g. when compiling on
// a supercomputer/cluster lacking OpenGL support (e.g. ORNL Titan):
//
// Linking  vmd_LINUXAMD64 ...
// /usr/lib64/libGL.so.1: undefined reference to `xcb_glx_set_client_info_arb'
// /usr/lib64/libGL.so.1: undefined reference to `xcb_glx_create_context_attribs_arb_checked'
// /usr/lib64/libGL.so.1: undefined reference to `xcb_glx_set_client_info_2arb'
// /usr/bin/ld: link errors found, deleting executable `vmd_LINUXAMD64'
// collect2: error: ld returned 1 exit status
// make: *** [vmd_LINUXAMD64] Error 1
//
extern "C" {
  typedef struct {
     unsigned int sequence;
  } xcb_void_cookie_t;
  static xcb_void_cookie_t fake_cookie = { 0 };
  xcb_void_cookie_t xcb_glx_set_client_info_arb(void) {
   return fake_cookie;
  }
  xcb_void_cookie_t xcb_glx_create_context_attribs_arb_checked(void) {
   return fake_cookie;
  }
  xcb_void_cookie_t xcb_glx_set_client_info_2arb(void) {
   return fake_cookie;
  }
}
#endif



