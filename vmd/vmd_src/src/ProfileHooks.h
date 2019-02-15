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
 *	$RCSfile: ProfileHooks.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.18 $	$Date: 2019/01/29 21:28:20 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Profiling utility routines and definitions.
 * Exemplary use of NVTX is shown here:
 *   https://devblogs.nvidia.com/cuda-pro-tip-generate-custom-application-profile-timelines-nvtx/
 *
 ***************************************************************************/
#ifndef PROFILEHOOKS_H
#define PROFILEHOOKS_H

#if defined(VMDNVTX)

#include <cuda_profiler_api.h>
#include <nvToolsExt.h>          // CUDA <= 9 has NVTX V2
// #include <nvtx3/nvToolsExt.h>   // CUDA >= 10 has NVTX V3
#include <cuda_runtime.h>
#include <pthread.h>

// C++ note: declaring const variables implies static (internal) linkage,
// and you have to explicitly specify "extern" to get external linkage.
const uint32_t VMD_nvtx_colors[] = {
  0xff00ff00, // 0 green
  0xff0000ff, // 1 blue
  0xffffff00, // 2 yellow
  0xffff00ff, // 3 purple
  0xff00ffff, // 4 teal
  0xffff0000, // 5 red
  0xffffffff, // 6 white
};
const int VMD_nvtx_colors_len = sizeof(VMD_nvtx_colors)/sizeof(uint32_t);


#define PROFILE_INITIALIZE() do { nvtxInitialize(NULL); } while(0) // terminate with semicolon

#define PROFILE_MARK(name,cid) \
  do { \
    /* create an ASCII event marker */ \
    /* nvtxMarkA(name); */ \
    int color_id = cid; \
    color_id = color_id % VMD_nvtx_colors_len; \
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = VMD_nvtx_colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxMarkEx(&eventAttrib); \
  } while(0) // terminate with semicolon


// start recording an event
#define PROFILE_PUSH_RANGE(name,cid) \
  do { \
    int color_id = cid; \
    color_id = color_id % VMD_nvtx_colors_len; \
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = VMD_nvtx_colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
  } while(0)  // must terminate with semi-colon

// stop recording an event
#define PROFILE_POP_RANGE(empty) \
  do { \
    nvtxRangePop(); \
  } while(0) // terminate with semicolon 

// embed event recording in class to automatically pop when destroyed
class VMD_NVTX_Tracer {
  public:
    VMD_NVTX_Tracer(const char *name, int cid = 0) { PROFILE_PUSH_RANGE(name, cid); }
    ~VMD_NVTX_Tracer() { PROFILE_POP_RANGE(); }
};

// include cid as part of the name
// call RANGE at beginning of function to push event recording
// destructor is automatically called on return to pop event recording
#define PROFILE_RANGE(name,cid) \
  VMD_NVTX_Tracer vmd_nvtx_tracer##cid(name,cid)
  // must terminate with semi-colon

#define PROFILE_MAIN_THREAD() \
  do { \
    nvtxNameOsThread(pthread_self(), "Main VMD thread"); \
  } while (0) // terminate with semicolon

#else

//
// Otherwise the NVTX profiling macros become no-ops.
//
#define PROFILE_MARK(name,cid)       do { } while(0) // terminate with semicolon
#define PROFILE_PUSH_RANGE(name,cid) do { } while(0) // terminate with semicolon
#define PROFILE_POP_RANGE()          do { } while(0) // terminate with semicolon
#define PROFILE_RANGE(namd,cid)      do { } while(0) // terminate with semicolon
#define PROFILE_MAIN_THREAD()        do { } while(0) // terminate with semicolon

#endif



//
// NVTX macros that may cause problems for the system profiler
//
#if 1 && defined(VMDNVTX)

#define PROFILE_START() \
  do { \
    cudaProfilerStart(); \
  } while (0) // terminate with semicolon

#define PROFILE_STOP() \
  do { \
    cudaDeviceSynchronize(); \
    cudaProfilerStop(); \
  } while (0) // terminate with semicolon


//
// An alternative to using NVTX to name threads is to use OS- or 
// runtime-specific threading APIs to assign string names independent
// of the profiling tools being used.  On Linux we can do this using
// pthread_setname_np() in combination with _GNU_SOURCE if we like.
// It is noteworthy that the pthread_setname_np() APIs are resitricted
// to 15 chars of thread name and 1 char for the terminating NUL char.
//
#define PROFILE_NAME_THREAD(name) \
  do { \
    nvtxNameOsThread(pthread_self(), name); \
  } while (0) // terminate with semicolon

#else

#define PROFILE_INITIALIZE()         do { } while(0) // terminate with semicolon
#define PROFILE_START()              do { } while(0) // terminate with semicolon
#define PROFILE_STOP()               do { } while(0) // terminate with semicolon
#define PROFILE_NAME_THREAD(name)    do { } while(0) // terminate with semicolon

#endif






#endif


