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
 *      $RCSfile: VMDApp.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.558 $      $Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************/
 
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <ctype.h> // for isalnum()

#include "VMDDisplayList.h"
#include "CoorPluginData.h"
#include "PluginMgr.h"
#include "MolFilePlugin.h"
#include "Matrix4.h"
#include "config.h"
#include "Inform.h"
#include "AtomSel.h"
#include "VMDTitle.h"
#include "DisplayDevice.h"
#include "PickList.h"
#include "PickModeList.h"
#include "MaterialList.h"
#include "Scene.h"
#include "CommandQueue.h"
#include "UIText.h"
#include "Stage.h"
#include "Axes.h"
#include "DisplayRocker.h"
#include "FPS.h"
#include "MoleculeList.h"
#include "Mouse.h"
#include "MobileInterface.h"
#include "Spaceball.h"
#ifdef WIN32
#include "Win32Joystick.h"
#endif
#include "GeometryList.h"
#include "FileRenderer.h"
#include "FileRenderList.h"
#include "CmdAnimate.h"
#include "CmdMol.h"
#include "CmdMaterial.h"
#include "CmdPlugin.h"
#include "CmdRender.h"   // to log the render commands we execute
#include "CmdTrans.h"
#include "CmdDisplay.h"
#include "CmdColor.h"
#include "CmdLabel.h"
#include "TextEvent.h"
#include "CmdMenu.h"
#include "P_UIVR.h"
#include "P_CmdTool.h"
#include "SymbolTable.h"
#include "VMDCollab.h"
#include "QMData.h"
#include "Orbital.h"
#include "QuickSurf.h"

#include "CUDAAccel.h"
#if defined(VMDOPENCL)
#include "OpenCLUtils.h"
#endif

#if defined(VMDNVENC)
#include "NVENCMgr.h"  // GPU-accelerated H.26[45] video encode/decode
#endif
#include "VideoStream.h" // GPU-accelerated H.26[45] video streaming 
#if defined(VMDLIBOPTIX)
#include "OptiXRenderer.h" // GPU-accelerated ray tracing engines
#endif

#if defined(VMDTHREADS) // System-specific threads code
#include "WKFThreads.h"
#endif

#ifdef MACVMD
#include "MacVMDDisplayDevice.h"
#endif

#ifdef VMDOPENGL        // OpenGL-specific files
#if defined(VMDFLTKOPENGL)
#include "FltkOpenGLDisplayDevice.h"
#else
#ifndef MACVMD
#include "OpenGLDisplayDevice.h"
#endif
#endif

#ifdef VMDCAVE           // CAVE-specific files
#include "cave_ogl.h"
#include "CaveDisplayDevice.h"
#include "CaveRoutines.h"
#include "CaveScene.h"
#endif // VMDCAVE
#endif // VMDOPENGL

// OpenGL Pbuffer off-screen rendering
#if defined(VMDOPENGLPBUFFER) || defined(VMDEGLPBUFFER)
#include "OpenGLPbufferDisplayDevice.h"
#endif

#include "VMDMenu.h"
#ifdef VMDFLTK           // If using FLTK GUI include FLTK objects
#include "FL/forms.H"

// New Fltk menus 
#include "MainFltkMenu.h"
#include "ColorFltkMenu.h"
#include "MaterialFltkMenu.h"
#include "DisplayFltkMenu.h"
#include "FileChooserFltkMenu.h"
#include "SaveTrajectoryFltkMenu.h"
#include "GeometryFltkMenu.h"
#include "GraphicsFltkMenu.h"
#include "RenderFltkMenu.h"
#include "ToolFltkMenu.h"

#endif // VMDFLTK

#ifdef VMDIMD
#include "IMDMgr.h"
#include "CmdIMD.h"
#endif

// FreeVR conflicts with the FLTK include files, so include it _after_ FLTK.
#ifdef VMDFREEVR     // FreeVR-specific files
#include "freevr.h"
#include "FreeVRDisplayDevice.h"
#include "FreeVRRoutines.h"
#include "FreeVRScene.h"
#endif

#if defined(VMDTKCON)
#include "vmdconsole.h"
#endif

#ifdef VMDMPI
#include "VMDMPI.h" 
#endif

#include "VMDApp.h"

// default atom selection
#define DEFAULT_ATOMSEL "all"

// XXX static data item
static unsigned long repserialnum;
static unsigned long texserialnum;

// static initialization
JString VMDApp::text_message;

#if defined(VMDXPLOR)  
VMDApp* VMDApp::obj = 0; ///< global object pointer used by VMD-XPLOR builds
#endif

VMDApp::VMDApp(int argc, char **argv, int mpion) {
#if defined(VMDXPLOR)  
  if (!obj) obj = this; ///< global object pointer used by VMD-XPLOR builds
#endif

  // initialize ALL member variables
  argc_m = argc;
  argv_m = (const char **)argv;
  mpienabled = mpion; // flag to enable/disable MPI functionality at runtime...
  menulist = NULL;
  nextMolID = 0; 
  stride_firsttime = 1;
  eofexit = 0;
  mouse = NULL;
  mobile = NULL;
  spaceball = NULL;
#ifdef WIN32
  win32joystick = NULL;
#endif
  vmdTitle = NULL; 
  fileRenderList = NULL;
  pluginMgr = NULL;
  uiText = NULL;
  uivr = NULL;
  uivs = NULL;
#ifdef VMDIMD
  imdMgr = NULL;
#endif
  display = NULL;
  scene = NULL;
  pickList = NULL;
  pickModeList = NULL;
  materialList = NULL;
  stage = NULL;
  axes = NULL;
  fps = NULL;
  commandQueue = NULL;
  moleculeList = NULL;
  geometryList = NULL;
  atomSelParser = NULL;
  anim = NULL;
  vmdcollab = NULL;
  thrpool = NULL;
  cuda = NULL;
  nvenc = NULL;
  strcpy(nodename, "");
  noderank  = 0; // init with MPI values later
  nodecount = 1; // init with MPI values later

  UpdateDisplay = TRUE;
  exitFlag = TRUE;
  ResetViewPending = FALSE;

  background_processing_clear();

  highlighted_molid  = highlighted_rep = -1;
}



// Permanently bind worker threads to CPUs to achieve 
// peak performance and reducing run-to-run jitter for
// graphics related algorithms.  This code simply performs
// a one-to-one mapping of threads to CPUs at present, and
// doesn't handle other cases with any greater logic yet.
extern "C" void * affinitize_threads(void *voidparms) {
  int tid, numthreads;
  wkf_threadpool_worker_getid(voidparms, &tid, &numthreads);
  int physcpus = wkf_thread_numphysprocessors();
  int setthreadaffinity=1; // enable affinity assignment by default

  // XXX need to add some kind of check to skip trying to set affinity
  //     if the OS, batch system, or other external CPU set management
  //     system has already restricted or set of CPUs we can use

#if defined(ARCH_SUMMIT)
  // XXX On the ORNL Summit system, the IBM LSF scheduler and 'jsrun'
  //     handle pinning of threads to CPU sockets/cores, and the application
  //     appears to have absolutely no control over this at present.   
  //     To prevent runtime errors when we try to map out and set
  //     the affinity of VMD CPU thread pool, we skip setting the affinity
  //     and live with whatever the scheduler has assigned.
  setthreadaffinity=0;
#endif

  if ((physcpus > 0) && setthreadaffinity) {
    int taffinity;
 
#if defined(ARCH_SUMMIT) || defined(ARCH_OPENPOWER)
#if 1
    int *affinitylist=NULL;
    int affinitycount=0;
    affinitylist = wkf_cpu_affinitylist(&affinitycount);

#if 0
    if (tid == 0) {
      printf("CPU affinity count: %d\n", affinitycount);
      int t;
      for (t=0; t<affinitycount; t++) {
        printf("thread[0] affinity[%d]: %d\n", t, affinitylist[t]);
      }
    }
#endif

    if (tid < affinitycount) {
      setthreadaffinity=1;
      taffinity=affinitylist[tid];
//      printf("thread[%d] affinity binding: %d\n", tid, taffinity);
    } else {
      setthreadaffinity=0;
//      printf("Skipping thread[%d] affinity binding\n", tid);
    }

    free(affinitylist);

#else

    // On POWER8/P9 SMT core hardware thread slots are numbered consecutively. 
    // To spread threads across all physical cores, threads are assigned by
    // multiplying threadID * SMT depth, modulo the total SMT core count.
    // Calculation of the active P8/P9 SMT state is a nightmarish exercise 
    // involving digging through tens of files in /proc and building up 
    // and cross referencing various CPU state tables, so far now we skip 
    // this entirely and use a hard-coded value that's applicable to the
    // HPC oriented systems in the field.  Alternately, we might use a 
    // scheme that decomposes the incoming thread affinity masks, since
    // that's another factor on production HPC systems.
    if (numthreads == physcpus) {
      taffinity=tid * 8 % physcpus;
    } else {
#if !defined(VMDMPI)
      if (tid == 0) 
        msgWarn << "Thread affinity binding: physical cpus != numthreads" << sendmsg;
#endif
      taffinity=tid * 8 % physcpus;
    }
#endif

#else
    // On Intel/AMD hardware, physically distinct CPU cores are numbered 
    // consecutively, and additional SMT "cores" appear as additional multiples
    // of the physical CPU core count.  It is therefore sufficient to 
    // use consecutive CPU indices to spread worker threads fully across CPUs
    // and SMT hardware threads
    if (numthreads == physcpus) {
      taffinity=tid; // set affinity of tid to same CPU id if numthreads == cpus
    } else {
#if !defined(VMDMPI)
      if (tid == 0) 
        msgWarn << "Thread affinity binding: physical cpus != numthreads" << sendmsg;
#endif

      taffinity=tid; // set affinity of tid to same CPU id if numthreads == cpus
    }
#endif

    // printf("Affinitizing thread[%d] to CPU[%d]...\n", tid, taffinity);
    wkf_thread_set_self_cpuaffinity(taffinity);
  }

  // mark CPU threads for display in profiling tools
  char threadname[1024];
  sprintf(threadname, "VMD CPU threadpool[%d]", tid);
  PROFILE_NAME_THREAD(threadname);

  return NULL;
}



// initialization routine for the library globals
int VMDApp::VMDinit(int argc, char **argv, const char *displaytype, 
                    int *displayLoc, int *displaySize) {
#if defined(VMDTHREADS)
  PROFILE_PUSH_RANGE("CPU thread pool init", 3);
  thrpool=wkf_threadpool_create(wkf_thread_numprocessors(), WKF_THREADPOOL_DEVLIST_CPUSONLY);

  // affinitize persistent threadpool to the available CPUs
  wkf_tasktile_t tile;
  memset(&tile, 0, sizeof(tile));
  wkf_threadpool_launch((wkf_threadpool_t *) thrpool, 
                        affinitize_threads, NULL, 1);
  PROFILE_POP_RANGE();
#endif

#if defined(VMDCUDA)
  PROFILE_PUSH_RANGE("GPU device pool init", 0);

  // register all usable CUDA GPU accelerator devices
  cuda = new CUDAAccel();

  PROFILE_POP_RANGE();
#endif

#if defined(VMDMPI)
  if (mpienabled) {
    PROFILE_PUSH_RANGE("MPI init", 5);

    // initialize MPI node info and print output here
    vmd_mpi_nodescan(&noderank, &nodecount, 
                     nodename, sizeof(nodename), 
                     (cuda != NULL) ? cuda->num_devices() : 0);

    PROFILE_POP_RANGE();
  }
#endif

  // only emit detailed GPU enumeration data if we're running on a single node
  if (nodecount == 1) {
    // CUDA and OpenCL GPU/multi-core acceleration
#if defined(VMDCUDA)
    cuda->print_cuda_devices();
#endif
#if defined(VMDOPENCL)
    vmd_cl_print_platform_info();
#endif
  }

#if defined(VMDNVENC)
  // It may eventually be desirable to relocate NVENC initialization after
  // OpenGL or Vulkan initialization has completed so that we have access 
  // to window handles and OpenGL or Vulkan context info at the point 
  // where video streaming is setup.  There don't presently appear to be
  // any APIs for graphics interop w/ NVENC.  We may need to instead use 
  // the GRID IFR library to achieve zero-copy streaming for OpenGL / Vulkan.
  // Direct use of NVENC seems most appropriate for ray tracing however, since
  // it allows CUDA interop.

  PROFILE_PUSH_RANGE("NVENC init", 0);
  nvenc = new NVENCMgr;
  int nvencrc;
  if ((nvencrc = nvenc->init()) == NVENCMGR_SUCCESS) {
    nvencrc = nvenc->open_session();
  }
  if (nvencrc == NVENCMGR_SUCCESS) {
    msgInfo << "NVENC GPU-accelerated video streaming available."
            << sendmsg;
  } else {
    msgInfo << "NVENC GPU-accelerated video streaming is not available." 
            << sendmsg;
  }
  PROFILE_POP_RANGE();
#endif


  PROFILE_PUSH_RANGE("UIObject init", 1);

  //
  // create commandQueue before all UIObjects!
  //
  commandQueue = new CommandQueue(); 


  // XXX This is currently only used for 64-bit MacOS X builds using Cocoa...
  // XXX Francois-Xavier Coudert has noted that this code path also corrected
  //     startup crashes he had been having with 32-bit builds using MacOS X
  //     10.7.x using FLTK 1.3.0.  It remains to be seen how stable the GUI is
  //     for the 32-bit build when the Cocoa workaround is active.  It has not
  //     been very stable for 64-bit builds thus far.
#if defined(ARCH_MACOSXX86_64)
  // Tcl/Tk _must_ be initialized before FLTK due to low-level implementation
  // details of Cocoa-based Tk and FLTK code required for 64-bit MacOS X.
  uiText = new UIText(this, (strcmp(displaytype, "TEXT") != 0), 0); // text user interface
#endif

  PROFILE_POP_RANGE();
  PROFILE_PUSH_RANGE("DisplayDevice init", 0);

  // initialize the display
  int *dloc = (displayLoc[0]  > 0 ? displayLoc  : (int *) NULL);

  int dsiz[2] = { 512, 512 };
  if (displaySize && displaySize[0] > 0 && displaySize[1] > 0) {
    dsiz[0] = displaySize[0];
    dsiz[1] = displaySize[1];
  }

  // initialize static data items
  repserialnum = 0; // initialize the display list serial number
  texserialnum = 0; // initialize the texture list serial number

#ifdef MACVMD
  display = new MacVMDDisplayDevice(this, dsiz);
#endif

  // check for a standard windowed display
  if (!strcmp(displaytype, "WIN") || !strcmp(displaytype, "OPENGL")) {
#if defined(VMDOPENGL)
#if defined(VMDFLTKOPENGL)
    display = new FltkOpenGLDisplayDevice(argc, argv, this, dsiz, dloc);
#else
    display = new OpenGLDisplayDevice;
    if (!display->init(argc, argv, this, dsiz, dloc)) {
      VMDexit("Unable to create OpenGL window.", 1, 7);
      return FALSE;
    } 
#endif
#endif
  }

#if defined(VMDOPENGLPBUFFER) || defined(VMDEGLPBUFFER)
  // check for OpenGL Pbuffer off-screen rendering context
  if (!strcmp(displaytype, "OPENGLPBUFFER")) {
    display = new OpenGLPbufferDisplayDevice;
    if (!display->init(argc, argv, this, dsiz, dloc)) {
      VMDexit("Unable to create OpenGL Pbuffer context.", 1, 7);
      return FALSE;
    } 
  }
#endif

#ifdef VMDCAVE
  if (!strcmp(displaytype, "CAVE") || !strcmp(displaytype, "CAVEFORMS")) {
    // The CAVE scene is allocated from shared memory and can be 
    // accessed by all of the rendering processes by virtue of its
    // class-specific operator new.
    scene = new CaveScene(this);
    msgInfo << "Cave shared memory scene created." << sendmsg;

    // the CAVE display device is created in process-private memory
    display = new CaveDisplayDevice;

    // set up pointers for cave_renderer, needs to be done before forking.
    set_cave_pointers(scene, display);

    // XXX this may cure a problem with specular highlights in CAVElib
    // by default CAVElib messes with the OpenGL modelview matrix, this
    // option prevents it from doing so, which should allow specular 
    // highlights to look right on multiple walls, but it breaks depth cueing.
    // CAVESetOption(CAVE_PROJ_USEMODELVIEW, 0);
    CAVESetOption(CAVE_GL_SAMPLES, 8); // enable multisample antialiasing
    CAVEInit();                        // fork off the rendering processes
    CAVEDisplay(cave_renderer, 0);     // set the fctn ptr for the renderers
    display->renderer_process = 0;     // this proc is the master process
    vmd_set_cave_is_initialized();     // flag successfull CAVE init

    if (!display->init(argc, argv, this, NULL, NULL)) {
      VMDexit("Unable to create CAVE display context.", 1, 7);
      return FALSE;
    }
    msgInfo << "CAVE Initialized" << sendmsg;
  }
#endif

#ifdef VMDFREEVR
  if (!strcmp(displaytype, "FREEVR") || !strcmp(displaytype, "FREEVRFORMS")) {
    // The FreeVR scene is allocated from shared memory and can be 
    // accessed by all of the rendering processes by virtue of its
    // class-specific operator new.
    scene = new FreeVRScene(this);
    msgInfo << "FreeVR shared memory scene created." << sendmsg;

    // the FreeVR display device is created in process-private memory
    display = new FreeVRDisplayDevice;

    // set up pointers for freevr_renderer, needs to be done before forking.
    set_freevr_pointers(scene, display);

    // set the function pointer for the per-screen renderers
    vrFunctionSetCallback(VRFUNC_ALL_DISPLAY, vrCallbackCreate(freevr_renderer, 1, &display));  
    vrStart();                         // fork off the rendering processes
    vrSystemSetName("VMD using FreeVR display form");
    vrSystemSetAuthors("John Stone, Justin Gullingsrud, Bill Sherman");
    vrSystemSetExtraInfo(VERSION_MSG);
    vrInputSet2switchDescription(0, "Terminate the FreeVR windows");
    vrInputSet2switchDescription(1, "Interface with the Graphics->Tools grab tool");
    vrInputSet2switchDescription(2, "Grab the world");
    vrInputSet2switchDescription(3, "Reset the world's position");
    vrInputSetValuatorDescription(0, "Rotate us in the world about the Y-axis");
    vrInputSetValuatorDescription(1, "Move us though the world in the direction we point");
    display->renderer_process = 0;     // this proc is the master process

    if (!display->init(argc, argv, this, NULL, NULL)) {
      VMDexit("Unable to create FreeVR display context.", 1, 7);
      return FALSE;
    }
    msgInfo << "FreeVR Initialized" << sendmsg;
  }
#endif

  // If no scene object has been created yet (e.g. CAVE/FreeVR), create one.
  if (!scene) {
    scene = new Scene;
  }

  // If a real display doesn't exist, create a stub display that eats commands
  if (!display) {
    display = new DisplayDevice("Default Display");
    if (!display->init(argc, argv, this, dsiz, dloc)) {
      VMDexit("Unable to create stub default display context.", 1, 7);
      return FALSE;
    }
  }

  // print any useful informational messages now that the display is setup
  if (display->get_num_processes() > 1) {
    msgInfo << "Started " << display->get_num_processes() 
            << " slave rendering processes." << sendmsg;
  }

  PROFILE_POP_RANGE();

  //
  // create other global objects for the program
  //
  PROFILE_PUSH_RANGE("Internal state init", 1);

  rocker = new DisplayRocker(&(scene->root));

  pluginMgr = new PluginMgr;

  atomSelParser = new SymbolTable;
  atomSelParser_init(atomSelParser);

  pickModeList = new PickModeList(this);
  pickList = new PickList(this);

  // create material list
  materialList = new MaterialList(&scene->root);

  // create other useful graphics objects
  axes = new Axes(display, &(scene->root));
  pickList->add_pickable(axes);
  
  fps = new FPS(display, &(scene->root));
  fps->off();

  // create the list of molecules (initially empty)
  moleculeList = new MoleculeList(this, scene);

  // create shared QuickSurf object used by all reps
  qsurf = new QuickSurf();

  anim = new Animation(this);
  anim->On();

  // create the list of geometry monitors (initially empty)
  geometryList = new GeometryList(this, &(scene->root));

  menulist = new NameList<VMDMenu *>;

  // If the text UI hasn't already been initialized then initialize it here.
  // Cocoa-based FLTK/Tk builds of VMD for 64-bit MacOS X must 
  // initialize Tcl/Tk first, so that FLTK works correctly.
  if (uiText == NULL)
    uiText = new UIText(this, display->supports_gui(), mpienabled); // text interface
  uiText->On();

  mouse = new Mouse(this); // mouse user interface
  mouse->On();

  mobile = new Mobile(this); // Smartphone/tablet network input device
  mobile->On();

  spaceball = new Spaceball(this); // Spaceball 6DOF input device
  spaceball->On();

#ifdef WIN32
  win32joystick = new Win32Joystick(this); // Win32 Joystick devices
  win32joystick->On();
#endif

#ifdef VMDIMD
  imdMgr = new IMDMgr(this);
  imdMgr->On();
#endif

#if 1 || defined(VMDNVPIPE)
  // only create a video streaming object if we have back-end encoder support
  uivs = new VideoStream(this);
  uivs->On();
#endif

  stage = new Stage(&(scene->root));
  pickList->add_pickable(stage);

  vmdcollab = new VMDCollab(this);
  vmdcollab->On();

  // make the classes which can render scenes to different file formats
  fileRenderList = new FileRenderList(this);

  display->queue_events(); // begin accepting UI events in graphics window

  // Create the menus; XXX currently this must be done _before_ calling
  // uiText->read_init() because otherwise the new menus created by the 
  // plugins loaded by the iit script won't be added to the main menu.
  activate_menus();

  PROFILE_POP_RANGE();
  PROFILE_PUSH_RANGE("Plugin shared obj init", 5);
  
  // XXX loading static plugins should be controlled by a startup option
  pluginMgr->load_static_plugins();

  // plugin_update must take place _after_ creation of uiText.
  plugin_update();

  PROFILE_POP_RANGE();
  PROFILE_PUSH_RANGE("Interp loop init", 3);

  VMDupdate(VMD_IGNORE_EVENTS); // flush cmd queue, prepare to enter event loop

  // Read the initialization code for the text interpreter.
  // We wait to do it now since that script might contain commands that use 
  // the event queue
  uiText->read_init();

  // If there's no text interpreter, show the main menu
#ifndef VMDTCL
#ifndef VMDPYTHON
  menu_show("main", 1);
#endif
#endif

#if defined(VMDTCL)
  // XXX hacks for Swift/T
  if (mpienabled) {
    commandQueue->runcommand(
      new TclEvalEvent("parallel swift_clone_communicator"));
  }
#endif
  
  PROFILE_POP_RANGE();

  // successful initialization.  Turn off the exit flag return success
  exitFlag = FALSE;
  return TRUE;
}

int VMDApp::num_menus() { return menulist->num(); }

int VMDApp::add_menu(VMDMenu *m) {
  if (menulist->typecode(m->get_name()) != -1) {
    msgErr << "Menu " << m->get_name() << " already exists." << sendmsg;
    return 0;  
  }
  menulist->add_name(m->get_name(), m);
  return 1;
}
 
int VMDApp::remove_menu(const char *name) {
  int id = menulist->typecode(name);
  if (id == -1) {
    msgErr << "Menu " << name << " does not exist." << sendmsg;
    return 0;  
  }
  NameList<VMDMenu *> *newmenulist = new NameList<VMDMenu *>;
  for (int i=0; i<menulist->num(); i++) {
    VMDMenu *menu = menulist->data(i);
    if (i == id) {
      delete menu;
    } else {
      newmenulist->add_name(menu->get_name(), menu);
    }
  }
  delete menulist;
  menulist = newmenulist;
  return 1;
}
 
void VMDApp::menu_add_extension(const char *shortname, const char *menu_path) {
  commandQueue->runcommand(new CmdMenuExtensionAdd(shortname,menu_path));
}

void VMDApp::menu_remove_extension(const char *shortname) {
  commandQueue->runcommand(new CmdMenuExtensionRemove(shortname));
}

const char *VMDApp::menu_name(int i) {
  return menulist->name(i);
}

int VMDApp::menu_id(const char *name) {
  return menulist->typecode(name);
}

int VMDApp::menu_status(const char *name) {
  int id = menulist->typecode(name);
  if (id == -1) return 0;
  return menulist->data(id)->active();
}

int VMDApp::menu_location(const char *name, int &x, int &y) {
  int id = menulist->typecode(name);
  if (id == -1) return 0;
  menulist->data(id)->where(x, y);
  return 1;
}

int VMDApp::menu_show(const char *name, int on) {
  int id = menulist->typecode(name);
  if (id == -1) return 0;
  VMDMenu *obj = menulist->data(name);
  if (on)
    obj->On();
  else
    obj->Off();
  commandQueue->runcommand(new CmdMenuShow(name, on));
  return 1;
}

int VMDApp::menu_move(const char *name, int x, int y) {
  int id = menulist->typecode(name);
  if (id == -1) return 0;
  menulist->data(id)->move(x, y);
  return 1;
}


int VMDApp::menu_select_mol(const char *name, int molno) {
  int id = menulist->typecode(name);
  if (id == -1) return 0;
  return menulist->data(id)->selectmol(molno);
}


// redraw the screen and update all things that need updating
int VMDApp::VMDupdate(int check_for_events) {
  PROFILE_PUSH_RANGE("VMDApp::VMDupdate()", 5);

  // clear background processing flag
  background_processing_clear();

  // see if there are any pending events; if so, get them
  if (check_for_events) {
    commandQueue->check_events();
#ifdef VMDGUI
#ifdef VMDFLTK
    // if we are using the FLTK library ...
    if (display->supports_gui()) {
#if defined(ARCH_MACOSX) && defined(VMDTCL)
      // don't call wait(0) since this causes Tcl/Tk to mishandle events
      Fl::flush();
#elif (defined(ARCH_MACOSXX86) || defined(ARCH_MACOSXX86_64)) && defined(VMDTCL)
      // starting with more recent revs of FLTK and Tcl/Tk, we should no
      // longer need to drop events
      Fl::wait(0);
#else
      Fl::wait(0);
#endif
    }
#endif
#endif
  } 
  
  // check if the user has requested to exit the program.
  if (exitFlag) {
    PROFILE_POP_RANGE();
    return FALSE;
  }
 
  commandQueue->execute_all(); // execute commands still in the queue

  //
  // if video streaming is active and we're a client with an active
  // connection, all local rendering is disabled..
  //
  if (uivs == NULL || !uivs->cli_connected()) {
    int needupdate = 0;
#if 1
    // Only prepare objects if display update is enabled
    // XXX avoid N^2 behavior when loading thousdands of molecules, 
    // don't prepare for drawing unless we have to.
    if (UpdateDisplay) {
      // If a resetview is pending, take care of it now before drawing
      if (ResetViewPending) 
        scene_resetview_newmoldata();

      needupdate = scene->prepare(); // prepare all objects for drawing
    } else {
      // XXX this has to be done currently to force trajectories to continue
      //     loading frames since they don't have their own UIObject yet.
      int molnum = moleculeList->num();
      int i;
      for (i=0; i<molnum; i++) {
        Molecule *mol = moleculeList->molecule(i);
        if (mol != NULL) {
          if (mol->get_new_frames()) {
            needupdate = 1; // need to update if any new frames were loaded
          }
        }
      }
    }
#else
    // XXX this has to be done currently to force trajectories to continue
    //     loading frames since they don't have their own UIObject yet.
    needupdate = scene->prepare(); // prepare all objects for drawing
#endif

    // turn off the spinning vmd when molecules are loaded
    if (vmdTitle && moleculeList->num() > 0) {
      delete vmdTitle;
      vmdTitle = NULL;
    }
  
    // Update the display (or sleep a tiny bit).
    if (UpdateDisplay && (needupdate || display->needRedraw())) {
      scene->draw(display);   // make the display redraw if necessary
      scene->draw_finished(); // perform any necessary post-drawing cleanup

      // if video streaming is active, we tell the encoder a new frame is ready
      if (uivs != NULL && uivs->srv_connected()) {
#if 0
        int xs, ys;
        unsigned char *img = display->readpixels_rgba4u(xs, ys);
        if (img != NULL) {
          uivs->srv_send_frame(img, xs * 4, xs, ys, 0);
          free(img);
        }
#elif 0
        uivs->video_frame_pending();
#endif
      }
    } else {
      // if not updating the display or doing background I/O, 
      // we sleep so we don't hog CPU.
      if (!needupdate && !background_processing())
        vmd_msleep(1); // sleep for 1 millisecond or more
    }
  } // no active video-streaming-client connection

  // XXX A hack to decrease CPU utilization on machines that have
  //     problems keeping up with the 3-D draw rate at full speed.
  if (getenv("VMDMSECDELAYHACK") != NULL) {
    // Delay the whole program for a user-specified number of milliseconds.  
    vmd_msleep(atoi(getenv("VMDMSECDELAYHACK"))); 
  }

  PROFILE_POP_RANGE();
  return TRUE;
}


// exit the program normally; first delete all the necessary objects
void VMDApp::VMDexit(const char *exitmsg, int exitcode, int pauseseconds) {

#if defined(VMDTKCON)
  // switch to text mode and flush all pending messages to the screen.
  vmdcon_use_text((void *)uiText->get_tcl_interp());
  vmdcon_purge();
#endif

#if defined(VMDMPI)
  // If MPI parallel execution is in effect, help console output complete before
  // final shutdown.
  if (mpienabled) { 
    fflush(stdout);
    vmd_mpi_barrier(); // wait for peers before printing final messages
    if (noderank == 0)
      vmd_msleep(250); // give peer output time to percolate to the console...
  }
#endif

  // only print exit status output on the first exit call
  if (exitFlag == 0 || (exitmsg != NULL && strlen(exitmsg))) {
    msgInfo << VERSION_MSG << sendmsg; 
    if (exitmsg && strlen(exitmsg)) {
      msgInfo << exitmsg << sendmsg;
    } else {
      msgInfo << "Exiting normally." << sendmsg;
    }
  }

  vmd_sleep(pauseseconds);  // sleep for requested number of seconds

  // make the VMDupdate event loop return FALSE.
  exitFlag = 1;
}

VMDApp::~VMDApp() {
  int i;

  // delete shared QuickSurf object used by all reps.
  // destroy QuickSurf object prior to other GPU-related teardown
  if (qsurf)          delete qsurf;

  // delete all objects we created during initialization
  if (fileRenderList) delete fileRenderList;
  if (mouse)          delete mouse;
  if (mobile)         delete mobile;
  if (spaceball)      delete spaceball;

#ifdef WIN32
  if (win32joystick)  delete win32joystick;
#endif

  if (uivr)           delete uivr;

  if (uivs)           delete uivs;
  uivs = NULL;        // prevent free mem accesses in event loops

  if (geometryList)   delete geometryList;

#ifdef VMDIMD
  if (imdMgr)         delete imdMgr;
  imdMgr = NULL;      // prevent free mem reads at moleculeList deletion
#endif

  if (vmdTitle)       delete vmdTitle;
  if (stage)          delete stage;
  if (axes)           delete axes;
  if (fps)            delete fps;
  if (pickModeList)   delete pickModeList;
  if (materialList)   delete materialList;
  if (pluginMgr)      delete pluginMgr;
  if (cuda)           delete cuda;
  delete vmdcollab;

  // delete all of the Forms;
  if (menulist) {
    for (i=0; i<menulist->num(); i++)
      delete menulist->data(i);
    delete menulist;
  }

#ifdef VMDGUI
  // Close all GUI windows
#ifdef VMDFLTK
  if (display->supports_gui()) {
    Fl::wait(0); // Give Fltk a chance to close the menu windows.  
  }
#endif
#endif

  // Tcl/Python interpreters can only be deleted after Tk forms are shutdown
  if (uiText)         delete uiText;

  // delete the list of user keys and descriptions 
  for (i=0; i<userKeys.num(); i++)
    delete [] userKeys.data(i);

  for (i=0; i<userKeyDesc.num(); i++)
    delete [] userKeyDesc.data(i);

  delete atomSelParser;
  delete anim;

  // delete commandQueue _after_ all UIObjects have been deleted; otherwise
  // they will try to unRegister with a deleted CommandQueue.
  delete commandQueue;

  // (these dependencies really suck) delete moleculelist after uiText
  // because text interfaces might use commands that require moleculeList to
  // be there.
  delete moleculeList;

  // picklist can't be deleted until the molecule is deleted, since
  // the DrawMolecule destructor needs it.
  delete pickList;
 
  delete display;
  delete scene;

#if defined(VMDTHREADS)
  if (thrpool) {
    wkf_threadpool_destroy((wkf_threadpool_t *) thrpool);
    thrpool=NULL;
  }
#endif
}

unsigned long VMDApp::get_repserialnum(void) {
  // first serial number returned will be 1, never return 0.
  repserialnum++;
  return repserialnum;
}

unsigned long VMDApp::get_texserialnum(void) {
  // first serial number returned will be 1, never return 0.
  texserialnum++;
  return texserialnum;
}

void VMDApp::show_stride_message() {
  if (stride_firsttime) {
    stride_firsttime = 0;
    msgInfo <<
     "In any publication of scientific results based in part or\n"
     "completely on the use of the program STRIDE, please reference:\n"
     " Frishman,D & Argos,P. (1995) Knowledge-based secondary structure\n"
     " assignment. Proteins: structure, function and genetics, 23, 566-579." 
      << "\n" << sendmsg;
  }
}

// this is a nasty hack until we get around to returning values from scripts
// properly.

#ifdef VMDTK
#include <tcl.h> 
#endif

char *VMDApp::vmd_choose_file(const char *title,
	const char *extension, 
	const char *extension_label, int do_save) {
  
 char *chooser = getenv("VMDFILECHOOSER");
 if (!chooser || !strupcmp(chooser, "TK")) {
    
#ifdef VMDTK
  JString t = title;
  JString ext = extension;
  JString label = extension_label;
  char *cmd = new char[300 + t.length() +ext.length() + label.length()];
  // no default extension for for saves/loads , because otherwise it 
  // automatically adds the file extension whether you specify it or not, and 
  // when the extension is * it won't let you save/load a file without an 
  // extension!
  if (do_save) {
    sprintf(cmd, "tk_getSaveFile -title {%s} -filetypes {{{%s} {%s}} {{All files} {*}}}", (const char *)t, (const char *)extension_label, (const char *)extension);
  } else {
    sprintf(cmd, "tk_getOpenFile -title {%s} -filetypes {{{%s} {%s}} {{All files} {*}}}", (const char *)t, (const char *)extension_label, (const char *)extension);
  }
  Tcl_Interp *interp = uiText->get_tcl_interp();
  if (interp) {
    int retval = Tcl_Eval(interp, cmd);
    delete [] cmd;
    if (retval == TCL_OK) {
      const char *result = Tcl_GetStringResult(interp);
      if (result == NULL || strlen(result) == 0) {
        return NULL;
      } else {
        return stringdup(result);
      }
    }
  }
  // fall through to next level on failure
#endif

 } 

#ifdef VMDFLTK
  return stringdup(fl_file_chooser(title, extension, NULL));
#endif

  char *result = new char[1024];
  if (fgets(result, 1024, stdin) == NULL)
    result[0] = '\0';

  return result;
}
 
// file renderer API
int VMDApp::filerender_num() {
  return fileRenderList->num();
}
const char *VMDApp::filerender_name(int n) {
  return fileRenderList->name(n);
}
const char *VMDApp::filerender_prettyname(int n) {
  return fileRenderList->pretty_name(n);
}
int VMDApp::filerender_valid(const char *method) {
  return (fileRenderList->find(method) != NULL);
}
// Find short renderer name from the "pretty" GUI renderer name
const char *VMDApp::filerender_shortname_from_prettyname(const char *pretty) {
  return fileRenderList->find_short_name_from_pretty_name(pretty);
}
int VMDApp::filerender_has_antialiasing(const char *method) {
  return (fileRenderList->has_antialiasing(method));
}
int VMDApp::filerender_aasamples(const char *method, int aasamples) {
  return fileRenderList->aasamples(method, aasamples);
}
int VMDApp::filerender_aosamples(const char *method, int aosamples) {
  return fileRenderList->aosamples(method, aosamples);
}
int VMDApp::filerender_imagesize(const char *method, int *w, int *h) {
  if (!w || !h) return FALSE;
  return fileRenderList->imagesize(method, w, h); 
}
int VMDApp::filerender_has_imagesize(const char *method) {
  return fileRenderList->has_imagesize(method);
}
int VMDApp::filerender_aspectratio(const char *method, float *aspect) {
  if (!aspect) return FALSE;
  return fileRenderList->aspectratio(method, aspect);
}
int VMDApp::filerender_numformats(const char *method) {
  return fileRenderList->numformats(method);
}
const char *VMDApp::filerender_get_format(const char *method, int i) {
  return fileRenderList->format(method, i);
}
const char *VMDApp::filerender_cur_format(const char *method) {
  return fileRenderList->format(method);
}
int VMDApp::filerender_set_format(const char *m, const char *fmt) {
  return fileRenderList->set_format(m, fmt);
}

int VMDApp::filerender_render(const char *m, const char *f, const char *e) {
  int retval = fileRenderList->render(f, m, e);
  if (retval) {
    commandQueue->runcommand(new CmdRender(f, m, e));
  }
  return retval;
}
const char *VMDApp::filerender_option(const char *m, const char *o) {
  FileRenderer *ren = fileRenderList->find(m);
  if (!ren) {
    return NULL;
  }
  if (o) {
    ren->set_exec_string(o);
    commandQueue->runcommand(new CmdRenderOption(m, o));
  }
  return ren->saved_exec_string();
}

// XXX The Scene doesn't return error codes, so I don't know if the commands
// worked or not.  I do what error checking I can here and hope that it works.
int VMDApp::scene_rotate_by(float angle, char ax, float incr) {
  if (uivs && uivs->cli_connected()) {
    uivs->cli_send_rotate_by(angle, ax);
    return TRUE;
  }

  if (ax < 'x' || ax > 'z') return FALSE;  // failed
  rocker->stop_rocking();
  if (incr) {
    int nsteps = (int)(fabs(angle / incr) + 0.5);
    incr = (float) (angle < 0.0f ? -fabs(incr) : fabs(incr));
    rocker->start_rocking(incr, ax, nsteps, TRUE);
    commandQueue->runcommand(new CmdRotate(angle, ax, CmdRotate::BY, incr)); 
  } else {
    scene->root.add_rot(angle, ax);
    commandQueue->runcommand(new CmdRotate(angle, ax, CmdRotate::BY));
  }
  return TRUE;
}
int VMDApp::scene_rotate_to(float angle, char ax) {
  if (ax < 'x' || ax > 'z') return FALSE;  // failed
  rocker->stop_rocking();
  scene->root.set_rot(angle, ax);
  commandQueue->runcommand(new CmdRotate(angle, ax, CmdRotate::TO));
  return TRUE;
}
int VMDApp::scene_rotate_by(const float *m) {
  Matrix4 mat(m);
  scene->root.add_rot(mat);
  commandQueue->runcommand(new CmdRotMat(mat, CmdRotMat::BY));
  return TRUE;
}
int VMDApp::scene_rotate_to(const float *m) {
  Matrix4 mat(m);
  scene->root.set_rot(mat);
  commandQueue->runcommand(new CmdRotMat(mat, CmdRotMat::TO));
  return TRUE;
}
int VMDApp::scene_translate_by(float x, float y, float z) {
  if (uivs && uivs->cli_connected()) {
    uivs->cli_send_translate_by(x, y, z);
    return TRUE;
  }

  scene->root.add_glob_trans(x, y, z);
  commandQueue->runcommand(new CmdTranslate(x,y,z,CmdTranslate::BY));
  return TRUE;
}
int VMDApp::scene_translate_to(float x, float y, float z) {
  scene->root.set_glob_trans(x, y, z);
  commandQueue->runcommand(new CmdTranslate(x,y,z,CmdTranslate::TO));
  return TRUE;
}
int VMDApp::scene_scale_by(float s) {
  if (uivs && uivs->cli_connected()) {
    uivs->cli_send_scale_by(s);
    return TRUE;
  }

  if (s <= 0) return FALSE; 
  scene->root.mult_scale(s);
  commandQueue->runcommand(new CmdScale(s, CmdScale::BY));
  return TRUE;
}
int VMDApp::scene_scale_to(float s) {
  if (s <= 0) return FALSE; 
  scene->root.set_scale(s);
  commandQueue->runcommand(new CmdScale(s, CmdScale::TO));
  return TRUE;
}
void VMDApp::scene_resetview_newmoldata() {
#if 1
  // XXX avoid N^2 behavior when loading thousands of molecules
  // we should only be resetting the view when necessary
  if (UpdateDisplay) {
    int nodisrupt = 0;

    if (getenv("VMDNODISRUPTHACK"))
      nodisrupt=1;

    if (nodisrupt && (moleculeList->num() > 1)) {
      moleculeList->center_top_molecule(); // new/top mol inherits current view
    } else {
      scene_resetview(); // reset all molecules to the newly loaded structure
    }
    ResetViewPending = FALSE;
  } else {
    ResetViewPending = TRUE;
  }
#else
  scene_resetview(); // reset all molecules to the newly loaded structure
#endif
}
void VMDApp::scene_resetview() {
  scene->root.reset_transformation();
  // center the view based on the displayed representations
  moleculeList->center_from_top_molecule_reps();
  moleculeList->center_all_molecules();
  commandQueue->runcommand(new CmdResetView);
}
int VMDApp::scene_rock(char ax, float step, int nsteps) {
  if (ax < 'x' || ax > 'z') return FALSE;  // failed
  rocker->start_rocking(step, ax, nsteps);
  commandQueue->runcommand(new CmdRockOn(step, ax, nsteps));
  return TRUE;
}
int VMDApp::scene_rockoff() {
  rocker->stop_rocking();
  commandQueue->runcommand(new CmdRockOff);
  return TRUE;
} 
int VMDApp::scene_stoprotation() {
  rocker->stop_rocking();
  mouse->stop_rotation();
  return TRUE;
}

int VMDApp::animation_num_dirs() {
  return Animation::ANIM_TOTAL_DIRS;
}

const char *VMDApp::animation_dir_name(int i) {
  if (i < 0 || i >= Animation::ANIM_TOTAL_DIRS) return NULL;
  return animationDirName[i]; 
}
   
int VMDApp::animation_set_dir(int d) {
  Animation::AnimDir dir = (Animation::AnimDir)d;
  anim->anim_dir(dir);
  commandQueue->runcommand(new CmdAnimDir(dir));
  return 1;
}

int VMDApp::animation_num_styles() {
  return Animation::ANIM_TOTAL_STYLES;
}

const char *VMDApp::animation_style_name(int i) {
  if (i < 0 || i >= Animation::ANIM_TOTAL_STYLES) return NULL;
  return animationStyleName[i];
}

int VMDApp::animation_set_style(int s) {
  Animation::AnimStyle style = (Animation::AnimStyle)s;
  anim->anim_style(style);
  commandQueue->runcommand(new CmdAnimStyle(style));
  return 1;
}

int VMDApp::animation_set_frame(int frame) {
    anim->goto_frame(frame);
    anim->anim_dir(Animation::ANIM_PAUSE);
    commandQueue->runcommand(new CmdAnimJump(frame));
    return 1;
}

int VMDApp::animation_set_stride(int stride) {
    anim->skip(stride);
    commandQueue->runcommand(new CmdAnimSkip(stride));
    return 1;
}

int VMDApp::animation_set_speed(float speed) {
    anim->speed(speed);
    commandQueue->runcommand(new CmdAnimSpeed(speed));
    return 1;
}

const char *VMDApp::filerender_default_option(const char *m) {
  FileRenderer *ren = fileRenderList->find(m);
  if (!ren) {
    return NULL;
  }
  return ren->default_exec_string();
}
const char *VMDApp::filerender_default_filename(const char *m) {
  FileRenderer *ren = fileRenderList->find(m);
  if (!ren) {
    return NULL;
  }
  return ren->default_filename();
}
 

// plugin stuff 

vmdplugin_t *VMDApp::get_plugin(const char *type, const char *name) {
  if (!pluginMgr) return NULL;
  if (!type || !name) return NULL;
  PluginList p;
  vmdplugin_t *plugin = NULL;
  if (pluginMgr->plugins(p, type, name)) {
    plugin = p[0];

    // loop over plugins and select the highest version number for a 
    // given plugin type/name combo.
    for (int i=1; i<p.num(); i++) {
      vmdplugin_t *curplugin = p[i];
      if (curplugin->majorv > plugin->majorv || 
          (curplugin->majorv == plugin->majorv && curplugin->minorv > plugin->minorv))
        plugin = curplugin;
    }
  } 
  return plugin;
}

int VMDApp::list_plugins(PluginList &p, const char *type) {
  if (!pluginMgr) return 0;
  return pluginMgr->plugins(p, type);
}

int VMDApp::plugin_dlopen(const char *filename) {
  if (!pluginMgr) {
    msgErr << "scan_plugins: no plugin manager available" << sendmsg;
    return -1;
  }
  if (!filename) return -1;
  return pluginMgr->load_sharedlibrary_plugins(filename);
}

void VMDApp::plugin_update() {
  commandQueue->runcommand(new CmdPluginUpdate);
}

void VMDApp::display_update_on(int ison) {
  UpdateDisplay = ison;
}

int VMDApp::display_update_status() {
  return (UpdateDisplay != 0);
}

void VMDApp::display_update() {
  int prevUpdateFlag = UpdateDisplay;
  UpdateDisplay = 1;
  VMDupdate(VMD_IGNORE_EVENTS);
  UpdateDisplay = prevUpdateFlag;
}

void VMDApp::display_update_ui() {
  VMDupdate(VMD_CHECK_EVENTS);
}

int VMDApp::num_color_categories() {
  return scene->num_categories();
}
const char *VMDApp::color_category(int n) {
  return scene->category_name(n);
}

int VMDApp::color_add_item(const char *cat, const char *name, const char *defcolor) {
  int init_color = scene->color_index(defcolor);
  if (init_color < 0) {
    msgErr << "Cannot add color item: invalid color name '" << defcolor << "'" << sendmsg;
    return FALSE;
  }
  int ind = scene->category_index(cat);
  if (ind < 0) {
    ind = scene->add_color_category(cat);
  }
  if (scene->add_color_item(ind, name, init_color) < 0) {
    return FALSE;
  }
  commandQueue->runcommand(new CmdColorItem(cat, name, defcolor));
  return TRUE;
}

int VMDApp::num_color_category_items(const char *category) {
  int colCatIndex = scene->category_index(category);
  if (colCatIndex < 0) return 0;
  return scene->num_category_items(colCatIndex);
}
const char *VMDApp::color_category_item(const char *category, int n) {
  int colCatIndex = scene->category_index(category);
  if (colCatIndex < 0) return 0;
  return scene->category_item_name(colCatIndex, n); // XXX check valid n
}
int VMDApp::num_colors() {
  return MAXCOLORS;
}
int VMDApp::num_regular_colors() {
  return REGCLRS;
}
const char *VMDApp::color_name(int n) {
  return scene->color_name(n);
}
int VMDApp::color_index(const char *color) {
  if (!color) return -1;
  // If it's a number in the valid range, return the number; otherwise return
  // -1.
  int i;
  if (sscanf(color, "%d", &i)) {
    if (i >= 0 && i < MAXCOLORS)
      return i;
    else
      return -1;
  }
  // look up the color by name.
  return scene->color_index(color);
} 
int VMDApp::color_value(const char *colorname, float *r, float *g, float *b) {
  int colIndex = color_index(colorname);
  if (colIndex < 0) return 0;
  const float *col = scene->color_value(colIndex);
  *r = col[0];
  *g = col[1];
  *b = col[2];
  return 1;
}
int VMDApp::color_default_value(const char *colorname, float *r, float *g, float *b) {
  int colIndex = color_index(colorname);
  if (colIndex < 0) return 0;
  const float *col = scene->color_default_value(colIndex);
  *r = col[0];
  *g = col[1];
  *b = col[2];
  return 1;
}
const char *VMDApp::color_mapping(const char *category, const char *item) {
  
  int colCatIndex = scene->category_index(category);
  if (colCatIndex < 0) return 0;

  int colNameIndex = scene->category_item_index(colCatIndex, item);
  if (colNameIndex < 0) return 0;
  
  int ind = scene->category_item_value(colCatIndex, colNameIndex);
  return scene->color_name(ind);
}

const char *VMDApp::color_get_restype(const char *resname) {
  int id = moleculeList->resTypes.typecode(resname);
  if (id < 0) return NULL;
  return moleculeList->resTypes.data(id);
}

int VMDApp::color_set_restype(const char *resname, const char *restype) {
  int cat = moleculeList->colorCatIndex[MLCAT_RESTYPES];
  int ind = scene->category_item_index(cat, restype);
  if (ind < 0) return FALSE;  // nonexistent restype category

  // Use the string stored in Scene rather than the one passed to this
  // function, since then we don't have to worry about copying it and
  // freeing it later.
  const char *stable_restype_name = scene->category_item_name(cat, ind);

  // if the resname doesn't have an entry yet, create one.
  int resname_id = moleculeList->resTypes.add_name(resname, restype);
  moleculeList->resTypes.set_data(resname_id, stable_restype_name);
  scene->root.color_changed(cat);
  return TRUE;
}

int VMDApp::colorscale_info(float *mid, float *min, float *max) {
  scene->colorscale_value(mid, min, max);
  return 1;
}
int VMDApp::num_colorscale_methods() {
  return scene->num_colorscale_methods();
}
int VMDApp::colorscale_method_current() {
  return scene->colorscale_method();
}
const char *VMDApp::colorscale_method_name(int n) {
  if (n < 0 || n >= scene->num_colorscale_methods()) return NULL;
  return scene->colorscale_method_name(n);
}
int VMDApp::colorscale_method_index(const char *method) {
  for (int i=0; i<scene->num_colorscale_methods(); i++) {
    if (!strupncmp(method, scene->colorscale_method_name(i),CMDLEN)) {
      return i;
    }
  }
  return -1;
}  

int VMDApp::get_colorscale_colors(int whichScale, 
      float min[3], float mid[3], float max[3]) {
  return scene->get_colorscale_colors(whichScale, min, mid, max);
}

int VMDApp::set_colorscale_colors(int whichScale, 
      const float min[3], const float mid[3], const float max[3]) {
  if (scene->set_colorscale_colors(whichScale, min, mid, max)) {
    commandQueue->runcommand(new CmdColorScaleColors(
          scene->colorscale_method_name(whichScale), mid, min, max));
    return TRUE;
  }
  return FALSE;
}


int VMDApp::color_change_name(const char *category, const char *colorname, 
                              const char *color) {

  if (!category || !colorname || !color) return 0;

  int colCatIndex = scene->category_index(category);
  if (colCatIndex < 0) return 0;
  
  int colNameIndex = scene->category_item_index(colCatIndex, colorname);
  if (colNameIndex < 0) return 0;

  int newIndex = color_index(color);
  if (newIndex < 0) return 0;
 
  // all systems go...
  scene->set_category_item(colCatIndex, colNameIndex, newIndex);
  
  // tell the rest of the world
  commandQueue->runcommand(new CmdColorName(category, colorname, color));
  return 1;
}


int VMDApp::color_change_namelist(int numcols, char **category, 
                                  char **colorname, char **color) {
  if (numcols < 1 || !category || !colorname || !color) return 0;

  int i;
  for (i=0; i<numcols; i++) {
    int colCatIndex = scene->category_index(category[i]);
    if (colCatIndex < 0) return 0;
  
    int colNameIndex = scene->category_item_index(colCatIndex, colorname[i]);
    if (colNameIndex < 0) return 0;

    int newIndex = color_index(color[i]);
    if (newIndex < 0) return 0;
 
    // all systems go...
    scene->set_category_item(colCatIndex, colNameIndex, newIndex);
  }
 
  // XXX need a new command for this 
  // commandQueue->runcommand(new CmdColorName(category, colorname, color));
  return 1;
}



int VMDApp::color_get_from_name(const char *category, const char *colorname, 
                     const char **color) {

  if (!category || !colorname) return 0;

  int colCatIndex = scene->category_index(category);
  if (colCatIndex < 0) return 0;
  
  int colNameIndex = scene->category_item_index(colCatIndex, colorname);
  if (colNameIndex < 0) return 0;
 
  // all systems go...
  int colIndex = scene->get_category_item(colCatIndex, colNameIndex);
  if (colIndex < 0) return 0;
  
  *color = color_name(colIndex);
      
  return 1;
}


int VMDApp::color_change_rgb(const char *color, float r, float g, float b) {
  int ind = color_index(color);
  if (ind < 0) return 0;
  float rgb[3] = {r, g, b};
  scene->set_color_value(ind, rgb);
  commandQueue->runcommand(new CmdColorChange(color, r, g, b));
  return 1;
}


int VMDApp::color_change_rgblist(int numcols, const char **colors, float *rgb3fv) {
  if (numcols < 1) return 0;

  int i;
  for (i=0; i<numcols; i++) {
    int ind = color_index(colors[i]);
    if (ind < 0) return 0;
    scene->set_color_value(ind, &rgb3fv[i*3]);
  }

  // XXX need a new command for this
  // commandQueue->runcommand(new CmdColorChange(color, r, g, b));
  return 1;
}
  
  
int VMDApp::colorscale_setvalues(float mid, float min, float max) {
  scene->set_colorscale_value(min, mid, max);
  commandQueue->runcommand(new CmdColorScaleSettings(mid, min, max));
  return 1;
}

int VMDApp::colorscale_setmethod(int method) {
  if (method < 0 || method >= scene->num_colorscale_methods()) return 0;
  scene->set_colorscale_method(method);
  commandQueue->runcommand(new CmdColorScaleMethod(
        scene->colorscale_method_name(method)));
  return 1;
}
  
  
int VMDApp::logfile_read(const char *path) {
  uiText->read_from_file(path);
  return 1;
}

int VMDApp::save_state() {
  char *file = vmd_choose_file(
    "Enter filename to save current VMD state:",  // Title
    "*.vmd",                                      // extension
    "VMD files",                                  // label
    1                                             // do_save
  );
  if (!file)
    return 1;
  int retval = uiText->save_state(file);
  delete [] file;
  return retval;
}
 
int VMDApp::num_molecules() {
  return moleculeList->num();
}


// This creates a blank molecule, for use by "mol new atoms" and the like...
int VMDApp::molecule_new(const char *name, int natoms, int docallbacks) {
  PROFILE_PUSH_RANGE("VMDApp::molecule_new()", 5);

  // if we aren't given a name for the molecule, use a temp name
  Molecule *newmol = new Molecule((name == NULL) ? "molecule" : name,
                                  this, &(scene->root));
  moleculeList->add_molecule(newmol);
  int molid = newmol->id();

  // If we aren't given a name for the molecule, we auto-generate one
  // from the assigned molid.
  if (name == NULL) {
    char buf[30];
    sprintf(buf, "molecule%d", molid);
#if 1
    // We rename the molecule for ourselves without calling molecule_rename()
    // in order to avoid triggering extra callbacks that cause expensive 
    // GUI redraws.
    newmol->rename(buf);

    // Add item to Molecule color category; default color should be the same as
    // the original molecule.  
    int ind = moleculeList->colorCatIndex[MLCAT_MOLECULES];
    scene->add_color_item(ind, buf, molid % VISCLRS);
#else
    molecule_rename(molid, buf);
#endif
  }

  if (natoms > 0) {
    int i;
    newmol->init_atoms(natoms);

    float *charge = newmol->charge();
    float defcharge = newmol->default_charge("X");
    for (i=0; i<natoms; i++)
      charge[i] = defcharge;

    float *mass = newmol->mass();
    float defmass = newmol->default_mass("X");
    for (i=0; i<natoms; i++)
      mass[i] = defmass;

    float *radius = newmol->radius();
    float defradius = newmol->default_radius("X");
    for (i=0; i<natoms; i++)
      radius[i] = defradius;

    float *beta = newmol->beta();
    float defbeta = newmol->default_beta();
    for (i=0; i<natoms; i++)
      beta[i] = defbeta;

    float *occupancy = newmol->occupancy();
    float defoccupancy = newmol->default_occup();
    for (i=0; i<natoms; i++)
      occupancy[i] = defoccupancy;

    // add all of the atoms in a single call, looping internally in add_atoms()
    if (0 > newmol->add_atoms(natoms, "X", "X", 0, "UNK", 0, "", "", " ", "")) {
      // if an error occured while adding an atom, we should delete
      // the offending molecule since the data is presumably inconsistent,
      // or at least not representative of what we tried to load
      msgErr << "VMDApp::molecule_new: molecule creation aborted" << sendmsg;
      return -1; // signal failure
    }
  }

  if (docallbacks) {
    commandQueue->runcommand(new CmdMolNew);
    commandQueue->runcommand(new MoleculeEvent(molid, MoleculeEvent::MOL_NEW));
    commandQueue->runcommand(new InitializeStructureEvent(molid, 1));
  }

  PROFILE_POP_RANGE();

  return molid; 
} 


int VMDApp::molecule_from_selection_list(const char *name, int mergemode,
                                         int numsels, AtomSel **sellist, 
                                         int docallbacks) {
  PROFILE_PUSH_RANGE("VMDApp::molecule_from_selection_lis()", 3);

  // sanity check on contents of atom selection list
  int natoms=0;
  int selidx, i=0, j=0;
  int havecoords=0, allhavecoords=0;
  for (selidx=0; selidx<numsels; selidx++) {
    natoms += sellist[selidx]->selected;
    if (sellist[selidx]->coordinates(moleculeList) != NULL)
      havecoords++;
  } 
  if (havecoords==numsels) {
    allhavecoords=1;
  }
  havecoords = (havecoords != 0);
  msgInfo << "Building new molecule with " << natoms 
          << " atoms from " << numsels << " selections." << sendmsg;
  if (allhavecoords) {
    msgInfo << "All atoms will be assigned atomic coordinates." << sendmsg;
  } else if (havecoords) {
    msgInfo << "Some atoms will be assigned atomic coordinates." << sendmsg;
  } else {
    msgInfo << "No atomic atomic coordinates assigned from selection list." << sendmsg;
  }

  // if we aren't given a name for the molecule, use a temp name
  Molecule *newmol = new Molecule((name == NULL) ? "molecule" : name,
                                  this, &(scene->root));
  moleculeList->add_molecule(newmol);
  int molid = newmol->id();

  // If we aren't given a name for the molecule, we auto-generate one
  // from the assigned molid.
  if (name == NULL) {
    char buf[30];
    sprintf(buf, "molecule%d", molid);
#if 1
    // We rename the molecule for ourselves without calling molecule_rename()
    // in order to avoid triggering extra callbacks that cause expensive 
    // GUI redraws.
    newmol->rename(buf);

    // Add item to Molecule color category; default color should be the same as
    // the original molecule.  
    int ind = moleculeList->colorCatIndex[MLCAT_MOLECULES];
    scene->add_color_item(ind, buf, molid % VISCLRS);
#else
    molecule_rename(molid, buf);
#endif
  }

  // initialize molecule and copy data from selected atoms into the
  // final molecule structure 
  newmol->init_atoms(natoms);

  // add a timestep for atomic coordinates, if we have them
  Timestep *ts = NULL;
  if (havecoords) {
    ts = new Timestep(natoms);
    newmol->append_frame(ts);
  }

  float *charge = newmol->charge();
  float *mass = newmol->mass();
  float *radius = newmol->radius();
  float *beta = newmol->beta();
  float *occupancy = newmol->occupancy();

  int naidx=0;
  for (selidx=0; selidx<numsels; selidx++) {
    const AtomSel *s = sellist[selidx];
    Molecule *sm = moleculeList->mol_from_id(s->molid());

#if 0
    printf("selected mol[%d]: '%s'\n" 
           "                 %d atoms, %d residues, %d frags, %d protein, %d nucleic,\n"
           "                 %d selected first: %d last: %d\n", 
      s->molid(), sm->molname(),
      sm->nAtoms, sm->nResidues, sm->nFragments, 
      sm->nProteinFragments, sm->nNucleicFragments,
      s->selected, s->firstsel, s->lastsel);
#endif

    // build a mapping from original atom indices to new atom indices
    int *atomindexmap = (int *) calloc(1, sm->nAtoms*sizeof(int));

    // initialize the atom index map to an invalid atom index value, so that
    // we can use this to eliminate bonds to atoms that aren't selected.
    for (j=0; j<sm->nAtoms; j++)
      atomindexmap[j] = -1;

    const float *s_charge = sm->charge();
    const float *s_mass = sm->mass();
    const float *s_radius = sm->radius();
    const float *s_beta = sm->beta();
    const float *s_occupancy = sm->occupancy();

    const Timestep *s_ts = s->timestep(moleculeList);

    // copy PBC unit cell information from first selection, when possible
    if (selidx == 0 && ts != NULL && s_ts != NULL) {
      ts->a_length = s_ts->a_length;
      ts->b_length = s_ts->b_length;
      ts->c_length = s_ts->c_length;
      ts->alpha = s_ts->alpha;
      ts->beta  = s_ts->beta;
      ts->gamma = s_ts->gamma;
    }

    int saidx=s->firstsel;
    while (saidx<=s->lastsel) {
      if (s->on[saidx]) {
        const MolAtom *satm = sm->atom(saidx);
#if 0
        printf("Adding satom[%d, %p] as atom[%d] nameidx: %d '%s'\n", 
               saidx, satm, naidx, satm->nameindex,
               sm->atomNames.name(satm->nameindex));
#endif

        newmol->add_atoms(1, 
                          sm->atomNames.name(satm->nameindex), 
                          sm->atomTypes.name(satm->typeindex),
                          satm->atomicnumber,
                          sm->resNames.name(satm->resnameindex),
                          satm->resid,
                          sm->chainNames.name(satm->chainindex),
                          sm->segNames.name(satm->segnameindex),
                          satm->insertionstr,
                          sm->altlocNames.name(satm->altlocindex));

        // copy atomic coordinates and other per-atom data if possible
        if (ts != NULL && s_ts != NULL) {
#if 0
          printf("Copying atomic coordinates for atom[%d] from orig[%d]...\n", naidx, saidx);
#endif
          long naddr = naidx*3;
          long saddr = saidx*3;

          // copy atomic coordinates
          ts->pos[naddr    ] = s_ts->pos[saddr    ];
          ts->pos[naddr + 1] = s_ts->pos[saddr + 1];
          ts->pos[naddr + 2] = s_ts->pos[saddr + 2];

          // XXX copy velocities, user fields, etc
        }

        // copy contents of the other per-atom fields
        charge[naidx] = s_charge[saidx];
        mass[naidx] = s_mass[saidx];
        radius[naidx] = s_radius[saidx];
        beta[naidx] = s_beta[saidx];
        occupancy[naidx] = s_occupancy[saidx];
     
        // build index map for bond/angle/dihedral/cterms
        atomindexmap[saidx] = naidx;

        naidx++; // increment total count of atoms built in new mol
      }

      // copy dataset flags from parent molecule(s)
      newmol->datasetflags |= sm->datasetflags;

      saidx++; // increment selected molecule atom index
    }

    // copy bonds to atoms that are also included in the selection
    j=0;
    for (saidx=s->firstsel; saidx<=s->lastsel; saidx++) {
      if (s->on[saidx]) {
        int newidx = naidx - s->selected + j; // index of new atom
        const MolAtom *satm = sm->atom(saidx);
#if 0
        printf("atom[%d] (orig[%d]): %d bonds\n", newidx, saidx, satm->bonds);
#endif
        int k;
        for (k=0; k<satm->bonds; k++) {
          float bondorder = 1;
          int bondtype = -1;
          int bto = satm->bondTo[k];
          int btmap = atomindexmap[bto];
          if (btmap >= 0) {
#if 0
            printf("+bond from[%d] to [%d]\n", newidx, btmap);
#endif
            // we must ensure that we do not duplicate bonds, so we
            // only add bonds to atoms with a higher index than our own
            if (btmap > newidx)
              newmol->add_bond(newidx, btmap, bondorder, bondtype);
          } 
#if 0
          else {
            printf("-bond from[%d] to [%d] (%d to %d)\n", newidx, btmap, saidx, bto);
          }
#endif
        }
        j++;
      }
    }

#if 0
    printf("processing angles...\n");
#endif
    // copy angles/dihedrals/impropers that are included in the selection
    int numangles = sm->num_angles();
    int numdihedrals = sm->num_dihedrals();
    int numimpropers = sm->num_impropers();
    int numcterms = sm->num_cterms();

    for (i=0; i<numangles; i++) {
      long i3addr = i*3L;
      int idx0 = atomindexmap[sm->angles[i3addr    ]];
      int idx1 = atomindexmap[sm->angles[i3addr + 1]];
      int idx2 = atomindexmap[sm->angles[i3addr + 2]];
      if ((idx0 >= 0) && (idx1 >= 0) && (idx2 >= 0)) {
        newmol->add_angle(idx0, idx1, idx2); 
      }
    }

#if 0
    printf("processing dihedrals...\n");
#endif
    for (i=0; i<numdihedrals; i++) {
      long i4addr = i*4L;
      int idx0 = atomindexmap[sm->dihedrals[i4addr    ]];
      int idx1 = atomindexmap[sm->dihedrals[i4addr + 1]];
      int idx2 = atomindexmap[sm->dihedrals[i4addr + 2]];
      int idx3 = atomindexmap[sm->dihedrals[i4addr + 3]];
      if ((idx0 >= 0) && (idx1 >= 0) && (idx2 >= 0) && (idx3 >= 0)) {
        newmol->add_dihedral(idx0, idx1, idx2, idx3);
      }
    }

#if 0
    printf("processing numimpropers...\n");
#endif
    for (i=0; i<numimpropers; i++) {
      long i4addr = i*4L;
      int idx0 = atomindexmap[sm->impropers[i4addr    ]];
      int idx1 = atomindexmap[sm->impropers[i4addr + 1]];
      int idx2 = atomindexmap[sm->impropers[i4addr + 2]];
      int idx3 = atomindexmap[sm->impropers[i4addr + 3]];
      if ((idx0 >= 0) && (idx1 >= 0) && (idx2 >= 0) && (idx3 >= 0)) {
        newmol->add_improper(idx0, idx1, idx2, idx3);
      }
    }

#if 0
    printf("processing numcterms...\n");
#endif
    for (i=0; i<numcterms; i++) {
      long i8addr = i*8L;
      int idx0 = atomindexmap[sm->cterms[i8addr    ]];
      int idx1 = atomindexmap[sm->cterms[i8addr + 1]];
      int idx2 = atomindexmap[sm->cterms[i8addr + 2]];
      int idx3 = atomindexmap[sm->cterms[i8addr + 3]];
      int idx4 = atomindexmap[sm->cterms[i8addr + 4]];
      int idx5 = atomindexmap[sm->cterms[i8addr + 5]];
      int idx6 = atomindexmap[sm->cterms[i8addr + 6]];
      int idx7 = atomindexmap[sm->cterms[i8addr + 7]];

      if ((idx0 >= 0) && (idx1 >= 0) && (idx2 >= 0) && (idx3 >= 0) &&
          (idx4 >= 0) && (idx5 >= 0) && (idx6 >= 0) && (idx7 >= 0)) {
        newmol->add_cterm(idx0, idx1, idx2, idx3, 
                          idx4, idx5, idx6, idx7);
      }
    }

    // XXX need to implement handling for 
    // angle types, dihedral types, and improper types (names)

    free(atomindexmap);
  }


#if 0
    // add all of the atoms in a single call, looping internally in add_atoms()
    if (0 > newmol->add_atoms(natoms, "X", "X", 0, "UNK", 0, "", "", " ", "")) {
      // if an error occured while adding an atom, we should delete
      // the offending molecule since the data is presumably inconsistent,
      // or at least not representative of what we tried to load
      msgErr << "VMDApp::molecule_new: molecule creation aborted" << sendmsg;
      PROFILE_POP_RANGE();
      return -1; // signal failure
    }
#endif

  // Cause VMD to analyze the newly-created atomic structure, so that
  // data such as the "residue" field get populated.
  newmol->analyze();

  if (docallbacks) {
    commandQueue->runcommand(new CmdMolNew);
    commandQueue->runcommand(new MoleculeEvent(molid, MoleculeEvent::MOL_NEW));
    commandQueue->runcommand(new InitializeStructureEvent(molid, 1));
  }

  PROFILE_POP_RANGE();

  return molid; 
} 


const char *VMDApp::guess_filetype(const char *filename) {
  const char *ext = strrchr(filename, '.');
  if (!ext) {
    // check for webpdb
    if (strlen(filename) == 4) {
      return "webpdb";
    }
    msgWarn << "Unable to ascertain filetype from filename '"
            << filename << "'; assuming pdb." << sendmsg;
    return "pdb";
  }
  ext++;
  char *s = strdup(ext);
  char *c = s;
  while (*c) { *c = tolower(*c); c++; }
  PluginList plugins;
  pluginMgr->plugins(plugins, "mol file reader", NULL);
  pluginMgr->plugins(plugins, "mol file converter", NULL);
  const char *bestname = NULL;
  int bestrank = 9999;
  for (int i=0; i<plugins.num(); i++) {
    // check against comma separated list of filename extensions, 
    // no spaces, as in: "pdb,ent,foo,bar,baz,ban"
    // Also keep track of the place in the list that the extension was
    // found - thus a plugin that lists the extension first can override
    // a plugin that comes earlier in the plugin list but lists the
    // extension later in its filename_extension string.
    MolFilePlugin p(plugins[i]);
    char *extbuf = strdup(p.extension());
    int extlen = strlen(extbuf);
    char *extcur = extbuf;
    char *extnext = NULL; 
    int currank = 1;
    while ((extcur - extbuf) < extlen) { 
      extnext = strchr(extcur, ','); // find next extension string
      if (extnext) {
        *extnext = '\0'; // NUL terminate this extension string
        extnext++;       // step to beginning of next extension string
      } else {
        extnext = extbuf + extlen; // no more extensions, last time through
      }
      if (!strcmp(s, extcur)) {
        if (!bestname || currank < bestrank) {
          bestname = p.name();
          bestrank = currank;
        }
      }
      extcur = extnext;
      ++currank;
    }
    free(extbuf);
  }
  free(s);
  return bestname;
}

int VMDApp::molecule_load(int molid, const char *filename, 
                          const char *filetype, const FileSpec *spec) {
  int original_molid = molid;

  // Call BaseMolecule::analyze() only once, when structure is first created.
  int first_structure = 0;

  // check for valid timestep parameters
  if (spec->last != -1 && spec->last < spec->first) {
    msgErr << "Invalid last frame: " << spec->last << sendmsg;
    return -1;
  }
  if (spec->stride < 1) {
    msgErr << "Invalid stride: " << spec->stride << sendmsg;
    return -1;
  }

  // if no filetype was given, try to obtain it from the filename.
  if (!filetype) {
    filetype = guess_filetype(filename);
    if (!filetype) {
      msgErr << "Could not determine filetype of file '" << filename 
             << "' from its name." << sendmsg;
      return -1;
    }
  }

  int waitfor = spec->waitfor;
  if (waitfor == 0 && molid == -1) {
    // It's not ok to load zero frames for a new molecule because coordinates
    // might be needed to determine bonds from atom distances.
    waitfor = 1;
    msgWarn << "Will load one coordinate frame for new molecule." << sendmsg;
  }
  
  // Prefer to use a direct reader plugin over a translator, if one
  // is available.  If not, then attempt to use a translator.
  vmdplugin_t *p = get_plugin("mol file reader", filetype);
  if (!p) p = get_plugin("mol file converter", filetype);
  if (!p) {
    msgErr << "Cannot read file of type " << filetype << sendmsg;
    return -1;
  }
  MolFilePlugin *plugin = new MolFilePlugin(p);
  if (plugin->init_read(filename)) {
    msgErr << "Could not read file " << filename << sendmsg;
    delete plugin;
    return -1;
  }

  // Check if VMD has to use page-aligned memory allocations for
  // timestep data, to allow for fast kernel-bypass unbuffered I/O APIs
  // Note: we make this call after the plugin's open_read() API has been
  // called, but before the first call to read_structure(), to ensure
  // that the called plugin sees that we have queried the required memory
  // alignment page size for all per-timestep data. 
  int ts_page_align_sz = 1;
  if (plugin->can_read_pagealigned_timesteps()) {
#if vmdplugin_ABIVERSION > 17
    ts_page_align_sz = plugin->read_timestep_pagealign_size();
#endif
  }

  // to avoid excessive error handling cleanup, we only push the 
  // profiler marker when we've made it through 80% of the error checks
  PROFILE_PUSH_RANGE("VMDApp::molecule_load()", 5);

  Molecule *newmol = NULL;
  if (molid == -1) {
#if 1
    // don't trigger callbacks for MOL_NEW etc yet, as we will be triggering
    // more of them below, and we wish to prevent the GUIs from exhibiting
    // quadratic behavior -- they regen their molecule choosers from scratch
    // in any case where the count of molecules doesn't change (e.g. when
    // getting redundant molecule new events from the same load event)
    molid = molecule_new(filename, 0, 0);
#else
    molid = molecule_new(filename, 0);
#endif
  }
  newmol = moleculeList->mol_from_id(molid);
  if (!newmol) {
    msgErr << "Invalid molecule " << molid << sendmsg;
    delete plugin;
    PROFILE_POP_RANGE();
    return -1;
  }

  // We've committed to loading as much as we can from the file, so go ahead
  // and record it.  This needs to be done before the InitializeStructure
  // event is triggered so that the list of filenames for the molecule is
  // complete and the filename can be used by Tcl/Python scripts that want
  // to process the filename when the InitializeStructure even occurs.
  char specstr[8192];
  sprintf(specstr, "first %d last %d step %d filebonds %d autobonds %d",
          spec->first, spec->last, spec->stride, 
          spec->filebonds, spec->autobonds);
  newmol->record_file(filename, filetype, specstr);

  //
  // Molecule file metadata
  // 
  if (plugin->can_read_metadata()) {
    if (plugin->read_metadata(newmol)) {
      msgErr << "Error reading metadata." << sendmsg;
    } 
  } else {
    // each file must have something, even if blank 
    newmol->record_database("", "");
    newmol->record_remarks("");
  }
 
  // 
  // Atomic structure and coordinate data
  //
  if (plugin->can_read_structure()) {
    if (!newmol->has_structure()) {
      // if the plugin can read structure data and there are a non-zero
      // number of atoms, we proceed with reading it, otherwise we bail out
      // of reading the structure, consider the outcome as having no structure,
      // and continue parsing for other data, e.g., file blocks containing
      // rawgraphics, density maps, or other contents independent of atomic
      // structure information.  Handling of the zero-atoms case is
      // required for correct handling of some of the (nascent) 
      // PDB hybrid modeling structure files hosted at PDB-Dev that contain
      // no atomic structure and only graphics glyph objects.
      if (plugin->natoms() > 0) {
        msgInfo << "Using plugin " << filetype << " for structure file " 
                << filename << sendmsg;

        int rc = plugin->read_structure(newmol, spec->filebonds, spec->autobonds);

        // it's not an error to get no structure data from formats that
        // don't always contain it, so only report an error when we 
        // expected to get structure data and got an error instead.
        if (rc != MOLFILE_SUCCESS && rc != MOLFILE_NOSTRUCTUREDATA) {
          // tell the user something went wrong, but keep going and try to
          // read other information in the file.  Perhaps it's better to
          // stop immediately and create no molecule if something goes wrong
          // at this stage?
          msgErr << "molecule_structure: Unable to read structure for molecule "
                 << molid << sendmsg;
          if (rc == MOLFILE_ERROR) {
            msgErr << "molecule_structure: severe error indicated by plugin "
                   << "aborting loading of molecule " << molid << sendmsg;
            delete plugin;
            PROFILE_POP_RANGE();
            return -1;
          }
        }

        // initialize structure if we loaded with no errors
        if (rc == MOLFILE_SUCCESS) {
          first_structure = 1;
          commandQueue->runcommand(new InitializeStructureEvent(molid, 1));
        }
      }
    } else {
      // the molecule already has structure information, so we only look for
      // extra/optional structure data.
      int rc = plugin->read_optional_structure(newmol, spec->filebonds);
      if (rc != MOLFILE_SUCCESS && rc != MOLFILE_NOSTRUCTUREDATA) {
        msgErr << 
          "Error reading optional structure information from coordinate file "
          << filename << sendmsg;
        msgErr << "Will ignore structure information in this file." << sendmsg;
      }
    }
  } else {
    // Can't read structure, but some plugins (e.g. dcd) can initialize the
    // atom number.  Do this if possible
    if (plugin->natoms() > 0) {
      if (!newmol->init_atoms(plugin->natoms())) {
        msgErr << "Invalid number of atoms in file: " << plugin->natoms()
               << sendmsg;
      }
    }
  }


  //
  // Read QM metadata and the actual data in one swoop
  // for now. We might have to separate these later.
  // 
  if (plugin->can_read_qm()) {
    if (plugin->read_qm_data(newmol)) {
      msgErr << "Error reading metadata." << sendmsg;
    } 
  }
  

  //
  // Volumetric data
  //   We proceed with attempting to volumetric density maps for any plugin
  //   that advertises the possibility of their existence.  This is done 
  //   independently of the availability of any atomic structure information. 
  if (plugin->can_read_volumetric()) {
    if (plugin->read_volumetric(newmol, spec->nvolsets, spec->setids)) {
      msgErr << "Error reading volumetric data." << sendmsg;
    } else {
      scene_resetview_newmoldata(); // reset the view so we can see the dataset.
      commandQueue->runcommand(new CmdMolVolume(molid));
    }
  }


  // 
  // Raw graphics
  //   We proceed with attempting to read graphics objects for any plugin
  //   that advertises the possibility of their existence.  This is done 
  //   independently of the availability of any atomic structure information. 
  if (plugin->can_read_graphics()) {
    if (plugin->read_rawgraphics(newmol, scene)) {
      msgErr << "Reading raw graphics failed." << sendmsg;
    } else {
      scene_resetview_newmoldata(); // reset the view so we can see the dataset.
    }
  }

  //
  // Timesteps:
  //   At present we only read timesteps if we have a non-zero atom count.
  //   It may become necessary to alter this logic if we later on have
  //   trajectory file formats that mix in non-atomic information that is
  //   present independent of any atomic coordinates.
  if ((plugin->can_read_timesteps() || plugin->can_read_qm_timestep()) &&
      (plugin->natoms() > 0)) {
    msgInfo << "Using plugin " << filetype << " for coordinates from file " 
            << filename << sendmsg;

    // Report use of direct block-based I/O when applicable
    if (ts_page_align_sz > 1)
      msgInfo << "  Direct I/O block size: " << ts_page_align_sz << sendmsg;

    if (!newmol->nAtoms) {
      msgErr << "Some frames from file '" << filename << "' could not be loaded"
        << sendmsg;
      msgErr << "because the number of atoms could not be determined.  Load a"
        << sendmsg;
      msgErr << "structure file first, then try loading this file again." << sendmsg;
    } else {
      // CoorPluginData also checks whether it has to use page-aligned 
      // memory allocations for incoming timestep data, to allow for 
      // fast kernel-bypass unbuffered I/O APIs, handling of padded and
      // aligned allocations is then taken care of in Timestep constructors
      CoorPluginData *data = new CoorPluginData(
          filename, newmol, plugin, 1, spec->first, spec->stride, spec->last);

      // Abort if there was a problem initializing coordinates, such as if
      // number of atoms doesn't match the topology. The plugin field is
      // used as a canary in the CoorPluginData constructor for this purpose.
      if (!data->is_valid()) {
        msgErr << "Problem loading coordinate data. Aborting loading of "
               << "molecule " << molid << sendmsg;
        delete data;
        delete plugin;
        PROFILE_POP_RANGE();
        return -1;
      }

      newmol->add_coor_file(data);
      if (waitfor < 0) {
        // drain the I/O queue of all frames, even those that didn't necessarily
        // come from this file.
        while (newmol->next_frame());
      } else {
        // read waitfor frames.
        for (int i=0; i<waitfor; i++)
          if (!newmol->next_frame()) break;
      }
    }
  } else {
    // Delete the plugin unless timesteps were loaded, in which case the 
    // plugin will be deleted by the CoorPluginData object.
    delete plugin;
    plugin = NULL;
  }

  // Now go back and analyze structure, since we may have had to load
  // timesteps first.
  if (first_structure) {
    // build structure information for this molecule
    newmol->analyze(); 

    // Must add color names here because atom names and such aren't defined
    // until there's a molecular structure.
    moleculeList->add_color_names(moleculeList->mol_index_from_id(molid));

    // force all colors and reps to be recalculated, since this may be 
    // loaded into a molecule that didn't previously contain atomic data
    newmol->force_recalc(DrawMolItem::COL_REGEN | DrawMolItem::SEL_REGEN);

    // since atom color definitions are now established, create a new 
    // representation using the default parameters.
    moleculeList->set_color((char *)moleculeList->default_color());
    moleculeList->set_representation((char *)moleculeList->default_representation());
    moleculeList->set_selection(moleculeList->default_selection());
    moleculeList->set_material((char *)moleculeList->default_material());
    molecule_addrep(newmol->id());
    scene_resetview_newmoldata(); // reset the view so we can see the dataset.
  }

  // If the molecule doesn't have any reps yet and we have volume data,
  // add a new Isosurface rep.
  if (!newmol->components() && newmol->num_volume_data()) {
    molecule_set_style("Isosurface");
    molecule_addrep(newmol->id());
  }

  commandQueue->runcommand(new CmdMolLoad(original_molid, filename, filetype, 
                           spec));

  PROFILE_POP_RANGE();
  return molid;
}


int VMDApp::molecule_savetrajectory(int molid, const char *fname, 
                                    const char *type, const FileSpec *spec) {
  Molecule *newmol = moleculeList->mol_from_id(molid);
  if (!newmol) {
    msgErr << "Invalid molecule id " << molid << sendmsg;
    return -1;
  }
  if (fname == NULL) {
    msgErr << "Invalid NULL filename string" << sendmsg;
    return -1;
  }

  int first = spec->first;
  int last = spec->last;
  int stride = spec->stride;
  int waitfor = spec->waitfor;
  int nframes = 0;
  int natoms = 0;
  const int *selection = spec->selection;
  CoorData *data = NULL;
  int savevoldata = (newmol->num_volume_data() > 0) && (spec->nvolsets != 0);

  // Determine number of atoms to write.
  natoms = newmol->nAtoms;
  if (selection) {
    natoms=0;
    // the selection cannot change as the trajectory is written out,
    // so this is a safe way to count selected atoms
    for (int i=0; i<newmol->nAtoms; i++)
      natoms += selection[i] ? 1 : 0;
  }

  // validate timesteps if we actually have atomic coordinates to write
  if (natoms > 0) {
    if (last == -1)
      last = newmol->numframes() - 1;

    if (last < first && last >= 0) {
      msgErr << "Invalid last frame: " << last << sendmsg;
      return -1;
    }
    
    if (stride == -1 || stride == 0)
      stride = 1; // save all frames 

    if (stride < 1) {
      msgErr << "Invalid stride: " << stride << sendmsg;
      return -1;
    }

    nframes = (last-first)/stride + 1;
    if (nframes < 1 && !savevoldata) {
      msgInfo << "Save Trajectory: 0 frames specified; no coordinates written."
              << sendmsg;
      return 0;
    }

    if (natoms < 1 && !savevoldata) {
      msgInfo << "Save Trajectory: 0 atoms in molecule or selection; no coordinates written."
              << sendmsg;
      return -1;
    }
  }

  // Prefer to use a direct reader plugin over a translator, if one
  // is available.  If not, then attempt to use a translator.
  vmdplugin_t *p = get_plugin("mol file reader", type);
  if (!p) p = get_plugin("mol file converter", type);
  MolFilePlugin *plugin = NULL;
  if (p) {
    plugin = new MolFilePlugin(p);
    if (plugin->init_write(fname, natoms)) {
      msgErr << "Unable to open file " << fname << " of type " << type
             << " for writing frames." << sendmsg;
      delete plugin;
      return -1;
    }
    data = new CoorPluginData(fname, newmol, plugin, 0, first, stride, last,
                              selection);
  } else {
    msgErr << "Unknown coordinate file type " << type << sendmsg;
    return -1;
  }
  if (data == NULL) {
    msgErr << "NULL data returned by plugin " << sendmsg;
    return -1;
  }    
  msgInfo << "Opened coordinate file " << fname << " for writing.";
  msgInfo << sendmsg;

  // to avoid excessive error handling cleanup, we only push the 
  // profiler marker when we've made it through 80% of the error checks
  PROFILE_PUSH_RANGE("VMDApp::molecule_savetrajectory()", 5);

  // XXX if writing volume sets was requested, do it here, since CoorPluginData 
  // doesn't know about that type of data.  CoorPluginData should be using
  // a FileSpec struct.
  if (savevoldata) {
    if (plugin->can_write_volumetric()) {
      for (int i=0; i<spec->nvolsets; i++) {
        if (plugin->write_volumetric(newmol, spec->setids[i]) != 
            MOLFILE_SUCCESS) {
          msgErr << "Failed to write volume set " << spec->setids[i]
                 << sendmsg;
        }
      }
    } else {
      msgErr << "Cannot write volsets to files of type " << type << sendmsg;
    }
  }
  
  // write waitfor frames before adding to the Molecule's queue.
  int numwritten = 0;
  if (waitfor < 0) {
    while (data->next(newmol) == CoorData::NOTDONE)
      numwritten++;
  
    // Don't add to the I/O queue, just complete the I/O transaction 
    // synchronously and trigger any necessary callbacks.
    // This prevents analysis scripts that don't return control to the
    // main loop from queueing up large amounts of I/Os that only needed
    // file closures and memory frees to be completed.
    newmol->close_coor_file(data);
  } else if (waitfor > 0) {
    for (int i=0; i<waitfor; i++) {
      if (data->next(newmol) == CoorData::DONE) break;
        numwritten++;
    }

    // Add the I/O to the asynchronous queue and let it continue 
    // with subsequent main loop event polling/updates.
    newmol->add_coor_file(data);
  }

  commandQueue->runcommand(new CmdAnimWriteFile(molid, fname, type,
                           first, last, stride));

  PROFILE_POP_RANGE();

  return numwritten;
}
 

int VMDApp::molecule_deleteframes(int molid, int first, int last, 
                                   int stride) {
  Molecule *mol = moleculeList->mol_from_id(molid);
  if (!mol) {
    msgErr << "Invalid molecule id " << molid << sendmsg;
    return 0;
  }
  if (!mol->numframes()) return TRUE;

  if (last == -1)
    last = mol->numframes()-1;
  if (last < first) {
    msgErr << "Invalid last frame: " << last << sendmsg;
    return 0;
  }
  
  if (stride==-1) stride=0; //delete all frames in range
  if (stride < 0) {
    msgErr << "Invalid stride: " << stride << sendmsg;
    return 0;
  }
  
  // keep every stride frame btw first and last
  int indexshift = first; // as frames are deleted, indices are shifted
  for (int i=0; i<=last-first; i++) {
    if (!stride || i%stride) {
      mol->delete_frame(indexshift+i);
      indexshift--;
    }
  }

  commandQueue->runcommand(new CmdAnimDelete(molid, first, last, stride));
  return 1;
}

int VMDApp::molecule_index_from_id(int id) {
  if (id < 0) return -1;
  return moleculeList->mol_index_from_id(id);
}
int VMDApp::molecule_id(int i) {
  if (i < 0 || i >= num_molecules()) return -1;
  Molecule *m = moleculeList->molecule(i);
  if (m == NULL)
    return -1;
  return m->id();
}
int VMDApp::molecule_valid_id(int molid) {
  return (moleculeList->mol_from_id(molid) != NULL);
}
int VMDApp::molecule_cancel_io(int molid) {
  Molecule *m = moleculeList->mol_from_id(molid);
  if (!m) return 0;
  m->cancel();
  commandQueue->runcommand(new CmdMolCancel(molid));
  return 1;
}

int VMDApp::molecule_delete(int molid) {
  if (moleculeList->del_molecule(molid)) {
    commandQueue->runcommand(new CmdMolDelete(molid));
    commandQueue->runcommand(new InitializeStructureEvent(molid, 0));
    commandQueue->runcommand(new MoleculeEvent(molid, MoleculeEvent::MOL_DELETE));
    // XXX this has the side effect of altering the 'top' molecule.
    // At present the GUI is checking for MOL_DEL in addition to MOL_TOP,
    // but really it'd be nicer if we generated appropriate events for 
    // side effect cases like this.
    return 1;
  }
  return 0; 
}

int VMDApp::molecule_delete_all(void) {
  int i, nummols, rc;
  int *molidlist;

  rc = 0;
  nummols = num_molecules();
  molidlist = new int[nummols];
 
  // save molid translation list before we delete them all
  for (i=0; i<nummols; i++) {
    molidlist[i] = moleculeList->molecule(i)->id();
  }

  // delete all molecules and process molecule event callbacks
  if (moleculeList->del_all_molecules()) {
    for (i=0; i<nummols; i++) {
      int molid = molidlist[i];
      commandQueue->runcommand(new CmdMolDelete(molid));
      commandQueue->runcommand(new InitializeStructureEvent(molid, 0));
      commandQueue->runcommand(new MoleculeEvent(molid, MoleculeEvent::MOL_DELETE));
    }

    // XXX this has the side effect of altering the 'top' molecule.
    // At present the GUI is checking for MOL_DEL in addition to MOL_TOP,
    // but really it'd be nicer if we generated appropriate events for 
    // side effect cases like this.
    rc=1;
  }

  delete [] molidlist;

  return rc; 
}

int VMDApp::molecule_activate(int molid, int onoff) {
  int ind = moleculeList->mol_index_from_id(molid);
  if (ind < 0) return 0;
  if (onoff)
    moleculeList->activate(ind);
  else
    moleculeList->inactivate(ind);
  commandQueue->runcommand(new CmdMolActive(molid, onoff));
  return 1;
}
int VMDApp::molecule_is_active(int molid) {
  int ind = moleculeList->mol_index_from_id(molid);
  if (ind < 0) return 0;
  return moleculeList->active(ind);
}
int VMDApp::molecule_fix(int molid, int onoff) {
  int ind = moleculeList->mol_index_from_id(molid);
  if (ind < 0) return 0;
  if (onoff)
    moleculeList->fix(ind);
  else
    moleculeList->unfix(ind);
  commandQueue->runcommand(new CmdMolFix(molid, onoff));
  return 1;
}
int VMDApp::molecule_is_fixed(int molid) {
  int ind = moleculeList->mol_index_from_id(molid);
  if (ind < 0) return 0;
  return moleculeList->fixed(ind);
}
int VMDApp::molecule_display(int molid, int onoff) {
  int ind = moleculeList->mol_index_from_id(molid);
  if (ind < 0) return 0;
  if (onoff)
    moleculeList->show(ind);
  else
    moleculeList->hide(ind);
  commandQueue->runcommand(new CmdMolOn(molid, onoff));
  return 1;
}
int VMDApp::molecule_is_displayed(int molid) {
  int ind = moleculeList->mol_index_from_id(molid);
  if (ind < 0) return 0;
  return moleculeList->displayed(ind);
}
int VMDApp::molecule_make_top(int molid) {
  int ind = moleculeList->mol_index_from_id(molid);
  if (ind < 0) return 0;
  moleculeList->make_top(ind);
  commandQueue->runcommand(new CmdMolTop(molid));
  return 1;
}
int VMDApp::molecule_top() {
  Molecule *m = moleculeList->top();
  if (!m) return -1;
  return m->id();
}
int VMDApp::num_molreps(int molid) {
  Molecule *m = moleculeList->mol_from_id(molid);
  if (!m) return 0;
  return m->components();
}
const char *VMDApp::molrep_get_style(int molid, int repid) {
  if (repid < 0 || repid >= num_molreps(molid)) return NULL;
  DrawMolItem *item = moleculeList->mol_from_id(molid)->component(repid);
  if (item == NULL)
    return NULL;
  return item->atomRep->cmdStr;
}
int VMDApp::molrep_set_style(int molid, int repid, const char *style) {
  int ind = moleculeList->mol_index_from_id(molid);
  if (ind < 0) return 0;
  if (!moleculeList->change_repmethod(repid, ind, (char *)style)) return 0;
  commandQueue->runcommand(
    new CmdMolChangeRepItem(repid, molid, CmdMolChangeRepItem::REP, style));
  return 1;
}
const char *VMDApp::molrep_get_color(int molid, int repid) {
  if (repid < 0 || repid >= num_molreps(molid)) return NULL;
  DrawMolItem *item = moleculeList->mol_from_id(molid)->component(repid);
  if (item == NULL)
    return NULL;
  return item->atomColor->cmdStr;
}
int VMDApp::molrep_set_color(int molid, int repid, const char *color) {
  int ind = moleculeList->mol_index_from_id(molid);
  if (ind < 0) return 0;
  if (!moleculeList->change_repcolor(repid, ind, (char *)color)) return 0;
  commandQueue->runcommand(
    new CmdMolChangeRepItem(repid, molid, CmdMolChangeRepItem::COLOR, color));
  return 1;
}
const char *VMDApp::molrep_get_selection(int molid, int repid) {
  if (repid < 0 || repid >= num_molreps(molid)) return NULL;
  DrawMolItem *item = moleculeList->mol_from_id(molid)->component(repid);
  if (item == NULL)
    return NULL;
  return item->atomSel->cmdStr;
}
int VMDApp::molrep_set_selection(int molid, int repid, const char *selection) {
  int ind = moleculeList->mol_index_from_id(molid);
  if (ind < 0) return FALSE;
  if (!moleculeList->change_repsel(repid, ind, selection)) 
      return FALSE;
  commandQueue->runcommand(
    new CmdMolChangeRepItem(repid, molid, CmdMolChangeRepItem::SEL, selection));
  return TRUE;
}
int VMDApp::molrep_numselected(int molid, int repid) {
  if (repid >= num_molreps(molid)) return  -1;
  DrawMolItem *item = moleculeList->mol_from_id(molid)->component(repid);
  if (item == NULL)
    return -1;
  return item->atomSel->selected;
}
const char *VMDApp::molrep_get_material(int molid, int repid) {
  if (repid >= num_molreps(molid)) return NULL;
  DrawMolItem *item = moleculeList->mol_from_id(molid)->component(repid);
  if (item == NULL)
    return NULL;
  return materialList->material_name(item->curr_material());
}
int VMDApp::molrep_set_material(int molid, int repid, const char *material) {
  int ind = moleculeList->mol_index_from_id(molid);
  if (ind < 0) return 0;
  if (!moleculeList->change_repmat(repid, ind, (char *)material)) return 0;
  commandQueue->runcommand(
    new CmdMolChangeRepItem(repid, molid, CmdMolChangeRepItem::MAT, material));
  return 1;
}
const char *VMDApp::molecule_get_style() {
  return moleculeList->representation();
}
int VMDApp::molecule_set_style(const char *style) {
  if (!moleculeList->set_representation((char *)style))
    return 0;
  commandQueue->runcommand(new CmdMolRep(style));
  return 1;
}
const char *VMDApp::molecule_get_color() {
  return moleculeList->color();
}
int VMDApp::molecule_set_color(const char *color) {
  if (!moleculeList->set_color((char *)color)) 
    return 0;
  commandQueue->runcommand(new CmdMolColor(color));
  return 1;
}
const char *VMDApp::molecule_get_selection() {
  return moleculeList->selection();
}
int VMDApp::molecule_set_selection(const char *selection) {
  if (!moleculeList->set_selection(selection))
    return FALSE;
  commandQueue->runcommand(new CmdMolSelect(selection));
  return TRUE;
}
const char *VMDApp::molecule_get_material() {
  return moleculeList->material();
}
int VMDApp::molecule_set_material(const char *material) {
  if (!moleculeList->set_material((char *)material))
    return 0;
  commandQueue->runcommand(new CmdMolMaterial(material));
  return 1;
}
int VMDApp::molecule_addrep(int molid) {
  int ind = moleculeList->mol_index_from_id(molid);
  if (ind < 0) return 0;
  if (!moleculeList->add_rep(ind)) return 0;
  commandQueue->runcommand(new CmdMolAddRep(molid));
  return 1;
}
int VMDApp::molecule_modrep(int molid, int repid) {
  int ind = moleculeList->mol_index_from_id(molid);
  if (ind < 0) return 0;
  if (!moleculeList->change_rep(repid, ind)) return 0;
  commandQueue->runcommand(new CmdMolChangeRep(molid, repid));
  return 1;
} 
int VMDApp::molrep_delete(int molid, int repid) {
  int ind = moleculeList->mol_index_from_id(molid);
  if (ind < 0) return 0;
  if (!moleculeList->del_rep(repid, ind)) return 0;
  commandQueue->runcommand(new CmdMolDeleteRep(repid, molid));
  return 1;
}

int VMDApp::molrep_get_selupdate(int molid, int repid) {
  if (repid >= num_molreps(molid)) return 0;
  DrawMolItem *item = moleculeList->mol_from_id(molid)->component(repid);
  if (item == NULL || item->atomSel == NULL)
    return 0;
  return item->atomSel->do_update; 
} 
int VMDApp::molrep_set_selupdate(int molid, int repid, int onoff) {
  if (repid >= num_molreps(molid)) return 0;
  DrawMolItem *item = moleculeList->mol_from_id(molid)->component(repid);
  if (item == NULL || item->atomSel == NULL)
    return 0;
  item->atomSel->do_update = onoff;
  commandQueue->runcommand(new CmdMolRepSelUpdate(repid, molid, onoff));
  return 1;
} 

int VMDApp::molrep_set_colorupdate(int molid, int repid, int onoff) {
  if (repid >= num_molreps(molid)) return 0;
  DrawMolItem *item = moleculeList->mol_from_id(molid)->component(repid);
  if (item == NULL || item->atomColor == NULL)
    return 0;
  item->atomColor->do_update = onoff;
  if (onoff) item->force_recalc(DrawMolItem::COL_REGEN);
  commandQueue->runcommand(new CmdMolRepColorUpdate(repid, molid, onoff));
  return 1;
} 
int VMDApp::molrep_get_colorupdate(int molid, int repid) {
  if (repid >= num_molreps(molid)) return 0;
  DrawMolItem *item = moleculeList->mol_from_id(molid)->component(repid);
  if (item == NULL || item->atomColor == NULL)
    return 0;
  return item->atomColor->do_update;
}

int VMDApp::molrep_set_smoothing(int molid, int repid, int n) {
  if (repid < 0 || repid >= num_molreps(molid)) return FALSE;
  DrawMolItem *item = moleculeList->mol_from_id(molid)->component(repid);
  if (item->get_smoothing() == n) return TRUE;
  if (item->set_smoothing(n)) {
    item->force_recalc(DrawMolItem::MOL_REGEN);
    commandQueue->runcommand(new CmdMolSmoothRep(molid, repid, n));
    return TRUE;
  }
  return FALSE;
}
int VMDApp::molrep_get_smoothing(int molid, int repid) {
  if (repid < 0 || repid >= num_molreps(molid)) return -1;
  DrawMolItem *item = moleculeList->mol_from_id(molid)->component(repid);
  return item->get_smoothing();
}

int VMDApp::molrep_set_pbc(int molid, int repid, int pbc) {
  if (repid < 0 || repid >= num_molreps(molid)) return FALSE;
  DrawMolItem *item = moleculeList->mol_from_id(molid)->component(repid);
  item->set_pbc(pbc);
  commandQueue->runcommand(new CmdMolShowPeriodic(molid, repid, pbc));
  return TRUE;
}
int VMDApp::molrep_get_pbc(int molid, int repid) {
  if (repid < 0 || repid >= num_molreps(molid)) return FALSE;
  const DrawMolItem *item = moleculeList->mol_from_id(molid)->component(repid);
  return item->get_pbc();
}

int VMDApp::molrep_set_pbc_images(int molid, int repid, int n) {
  if (n < 1) return FALSE;
  if (repid < 0 || repid >= num_molreps(molid)) return FALSE;
  DrawMolItem *item = moleculeList->mol_from_id(molid)->component(repid);
  item->set_pbc_images(n);
  commandQueue->runcommand(new CmdMolNumPeriodic(molid, repid, n));
  return TRUE;
}
int VMDApp::molrep_get_pbc_images(int molid, int repid) {
  if (repid < 0 || repid >= num_molreps(molid)) return -1;
  const DrawMolItem *item = moleculeList->mol_from_id(molid)->component(repid);
  return item->get_pbc_images();
}


int VMDApp::molecule_add_instance(int molid, Matrix4 & inst) {
  Molecule *m = moleculeList->mol_from_id(molid);
  if (!m) return 0;
  m->add_instance(inst);

  // commandQueue->runcommand(new CmdMolVolume(molid));
  return 1;
}
int VMDApp::molecule_num_instances(int molid) {
  Molecule *m = moleculeList->mol_from_id(molid);
  if (!m) return -1;
  return m->num_instances();
}
int VMDApp::molecule_delete_all_instances(int molid) {
  Molecule *m = moleculeList->mol_from_id(molid);
  if (!m) return -1;
  m->clear_instances();
  // commandQueue->runcommand(new CmdMolVolume(molid));
  return 1;
}

int VMDApp::molrep_set_instances(int molid, int repid, int inst) {
  if (repid < 0 || repid >= num_molreps(molid)) return FALSE;
  DrawMolItem *item = moleculeList->mol_from_id(molid)->component(repid);
  item->set_instances(inst);
  commandQueue->runcommand(new CmdMolShowInstances(molid, repid, inst));
  return TRUE;
}
int VMDApp::molrep_get_instances(int molid, int repid) {
  if (repid < 0 || repid >= num_molreps(molid)) return FALSE;
  const DrawMolItem *item = moleculeList->mol_from_id(molid)->component(repid);
  return item->get_instances();
}

int VMDApp::molecule_set_dataset_flag(int molid, const char *dataflagstr,
                                      int setval) {
  int dataflag=BaseMolecule::NODATA;
  Molecule *m = moleculeList->mol_from_id(molid);
  if (!m) return 0;

  // which flag to set/clear
  if (!strcmp("insertion", dataflagstr)) {
    dataflag=BaseMolecule::INSERTION;
  } else if (!strcmp("occupancy", dataflagstr)) {
    dataflag=BaseMolecule::OCCUPANCY;
  } else if (!strcmp("beta", dataflagstr)) {
    dataflag=BaseMolecule::BFACTOR;
  } else if (!strcmp("mass", dataflagstr)) {
    dataflag=BaseMolecule::MASS;
  } else if (!strcmp("charge", dataflagstr)) {
    dataflag=BaseMolecule::CHARGE;
  } else if (!strcmp("radius", dataflagstr)) {
    dataflag=BaseMolecule::RADIUS;
  } else if (!strcmp("altloc", dataflagstr)) {
    dataflag=BaseMolecule::ALTLOC;
  } else if (!strcmp("atomicnumber", dataflagstr)) {
    dataflag=BaseMolecule::ATOMICNUMBER;
  } else if (!strcmp("bonds", dataflagstr)) {
    dataflag=BaseMolecule::BONDS;
  } else if (!strcmp("bondorders", dataflagstr)) {
    dataflag=BaseMolecule::BONDORDERS;
  } else if (!strcmp("bondtypes", dataflagstr)) {
    dataflag=BaseMolecule::BONDTYPES;
  } else if (!strcmp("angles", dataflagstr)) {
    dataflag=BaseMolecule::ANGLES;
  } else if (!strcmp("angletypes", dataflagstr)) {
    dataflag=BaseMolecule::ANGLETYPES;
  } else if (!strcmp("cterms", dataflagstr)) {
    dataflag=BaseMolecule::CTERMS;
  } else if (!strcmp("all", dataflagstr)) {
    dataflag=
      BaseMolecule::INSERTION    |
      BaseMolecule::OCCUPANCY    |
      BaseMolecule::BFACTOR      |
      BaseMolecule::MASS         |
      BaseMolecule::CHARGE       |
      BaseMolecule::RADIUS       |
      BaseMolecule::ALTLOC       |
      BaseMolecule::ATOMICNUMBER |
      BaseMolecule::BONDS        |
      BaseMolecule::BONDORDERS   |
      BaseMolecule::BONDTYPES    |
      BaseMolecule::ANGLES       |
      BaseMolecule::ANGLETYPES   |
      BaseMolecule::CTERMS;
  }

  // return an error if the flag string is unknown
  if (dataflag == BaseMolecule::NODATA)
    return 0;

  // set/unset the flag if we recognized it
  if (setval)
    m->set_dataset_flag(dataflag); 
  else
    m->unset_dataset_flag(dataflag); 

  return 1;
}


int VMDApp::molecule_reanalyze(int molid) {
  Molecule *m = moleculeList->mol_from_id(molid);
  if (!m) return 0;

  // (re)analyze the molecular structure, since bonds may have been changed
  m->analyze();

  // force all reps and selections to be recalculated
  m->force_recalc(DrawMolItem::COL_REGEN | 
                  DrawMolItem::MOL_REGEN |
                  DrawMolItem::SEL_REGEN);

  // regen secondary structure as well
  m->invalidate_ss();

  // log the command
  commandQueue->runcommand(new CmdMolReanalyze(molid));
  commandQueue->runcommand(new MoleculeEvent(molid, MoleculeEvent::MOL_REGEN));
  return TRUE;
}
int VMDApp::molecule_bondsrecalc(int molid) {
  Molecule *m = moleculeList->mol_from_id(molid);
  if (!m) return 0;
  if (m->recalc_bonds()) return 0;
  commandQueue->runcommand(new CmdMolBondsRecalc(molid));
  commandQueue->runcommand(new MoleculeEvent(molid, MoleculeEvent::MOL_REGEN));
  return 1;
}
int VMDApp::molecule_ssrecalc(int molid) {
  Molecule *m = moleculeList->mol_from_id(molid);
  if (!m) return FALSE;
  if (!m->recalc_ss()) return FALSE;
  commandQueue->runcommand(new CmdMolSSRecalc(molid));
  return TRUE;
}
int VMDApp::molecule_numatoms(int molid) {
  Molecule *m = moleculeList->mol_from_id(molid);
  if (!m) return -1;
  return m->nAtoms;
}
int VMDApp::molecule_numframes(int molid) {
  Molecule *m = moleculeList->mol_from_id(molid);
  if (!m) return -1;
  return m->numframes();
} 
int VMDApp::molecule_frame(int molid) {
  Molecule *m = moleculeList->mol_from_id(molid);
  if (!m) return -1;
  return m->frame();
} 
int VMDApp::molecule_dupframe(int molid, int frame) {
  Molecule *m = moleculeList->mol_from_id(molid);
  if (!m) {
    msgErr << "molecule_dupframe: invalid molecule" << sendmsg;
    return FALSE;
  }
  if (frame >= m->numframes()) {
    msgErr << "molecule_dupframe: frame out of range" << sendmsg;
    return FALSE;
  }
  if (frame == -1) {
    m->duplicate_frame(m->current());
  } else {
    m->duplicate_frame(m->get_frame(frame));
  }
  commandQueue->runcommand(new CmdAnimDup(frame, molid));
  return TRUE;
}

const char *VMDApp::molecule_name(int molid) {
  Molecule *m = moleculeList->mol_from_id(molid);
  if (!m) return NULL;
  return m->molname();
} 
int VMDApp::molecule_rename(int molid, const char *newname) {
  Molecule *m = moleculeList->mol_from_id(molid);
  if (!m) return 0;
  if (!newname) return 0;
  if (!m->rename(newname)) return 0;
  
  // Add item to Molecule color category; default color should be the same as
  // the original molecule.  
  int ind = moleculeList->colorCatIndex[MLCAT_MOLECULES];
  scene->add_color_item(ind, newname, m->id() % VISCLRS);
  
  commandQueue->runcommand(new CmdMolRename(molid, newname));
  commandQueue->runcommand(new MoleculeEvent(molid, MoleculeEvent::MOL_RENAME));
  return 1;
}

/// Create a new wavefunction object based on existing wavefunction
/// <waveid> with orbitals localized using the Pipek-Mezey algorithm.
int VMDApp::molecule_orblocalize(int molid, int waveid) {
  Molecule *m = moleculeList->mol_from_id(molid);
  if (!m) return 0;

  float *expandedbasis = NULL;
  int *numprims = NULL;
  m->qm_data->expand_basis_array(expandedbasis, numprims);

  int i;
  for (i=0; i<m->numframes(); i++) {
    msgInfo << "Localizing orbitals for wavefunction " << waveid
            << " in frame " << i << sendmsg;
    m->qm_data->orblocalize(m->get_frame(i), waveid, expandedbasis, numprims);
  }

  delete [] expandedbasis;
  delete [] numprims;
  // XXX need to add commandQueue->runcommand()
  return 1;
}

int VMDApp::molecule_add_volumetric(int molid, const char *dataname, 
    const float origin[3], const float xaxis[3], const float yaxis[3],
    const float zaxis[3], int xsize, int ysize, int zsize, float *datablock) {
  PROFILE_PUSH_RANGE("VMDApp::molecule_add_volumetric()", 3);
 
  Molecule *m = moleculeList->mol_from_id(molid);
  if (!m) return 0;
  m->add_volume_data(dataname, origin, xaxis, yaxis, zaxis, xsize, ysize, 
    zsize, datablock);

  scene_resetview_newmoldata(); // reset the view so we can see the dataset.
  commandQueue->runcommand(new CmdMolVolume(molid));

  PROFILE_POP_RANGE();
  return 1;
}

int VMDApp::molecule_add_volumetric(int molid, const char *dataname, 
    const double origin[3], const double xaxis[3], const double yaxis[3],
    const double zaxis[3], int xsize, int ysize, int zsize, float *datablock) {
  PROFILE_PUSH_RANGE("VMDApp::molecule_add_volumetric()", 3);
  
  Molecule *m = moleculeList->mol_from_id(molid);
  if (!m) return 0;
  m->add_volume_data(dataname, origin, xaxis, yaxis, zaxis, xsize, ysize, 
    zsize, datablock);

  scene_resetview_newmoldata(); // reset the view so we can see the dataset.
  commandQueue->runcommand(new CmdMolVolume(molid));

  PROFILE_POP_RANGE();
  return 1;
}

void VMDApp::set_mouse_callbacks(int on) {
  mouse->set_callbacks(on);
}

void VMDApp::set_mouse_rocking(int on) {
  mouse->set_rocking(on);
}

int VMDApp::num_clipplanes() {
  return VMD_MAX_CLIP_PLANE;
}
int VMDApp::molrep_get_clipplane(int molid, int repid, int clipid,
                        float *center, float *normal, float *color, int *mode) {
  Molecule *mol = moleculeList->mol_from_id(molid);
  if (!mol) return 0;
  Displayable *d = mol->component(repid);
  if (!d) return 0;
  const VMDClipPlane *c = d->clipplane(clipid);
  if (!c) return 0;
  memcpy(center, c->center, 3L*sizeof(float));
  memcpy(normal, c->normal, 3L*sizeof(float));
  memcpy(color, c->color, 3L*sizeof(float));
  *mode = c->mode;
  return 1;
}
int VMDApp::molrep_set_clipcenter(int molid, int repid, int clipid,
                                 const float *center) {
  Molecule *mol = moleculeList->mol_from_id(molid);
  if (!mol) return 0;
  Displayable *d = mol->component(repid);
  if (!d) return 0;
  return d->set_clip_center(clipid, center);
}
int VMDApp::molrep_set_clipnormal(int molid, int repid, int clipid,
                                 const float *normal) {
  Molecule *mol = moleculeList->mol_from_id(molid);
  if (!mol) return 0;
  Displayable *d = mol->component(repid);
  if (!d) return 0;
  return d->set_clip_normal(clipid, normal);
}
int VMDApp::molrep_set_clipcolor(int molid, int repid, int clipid,
                                 const float *color) {
  Molecule *mol = moleculeList->mol_from_id(molid);
  if (!mol) return 0;
  Displayable *d = mol->component(repid);
  if (!d) return 0;
  return d->set_clip_color(clipid, color);
}
int VMDApp::molrep_set_clipstatus(int molid, int repid, int clipid, int mode) {
  Molecule *mol = moleculeList->mol_from_id(molid);
  if (!mol) return 0;
  Displayable *d = mol->component(repid);
  if (!d) return 0;
  return d->set_clip_status(clipid, mode);
}
 
const char *VMDApp::molrep_get_name(int molid, int repid) {
  Molecule *mol = moleculeList->mol_from_id(molid);  
  if (!mol) return NULL;
  return mol->get_component_name(repid);
}

int VMDApp::molrep_get_by_name(int molid, const char *name) {
  Molecule *mol = moleculeList->mol_from_id(molid);  
  if (!mol) return -1;
  return mol->get_component_by_name(name);
}

int VMDApp::molrep_get_scaleminmax(int molid, int repid, float *min, float *max) {
  if (repid < 0 || repid >= num_molreps(molid)) return FALSE;
  const DrawMolItem *item = moleculeList->mol_from_id(molid)->component(repid);
#if 0
  // XXX Axel's color data range auto scale patch to fix the "two clicks"
  // problem he discovered with some volumetric datasets.  Needs closer
  // examination when I have a few minutes, but this will get him by for
  // the CPMD tutorial.
  Molecule *mol = moleculeList->mol_from_id(molid);
  item->atomColor->rescale_colorscale_minmax();
  item->atomColor->find(mol);
#endif
  item->atomColor->get_colorscale_minmax(min, max);
  return TRUE;
}
int VMDApp::molrep_set_scaleminmax(int molid, int repid, float min, float max) {
  if (repid < 0 || repid >= num_molreps(molid)) return FALSE;
  DrawMolItem *item = moleculeList->mol_from_id(molid)->component(repid);
  if (item->atomColor->set_colorscale_minmax(min, max)) {
    item->force_recalc(DrawMolItem::COL_REGEN);
    commandQueue->runcommand(new CmdMolScaleMinmax(molid, repid, min, max));
    return TRUE;
  }
  return FALSE;
}
int VMDApp::molrep_reset_scaleminmax(int molid, int repid) {
  if (repid < 0 || repid >= num_molreps(molid)) return FALSE;
  DrawMolItem *item = moleculeList->mol_from_id(molid)->component(repid);
  item->atomColor->rescale_colorscale_minmax();
  item->force_recalc(DrawMolItem::COL_REGEN);
  commandQueue->runcommand(new CmdMolScaleMinmax(molid, repid, 0, 0, 1));
  return TRUE;
}

int VMDApp::molrep_set_drawframes(int molid, int repid, const char *framesel) {
  if (repid < 0 || repid >= num_molreps(molid)) return FALSE;
  if (!framesel) {
    msgErr << "molrep_set_drawframes: Error, framesel is NULL!" << sendmsg;
    return FALSE;
  }
  DrawMolItem *item = moleculeList->mol_from_id(molid)->component(repid);
  item->set_drawframes(framesel);
  commandQueue->runcommand(new CmdMolDrawFrames(molid, repid, framesel));
  return TRUE;
}

const char *VMDApp::molrep_get_drawframes(int molid, int repid) {
  if (repid < 0 || repid >= num_molreps(molid)) return NULL;
  const DrawMolItem *item = moleculeList->mol_from_id(molid)->component(repid);
  return item->get_drawframes();
}

int VMDApp::molrep_show(int molid, int repid, int onoff) {
  if (repid < 0 || repid >= num_molreps(molid)) return FALSE;
  moleculeList->mol_from_id(molid)->show_rep(repid, onoff);
  commandQueue->runcommand(new CmdMolShowRep(molid, repid, onoff));
  return TRUE;
}

int VMDApp::molrep_is_shown(int molid, int repid) {
  if (repid < 0 || repid >= num_molreps(molid)) return FALSE;
  DrawMolItem *item = moleculeList->mol_from_id(molid)->component(repid);
  return item->displayed();
}


//
// IMD methods
//
int VMDApp::imd_connect(int molid, const char *host, int port) {
#ifdef VMDIMD 
  Molecule *mol = moleculeList->mol_from_id(molid);  
  if (!mol) return 0;
  if (!imdMgr) return 0;
  if (imdMgr->connect(mol, host, port)) {
    // tell the world
    commandQueue->runcommand(new CmdIMDConnect(molid,host, port));
    return 1;
  }
#endif
  return 0;
}

int VMDApp::imd_connected(int molid) {
#ifdef VMDIMD
  if (!imdMgr) return 0;
  Molecule *mol = imdMgr->get_imdmol();
  if (mol) {
    return (mol->id() == molid);
  }
#endif
  return 0;
}

int VMDApp::imd_sendforces(int num, const int *ind, const float *forces) {
#ifdef VMDIMD
  if (!imdMgr) return 0;
  return imdMgr->send_forces(num, ind, forces);
#endif
  return 0;
}

int VMDApp::imd_disconnect(int molid) {
#ifdef VMDIMD
  if (!imdMgr) return FALSE;
  Molecule *mol  = imdMgr->get_imdmol();
  if (mol && mol->id() == molid) {
    imdMgr->detach();
    return TRUE;
  }
#endif
  return FALSE;
}


//
// VideoStream methods
//
int VMDApp::vs_connect(const char *host, int port) {
#ifdef VMDNVPIPE
  if (!uivs) return 0;
  if (uivs->cli_connect(host, port)) {
    // tell the world
//    commandQueue->runcommand(new CmdVSConnect(host, port));
msgInfo << "VMDApp: VideoStream connected" << sendmsg;
    return 1;
  }
#endif
  return 0;
}

int VMDApp::vs_connected() {
#ifdef VMDNVPIPE
  if (!uivs) return 0;
  // XXX check for connection here
#endif
  return 0;
}

int VMDApp::vs_disconnect() {
#ifdef VMDNVPIPE
  if (!uivs) return FALSE;
  if (uivs) {
    uivs->cli_disconnect();
    return TRUE;
  }
#endif
  return FALSE;
}



//
// Display methods
//

void VMDApp::display_set_screen_height(float ht) {
  if (!display) return;
  display->screen_height(ht);
  commandQueue->runcommand(new CmdDisplayScreenHeight(ht));
}
float VMDApp::display_get_screen_height() {
  if (!display) return 0.0f;
  return display->screen_height();
}
void VMDApp::display_set_screen_distance(float d) {
  if (!display) return;
  display->distance_to_screen(d);
  commandQueue->runcommand(new CmdDisplayScreenDistance(d));
}
float VMDApp::display_get_screen_distance() {
  if (!display) return 0.0f;
  return display->distance_to_screen();
}
void VMDApp::display_set_position(int x, int y) {
  if (display)
    display->reposition_window(x, y);
}
#if 0
void VMDApp::display_get_position(int *x, int *y) {
  if (display) {
    display->window_position(x, y);
  }
}
#endif
void VMDApp::display_set_size(int w, int h) {
  if (display) {
    display->resize_window(w, h);
    // do an update so that the new size of the window becomes immediately
    // available.  
    display_update_ui();
  }
}
void VMDApp::display_get_size(int *w, int *h) {
  if (display) {
    *w = display->xSize;
    *h = display->ySize;
  }
}
void VMDApp::display_titlescreen() {
  if (display && display->supports_gui()) {
    delete vmdTitle;
    vmdTitle = new VMDTitle(display, &(scene->root));
  }
}

int VMDApp::display_set_stereo(const char *mode) {
  if (!mode) 
    return FALSE;

  int i, j;
  for (i=0; i<display->num_stereo_modes(); i++) {
    if (!strcmp(mode, display->stereo_name(i))) {
      display->set_stereo_mode(i);
      commandQueue->runcommand(new CmdDisplayStereo(mode));
      return TRUE;
    }
  }

  // Backwards compatibility with old scripts...
  const char *OldStereoNames[] = { 
    "CrystalEyes", "CrystalEyesReversed", "CrossEyes" 
  };
  const char *NewStereoNames[] = {
    "QuadBuffered", "QuadBuffered", "SideBySide" 
  };
  for (j=0; j<3; j++) {
    if (!strcmp(mode, OldStereoNames[j])) {
      for (i=0; i<display->num_stereo_modes(); i++) {
        if (!strcmp(NewStereoNames[j], display->stereo_name(i))) {
          display->set_stereo_mode(i);
          commandQueue->runcommand(new CmdDisplayStereo(NewStereoNames[j]));

          // preserve the swapped eye behavior of the old stereo mode names
          if (!strcmp(mode, "CrystalEyesReversed") ||
              !strcmp(mode, "CrossEyes")) {
            display_set_stereo_swap(1); 
          } 
          return TRUE;
        }
      }
    }
  }

  msgErr << "Illegal stereo mode: " << mode << sendmsg;
  return FALSE;
}

int VMDApp::display_set_stereo_swap(int onoff) {
  if (!onoff) {
    display->set_stereo_swap(0);
    commandQueue->runcommand(new CmdDisplayStereoSwap(0));
    return TRUE;
  }

  display->set_stereo_swap(1);
  commandQueue->runcommand(new CmdDisplayStereoSwap(1));
  return TRUE;
}

int VMDApp::display_set_cachemode(const char *mode) {
  if (!mode) return FALSE;
  for (int i=0; i<display->num_cache_modes(); i++) {
    if (!strcmp(mode, display->cache_name(i))) {
      display->set_cache_mode(i);
      commandQueue->runcommand(new CmdDisplayCacheMode(mode));
      return TRUE;
    }
  }
  msgErr << "Illegal cache mode: " << mode << sendmsg;
  return FALSE;
}

int VMDApp::display_set_rendermode(const char *mode) {
  if (!mode) return FALSE;
  for (int i=0; i<display->num_render_modes(); i++) {
    if (!strcmp(mode, display->render_name(i))) {
      display->set_render_mode(i);
      commandQueue->runcommand(new CmdDisplayRenderMode(mode));
      return TRUE;
    }
  }
  msgErr << "Illegal rendering mode: " << mode << sendmsg;
  return FALSE;
}

int VMDApp::display_set_eyesep(float sep) {
  if (sep < 0) return FALSE;
  display->set_eyesep(sep);
  commandQueue->runcommand(new CmdDisplayEyesep(sep));
  return TRUE;
}

int VMDApp::display_set_focallen(float flen) {
  if (!display->set_eye_dist(flen)) return FALSE;
  commandQueue->runcommand(new CmdDisplayFocallen(flen));
  return TRUE;
}

int VMDApp::display_set_projection(const char *proj) {
  if (!display->set_projection(proj)) return FALSE;
  commandQueue->runcommand(new CmdDisplayProj(proj));
  return TRUE;
}

int VMDApp::display_projection_is_perspective(void) {
  if (display->projection() == DisplayDevice::ORTHOGRAPHIC)
    return FALSE;

  return TRUE;
}

int VMDApp::display_set_aa(int onoff) {
  if (!display->aa_available()) return FALSE;
  if (onoff) display->aa_on(); else display->aa_off();
  commandQueue->runcommand(new CmdDisplayAAOn(onoff));
  return TRUE;
}

int VMDApp::display_set_depthcue(int onoff) {
  if (!display->cueing_available()) return FALSE;
  if (onoff) display->cueing_on(); else display->cueing_off();
  commandQueue->runcommand(new CmdDisplayDepthcueOn(onoff));
  return TRUE;
}

int VMDApp::display_set_culling(int onoff) {
  if (!display->culling_available()) return FALSE;
  if (onoff) display->culling_on(); else display->culling_off();
  commandQueue->runcommand(new CmdDisplayCullingOn(onoff));
  return TRUE;
}

int VMDApp::display_set_fps(int onoff) {
  if (onoff) fps->on(); else fps->off();
  commandQueue->runcommand(new CmdDisplayFPSOn(onoff));
  return TRUE;
}

int VMDApp::display_set_background_mode(int mode) {
  scene->set_background_mode(mode);
  commandQueue->runcommand(new CmdDisplayBackgroundGradientOn(mode));
  return TRUE;
}

int VMDApp::display_set_nearclip(float amt, int isdelta) {
  if (isdelta) {
    display->addto_near_clip(amt); 
    commandQueue->runcommand(new CmdDisplayClipNearRel(amt));
  } else {
    // prevent illegal near clipping plane values from causing problems
    if (amt <= 0.0)
      amt = 0.001f;
    display->set_near_clip(amt);
    commandQueue->runcommand(new CmdDisplayClipNear(amt));
  }
  return TRUE;
}

int VMDApp::display_set_farclip(float amt, int isdelta) {
  if (isdelta) {
    display->addto_far_clip(amt); 
    commandQueue->runcommand(new CmdDisplayClipFarRel(amt));
  } else {
    display->set_far_clip(amt);
    commandQueue->runcommand(new CmdDisplayClipFar(amt));
  }
  return TRUE;
}

int VMDApp::stage_set_location (const char *pos) {
  if (!pos) return FALSE;
  for (int i=0; i<stage->locations(); i++) {
    if (!strupcmp(pos, stage->loc_description(i))) {
      stage->location(i);
      commandQueue->runcommand(new CmdDisplayStageLocation(pos));
      return TRUE;
    }
  }
  return FALSE;
}

int VMDApp::stage_set_numpanels(int num) {
  if (!stage->panels(num)) return FALSE;
  commandQueue->runcommand(new CmdDisplayStagePanels(num));
  return TRUE;
}

int VMDApp::stage_set_size(float sz) {
  if (!stage->size(sz)) return FALSE;
  commandQueue->runcommand(new CmdDisplayStageSize(sz));
  return TRUE;
}

int VMDApp::axes_set_location (const char *pos) {
  if (!pos) return FALSE;
  for (int i=0; i<axes->locations(); i++) {
    if (!strupcmp(pos, axes->loc_description(i))) {
      axes->location(i);
      commandQueue->runcommand(new CmdDisplayAxes(pos));
      return TRUE;
    }
  }
  return FALSE;
}

int VMDApp::light_on(int n, int onoff) {
  if (n<0 || n >= DISP_LIGHTS) return FALSE;
  scene->activate_light(n, onoff);
  commandQueue->runcommand(new CmdDisplayLightOn(n, onoff));
  return TRUE;
}
int VMDApp::light_highlight(int n, int onoff) {
  if (n<0 || n >= DISP_LIGHTS) return FALSE;
  scene->highlight_light(n, onoff);
  commandQueue->runcommand(new CmdDisplayLightHL(n, onoff));
  return TRUE;
}

int VMDApp::light_rotate(int n, float amt, char axis) {
  if (n<0 || n >= DISP_LIGHTS) return FALSE;
  scene->rotate_light(n, amt, axis);
  commandQueue->runcommand(new CmdDisplayLightRot(n, amt, axis));
  return TRUE;
}

int VMDApp::light_move(int n, const float *newpos) {
  if (n<0 || n >= DISP_LIGHTS) return FALSE;
  scene->move_light(n, newpos);
  commandQueue->runcommand(new CmdDisplayLightMove(n, newpos));
  return TRUE;
}

int VMDApp::depthcue_set_mode(const char *mode) {
  if (!display->set_cue_mode(mode)) return FALSE;
  commandQueue->runcommand(new CmdDisplayCueMode(mode));
  return TRUE;
}

int VMDApp::depthcue_set_start(float val) {
  if (!display->set_cue_start(val)) return FALSE;
  commandQueue->runcommand(new CmdDisplayCueStart(val));
  return TRUE;
}

int VMDApp::depthcue_set_end(float val) {
  if (!display->set_cue_end(val)) return FALSE;
  commandQueue->runcommand(new CmdDisplayCueEnd(val));
  return TRUE;
}

int VMDApp::depthcue_set_density(float val) {
  if (!display->set_cue_density(val)) return FALSE;
  commandQueue->runcommand(new CmdDisplayCueDensity(val));
  return TRUE;
}

int VMDApp::display_set_shadows(int onoff) {
  if (!display->set_shadow_mode(onoff)) return FALSE;
  commandQueue->runcommand(new CmdDisplayShadowOn(onoff));
  return TRUE;
} 

int VMDApp::display_set_ao(int onoff) {
  if (!display->set_ao_mode(onoff)) return FALSE;
  commandQueue->runcommand(new CmdDisplayAOOn(onoff));
  return TRUE;
} 

int VMDApp::display_set_ao_ambient(float val) {
  if (!display->set_ao_ambient(val)) return FALSE;
  commandQueue->runcommand(new CmdDisplayAOAmbient(val));
  return TRUE;
}

int VMDApp::display_set_ao_direct(float val) {
  if (!display->set_ao_direct(val)) return FALSE;
  commandQueue->runcommand(new CmdDisplayAODirect(val));
  return TRUE;
}

int VMDApp::display_set_dof(int onoff) {
  if (!display->set_dof_mode(onoff)) return FALSE;
  commandQueue->runcommand(new CmdDisplayDoFOn(onoff));
  return TRUE;
} 

int VMDApp::display_set_dof_fnumber(float f) {
  if (!display->set_dof_fnumber(f)) return FALSE;
  commandQueue->runcommand(new CmdDisplayDoFFNumber(f));
  return TRUE;
}

int VMDApp::display_set_dof_focal_dist(float d) {
  if (!display->set_dof_focal_dist(d)) return FALSE;
  commandQueue->runcommand(new CmdDisplayDoFFocalDist(d));
  return TRUE;
}

void VMDApp::deactivate_uitext_stdin() {
  if (uiText)
    uiText->Off();
}

int VMDApp::activate_menus() {
  // XXX This should control Tk menus as well; at present Tk menus are 
  // available whenever the display supports GUI's.

#ifdef VMDGUI
  if(display->supports_gui()) {

    delete uivr;
    uivr = new UIVR(this);
    uivr->On();

    // if we are using the FLTK library ...
#ifdef VMDFLTK
  VMDMenu *obj;
  obj = new MainFltkMenu(this);
  menulist->add_name(obj->get_name(), obj);
  obj = new ColorFltkMenu(this);
  menulist->add_name(obj->get_name(), obj);
  obj = new MaterialFltkMenu(this);
  menulist->add_name(obj->get_name(), obj);
  obj = new DisplayFltkMenu(this);
  menulist->add_name(obj->get_name(), obj);
  obj = new FileChooserFltkMenu(this);
  menulist->add_name(obj->get_name(), obj);
  obj = new GeometryFltkMenu(this);
  menulist->add_name(obj->get_name(), obj);
  obj = new GraphicsFltkMenu(this);
  menulist->add_name(obj->get_name(), obj);
  obj = new RenderFltkMenu(this);
  menulist->add_name(obj->get_name(), obj);
  obj = new SaveTrajectoryFltkMenu(this);
  menulist->add_name(obj->get_name(), obj);
  obj = new ToolFltkMenu(this);
  menulist->add_name(obj->get_name(), obj);
#endif /*VMDFLTK*/
  }
  return TRUE;
#endif /*VMDGUI*/
  
  // no menus available
  return FALSE;
}

int VMDApp::label_add(const char *category, int n, const int *molids, 
    const int *atomids, const int *cells, float k, int toggle) {
  if (!category || !molids || !atomids) return -1;
  int rc = geometryList->add_geometry(category, molids, atomids, cells, k, 
      toggle);
  if (rc >= 0) {
    if (!strcmp(category, "Springs"))
      commandQueue->runcommand(new CmdLabelAddspring(molids[0], atomids[0],
          atomids[1], k));
    else 
      commandQueue->runcommand(new CmdLabelAdd(category, n, (int *)molids, (int *)atomids));
  }
  return rc;
}

int VMDApp::label_show (const char *category, int n, int onoff) {
  if (!category) return FALSE;
  if (geometryList->show_geometry(category, n, onoff)) {
    commandQueue->runcommand(new CmdLabelShow(category, n, onoff));
    return TRUE;
  }
  return FALSE;
}

float VMDApp::label_get_text_size() const {
  return geometryList->getTextSize();
}

int VMDApp::label_set_text_size(float newsize) {
  if (geometryList->setTextSize(newsize)) {
    commandQueue->runcommand(new CmdLabelTextSize(newsize));
    return TRUE;
  }
  return FALSE;
}

float VMDApp::label_get_text_thickness() const {
  return geometryList->getTextThickness();
}

int VMDApp::label_set_text_thickness(float newthick) {
  if (geometryList->setTextThickness(newthick)) {
    commandQueue->runcommand(new CmdLabelTextThickness(newthick));
    return TRUE;
  }
  return FALSE;
}

int VMDApp::label_set_textoffset(const char *nm, int n, float x, float y) {
  float delta[2] = { x, y };
  if (geometryList->setTextOffset(nm, n, delta)) {
    commandQueue->runcommand(new CmdLabelTextOffset(nm, n, x, y));
    return TRUE;
  }
  return FALSE;
}

int VMDApp::label_set_textformat(const char *nm, int n, const char *format) {
  if (geometryList->setTextFormat(nm, n, format)) {
    commandQueue->runcommand(new CmdLabelTextFormat(nm, n, format));
    return TRUE;
  }
  return FALSE;
}

int VMDApp::label_delete(const char *category, int n) {
  if (!category) return FALSE;
  if (geometryList->del_geometry(category, n)) {
    commandQueue->runcommand(new CmdLabelDelete(category, n));
    return TRUE;
  }
  return FALSE;
}

int VMDApp::tool_create(const char *type, int argc, const char **argv) {
  if (!uivr) return FALSE;
  if (!uivr->add_tool_with_USL(type, argc, argv)) return FALSE;
  commandQueue->runcommand(new CmdToolCreate(type, argc, argv));
  return TRUE;
}

int VMDApp::tool_change_type(int toolnum, const char *type) {
  if (!uivr) return FALSE;
  if (!uivr->change_type(toolnum, type)) return FALSE;
  commandQueue->runcommand(new CmdToolChange(type, toolnum));
  return TRUE;
}

int VMDApp::tool_delete(int toolnum) {
  if (!uivr) return FALSE;
  if (!uivr->remove_tool(toolnum)) return FALSE;
  commandQueue->runcommand(new CmdToolDelete(toolnum));
  // XXXX fix this for multiple tools
  commandQueue->runcommand(new PickAtomCallbackEvent(-1,-1,"uivr"));
  return TRUE;
}

int VMDApp::tool_set_position_scale(int toolnum, float newval) {
  if (!uivr) return FALSE;
  if (uivr->set_position_scale(toolnum, newval)) {
    commandQueue->runcommand(new CmdToolScale(newval, toolnum));
    return TRUE;
  }
  return FALSE;
}

int VMDApp::tool_set_force_scale(int toolnum, float newval) {
  if (!uivr) return FALSE;
  if (uivr->set_force_scale(toolnum, newval)) {
    commandQueue->runcommand(new CmdToolScaleForce(newval, toolnum));
    return TRUE;
  }
  return FALSE;
}

int VMDApp::tool_set_spring_scale(int toolnum, float newval) {
  if (!uivr) return FALSE;
  if (uivr->set_spring_scale(toolnum, newval)) {
    commandQueue->runcommand(new CmdToolScaleSpring(newval, toolnum));
    return TRUE;
  }
  return FALSE;
}

const char *VMDApp::material_add(const char *name, const char *copy) {
  const char *newname = materialList->add_material(name, copy);
  if (newname) {
    commandQueue->runcommand(new CmdMaterialAdd(name, copy));
  }
  return newname;
}

int VMDApp::material_delete(const char *name) {
  char * strname = stringdup(name);
  int ind = materialList->material_index(strname);
  if (materialList->delete_material(ind)) {
    commandQueue->runcommand(new CmdMaterialDelete(strname));
    delete [] strname;
    return TRUE;
  }
  delete [] strname;
  return FALSE;
}

int VMDApp::material_rename(const char *prevname, const char *newname) {
  char * oldname = stringdup(prevname);
  int ind = materialList->material_index(oldname);
  if (ind < 0) {
    msgErr << "material rename: '" << oldname << "' does not exist."   
           << sendmsg;
    delete [] oldname;
    return FALSE;
  }
  int n = strlen(newname);
  if (!n) return FALSE;
  for (size_t i=0; i<strlen(newname); i++) {
    if (!isalnum(newname[i])) {
      msgErr << "material rename: new name contains non-alphanumeric character"
             << sendmsg;
      delete [] oldname;
      return FALSE;
    }
  }
  if (materialList->material_index(newname) >= 0) {
    msgErr << "material rename: '" << newname << "' already exists." 
           << sendmsg;
    delete [] oldname;
    return FALSE;
  }
  materialList->set_name(ind, newname);
  commandQueue->runcommand(new CmdMaterialRename(oldname, newname));
  delete [] oldname;
  return TRUE;
}

int VMDApp::material_change(const char *name, int property, float val) {
  int ind = materialList->material_index(name);
  if (ind < 0) return FALSE;
  switch (property) {
    case MAT_AMBIENT: materialList->set_ambient(ind, val); break;
    case MAT_SPECULAR: materialList->set_specular(ind, val); break;
    case MAT_DIFFUSE: materialList->set_diffuse(ind, val); break;
    case MAT_SHININESS: materialList->set_shininess(ind, val); break;
    case MAT_MIRROR: materialList->set_mirror(ind, val); break;
    case MAT_OPACITY: materialList->set_opacity(ind, val); break;
    case MAT_OUTLINE: materialList->set_outline(ind, val); break;
    case MAT_OUTLINEWIDTH: materialList->set_outlinewidth(ind, val); break;
    case MAT_TRANSMODE: materialList->set_transmode(ind, val); break;
  }
  commandQueue->runcommand(new CmdMaterialChange(name, property, val));
  return TRUE;
}

int VMDApp::material_restore_default(int ind) {
  if (materialList->restore_default(ind)) {
    commandQueue->runcommand(new CmdMaterialDefault(ind));
    return TRUE;
  }
  return FALSE;
}

int VMDApp::mouse_set_mode(int mm, int ms) {
  if (!mouse->move_mode((Mouse::MoveMode)mm, ms)) {
    msgErr << "Illegal mouse mode: " << mm << " " << ms << sendmsg;
    return FALSE;
  }
  
  // If mouse mode is a picking mode, set it here
  switch (mm) {
    case Mouse::PICK:        pickModeList->set_pick_mode(PickModeList::PICK); break;
    case Mouse::QUERY:       pickModeList->set_pick_mode(PickModeList::QUERY); break;
    case Mouse::CENTER:      pickModeList->set_pick_mode(PickModeList::CENTER); break;
    case Mouse::LABELATOM:   pickModeList->set_pick_mode(PickModeList::LABELATOM); break;
    case Mouse::LABELBOND:   pickModeList->set_pick_mode(PickModeList::LABELBOND); break;
    case Mouse::LABELANGLE:  pickModeList->set_pick_mode(PickModeList::LABELANGLE); break;
    case Mouse::LABELDIHEDRAL:  pickModeList->set_pick_mode(PickModeList::LABELDIHEDRAL); break;
    case Mouse::MOVEATOM:    pickModeList->set_pick_mode(PickModeList::MOVEATOM); break;
    case Mouse::MOVERES:     pickModeList->set_pick_mode(PickModeList::MOVERES); break;
    case Mouse::MOVEFRAG:    pickModeList->set_pick_mode(PickModeList::MOVEFRAG); break;
    case Mouse::MOVEMOL:     pickModeList->set_pick_mode(PickModeList::MOVEMOL); break;
    case Mouse::FORCEATOM:   pickModeList->set_pick_mode(PickModeList::FORCEATOM); break;
    case Mouse::FORCERES:    pickModeList->set_pick_mode(PickModeList::FORCERES); break;
    case Mouse::FORCEFRAG:   pickModeList->set_pick_mode(PickModeList::FORCEFRAG); break;
    case Mouse::MOVEREP:     pickModeList->set_pick_mode(PickModeList::MOVEREP); break;
    case Mouse::ADDBOND:     pickModeList->set_pick_mode(PickModeList::ADDBOND); break;
    default: break;
  }
  
  commandQueue->runcommand(new CmdMouseMode(mm, ms));
  return TRUE;
}


int VMDApp::mobile_set_mode(int mm) {
  if (!mobile->move_mode((Mobile::MoveMode) mm)) {
    msgErr << "Illegal mobile mode: " << mm << " " << sendmsg;
    return FALSE;
  }
  commandQueue->runcommand(new CmdMobileMode(mm));
  return TRUE;
}

int VMDApp::mobile_get_mode() {
  return mobile->get_move_mode();
}

void VMDApp::mobile_get_client_list(ResizeArray <JString*>* &nick, 
                         ResizeArray <JString*>* &ip, ResizeArray <bool>* &active) 
{
  mobile->get_client_list(nick, ip, active);
}

int VMDApp::mobile_network_port(int port) {
  mobile->network_port(port);
  //  commandQueue->runcommand(new CmdMobileNetworkPort(port));
  return TRUE;
}

int VMDApp::mobile_get_network_port() {
  return mobile->get_port();
}

int VMDApp::mobile_get_APIsupported() {
  return mobile->get_APIsupported();
}

  /// Set the currently active client, identified by nick and ip
int VMDApp::mobile_set_activeClient(const char *nick, const char *ip) {
  return mobile->set_activeClient(nick, ip);
}

  /// Send a message to a specific client
int VMDApp::mobile_sendMsg(const char *nick, const char *ip, 
                           const char *msgType, const char *msg) {
  return mobile->sendMsgToClient(nick, ip, msgType, msg);
}


/// return the current mobile interface event data,
/// used by the UIVR MobileTracker interface
void VMDApp::mobile_get_tracker_status(float &tx, float &ty, float &tz,
                                       float &rx, float &ry, float &rz,
                                       int &buttons) {
  if (mobile != NULL) {
    mobile->get_tracker_status(tx, ty, tz, rx, ry, rz, buttons);
  } else {
    tx=ty=tz=rx=ry=rz=0.0f;
    buttons=0;
  }
}


int VMDApp::spaceball_set_mode(int mm) {
  if (!spaceball->move_mode((Spaceball::MoveMode) mm)) {
    msgErr << "Illegal spaceball mode: " << mm << " " << sendmsg;
    return FALSE;
  }
  commandQueue->runcommand(new CmdSpaceballMode(mm));
  return TRUE;
}


int VMDApp::spaceball_set_sensitivity(float s) {
  spaceball->set_sensitivity(s);
  //  commandQueue->runcommand(new CmdSpaceballSensitivity(s));
  return TRUE;
}


int VMDApp::spaceball_set_null_region(int nr) {
  spaceball->set_null_region(nr);
  //  commandQueue->runcommand(new CmdSpaceballNullRegion(s));
  return TRUE;
}


/// return the current spaceball event data,
/// used by the UIVR SpaceballTracker interface
void VMDApp::spaceball_get_tracker_status(float &tx, float &ty, float &tz,
                                          float &rx, float &ry, float &rz,
                                          int &buttons) {
  if (spaceball != NULL) {
    spaceball->get_tracker_status(tx, ty, tz, rx, ry, rz, buttons);
  } else {
    tx=ty=tz=rx=ry=rz=0.0f;
    buttons=0;
  }
}


int VMDApp::textinterp_change(const char *name) {
  return uiText->change_interp(name);
}

//
// MPI related routines
//
void VMDApp::par_barrier() {
#if defined(VMDMPI)
  // perform a barrier if running in parallel
  if (mpienabled)
    vmd_mpi_barrier();
#endif
}

